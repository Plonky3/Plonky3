//! Folds through byte tables: each eight rows map a byte of set bits to the sum of their weights.
//!
//! Weights are split into 32-bit pieces held in 64-bit lanes, so fewer than `2^32` rows never carry.

use alloc::vec;
use alloc::vec::Vec;

use p3_maybe_rayon::prelude::*;

/// Byte groups of one table tile: 256 rows, half a cache line of every column.
const TILE_GROUPS: usize = 32;

/// `mu_c = sum_b gamma[b] bit(c, b)` for every column `c` of `2^log_rows` rows.
///
/// `pieces`, one or two, 32-bit pieces must hold every weight.
///
/// # Panics
///
/// A column is not whole words, or `gamma` does not hold one weight per row.
pub(crate) fn fold(words: &[u64], log_rows: usize, gamma: &[u64], pieces: usize) -> Vec<u128> {
    assert!(log_rows >= 6, "a column is whole words");
    assert_eq!(gamma.len(), 1 << log_rows, "one weight per row");
    match pieces {
        1 => fold_pieces::<1>(words, log_rows, gamma),
        2 => fold_pieces::<2>(words, log_rows, gamma),
        _ => unreachable!("a weight of at most 64 bits"),
    }
}

/// The fold with weights of `P` pieces.
fn fold_pieces<const P: usize>(words: &[u64], log_rows: usize, gamma: &[u64]) -> Vec<u128> {
    let column_words = 1 << (log_rows - 6);
    let columns = words.len() / column_words;
    let tile = TILE_GROUPS.min(1 << (log_rows - 3));
    let tiles = (1 << (log_rows - 3)) / tile;
    // Every tile builds its tables and reads its bytes of every column.
    let item_bytes = tile * 256 * P * 8 + columns * tile;
    let sums = (0..tiles)
        .into_par_iter()
        .with_min_task_bytes(item_bytes)
        .par_fold_reduce(
            || (vec![[0u64; P]; columns], vec![[0u64; P]; tile * 256]),
            |(mut sums, mut table), index| {
                let first_group = index * tile;
                build_tables(&gamma[8 * first_group..][..8 * tile], &mut table);
                accumulate(words, column_words, first_group / 8, &table, &mut sums);
                (sums, table)
            },
            |(mut left, table), (right, _)| {
                for (l, r) in left.iter_mut().zip(&right) {
                    for (a, b) in l.iter_mut().zip(r) {
                        *a += b;
                    }
                }
                (left, table)
            },
        )
        .0;
    // Lane p carries weight 2^(32 p).
    sums.iter()
        .map(|lanes| {
            lanes
                .iter()
                .enumerate()
                .map(|(p, &lane)| u128::from(lane) << (32 * p))
                .sum()
        })
        .collect()
}

/// Entry `256 k + x` is the sum of the pieces of rows `8 k + s` for every set bit `s` of `x`.
fn build_tables<const P: usize>(gamma: &[u64], table: &mut [[u64; P]]) {
    for (rows, entries) in gamma
        .as_chunks::<8>()
        .0
        .iter()
        .zip(table.as_chunks_mut::<256>().0)
    {
        let split: [[u64; P]; 8] =
            core::array::from_fn(|s| core::array::from_fn(|p| (rows[s] >> (32 * p)) & 0xffff_ffff));
        entries[0] = [0; P];
        // Each byte adds its lowest set bit to the entry without it.
        for x in 1..256usize {
            let (rest, bit) = (x & (x - 1), x.trailing_zeros() as usize);
            entries[x] = core::array::from_fn(|p| entries[rest][p] + split[bit][p]);
        }
    }
}

/// Add the table rows the tile's bytes select to every column's sums.
fn accumulate<const P: usize>(
    words: &[u64],
    column_words: usize,
    first_word: usize,
    table: &[[u64; P]],
    sums: &mut [[u64; P]],
) {
    let tile_words = table.len() / (256 * 8);
    for (column, sum) in words.chunks_exact(column_words).zip(sums.iter_mut()) {
        let bytes = column[first_word..][..tile_words]
            .iter()
            .flat_map(|word| word.to_le_bytes());
        for (entries, byte) in table.as_chunks::<256>().0.iter().zip(bytes) {
            let entry = &entries[byte as usize];
            for (s, e) in sum.iter_mut().zip(entry) {
                *s += e;
            }
        }
    }
}

/// The fold one set bit at a time.
#[cfg(test)]
pub(crate) fn fold_reference(words: &[u64], log_rows: usize, gamma: &[u64]) -> Vec<u128> {
    let column_words = 1 << (log_rows - 6);
    words
        .chunks_exact(column_words)
        .map(|column| {
            (0..1 << log_rows)
                .filter(|b| column[b / 64] >> (b % 64) & 1 == 1)
                .map(|b| u128::from(gamma[b]))
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn the_table_fold_is_the_set_bit_fold(
            log_rows in 6usize..=12,
            log_columns in 0usize..=4,
            pieces in 1usize..=2,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let words: Vec<u64> = (0..1 << (log_rows + log_columns - 6)).map(|_| rng.next_u64()).collect();
            // Weights below 2^(32 pieces), so they have that many pieces.
            let gamma: Vec<u64> = (0..1 << log_rows)
                .map(|_| rng.next_u64() >> (64 - 32 * pieces))
                .collect();
            prop_assert_eq!(
                fold(&words, log_rows, &gamma, pieces),
                fold_reference(&words, log_rows, &gamma)
            );
        }
    }

    #[test]
    fn full_weights_on_full_columns_reach_the_largest_fold() {
        // Every bit set and every weight 2^64 - 1 over 2^8 rows: every lane is as full as a tile makes it.
        let gamma = vec![u64::MAX; 256];
        let words = [u64::MAX; 8];
        let largest = u128::from(u64::MAX) << 8;
        assert_eq!(fold(&words, 8, &gamma, 2), [largest; 2]);
        assert_eq!(fold_reference(&words, 8, &gamma), [largest; 2]);
    }
}
