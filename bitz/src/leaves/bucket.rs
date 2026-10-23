//! The first two rounds of the bottom layer, through buckets of column weights (Section 6.2).

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;

use super::tables::FoldTables;
use super::{BitLeaves, add, coefficients, deal_chunk, karatsuba, quadratic};

/// Round one: a block is one byte of a column, so `H_b[x] = sum_c eq(c) [byte_b(c) = x]` serves every tree.
///
/// Rows `s` and `4 + s` of a byte are the two endpoints of child `s`, so the quartic of a byte splits into quadratics of four bits each.
pub(super) fn byte_round<F: Field, EF: Field>(
    leaves: &BitLeaves<'_, F, EF>,
    rows: &[EF],
    columns: &[EF],
    trees: &[EF],
) -> [EF; 5] {
    let (tile_words, tiles) = leaves.tiles();
    let bytes = 8 * tile_words;
    let partials: Vec<[EF; 5]> = (0..tiles)
        .into_par_iter()
        .map(|tile| {
            let mut buckets = EF::zero_vec(bytes * 256);
            for (c, &weight) in columns.iter().enumerate() {
                for (wi, &word) in leaves
                    .column(c, tile * tile_words, tile_words)
                    .iter()
                    .enumerate()
                {
                    for (byte, &x) in word.to_le_bytes().iter().enumerate() {
                        buckets[(8 * wi + byte) * 256 + x as usize] += weight;
                    }
                }
            }
            let mut sum = [EF::ZERO; 5];
            for (t, images) in leaves.images.iter().enumerate() {
                for (byte, buckets) in buckets.as_chunks::<256>().0.iter().enumerate() {
                    let block = tile * bytes + byte;
                    let y = &images[8 * block..][..8];
                    // Child slots `(0, 1)` read bits `0, 1, 4, 5`, slots `(2, 3)` bits `2, 3, 6, 7`.
                    let half = |s: usize| -> [[EF; 3]; 16] {
                        core::array::from_fn(|x| {
                            let leaf = |bit: usize, row: usize| {
                                if x >> bit & 1 == 1 { y[row] } else { EF::ONE }
                            };
                            quadratic(leaf(0, s), leaf(2, s + 4), leaf(1, s + 1), leaf(3, s + 5))
                        })
                    };
                    let (low, high) = (half(0), half(2));
                    let mut products = [EF::ZERO; 6];
                    for (a, low) in low.iter().enumerate() {
                        // `G[a] = sum_x H[x] high(x)` over the bytes whose low half is `a`.
                        let mut folded = [EF::ZERO; 3];
                        for (b, high) in high.iter().enumerate() {
                            let x = (a & 3) | (b & 3) << 2 | (a >> 2) << 4 | (b >> 2) << 6;
                            for (folded, &h) in folded.iter_mut().zip(high) {
                                *folded += buckets[x] * h;
                            }
                        }
                        karatsuba(&mut products, *low, folded);
                    }
                    let weight = rows[block] * trees[t];
                    sum = add(sum, coefficients(products).map(|c| weight * c));
                }
            }
            sum
        })
        .collect();
    partials.into_iter().fold([EF::ZERO; 5], add)
}

/// Round two: a block is four bits of each child slot, so slots `(0, 1)` and `(2, 3)` each index a byte table of quadratics.
///
/// Buckets `G[a] = sum_c eq(c) high(b_c)` over the columns whose low byte is `a` leave one product per column.
pub(super) fn pair_round<F: Field, EF: ExtensionField<F>>(
    leaves: &BitLeaves<'_, F, EF>,
    bound: &[EF],
    rows: &[EF],
    columns: &[EF],
    trees: &[EF],
) -> [EF; 5] {
    let (tile_words, tiles) = leaves.tiles();
    let blocks = tile_words * 4;
    let weights = Point::new(bound).equality_weights_lsb();
    let chunks = leaves.column_chunks();
    let tasks = leaves.images.len() * tiles * chunks.len();
    let partials: Vec<[EF; 5]> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let (t, tile, chunk) = (
                task / (tiles * chunks.len()),
                task / chunks.len() % tiles,
                task % chunks.len(),
            );
            let range = chunks[chunk].clone();
            let (dealt, _) = deal_chunk::<F, EF>(leaves, 0, tile, range.clone(), columns);
            let stride = dealt.len() / tile_words;
            let mut endpoints = FoldTables::new(0, 1, 8);
            let mut low = [[EF::ZERO; 3]; 256];
            let mut high = [[EF::ZERO; 3]; 256];
            let mut buckets = [[EF::ZERO; 3]; 256];
            let mut sum = [EF::ZERO; 5];
            for b in 0..blocks {
                let block = tile * blocks + b;
                endpoints.clear();
                for s in 0..4 {
                    for side in 0..2 {
                        endpoints.push(&leaves.images[t], &weights, s, (2 * block + side) << 1);
                    }
                }
                // A slot's nibble holds its even endpoint's two bits, then its odd endpoint's.
                let lines: [[[EF; 2]; 16]; 4] = core::array::from_fn(|s| {
                    core::array::from_fn(|nibble| {
                        [0, 1].map(|side| {
                            endpoints.eval(2 * s + side, (nibble >> (2 * side) & 3) as u64)
                        })
                    })
                });
                for x in 0..256 {
                    let line = |s: usize, nibble: usize| lines[s][nibble];
                    let ([e0, o0], [e1, o1]) = (line(0, x & 15), line(1, x >> 4));
                    low[x] = quadratic(e0, o0, e1, o1);
                    let ([e2, o2], [e3, o3]) = (line(2, x & 15), line(3, x >> 4));
                    high[x] = quadratic(e2, o2, e3, o3);
                }
                buckets.fill([EF::ZERO; 3]);
                let shift = 4 * (b % 4);
                let words = &dealt[b / 4 * stride..][..range.len()];
                for (&word, &weight) in words.iter().zip(&columns[range.clone()]) {
                    let byte = |s: usize| {
                        ((word >> (16 * s + shift)) & 15
                            | ((word >> (16 * s + 16 + shift)) & 15) << 4)
                            as usize
                    };
                    let (a, x) = (byte(0), byte(2));
                    for (bucket, &h) in buckets[a].iter_mut().zip(&high[x]) {
                        *bucket += weight * h;
                    }
                }
                let mut products = [EF::ZERO; 6];
                for (low, bucket) in low.iter().zip(&buckets) {
                    karatsuba(&mut products, *low, *bucket);
                }
                let weight = rows[block] * trees[t];
                sum = add(sum, coefficients(products).map(|c| weight * c));
            }
            sum
        })
        .collect();
    partials.into_iter().fold([EF::ZERO; 5], add)
}
