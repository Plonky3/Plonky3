//! The square bit transpose: row `i` of the result is column `i` of the input.
//!
//! A `2n x 2n` matrix is four `n x n` quadrants, and transposing swaps the off-diagonal pair:
//!
//! ```text
//!     | A  B |            | A^T  C^T |
//!     | C  D |    ->      | B^T  D^T |
//! ```
//!
//! Doing that swap at every quadrant size costs a shift, a mask and two exclusive ors a pair.
//! So `log2(B)` passes over the rows leave nothing left to transpose.
//!
//! This is Hacker's Delight figure 7-6, mirrored because column `0` is the lowest bit here.
//!
//! A `nB x nB` matrix is an `n x n` grid of `B x B` blocks, so the identity applies a level up.
//! Phase 1 transposes each block in place, phase 2 swaps block `(i, j)` with block `(j, i)`.
//!
//! Phase 1 treats every word alike, so it sweeps a run of rows flat, at the vector width.
//!
//! Only square, only a power of two, one row per lane, and in place.

use core::slice;

use super::packing::{PackedGf2, block_mask};
use super::underlier::{Underlier, Word};

/// The backing words of a run of packings, as one flat slice.
///
/// A run of `k` packings is `k * n` contiguous words, so a pass sweeps it straight through.
#[inline]
const fn words_of<U: Underlier>(rows: &mut [PackedGf2<U>]) -> &mut [U::Word] {
    //     size      exactly the backing words, so no padding between packings
    //     align     at least a word's, so every packing starts word-aligned
    const {
        assert!(size_of::<PackedGf2<U>>() == U::WORDS * size_of::<U::Word>());
        assert!(align_of::<PackedGf2<U>>() >= align_of::<U::Word>());
    }

    let len = rows.len() * U::WORDS;

    // SAFETY: by the assertions the run is exactly `len` contiguous aligned words.
    // They lie in one allocation, and every bit pattern of a word is valid.
    unsafe { slice::from_raw_parts_mut(rows.as_mut_ptr().cast::<U::Word>(), len) }
}

/// Exchange the off-diagonal quadrants of every `2 * STEP` block, in every word of a row.
///
/// The quadrant side is a constant, so the compiler knows every trip count.
/// That is what stops it vectorising the short passes blindly.
///
/// A quadrant as wide as a word belongs to the grid exchange, so that call folds away.
// The chunk size `2 * STEP` is not a valid const generic argument, so the lint's fix is out.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn stage<U: Underlier, const STEP: usize>(group: &mut [PackedGf2<U>]) {
    if STEP >= <U::Word as Word>::BITS {
        return;
    }

    // The column range of the lower-left quadrant of each block.
    let mask = U::Word::from_u64(block_mask(STEP));

    // Upper quadrants in the first half of each run, lower ones in the second.
    for run in group.chunks_exact_mut(2 * STEP) {
        let (top, bottom) = run.split_at_mut(STEP);

        // The halves pair off word for word, so the sweep stays flat.
        //
        //     upper right   bits [STEP, 2*STEP) of a
        //     lower left    bits [0, STEP)      of b
        for (a, b) in words_of(top).iter_mut().zip(words_of(bottom)) {
            // Their difference, aligned at the bottom, then added back to each side.
            let diff = ((*a >> STEP) ^ *b) & mask;

            *a = *a ^ (diff << STEP);
            *b = *b ^ diff;
        }
    }
}

impl<U: Underlier> PackedGf2<U> {
    /// Replace the square bit matrix whose row `i` is `rows[i]` by its transpose.
    ///
    /// Lane `c` of row `r` becomes lane `r` of row `c`.
    ///
    /// # Panics
    /// Panics unless the matrix has exactly as many rows as the packing has lanes.
    pub fn transpose(rows: &mut [Self]) {
        assert_eq!(
            rows.len(),
            Self::WIDTH,
            "a bit transpose needs one row per lane"
        );
        let block = Self::WORD_BITS;

        // Phase 1, a row-group at a time; a pass wider than the word is skipped inside it.
        for group in rows.chunks_exact_mut(block) {
            stage::<U, 32>(group);
            stage::<U, 16>(group);
            stage::<U, 8>(group);
            stage::<U, 4>(group);
            stage::<U, 2>(group);
            stage::<U, 1>(group);
        }

        // Phase 2: block `(i, j)` is word `j` of rows `block*i ..`, so no bits move.
        // A single-word row has no grid, and both loops are empty.
        for i in 0..U::WORDS {
            for j in (i + 1)..U::WORDS {
                // Disjoint row ranges, because `i < j`.
                let (above, below) = rows.split_at_mut(block * j);

                for r in 0..block {
                    let upper = above[block * i + r].words()[j];
                    let lower = below[r].words()[i];
                    above[block * i + r].words_mut()[j] = lower;
                    below[r].words_mut()[i] = upper;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use crate::{Gf2, PackedGf2x8, PackedGf2x64, PackedGf2x128};

    /// One test module per matrix side, against the definition read lane by lane.
    macro_rules! transpose_tests {
        ($module:ident, $name:ident, $underlier:ty, $word:ty, $words:literal, $width:literal) => {
            mod $module {
                use super::*;
                use crate::bitslice::underlier::Underlier;
                use crate::$name;

                fn from_words(words: [$word; $words]) -> $name {
                    $name::new(<$underlier>::from_words_fn(|i| words[i]))
                }

                /// Row `r` of the result is column `r` of the input, one lane at a time.
                fn reference(rows: &[$name; $width]) -> [$name; $width] {
                    core::array::from_fn(|r| $name::from_fn(|c| rows[c].get(r)))
                }

                /// Entry `(r, c)` depends on its position, so any misplacement shows.
                fn fixture() -> [$name; $width] {
                    core::array::from_fn(|r| {
                        $name::from_fn(|c| Gf2::from_bool((r * 7 + c * 13 + r * c) % 3 == 0))
                    })
                }

                #[test]
                fn transpose_matches_the_definition() {
                    let mut rows = fixture();
                    let expected = reference(&rows);

                    $name::transpose(&mut rows);

                    assert_eq!(rows, expected);
                }

                #[test]
                fn transposing_twice_is_the_identity() {
                    // An involution: two passes restore the input.
                    let original = fixture();
                    let mut rows = original;

                    $name::transpose(&mut rows);
                    $name::transpose(&mut rows);

                    assert_eq!(rows, original);
                }

                #[test]
                fn the_extremes_are_fixed_or_swapped_as_expected() {
                    // Symmetric matrices are left alone.
                    let mut zeros = [$name::ZERO; $width];
                    $name::transpose(&mut zeros);
                    assert_eq!(zeros, [$name::ZERO; $width]);

                    let mut ones = [$name::ONE; $width];
                    $name::transpose(&mut ones);
                    assert_eq!(ones, [$name::ONE; $width]);

                    // Entry `(0, W-1)` moves to the mirrored corner, `(W-1, 0)`.
                    let mut corner = [$name::ZERO; $width];
                    corner[0].set($width - 1, Gf2::ONE);
                    $name::transpose(&mut corner);

                    let mut expected = [$name::ZERO; $width];
                    expected[$width - 1].set(0, Gf2::ONE);
                    assert_eq!(corner, expected);
                }

                #[test]
                fn the_identity_matrix_is_its_own_transpose() {
                    // The diagonal is the fixed set, so this must not change at all.
                    let identity: [$name; $width] =
                        core::array::from_fn(|r| $name::from_fn(|c| Gf2::from_bool(r == c)));
                    let mut rows = identity;

                    $name::transpose(&mut rows);

                    assert_eq!(rows, identity);
                }

                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(16))]

                    #[test]
                    fn transpose_matches_the_definition_on_random_matrices(
                        seed in any::<u64>(),
                    ) {
                        // A xorshift fills the matrix, so the printed seed reproduces it.
                        let mut state = seed | 1;
                        let mut next = move || {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            state
                        };
                        let mut rows: [$name; $width] = core::array::from_fn(|_| {
                            from_words(core::array::from_fn(|_| next() as $word))
                        });

                        let expected = reference(&rows);
                        $name::transpose(&mut rows);

                        prop_assert_eq!(rows, expected);
                    }
                }
            }
        };
    }

    transpose_tests!(x8, PackedGf2x8, u8, u8, 1, 8);
    transpose_tests!(x16, PackedGf2x16, u16, u16, 1, 16);
    transpose_tests!(x32, PackedGf2x32, u32, u32, 1, 32);
    transpose_tests!(x64, PackedGf2x64, u64, u64, 1, 64);
    transpose_tests!(x128, PackedGf2x128, crate::M128, u64, 2, 128);
    transpose_tests!(x256, PackedGf2x256, crate::M256, u64, 4, 256);
    transpose_tests!(x512, PackedGf2x512, crate::M512, u64, 8, 512);

    #[test]
    fn a_hand_written_eight_by_eight_transposes_as_drawn() {
        // Column 0 is drawn leftmost and is the lowest bit, so the literals read backwards.
        //
        //     |         input       output
        //     |         c0....c7    c0....c7
        //     |     r0  11111111    10000000
        //     |     r1  00000000    10000001
        //     |     r2  00000000    10000001
        //     |     r3  00010000    10010001
        //     |     r4  00000000    10000001
        //     |     r5  00000000    10000001
        //     |     r6  00000000    10000001
        //     |     r7  01111111    10000001
        //
        // Row 0 is every column, so it becomes column 0: every output row's bit 0.
        // The lone entry at `(3, 3)` is on the diagonal and stays put.
        let input = [
            0b1111_1111u8,
            0b0000_0000,
            0b0000_0000,
            0b0000_1000,
            0b0000_0000,
            0b0000_0000,
            0b0000_0000,
            0b1111_1110,
        ];
        let expected = [
            0b0000_0001u8,
            0b1000_0001,
            0b1000_0001,
            0b1000_1001,
            0b1000_0001,
            0b1000_0001,
            0b1000_0001,
            0b1000_0001,
        ];

        let mut rows: [PackedGf2x8; 8] = core::array::from_fn(|r| PackedGf2x8::new(input[r]));
        PackedGf2x8::transpose(&mut rows);

        let got: [u8; 8] = core::array::from_fn(|r| rows[r].words()[0]);
        assert_eq!(got, expected);
    }

    #[test]
    fn the_transpose_reads_lanes_the_way_the_packings_write_them() {
        // Invariant: the transpose and the lane accessors cannot drift apart.
        //
        //     before:  row r, lane c
        //     after:   row c, lane r, and nothing else
        for r in [0usize, 1, 31, 62, 63] {
            for c in [0usize, 1, 32, 63] {
                let mut rows = [PackedGf2x64::ZERO; 64];
                rows[r].set(c, Gf2::ONE);

                PackedGf2x64::transpose(&mut rows);

                let mut expected = [PackedGf2x64::ZERO; 64];
                expected[c].set(r, Gf2::ONE);
                assert_eq!(rows, expected, "entry ({r}, {c})");
            }
        }
    }

    #[test]
    fn a_wide_transpose_moves_entries_across_word_boundaries() {
        // Invariant: an entry crosses blocks, the step a per-word kernel would miss.
        //
        //     entry (10, 100)   block (0, 1), word 1 of row 10
        //     entry (100, 10)   block (1, 0), word 0 of row 100
        let mut rows = [PackedGf2x128::ZERO; 128];
        rows[10].set(100, Gf2::ONE);

        PackedGf2x128::transpose(&mut rows);

        let mut expected = [PackedGf2x128::ZERO; 128];
        expected[100].set(10, Gf2::ONE);
        assert_eq!(rows, expected);
    }
}
