//! The square bit transpose: row `i` of the result is column `i` of the input.
//!
//! # What it is for
//!
//! A bit slice is one row of a bit matrix, and the proving stack needs both readings of the
//! same bits:
//!
//! ```text
//!     row-major     one slice = one row of the trace, W columns at a time
//!     column-major  one slice = one column of the trace, W rows at a time
//! ```
//!
//! Turning one into the other is exactly this transpose.
//!
//! # The algorithm
//!
//! A `2n x 2n` bit matrix splits into four `n x n` quadrants:
//!
//! ```text
//!     | A  B |            | A^T  C^T |
//!     | C  D |    ->      | B^T  D^T |
//! ```
//!
//! So transposing means swapping the two off-diagonal quadrants and transposing each one.
//!
//! Doing the swap first, for every quadrant size at once, is what makes it cheap: with one
//! row per machine word, swapping the two off-diagonal quadrants of every `2n`-block is a
//! shift, a mask and two exclusive ors per row pair.
//!
//! Repeating that for `n = B/2, B/4, ..., 1` leaves nothing to transpose, so the recursion
//! flattens into `log2(B)` passes over the rows.
//!
//! This is the blocked swap of Hacker's Delight, figure 7-6, mirrored for a lane order that
//! puts column `0` at the least significant bit rather than the most.
//!
//! # Rows wider than a word
//!
//! A row of `n` words is an `n`-wide strip of `B x B` blocks, so a `nB x nB` matrix is an
//! `n x n` grid of them, and the identity above applies one level up:
//!
//! ```text
//!     phase 1   transpose every block where it lies
//!     phase 2   swap block (i, j) with block (j, i)
//! ```
//!
//! Phase 1 acts inside a word and pairs rows within a group of `B`, so it treats every word
//! of a row alike, and runs as one flat sweep over the words of a whole run of rows.
//!
//! Keeping that sweep contiguous is what keeps it at the target's vector width.
//! Gathering one block into a buffer instead costs a strided read per block, which a wide
//! target turns into gather instructions that run slower than the transpose itself.
//!
//! Phase 2 is a plain exchange of words, with no arithmetic at all, and costs about one
//! percent of the whole.
//!
//! # Why in place, and why on the packings
//!
//! In place, because a `512 x 512` matrix is 32 KB and the algorithm never needs a second
//! copy of it: every pass rewrites the rows it reads.
//!
//! On the packings rather than on raw integers, because a row of the matrix *is* a packing:
//! the type then carries the lane order, and the row count is checked against the width the
//! type advertises.
//!
//! # Only square, only a power of two
//!
//! The blocked swap needs the two off-diagonal quadrants to have the same shape, which holds
//! only for a square matrix whose side is a power of two.
//!
//! Every packing width is such a side, so the only shape accepted is exactly as many rows as
//! the packing has lanes.
//!
//! A rectangular bit matrix is not supported.
//! Cut it into square blocks and transpose the blocks.

use core::slice;

use super::packing::{PackedGf2, block_mask};
use super::underlier::{Underlier, Word};

/// The backing words of a run of packings, as one flat slice.
///
/// Packings are contiguous and a packing is exactly its words, so a run of `k` of them is
/// `k * n` words with nothing in between.
///
/// Flattening lets a pass that treats every word alike run as one straight sweep, which keeps
/// it at the target's vector width instead of a strided access per word index.
#[inline]
const fn words_of<U: Underlier>(rows: &mut [PackedGf2<U>]) -> &mut [U::Word] {
    // The layout this rests on, checked at compile time.
    //
    //     size      exactly the backing words, so no padding between packings
    //     align     at least a word's, so every packing starts word-aligned
    const {
        assert!(size_of::<PackedGf2<U>>() == U::WORDS * size_of::<U::Word>());
        assert!(align_of::<PackedGf2<U>>() >= align_of::<U::Word>());
    }

    let len = rows.len() * U::WORDS;

    // SAFETY: by the assertions above, the run occupies exactly `len` contiguous, suitably
    // aligned words of the same allocation, and every bit pattern of a word is valid.
    unsafe { slice::from_raw_parts_mut(rows.as_mut_ptr().cast::<U::Word>(), len) }
}

/// One pass: exchange the two off-diagonal quadrants of every block of `2 * STEP` rows and
/// `2 * STEP` columns, in every word of every row.
///
/// The quadrant side is a constant so the compiler knows every trip count, and can unroll the
/// short passes instead of vectorising them blindly.
///
/// A quadrant at least as wide as a word lies across words, which is the grid exchange rather
/// than this pass, so the call folds away at compile time.
// The chunk size is `2 * STEP`, which is not a valid const generic argument on stable, so
// the fixed-size chunking the lint suggests is not expressible here.
#[allow(clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn stage<U: Underlier, const STEP: usize>(group: &mut [PackedGf2<U>]) {
    if STEP >= <U::Word as Word>::BITS {
        return;
    }

    // Selects the low `STEP` bits of every `2 * STEP`-bit block, which is the column range
    // of the lower-left quadrant of each block.
    let mask = U::Word::from_u64(block_mask(STEP));

    // Rows split into runs of `2 * STEP`: the first half holds the upper quadrants, the
    // second half the lower ones.
    for run in group.chunks_exact_mut(2 * STEP) {
        let (top, bottom) = run.split_at_mut(STEP);

        // Word `w` of row `k` of the top half pairs with word `w` of row `k` of the bottom
        // half, so the two halves pair off word for word and the sweep stays flat.
        //
        //     upper right = bits [STEP, 2*STEP) of a
        //     lower left  = bits [0, STEP)      of b
        for (a, b) in words_of(top).iter_mut().zip(words_of(bottom)) {
            // The difference of the two quadrants, aligned at the bottom of the word.
            let diff = ((*a >> STEP) ^ *b) & mask;

            // Adding the difference to each side is the exchange.
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

        // Phase 1: transpose every block where it lies, a row-group at a time.
        //
        // The ladder covers every quadrant side up to half a 64-bit word; the passes wider
        // than the backing word are skipped inside the pass itself.
        for group in rows.chunks_exact_mut(block) {
            stage::<U, 32>(group);
            stage::<U, 16>(group);
            stage::<U, 8>(group);
            stage::<U, 4>(group);
            stage::<U, 2>(group);
            stage::<U, 1>(group);
        }

        // Phase 2: swap the grid of blocks across its own diagonal.
        //
        // Block `(i, j)` is word `j` of the rows `block*i .. block*i + block`, so the
        // exchange is word for word and touches no bits.
        //
        // A row that is a single word has no grid, and both loops are empty.
        for i in 0..U::WORDS {
            for j in (i + 1)..U::WORDS {
                // The two blocks lie in disjoint row ranges because `i < j`.
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

    /// One test module per matrix side.
    ///
    /// The reference is the definition itself, read through the lane accessors, so a kernel
    /// that agrees with it cannot disagree with the lane order the packings promise.
    macro_rules! transpose_tests {
        ($module:ident, $name:ident, $underlier:ty, $word:ty, $words:literal, $width:literal) => {
            mod $module {
                use super::*;
                use crate::bitslice::underlier::Underlier;
                use crate::$name;

                /// A packing over the given backing words, lowest lanes first.
                fn from_words(words: [$word; $words]) -> $name {
                    $name::new(<$underlier>::from_words_fn(|i| words[i]))
                }

                /// The transpose written straight from its definition.
                ///
                /// Row `r` of the result is column `r` of the input, one lane at a time.
                fn reference(rows: &[$name; $width]) -> [$name; $width] {
                    core::array::from_fn(|r| $name::from_fn(|c| rows[c].get(r)))
                }

                /// A matrix whose entry `(r, c)` is a fixed, easily distinguished function of
                /// its position.
                ///
                /// Any permutation that sends an entry to the wrong place changes the value.
                fn fixture() -> [$name; $width] {
                    core::array::from_fn(|r| {
                        $name::from_fn(|c| Gf2::from_bool((r * 7 + c * 13 + r * c) % 3 == 0))
                    })
                }

                #[test]
                fn transpose_matches_the_definition() {
                    // Fixture state: entry (r, c) set iff 7r + 13c + rc is a multiple of three.
                    let mut rows = fixture();
                    let expected = reference(&rows);

                    $name::transpose(&mut rows);

                    assert_eq!(rows, expected);
                }

                #[test]
                fn transposing_twice_is_the_identity() {
                    // Transposition is an involution, so two passes must restore the input.
                    let original = fixture();
                    let mut rows = original;

                    $name::transpose(&mut rows);
                    $name::transpose(&mut rows);

                    assert_eq!(rows, original);
                }

                #[test]
                fn the_extremes_are_fixed_or_swapped_as_expected() {
                    // All zeros and all ones are symmetric, so the transpose leaves them alone.
                    let mut zeros = [$name::ZERO; $width];
                    $name::transpose(&mut zeros);
                    assert_eq!(zeros, [$name::ZERO; $width]);

                    let mut ones = [$name::ONE; $width];
                    $name::transpose(&mut ones);
                    assert_eq!(ones, [$name::ONE; $width]);

                    // A single entry at a corner moves to the mirrored corner.
                    //
                    //     entry (0, W-1)   ->   entry (W-1, 0)
                    let mut corner = [$name::ZERO; $width];
                    corner[0].set($width - 1, Gf2::ONE);
                    $name::transpose(&mut corner);

                    let mut expected = [$name::ZERO; $width];
                    expected[$width - 1].set(0, Gf2::ONE);
                    assert_eq!(corner, expected);
                }

                #[test]
                fn the_identity_matrix_is_its_own_transpose() {
                    // The diagonal is the fixed set of the transpose, so lighting only the
                    // diagonal gives a matrix the kernel must not change at all.
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
                        // A cheap xorshift fills the matrix, so the case index alone decides
                        // the whole fixture and a failure reproduces from the printed seed.
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
        // Invariant: row `i` of the output is column `i` of the input.
        //
        // The matrix below is drawn with column 0 on the left, which is the least significant
        // bit of each row, so the literals read right to left against the picture.
        //
        //     input                       output
        //     c0 c1 c2 c3 c4 c5 c6 c7     c0 c1 c2 c3 c4 c5 c6 c7
        // r0   1  1  1  1  1  1  1  1  r0  1  0  0  0  0  0  0  0
        // r1   0  0  0  0  0  0  0  0  r1  1  0  0  0  0  0  0  1
        // r2   0  0  0  0  0  0  0  0  r2  1  0  0  0  0  0  0  1
        // r3   0  0  0  1  0  0  0  0  r3  1  0  0  1  0  0  0  1
        // r4   0  0  0  0  0  0  0  0  r4  1  0  0  0  0  0  0  1
        // r5   0  0  0  0  0  0  0  0  r5  1  0  0  0  0  0  0  1
        // r6   0  0  0  0  0  0  0  0  r6  1  0  0  0  0  0  0  1
        // r7   0  1  1  1  1  1  1  1  r7  1  0  0  0  0  0  0  1
        //
        // Row 0 is every column, so it becomes column 0 of the output: every row's bit 0.
        //
        // Row 7 is every column but the first, so it becomes column 7 of the output: every
        // row's bit 7, except row 0's.
        //
        // The lone entry at (3, 3) sits on the diagonal and therefore does not move.
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
        // Invariant: the two halves of this module cannot drift apart.
        //
        // Setting lane `c` of row `r` and transposing must leave lane `r` of row `c` set, for
        // every position in a `64 x 64` matrix, with nothing else touched.
        //
        //     before:  row r, lane c
        //     after:   row c, lane r
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
        // Invariant: the block decomposition must carry an entry from block (i, j) into
        // block (j, i), which is the step a purely per-word kernel would miss.
        //
        // Fixture: a `128 x 128` matrix with one entry in the upper-right `64 x 64` block.
        //
        //     entry (10, 100)   lives in block (0, 1), word 1 of row 10
        //     entry (100, 10)   lives in block (1, 0), word 0 of row 100
        let mut rows = [PackedGf2x128::ZERO; 128];
        rows[10].set(100, Gf2::ONE);

        PackedGf2x128::transpose(&mut rows);

        let mut expected = [PackedGf2x128::ZERO; 128];
        expected[100].set(10, Gf2::ONE);
        assert_eq!(rows, expected);
    }
}
