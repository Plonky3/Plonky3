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
//!
//! With AVX-512 VBMI and GFNI, 64-bit words take a byte-shuffle form instead.

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

/// The bit transpose of an 8 x 8 matrix whose byte `r` is row `r`: bit `8r + c` moves to `8c + r`.
///
/// Three exchange rounds at block sizes 1, 2 and 4.
#[inline]
#[must_use]
pub const fn transpose_8x8(mut value: u64) -> u64 {
    let mut exchange = (value ^ (value >> 7)) & 0x00aa_00aa_00aa_00aa;
    value ^= exchange ^ (exchange << 7);
    exchange = (value ^ (value >> 14)) & 0x0000_cccc_0000_cccc;
    value ^= exchange ^ (exchange << 14);
    exchange = (value ^ (value >> 28)) & 0x0000_0000_f0f0_f0f0;
    value ^ exchange ^ (exchange << 28)
}

/// The transpose of 64-bit words by byte shuffles and affine maps over `GF(2)`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
mod gfni {
    use core::any::Any;
    use core::arch::x86_64::*;

    use super::super::packing::PackedGf2;
    use super::super::underlier::{Underlier, Word};
    use super::words_of;

    /// Transpose a square matrix of 64-bit words, whatever a row's word count.
    ///
    /// With `n` words a row, the matrix is an `n x n` grid of 64 x 64 blocks.
    /// Block `(i, j)` is word `j` of rows `64*i ..`.
    /// Transposing the whole is transposing each block and swapping block `(i, j)` with `(j, i)`.
    ///
    /// Not inlined, so the generic transpose stays small for every other word width.
    #[inline(never)]
    pub(super) fn transpose<U: Underlier>(rows: &mut [PackedGf2<U>]) {
        // A constant once the underlier is known, so every index below folds.
        let n = U::WORDS;
        let words = words_of(rows);
        assert_eq!(
            words.len(),
            64 * n * n,
            "a bit transpose needs one row per lane"
        );

        // A one-word row is a single block, transposed where it lies.
        if n == 1 {
            transpose_words(words.try_into().expect("one block of 64 words"));
            return;
        }

        // Row group `i` is rows `64*i ..`; its column `j` of words is block `(i, j)`.
        for i in 0..n {
            let (upper, lower) = words[64 * n * i..].split_at_mut(64 * n);

            // A diagonal block stays where it is.
            let mut diagonal = column(upper, n, i);
            transpose_words(&mut diagonal);
            set_column(upper, n, i, &diagonal);

            for (j, lower) in ((i + 1)..n).zip(lower.chunks_exact_mut(64 * n)) {
                let mut above = column(upper, n, j);
                let mut below = column(lower, n, i);
                transpose_words(&mut above);
                transpose_words(&mut below);

                // Each transposed block lands in the other's place.
                set_column(upper, n, j, &below);
                set_column(lower, n, i, &above);
            }
        }
    }

    /// Word `j` of each of the 64 rows of `n` words in `group`.
    #[inline(always)]
    fn column<T: Word>(group: &[T], n: usize, j: usize) -> [T; 64] {
        let mut out = [T::ZERO; 64];
        for (word, row) in out.iter_mut().zip(group.chunks_exact(n)) {
            *word = row[j];
        }
        out
    }

    /// Overwrite word `j` of each of the 64 rows of `n` words in `group`.
    #[inline(always)]
    fn set_column<T: Word>(group: &mut [T], n: usize, j: usize, column: &[T; 64]) {
        for (row, &word) in group.chunks_exact_mut(n).zip(column) {
            row[j] = word;
        }
    }

    /// Transpose a block of 64-bit words, which are `u64`s seen through the word trait.
    #[inline]
    fn transpose_words<T: Word>(block: &mut [T; 64]) {
        let block = (block as &mut dyn Any)
            .downcast_mut::<[u64; 64]>()
            .expect("a 64-bit word is a u64");
        transpose_block(block);
    }

    /// Transpose the 64 x 64 block whose row `r` is word `r`.
    ///
    /// Write bit `(r, c)` with `r = 8R + r'` and `c = 8C + c'`.
    /// Register `R` holds rows `8R ..`, so the bit is at register `R`, word `r'`, byte `C`, bit `c'`.
    /// The transpose moves it to register `C`, word `c'`, byte `R`, bit `r'`:
    ///
    /// ```text
    ///     (R; r', C, c')
    ///       -> (R; C, c', r')    a byte permute and an affine map, per register
    ///       -> (C; R, c', r')    an 8 x 8 transpose of words across the registers
    ///       -> (C; c', R, r')    a byte permute, per register
    /// ```
    ///
    /// `vgf2p8affineqb` with the constant `0x8040201008040201` computes, per word `A`:
    ///
    /// ```text
    ///     result byte j bit i  =  bit j of A's byte 7 - i
    /// ```
    ///
    /// So the first permute also reverses the rows, and the map undoes it.
    ///
    /// The word transpose pairs words inside each 128-bit lane first.
    /// Only its last two rounds cross lanes, one 128-bit lane or 256-bit half at a time.
    #[inline]
    fn transpose_block(block: &mut [u64; 64]) {
        /// Byte `8C + c'` takes byte `C` of word `7 - c'`, rows reversed for the affine map.
        const GATHER: [u8; 64] = {
            let mut idx = [0u8; 64];
            let mut i = 0;
            while i < 64 {
                idx[i] = ((7 - i % 8) * 8 + i / 8) as u8;
                i += 1;
            }
            idx
        };

        /// Byte `8c' + R` takes byte `8R + c'`: an 8 x 8 byte transpose.
        const SCATTER: [u8; 64] = {
            let mut idx = [0u8; 64];
            let mut i = 0;
            while i < 64 {
                idx[i] = ((i % 8) * 8 + i / 8) as u8;
                i += 1;
            }
            idx
        };

        /// The low 128-bit lane of each 256-bit half of `a`, then of `b`.
        const LOW_LANES: [u64; 8] = [0, 1, 8, 9, 4, 5, 12, 13];

        /// The high 128-bit lane of each 256-bit half of `a`, then of `b`.
        const HIGH_LANES: [u64; 8] = [2, 3, 10, 11, 6, 7, 14, 15];

        // SAFETY: every load and store is 64 bytes inside the block or a constant table.
        unsafe {
            let gather = _mm512_loadu_si512(GATHER.as_ptr().cast());
            let scatter = _mm512_loadu_si512(SCATTER.as_ptr().cast());
            let low_lanes = _mm512_loadu_si512(LOW_LANES.as_ptr().cast());
            let high_lanes = _mm512_loadu_si512(HIGH_LANES.as_ptr().cast());
            let unit = _mm512_set1_epi64(0x8040_2010_0804_0201_u64 as i64);

            // Phase 1: word `C` byte `c'` of register `R` becomes column `8C + c'` of its rows.
            let y: [__m512i; 8] = core::array::from_fn(|r| {
                let rows = _mm512_loadu_si512(block.as_ptr().add(8 * r).cast());
                _mm512_gf2p8affine_epi64_epi8::<0>(unit, _mm512_permutexvar_epi8(gather, rows))
            });

            // Phase 2: word `C` of register `R` moves to word `R` of register `C`.
            //
            //     distance 1   unpack, inside each 128-bit lane
            //     distance 2   two-source permute of 128-bit lanes
            //     distance 4   shuffle of 256-bit halves
            let t: [__m512i; 8] = core::array::from_fn(|i| {
                let (a, b) = (y[i & !1], y[i | 1]);
                if i & 1 == 0 {
                    _mm512_unpacklo_epi64(a, b)
                } else {
                    _mm512_unpackhi_epi64(a, b)
                }
            });
            let u: [__m512i; 8] = core::array::from_fn(|i| {
                let (a, b) = (t[i & !2], t[i | 2]);
                if i & 2 == 0 {
                    _mm512_permutex2var_epi64(a, low_lanes, b)
                } else {
                    _mm512_permutex2var_epi64(a, high_lanes, b)
                }
            });
            let w: [__m512i; 8] = core::array::from_fn(|i| {
                let (a, b) = (u[i & !4], u[i | 4]);
                if i & 4 == 0 {
                    _mm512_shuffle_i64x2::<0b01_00_01_00>(a, b)
                } else {
                    _mm512_shuffle_i64x2::<0b11_10_11_10>(a, b)
                }
            });

            // Phase 3: word `R` byte `c'` of register `C` becomes byte `R` of row `8C + c'`.
            for (c, w) in w.into_iter().enumerate() {
                let rows = _mm512_permutexvar_epi8(scatter, w);
                _mm512_storeu_si512(block.as_mut_ptr().add(8 * c).cast(), rows);
            }
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

        // 64-bit words have a byte-shuffle form of the whole transpose where the build allows it.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512vbmi",
            target_feature = "gfni"
        ))]
        if Self::WORD_BITS == 64 {
            gfni::transpose(rows);
            return;
        }

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
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::{Rng, RngExt};

    use super::super::packing::PackedGf2;
    use super::super::underlier::Underlier;
    use crate::{Gf2, PackedGf2x8, PackedGf2x64, PackedGf2x128, transpose_8x8};

    /// A block of `N` 64-bit words, for word counts no packing of the crate has.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    #[repr(transparent)]
    struct Words<const N: usize>([u64; N]);

    impl<const N: usize> Default for Words<N> {
        fn default() -> Self {
            Self([0; N])
        }
    }

    // SAFETY: a transparent array of `N` words has their size and alignment.
    unsafe impl<const N: usize> Underlier for Words<N> {
        type Word = u64;

        const WORDS: usize = N;
        const ZERO: Self = Self([0; N]);
        const ONES: Self = Self([u64::MAX; N]);

        fn words(&self) -> &[u64] {
            &self.0
        }

        fn words_mut(&mut self) -> &mut [u64] {
            &mut self.0
        }

        fn from_words_fn(f: impl FnMut(usize) -> u64) -> Self {
            Self(core::array::from_fn(f))
        }

        fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
            Self(core::array::from_fn(|_| rng.random()))
        }
    }

    /// Transpose a random `64N x 64N` matrix and compare every entry with its mirror.
    fn check_any_word_count<const N: usize>() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let original: Vec<PackedGf2<Words<N>>> = (0..64 * N)
            .map(|_| {
                PackedGf2::new(Words(core::array::from_fn(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                })))
            })
            .collect();

        let mut rows = original.clone();
        PackedGf2::transpose(&mut rows);

        for (r, row) in rows.iter().enumerate() {
            for (c, column) in original.iter().enumerate() {
                assert_eq!(row.get(c), column.get(r), "entry ({r}, {c})");
            }
        }
    }

    #[test]
    fn every_word_count_the_underlier_contract_allows_transposes() {
        // Invariant: no valid underlier panics, powers of two or not.
        check_any_word_count::<3>();
        check_any_word_count::<16>();
    }

    proptest! {
        #[test]
        fn transpose_8x8_moves_every_bit_across_the_diagonal(value in any::<u64>()) {
            // Invariant: bit 8r + c of the input is bit 8c + r of the output.
            let out = transpose_8x8(value);
            for r in 0..8 {
                for c in 0..8 {
                    prop_assert_eq!((out >> (8 * c + r)) & 1, (value >> (8 * r + c)) & 1);
                }
            }
        }
    }

    #[test]
    fn a_slice_of_words_reads_as_packings_in_place() {
        // Fixture state: the identity matrix, transposed through the in-place view.
        let mut words: [u64; 64] = core::array::from_fn(|r| 1 << r);
        PackedGf2x64::transpose(PackedGf2x64::from_bits_slice_mut(&mut words));
        assert_eq!(words, core::array::from_fn(|r| 1 << r));
        assert_eq!(PackedGf2x64::from_bits_slice(&words)[5].to_bits(), 1 << 5);
    }

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
        //
        // Every one of the 4096 entries, so each route through a shuffle network is pinned.
        for r in 0..64 {
            for c in 0..64 {
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
