//! Multiplication by a small tower subfield element, as a byte-wise `GF(2)` linear map.
//!
//! The tower is byte-aligned above `GF(2^8)`.
//! So a level of `B` bytes has the groups of `D` bytes as coordinates, for each `D` dividing it.
//!
//! Scaling by a subfield element acts on every group alike.
//! That map is a `D x D` matrix of `8 x 8` blocks, and one block is one instruction.

use crate::lanes::{ByteLanes, butterfly_run, group_mask};

/// The `8 x 8` `GF(2)` blocks of a linear map on a group of `D` bytes.
///
/// ```text
///     out_byte_k = sum_j blocks[k][j] . in_byte_j
/// ```
pub(crate) type Blocks<const D: usize> = [[u64; D]; D];

/// Multiply by the tower generator `X` of the level of `bits` bits.
///
/// ```text
///     X^2 = a X + 1                       a is the generator of the level below
///     X . (a_0 + a_1 X) = a_1 + (a_0 + a . a_1) X
/// ```
const fn mul_alpha(a: u64, bits: usize) -> u64 {
    // The base of the tower has `X = 1`, so nothing moves.
    if bits == 1 {
        return a;
    }
    let half = bits / 2;
    let mask = u64::MAX >> (64 - half);

    // The coefficients of `a_0 + a_1 X` over the level below.
    let lo = a & mask;
    let hi = a >> half;

    // The old `X` coefficient drops down, and the modulus folds back into the new one.
    hi | ((lo ^ mul_alpha(hi, half)) << half)
}

/// Multiply in the tower level of `bits` bits, in the tower basis.
///
/// Karatsuba over the quadratic extension, so the middle coefficient costs one product:
///
/// ```text
///     (a_0 + a_1 X)(b_0 + b_1 X)  =  a_0 b_0 + a_1 b_1  +  (a_0 b_1 + a_1 b_0 + a a_1 b_1) X
/// ```
const fn tower_mul(a: u64, b: u64, bits: usize) -> u64 {
    // `GF(2)` multiplication is the logical and.
    if bits == 1 {
        return a & b;
    }
    let half = bits / 2;
    let mask = u64::MAX >> (64 - half);

    // The coefficients of both operands over the level below.
    let (a0, a1) = (a & mask, a >> half);
    let (b0, b1) = (b & mask, b >> half);

    // The two diagonal products, then the middle one recovered from their sum.
    let low = tower_mul(a0, b0, half);
    let high = tower_mul(a1, b1, half);
    let middle = tower_mul(a0 ^ a1, b0 ^ b1, half) ^ low ^ high;

    // `X^2 = a X + 1` sends the top product into both coefficients.
    (low ^ high) | ((middle ^ mul_alpha(high, half)) << half)
}

/// The blocks of multiplication by `scalar` in the tower level of `D` bytes.
///
/// Column `8j + b` of the map is the scalar times the basis element of that input bit.
/// The instruction's layout then fixes where each entry of that column belongs:
///
/// ```text
///     out_byte_k[i] = sum_b blocks[k][j][8(7 - i) + b] . in_byte_j[b]
/// ```
const fn map_blocks<const D: usize>(scalar: u64) -> Blocks<D> {
    let bits = 8 * D;
    let mut blocks = [[0u64; D]; D];

    // Walk the input bits, one column of the map at a time.
    let mut j = 0;
    while j < D {
        let mut b = 0;
        while b < 8 {
            // The image of the basis element at this bit position.
            let column = tower_mul(scalar, 1 << (8 * j + b), bits);

            // Scatter that column's bits into the quadword each output byte reads.
            let mut k = 0;
            while k < D {
                let mut i = 0;
                while i < 8 {
                    if (column >> (8 * k + i)) & 1 == 1 {
                        blocks[k][j] |= 1 << (8 * (7 - i) + b);
                    }
                    i += 1;
                }
                k += 1;
            }
            b += 1;
        }
        j += 1;
    }
    blocks
}

/// The blocks of multiplication by every element of the tower `GF(2^8)`, which is 2 KiB.
static BYTE_BLOCKS: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut c = 0;
    while c < 256 {
        table[c] = map_blocks::<1>(c as u64)[0][0];
        c += 1;
    }
    table
};

/// The blocks of multiplication in the tower `GF(2^16)`, split by byte of the multiplier.
///
/// Blocks are linear in the multiplier, so one entry per byte sums to the whole map.
/// Two tables of 256 entries come to 16 KiB, against 2 MiB for one unsplit table.
static WORD_BLOCKS: [[Blocks<2>; 256]; 2] = {
    let mut table = [[[[0u64; 2]; 2]; 256]; 2];
    let mut byte = 0;
    while byte < 2 {
        let mut c = 0;
        while c < 256 {
            table[byte][c] = map_blocks::<2>((c as u64) << (8 * byte));
            c += 1;
        }
        byte += 1;
    }
    table
};

/// The blocks of multiplication by a one-byte multiplier.
#[inline]
pub(crate) fn byte_blocks(scalar: u8) -> Blocks<1> {
    [[BYTE_BLOCKS[scalar as usize]]]
}

/// The blocks of multiplication by a two-byte multiplier.
#[inline]
pub(crate) fn word_blocks(scalar: u16) -> Blocks<2> {
    let low = &WORD_BLOCKS[0][scalar as u8 as usize];
    let high = &WORD_BLOCKS[1][(scalar >> 8) as usize];

    // Blocks are linear in the multiplier, so the two halves combine by exclusive or.
    core::array::from_fn(|k| core::array::from_fn(|j| low[k][j] ^ high[k][j]))
}

/// The blocks of multiplication by a four-byte multiplier, from the two-byte table alone.
///
/// The four-byte level is `GF(2^16)[X]` modulo `X^2 + a X + 1`.
/// A multiplier `t_0 + t_1 X` therefore acts as a `2 x 2` matrix over the level below:
///
/// ```text
///     out_0  =  t_0 v_0  +  t_1 v_1
///     out_1  =  t_1 v_0  +  (t_0 + a t_1) v_1
/// ```
#[inline]
pub(crate) fn dword_blocks(scalar: u32) -> Blocks<4> {
    // The coefficients of the multiplier over the two-byte level.
    let (t0, t1) = (scalar as u16, (scalar >> 16) as u16);

    // The modulus folds the top product back into both coefficients.
    let folded = t0 ^ mul_alpha(t1 as u64, 16) as u16;

    // Each entry of that matrix expands into the byte blocks of its own multiplier.
    let matrix = [[t0, t1], [t1, folded]];
    let expanded: [[Blocks<2>; 2]; 2] =
        core::array::from_fn(|k| core::array::from_fn(|j| word_blocks(matrix[k][j])));

    // Byte `2k + i` of the output reads byte `2j + m` of the input through entry `[k][j]`.
    core::array::from_fn(|row| {
        core::array::from_fn(|col| expanded[row / 2][col / 2][row % 2][col % 2])
    })
}

/// The image of one register under a `D`-byte-group map.
///
/// A register interleaves the groups, so byte position `p` carries coordinate `p mod D`:
///
/// ```text
///     x                   in_0 in_1 .. in_{D-1} in_0 in_1 ..
///     rotate(x, r)        in_r in_{r+1} ..                      indices modulo D
/// ```
///
/// At rotation `r = (j - k) mod D` output coordinate `k` and its input `j` share a position.
/// So one rotation serves every output at once, under the block the write mask selects.
#[inline(always)]
fn map_groups<L: ByteLanes, const D: usize>(x: L, blocks: &Blocks<D>) -> L {
    // One rotation's contribution: residue-`k` positions take the block for input `k + r`.
    let fold = |rotated: L, r: usize| {
        let mut acc = rotated.affine(blocks[0][r]);
        for k in 1..D {
            acc = acc.affine_merge(rotated, blocks[k][(k + r) % D], group_mask(D, k));
        }
        acc
    };

    // The unrotated register already pairs each coordinate with itself.
    let mut out = fold(x, 0);
    for r in 1..D {
        out = out.xor(fold(x.rotate_group::<D>(r), r));
    }
    out
}

/// Apply the butterfly with a `D`-byte-group twiddle, returning the bytes it covered.
///
/// # Panics
/// Panics if the two runs have different lengths.
#[inline]
pub(crate) fn subfield_butterfly<L: ByteLanes, const D: usize, const INVERSE: bool>(
    lo: &mut [u8],
    hi: &mut [u8],
    blocks: &Blocks<D>,
) -> usize {
    butterfly_run::<L, INVERSE>(lo, hi, |x| map_groups::<L, D>(x, blocks))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, TowerLevel};
    use proptest::prelude::*;

    use super::{byte_blocks, dword_blocks, map_blocks, tower_mul, word_blocks};
    use crate::test_util::{LANE_BYTES, Model};

    /// Zero, one, a lone top basis element of each level, a full lower coordinate, all bits.
    const CORNERS: [u32; 8] = [0, 1, 0xff, 0x80, 0x8000, 0xffff, 0x8000_0000, u32::MAX];

    // Everything below rearranges this product, so a mismatch here invalidates all of it.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn the_const_tower_product_matches_the_field(a in any::<u32>(), b in any::<u32>()) {
            // Eight bits: the base case, one turn of the recursion above `GF(2)`.
            let want = BinaryField8::from_repr(a as u8) * BinaryField8::from_repr(b as u8);
            prop_assert_eq!(tower_mul(a as u8 as u64, b as u8 as u64, 8), want.to_repr() as u64);

            // Sixteen bits: where the two-byte table comes from.
            let want = BinaryField16::from_repr(a as u16) * BinaryField16::from_repr(b as u16);
            prop_assert_eq!(
                tower_mul(a as u16 as u64, b as u16 as u64, 16),
                want.to_repr() as u64
            );

            // Thirty-two bits: the level the four-byte blocks are assembled for.
            let want = BinaryField32::from_repr(a) * BinaryField32::from_repr(b);
            prop_assert_eq!(tower_mul(a as u64, b as u64, 32), want.to_repr() as u64);
        }
    }

    /// The image of a byte under one `8 x 8` block, in the instruction's own convention.
    fn apply_block(matrix: u64, input: u8) -> u8 {
        let rows = matrix.to_le_bytes();
        let mut out = 0u8;

        // Output bit `i` reads row `7 - i`, pairing bit `b` of it with bit `b` of the input.
        for i in 0..8 {
            if (rows[7 - i] & input).count_ones() % 2 == 1 {
                out |= 1 << i;
            }
        }
        out
    }

    /// One group of `D` bytes pushed through the blocks, byte by byte.
    fn through_blocks<const D: usize>(blocks: &super::Blocks<D>, input: [u8; D]) -> [u8; D] {
        // Output byte `k` sums the images of every input byte of the same group.
        core::array::from_fn(|k| {
            (0..D).fold(0u8, |acc, j| acc ^ apply_block(blocks[k][j], input[j]))
        })
    }

    #[test]
    fn a_one_byte_map_scales_a_byte_of_the_tower() {
        // Every multiplier against every value is 65536 cases, so this sweep is exhaustive.
        for c in 0..=u8::MAX {
            let blocks = byte_blocks(c);
            for v in 0..=u8::MAX {
                let want = BinaryField8::from_repr(c) * BinaryField8::from_repr(v);
                assert_eq!(through_blocks(&blocks, [v]), [want.to_repr()], "c {c:#x}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        /// The two-byte blocks reproduce a `GF(2^16)` product, including the byte split.
        #[test]
        fn a_two_byte_map_scales_a_word_of_the_tower(c in any::<u16>(), v in any::<u16>()) {
            let want = BinaryField16::from_repr(c) * BinaryField16::from_repr(v);
            let got = through_blocks(&word_blocks(c), v.to_le_bytes());
            prop_assert_eq!(u16::from_le_bytes(got), want.to_repr());
        }

        /// The four-byte blocks reproduce a `GF(2^32)` product through the two-byte ones.
        #[test]
        fn a_four_byte_map_scales_a_doubleword_of_the_tower(c in any::<u32>(), v in any::<u32>()) {
            let want = BinaryField32::from_repr(c) * BinaryField32::from_repr(v);
            let got = through_blocks(&dword_blocks(c), v.to_le_bytes());
            prop_assert_eq!(u32::from_le_bytes(got), want.to_repr());
        }
    }

    #[test]
    fn the_assembled_tables_agree_with_the_direct_construction() {
        // Invariant: `blocks(a + b) = blocks(a) + blocks(b)`, which the byte split rests on.
        // Building each map from its own columns is what checks that independently.
        for &c in &CORNERS {
            assert_eq!(
                word_blocks(c as u16),
                map_blocks::<2>(c as u16 as u64),
                "word {c:#x}"
            );
            assert_eq!(dword_blocks(c), map_blocks::<4>(c as u64), "dword {c:#x}");
        }
    }

    #[test]
    fn the_maps_scale_at_the_corners() {
        // Every corner as the multiplier, against every corner as the value.
        for &c in &CORNERS {
            for &v in &CORNERS {
                let want = BinaryField32::from_repr(c) * BinaryField32::from_repr(v);
                let got = through_blocks(&dword_blocks(c), v.to_le_bytes());
                assert_eq!(
                    u32::from_le_bytes(got),
                    want.to_repr(),
                    "c {c:#x}, v {v:#x}"
                );
            }
        }
    }

    /// Three registers of payload, so a kernel runs its loop more than once.
    fn sample(seed: usize) -> Vec<u8> {
        (0..LANE_BYTES * 3)
            .map(|i| ((i * 31 + seed * 7 + 1) % 251) as u8)
            .collect()
    }

    /// The butterfly at the field level, one element at a time, as the reference.
    fn reference<F: TowerLevel>(lo: &mut [F], hi: &mut [F], t: F, inverse: bool) {
        for (u, v) in lo.iter_mut().zip(hi) {
            if inverse {
                // Recover the upper half, then take the scaled result out of the lower one.
                *v += *u;
                *u += t * *v;
            } else {
                // Scale the upper half into the lower one, then sum both into the upper one.
                *u += t * *v;
                *v += *u;
            }
        }
    }

    /// Reinterpret a byte buffer as elements of a byte-aligned tower level.
    // The element width comes from a type parameter, which a const chunk size cannot take.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn as_level<F: TowerLevel>(bytes: &[u8]) -> Vec<F> {
        bytes
            .chunks_exact(size_of::<F>())
            .map(|c| F::from_le_byte_iter(c.iter().copied()))
            .collect()
    }

    /// Drive one kernel over the model backend and compare against the field-level loop.
    fn kernel_agrees<F: TowerLevel, const D: usize, const INVERSE: bool>(
        blocks: &super::Blocks<D>,
        twiddle: F,
    ) -> Result<(), TestCaseError> {
        // Two runs with nothing in common, so a kernel that mixed them would show it.
        let (mut lo, mut hi) = (sample(1), sample(2));

        // The same bytes read as field elements, transformed the slow way.
        let (mut want_lo, mut want_hi) = (as_level::<F>(&lo), as_level::<F>(&hi));
        reference(&mut want_lo, &mut want_hi, twiddle, INVERSE);

        // The payload is a whole number of registers, so the kernel must cover all of it.
        let covered = super::subfield_butterfly::<Model, D, INVERSE>(&mut lo, &mut hi, blocks);
        prop_assert_eq!(covered, lo.len());

        prop_assert_eq!(as_level::<F>(&lo), want_lo);
        prop_assert_eq!(as_level::<F>(&hi), want_hi);
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// One byte: the same block repeats on every byte of an eight-byte level.
        #[test]
        fn the_one_byte_kernel_matches_the_field(t in any::<u8>()) {
            let blocks = byte_blocks(t);
            let twiddle = BinaryField64::from_repr(t as u64);
            kernel_agrees::<BinaryField64, 1, false>(&blocks, twiddle)?;
            kernel_agrees::<BinaryField64, 1, true>(&blocks, twiddle)?;
        }

        /// Two bytes: each pair mixes, and nothing crosses a pair.
        #[test]
        fn the_two_byte_kernel_matches_the_field(t in any::<u16>()) {
            let blocks = word_blocks(t);
            let twiddle = BinaryField64::from_repr(t as u64);
            kernel_agrees::<BinaryField64, 2, false>(&blocks, twiddle)?;
            kernel_agrees::<BinaryField64, 2, true>(&blocks, twiddle)?;
        }

        /// Four bytes: all four rotations and all four write masks are in play.
        #[test]
        fn the_four_byte_kernel_matches_the_field(t in any::<u32>()) {
            let blocks = dword_blocks(t);
            let twiddle = BinaryField64::from_repr(t as u64);
            kernel_agrees::<BinaryField64, 4, false>(&blocks, twiddle)?;
            kernel_agrees::<BinaryField64, 4, true>(&blocks, twiddle)?;
        }
    }

    #[test]
    fn the_kernels_match_the_field_at_the_corners() {
        // Every corner multiplier, at the narrowest level each group size can run on.
        for &t in &CORNERS {
            kernel_agrees::<BinaryField8, 1, false>(
                &byte_blocks(t as u8),
                BinaryField8::from_repr(t as u8),
            )
            .unwrap_or_else(|e| panic!("one-byte twiddle {t:#x}: {e}"));

            kernel_agrees::<BinaryField16, 2, true>(
                &word_blocks(t as u16),
                BinaryField16::from_repr(t as u16),
            )
            .unwrap_or_else(|e| panic!("two-byte twiddle {t:#x}: {e}"));

            kernel_agrees::<BinaryField32, 4, false>(&dword_blocks(t), BinaryField32::from_repr(t))
                .unwrap_or_else(|e| panic!("four-byte twiddle {t:#x}: {e}"));
        }
    }

    #[test]
    fn a_run_shorter_than_a_register_is_left_to_the_caller() {
        // Fixture state: one byte short of a register, so no whole register fits.
        let mut lo = vec![1u8; LANE_BYTES - 1];
        let mut hi = vec![2u8; LANE_BYTES - 1];
        let blocks = byte_blocks(3);

        // The kernel reports zero bytes covered and leaves both sides untouched.
        let covered = super::subfield_butterfly::<Model, 1, false>(&mut lo, &mut hi, &blocks);
        assert_eq!(covered, 0);
        assert!(lo.iter().all(|&b| b == 1) && hi.iter().all(|&b| b == 2));

        // An empty pair is the same story with nothing to leave behind.
        assert_eq!(
            super::subfield_butterfly::<Model, 1, false>(&mut [], &mut [], &blocks),
            0
        );
    }
}
