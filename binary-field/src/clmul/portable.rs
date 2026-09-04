//! The software backend, for targets with no carryless-multiply instruction.
//!
//! An integer multiply is a carryless multiply plus carries.
//!
//! Masking an operand down to every fourth bit spaces its partial products four apart.
//! The three positions in between are then free for carries to land in.
//! Masking the result again discards them.
//!
//! Four masked operands per side cover every bit of the input.
//! This is the routine BearSSL's constant-time GHASH is built on.

/// Every fourth bit, starting at bit 0.
const NIBBLE_0: u64 = 0x1111_1111_1111_1111;
/// Every fourth bit, starting at bit 1.
const NIBBLE_1: u64 = NIBBLE_0 << 1;
/// Every fourth bit, starting at bit 2.
const NIBBLE_2: u64 = NIBBLE_0 << 2;
/// Every fourth bit, starting at bit 3.
const NIBBLE_3: u64 = NIBBLE_0 << 3;

/// The low 64 bits of the carryless product of two 64-bit polynomials over `GF(2)`.
///
/// # Algorithm
///
/// Split each operand by the residue of its bit positions modulo 4.
/// A product of residue `r` by residue `s` lands entirely in residue `r + s`.
/// Sixteen integer products cover the four residues of the output.
///
/// # Why the carries never escape
///
/// Bit `p` counts the pairs `(i, j)` with `i + j = p` drawn from the two masks.
/// A 64-bit word holds at most 16 positions in one residue class.
///
/// At most 15 pairs reach any `p <= 56`, which fits the three free bits above `p`.
/// Only `p = 60` reaches 16, and its carry leaves the word entirely.
#[inline]
pub(super) const fn bmul64(x: u64, y: u64) -> u64 {
    // Bits of the first operand, one residue class per word.
    let (x0, x1) = (x & NIBBLE_0, x & NIBBLE_1);
    let (x2, x3) = (x & NIBBLE_2, x & NIBBLE_3);

    // Bits of the second operand, likewise.
    let (y0, y1) = (y & NIBBLE_0, y & NIBBLE_1);
    let (y2, y3) = (y & NIBBLE_2, y & NIBBLE_3);

    // Each line gathers the four products belonging to one output residue.
    //
    //     residue 0 <- (0,0) (1,3) (2,2) (3,1)
    //     residue 1 <- (0,1) (1,0) (2,3) (3,2)
    //     residue 2 <- (0,2) (1,1) (2,0) (3,3)
    //     residue 3 <- (0,3) (1,2) (2,1) (3,0)
    let z0 = x0.wrapping_mul(y0) ^ x1.wrapping_mul(y3) ^ x2.wrapping_mul(y2) ^ x3.wrapping_mul(y1);
    let z1 = x0.wrapping_mul(y1) ^ x1.wrapping_mul(y0) ^ x2.wrapping_mul(y3) ^ x3.wrapping_mul(y2);
    let z2 = x0.wrapping_mul(y2) ^ x1.wrapping_mul(y1) ^ x2.wrapping_mul(y0) ^ x3.wrapping_mul(y3);
    let z3 = x0.wrapping_mul(y3) ^ x1.wrapping_mul(y2) ^ x2.wrapping_mul(y1) ^ x3.wrapping_mul(y0);

    // Keep only the positions each line is exact at, dropping the carries in between.
    (z0 & NIBBLE_0) | (z1 & NIBBLE_1) | (z2 & NIBBLE_2) | (z3 & NIBBLE_3)
}

/// The carryless product of two 64-bit polynomials over `GF(2)`, as its two 64-bit halves.
///
/// # Algorithm
///
/// Reversing both operands reverses their product.
/// Running the masked multiply on the reversed operands therefore returns the top half.
///
/// ```text
///     a * b            = p_0 + p_1 x + ... + p_126 x^126
///     rev(a) * rev(b)  = p_126 + p_125 x + ... + p_0 x^126
///
///     its low 64 bits  = p_126 down to p_63
///     reversed again   = p_63, p_64, ..., p_126
///     shifted by one   = p_64, ..., p_126          the high half
/// ```
#[inline]
const fn clmul_64x64_halves(a: u64, b: u64) -> (u64, u64) {
    // Coefficients 0 through 63, straight from the masked multiply.
    let low = bmul64(a, b);

    // Coefficients 64 through 126, from the reversed operands.
    // The shift drops coefficient 63, which the low half already carries.
    let high = bmul64(a.reverse_bits(), b.reverse_bits()).reverse_bits() >> 1;

    (low, high)
}

/// The carryless product of two 64-bit polynomials over `GF(2)`.
#[inline]
pub(super) const fn clmul_64x64(a: u64, b: u64) -> u128 {
    let (low, high) = clmul_64x64_halves(a, b);
    (low as u128) | ((high as u128) << 64)
}

/// Multiplication in `GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`.
///
/// Both operands and the result are in the polynomial basis.
///
/// # Algorithm
///
/// Karatsuba over the 64-bit halves.
/// Writing `a = a0 + a1 x^64` and `b = b0 + b1 x^64`:
///
/// ```text
///     middle = (a0 + a1)(b0 + b1) + a0 b0 + a1 b1
/// ```
///
/// # Why Karatsuba here
///
/// A half product costs sixteen integer multiplies.
/// Saving one is worth the four extra exclusive ors it costs.
/// A hardware carryless multiply is cheap enough that the trade goes the other way.
#[inline]
pub(crate) const fn poly_mul_128(a: u128, b: u128) -> u128 {
    // Reduce the full polynomial product once.
    let (low, high) = wide_mul_128(a, b);
    super::reduce_128(low, high)
}

/// Compute the two halves of an unreduced 128-bit polynomial product.
#[inline]
const fn wide_mul_128(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);

    // The two diagonal half products.
    let (l0, l1) = clmul_64x64_halves(a0, b0);
    let (h0, h1) = clmul_64x64_halves(a1, b1);

    // The third product, less the two above, is the sum of the cross terms.
    let (m0, m1) = clmul_64x64_halves(a0 ^ a1, b0 ^ b1);
    let (m0, m1) = (m0 ^ l0 ^ h0, m1 ^ l1 ^ h1);

    // Place the middle coefficient at x^64 and split the 256-bit product in two.
    //
    //     low  = l0 + (l1 + m0) x^64
    //     high = (h0 + m1) + h1 x^64          weighted by x^128
    let low = (l0 as u128) | (((l1 ^ m0) as u128) << 64);
    let high = ((h0 ^ m1) as u128) | ((h1 as u128) << 64);

    (low, high)
}

/// Sum polynomial products before reducing modulo the GHASH polynomial.
#[inline]
pub(crate) fn poly_dot_128(pairs: impl Iterator<Item = (u128, u128)>) -> u128 {
    // Reduction is linear over GF(2), so XOR accumulation needs no carry space.
    let (low, high) = pairs.fold((0, 0), |(low, high), (a, b)| {
        let (l, h) = wide_mul_128(a, b);
        (low ^ l, high ^ h)
    });
    super::reduce_128(low, high)
}

/// Interleaves the bits of a 64-bit word with zeros, widening it to 128 bits.
///
/// Each round doubles the gap between neighbouring bits.
/// After six rounds every original bit sits at twice its index.
#[inline]
const fn spread_bits_64(v: u64) -> u128 {
    let mut x = v as u128;

    // Separate into groups of 32 bits, then 16, 8, 4, 2, and finally single bits.
    x = (x | (x << 32)) & 0x0000_0000_ffff_ffff_0000_0000_ffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff_0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    x
}

/// Squaring in `GF(2^128)`, taking and returning the polynomial representation.
///
/// Squaring is the Frobenius map, so in characteristic 2 the coefficients simply spread out:
///
/// ```text
///     (sum_i c_i x^i)^2 = sum_i c_i x^(2i)
/// ```
///
/// Only shifts and masks are involved.
/// This backend therefore squares far faster than it multiplies.
#[inline]
pub(crate) const fn poly_square_128(a: u128) -> u128 {
    // Spreading each half gives the 256-bit square directly, with no middle term.
    super::reduce_128(spread_bits_64(a as u64), spread_bits_64((a >> 64) as u64))
}

/// Multiply by a polynomial of degree below 64 with two unreduced half products.
#[inline]
pub(crate) const fn poly_mul_128_by_64(a: u128, b: u64) -> u128 {
    // The high half of the second operand is zero, leaving just two coefficients.
    let low = clmul_64x64(a as u64, b);
    let middle = clmul_64x64((a >> 64) as u64, b);
    super::reduce_128(low ^ (middle << 64), middle >> 64)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{bmul64, clmul_64x64, spread_bits_64};
    use crate::clmul::basis::{TAIL_128, poly_mul};

    /// Bit `i` of the input, moved to position `2i`, one bit at a time.
    fn spread_bits_64_reference(v: u64) -> u128 {
        (0..64)
            .filter(|i| (v >> i) & 1 == 1)
            .fold(0u128, |acc, i| acc | (1u128 << (2 * i)))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4000))]

        #[test]
        fn the_masked_multiply_matches_the_bit_serial_product(a: u64, b: u64) {
            // The bit-serial loop is the definition of a carryless product.
            let expected = super::super::scalar_clmul_64x64(a, b);

            // Both halves, recovered through the bit reversal.
            prop_assert_eq!(clmul_64x64(a, b), expected);

            // The low half alone, which is all the masked multiply promises.
            prop_assert_eq!(bmul64(a, b), expected as u64);
        }

        #[test]
        fn the_bit_spread_matches_a_bit_at_a_time(v: u64) {
            prop_assert_eq!(spread_bits_64(v), spread_bits_64_reference(v));
        }

        #[test]
        fn the_software_product_matches_bit_serial_modular_multiplication(a: u128, b: u128) {
            // Karatsuba plus the fold, against multiplication straight from the modulus.
            prop_assert_eq!(super::poly_mul_128(a, b), poly_mul(a, b, 128, TAIL_128));
        }

        #[test]
        fn the_software_square_matches_the_software_product(a: u128) {
            prop_assert_eq!(super::poly_square_128(a), super::poly_mul_128(a, a));
        }
    }

    #[test]
    fn the_masked_multiply_is_exact_where_the_carries_are_worst() {
        // Invariant: the carries stay inside the holes for every operand pair.
        //
        // Worst cases: the operands packing the most set bits into one residue class.
        //
        //     0xffff...  every position in every class
        //     0x8888...  every position in class 3
        //     0x1111...  every position in class 0
        //     1 << 63    the single highest position, where the carry leaves the word
        const WORST: [u64; 4] = [
            u64::MAX,
            0x8888_8888_8888_8888,
            0x1111_1111_1111_1111,
            1 << 63,
        ];

        for a in WORST {
            for b in WORST {
                assert_eq!(
                    clmul_64x64(a, b),
                    super::super::scalar_clmul_64x64(a, b),
                    "{a:#x} * {b:#x}"
                );
            }
        }
    }
}
