//! Itoh–Tsujii inversion with precomputed Frobenius maps.
//!
//! The inverse of a nonzero `x` is `x^(2^128 - 2)`.
//! An addition chain reaches that exponent in ten products, once squaring is cheap.
//!
//! Squaring `2^k` times is `GF(2)`-linear, so each such step is one fixed matrix.
//! The five the chain needs are tabulated a byte at a time, at compile time.

use super::{poly_mul_128, poly_square_128, portable};

/// One byte-indexed linear map occupies 64 KiB.
type PowerMap = [[u128; 256]; 16];

/// Build a linear map for a fixed number of squarings at compile time.
const fn power_map(squarings: usize) -> PowerMap {
    let mut table = [[0; 256]; 16];
    let mut byte = 0;
    while byte < 16 {
        // A linear map is determined by the images of the eight bits of this byte.
        let mut bit = 0;
        while bit < 8 {
            let mut image = 1u128 << (8 * byte + bit);
            let mut step = 0;
            while step < squarings {
                image = portable::poly_square_128(image);
                step += 1;
            }
            table[byte][1 << bit] = image;
            bit += 1;
        }

        // Every remaining byte is the XOR of one bit and a smaller byte.
        let mut value = 1usize;
        while value < 256 {
            let low = 1 << value.trailing_zeros();
            table[byte][value] = table[byte][low] ^ table[byte][value ^ low];
            value += 1;
        }
        byte += 1;
    }
    table
}

/// Three squarings advance the exponent index from 3 to 6.
static POW_3: PowerMap = power_map(3);
/// Seven squarings serve both the 7-to-14 and 56-to-63 steps.
static POW_7: PowerMap = power_map(7);
/// Fourteen squarings advance the exponent index from 14 to 28.
static POW_14: PowerMap = power_map(14);
/// Twenty-eight squarings advance the exponent index from 28 to 56.
static POW_28: PowerMap = power_map(28);
/// Sixty-three squarings advance the exponent index from 63 to 126.
static POW_63: PowerMap = power_map(63);

/// Apply a Frobenius map through operand-indexed lookups.
#[inline]
fn apply(table: &PowerMap, value: u128) -> u128 {
    // Each byte contributes independently because squaring is GF(2)-linear.
    table
        .iter()
        .zip(value.to_le_bytes())
        .fold(0, |acc, (row, byte)| acc ^ row[usize::from(byte)])
}

/// Invert a nonzero polynomial-basis element, mapping zero to zero.
///
/// Operand-indexed tables make this unsuitable for secret inputs requiring constant-time access.
#[inline]
pub(crate) fn poly_inverse_128(x: u128) -> u128 {
    // Invariant:
    //     beta_k = x^(2^k - 1)
    //     beta_(a+b) = beta_a^(2^b) * beta_b
    let b2 = poly_mul_128(poly_square_128(x), x);
    let b3 = poly_mul_128(poly_square_128(b2), x);
    let b6 = poly_mul_128(apply(&POW_3, b3), b3);
    let b7 = poly_mul_128(poly_square_128(b6), x);

    // Doubling the exponent index reuses the same intermediate on both sides.
    let b14 = poly_mul_128(apply(&POW_7, b7), b7);
    let b28 = poly_mul_128(apply(&POW_14, b14), b14);
    let b56 = poly_mul_128(apply(&POW_28, b28), b28);

    // Finish 56 + 7 = 63, then 2 * 63 + 1 = 127.
    let b63 = poly_mul_128(apply(&POW_7, b56), b7);
    let b126 = poly_mul_128(apply(&POW_63, b63), b63);
    let b127 = poly_mul_128(poly_square_128(b126), x);

    // Squaring gives x^(2^128 - 2), the multiplicative inverse for nonzero x.
    poly_square_128(b127)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::clmul::basis::{TAIL_128, poly_mul};

    #[test]
    fn power_maps_match_the_modulus_on_every_basis_vector() {
        // Linearity reduces equality of the maps to all 128 input basis vectors.
        for (count, table) in [
            (3, &POW_3),
            (7, &POW_7),
            (14, &POW_14),
            (28, &POW_28),
            (63, &POW_63),
        ] {
            for bit in 0..128 {
                let input = 1u128 << bit;
                let mut expected = input;
                // Use bit-serial modular multiplication as an independent oracle.
                for _ in 0..count {
                    expected = poly_mul(expected, expected, 128, TAIL_128);
                }
                assert_eq!(apply(table, input), expected);
            }
        }
    }

    proptest! {
        #[test]
        fn inversion_matches_the_multiplicative_identity(x: u128) {
            // The bit-serial product characterizes the inverse independently of every backend.
            let inverse = poly_inverse_128(x);
            prop_assert_eq!(poly_mul(x, inverse, 128, TAIL_128), u128::from(x != 0));
        }
    }

    #[test]
    fn inversion_handles_zero_and_one() {
        // Every step maps zero to zero; the public field API rejects it before inversion.
        assert_eq!(poly_inverse_128(0), 0);
        assert_eq!(poly_inverse_128(1), 1);
    }
}
