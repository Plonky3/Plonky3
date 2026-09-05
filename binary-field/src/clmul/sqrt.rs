//! Square roots by separating even and odd polynomial coefficients.

use super::basis::{TAIL_128, poly_mul};

/// The unique square root of the polynomial variable modulo the GHASH polynomial.
const ROOT_X: u128 = 0x2492_4924_9249_2492_6db6_db6d_b6db_6da4;

// Pin the constant to the modulus, independently of the selected arithmetic backend.
const _: () = assert!(poly_mul(ROOT_X, ROOT_X, 128, TAIL_128) == 2);

/// Gather the even-numbered coefficients into the low half of a word.
///
/// A bit-extract instruction does this in one step on recent x86.
/// On Zen 1 and Zen 2 it is microcoded, at around eighteen cycles.
///
/// Square roots are rare enough not to be worth that cliff, so the six shifts run everywhere.
#[inline]
const fn compact_even(mut x: u64) -> u64 {
    // Separate coefficients at positions 0, 2, ..., 62 from the odd coefficients.
    x &= 0x5555_5555_5555_5555;
    // Merge adjacent groups until the 32 selected bits are contiguous.
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    (x | (x >> 16)) & 0xffff_ffff
}

/// Separate the coefficients of even and odd degree into two 64-bit polynomials.
#[inline]
const fn unshuffle(a: u128) -> (u64, u64) {
    // Each half supplies 32 coefficients to each output polynomial.
    let (lo, hi) = (a as u64, (a >> 64) as u64);
    (
        compact_even(lo) | (compact_even(hi) << 32),
        compact_even(lo >> 1) | (compact_even(hi >> 1) << 32),
    )
}

/// Compute a square root with one field product and no operand-indexed tables.
#[inline]
pub(crate) fn poly_sqrt_128(a: u128) -> u128 {
    // Invariant:
    //     a = even(x)^2 + x * odd(x)^2
    //     sqrt(a) = even(x) + sqrt(x) * odd(x)
    let (even, odd) = unshuffle(a);
    u128::from(even) ^ super::poly_mul_128_by_64(ROOT_X, odd)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn unshuffle_matches_individual_coefficients(a: u128) {
            // Trace each input bit to its position in the two output polynomials.
            let (even, odd) = unshuffle(a);
            for i in 0..64 {
                prop_assert_eq!((even >> i) & 1, ((a >> (2 * i)) & 1) as u64);
                prop_assert_eq!((odd >> i) & 1, ((a >> (2 * i + 1)) & 1) as u64);
            }
        }
    }

    #[test]
    fn every_basis_vector_squares_back() {
        // A linear map is determined by its action on all 128 basis vectors.
        for i in 0..128 {
            let a = 1u128 << i;
            let root = poly_sqrt_128(a);
            assert_eq!(poly_mul(root, root, 128, TAIL_128), a);
        }
        // Include both additive extremes explicitly.
        assert_eq!(poly_sqrt_128(0), 0);
        let root = poly_sqrt_128(u128::MAX);
        assert_eq!(poly_mul(root, root, 128, TAIL_128), u128::MAX);
    }
}
