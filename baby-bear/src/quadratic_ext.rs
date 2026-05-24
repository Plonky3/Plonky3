//! Arithmetic helpers for the degree-2 extension `F_p[X] / (X^2 - 11)` of BabyBear.
//!
//! The constant `W = 11` is a quadratic non-residue modulo
//! `p = 2^31 - 2^27 + 1`, so `X^2 - W` is irreducible and `F_{p^2}` is well
//! defined as `F_p[α]` with `α^2 = 11`.
//!
//! No `Field` machinery is added here — only the operations needed by callers
//! (e.g. an XHash-style extension S-box): multiplication, squaring, and a
//! fixed-exponent power map for the seventh power. The mul/square routines use
//! [`PrimeCharacteristicRing::dot_product`] so that the Monty reductions fuse
//! into a single reduction per output coefficient (same shape as
//! `p3_field::extension::binomial_extension::quadratic_mul`).

use p3_field::PrimeCharacteristicRing;

use crate::BabyBear;

/// `W` in `X^2 - W`, i.e. `α^2 = 11`.
pub const QUADRATIC_W: BabyBear = BabyBear::new(11);

/// Multiply two elements of `F_p[α] / (α^2 - 11)`.
///
/// Each element is represented as `[a0, a1]` for `a0 + a1·α`. The product is
/// `(a0·b0 + 11·a1·b1) + (a0·b1 + a1·b0)·α`.
#[inline]
pub fn fp2_mul(a: &[BabyBear; 2], b: &[BabyBear; 2]) -> [BabyBear; 2] {
    let b1_w = b[1] * QUADRATIC_W;
    let c0 = BabyBear::dot_product::<2>(a, &[b[0], b1_w]);
    let c1 = BabyBear::dot_product::<2>(a, &[b[1], b[0]]);
    [c0, c1]
}

/// Square an element of `F_p[α] / (α^2 - 11)`.
///
/// `(a0 + a1·α)^2 = (a0^2 + 11·a1^2) + (2·a0·a1)·α`.
#[inline]
pub fn fp2_square(a: &[BabyBear; 2]) -> [BabyBear; 2] {
    let a1_w = a[1] * QUADRATIC_W;
    let c0 = BabyBear::dot_product::<2>(a, &[a[0], a1_w]);
    let c1 = a[0] * a[1].double();
    [c0, c1]
}

/// Raise `a ∈ F_{p^2}` to the seventh power.
///
/// Uses the chain `x → x^2 → x^3 → x^6 → x^7` (2 squarings + 2 multiplications).
#[inline]
pub fn fp2_pow7(a: &[BabyBear; 2]) -> [BabyBear; 2] {
    let x2 = fp2_square(a);
    let x3 = fp2_mul(&x2, a);
    let x6 = fp2_square(&x3);
    fp2_mul(&x6, a)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn fp2_mul_zero() {
        let a = [BabyBear::from_u32(3), BabyBear::from_u32(5)];
        let z = [BabyBear::ZERO, BabyBear::ZERO];
        assert_eq!(fp2_mul(&a, &z), z);
    }

    #[test]
    fn fp2_mul_identity() {
        let a = [BabyBear::from_u32(3), BabyBear::from_u32(5)];
        let one = [BabyBear::ONE, BabyBear::ZERO];
        assert_eq!(fp2_mul(&a, &one), a);
    }

    #[test]
    fn fp2_square_matches_mul() {
        let a = [BabyBear::from_u32(7), BabyBear::from_u32(13)];
        assert_eq!(fp2_square(&a), fp2_mul(&a, &a));
    }

    #[test]
    fn fp2_pow7_matches_repeated_mul() {
        let a = [BabyBear::from_u32(9), BabyBear::from_u32(2)];
        let mut acc = a;
        for _ in 0..6 {
            acc = fp2_mul(&acc, &a);
        }
        assert_eq!(fp2_pow7(&a), acc);
    }
}
