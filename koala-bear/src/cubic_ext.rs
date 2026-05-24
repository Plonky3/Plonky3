//! Arithmetic helpers for the degree-3 extension `F_p[X] / (X^3 + X + 4)` of KoalaBear.
//!
//! `p = 2^31 - 2^24 + 1` satisfies `gcd(3, p - 1) = 1`, so cubing is a bijection
//! on `F_p`. As a consequence every polynomial of the form `X^3 - W` has a root
//! in `F_p` and is reducible — `F_{p^3}` therefore has to be built from a
//! non-binomial irreducible. The trinomial `X^3 + X + 4` is irreducible over
//! `F_p` (verified directly), so we use it and write `α^3 = -α - 4`.
//!
//! No `Field` machinery is added here — only the operations needed by callers
//! (e.g. an XHash-style extension S-box): multiplication, squaring, and a
//! fixed-exponent cube map. [`fp3_mul`] uses
//! [`PrimeCharacteristicRing::dot_product`] so each output coefficient costs a
//! single fused Monty reduction (same shape as
//! `p3_field::extension::cubic_extension::trinomial_cubic_mul`).

use p3_field::PrimeCharacteristicRing;

use crate::KoalaBear;

/// Multiply two elements of `F_p[α] / (α^3 + α + 4)`.
///
/// With `α^3 = −α − 4` and `α^4 = −α^2 − 4α`, the schoolbook product reduces to
/// three length-3 dot products of `a` against fixed combinations of `b`:
/// ```text
///   c0 = a0·b0       + a1·(−4·b2)        + a2·(−4·b1)
///   c1 = a0·b1       + a1·(b0 − b2)      + a2·(−b1 − 4·b2)
///   c2 = a0·b2       + a1·b1             + a2·(b0 − b2)
/// ```
#[inline]
pub fn fp3_mul(a: &[KoalaBear; 3], b: &[KoalaBear; 3]) -> [KoalaBear; 3] {
    let neg_b1 = -b[1];
    let neg_b2 = -b[2];
    let m4_b1 = neg_b1.double().double();
    let m4_b2 = neg_b2.double().double();
    let b0_minus_b2 = b[0] + neg_b2;
    let neg_b1_minus_4b2 = neg_b1 + m4_b2;

    let c0 = KoalaBear::dot_product::<3>(a, &[b[0], m4_b2, m4_b1]);
    let c1 = KoalaBear::dot_product::<3>(a, &[b[1], b0_minus_b2, neg_b1_minus_4b2]);
    let c2 = KoalaBear::dot_product::<3>(a, &[b[2], b[1], b0_minus_b2]);
    [c0, c1, c2]
}

/// Square an element of `F_p[α] / (α^3 + α + 4)`.
///
/// 3 squarings + 3 multiplications. The dot-product form used in [`fp3_mul`]
/// requires 9 base mults, so for squaring we keep the explicit expansion.
#[inline]
pub fn fp3_square(a: &[KoalaBear; 3]) -> [KoalaBear; 3] {
    let [a0, a1, a2] = *a;
    let a0a1 = a0 * a1;
    let a0a2 = a0 * a2;
    let a1a2 = a1 * a2;
    let a0sq = a0 * a0;
    let a1sq = a1 * a1;
    let a2sq = a2 * a2;
    // α^3 → −α − 4, α^4 → −α^2 − 4·α.
    let four_a1a2 = a1a2.double().double();
    let four_a2sq = a2sq.double().double();
    let r0 = a0sq - four_a1a2.double();
    let r1 = a0a1.double() - a1a2.double() - four_a2sq;
    let r2 = a0a2.double() + a1sq - a2sq;
    [r0, r1, r2]
}

/// Cube an element of `F_{p^3}` via `x → x^2 → x^3` (1 squaring + 1 multiplication).
#[inline]
pub fn fp3_cube(a: &[KoalaBear; 3]) -> [KoalaBear; 3] {
    let s = fp3_square(a);
    fp3_mul(&s, a)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    fn lift(v: u32) -> KoalaBear {
        KoalaBear::from_u32(v)
    }

    #[test]
    fn fp3_mul_zero_and_identity() {
        let a = [lift(3), lift(5), lift(7)];
        let zero = [KoalaBear::ZERO; 3];
        let one = [KoalaBear::ONE, KoalaBear::ZERO, KoalaBear::ZERO];
        assert_eq!(fp3_mul(&a, &zero), zero);
        assert_eq!(fp3_mul(&a, &one), a);
    }

    #[test]
    fn fp3_square_matches_mul() {
        let a = [lift(11), lift(13), lift(17)];
        assert_eq!(fp3_square(&a), fp3_mul(&a, &a));
    }

    #[test]
    fn fp3_cube_matches_repeated_mul() {
        let a = [lift(2), lift(3), lift(5)];
        let a2 = fp3_mul(&a, &a);
        let expected = fp3_mul(&a2, &a);
        assert_eq!(fp3_cube(&a), expected);
    }

    /// α^3 = −α − 4 in F_p[α] / (α^3 + α + 4).
    #[test]
    fn fp3_alpha_cubed_reduces() {
        let alpha = [KoalaBear::ZERO, KoalaBear::ONE, KoalaBear::ZERO];
        let alpha_cubed = fp3_cube(&alpha);
        let expected = [-KoalaBear::from_u32(4), -KoalaBear::ONE, KoalaBear::ZERO];
        assert_eq!(alpha_cubed, expected);
    }
}
