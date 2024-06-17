use p3_field::{extension::Complex, ExtensionField, Field};

use crate::point::Point;

/// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
/// ```ignore
/// (x, y) = ((1-t^2)/(1+t^2), 2t/(1+t^2))
/// ```
/// Returns `None` if `t^2 = -1` (corresponding to the point at infinity).
/*
pub(crate) fn univariate_to_point<F: Field>(t: F) -> Option<Complex<F>> {
    let t2 = t.square();
    let inv_denom = (F::one() + t2).try_inverse()?;
    Some(Complex::new(
        (F::one() - t2) * inv_denom,
        t.double() * inv_denom,
    ))
}
*/

/// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
/// ```ignore
/// t = y / (x + 1)
/// ```
/// If F has i, this should return that instead, but we don't have access..
/*
pub(crate) fn point_to_univariate<F: Field>(p: Complex<F>) -> Option<F> {
    p.imag().try_div(p.real() + F::one())
}
*/

/// Formula for the group operation in univariate coordinates
/// Circle STARKs, Section 3.1, Remark 4: (page 5 of the first revision PDF)
/// same as above, this *could* handle point at infinity if we had Field::try_sqrt
#[allow(unused)]
pub(crate) fn rotate_univariate<F: Field, EF: ExtensionField<F>>(t1: EF, t2: F) -> Option<EF> {
    (t1 + t2).try_div(EF::one() - t1 * t2)
}

/// Evaluate the vanishing polynomial for the standard position coset of size 2^log_n
/// at the point `p` (which has x coordinate `p_x`).
/// Circle STARKs, Section 3.3, Equation 8 (page 10 of the first revision PDF)
pub(crate) fn v_n<F: Field>(mut p_x: F, log_n: usize) -> F {
    for _ in 0..(log_n - 1) {
        p_x = p_x.square().double() - F::one();
    }
    p_x
}

/// Evaluate the formal derivative of `v_n` at the point `p` (which has x coordinate `p_x`).
/// Circle STARKs, Section 5.1, Remark 15 (page 21 of the first revision PDF)
fn v_n_prime<F: Field>(p_x: F, log_n: usize) -> F {
    F::two().exp_u64((2 * (log_n - 1)) as u64) * (1..log_n).map(|i| v_n(p_x, i)).product()
}

/// Simple zero at (1,0), simple pole at (-1,0)
/// Circle STARKs, Section 5.1, Lemma 11 (page 21 of the first revision PDF)
/// panics if called with x = -1
pub(crate) fn v_0<F: Field>(p: Complex<F>) -> F {
    p.imag() / (p.real() + F::one())
}

/// Evaluate the single-point vanishing function v_p(x). Used for DEEP quotient.
/// Circle STARKs, Section 3.3, Equation 11 (page 11 of the first edition PDF).
/// Simple zero at p, simple pole at +-infinity.
/*
pub(crate) fn v_p<F: Field, EF: ExtensionField<F>>(p: Complex<EF>, x: Complex<F>) -> Complex<EF> {
    let x_rotate_p: Complex<EF> = x.rotate(p.conjugate());
    Complex::new(EF::one() - x_rotate_p.real(), -x_rotate_p.imag())
}
*/

pub(crate) fn v_p<F: Field, EF: ExtensionField<F>>(p: Point<EF>, x: Point<F>) -> (EF, EF) {
    let x_minus_p = -p + x;
    (EF::one() - x_minus_p.x, -x_minus_p.y)
}

/// The concrete value of the selector s_P = v_n / (v_0 . T_p⁻¹) at P, used for normalization to 1.
/// Circle STARKs, Section 5.1, Remark 16 (page 22 of the first revision PDF)
pub(crate) fn s_p_at_p<F: Field>(p_x: F, p_y: F, log_n: usize) -> F {
    -F::two() * v_n_prime(p_x, log_n) * p_y
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_mersenne_31::Mersenne31;

    use super::*;

    type F = Mersenne31;
    type C = Complex<F>;

    /*
    #[test]
    fn test_uni_to_point() {
        // 0 -> (1, 0)
        assert_eq!(univariate_to_point(F::zero()), Some(C::new_real(F::one())));
        // 1 -> (0, 1)
        assert_eq!(univariate_to_point(F::one()), Some(C::new_imag(F::one())));
        // -1 -> (0, -1)
        assert_eq!(
            univariate_to_point(F::neg_one()),
            Some(C::new_imag(F::neg_one()))
        );
    }
    */

    #[test]
    fn test_s_p_at_p() {
        // from sage
        assert_eq!(
            s_p_at_p(
                // random point on the circle
                F::from_canonical_u32(383393203),
                F::from_canonical_u32(415518596),
                3
            ),
            F::from_canonical_u32(1612953309)
        );
    }
}
