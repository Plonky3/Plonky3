use p3_field::extension::Complex;
#[cfg(test)]
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field};
#[cfg(test)]
use p3_matrix::dense::RowMajorMatrix;
#[cfg(test)]
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

/// Get the cfft polynomial basis.
/// The basis consists off all multi-linear products of: y, x, 2x^2 - 1, 2(2x^2 - 1)^2 - 1, ...
/// The ordering of these basis elements is the bit reversal of the sequence: 1, y, x, xy, (2x^2 - 1), (2x^2 - 1)y, ...
/// We also need to throw in a couple of negative signs for technical reasons.
#[cfg(test)]
pub(crate) fn circle_basis<F: ComplexExtendable>(point: Complex<F>, log_n: usize) -> Vec<F> {
    if log_n == 0 {
        return vec![F::one()];
    }

    let mut basis = vec![F::one()];
    basis.reserve(1 << log_n);

    // First compute the repeated applications of the squaring map π(x) = 2x^2 - 1
    let mut cur = point.real();
    for _ in 0..(log_n - 1) {
        for i in 0..basis.len() {
            basis.push(basis[i] * cur);
        }
        cur = F::two() * cur.square() - F::one();
    }

    // Bit reverse, and compute the second half of the array,
    // which is just each element of the first half times y
    reverse_slice_index_bits(&mut basis);
    for i in 0..basis.len() {
        basis.push(basis[i] * point.imag());
    }

    // Negate each element each time the binary representation of its index has a pair of adjacent ones,
    // or equivalently, if the number of adjacent ones is odd.
    // This comes from a peculiarity in how we compute the CFFT:
    // The butterfly zips the first half of the domain with the second half reversed, because that maps each point
    // to its involution. After each layer, the second half is still in reverse order, so we should use the twiddles
    // in reverse order as well, but we ignore that and use the same twiddles for both halves.
    // Using t(g^(N-k)) instead of t(g^k) just adds a negative sign. It turns out the number of negations is the number
    // of adjacent ones in the index.
    for (i, val) in basis.iter_mut().enumerate() {
        let num_adjacent_ones = (i & (i >> 1)).count_ones();
        if num_adjacent_ones % 2 == 1 {
            *val = -*val;
        }
    }

    basis
}

#[cfg(test)]
pub(crate) fn eval_circle_polys<F: ComplexExtendable>(
    coeffs: &RowMajorMatrix<F>,
    point: Complex<F>,
) -> Vec<F> {
    use itertools::izip;
    use p3_matrix::Matrix;

    let log_n = log2_strict_usize(coeffs.height());
    let mut accs = vec![F::zero(); coeffs.width()];
    for (row, basis) in coeffs.rows().zip(circle_basis(point, log_n)) {
        for (acc, coeff) in izip!(&mut accs, row) {
            *acc += *coeff * basis;
        }
    }
    accs
}

/// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
/// (x, y) = ((1-t^2)/(1+t^2), 2t/(1+t^2))
/// t = ±i goes to (-1, 0)
pub(crate) fn univariate_to_point<F: Field>(t: F) -> Complex<F> {
    let t2 = t.square();
    if let Some(inv_denom) = (F::one() + t2).try_inverse() {
        Complex::new((F::one() - t2) * inv_denom, (F::two() * t) * inv_denom)
    } else {
        Complex::new_real(F::neg_one())
    }
}

/// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
/// t = y / (x + 1)
/// If F has i, this should return that instead, but we don't have access..
pub(crate) fn point_to_univariate<F: Field>(p: Complex<F>) -> Option<F> {
    p.imag().try_div(p.real() + F::one())
}

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
        p_x = F::two() * p_x.square() - F::one();
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

    #[test]
    fn test_uni_to_point() {
        // 0 -> (1, 0)
        assert_eq!(univariate_to_point(F::zero()), C::new_real(F::one()));
        // 1 -> (0, 1)
        assert_eq!(univariate_to_point(F::one()), C::new_imag(F::one()));
        // -1 -> (0, -1)
        assert_eq!(univariate_to_point(F::neg_one()), C::new_imag(F::neg_one()));
        // to make infinity, we need an extension that includes i
        // for example, the complex extension itself, although any
        // even extension of M31 will have p^i = 1 (mod 4)
        // i -> (-1, 0)
        assert_eq!(
            univariate_to_point(C::new_imag(F::one())),
            Complex::new_real(C::neg_one()),
        );
        // -i -> (-1, 0)
        assert_eq!(
            univariate_to_point(C::new_imag(F::neg_one())),
            Complex::new_real(C::neg_one()),
        );
    }

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
