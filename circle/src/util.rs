use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{batch_multiplicative_inverse, AbstractField, ExtensionField, Field};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::Matrix;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::instrument;

use crate::domain::cfft_domain;

pub(crate) fn circle_basis<F: ComplexExtendable>(point: Complex<F>, log_n: usize) -> Vec<F> {
    if log_n == 0 {
        return vec![F::one()];
    }
    let mut basis = vec![F::one()];
    basis.reserve(1 << log_n);
    let mut cur = point.real();
    for _ in 0..(log_n - 1) {
        for i in 0..basis.len() {
            basis.push(basis[i] * cur);
        }
        cur = F::two() * cur.square() - F::one();
    }
    reverse_slice_index_bits(&mut basis);
    for i in 0..basis.len() {
        basis.push(basis[i] * point.imag());
    }
    for (i, val) in basis.iter_mut().enumerate() {
        let num_adjacent_ones = (i & (i >> 1)).count_ones();
        if num_adjacent_ones % 2 == 1 {
            *val = -*val;
        }
    }
    basis
}

pub(crate) fn eval_circle_polys<F: ComplexExtendable>(
    coeffs: &RowMajorMatrix<F>,
    point: Complex<F>,
) -> Vec<F> {
    let log_n = log2_strict_usize(coeffs.height());
    let mut accs = vec![F::zero(); coeffs.width()];
    for (row, basis) in coeffs.rows().zip(circle_basis(point, log_n)) {
        for (acc, coeff) in izip!(&mut accs, row) {
            *acc += *coeff * basis;
        }
    }
    accs
}

// Page 4 of the paper:
// (x, y) = ((1-t^2)/(1+t^2), 2t/(1+t^2))
// t = ±i goes to (-1, 0)
pub(crate) fn univariate_to_point<F: Field>(t: F) -> Complex<F> {
    let t2 = t.square();
    if let Some(inv_denom) = (F::one() + t2).try_inverse() {
        Complex::new((F::one() - t2) * inv_denom, (F::two() * t) * inv_denom)
    } else {
        Complex::new_real(F::neg_one())
    }
}

// t = y / (x + 1)
// If F has i, this should return that instead, but we don't have access..
pub(crate) fn point_to_univariate<F: Field>(p: Complex<F>) -> Option<F> {
    Some(p.imag() * (p.real() + F::one()).try_inverse()?)
}

// Page 5
// same as above, this *could* handle point at infinity if we had Field::try_sqrt
pub(crate) fn rotate_univariate<F: Field, EF: ExtensionField<F>>(t1: EF, t2: F) -> Option<EF> {
    Some((t1 + t2) * (EF::one() - t1 * t2).try_inverse()?)
}

// the zero poly for cfft domain sized log_n, evaluated at p_x
pub(crate) fn v_n<F: Field>(mut p_x: F, log_n: usize) -> F {
    for _ in 0..(log_n - 1) {
        p_x = F::two() * p_x.square() - F::one();
    }
    p_x
}

fn v_n_prime<F: Field>(p_x: F, log_n: usize) -> F {
    F::two().exp_u64((2 * (log_n - 1)) as u64) * (1..log_n).map(|i| v_n(p_x, i)).product()
}

// Simple zero at (1,0), simple pole at (-1,0)
// panics if called with x = -1
pub(crate) fn v_0<F: Field>(p: Complex<F>) -> F {
    p.imag() / (p.real() + F::one())
}

pub(crate) fn s_p_at_p<F: Field>(p_x: F, p_y: F, log_n: usize) -> F {
    -F::two() * v_n_prime(p_x, log_n) * p_y
}

// this can be optimized a TON
#[instrument(skip(zeta))]
pub(crate) fn lagrange_basis<F: ComplexExtendable, EF: ExtensionField<F>>(
    zeta: Complex<EF>,
    log_n: usize,
) -> Vec<EF> {
    let domain = cfft_domain::<F>(log_n).collect_vec();

    // the denominator so that the lagrange basis is normalized to 1
    // this depends only domain, so should be precomputed
    let lagrange_normalizer: Vec<F> = domain
        .iter()
        .map(|p| s_p_at_p(p.real(), p.imag(), log_n))
        .collect();

    let basis = domain
        .into_iter()
        .zip(&lagrange_normalizer)
        .map(|(p, &ln)| {
            // ext * base
            v_0(p.conjugate().rotate(zeta)) * ln
        })
        .collect_vec();

    batch_multiplicative_inverse(&basis)
}

// tranposed matrix-vector product: Mᵀv
#[instrument(skip_all, fields(dims = %m.dimensions()))]
pub(crate) fn gemv_tr<'a, F: Field, EF: ExtensionField<F>>(
    m: RowMajorMatrixView<'a, F>,
    v: Vec<EF>,
) -> Vec<EF> {
    assert_eq!(m.height(), v.len());
    let mut accs = vec![EF::zero(); m.width()];
    for (row, vx) in m.rows().zip(v) {
        for (acc, mx) in izip!(&mut accs, row) {
            *acc += vx * *mx;
        }
    }
    accs
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
