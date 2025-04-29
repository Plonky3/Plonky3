//! Tools for Lagrange interpolation.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_field::{ExtensionField, TwoAdicField, batch_multiplicative_inverse, scale_vec};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

/// Given evaluations of a batch of polynomials over the canonical power-of-two subgroup, evaluate
/// the polynomials at `point`.
///
/// This assumes the point is not in the subgroup, otherwise the behavior is undefined.
pub fn interpolate_subgroup<F, EF, Mat>(subgroup_evals: &Mat, point: EF) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: Matrix<F>,
{
    interpolate_coset(subgroup_evals, F::ONE, point)
}

/// Given evaluations of a batch of polynomials over the given coset of the canonical power-of-two
/// subgroup, evaluate the polynomials at `point`.
///
/// This assumes the point is not in the coset, otherwise the behavior is undefined.
///
/// The `coset_evals` must be given in standard (not bit-reversed) order.
pub fn interpolate_coset<F, EF, Mat>(coset_evals: &Mat, shift: F, point: EF) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: Matrix<F>,
{
    let height = coset_evals.height();
    let log_height = log2_strict_usize(height);
    let coset = F::two_adic_generator(log_height)
        .shifted_powers(shift)
        .take(height)
        .collect::<Vec<_>>();

    // Compute `1/(z - gh^i)` for each elements of the coset.
    let diffs: Vec<EF> = coset.par_iter().map(|&g| point - g).collect();
    let diff_invs = batch_multiplicative_inverse(&diffs);

    interpolate_coset_with_precomputation(coset_evals, shift, point, &coset, &diff_invs)
}

/// Given evaluations of a batch of polynomials over the given coset of the
/// canonical power-of-two subgroup, evaluate the polynomials at `point`.
///
/// This assumes the point is not in the coset, otherwise the behavior is undefined.
///
/// This function takes the precomputed `subgroup` points and `diff_invs` (the
/// inverses of the differences between the evaluation point and each shifted
/// subgroup element), and should be preferred over `interpolate_coset` when
/// repeatedly called with the same subgroup and/or point.
///
/// Unlike `interpolate_coset`, the parameters `subgroup`, `coset_evals`, and
/// `diff_invs` may use any indexing scheme, as long as they are all consistent.
pub fn interpolate_coset_with_precomputation<F, EF, Mat>(
    coset_evals: &Mat,
    shift: F,
    point: EF,
    coset: &[F],
    diff_invs: &[EF],
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: Matrix<F>,
{
    // Slight variation of this approach: https://hackmd.io/@vbuterin/barycentric_evaluation

    debug_assert_eq!(coset.len(), diff_invs.len());
    debug_assert_eq!(coset.len(), coset_evals.height());

    // We start with the evaluations of a polynomial `f` over a coset `gH` of size `N` and want to compute `f(z)`.
    // First note that `g^N - z^N` is equal to `0` at all points in the coset. Thus `(z^N - g^N)/(z - gh^i)`
    // is equal to `0` at all points except for `gh^i` where it is equal to `N * (gh^i)^{N - 1} = N * g^N * (gh^{i})^{-1}`.
    // Hence `L_{i}(x) = h^i * (g^N - x^N)/(N * g^{N - 1} * (gh^i - x))` will be equal to `1` at `gh^i` and `0`
    // at all other points in the coset. This means that we can compute `f(z)` as:
    //          `\sum_i L_{i}(z) f(gh^i) = (z^N - g^N)/(N * g^N) * \sum_i gh^i/(z - gh^i) f(gh^i)`.

    let height = coset_evals.height();
    let log_height = log2_strict_usize(height);

    // Compute `gh^i/(z - gh^i)` for each i.
    let col_scale: Vec<_> = coset
        .par_iter()
        .zip(diff_invs)
        .map(|(&sg, &diff_inv)| diff_inv * sg)
        .collect();

    // For each column polynomial `fj`, compute `\sum_i h^i/(gh^i - z) * fj(gh^i)`.
    let sum = coset_evals.columnwise_dot_product(&col_scale);

    let point_pow_height = point.exp_power_of_2(log_height);
    let shift_pow_height = shift.exp_power_of_2(log_height);

    // Compute the vanishing polynomial of the coset.
    // This is `Z_{sH}(z) = z^N - g^N`.
    let vanishing_polynomial = point_pow_height - shift_pow_height;

    // Compute N * g^N
    // In principle, height could be bigger than the characteristic of F.
    let denominator = shift_pow_height.mul_2exp_u64(log_height as u64);

    // TODO: It should be possible to remove this inverse via a simple refactor.
    // We simply need to pass in shift_inv as well as shift. This would also
    // let us get rid of one of the exp_power_of_2 calls using `(z^N - g^N)/(N * g^N) = ((z/g)^N - 1)/N`.
    scale_vec(vanishing_polynomial * denominator.inverse(), sum)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{Field, PrimeCharacteristicRing, batch_multiplicative_inverse};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;

    use crate::{interpolate_coset, interpolate_coset_with_precomputation, interpolate_subgroup};

    #[test]
    fn test_interpolate_subgroup() {
        // x^2 + 2 x + 3
        type F = BabyBear;
        let evals = [
            6, 886605102, 1443543107, 708307799, 2, 556938009, 569722818, 1874680944,
        ]
        .map(F::from_u32);
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);
        let point = F::from_u16(100);
        let result = interpolate_subgroup(&evals_mat, point);
        assert_eq!(result, vec![F::from_u16(10203)]);
    }

    #[test]
    fn test_interpolate_coset() {
        // x^2 + 2 x + 3
        type F = BabyBear;
        let shift = F::GENERATOR;
        let evals = [
            1026, 129027310, 457985035, 994890337, 902, 1988942953, 1555278970, 913671254,
        ]
        .map(F::from_u32);
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);
        let point = F::from_u16(100);
        let result = interpolate_coset(&evals_mat, shift, point);
        assert_eq!(result, vec![F::from_u16(10203)]);

        use p3_field::TwoAdicField;
        let n = evals.len();
        let k = log2_strict_usize(n);

        let coset: Vec<_> = F::two_adic_generator(k)
            .shifted_powers(shift)
            .take(n)
            .collect();

        let denom: Vec<_> = coset.iter().map(|&w| point - w).collect();

        let denom = batch_multiplicative_inverse(&denom);
        let result =
            interpolate_coset_with_precomputation(&evals_mat, shift, point, &coset, &denom);
        assert_eq!(result, vec![F::from_u16(10203)]);
    }
}
