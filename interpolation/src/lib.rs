//! Tools for Lagrange interpolation.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, scale_vec,
    two_adic_coset_zerofier, ExtensionField, Field, TwoAdicField,
};
use p3_matrix::MatrixRows;
use p3_util::log2_strict_usize;

/// Given evaluations of a batch of polynomials over the canonical power-of-two subgroup, evaluate
/// the polynomials at `point`.
pub fn interpolate_subgroup<F, EF, Mat>(subgroup_evals: &Mat, point: EF) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: MatrixRows<F>,
{
    interpolate_coset(subgroup_evals, F::one(), point)
}

/// Given evaluations of a batch of polynomials over the given coset of the canonical power-of-two
/// subgroup, evaluate the polynomials at `point`.
pub fn interpolate_coset<F, EF, Mat>(coset_evals: &Mat, shift: F, point: EF) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Mat: MatrixRows<F>,
{
    // Slight variation of this approach: https://hackmd.io/@vbuterin/barycentric_evaluation

    let width = coset_evals.width();
    let height = coset_evals.height();
    let log_height = log2_strict_usize(height);
    let g = F::two_adic_generator(log_height);

    let diffs: Vec<EF> = cyclic_subgroup_coset_known_order(g, shift, height)
        .map(|subgroup_i| point - subgroup_i)
        .collect();
    let diff_invs = batch_multiplicative_inverse(&diffs);

    let mut sum = vec![EF::zero(); width];
    for (i, (subgroup_i, diff_inv)) in g.powers().zip(diff_invs).enumerate() {
        let row = coset_evals.row(i).into_iter();
        add_scaled_base_slice_in_place(&mut sum, row, diff_inv * subgroup_i);
    }

    let zerofier = two_adic_coset_zerofier::<EF>(log_height, EF::from_base(shift), point);
    let denominator = F::from_canonical_usize(height) * shift.exp_u64(height as u64 - 1);
    scale_vec(zerofier * denominator.inverse(), sum)
}

/// `x += y * s`, where `s` is a scalar.
pub fn add_scaled_base_slice_in_place<F, EF, Y>(x: &mut [EF], y: Y, s: EF)
where
    F: Field,
    EF: ExtensionField<F>,
    Y: Iterator<Item = F>,
{
    // TODO: Use PackedField
    x.iter_mut().zip(y).for_each(|(x_i, y_i)| *x_i += s * y_i);
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;

    use crate::{interpolate_coset, interpolate_subgroup};

    #[test]
    fn test_interpolate_subgroup() {
        // x^2 + 2 x + 3
        type F = BabyBear;
        let evals = [
            6, 886_605_102, 1_443_543_107, 708_307_799, 2, 556_938_009, 569_722_818, 1_874_680_944,
        ]
        .map(F::from_canonical_u32);
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);
        let point = F::from_canonical_u32(100);
        let result = interpolate_subgroup(&evals_mat, point);
        assert_eq!(result, vec![F::from_canonical_u32(10203)]);
    }

    #[test]
    fn test_interpolate_coset() {
        // x^2 + 2 x + 3
        type F = BabyBear;
        let shift = F::generator();
        let evals = [
            1026, 129_027_310, 457_985_035, 994_890_337, 902, 1_988_942_953, 1_555_278_970, 913_671_254,
        ]
        .map(F::from_canonical_u32);
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);
        let point = F::from_canonical_u32(100);
        let result = interpolate_coset(&evals_mat, shift, point);
        assert_eq!(result, vec![F::from_canonical_u32(10203)]);
    }
}
