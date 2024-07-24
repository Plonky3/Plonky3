use std::collections::BTreeMap;

use itertools::{izip, Itertools};
use p3_commit::OpenedValues;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, ExtensionField, TwoAdicField,
};
use p3_fri::Codeword;
use p3_matrix::Matrix;
use p3_util::{linear_map::LinearMap, log2_strict_usize, reverse_slice_index_bits};
use tracing::instrument;

use crate::RsCode;

/*

A quick rundown of the optimizations in this function:
We are trying to compute sum_i alpha^i * (p(X) - y)/(X - z),
for each z an opening point, y = p(z). Each p(X) is given as evaluations in bit-reversed order
in the columns of the matrices. y is computed by barycentric interpolation.
X and p(X) are in the base field; alpha, y and z are in the extension.
The primary goal is to minimize extension multiplications.

- Instead of computing all alpha^i, we just compute alpha^i for i up to the largest width
of a matrix, then multiply by an "alpha offset" when accumulating.
      a^0 x0 + a^1 x1 + a^2 x2 + a^3 x3 + ...
    = a^0 ( a^0 x0 + a^1 x1 ) + a^2 ( a^0 x2 + a^1 x3 ) + ...
    (see `alpha_pows`, `alpha_pow_offset`, `num_reduced`)

- For each unique point z, we precompute 1/(X-z) for the largest subgroup opened at this point.
Since we compute it in bit-reversed order, smaller subgroups can simply truncate the vector.
    (see `inv_denoms`)

- Then, for each matrix (with columns p_i) and opening point z, we want:
    for each row (corresponding to subgroup element X):
        reduced[X] += alpha_offset * sum_i [ alpha^i * inv_denom[X] * (p_i[X] - y[i]) ]

    We can factor out inv_denom, and expand what's left:
        reduced[X] += alpha_offset * inv_denom[X] * sum_i [ alpha^i * p_i[X] - alpha^i * y[i] ]

    And separate the sum:
        reduced[X] += alpha_offset * inv_denom[X] * [ sum_i [ alpha^i * p_i[X] ] - sum_i [ alpha^i * y[i] ] ]

    And now the last sum doesn't depend on X, so we can precompute that for the matrix, too.
    So the hot loop (that depends on both X and i) is just:
        sum_i [ alpha^i * p_i[X] ]

    with alpha^i an extension, p_i[X] a base

*/

pub fn deep_reduce_matrices<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    opened_values: &OpenedValues<EF>,
) -> Vec<Codeword<EF, RsCode<EF>>> {
    todo!()
}

#[instrument(skip_all)]
fn compute_inverse_denominators<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mats_and_points: &[(Vec<M>, &Vec<Vec<EF>>)],
    coset_shift: F,
) -> LinearMap<EF, Vec<EF>> {
    let mut max_log_height_for_point: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height());
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Compute the largest subgroup we will use, in bitrev order.
    let max_log_height = *max_log_height_for_point.values().max().unwrap();
    let mut subgroup = cyclic_subgroup_coset_known_order(
        F::two_adic_generator(max_log_height),
        coset_shift,
        1 << max_log_height,
    )
    .collect_vec();
    reverse_slice_index_bits(&mut subgroup);

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            (
                z,
                batch_multiplicative_inverse(
                    &subgroup[..(1 << log_height)]
                        .iter()
                        .map(|&x| EF::from_base(x) - z)
                        .collect_vec(),
                ),
            )
        })
        .collect()
}
