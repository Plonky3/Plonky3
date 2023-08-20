use alloc::vec::Vec;

use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, cyclic_subgroup_known_order,
    scale_vec, sum_vecs, ExtensionField, Field, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::VerticalPair;
use p3_matrix::{Matrix, MatrixRows};
use p3_util::log2_strict_usize;

use crate::{TwoAdicCosetLde, TwoAdicLde, TwoAdicSubgroupLde, UndefinedLde};

/// A naive quadratic-time implementation of `Lde`, intended for testing.
pub struct NaiveUndefinedLde;

/// A naive quadratic-time implementation of `TwoAdicSubgroupLde`, intended for testing.
pub struct NaiveSubgroupLde;

/// A naive quadratic-time implementation of `TwoAdicCosetLde`, intended for testing.
pub struct NaiveCosetLde;

impl<F, In> UndefinedLde<F, F, In> for NaiveUndefinedLde
where
    F: Field,
    In: MatrixRows<F>,
{
    type Out = VerticalPair<F, In, RowMajorMatrix<F>>;

    fn lde_batch(&self, polys: In, extended_height: usize) -> Self::Out {
        let original_height = polys.height();
        let original_domain: Vec<F> = (0..original_height)
            .map(|x| F::from_canonical_usize(x))
            .collect();
        let weights = barycentric_weights(&original_domain);

        let added_values = (original_height..extended_height)
            .map(|x| F::from_canonical_usize(x))
            .flat_map(|x| interpolate(&original_domain, &polys, x, &weights))
            .collect();
        let extension = RowMajorMatrix::new(added_values, polys.width());
        VerticalPair::new(polys, extension)
    }
}

impl<Val, Domain> TwoAdicLde<Val, Domain> for NaiveSubgroupLde
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
{
    fn lde_batch(&self, polys: RowMajorMatrix<Val>, added_bits: usize) -> RowMajorMatrix<Domain> {
        let bits = log2_strict_usize(polys.height());
        let g = Domain::primitive_root_of_unity(bits);
        let subgroup = cyclic_subgroup_known_order::<Domain>(g, 1 << bits).collect::<Vec<_>>();
        let weights = barycentric_weights(&subgroup);

        let lde_bits = bits + added_bits;
        let lde_subgroup = cyclic_subgroup_known_order::<Domain>(g, 1 << lde_bits);

        let polys_fe = polys.map(|x| Domain::from_base(x));
        let values = lde_subgroup
            .flat_map(|x| interpolate(&subgroup, &polys_fe, x, &weights))
            .collect();
        RowMajorMatrix::new(values, polys.width())
    }
}

impl<Val, Domain> TwoAdicLde<Val, Domain> for NaiveCosetLde
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
{
    fn lde_batch(&self, polys: RowMajorMatrix<Val>, added_bits: usize) -> RowMajorMatrix<Domain> {
        let bits = log2_strict_usize(polys.height());
        let g = Domain::primitive_root_of_unity(bits);
        let subgroup = cyclic_subgroup_known_order::<Domain>(g, 1 << bits).collect::<Vec<_>>();
        let weights = barycentric_weights(&subgroup);

        let lde_bits = bits + added_bits;
        let lde_subgroup =
            cyclic_subgroup_coset_known_order(g, self.shift(lde_bits), 1 << lde_bits);

        let polys_fe = polys.map(|x| Domain::from_base(x));
        let values = lde_subgroup
            .flat_map(|x| interpolate(&subgroup, &polys_fe, x, &weights))
            .collect();
        RowMajorMatrix::new(values, polys.width())
    }
}

impl<Val, Domain> TwoAdicSubgroupLde<Val, Domain> for NaiveSubgroupLde
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
{
}

impl<Val, Domain> TwoAdicCosetLde<Val, Domain> for NaiveCosetLde
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
{
    fn shift(&self, _lde_bits: usize) -> Domain {
        Domain::multiplicative_group_generator()
    }
}

// TODO: Move to interpolation crate?
fn barycentric_weights<F: Field>(points: &[F]) -> Vec<F> {
    let n = points.len();
    batch_multiplicative_inverse(
        &(0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| points[i] - points[j])
                    .product::<F>()
            })
            .collect::<Vec<_>>(),
    )
}

// TODO: Move to interpolation crate?
fn interpolate<F: Field, Mat: MatrixRows<F>>(
    points: &[F],
    values: &Mat,
    x: F,
    barycentric_weights: &[F],
) -> Vec<F> {
    // If x is in the list of points, the Lagrange formula would divide by zero.
    for (i, &x_i) in points.iter().enumerate() {
        if x_i == x {
            return values.row(i).into_iter().collect();
        }
    }

    let l_x: F = points.iter().map(|&x_i| x - x_i).product();

    let sum = sum_vecs((0..points.len()).map(|i| {
        let x_i = points[i];
        let y_i = values.row(i).into_iter().collect();
        let w_i = barycentric_weights[i];
        scale_vec(w_i / (x - x_i), y_i)
    }));

    scale_vec(l_x, sum)
}
