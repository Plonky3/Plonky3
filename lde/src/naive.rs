use crate::TwoAdicSubgroupLDE;
use alloc::vec::Vec;
use p3_field::{batch_multiplicative_inverse, cyclic_subgroup_known_order};
use p3_field::{Field, FieldExtension, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

/// A naive quadratic-time implementation of `TwoAdicLDE`, intended for testing.
pub struct NaiveLDE;

impl<F, FE> TwoAdicSubgroupLDE<F, FE> for NaiveLDE
where
    F: Field,
    FE: FieldExtension<F> + TwoAdicField,
{
    type Res = FE;

    fn subgroup_lde_batch(
        &self,
        polys: RowMajorMatrix<F>,
        added_bits: usize,
    ) -> RowMajorMatrix<Self::Res> {
        let bits = log2_strict_usize(polys.height());
        let g = FE::primitive_root_of_unity(bits);
        let subgroup = cyclic_subgroup_known_order::<FE>(g, 1 << bits).collect::<Vec<_>>();
        let weights = barycentric_weights(&subgroup);

        let lde_bits = bits + added_bits;
        let lde_subgroup = cyclic_subgroup_known_order::<FE>(g, 1 << lde_bits);

        let polys_fe = polys.map(|x| FE::from_base(x));
        let values = lde_subgroup
            .map(|x| interpolate(&subgroup, &polys_fe, x, &weights))
            .flatten()
            .collect();
        RowMajorMatrix::new(values, polys.width())
    }
}

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

fn interpolate<F: Field>(
    points: &[F],
    values: &RowMajorMatrix<F>,
    x: F,
    barycentric_weights: &[F],
) -> Vec<F> {
    // If x is in the list of points, the Lagrange formula would divide by zero.
    for (i, &x_i) in points.iter().enumerate() {
        if x_i == x {
            return values.row(i).to_vec();
        }
    }

    let l_x: F = points.iter().map(|&x_i| x - x_i).product();

    let sum = sum_vecs((0..points.len()).map(|i| {
        let x_i = points[i];
        let y_i = values.row(i).to_vec();
        let w_i = barycentric_weights[i];
        scale_vec(w_i / (x - x_i), y_i)
    }));

    scale_vec(l_x, sum)
}

fn add_vecs<F: Field>(v: Vec<F>, w: Vec<F>) -> Vec<F> {
    assert_eq!(v.len(), w.len());
    v.into_iter().zip(w).map(|(x, y)| x + y).collect()
}

fn sum_vecs<F: Field, I: Iterator<Item = Vec<F>>>(iter: I) -> Vec<F> {
    iter.reduce(|v, w| add_vecs(v, w))
        .expect("sum_vecs: empty iterator")
}

fn scale_vec<F: Field>(s: F, vec: Vec<F>) -> Vec<F> {
    vec.into_iter().map(|x| s * x).collect()
}
