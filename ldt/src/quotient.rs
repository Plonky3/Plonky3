use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::{debug_assert, debug_assert_eq};
use p3_field::eval_poly;
use p3_matrix::dense::RowMajorMatrix;

use itertools::{izip, Itertools};
use p3_commit::Mmcs;
use p3_field::{
    add_vecs, batch_multiplicative_inverse, binomial_expand, cyclic_subgroup_coset_known_order,
    extension::HasFrobenius, scale_vec, ExtensionField, Field, TwoAdicField,
};
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;

/// A wrapper around an Inner MMCS, which transforms each inner value to
/// `(inner - opened_point) / (x - opened_eval)`.
///
/// `(inner - r(X))/m(X)`
///
/// Since there can be multiple opening points, for each matrix, this transforms an inner opened row
/// into a concatenation of rows, transformed as above, for each point.
#[derive(Clone)]
pub struct QuotientMmcs<F, Inner: Mmcs<F>> {
    pub(crate) inner: Inner,

    /// For each matrix, a list of claimed openings, one for each point that we open that batch of
    /// polynomials at.
    pub(crate) openings: Vec<Vec<Opening<F>>>,

    // The coset shift for the inner MMCS's evals, to correct `x` in the denominator.
    pub(crate) coset_shift: F,
}

/// A claimed opening.
#[derive(Clone, Debug)]
pub(crate) struct Opening<F> {
    pub(crate) minpoly: Vec<F>,
    pub(crate) remainder_polys: Vec<Vec<F>>,
}

impl<F: Field> Opening<F> {
    pub(crate) fn new<EF: HasFrobenius<F>>(point: EF, values: Vec<EF>) -> Self {
        let remainder_polys = Self::compute_remainder_polys(point, &values);
        Self {
            minpoly: point.minimal_poly(),
            remainder_polys,
        }
    }
    fn compute_remainder_polys<EF: HasFrobenius<F>>(point: EF, values: &[EF]) -> Vec<Vec<F>> {
        // compute lagrange basis for [point, Frob point, Frob^2 point, ..]
        let xs = point.galois_group();
        let w = xs[1..]
            .iter()
            .map(|&xi| xs[0] - xi)
            .product::<EF>()
            .inverse();
        let l_point = scale_vec(w, binomial_expand(&xs[1..]));
        debug_assert_eq!(l_point.len(), EF::D);
        // interpolate at [(pt,value),(Frob pt, Frob alpha),..]
        let mut rs = vec![];
        for &v in values {
            let mut l_point_frob = scale_vec(v, l_point.clone());
            let mut r = l_point_frob.clone();
            for _ in 1..EF::D {
                l_point_frob.iter_mut().for_each(|c| *c = c.frobenius());
                r = add_vecs(r, l_point_frob.clone());
            }
            rs.push(
                r.into_iter()
                    .map(|c| {
                        debug_assert!(c.is_in_basefield());
                        c.as_base_slice()[0]
                    })
                    .collect(),
            );
        }
        rs
    }
}

impl<F, Inner> Mmcs<F> for QuotientMmcs<F, Inner>
where
    F: TwoAdicField,
    Inner: Mmcs<F>,
    for<'a> Inner::Mat<'a>: MatrixRowSlices<F>,
{
    type ProverData = Inner::ProverData;
    type Commitment = Inner::Commitment;
    type Proof = Inner::Proof;
    type Error = Inner::Error;
    type Mat<'a> = QuotientMatrix<F, Inner::Mat<'a>> where Self: 'a;

    fn open_batch(
        &self,
        index: usize,
        prover_data: &Self::ProverData,
    ) -> (Vec<Vec<F>>, Self::Proof) {
        let (inner_values, proof) = self.inner.open_batch(index, prover_data);
        let matrix_heights = self.inner.get_matrix_heights(prover_data);
        let max_height = *matrix_heights.iter().max().unwrap();
        let log_max_height = log2_strict_usize(max_height);

        let quotients = izip!(inner_values, self.openings.clone(), matrix_heights)
            .map(|(inner_row, openings_for_mat, height)| {
                let log2_height = log2_strict_usize(height);
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                let x = self.coset_shift
                    * F::two_adic_generator(log2_height).exp_u64(reduced_index as u64);
                openings_for_mat
                    .iter()
                    .flat_map(
                        |Opening {
                             minpoly,
                             remainder_polys,
                         }| {
                            inner_row.iter().zip_eq(remainder_polys).map(
                                move |(&inner_value, r)| {
                                    (inner_value - eval_poly(r, x)) / eval_poly(&minpoly, x)
                                },
                            )
                        },
                    )
                    .collect()
            })
            .collect();

        (quotients, proof)
    }

    fn get_matrices<'a>(&'a self, prover_data: &'a Self::ProverData) -> Vec<Self::Mat<'a>> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .zip(self.openings.clone())
            .map(|(inner, openings)| {
                let height = inner.height();
                let log2_height = log2_strict_usize(height);
                let g = F::two_adic_generator(log2_height);
                let subgroup =
                    cyclic_subgroup_coset_known_order(g, self.coset_shift, height).collect_vec();

                let denominators: Vec<F> = subgroup
                    .iter()
                    .flat_map(|&x| {
                        openings
                            .iter()
                            .map(move |opening| eval_poly(&opening.minpoly, x))
                    })
                    .collect();
                let inv_denominators = RowMajorMatrix::new(
                    batch_multiplicative_inverse(&denominators),
                    openings.len(),
                );

                dbg!(inner.dimensions());
                dbg!(subgroup.len());
                dbg!(inv_denominators.dimensions());
                dbg!(&openings);

                QuotientMatrix {
                    inner,
                    subgroup,
                    openings,
                    inv_denominators,
                    _phantom: PhantomData,
                }
            })
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_quotient_values: &[Vec<F>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        // quotient = (original - r(X))/m(X)
        // original = quotient * m(X) + r(X)

        let log_max_height = dimensions
            .iter()
            .map(|dims| log2_strict_usize(dims.height))
            .max()
            .unwrap();

        let opened_original_values = izip!(opened_quotient_values, &self.openings, dimensions)
            .map(|(quotient_row, openings, dims)| {
                let log_height = log2_strict_usize(dims.height);
                let bits_reduced = log_max_height - log_height;
                let reduced_index = index >> bits_reduced;
                let x = self.coset_shift
                    * F::two_adic_generator(log_height).exp_u64(reduced_index as u64);

                let original_width = quotient_row.len() / openings.len();
                let original_row_repeated: Vec<Vec<F>> = quotient_row
                    .chunks(original_width)
                    .zip(openings)
                    .map(|(quotient_row_chunk, opening)| {
                        quotient_row_chunk
                            .iter()
                            .zip(&opening.remainder_polys)
                            .map(|(&quotient_value, r)| {
                                quotient_value * eval_poly(&opening.minpoly, x) + eval_poly(&r, x)
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                get_repeated(original_row_repeated.into_iter())
            })
            .collect_vec();

        self.inner
            .verify_batch(commit, dimensions, index, &opened_original_values, proof)
    }
}

pub struct QuotientMatrix<F, Inner: MatrixRowSlices<F>> {
    inner: Inner,
    subgroup: Vec<F>,
    openings: Vec<Opening<F>>,
    /// For each row (associated with a subgroup element `x`), for each opening point,
    /// this holds `1 / (m(X))`.
    inv_denominators: RowMajorMatrix<F>,
    _phantom: PhantomData<F>,
}

impl<F, Inner: MatrixRowSlices<F>> Matrix<F> for QuotientMatrix<F, Inner> {
    fn width(&self) -> usize {
        self.inner.width() * self.openings.len()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F: Field, Inner: MatrixRowSlices<F>> MatrixRows<F> for QuotientMatrix<F, Inner> {
    type Row<'a> = QuotientMatrixRow<'a, F> where Inner: 'a;

    #[inline]
    fn row(&self, r: usize) -> Self::Row<'_> {
        QuotientMatrixRow {
            x: self.subgroup[r],
            openings: &self.openings,
            inv_denominator: self.inv_denominators.row_slice(r),
            inner_row: self.inner.row_slice(r),
            opening_index: 0,
            inner_col_index: 0,
        }
    }
}

pub struct QuotientMatrixRow<'a, F> {
    x: F,
    openings: &'a [Opening<F>],
    /// `1 / (x - opened_point)`
    inv_denominator: &'a [F],
    inner_row: &'a [F],
    opening_index: usize,
    inner_col_index: usize,
}

impl<'a, F: Field> Iterator for QuotientMatrixRow<'a, F> {
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.inner_col_index == self.inner_row.len() {
            self.opening_index += 1;
            self.inner_col_index = 0;
        }
        if self.opening_index == self.openings.len() {
            return None;
        }
        let eval = self.inner_row[self.inner_col_index];
        let opening = &self.openings[self.opening_index];
        let numerator = eval - eval_poly(&opening.remainder_polys[self.opening_index], self.x);
        let result = numerator * self.inv_denominator[self.opening_index];
        self.inner_col_index += 1;
        Some(result)
    }
}

/// Checks that the given iterator contains repetitions of a single item, and return that item.
fn get_repeated<T: Eq + Debug, I: Iterator<Item = T>>(mut iter: I) -> T {
    let first = iter.next().expect("get_repeated on empty iterator");
    for x in iter {
        debug_assert_eq!(x, first, "{:?} != {:?}", x, first);
    }
    first
}

fn to_base<F: Field, EF: ExtensionField<F>>(vec: Vec<EF>) -> Vec<F> {
    vec.into_iter()
        .map(|x| {
            let base = x.as_base_slice();
            for b in &base[1..] {
                assert!(b.is_zero());
            }
            base[0]
        })
        .collect()
}
