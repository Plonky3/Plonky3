use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_commit::Mmcs;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, ExtensionField, Field,
    TwoAdicField,
};
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};
use p3_util::log2_strict_usize;

/// A wrapper around an Inner MMCS, which transforms each inner value to
/// `(inner - opened_point) / (x - opened_eval)`.
///
/// Since there can be multiple opening points, for each matrix, this transforms an inner opened row
/// into a concatenation of rows, transformed as above, for each point.
#[derive(Clone)]
pub struct QuotientMmcs<F, EF, Inner: Mmcs<F>> {
    pub(crate) inner: Inner,

    /// For each matrix, a list of claimed openings, one for each point that we open that batch of
    /// polynomials at.
    pub(crate) openings: Vec<Vec<Opening<EF>>>,

    // The coset shift for the inner MMCS's evals, to correct `x` in the denominator.
    pub(crate) coset_shift: F,
}

/// A claimed opening.
#[derive(Clone)]
pub(crate) struct Opening<F> {
    pub(crate) point: F,
    pub(crate) values: Vec<F>,
}

impl<F, EF, Inner> Mmcs<EF> for QuotientMmcs<F, EF, Inner>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Inner: Mmcs<F>,
    for<'a> Inner::Mat<'a>: MatrixRowSlices<F>,
{
    type ProverData = Inner::ProverData;
    type Commitment = Inner::Commitment;
    type Proof = Inner::Proof;
    type Error = Inner::Error;
    type Mat<'a> = QuotientMatrix<F, EF, Inner::Mat<'a>> where Self: 'a;

    fn open_batch(
        &self,
        index: usize,
        prover_data: &Self::ProverData,
    ) -> (Vec<Vec<EF>>, Self::Proof) {
        let (inner_values, proof) = self.inner.open_batch(index, prover_data);
        let matrix_heights = self.inner.get_matrix_heights(prover_data);
        let max_height = *matrix_heights.iter().max().unwrap();
        let log_max_height = log2_strict_usize(max_height);

        let quotients = izip!(inner_values, self.openings.clone(), matrix_heights)
            .map(|(inner_row, openings_for_mat, height)| {
                let log2_height = log2_strict_usize(height);
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                let x = F::two_adic_generator(log2_height).exp_u64(reduced_index as u64);
                openings_for_mat
                    .iter()
                    .flat_map(|Opening { point, values }| {
                        inner_row
                            .iter()
                            .zip(values)
                            .map(|(&inner_value, &opened_value)| {
                                (EF::from_base(inner_value) - opened_value)
                                    / (EF::from_base(x) - *point)
                            })
                    })
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
                let subgroup = cyclic_subgroup_coset_known_order(g, self.coset_shift, height);

                let denominators: Vec<EF> = subgroup
                    .flat_map(|x| {
                        openings
                            .iter()
                            .map(move |opening| EF::from_base(x) - opening.point)
                    })
                    .collect();
                let inv_denominators = batch_multiplicative_inverse(&denominators);

                QuotientMatrix {
                    inner,
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
        opened_quotient_values: &[Vec<EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        // quotient = (original - opened_eval) / (x - opened_point)
        // original = quotient * (x - opened_point) + opened_eval

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
                let x = F::two_adic_generator(log_height).exp_u64(reduced_index as u64);

                let original_width = quotient_row.len() / openings.len();
                let original_row_repeated: Vec<Vec<EF>> = quotient_row
                    .chunks(original_width)
                    .zip(openings)
                    .map(|(quotient_row_chunk, opening)| {
                        quotient_row_chunk
                            .iter()
                            .zip(&opening.values)
                            .map(|(&quotient_value, &opened_value)| {
                                quotient_value * (EF::from_base(x) - opening.point) + opened_value
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                let original_row = get_repeated(original_row_repeated.into_iter());
                to_base::<F, EF>(original_row)
            })
            .collect_vec();

        self.inner
            .verify_batch(commit, dimensions, index, &opened_original_values, proof)
    }
}

pub struct QuotientMatrix<F, EF, Inner: MatrixRowSlices<F>> {
    inner: Inner,
    openings: Vec<Opening<EF>>,
    /// For each row (associated with a subgroup element `x`), for each opening point,
    /// this holds `1 / (x - opened_point)`.
    inv_denominators: Vec<EF>,
    _phantom: PhantomData<F>,
}

impl<F, EF, Inner: MatrixRowSlices<F>> Matrix<EF> for QuotientMatrix<F, EF, Inner> {
    fn width(&self) -> usize {
        self.inner.width() * self.openings.len()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F: Field, EF: ExtensionField<F>, Inner: MatrixRowSlices<F>> MatrixRows<EF>
    for QuotientMatrix<F, EF, Inner>
{
    type Row<'a> = QuotientMatrixRow<'a, F, EF> where Inner: 'a;

    #[inline]
    fn row(&self, r: usize) -> Self::Row<'_> {
        let num_openings = self.openings.len();
        QuotientMatrixRow {
            openings: &self.openings,
            inv_denominator: &self.inv_denominators[r * num_openings..(r + 1) * num_openings],
            inner_row: self.inner.row_slice(r),
            opening_index: 0,
            inner_col_index: 0,
        }
    }
}

pub struct QuotientMatrixRow<'a, F, EF> {
    openings: &'a [Opening<EF>],
    /// `1 / (x - opened_point)`
    inv_denominator: &'a [EF],
    inner_row: &'a [F],
    opening_index: usize,
    inner_col_index: usize,
}

impl<'a, F, EF> Iterator for QuotientMatrixRow<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Item = EF;

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
        let opened_value = opening.values[self.inner_col_index];
        let numerator: EF = EF::from_base(eval) - opened_value;
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
