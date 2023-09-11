use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_commit::{Dimensions, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, MatrixRows};
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

    pub(crate) _phantom_f: PhantomData<F>,
}

/// A claimed opening.
#[derive(Clone)]
pub(crate) struct Opening<F> {
    pub(crate) point: F,
    pub(crate) values: Vec<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>, Inner: Mmcs<F>> Mmcs<EF>
    for QuotientMmcs<F, EF, Inner>
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
                let x = F::primitive_root_of_unity(log2_height).exp_u64(reduced_index as u64);
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
                let g = F::primitive_root_of_unity(log2_height);
                let subgroup = g.powers().take(height).collect();
                QuotientMatrix {
                    inner,
                    subgroup,
                    openings,
                }
            })
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_quotient_values: Vec<Vec<EF>>,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        // quotient = (original - opened_eval) / (x - opened_point)
        // original = quotient * (x - opened_point) + opened_eval

        let log_max_height = dimensions
            .iter()
            .map(|dims| dims.log2_height)
            .max()
            .unwrap();

        let opened_original_values = izip!(opened_quotient_values, &self.openings, dimensions)
            .map(|(quotient_row, openings, dims)| {
                let bits_reduced = log_max_height - dims.log2_height;
                let reduced_index = index >> bits_reduced;
                let x = F::primitive_root_of_unity(dims.log2_height).exp_u64(reduced_index as u64);

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
            .collect();

        self.inner
            .verify_batch(commit, dimensions, index, opened_original_values, proof)
    }
}

pub struct QuotientMatrix<F, EF, Inner: MatrixRows<F>> {
    inner: Inner,
    subgroup: Vec<F>,
    openings: Vec<Opening<EF>>,
}

impl<F, EF, Inner: MatrixRows<F>> Matrix<EF> for QuotientMatrix<F, EF, Inner> {
    fn width(&self) -> usize {
        self.inner.width() * self.openings.len()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F: Field, EF: ExtensionField<F>, Inner: MatrixRows<F>> MatrixRows<EF>
    for QuotientMatrix<F, EF, Inner>
{
    type Row<'a> = QuotientMatrixRow<'a, F, EF, <Inner::Row<'a> as IntoIterator>::IntoIter> where Inner: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        let inner_row = r / self.openings.len();
        let opening_index = r % self.openings.len();

        let x = self.subgroup[inner_row];
        let opening = &self.openings[opening_index];
        let inv_denominator: EF = (EF::from_base(x) - opening.point).inverse();

        QuotientMatrixRow {
            opened_values: opening.values.iter(),
            inner_row_iter: self.inner.row(inner_row).into_iter(),
            inv_denominator,
            _phantom_f: PhantomData,
        }
    }
}

pub struct QuotientMatrixRow<'a, F, EF, InnerRowIter> {
    opened_values: core::slice::Iter<'a, EF>,
    inner_row_iter: InnerRowIter,
    /// `1 / (x - opened_point)`
    inv_denominator: EF,
    _phantom_f: PhantomData<F>,
}

impl<'a, F, EF, InnerRowIter> Iterator for QuotientMatrixRow<'a, F, EF, InnerRowIter>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerRowIter: Iterator<Item = F>,
{
    type Item = EF;

    fn next(&mut self) -> Option<Self::Item> {
        let opt_eval = self.inner_row_iter.next();
        let opt_opened_value = self.opened_values.next();

        if let (Some(eval), Some(opened_value)) = (opt_eval, opt_opened_value) {
            let num: EF = EF::from_base(eval) - *opened_value;
            Some(num * self.inv_denominator)
        } else {
            debug_assert!(opt_eval.is_none());
            debug_assert!(opt_opened_value.is_none());
            None
        }
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
