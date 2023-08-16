use alloc::vec::Vec;

use p3_commit::{Dimensions, Mmcs};
use p3_field::{Field, TwoAdicField};
use p3_matrix::{Matrix, MatrixRows};
use p3_util::log2_strict_usize;

/// A wrapper around an Inner MMCS, which transforms each inner value to
/// `(inner - opened_point) / (x - opened_eval)`.
pub(crate) struct QuotientMmcs<F, Inner: Mmcs<F>> {
    inner: Inner,
    opened_point: F,
    opened_eval: F,
}

impl<F: TwoAdicField, Inner: Mmcs<F>> Mmcs<F> for QuotientMmcs<F, Inner> {
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
        let (openings, proof) = self.inner.open_batch(index, prover_data);
        let matrix_heights = self.inner.get_matrix_heights(prover_data);
        let max_height = *matrix_heights.iter().max().unwrap();
        let log_max_height = log2_strict_usize(max_height);

        let quotients = openings
            .into_iter()
            .zip(matrix_heights)
            .map(|(row, height)| {
                let log2_height = log2_strict_usize(height);
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                let x = F::primitive_root_of_unity(log2_height).exp_u64(reduced_index as u64);
                row.into_iter()
                    .map(|value| (value - self.opened_eval) / (x - self.opened_point))
                    .collect()
            })
            .collect();

        (quotients, proof)
    }

    fn get_matrices<'a>(&'a self, prover_data: &'a Self::ProverData) -> Vec<Self::Mat<'a>> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|inner| {
                let height = inner.height();
                let g = F::primitive_root_of_unity(log2_strict_usize(height));
                let subgroup = g.powers().take(height).collect();
                QuotientMatrix {
                    inner,
                    subgroup,
                    opened_point: self.opened_point,
                    opened_eval: self.opened_eval,
                }
            })
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_quotient_values: Vec<Vec<F>>,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        // quotient = (original - opened_eval) / (x - opened_point)
        // original = quotient * (x - opened_point) + opened_eval

        let log_max_height = dimensions
            .iter()
            .map(|dims| dims.log2_height)
            .max()
            .unwrap();

        let opened_original_values = opened_quotient_values
            .into_iter()
            .zip(dimensions)
            .map(|(row, dims)| {
                let bits_reduced = log_max_height - dims.log2_height;
                let reduced_index = index >> bits_reduced;
                let x = F::primitive_root_of_unity(dims.log2_height).exp_u64(reduced_index as u64);
                row.into_iter()
                    .map(|quotient| quotient * (x - self.opened_point) + self.opened_eval)
                    .collect()
            })
            .collect();
        self.inner
            .verify_batch(commit, dimensions, index, opened_original_values, proof)
    }
}

pub(crate) struct QuotientMatrix<F, Inner: MatrixRows<F>> {
    inner: Inner,
    subgroup: Vec<F>,
    opened_point: F,
    opened_eval: F,
}

impl<F, Inner: MatrixRows<F>> Matrix<F> for QuotientMatrix<F, Inner> {
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F: TwoAdicField, Inner: MatrixRows<F>> MatrixRows<F> for QuotientMatrix<F, Inner> {
    type Row<'a> = QuotientRow<F, <Inner::Row<'a> as IntoIterator>::IntoIter> where Inner: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        QuotientRow {
            inner: self.inner.row(r).into_iter(),
            current_point: self.subgroup[r],
            opened_point: self.opened_point,
            opened_eval: self.opened_eval,
        }
    }
}

pub(crate) struct QuotientRow<F, Inner: Iterator<Item = F>> {
    inner: Inner,
    current_point: F,
    opened_point: F,
    opened_eval: F,
}

impl<F: Field, Inner: Iterator<Item = F>> Iterator for QuotientRow<F, Inner> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|value| (value - self.opened_eval) / (self.current_point - self.opened_point))
    }
}
