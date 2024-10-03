use alloc::vec::Vec;

use p3_field::TwoAdicField;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::instrument;

use crate::TwoAdicSubgroupDft;

#[derive(Default, Clone, Debug)]
pub struct ChunkedDft<Inner, const CHUNK_SIZE: usize> {
    inner: Inner,
}

impl<F, Inner, const CHUNK_SIZE: usize> TwoAdicSubgroupDft<F> for ChunkedDft<Inner, CHUNK_SIZE>
where
    F: TwoAdicField,
    Inner: TwoAdicSubgroupDft<F, Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>>,
{
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

    #[instrument(skip_all)]
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let input_chunks = split::<F, CHUNK_SIZE>(mat);
        let dft_chunks = input_chunks
            .into_iter()
            .map(|chunk| self.inner.dft_batch(chunk))
            .collect();
        join::<F, CHUNK_SIZE>(dft_chunks)
    }

    #[instrument(skip_all)]
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        let input_chunks = split::<F, CHUNK_SIZE>(mat);
        let dft_chunks = input_chunks
            .into_iter()
            .map(|chunk| self.inner.coset_lde_batch(chunk, added_bits, shift))
            .collect();
        join::<F, CHUNK_SIZE>(dft_chunks)
    }
}

#[instrument(skip_all)]
fn split<F: TwoAdicField, const CHUNK_SIZE: usize>(
    mat: RowMajorMatrix<F>,
) -> Vec<RowMajorMatrix<F>> {
    let height = mat.height();
    let num_chunks = mat.width.div_ceil(CHUNK_SIZE);
    (0..num_chunks)
        .into_par_iter()
        .map(|i| {
            let col_start = i * CHUNK_SIZE;
            let col_end = ((i + 1) * CHUNK_SIZE).min(mat.width);
            let width = col_end - col_start;
            let mut chunk = Vec::with_capacity(width * height);
            for r in 0..height {
                chunk.extend(mat.row(r).skip(col_start).take(width));
            }
            RowMajorMatrix::new(chunk, width)
        })
        .collect()
}

#[instrument(skip_all)]
fn join<F: TwoAdicField, const CHUNK_SIZE: usize>(
    chunks: Vec<BitReversedMatrixView<RowMajorMatrix<F>>>,
) -> BitReversedMatrixView<RowMajorMatrix<F>> {
    let height = chunks[0].height();
    let width = chunks.iter().map(|c| c.width()).sum();
    let mut mat = Vec::with_capacity(width * height);
    for r in 0..height {
        for chunk in &chunks {
            mat.extend(chunk.inner.row(r));
        }
    }
    let matrix = RowMajorMatrix::new(mat, width);
    BitReversalPerm::new_view(matrix)
}
