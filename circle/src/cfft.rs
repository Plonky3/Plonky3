use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;
use itertools::izip;
use p3_field::{
    extension::{Complex, ComplexExtendable},
    Field,
};
use p3_matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix, MatrixRows};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::twiddles::TwiddleCache;

#[derive(Default, Clone)]
pub struct Cfft<F>(Rc<RefCell<TwiddleCache<F>>>);

impl<F: ComplexExtendable> Cfft<F> {
    pub fn cfft(&self, vec: Vec<F>) -> Vec<F> {
        self.cfft_batch(RowMajorMatrix::new_col(vec)).values
    }
    pub fn cfft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let log_n = log2_strict_usize(mat.height());
        self.coset_cfft_batch(mat, F::circle_two_adic_generator(log_n + 1))
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    pub fn coset_cfft_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        shift: Complex<F>,
    ) -> RowMajorMatrix<F> {
        let n = mat.height();
        let log_n = log2_strict_usize(n);
        let width = mat.width();

        let mut cache = self.0.borrow_mut();
        let twiddles = cache.get_twiddles(log_n, shift, true);

        for (i, twiddle) in twiddles.iter().enumerate() {
            let block_size = 1 << (log_n - i);
            let half_block_size = block_size >> 1;

            for chunks in mat.values.chunks_exact_mut(block_size * width) {
                let (low_chunks, high_chunks) = chunks.split_at_mut(half_block_size * width);

                for (twiddle, lo, hi) in izip!(
                    twiddle,
                    low_chunks.chunks_exact_mut(width),
                    high_chunks.chunks_exact_mut(width).rev(),
                ) {
                    butterfly_cfft(lo, hi, *twiddle)
                }
            }
        }
        let inv_height = F::from_canonical_usize(n).inverse();
        mat.map(|x| x * inv_height)
    }

    pub fn icfft(&self, vec: Vec<F>) -> Vec<F> {
        self.icfft_batch(RowMajorMatrix::new_col(vec)).values
    }
    pub fn icfft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let log_n = log2_strict_usize(mat.height());
        self.coset_icfft_batch(mat, F::circle_two_adic_generator(log_n + 1))
    }
    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    pub fn coset_icfft_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        shift: Complex<F>,
    ) -> RowMajorMatrix<F> {
        let n = mat.height();
        let log_n = log2_strict_usize(n);
        let width = mat.width();

        let mut cache = self.0.borrow_mut();
        let twiddles = cache.get_twiddles(log_n, shift, false);

        for (i, twiddle) in twiddles.iter().rev().enumerate() {
            let block_size = 1 << (i + 1);
            let half_block_size = block_size >> 1;

            for chunks in mat.values.chunks_exact_mut(block_size * width) {
                let (low_chunks, high_chunks) = chunks.split_at_mut(half_block_size * width);

                for (twiddle, lo, hi) in izip!(
                    twiddle,
                    low_chunks.chunks_exact_mut(width),
                    high_chunks.chunks_exact_mut(width).rev(),
                ) {
                    butterfly_icfft(lo, hi, *twiddle)
                }
            }
        }

        mat
    }

    pub fn lde_batch(&self, mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        self.icfft_batch(pad_coeffs(self.cfft_batch(mat), added_bits))
    }
}

fn butterfly_cfft<F: Field>(low_chunk: &mut [F], high_chunk: &mut [F], twiddle: F) {
    for (low, high) in low_chunk.iter_mut().zip(high_chunk) {
        let sum = *low + *high;
        let diff = (*low - *high) * twiddle;
        *low = sum;
        *high = diff;
    }
}

fn butterfly_icfft<F: Field>(low_chunk: &mut [F], high_chunk: &mut [F], twiddle: F) {
    for (low, high) in low_chunk.iter_mut().zip(high_chunk) {
        let high_twiddle = *high * twiddle;
        let sum = *low + high_twiddle;
        let diff = *low - high_twiddle;
        *low = sum;
        *high = diff;
    }
}

pub(crate) fn pad_coeffs<F: Field>(
    mut coeffs: RowMajorMatrix<F>,
    added_bits: usize,
) -> RowMajorMatrix<F> {
    coeffs = coeffs.bit_reverse_rows().to_row_major_matrix();
    coeffs.expand_to_height(coeffs.height() << added_bits);
    coeffs.bit_reverse_rows().to_row_major_matrix()
}
