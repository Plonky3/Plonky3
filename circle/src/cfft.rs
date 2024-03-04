use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;
use itertools::izip;
use p3_dft::divide_by_height;
use p3_field::{
    extension::{Complex, ComplexExtendable},
    AbstractField, Field,
};
use p3_matrix::{
    bitrev::BitReversableMatrix,
    dense::{RowMajorMatrix, RowMajorMatrixViewMut},
    Matrix, MatrixRows,
};
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
            assert_eq!(twiddle.len(), half_block_size);

            mat.par_row_chunks_mut(block_size).for_each(|mut chunk| {
                for (i, &t) in twiddle.iter().enumerate() {
                    let ((pfx_lo, packed_lo, sfx_lo), (pfx_hi, packed_hi, sfx_hi)) =
                        chunk.packing_aligned_rows(i, block_size - i - 1);
                    butterfly(pfx_lo, pfx_hi, t);
                    butterfly(packed_lo, packed_hi, t.into());
                    butterfly(sfx_lo, sfx_hi, t);
                }
            });
        }
        divide_by_height(&mut mat);
        mat
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
            assert_eq!(twiddle.len(), half_block_size);

            mat.par_row_chunks_mut(block_size).for_each(|mut chunk| {
                for (i, &t) in twiddle.iter().enumerate() {
                    let ((pfx_lo, packed_lo, sfx_lo), (pfx_hi, packed_hi, sfx_hi)) =
                        chunk.packing_aligned_rows(i, block_size - i - 1);
                    ibutterfly(pfx_lo, pfx_hi, t);
                    ibutterfly(packed_lo, packed_hi, t.into());
                    ibutterfly(sfx_lo, sfx_hi, t);
                }
            });
        }

        mat
    }

    pub fn lde_batch(&self, mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        self.icfft_batch(pad_coeffs(self.cfft_batch(mat), added_bits))
    }
}

#[inline(always)]
fn butterfly<F: AbstractField + Copy>(lo_chunk: &mut [F], hi_chunk: &mut [F], twiddle: F) {
    for (lo, hi) in lo_chunk.iter_mut().zip(hi_chunk) {
        let sum = *lo + *hi;
        let diff = (*lo - *hi) * twiddle;
        *lo = sum;
        *hi = diff;
    }
}

#[inline(always)]
fn ibutterfly<F: AbstractField + Copy>(lo_chunk: &mut [F], hi_chunk: &mut [F], twiddle: F) {
    for (lo, hi) in lo_chunk.iter_mut().zip(hi_chunk) {
        let hi_twiddle = *hi * twiddle;
        let sum = *lo + hi_twiddle;
        let diff = *lo - hi_twiddle;
        *lo = sum;
        *hi = diff;
    }
}

fn butterflyfdsads<F: Field>(
    mat: &mut RowMajorMatrixViewMut<F>,
    idx_lo: usize,
    idx_hi: usize,
    twiddle: F,
) {
    let ((pfx_lo, packed_lo, sfx_lo), (pfx_hi, packed_hi, sfx_hi)) =
        mat.packing_aligned_rows(idx_lo, idx_hi);
    for (lo, hi) in pfx_lo.iter_mut().zip(pfx_hi) {
        let sum = *lo + *hi;
        let diff = (*lo - *hi) * twiddle;
        *lo = sum;
        *hi = diff;
    }
    for (lo, hi) in packed_lo.iter_mut().zip(packed_hi) {
        let sum = *lo + *hi;
        let diff = (*lo - *hi) * twiddle;
        *lo = sum;
        *hi = diff;
    }
    for (lo, hi) in sfx_lo.iter_mut().zip(sfx_hi) {
        let sum = *lo + *hi;
        let diff = (*lo - *hi) * twiddle;
        *lo = sum;
        *hi = diff;
    }
}

#[instrument(skip_all, fields(dims = %coeffs.dimensions(), added_bits))]
pub(crate) fn pad_coeffs<F: Field>(
    mut coeffs: RowMajorMatrix<F>,
    added_bits: usize,
) -> RowMajorMatrix<F> {
    coeffs = coeffs.bit_reverse_rows().to_row_major_matrix();
    coeffs.expand_to_height(coeffs.height() << added_bits);
    coeffs.bit_reverse_rows().to_row_major_matrix()
}

#[cfg(test)]
mod tests {
    use p3_mersenne_31::Mersenne31;
    use rand::{thread_rng, Rng};

    use crate::{
        domain::CircleDomain,
        util::{eval_circle_polys, univariate_to_point},
    };

    use super::*;

    type F = Mersenne31;

    fn do_test_cfft(log_n: usize) {
        let n = 1 << log_n;
        let cfft = Cfft::default();

        let shift: Complex<F> = univariate_to_point(thread_rng().gen());

        let evals = RowMajorMatrix::<F>::rand(&mut thread_rng(), n, 1 << 5);
        let coeffs = cfft.coset_cfft_batch(evals.clone(), shift);

        assert_eq!(evals.clone(), cfft.coset_icfft_batch(coeffs.clone(), shift));

        let d = CircleDomain { shift, log_n };
        for (pt, ys) in d.points().zip(evals.rows()) {
            assert_eq!(ys, eval_circle_polys(&coeffs, pt));
        }
    }

    #[test]
    fn test_cfft() {
        do_test_cfft(5);
        do_test_cfft(8);
    }
}
