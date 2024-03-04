use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;
use itertools::izip;
use p3_commit::PolynomialSpace;
use p3_dft::{bit_reversed_zero_pad, divide_by_height};
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

use crate::{domain::CircleDomain, twiddles::TwiddleCache};

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

    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    pub fn lde(
        &self,
        mut mat: RowMajorMatrix<F>,
        src_domain: CircleDomain<F>,
        target_domain: CircleDomain<F>,
    ) -> RowMajorMatrix<F> {
        assert_eq!(mat.height(), src_domain.size());

        // CFFT
        // let mut coeffs = self.coset_cfft_batch(evals, src_domain.shift);

        let mut cache = self.0.borrow_mut();
        let f_twiddles = cache.get_twiddles(src_domain.log_n, src_domain.shift, true);

        for (i, twiddle) in f_twiddles.iter().enumerate() {
            let block_size = 1 << (src_domain.log_n - i);
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

        assert!(target_domain.size() >= src_domain.size());

        let added_bits = target_domain.log_n - src_domain.log_n;

        /*
        To do an LDE, we could interleave zeros into the coefficients, but
        the first `added_bits` layers will simply fill out the zeros with the
        lower order values. (In `ibutterfly`, `hi` will start as zero, and
        both `lo` and `hi` are set to `lo`). So instead, we do the tiling directly
        and skip the first `added_bits` layers.
        */
        // bit_reversed_zero_pad(&mut mat, added_bits);
        mat = tile_rows(mat, 1 << added_bits);

        assert_eq!(mat.height(), target_domain.size());

        // ICFFT
        // self.coset_icfft_batch(mat, target_domain.shift)

        let i_twiddles = cache.get_twiddles(target_domain.log_n, target_domain.shift, false);

        for (i, twiddle) in i_twiddles.iter().rev().enumerate().skip(added_bits) {
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

// Repeats rows
// this can be micro-optimized
fn tile_rows<F: Copy>(mut mat: RowMajorMatrix<F>, repetitions: usize) -> RowMajorMatrix<F> {
    let mut values = Vec::with_capacity(mat.values.len() * repetitions);
    for r in mat.rows() {
        for _ in 0..repetitions {
            values.extend_from_slice(r);
        }
    }
    mat.values = values;
    mat
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

    fn do_test_lde(log_n: usize, added_bits: usize) {
        let n = 1 << log_n;
        let cfft = Cfft::<F>::default();

        let shift: Complex<F> = univariate_to_point(thread_rng().gen());

        let evals = RowMajorMatrix::<F>::rand(&mut thread_rng(), n, 1);
        let src_domain = CircleDomain { log_n, shift };
        let target_domain = CircleDomain::standard(log_n + added_bits);

        let mut coeffs = cfft.coset_cfft_batch(evals.clone(), src_domain.shift);
        bit_reversed_zero_pad(&mut coeffs, added_bits);
        let expected = cfft.coset_icfft_batch(coeffs, target_domain.shift);

        let actual = cfft.lde(evals, src_domain, target_domain);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_lde() {
        do_test_lde(3, 1);
    }
}
