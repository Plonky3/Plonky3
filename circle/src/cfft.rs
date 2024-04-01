//! The Circle FFT and its inverse, as detailed in
//! Circle STARKs, Section 4.2 (page 14 of the first revision PDF)
//! This code is based on Angus Gruen's implementation, which uses a slightly
//! different cfft basis than that of the paper. Basically, it continues using the
//! same twiddles for the second half of the chunk, which only changes the sign of the
//! resulting basis. For a full explanation see the comments in `util::circle_basis`.
//! This alternate basis doesn't cause any change to the code apart from our testing functions.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use p3_commit::PolynomialSpace;
use p3_dft::divide_by_height;
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{AbstractField, Field, PackedValue};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::domain::CircleDomain;
use crate::twiddles::TwiddleCache;

#[derive(Default, Clone)]
pub struct Cfft<F>(Rc<RefCell<TwiddleCache<F>>>);

impl<F: ComplexExtendable> Cfft<F> {
    pub fn cfft(&self, vec: Vec<F>) -> Vec<F> {
        self.cfft_batch(DenseMatrix::new_col(vec)).values
    }
    pub fn cfft_batch(&self, mat: DenseMatrix<F>) -> DenseMatrix<F> {
        let log_n = log2_strict_usize(mat.height());
        self.coset_cfft_batch(mat, F::circle_two_adic_generator(log_n + 1))
    }
    /// The cfft: interpolating evaluations over a domain to the (sign-switched) cfft basis
    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    pub fn coset_cfft_batch(&self, mut mat: DenseMatrix<F>, shift: Complex<F>) -> DenseMatrix<F> {
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
                    let (lo, hi) = chunk.row_pair_mut(i, block_size - i - 1);
                    let (lo_packed, lo_suffix) = F::Packing::pack_slice_with_suffix_mut(lo);
                    let (hi_packed, hi_suffix) = F::Packing::pack_slice_with_suffix_mut(hi);
                    dif_butterfly(lo_packed, hi_packed, t.into());
                    dif_butterfly(lo_suffix, hi_suffix, t);
                }
            });
        }
        // TODO: omit this?
        divide_by_height(&mut mat);
        mat
    }

    pub fn icfft(&self, vec: Vec<F>) -> Vec<F> {
        self.icfft_batch(RowMajorMatrix::new_col(vec)).values
    }
    pub fn icfft_batch(&self, mat: DenseMatrix<F>) -> DenseMatrix<F> {
        let log_n = log2_strict_usize(mat.height());
        self.coset_icfft_batch(mat, F::circle_two_adic_generator(log_n + 1))
    }
    /// The icfft: evaluating a polynomial in monomial basis over a domain
    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    pub fn coset_icfft_batch(&self, mat: DenseMatrix<F>, shift: Complex<F>) -> DenseMatrix<F> {
        self.coset_icfft_batch_skipping_first_layers(mat, shift, 0)
    }
    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    fn coset_icfft_batch_skipping_first_layers(
        &self,
        mut mat: DenseMatrix<F>,
        shift: Complex<F>,
        num_skipped_layers: usize,
    ) -> DenseMatrix<F> {
        let n = mat.height();
        let log_n = log2_strict_usize(n);

        let mut cache = self.0.borrow_mut();
        let twiddles = cache.get_twiddles(log_n, shift, false);

        for (i, twiddle) in twiddles.iter().rev().enumerate().skip(num_skipped_layers) {
            let block_size = 1 << (i + 1);
            let half_block_size = block_size >> 1;
            assert_eq!(twiddle.len(), half_block_size);

            mat.par_row_chunks_mut(block_size).for_each(|mut chunk| {
                for (i, &t) in twiddle.iter().enumerate() {
                    let (lo, hi) = chunk.row_pair_mut(i, block_size - i - 1);
                    let (lo_packed, lo_suffix) = F::Packing::pack_slice_with_suffix_mut(lo);
                    let (hi_packed, hi_suffix) = F::Packing::pack_slice_with_suffix_mut(hi);
                    dit_butterfly(lo_packed, hi_packed, t.into());
                    dit_butterfly(lo_suffix, hi_suffix, t);
                }
            });
        }

        mat
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions()))]
    pub fn lde(
        &self,
        mut mat: DenseMatrix<F>,
        src_domain: CircleDomain<F>,
        target_domain: CircleDomain<F>,
    ) -> DenseMatrix<F> {
        assert_eq!(mat.height(), src_domain.size());
        assert!(target_domain.size() >= src_domain.size());
        let added_bits = target_domain.log_n - src_domain.log_n;

        // CFFT
        mat = self.coset_cfft_batch(mat, src_domain.shift);

        /*
        To do an LDE, we could interleave zeros into the coefficients, but
        the first `added_bits` layers will simply fill out the zeros with the
        lower order values. (In `ibutterfly`, `hi` will start as zero, and
        both `lo` and `hi` are set to `lo`). So instead, we do the tiling directly
        and skip the first `added_bits` layers.
        */
        let tiled_mat = tile_rows(mat, 1 << added_bits);
        debug_assert_eq!(tiled_mat.height(), target_domain.size());

        self.coset_icfft_batch_skipping_first_layers(tiled_mat, target_domain.shift, added_bits)
    }
}

/// Division-in-frequency
#[inline(always)]
fn dif_butterfly<F: AbstractField + Copy>(lo_chunk: &mut [F], hi_chunk: &mut [F], twiddle: F) {
    for (lo, hi) in lo_chunk.iter_mut().zip(hi_chunk) {
        let sum = *lo + *hi;
        let diff = (*lo - *hi) * twiddle;
        *lo = sum;
        *hi = diff;
    }
}

/// Division-in-time
#[inline(always)]
fn dit_butterfly<F: AbstractField + Copy>(lo_chunk: &mut [F], hi_chunk: &mut [F], twiddle: F) {
    for (lo, hi) in lo_chunk.iter_mut().zip(hi_chunk) {
        let hi_twiddle = *hi * twiddle;
        let sum = *lo + hi_twiddle;
        let diff = *lo - hi_twiddle;
        *lo = sum;
        *hi = diff;
    }
}

// Repeats rows
// TODO this can be micro-optimized
fn tile_rows<F: Field>(mat: impl Matrix<F>, repetitions: usize) -> RowMajorMatrix<F> {
    let mut values = Vec::with_capacity(mat.width() * mat.height() * repetitions);
    for r in 0..mat.height() {
        let s = mat.row_slice(r);
        for _ in 0..repetitions {
            values.extend_from_slice(s.as_ref());
        }
    }
    RowMajorMatrix::new(values, mat.width())
}

#[cfg(test)]
mod tests {
    use p3_dft::bit_reversed_zero_pad;
    use p3_mersenne_31::Mersenne31;
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::domain::CircleDomain;
    use crate::util::{eval_circle_polys, univariate_to_point};

    type F = Mersenne31;

    fn do_test_cfft(log_n: usize) {
        let n = 1 << log_n;
        let cfft = Cfft::default();

        let shift: Complex<F> = univariate_to_point(thread_rng().gen()).unwrap();

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

        let shift: Complex<F> = univariate_to_point(thread_rng().gen()).unwrap();

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
