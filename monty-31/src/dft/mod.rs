//! An implementation of the FFT for `MontyField31`
extern crate alloc;

use alloc::vec::Vec;
use core::cell::RefCell;

use p3_dft::util::{coset_shift_cols, coset_shift_rows};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, Field};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::split_at_mut_unchecked;
use tracing::{debug_span, instrument};

mod backward;
mod forward;

use crate::{FieldParameters, MontyField31, MontyParameters, TwoAdicData};

/// Radix-2 decimation-in-frequency FFT
#[derive(Clone, Debug, Default)]
pub struct Radix2Dif<F> {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    twiddles: RefCell<Vec<Vec<F>>>,
    inv_twiddles: RefCell<Vec<Vec<F>>>,
}

impl<MP: FieldParameters + TwoAdicData> Radix2Dif<MontyField31<MP>> {
    pub fn new(n: usize) -> Self {
        Self {
            twiddles: RefCell::new(MontyField31::roots_of_unity_table(n)),
            inv_twiddles: RefCell::new(MontyField31::inv_roots_of_unity_table(n)),
        }
    }

    #[inline]
    fn decimation_in_freq_dft(
        mat: &mut RowMajorMatrix<MontyField31<MP>>,
        twiddles: &Vec<Vec<MontyField31<MP>>>,
    ) {
        let lg_fft_len = p3_util::log2_ceil_usize(mat.width());

        let roots_idx = (twiddles.len() + 1) - lg_fft_len;
        let twiddles = &twiddles[roots_idx..];

        mat.par_rows_mut()
            .for_each(|v| MontyField31::forward_fft(v, twiddles))
    }

    #[inline]
    fn decimation_in_time_dft(
        mat: &mut RowMajorMatrix<MontyField31<MP>>,
        twiddles: &Vec<Vec<MontyField31<MP>>>,
    ) {
        let lg_fft_len = p3_util::log2_ceil_usize(mat.width());

        let roots_idx = (twiddles.len() + 1) - lg_fft_len;
        let twiddles = &twiddles[roots_idx..];

        mat.par_rows_mut()
            .for_each(|v| MontyField31::backward_fft(v, twiddles))
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    #[inline]
    pub fn dft_batch_rows(&self, mat: &mut RowMajorMatrix<MontyField31<MP>>)
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        // Compute twiddle factors, or take memoized ones if already available.
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        // TODO: This currently maintains both tables at the same length, but this is
        // unnecessary and somewhat expensive.
        debug_span!("maybe calculate twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.twiddles.borrow().len();
            if mat.width() > curr_max_fft_len {
                let new_twiddles = MontyField31::roots_of_unity_table(mat.width());
                self.twiddles.replace(new_twiddles);
            }
        });

        // TODO: We're only cloning because of the RefCell; it
        // shouldn't be necessary, though it only costs ~20μs.
        let twiddles = debug_span!("clone twiddles").in_scope(|| self.twiddles.borrow().clone());

        debug_span!("parallel dft", n_dfts = mat.height(), lengths = mat.width())
            .in_scope(|| Self::decimation_in_time_dft(mat, &twiddles));
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    #[inline]
    pub fn idft_batch_rows(&self, mat: &mut RowMajorMatrix<MontyField31<MP>>)
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        // Compute twiddle factors, or take memoized ones if already available.
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        // TODO: This currently maintains both tables at the same length, but this is
        // unnecessary and somewhat expensive.
        debug_span!("maybe calculate inv_twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.inv_twiddles.borrow().len();
            if mat.width() > curr_max_fft_len {
                let new_twiddles = MontyField31::inv_roots_of_unity_table(mat.width());
                self.inv_twiddles.replace(new_twiddles);
            }
        });

        // TODO: We're only cloning because of the RefCell; it
        // shouldn't be necessary, though it only costs ~20μs.
        let inv_twiddles =
            debug_span!("clone inv_twiddles").in_scope(|| self.inv_twiddles.borrow().clone());

        debug_span!(
            "parallel idft",
            n_dfts = mat.height(),
            lengths = mat.width()
        )
        .in_scope(|| Self::decimation_in_freq_dft(mat, &inv_twiddles));

        let inv_len = MontyField31::from_canonical_usize(mat.width()).inverse();
        // TODO: mat.scale() is not parallelised...
        mat.scale(inv_len);
    }

    pub fn idft_batch_cols_transposed_bitrevd(
        &self,
        mat: &RowMajorMatrix<MontyField31<MP>>,
        out: &mut RowMajorMatrix<MontyField31<MP>>,
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        if mat.height() <= 1 {
            out.values.copy_from_slice(&mat.values);
            return;
        }

        // transpose input
        debug_span!("pre-transpose", nrows = mat.height(), ncols = mat.width())
            .in_scope(|| mat.transpose_into(out));

        self.idft_batch_rows(out);
    }

    pub fn idft_batch_cols_bitrevd(
        &self,
        mat: &mut RowMajorMatrix<MontyField31<MP>>,
        out: &mut RowMajorMatrix<MontyField31<MP>>,
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        self.idft_batch_cols_transposed_bitrevd(mat, out);

        // transpose output
        debug_span!("post-transpose", nrows = out.height(), ncols = out.width())
            .in_scope(|| out.transpose_into(mat));
    }

    pub fn dft_batch_cols_transposed_bitrevd(
        &self,
        mat: &RowMajorMatrix<MontyField31<MP>>,
        out: &mut RowMajorMatrix<MontyField31<MP>>,
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        if mat.height() <= 1 {
            out.values.copy_from_slice(&mat.values);
            return;
        }

        // transpose input
        debug_span!("pre-transpose", nrows = mat.height(), ncols = mat.width())
            .in_scope(|| mat.transpose_into(out));

        self.dft_batch_rows(out);
    }

    pub fn dft_batch_cols_bitrevd(
        &self,
        mat: &mut RowMajorMatrix<MontyField31<MP>>,
        out: &mut RowMajorMatrix<MontyField31<MP>>,
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        self.dft_batch_cols_transposed_bitrevd(mat, out);

        // transpose output
        debug_span!("post-transpose", nrows = out.height(), ncols = out.width())
            .in_scope(|| out.transpose_into(mat));
    }
}

/// DFT implementation that uses DIT for forward direction and DIF for backward.
/// Hence a (coset) LDE
///
///   - IDFT
///   - zero extend
///   - DFT
///
/// expands to
///
///   - invDFT DIF
///     - result is bit-reversed
///   - zero extend bitrevd result
///   - DFT DIT
///     - input is already bit-reversed
///
impl<MP: MontyParameters + FieldParameters + TwoAdicData> TwoAdicSubgroupDft<MontyField31<MP>>
    for Radix2Dif<MontyField31<MP>>
{
    type Evaluations = RowMajorMatrix<MontyField31<MP>>;

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn dft_batch(&self, mat: RowMajorMatrix<MontyField31<MP>>) -> Self::Evaluations
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(mat.height(), mat.width()));

        // TODO: In principle bit reversal should only be necessary when
        // doing the transform inplace, though it might still be
        // beneficial for memory coherence.
        let mut mat =
            debug_span!("initial bitrev").in_scope(|| mat.bit_reverse_rows().to_row_major_matrix());

        debug_span!("dft batch").in_scope(|| self.dft_batch_cols_bitrevd(&mut mat, &mut scratch));
        mat
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn idft_batch(
        &self,
        mut mat: RowMajorMatrix<MontyField31<MP>>,
    ) -> RowMajorMatrix<MontyField31<MP>>
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(mat.height(), mat.width()));

        debug_span!("idft batch").in_scope(|| self.idft_batch_cols_bitrevd(&mut mat, &mut scratch));

        // TODO: In principle bit reversal should only be necessary when
        // doing the transform inplace, though it might still be
        // beneficial for memory coherence.
        debug_span!("final bitrev").in_scope(|| mat.bit_reverse_rows().to_row_major_matrix())
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<MontyField31<MP>>,
        added_bits: usize,
        shift: MontyField31<MP>,
    ) -> Self::Evaluations {
        let result_height = mat
            .height()
            .checked_shl(added_bits.try_into().unwrap())
            .unwrap();

        let mut coeffs = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(mat.height(), mat.width()));

        // TODO: Ensure that we only calculate twiddle factors once;
        // at the moment we calculate a smaller table first then the
        // bigger table, but if we did the bigger table first we
        // wouldn't need to do the smaller table at all.

        self.idft_batch_cols_transposed_bitrevd(&mat, &mut coeffs);

        // At this point the inverse FFT of each column of `mat` appears
        // as a row in `coeffs`.

        // Extend coeffs by a suitable number of zeros
        // FIXME: do row-wise and incorporate bitrev; base on bit_reversed_zero_pad
        // TODO: Shouldn't resize here, should allocate once at the start
        let mut padded = coeffs.bit_reversed_zero_pad_rows(added_bits);

        // TODO: coset_shift_rows will have poor memory coherence; could be
        // a good argument for integrating coset shift into twiddles?
        // TODO: Half of coeffs was just set to zero; make sure we're not
        // pointlessly multiplying zero values by shift powers
        coset_shift_rows(&mut padded, shift);

        self.dft_batch_rows(&mut padded);

        // FIXME: Find a way to reuse the scratch space from above
        let mut out = debug_span!("allocate output space")
            .in_scope(|| RowMajorMatrix::default(mat.width(), result_height));

        // transpose output
        debug_span!(
            "post-transpose",
            nrows = padded.height(),
            ncols = padded.width()
        )
        .in_scope(|| padded.transpose_into(&mut out));

        out
    }
}
