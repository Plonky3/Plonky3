//! An implementation of the FFT for `MontyField31`
extern crate alloc;

use alloc::vec::Vec;
use core::cell::RefCell;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, Field};
use p3_matrix::bitrev::{BitReversableMatrix, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::split_at_mut_unchecked;
use tracing::{debug_span, instrument};

mod backward;
mod forward;

use crate::{FieldParameters, MontyField31, MontyParameters, TwoAdicData};

#[instrument(level = "debug", skip_all)]
fn zero_pad_rows<T>(output: &mut [T], input: &[T], nrows: usize, ncols: usize, added_bits: usize)
where
    T: Copy + Default + Send + Sync,
{
    if added_bits == 0 {
        output.copy_from_slice(input);
        return;
    }

    let new_ncols = ncols << added_bits;
    assert_eq!(input.len(), nrows * ncols);
    assert_eq!(output.len(), nrows * new_ncols);

    output
        .par_chunks_exact_mut(new_ncols)
        .zip(input.par_chunks_exact(ncols))
        .for_each(|(padded_row, row)| {
            padded_row[..ncols].copy_from_slice(row);
        });
}

/// Multiply each element of column `j` of `mat` by `shift**j`.
#[instrument(level = "debug", skip_all)]
fn coset_shift_rows<F: Field>(mat: &mut [F], ncols: usize, shift: F, scale: F) {
    let powers = shift.shifted_powers(scale).take(ncols).collect::<Vec<_>>();
    mat.par_chunks_exact_mut(ncols).for_each(|row| {
        row.iter_mut().zip(&powers).for_each(|(coeff, &weight)| {
            *coeff *= weight;
        })
    });
}

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
        mat: &mut [MontyField31<MP>],
        ncols: usize,
        twiddles: &[Vec<MontyField31<MP>>],
    ) {
        if ncols > 1 {
            let lg_fft_len = p3_util::log2_ceil_usize(ncols);
            let roots_idx = (twiddles.len() + 1) - lg_fft_len;
            let twiddles = &twiddles[roots_idx..];

            mat.par_chunks_exact_mut(ncols)
                .for_each(|v| MontyField31::forward_fft(v, twiddles))
        }
    }

    #[inline]
    fn decimation_in_time_dft(
        mat: &mut [MontyField31<MP>],
        ncols: usize,
        twiddles: &[Vec<MontyField31<MP>>],
    ) {
        if ncols > 1 {
            let lg_fft_len = p3_util::log2_ceil_usize(ncols);
            let roots_idx = (twiddles.len() + 1) - lg_fft_len;
            let twiddles = &twiddles[roots_idx..];

            mat.par_chunks_exact_mut(ncols)
                .for_each(|v| MontyField31::backward_fft(v, twiddles))
        }
    }

    /// Compute twiddle factors, or take memoized ones if already available.
    fn update_twiddles(&self, fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        debug_span!("maybe calculate twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.twiddles.borrow().len();
            if fft_len > curr_max_fft_len {
                let new_twiddles = MontyField31::roots_of_unity_table(fft_len);
                self.twiddles.replace(new_twiddles);
            }
        });
    }

    fn update_inv_twiddles(&self, inv_fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        debug_span!("maybe calculate inv_twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.inv_twiddles.borrow().len();
            if inv_fft_len > curr_max_fft_len {
                let new_twiddles = MontyField31::inv_roots_of_unity_table(inv_fft_len);
                self.inv_twiddles.replace(new_twiddles);
            }
        });
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
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<MontyField31<MP>>>;

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn dft_batch(&self, mut mat: RowMajorMatrix<MontyField31<MP>>) -> Self::Evaluations
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let nrows = mat.height();
        let ncols = mat.width();
        if nrows <= 1 {
            return mat.bit_reverse_rows();
        }

        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(nrows, ncols));

        self.update_twiddles(nrows);
        let twiddles = self.twiddles.borrow().clone();

        // transpose input
        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(&mat.values, &mut scratch.values, ncols, nrows));

        debug_span!("dft batch", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_freq_dft(&mut scratch.values, nrows, &twiddles));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = nrows)
            .in_scope(|| transpose::transpose(&scratch.values, &mut mat.values, nrows, ncols));

        mat.bit_reverse_rows()
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn idft_batch(&self, mat: RowMajorMatrix<MontyField31<MP>>) -> RowMajorMatrix<MontyField31<MP>>
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let nrows = mat.height();
        let ncols = mat.width();
        if nrows <= 1 {
            return mat;
        }

        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(nrows, ncols));

        // TODO: Consider doing this in-place?
        // TODO: Use faster bit-reversal algo
        let mut mat =
            debug_span!("initial bitrev").in_scope(|| mat.bit_reverse_rows().to_row_major_matrix());

        self.update_inv_twiddles(nrows);
        let inv_twiddles = self.inv_twiddles.borrow().clone();

        // transpose input
        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(&mat.values, &mut scratch.values, ncols, nrows));

        debug_span!("idft", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_time_dft(&mut scratch.values, nrows, &inv_twiddles));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = nrows)
            .in_scope(|| transpose::transpose(&scratch.values, &mut mat.values, nrows, ncols));

        let inv_len = MontyField31::from_canonical_usize(nrows).inverse();
        debug_span!("scale").in_scope(|| mat.scale(inv_len));
        mat
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<MontyField31<MP>>,
        added_bits: usize,
        shift: MontyField31<MP>,
    ) -> Self::Evaluations {
        let nrows = mat.height();
        let ncols = mat.width();

        let result_nrows = nrows << added_bits;

        let input_size = nrows * ncols;
        let output_size = result_nrows * ncols;

        // TODO: Consider doing this in-place?
        // TODO: Use faster bit-reversal algo
        let mat = debug_span!("bit-reverse input trace")
            .in_scope(|| mat.bit_reverse_rows().to_row_major_matrix());

        // Allocate twice the space of the result, so we can do the final transpose
        // from the second half into the first half.
        //
        // NB: The unsafe version below takes about 10μs, whereas doing
        //   let mut scratch = vec![MontyField31::zero(); 2 * output_size]);
        // takes about 320ms. Safety is expensive.
        let mut scratch =
            debug_span!("allocate scratch space").in_scope(|| Vec::with_capacity(2 * output_size));
        unsafe {
            scratch.set_len(2 * output_size);
        }

        // Split `scratch` into halves `output` and `padded`, each of length `output_size`.
        let (output, padded) = unsafe { split_at_mut_unchecked(&mut scratch, output_size) };

        // `coeffs` will hold the result of the inverse FFT
        let coeffs = &mut output[..input_size];

        // TODO: Ensure that we only calculate twiddle factors once;
        // at the moment we calculate a smaller table first then the
        // bigger table, but if we did the bigger table first we
        // wouldn't need to do the smaller table at all.

        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(&mat.values, coeffs, ncols, nrows));

        // Apply inverse DFT
        self.update_inv_twiddles(nrows);
        let inv_twiddles = self.inv_twiddles.borrow().clone();
        debug_span!("inverse dft batch", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_time_dft(coeffs, nrows, &inv_twiddles));

        // At this point the inverse FFT of each column of `mat` appears
        // as a row in `coeffs`.

        // TODO: consider integrating coset shift into twiddles?
        let inv_len = MontyField31::from_canonical_usize(nrows).inverse();
        coset_shift_rows(coeffs, nrows, shift, inv_len);

        // Extend coeffs by a suitable number of zeros
        padded.fill(MontyField31::zero());
        zero_pad_rows(padded, coeffs, ncols, nrows, added_bits);

        // Apply DFT
        self.update_twiddles(result_nrows);
        let twiddles = self.twiddles.borrow().clone();
        debug_span!("dft batch", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_freq_dft(padded, result_nrows, &twiddles));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = result_nrows)
            .in_scope(|| transpose::transpose(padded, output, result_nrows, ncols));

        // The first half of `scratch` corresponds to `output`.
        // NB: truncating here probably leaves the second half of the vector (being the
        // size of the output) still allocated as "capacity"; this will never be used
        // which is somewhat wasteful.
        scratch.truncate(output_size);
        RowMajorMatrix::new(scratch, ncols).bit_reverse_rows()
    }
}
