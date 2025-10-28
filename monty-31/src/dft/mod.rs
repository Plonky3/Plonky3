//! An implementation of the FFT for `MontyField31`
extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;

use itertools::izip;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use spin::RwLock;
use tracing::{debug_span, instrument};

mod backward;
mod forward;

use crate::{FieldParameters, MontyField31, MontyParameters, TwoAdicData};

/// Multiply each element of column `j` of `mat` by `shift**j`.
#[instrument(level = "debug", skip_all)]
fn coset_shift_and_scale_rows<F: Field>(
    out: &mut [F],
    out_ncols: usize,
    mat: &[F],
    ncols: usize,
    shift: F,
    scale: F,
) {
    let powers = shift.shifted_powers(scale).collect_n(ncols);
    out.par_chunks_exact_mut(out_ncols)
        .zip(mat.par_chunks_exact(ncols))
        .for_each(|(out_row, in_row)| {
            izip!(out_row.iter_mut(), in_row, &powers).for_each(|(out, &coeff, &weight)| {
                *out = coeff * weight;
            });
        });
}

/// Recursive DFT, decimation-in-frequency in the forward direction,
/// decimation-in-time in the backward (inverse) direction.
#[derive(Clone, Debug, Default)]
pub struct RecursiveDft<F> {
    /// Forward twiddle tables
    #[allow(clippy::type_complexity)]
    twiddles: Arc<RwLock<Arc<[Vec<F>]>>>,
    /// Inverse twiddle tables
    #[allow(clippy::type_complexity)]
    inv_twiddles: Arc<RwLock<Arc<[Vec<F>]>>>,
}

impl<MP: FieldParameters + TwoAdicData> RecursiveDft<MontyField31<MP>> {
    pub fn new(n: usize) -> Self {
        let res = Self {
            twiddles: Arc::default(),
            inv_twiddles: Arc::default(),
        };
        res.update_twiddles(n);
        res
    }

    #[inline]
    fn decimation_in_freq_dft(
        mat: &mut [MontyField31<MP>],
        ncols: usize,
        twiddles: &[Vec<MontyField31<MP>>],
    ) {
        if ncols > 1 {
            let lg_fft_len = log2_strict_usize(ncols);
            let twiddles = &twiddles[..(lg_fft_len - 1)];

            mat.par_chunks_exact_mut(ncols)
                .for_each(|v| MontyField31::forward_fft(v, twiddles));
        }
    }

    #[inline]
    fn decimation_in_time_dft(
        mat: &mut [MontyField31<MP>],
        ncols: usize,
        twiddles: &[Vec<MontyField31<MP>>],
    ) {
        if ncols > 1 {
            let lg_fft_len = p3_util::log2_strict_usize(ncols);
            let twiddles = &twiddles[..(lg_fft_len - 1)];

            mat.par_chunks_exact_mut(ncols)
                .for_each(|v| MontyField31::backward_fft(v, twiddles));
        }
    }

    /// Compute twiddle factors, or take memoized ones if already available.
    #[instrument(skip_all)]
    fn update_twiddles(&self, fft_len: usize) {
        // As we don't save the twiddles for the final layer where
        // the only twiddle is 1, roots_of_unity_table(fft_len)
        // returns a vector of twiddles of length log_2(fft_len) - 1.
        // let curr_max_fft_len = 2 << self.twiddles.read().len();
        let need = log2_strict_usize(fft_len);
        let snapshot = self.twiddles.read().clone();
        let have = snapshot.len() + 1;
        if have >= need {
            return;
        }

        let missing_twiddles = MontyField31::get_missing_twiddles(need, have);

        let missing_inv_twiddles = missing_twiddles
            .iter()
            .map(|ts| {
                core::iter::once(MontyField31::ONE)
                    .chain(
                        ts[1..]
                            .iter()
                            .rev()
                            .map(|&t| MontyField31::new_monty(MP::PRIME - t.value)),
                    )
                    .collect()
            })
            .collect::<Vec<_>>();
        // Helper closure to extend a table under its lock.
        let extend_table = |lock: &RwLock<Arc<[Vec<_>]>>, missing: &[Vec<_>]| {
            let mut w = lock.write();
            let current_len = w.len();
            // Double-check if an update is still needed after acquiring the write lock.
            if (current_len + 1) < need {
                let mut v = w.to_vec();
                // Append only the portion needed in case another thread did a partial update.
                let extend_from = current_len.saturating_sub(current_len);
                v.extend_from_slice(&missing[extend_from..]);
                *w = v.into();
            }
        };
        // Atomically update each table. This two-step process is the source of the race condition.
        extend_table(&self.twiddles, &missing_twiddles);
        extend_table(&self.inv_twiddles, &missing_inv_twiddles);
    }

    fn get_twiddles(&self) -> Arc<[Vec<MontyField31<MP>>]> {
        self.twiddles.read().clone()
    }

    fn get_inv_twiddles(&self) -> Arc<[Vec<MontyField31<MP>>]> {
        self.inv_twiddles.read().clone()
    }
}

/// DFT implementation that uses DIT for the inverse "backward"
/// direction and DIF for the "forward" direction.
///
/// The API mandates that the LDE is applied column-wise on the
/// _row-major_ input. This is awkward for memory coherence, so the
/// algorithm here transposes the input and operates on the rows in
/// the typical way, then transposes back again for the output. Even
/// for modestly large inputs, the cost of the two transposes
/// outweighed by the improved performance from operating row-wise.
///
/// The choice of DIT for inverse and DIF for "forward" transform mean
/// that a (coset) LDE
///
/// - IDFT / zero extend / DFT
///
/// expands to
///
///   - bit-reverse input
///   - invDFT DIT
///     - result is in "correct" order
///   - coset shift and zero extend result
///   - DFT DIF on result
///     - output is bit-reversed, as required for FRI.
///
/// Hence the only bit-reversal that needs to take place is on the input.
///
impl<MP: MontyParameters + FieldParameters + TwoAdicData> TwoAdicSubgroupDft<MontyField31<MP>>
    for RecursiveDft<MontyField31<MP>>
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
        let twiddles = self.get_twiddles();

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

        let mut mat =
            debug_span!("initial bitrev").in_scope(|| mat.bit_reverse_rows().to_row_major_matrix());

        self.update_twiddles(nrows);
        let inv_twiddles = self.get_inv_twiddles();

        // transpose input
        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(&mat.values, &mut scratch.values, ncols, nrows));

        debug_span!("idft", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_time_dft(&mut scratch.values, nrows, &inv_twiddles));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = nrows)
            .in_scope(|| transpose::transpose(&scratch.values, &mut mat.values, nrows, ncols));

        let log_rows = log2_ceil_usize(nrows);
        let inv_len = MontyField31::ONE.div_2exp_u64(log_rows as u64);
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

        if nrows == 1 {
            let dupd_rows = core::iter::repeat_n(mat.values, result_nrows)
                .flatten()
                .collect();
            return RowMajorMatrix::new(dupd_rows, ncols).bit_reverse_rows();
        }

        let input_size = nrows * ncols;
        let output_size = result_nrows * ncols;

        let mat = mat.bit_reverse_rows().to_row_major_matrix();

        // Allocate space for the output and the intermediate state.
        let (mut output, mut padded) = debug_span!("allocate scratch space").in_scope(|| {
            // Safety: These are pretty dodgy, but work because MontyField31 is #[repr(transparent)]
            let output = MontyField31::<MP>::zero_vec(output_size);
            let padded = MontyField31::<MP>::zero_vec(output_size);
            (output, padded)
        });

        // `coeffs` will hold the result of the inverse FFT; use the
        // output storage as scratch space.
        let coeffs = &mut output[..input_size];

        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(&mat.values, coeffs, ncols, nrows));

        // Apply inverse DFT; result is not yet normalised.
        self.update_twiddles(result_nrows);
        let inv_twiddles = self.get_inv_twiddles();
        debug_span!("inverse dft batch", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_time_dft(coeffs, nrows, &inv_twiddles));

        // At this point the inverse FFT of each column of `mat` appears
        // as a row in `coeffs`.

        // Normalise inverse DFT and coset shift in one go.
        let log_rows = log2_ceil_usize(nrows);
        let inv_len = MontyField31::ONE.div_2exp_u64(log_rows as u64);
        coset_shift_and_scale_rows(&mut padded, result_nrows, coeffs, nrows, shift, inv_len);

        // `padded` is implicitly zero padded since it was initialised
        // to zeros when declared above.

        let twiddles = self.get_twiddles();

        // Apply DFT
        debug_span!("dft batch", n_dfts = ncols, fft_len = result_nrows)
            .in_scope(|| Self::decimation_in_freq_dft(&mut padded, result_nrows, &twiddles));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = result_nrows)
            .in_scope(|| transpose::transpose(&padded, &mut output, result_nrows, ncols));

        RowMajorMatrix::new(output, ncols).bit_reverse_rows()
    }
}
