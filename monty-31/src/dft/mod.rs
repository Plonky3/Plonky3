//! An implementation of the FFT for `MontyField31`
extern crate alloc;

use alloc::vec::Vec;
use core::cell::RefCell;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, Field, PackedValue};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{reverse_slice_index_bits, split_at_mut_unchecked};
use tracing::{debug_span, instrument};

mod backward;
mod forward;

use crate::{FieldParameters, MontyField31, MontyParameters, TwoAdicData};

#[inline]
fn scale<T>(vec: &mut [T], scale: T)
where
    T: Field,
{
    let (packed, sfx) = T::Packing::pack_slice_with_suffix_mut(vec);
    let packed_scale: T::Packing = scale.into();
    packed.par_iter_mut().for_each(|x| *x *= packed_scale);
    sfx.iter_mut().for_each(|x| *x *= scale);
}

#[instrument(level = "debug", skip_all)]
fn zero_pad_bit_reversed_rows<T>(
    output: &mut [T],
    input: &[T],
    nrows: usize,
    ncols: usize,
    added_bits: usize,
) where
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
            padded_row
                .chunks_exact_mut(1 << added_bits)
                .zip(row)
                .for_each(|(chunk, &r)| {
                    chunk[0] = r;
                });
        });
}

/// Multiply each element of column `j` of `mat` by `shift**j`.
///
/// TODO: This might be quite slow
#[instrument(level = "debug", skip_all)]
fn coset_shift_rows<F: Field>(mat: &mut [F], ncols: usize, shift: F) {
    let mut powers = shift.powers().take(ncols).collect::<Vec<_>>();
    reverse_slice_index_bits(&mut powers);

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
        row_len: usize,
        twiddles: &Vec<Vec<MontyField31<MP>>>,
    ) {
        let lg_fft_len = p3_util::log2_ceil_usize(row_len);
        let roots_idx = (twiddles.len() + 1) - lg_fft_len;
        let twiddles = &twiddles[roots_idx..];

        mat.par_chunks_exact_mut(row_len)
            .for_each(|v| MontyField31::forward_fft(v, twiddles))
    }

    #[inline]
    fn decimation_in_time_dft(
        mat: &mut [MontyField31<MP>],
        row_len: usize,
        twiddles: &Vec<Vec<MontyField31<MP>>>,
    ) {
        let lg_fft_len = p3_util::log2_ceil_usize(row_len);
        let roots_idx = (twiddles.len() + 1) - lg_fft_len;
        let twiddles = &twiddles[roots_idx..];

        mat.par_chunks_exact_mut(row_len)
            .for_each(|v| MontyField31::backward_fft(v, twiddles))
    }

    #[instrument(skip_all, fields(nrows = %mat.len()/row_len, row_len, added_bits))]
    #[inline]
    pub fn dft_batch_rows(&self, mat: &mut [MontyField31<MP>], row_len: usize)
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        // Compute twiddle factors, or take memoized ones if already available.
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        debug_span!("maybe calculate twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.twiddles.borrow().len();
            if row_len > curr_max_fft_len {
                let new_twiddles = MontyField31::roots_of_unity_table(row_len);
                self.twiddles.replace(new_twiddles);
            }
        });

        // TODO: We're only cloning because of the RefCell; it
        // shouldn't be necessary, though it only costs ~20μs.
        let twiddles = debug_span!("clone twiddles").in_scope(|| self.twiddles.borrow().clone());

        debug_span!("parallel dft") //, n_dfts = mat.height(), lengths = mat.width())
            .in_scope(|| Self::decimation_in_time_dft(mat, row_len, &twiddles));
    }

    #[instrument(skip_all, fields(nrows = %mat.len()/ncols, ncols, added_bits))]
    #[inline]
    pub fn idft_batch_rows(&self, mat: &mut [MontyField31<MP>], ncols: usize)
    where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        // Compute twiddle factors, or take memoized ones if already available.
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.
        debug_span!("maybe calculate inv_twiddles").in_scope(|| {
            let curr_max_fft_len = 1 << self.inv_twiddles.borrow().len();
            if ncols > curr_max_fft_len {
                let new_twiddles = MontyField31::inv_roots_of_unity_table(ncols);
                self.inv_twiddles.replace(new_twiddles);
            }
        });

        // TODO: We're only cloning because of the RefCell; it
        // shouldn't be necessary, though it only costs ~20μs.
        let inv_twiddles =
            debug_span!("clone inv_twiddles").in_scope(|| self.inv_twiddles.borrow().clone());

        debug_span!("parallel idft", n_dfts = mat.len() / ncols, fft_len = ncols)
            .in_scope(|| Self::decimation_in_freq_dft(mat, ncols, &inv_twiddles));

        let inv_len = MontyField31::from_canonical_usize(ncols).inverse();
        // TODO: mat.scale() is not parallelised...
        debug_span!("scale").in_scope(|| scale(mat, inv_len));
    }

    pub fn idft_batch_cols_transposed_bitrevd(
        &self,
        mat: &[MontyField31<MP>],
        ncols: usize,
        out: &mut [MontyField31<MP>],
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let nrows = mat.len() / ncols;
        if nrows <= 1 {
            out.copy_from_slice(mat);
            return;
        }

        // transpose input
        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(mat, out, ncols, nrows));

        self.idft_batch_rows(out, nrows);
    }

    pub fn idft_batch_cols_bitrevd(
        &self,
        mat: &mut [MontyField31<MP>],
        ncols: usize,
        out: &mut [MontyField31<MP>],
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        self.idft_batch_cols_transposed_bitrevd(mat, ncols, out);

        let nrows = mat.len() / ncols;

        // transpose output
        debug_span!("post-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(out, mat, nrows, ncols));
    }

    pub fn dft_batch_cols_transposed_bitrevd(
        &self,
        mat: &[MontyField31<MP>],
        ncols: usize,
        out: &mut [MontyField31<MP>],
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        let nrows = mat.len() / ncols;
        if nrows <= 1 {
            out.copy_from_slice(mat);
            return;
        }

        // transpose input
        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(mat, out, ncols, nrows));

        self.dft_batch_rows(out, nrows);
    }

    pub fn dft_batch_cols_bitrevd(
        &self,
        mat: &mut [MontyField31<MP>],
        ncols: usize,
        out: &mut [MontyField31<MP>],
    ) where
        MP: MontyParameters + FieldParameters + TwoAdicData,
    {
        self.dft_batch_cols_transposed_bitrevd(mat, ncols, out);

        let nrows = mat.len() / ncols;

        // transpose output
        debug_span!("post-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(out, mat, nrows, ncols));
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

        let ncols = mat.width();
        debug_span!("dft batch")
            .in_scope(|| self.dft_batch_cols_bitrevd(&mut mat.values, ncols, &mut scratch.values));
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

        let ncols = mat.width();
        debug_span!("idft batch")
            .in_scope(|| self.idft_batch_cols_bitrevd(&mut mat.values, ncols, &mut scratch.values));

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

        let input_size = mat.height() * mat.width();
        let output_size = result_height * mat.width();

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
        let (output, mut padded) = unsafe { split_at_mut_unchecked(&mut scratch, output_size) };

        // `coeffs` will hold the result of the inverse FFT
        let mut coeffs = &mut output[..input_size];

        // TODO: Ensure that we only calculate twiddle factors once;
        // at the moment we calculate a smaller table first then the
        // bigger table, but if we did the bigger table first we
        // wouldn't need to do the smaller table at all.

        // Apply inverse DFT
        self.idft_batch_cols_transposed_bitrevd(&mat.values, mat.width(), &mut coeffs);

        // At this point the inverse FFT of each column of `mat` appears
        // as a row in `coeffs`.

        // TODO: consider integrating coset shift into twiddles?
        coset_shift_rows(coeffs, mat.height(), shift);

        // Extend coeffs by a suitable number of zeros
        zero_pad_bit_reversed_rows(padded, coeffs, mat.width(), mat.height(), added_bits);

        // Apply DFT
        self.dft_batch_rows(&mut padded, result_height);

        // transpose output
        debug_span!("post-transpose", nrows = mat.width(), ncols = result_height)
            .in_scope(|| transpose::transpose(padded, output, result_height, mat.width()));

        // The first half of `scratch` corresponds to `output`.
        // NB: truncating here probably leaves the second half of the vector (being the
        // size of the output) still allocated as "capacity"; this will never be used
        // which is somewhat wasteful.
        scratch.truncate(output_size);
        RowMajorMatrix::new(scratch, mat.width())
    }
}
