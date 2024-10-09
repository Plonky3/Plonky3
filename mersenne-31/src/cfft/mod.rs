//! An implementation of the FFT for `Mersenne31`
extern crate alloc;

use alloc::vec::Vec;
use core::cell::RefCell;

use itertools::izip;
use p3_circle::{compute_twiddles_no_bit_rev, CircleDomain};
use p3_dft::divide_by_height;
use p3_field::{batch_multiplicative_inverse, AbstractField, Field, PackedField, PackedValue};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{debug_span, instrument};

mod backward;
mod forward;

use crate::Mersenne31;

/// Multiply each element of column `j` of `mat` by `shift**j`.
#[instrument(level = "debug", skip_all)]
fn coset_shift_and_scale_rows<F: Field, PF: PackedField<Scalar = F>>(
    out: &mut [PF],
    out_ncols: usize,
    mat: &[PF],
    ncols: usize,
    shift: F,
    scale: F,
) {
    let powers = shift.shifted_powers(scale).take(ncols).collect::<Vec<_>>();
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
pub struct RecursiveCfftMersenne31 {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    y_twiddles: RefCell<Vec<Vec<Mersenne31>>>,
    y_inv_twiddles: RefCell<Vec<Vec<Mersenne31>>>,
    x_twiddles: RefCell<Vec<Vec<Mersenne31>>>,
    x_inv_twiddles: RefCell<Vec<Vec<Mersenne31>>>,
}

impl RecursiveCfftMersenne31 {
    pub fn new(n: usize) -> Self {
        let res = Self {
            y_twiddles: RefCell::default(),
            y_inv_twiddles: RefCell::default(),
            x_twiddles: RefCell::default(),
            x_inv_twiddles: RefCell::default(),
        };
        res.update_twiddles(n);
        res
    }

    #[inline]
    fn decimation_in_freq_dft(&self, mat: &mut [<Mersenne31 as Field>::Packing], ncols: usize) {
        if ncols > 1 {
            let lg_fft_len = p3_util::log2_ceil_usize(ncols);
            let twiddles = self.x_inv_twiddles.borrow();
            let roots_idx = twiddles.len() + 1 - lg_fft_len;
            let twiddles = &twiddles[roots_idx..];
            let y_inv_twiddle = &self.y_inv_twiddles.borrow()[roots_idx];

            mat.par_chunks_exact_mut(ncols).for_each(|v| {
                Mersenne31::backward_pass(v, y_inv_twiddle);

                let n = v.len();
                assert!(n > 1);

                let (v0, v1) = unsafe { v.split_at_mut_unchecked(n / 2) };
                Mersenne31::backward_fft(v0, twiddles);
                Mersenne31::backward_fft(v1, twiddles);
            })
        }
    }

    #[inline]
    fn decimation_in_time_dft(&self, mat: &mut [<Mersenne31 as Field>::Packing], ncols: usize) {
        if ncols > 1 {
            let lg_fft_len = p3_util::log2_ceil_usize(ncols);
            let twiddles = self.x_twiddles.borrow();
            let roots_idx = twiddles.len() + 1 - lg_fft_len;
            let twiddles = &twiddles[roots_idx..];
            let y_twiddle = &self.y_twiddles.borrow()[roots_idx];

            mat.par_chunks_exact_mut(ncols).for_each(|v| {
                let n = v.len();
                assert!(n > 1);

                let (v0, v1) = unsafe { v.split_at_mut_unchecked(n / 2) };
                Mersenne31::forward_fft(v0, twiddles);
                Mersenne31::forward_fft(v1, twiddles);

                Mersenne31::forward_pass(v, y_twiddle);
            })
        }
    }

    /// Compute twiddle factors, or take memoized ones if already available.
    #[instrument(skip_all)]
    fn update_twiddles(&self, fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.

        // Need to do something different for the CFFT due to the y twiddles being different.

        // As we don't save the twiddles for the final layer where
        // the only twiddle is 1, roots_of_unity_table(fft_len)
        // returns a vector of twiddles of length log_2(fft_len) - 1.

        let curr_max_fft_len = 1 << (1 + self.x_twiddles.borrow().len());
        if fft_len > curr_max_fft_len {
            let log_n = log2_strict_usize(fft_len);
            let (new_x_twiddles, new_y_twiddles): (Vec<_>, Vec<_>) =
                compute_twiddles_no_bit_rev(CircleDomain::standard(log_n));
            // We can obtain the inverse twiddles by inverting the twiddles.
            let new_x_inv_twiddles = new_x_twiddles
                .iter()
                .map(|ts| batch_multiplicative_inverse(ts))
                .collect();

            let new_y_inv_twiddles = new_y_twiddles
                .iter()
                .map(|ts| batch_multiplicative_inverse(ts))
                .collect();

            self.x_twiddles.replace(new_x_twiddles);
            self.x_inv_twiddles.replace(new_x_inv_twiddles);
            self.y_twiddles.replace(new_y_twiddles);
            self.y_inv_twiddles.replace(new_y_inv_twiddles);
        }
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
impl RecursiveCfftMersenne31 {
    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    pub fn to_coefficients_dft_batch(
        &self,
        mat: RowMajorMatrix<Mersenne31>,
    ) -> RowMajorMatrix<Mersenne31> {
        let nrows = mat.height();
        let ncols = mat.width();
        assert_eq!(ncols % <Mersenne31 as Field>::Packing::WIDTH, 0);
        let ncols = mat.width() / <Mersenne31 as Field>::Packing::WIDTH;

        let mut mat = mat.bit_reverse_rows().to_row_major_matrix();

        let packedmat = <Mersenne31 as Field>::Packing::pack_slice_mut(&mut mat.values);

        let mut scratch = debug_span!("allocate scratch space")
            .in_scope(|| RowMajorMatrix::default(nrows, ncols));

        self.update_twiddles(nrows);

        // transpose input
        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(packedmat, &mut scratch.values, ncols, nrows));

        debug_span!("dft batch", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_freq_dft(self, &mut scratch.values, nrows));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = nrows)
            .in_scope(|| transpose::transpose(&scratch.values, packedmat, nrows, ncols));

        divide_by_height(&mut mat);

        mat.bit_reverse_rows().to_row_major_matrix()
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    pub fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Mersenne31>,
        added_bits: usize,
    ) -> RowMajorMatrix<Mersenne31> {
        let nrows = mat.height();
        let ncols = mat.width();
        let result_nrows = nrows << added_bits;

        let pack_width = <Mersenne31 as Field>::Packing::WIDTH;
        assert_eq!(ncols % pack_width, 0);

        let input_size = nrows * ncols;
        let output_size = result_nrows * ncols;

        let ncols_packed = ncols / pack_width;

        let mut mat = mat.bit_reverse_rows().to_row_major_matrix();

        self.update_twiddles(result_nrows);

        divide_by_height(&mut mat);

        let packedmat = <Mersenne31 as Field>::Packing::pack_slice(&mat.values);

        // Allocate space for the output and the intermediate state.
        // NB: The unsafe version below takes well under 1ms, whereas doing
        //   vec![Mersenne31::zero(); output_size])
        // takes 100s of ms. Safety is expensive.
        let (mut output, mut padded) = debug_span!("allocate scratch space").in_scope(|| {
            // Safety: These are pretty dodgy, but work because Mersenne31 is #[repr(transparent)]
            let output = Mersenne31::zero_vec(output_size);
            let padded = Mersenne31::zero_vec(output_size);
            (output, padded)
        });

        let packed_output = <Mersenne31 as Field>::Packing::pack_slice_mut(&mut output);
        let packed_padded = <Mersenne31 as Field>::Packing::pack_slice_mut(&mut padded);

        // `coeffs` will hold the result of the inverse FFT; use the
        // output storage as scratch space.
        let coeffs = &mut packed_output[..input_size / pack_width];

        debug_span!("pre-transpose", nrows, ncols)
            .in_scope(|| transpose::transpose(packedmat, coeffs, ncols_packed, nrows));

        // Apply inverse DFT; result is not yet normalised.

        debug_span!("inverse dft batch", n_dfts = ncols, fft_len = nrows)
            .in_scope(|| Self::decimation_in_freq_dft(self, coeffs, nrows));

        // `padded` is implicitly zero padded since it was initialised
        // to zeros when declared above.

        // This is reasonably slow.
        // Replacing it with copy_from_slice gives a 14% speed up. (Though of course that's not valid.)
        debug_span!("Widen Data", n_dfts = ncols, fft_len = nrows).in_scope(|| {
            packed_padded
                .iter_mut()
                .step_by(2)
                .zip(coeffs)
                .for_each(|(val, coeff)| *val = *coeff)
        });

        // Apply DFT
        debug_span!("dft batch", n_dfts = ncols, fft_len = result_nrows)
            .in_scope(|| Self::decimation_in_time_dft(self, packed_padded, result_nrows));

        // transpose output
        debug_span!("post-transpose", nrows = ncols, ncols = result_nrows).in_scope(|| {
            transpose::transpose(packed_padded, packed_output, result_nrows, ncols_packed)
        });

        RowMajorMatrix::new(output, ncols)
            .bit_reverse_rows()
            .to_row_major_matrix()
    }
}

#[cfg(test)]
mod tests {
    use p3_circle::{CircleDomain, CircleEvaluations};
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use rand::thread_rng;

    use super::RecursiveCfftMersenne31;
    use crate::cfft::{backward, forward};
    use crate::Mersenne31;

    type F = Mersenne31;

    #[test]
    fn check_twiddles() {
        for (&twiddle, twiddle_inv) in backward::INV_TWIDDLES_32.iter().zip(forward::_TWIDDLES_32) {
            assert_eq!(F::one(), twiddle * twiddle_inv);
        }

        for (&twiddle, twiddle_inv) in backward::INV_TWIDDLES_16.iter().zip(forward::_TWIDDLES_16) {
            assert_eq!(F::one(), twiddle * twiddle_inv);
        }

        for (&twiddle, twiddle_inv) in backward::INV_TWIDDLES_8.iter().zip(forward::_TWIDDLES_8) {
            assert_eq!(F::one(), twiddle * twiddle_inv);
        }

        for (&twiddle, twiddle_inv) in backward::INV_TWIDDLES_4.iter().zip(forward::_TWIDDLES_4) {
            assert_eq!(F::one(), twiddle * twiddle_inv);
        }
    }

    #[test]
    fn test_to_coeffs() {
        let m = RowMajorMatrix::<F>::rand(&mut thread_rng(), 1 << 8, 1 << 6);
        let rotated = CircleEvaluations::from_natural_order(CircleDomain::standard(8), m);
        let copy = rotated.clone().to_cfft_order().to_row_major_matrix();

        let cfft = RecursiveCfftMersenne31::new(1 << 3);

        let output_circle = rotated.interpolate();
        let output_cfft = cfft.to_coefficients_dft_batch(copy);

        assert_eq!(output_circle.values, output_cfft.values)
    }

    #[test]
    fn test_cfft() {
        let m = RowMajorMatrix::<F>::rand(&mut thread_rng(), 1 << 8, 1 << 6);
        let rotated = CircleEvaluations::from_natural_order(CircleDomain::standard(8), m);
        let copy = rotated.clone().to_cfft_order().to_row_major_matrix();

        let cfft = RecursiveCfftMersenne31::new(2);

        let output_circle = rotated.extrapolate(CircleDomain::standard(8 + 1));
        let output_cfft = cfft.coset_lde_batch(copy, 1);

        assert_eq!(
            output_circle.to_cfft_order().to_row_major_matrix().values,
            output_cfft.values
        )
    }
}
