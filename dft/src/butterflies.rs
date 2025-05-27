use core::mem::MaybeUninit;

use itertools::izip;
use p3_field::{Field, PackedField, PackedValue};

/// A butterfly operation used in NTT to combine two values into a new pair.
///
/// This trait defines how to transform two elements (or vectors of elements)
/// according to the structure of a butterfly gate.
///
/// In an NTT, butterflies are the core units that recursively combine values
/// across layers. Each butterfly computes:
/// ```text
///   (a + b * twiddle, a - b * twiddle)   // DIT
/// or
///   (a + b, (a - b) * twiddle)           // DIF
/// ```
/// The transformation can be applied:
/// - in-place (mutating input values)
/// - to full rows of values (arrays of field elements)
/// - out-of-place (writing results to separate destination buffers)
///
/// Different butterfly variants (DIT, DIF, or twiddle-free) define the exact formula.
pub trait Butterfly<F: Field>: Copy + Send + Sync {
    /// Applies the butterfly transformation to two packed field values.
    ///
    /// This method takes two inputs `x_1` and `x_2` and returns two outputs `(y_1, y_2)`
    /// depending on the butterfly type.
    /// ```text
    /// Example (DIF):
    ///   Input:  x_1 = a, x_2 = b
    ///   Output: (a + b, (a - b) * twiddle)
    /// ```
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF);

    /// Applies the butterfly in-place to two packed values.
    ///
    /// Mutates both `x_1` and `x_2` directly, storing the result of `apply`.
    #[inline]
    fn apply_in_place<PF: PackedField<Scalar = F>>(&self, x_1: &mut PF, x_2: &mut PF) {
        (*x_1, *x_2) = self.apply(*x_1, *x_2);
    }

    /// Applies the butterfly transformation to two rows of scalar field values.
    ///
    /// Each row is a slice of `F`. This function processes the rows in packed
    /// chunks using SIMD where possible, and falls back to scalar operations
    /// for the suffix (remaining elements).
    ///
    /// The transformation is done in-place.
    #[inline]
    fn apply_to_rows(&self, row_1: &mut [F], row_2: &mut [F]) {
        let (shorts_1, suffix_1) = F::Packing::pack_slice_with_suffix_mut(row_1);
        let (shorts_2, suffix_2) = F::Packing::pack_slice_with_suffix_mut(row_2);
        debug_assert_eq!(shorts_1.len(), shorts_2.len());
        debug_assert_eq!(suffix_1.len(), suffix_2.len());
        for (x_1, x_2) in shorts_1.iter_mut().zip(shorts_2) {
            self.apply_in_place(x_1, x_2);
        }
        for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2) {
            self.apply_in_place(x_1, x_2);
        }
    }

    /// Applies the butterfly out-of-place to two source rows.
    ///
    /// This version does not overwrite the source. Instead, it writes the
    /// result of each butterfly to separate destination slices (which may
    /// be uninitialized memory).
    ///
    /// This is useful when performing LDE's where the size of the output is larger than the size of the input.
    ///
    /// - `src_1`, `src_2`: input slices
    /// - `dst_1`, `dst_2`: output slices to write to (must be MaybeUninit)
    #[inline]
    fn apply_to_rows_oop(
        &self,
        src_1: &[F],
        dst_1: &mut [MaybeUninit<F>],
        src_2: &[F],
        dst_2: &mut [MaybeUninit<F>],
    ) {
        let (src_shorts_1, src_suffix_1) = F::Packing::pack_slice_with_suffix(src_1);
        let (src_shorts_2, src_suffix_2) = F::Packing::pack_slice_with_suffix(src_2);
        let (dst_shorts_1, dst_suffix_1) =
            F::Packing::pack_maybe_uninit_slice_with_suffix_mut(dst_1);
        let (dst_shorts_2, dst_suffix_2) =
            F::Packing::pack_maybe_uninit_slice_with_suffix_mut(dst_2);
        debug_assert_eq!(src_shorts_1.len(), src_shorts_2.len());
        debug_assert_eq!(src_suffix_1.len(), src_suffix_2.len());
        debug_assert_eq!(dst_shorts_1.len(), dst_shorts_2.len());
        debug_assert_eq!(dst_suffix_1.len(), dst_suffix_2.len());
        for (s_1, s_2, d_1, d_2) in izip!(src_shorts_1, src_shorts_2, dst_shorts_1, dst_shorts_2) {
            let (res_1, res_2) = self.apply(*s_1, *s_2);
            d_1.write(res_1);
            d_2.write(res_2);
        }
        for (s_1, s_2, d_1, d_2) in izip!(src_suffix_1, src_suffix_2, dst_suffix_1, dst_suffix_2) {
            let (res_1, res_2) = self.apply(*s_1, *s_2);
            d_1.write(res_1);
            d_2.write(res_2);
        }
    }
}

/// DIF (Decimation-In-Frequency) butterfly operation.
///
/// Used in the *output-ordering* variant of NTT.
/// This butterfly computes:
/// ```text
///   output_1 = x1 + x2
///   output_2 = (x1 - x2) * twiddle
/// ```
/// The twiddle factor is applied after subtraction.
/// Suitable for DIF-style recursive transforms.
#[derive(Copy, Clone)]
pub struct DifButterfly<F>(pub F);

impl<F: Field> Butterfly<F> for DifButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        (x_1 + x_2, (x_1 - x_2) * self.0)
    }
}

/// DIT (Decimation-In-Time) butterfly operation.
///
/// Used in the *input-ordering* variant of NTT/FFT.
/// This butterfly computes:
/// ```text
///   output_1 = x1 + x2 * twiddle
///   output_2 = x1 - x2 * twiddle
/// ```
/// The twiddle factor is applied to x2 before combining.
/// Suitable for DIT-style recursive transforms.
#[derive(Copy, Clone)]
pub struct DitButterfly<F>(pub F);

impl<F: Field> Butterfly<F> for DitButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        let x_2_twiddle = x_2 * self.0;
        (x_1 + x_2_twiddle, x_1 - x_2_twiddle)
    }
}

/// Butterfly with no twiddle factor (`twiddle = 1`).
///
/// This is used when no root-of-unity scaling is needed.
/// It works for either DIT or DIF, and is often used at
/// the final or base level of a transform tree.
///
/// This butterfly computes:
/// ```text
///   - output_1 = x1 + x2
///   - output_2 = x1 - x2
/// ```
#[derive(Copy, Clone)]
pub struct TwiddleFreeButterfly;

impl<F: Field> Butterfly<F> for TwiddleFreeButterfly {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        (x_1 + x_2, x_1 - x_2)
    }
}
