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
#[repr(transparent)] // Allows safe transmutes from F to this.
pub struct DifButterfly<F>(pub F);

impl<F: Field> Butterfly<F> for DifButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        (x_1 + x_2, (x_1 - x_2) * self.0)
    }
}

/// DIF (Decimation-In-Frequency) butterfly operation where `x_2` is guaranteed to be zero.
///
/// Useful in scenarios where the input has just been padded with zeros.
///
/// Used in the *output-ordering* variant of NTT.
/// This butterfly computes:
/// ```text
///   output_1 = x1
///   output_2 = x1 * twiddle
/// ```
#[derive(Copy, Clone)]
#[repr(transparent)] // Allows safe transmutes from F to this.
pub struct DifButterflyZeros<F>(pub F);

impl<F: Field> Butterfly<F> for DifButterflyZeros<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        debug_assert!(x_2.as_slice().iter().all(|x| x.is_zero())); // Slightly convoluted but PF may not implement equality.
        (x_1, x_1 * self.0)
    }

    #[inline]
    fn apply_to_rows(&self, row_1: &mut [F], row_2: &mut [F]) {
        let (shorts_1, suffix_1) = F::Packing::pack_slice_with_suffix(row_1);
        let (shorts_2, suffix_2) = F::Packing::pack_slice_with_suffix_mut(row_2);
        debug_assert_eq!(shorts_1.len(), shorts_2.len());
        debug_assert_eq!(suffix_1.len(), suffix_2.len());
        for (x_1, x_2) in shorts_1.iter().zip(shorts_2) {
            debug_assert!(x_2.as_slice().iter().all(|x| x.is_zero())); // Slightly convoluted but PF may not implement equality.
            *x_2 = *x_1 * self.0; // x_2 is guaranteed to be zero, so we just set it to x_1 * twiddle. 
        }
        for (x_1, x_2) in suffix_1.iter().zip(suffix_2) {
            debug_assert!(x_2.is_zero());
            *x_2 = *x_1 * self.0; // x_2 is guaranteed to be zero, so we just set it to x_1 * twiddle. 
        }
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
#[repr(transparent)] // Allows safe transmutes from F to this.
pub struct DitButterfly<F>(pub F);

impl<F: Field> Butterfly<F> for DitButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        let x_2_twiddle = x_2 * self.0;
        (x_1 + x_2_twiddle, x_1 - x_2_twiddle)
    }

    /// Override `apply_to_rows` to pre-broadcast the twiddle factor into a packed field
    /// once before the inner loop, avoiding a scalar-to-vector broadcast on each packed
    /// multiplication. For wide rows (e.g., 256 columns with AVX512 width=16, giving 16
    /// packed iterations per row-pair), this eliminates 15 redundant broadcasts per call.
    /// Manually unroll the inner packed loop to expose multiple independent mul chains
    /// to the compiler's scheduler, hiding the ~12–15 cyc Montgomery mul latency.
    #[inline]
    fn apply_to_rows(&self, row_1: &mut [F], row_2: &mut [F]) {
        let (shorts_1, suffix_1) = F::Packing::pack_slice_with_suffix_mut(row_1);
        let (shorts_2, suffix_2) = F::Packing::pack_slice_with_suffix_mut(row_2);
        debug_assert_eq!(shorts_1.len(), shorts_2.len());
        debug_assert_eq!(suffix_1.len(), suffix_2.len());
        let twiddle_packed = F::Packing::from(self.0);
        let mut c1 = shorts_1.chunks_exact_mut(4);
        let mut c2 = shorts_2.chunks_exact_mut(4);
        for (p1, p2) in (&mut c1).zip(&mut c2) {
            let a1 = p1[0];
            let b1 = p1[1];
            let c1_ = p1[2];
            let d1 = p1[3];
            let a2 = p2[0];
            let b2 = p2[1];
            let c2_ = p2[2];
            let d2 = p2[3];
            let a2t = a2 * twiddle_packed;
            let b2t = b2 * twiddle_packed;
            let c2t = c2_ * twiddle_packed;
            let d2t = d2 * twiddle_packed;
            p1[0] = a1 + a2t;
            p2[0] = a1 - a2t;
            p1[1] = b1 + b2t;
            p2[1] = b1 - b2t;
            p1[2] = c1_ + c2t;
            p2[2] = c1_ - c2t;
            p1[3] = d1 + d2t;
            p2[3] = d1 - d2t;
        }
        for (x_1, x_2) in c1
            .into_remainder()
            .iter_mut()
            .zip(c2.into_remainder().iter_mut())
        {
            let x_2_twiddle = *x_2 * twiddle_packed;
            let new_x1 = *x_1 + x_2_twiddle;
            *x_2 = *x_1 - x_2_twiddle;
            *x_1 = new_x1;
        }
        for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2.iter_mut()) {
            self.apply_in_place(x_1, x_2);
        }
    }

    /// Out-of-place variant with matching unroll factor.
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
        let twiddle_packed = F::Packing::from(self.0);
        let n = src_shorts_1.len();
        let n4 = n - (n & 3);
        let mut i = 0;
        while i < n4 {
            let a1 = src_shorts_1[i];
            let b1 = src_shorts_1[i + 1];
            let c1 = src_shorts_1[i + 2];
            let d1 = src_shorts_1[i + 3];
            let a2 = src_shorts_2[i];
            let b2 = src_shorts_2[i + 1];
            let c2 = src_shorts_2[i + 2];
            let d2 = src_shorts_2[i + 3];
            let a2t = a2 * twiddle_packed;
            let b2t = b2 * twiddle_packed;
            let c2t = c2 * twiddle_packed;
            let d2t = d2 * twiddle_packed;
            dst_shorts_1[i].write(a1 + a2t);
            dst_shorts_2[i].write(a1 - a2t);
            dst_shorts_1[i + 1].write(b1 + b2t);
            dst_shorts_2[i + 1].write(b1 - b2t);
            dst_shorts_1[i + 2].write(c1 + c2t);
            dst_shorts_2[i + 2].write(c1 - c2t);
            dst_shorts_1[i + 3].write(d1 + d2t);
            dst_shorts_2[i + 3].write(d1 - d2t);
            i += 4;
        }
        while i < n {
            let s1 = src_shorts_1[i];
            let s2 = src_shorts_2[i];
            let x_2_twiddle = s2 * twiddle_packed;
            dst_shorts_1[i].write(s1 + x_2_twiddle);
            dst_shorts_2[i].write(s1 - x_2_twiddle);
            i += 1;
        }
        for (s_1, s_2, d_1, d_2) in izip!(src_suffix_1, src_suffix_2, dst_suffix_1, dst_suffix_2) {
            let (res_1, res_2) = self.apply(*s_1, *s_2);
            d_1.write(res_1);
            d_2.write(res_2);
        }
    }
}

/// DIT (Decimation-In-Time) butterfly operation with a post-multiplication scale factor.
///
/// This butterfly computes:
/// ```text
///   output_1 = (x1 + x2 * twiddle) * scale
///   output_2 = (x1 - x2 * twiddle) * scale
/// ```
/// which is equivalent to:
/// ```text
///   output_1 = x1 * scale + x2 * (twiddle * scale)
///   output_2 = x1 * scale - x2 * (twiddle * scale)
/// ```
///
/// This is used to merge a uniform scaling step (e.g., 1/N normalization in inverse DFT)
/// into a butterfly pass, avoiding a separate memory pass over the data.
///
/// The struct stores `scale` and `twiddle_times_scale = twiddle * scale` so that the
/// `apply` method only needs 2 multiplications instead of 3.
#[derive(Copy, Clone)]
pub struct ScaledDitButterfly<F> {
    pub twiddle: F,
    pub scale: F,
    /// Precomputed product `twiddle * scale` to reduce multiplications in the hot loop.
    pub twiddle_times_scale: F,
}

impl<F: Field> ScaledDitButterfly<F> {
    /// Construct a `ScaledDitButterfly`, precomputing `twiddle * scale`.
    #[inline]
    pub fn new(twiddle: F, scale: F) -> Self {
        Self {
            twiddle,
            scale,
            twiddle_times_scale: twiddle * scale,
        }
    }
}

impl<F: Field> Butterfly<F> for ScaledDitButterfly<F> {
    #[inline]
    fn apply<PF: PackedField<Scalar = F>>(&self, x_1: PF, x_2: PF) -> (PF, PF) {
        // 2 multiplications instead of 3:
        //   x1_s   = x1 * scale
        //   x2_ts  = x2 * (twiddle * scale)   [precomputed]
        //   out1   = x1_s + x2_ts
        //   out2   = x1_s - x2_ts
        let x_1_scale = x_1 * self.scale;
        let x_2_twiddle_scale = x_2 * self.twiddle_times_scale;
        (x_1_scale + x_2_twiddle_scale, x_1_scale - x_2_twiddle_scale)
    }

    /// Override `apply_to_rows` to pre-broadcast both `scale` and `twiddle_times_scale`
    /// into packed fields once before the inner loop.
    #[inline]
    fn apply_to_rows(&self, row_1: &mut [F], row_2: &mut [F]) {
        let (shorts_1, suffix_1) = F::Packing::pack_slice_with_suffix_mut(row_1);
        let (shorts_2, suffix_2) = F::Packing::pack_slice_with_suffix_mut(row_2);
        debug_assert_eq!(shorts_1.len(), shorts_2.len());
        debug_assert_eq!(suffix_1.len(), suffix_2.len());
        let scale_packed = F::Packing::from(self.scale);
        let twiddle_times_scale_packed = F::Packing::from(self.twiddle_times_scale);
        // ScaledDitButterfly has 2 muls per butterfly (scale + twiddle_scale), so unroll-4
        // exposes 8 independent mul chains — better ILP than unroll-2's 4 chains.
        let mut c1 = shorts_1.chunks_exact_mut(4);
        let mut c2 = shorts_2.chunks_exact_mut(4);
        for (p1, p2) in (&mut c1).zip(&mut c2) {
            let a1 = p1[0];
            let b1 = p1[1];
            let c1_ = p1[2];
            let d1 = p1[3];
            let a2 = p2[0];
            let b2 = p2[1];
            let c2_ = p2[2];
            let d2 = p2[3];
            let a1s = a1 * scale_packed;
            let b1s = b1 * scale_packed;
            let c1s = c1_ * scale_packed;
            let d1s = d1 * scale_packed;
            let a2t = a2 * twiddle_times_scale_packed;
            let b2t = b2 * twiddle_times_scale_packed;
            let c2t = c2_ * twiddle_times_scale_packed;
            let d2t = d2 * twiddle_times_scale_packed;
            p1[0] = a1s + a2t;
            p2[0] = a1s - a2t;
            p1[1] = b1s + b2t;
            p2[1] = b1s - b2t;
            p1[2] = c1s + c2t;
            p2[2] = c1s - c2t;
            p1[3] = d1s + d2t;
            p2[3] = d1s - d2t;
        }
        for (x_1, x_2) in c1
            .into_remainder()
            .iter_mut()
            .zip(c2.into_remainder().iter_mut())
        {
            let x_1_scale = *x_1 * scale_packed;
            let x_2_twiddle_scale = *x_2 * twiddle_times_scale_packed;
            *x_1 = x_1_scale + x_2_twiddle_scale;
            *x_2 = x_1_scale - x_2_twiddle_scale;
        }
        for (x_1, x_2) in suffix_1.iter_mut().zip(suffix_2.iter_mut()) {
            self.apply_in_place(x_1, x_2);
        }
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
