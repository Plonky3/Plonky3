//! A file containing a collection of architecture-specific interleaving functions.
//! Used for PackedFields to implement interleaving operations.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod interleave {
    use core::arch::aarch64::{self, uint32x4_t};

    #[inline]
    #[must_use]
    /// Interleave two vectors of 32-bit integers.
    ///
    /// Maps `[a0, ..., a3], [b0, ..., b3], ` to `[a0, b0, ...], [..., a3, b3]`.
    pub fn interleave_u32(v0: uint32x4_t, v1: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
        // We want this to compile to:
        //      trn1  res0.4s, v0.4s, v1.4s
        //      trn2  res1.4s, v0.4s, v1.4s
        // throughput: .5 cyc/2 vec (16 els/cyc)
        // latency: 2 cyc
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            (aarch64::vtrn1q_u32(v0, v1), aarch64::vtrn2q_u32(v0, v1))
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 64-bit integers.
    ///
    /// Maps `[a0, a1], [b0, b1], ` to `[a0, b0], [a1, b1]`.
    pub fn interleave_u64(v0: uint32x4_t, v1: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
        // We want this to compile to:
        //      trn1  res0.2d, v0.2d, v1.2d
        //      trn2  res1.2d, v0.2d, v1.2d
        // throughput: .5 cyc/2 vec (16 els/cyc)
        // latency: 2 cyc

        // To transpose 64-bit blocks, cast the [u32; 4] vectors to [u64; 2], transpose, and cast back.
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let v0 = aarch64::vreinterpretq_u64_u32(v0);
            let v1 = aarch64::vreinterpretq_u64_u32(v1);
            (
                aarch64::vreinterpretq_u32_u64(aarch64::vtrn1q_u64(v0, v1)),
                aarch64::vreinterpretq_u32_u64(aarch64::vtrn2q_u64(v0, v1)),
            )
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub mod interleave {
    use core::arch::x86_64::{self, __m256i};

    #[inline]
    #[must_use]
    /// Interleave two vectors of 32-bit integers.
    ///
    /// Maps `[a0, ..., a7], [b0, ..., b7], ` to `[a0, b0, ...], [..., a7, b7]`.
    pub fn interleave_u32(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        // We want this to compile to:
        //      vpsllq    t, a, 32
        //      vpsrlq    u, b, 32
        //      vpblendd  res0, a, u, aah
        //      vpblendd  res1, t, b, aah
        // throughput: 1.33 cyc/2 vec (12 els/cyc)
        // latency: (1 -> 1)  1 cyc
        //          (1 -> 2)  2 cyc
        //          (2 -> 1)  2 cyc
        //          (2 -> 2)  1 cyc
        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.

            // We currently have:
            //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
            //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
            // First form
            //   t = [ a1   0  a3   0  a5   0  a7   0 ].
            //   u = [  0  b0   0  b2   0  b4   0  b6 ].
            let t = x86_64::_mm256_srli_epi64::<32>(a);
            let u = x86_64::_mm256_slli_epi64::<32>(b);

            // Then
            //   res0 = [ a0  b0  a2  b2  a4  b4  a6  b6 ],
            //   res1 = [ a1  b1  a3  b3  a5  b5  a7  b7 ].
            (
                x86_64::_mm256_blend_epi32::<0b10101010>(a, u),
                x86_64::_mm256_blend_epi32::<0b10101010>(t, b),
            )
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 64-bit integers.
    ///
    /// Maps `[a0, ..., a3], [b0, ..., b3], ` to `[a0, b0, ...], [..., a3, b3]`.
    pub fn interleave_u64(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        // We want this to compile to:
        //      vpunpcklqdq   res0, a, b
        //      vpunpckhqdq   res1, a, b
        // throughput: 1 cyc/2 vec (16 els/cyc)
        // latency: 1 cyc

        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.
            (
                x86_64::_mm256_unpacklo_epi64(a, b),
                x86_64::_mm256_unpackhi_epi64(a, b),
            )
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 128-bit integers.
    ///
    /// Maps `[a0, a1], [b0, b1], ` to `[a0, b0], [a1, b1]`.
    pub fn interleave_u128(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        // We want this to compile to:
        //      vperm2i128  t, a, b, 21h
        //      vpblendd    res0, a, t, f0h
        //      vpblendd    res1, t, b, f0h
        // throughput: 1 cyc/2 vec (16 els/cyc)
        // latency: 4 cyc

        unsafe {
            // Safety: If this code got compiled then AVX2 intrinsics are available.

            // We currently have:
            //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
            //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
            // First form
            //   t = [ a4  a5  a6  a7  b0  b1  b2  b3 ].
            let t = x86_64::_mm256_permute2x128_si256::<0x21>(a, b);

            // Then
            //   res0 = [ a0  a1  a2  a3  b0  b1  b2  b3 ],
            //   res1 = [ a4  a5  a6  a7  b4  b5  b6  b7 ].
            (
                x86_64::_mm256_blend_epi32::<0b11110000>(a, t),
                x86_64::_mm256_blend_epi32::<0b11110000>(t, b),
            )
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod interleave {
    use core::arch::x86_64::{self, __m512i, __mmask8, __mmask16};
    use core::mem::transmute;

    const EVENS: __mmask16 = 0b0101010101010101;
    const EVENS4: __mmask16 = 0x0f0f;

    // vpshrdq requires AVX-512VBMI2.
    #[cfg(target_feature = "avx512vbmi2")]
    #[inline]
    #[must_use]
    fn interleave1_antidiagonal(x: __m512i, y: __m512i) -> __m512i {
        unsafe {
            // Safety: If this code got compiled then AVX-512VBMI2 intrinsics are available.
            x86_64::_mm512_shrdi_epi64::<32>(x, y)
        }
    }

    // If we can't use vpshrdq, then do a vpermi2d, but we waste a register and double the latency.
    #[cfg(not(target_feature = "avx512vbmi2"))]
    #[inline]
    #[must_use]
    fn interleave1_antidiagonal(x: __m512i, y: __m512i) -> __m512i {
        const INTERLEAVE1_INDICES: __m512i = unsafe {
            // Safety: `[u32; 16]` is trivially transmutable to `__m512i`.
            transmute::<[u32; WIDTH], _>([
                0x01, 0x10, 0x03, 0x12, 0x05, 0x14, 0x07, 0x16, 0x09, 0x18, 0x0b, 0x1a, 0x0d, 0x1c,
                0x0f, 0x1e,
            ])
        };
        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.
            x86_64::_mm512_permutex2var_epi32(x, INTERLEAVE1_INDICES, y)
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 32-bit integers.
    ///
    /// Maps `[a0, ..., a15], [b0, ..., b15], ` to `[a0, b0, ...], [..., a15, b15]`.
    pub fn interleave_u32(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        // If we have AVX-512VBMI2, we want this to compile to:
        //      vpshrdq    t, x, y, 32
        //      vpblendmd  res0 {EVENS}, t, x
        //      vpblendmd  res1 {EVENS}, y, t
        // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
        // latency: 2 cyc
        //
        // Otherwise, we want it to compile to:
        //      vmovdqa32  t, INTERLEAVE1_INDICES
        //      vpermi2d   t, x, y
        //      vpblendmd  res0 {EVENS}, t, x
        //      vpblendmd  res1 {EVENS}, y, t
        // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
        // latency: 4 cyc

        // We currently have:
        //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
        //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
        // First form
        //   t = [ x1  y0  x3  y2  x5  y4  x7  y6  x9  y8  xb  ya  xd  yc  xf  ye ].
        let t = interleave1_antidiagonal(x, y);

        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.

            // Then
            //   res0 = [ x0  y0  x2  y2  x4  y4  x6  y6  x8  y8  xa  ya  xc  yc  xe  ye ],
            //   res1 = [ x1  y1  x3  y3  x5  y5  x7  y7  x9  y9  xb  yb  xd  yd  xf  yf ].
            (
                x86_64::_mm512_mask_blend_epi32(EVENS, t, x),
                x86_64::_mm512_mask_blend_epi32(EVENS, y, t),
            )
        }
    }

    #[inline]
    #[must_use]
    fn shuffle_epi64<const MASK: i32>(a: __m512i, b: __m512i) -> __m512i {
        // The instruction is only available in the floating-point flavor; this distinction is only for
        // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
        unsafe {
            let a = x86_64::_mm512_castsi512_pd(a);
            let b = x86_64::_mm512_castsi512_pd(b);
            x86_64::_mm512_castpd_si512(x86_64::_mm512_shuffle_pd::<MASK>(a, b))
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 64-bit integers.
    ///
    /// Maps `[a0, ..., a7], [b0, ..., b7], ` to `[a0, b0, ...], [..., a7, b7]`.
    pub fn interleave_u64(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        // We want this to compile to:
        //      vshufpd    t, x, y, 55h
        //      vpblendmq  res0 {EVENS}, t, x
        //      vpblendmq  res1 {EVENS}, y, t
        // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
        // latency: 2 cyc

        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.

            // We currently have:
            //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
            //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
            // First form
            //   t = [ x2  x3  y0  y1  x6  x7  y4  y5  xa  xb  y8  y9  xe  xf  yc  yd ].
            let t = shuffle_epi64::<0b01010101>(x, y);

            // Then
            //   res0 = [ x0  x1  y0  y1  x4  x5  y4  y5  x8  x9  y8  y9  xc  xd  yc  yd ],
            //   res1 = [ x2  x3  y2  y3  x6  x7  y6  y7  xa  xb  ya  yb  xe  xf  ye  yf ].
            (
                x86_64::_mm512_mask_blend_epi64(EVENS as __mmask8, t, x),
                x86_64::_mm512_mask_blend_epi64(EVENS as __mmask8, y, t),
            )
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 128-bit integers.
    ///
    /// Maps `[a0, ..., a3], [b0, ..., b3], ` to `[a0, b0, ...], [..., a3, b3]`.
    pub fn interleave_u128(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        // We want this to compile to:
        //      vmovdqa64   t, INTERLEAVE4_INDICES
        //      vpermi2q    t, x, y
        //      vpblendmd   res0 {EVENS4}, t, x
        //      vpblendmd   res1 {EVENS4}, y, t
        // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
        // latency: 4 cyc

        const INTERLEAVE4_INDICES: __m512i = unsafe {
            // Safety: `[u64; 8]` is trivially transmutable to `__m512i`.
            transmute::<[u64; 8], _>([0o02, 0o03, 0o10, 0o11, 0o06, 0o07, 0o14, 0o15])
        };

        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.

            // We currently have:
            //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
            //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
            // First form
            //   t = [ x4  x5  x6  x7  y0  y1  y2  y3  xc  xd  xe  xf  y8  y9  ya  yb ].
            let t = x86_64::_mm512_permutex2var_epi64(x, INTERLEAVE4_INDICES, y);

            // Then
            //   res0 = [ x0  x1  x2  x3  y0  y1  y2  y3  x8  x9  xa  xb  y8  y9  ya  yb ],
            //   res1 = [ x4  x5  x6  x7  y4  y5  y6  y7  xc  xd  xe  xf  yc  yd  ye  yf ].
            (
                x86_64::_mm512_mask_blend_epi32(EVENS4, t, x),
                x86_64::_mm512_mask_blend_epi32(EVENS4, y, t),
            )
        }
    }

    #[inline]
    #[must_use]
    /// Interleave two vectors of 256-bit integers.
    ///
    /// Maps `[a0, a1], [b0, b1], ` to `[a0, b0], [a1, b1]`.
    pub fn interleave_u256(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
        // We want this to compile to:
        //      vshufi64x2  t, x, b, 4eh
        //      vpblendmq   res0 {EVENS4}, t, x
        //      vpblendmq   res1 {EVENS4}, y, t
        // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
        // latency: 4 cyc

        unsafe {
            // Safety: If this code got compiled then AVX-512F intrinsics are available.

            // We currently have:
            //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
            //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
            // First form
            //   t = [ x8  x9  xa  xb  xc  xd  xe  xf  y0  y1  y2  y3  y4  y5  y6  y7 ].
            let t = x86_64::_mm512_shuffle_i64x2::<0b01_00_11_10>(x, y);

            // Then
            //   res0 = [ x0  x1  x2  x3  x4  x5  x6  x7  y0  y1  y2  y3  y4  y5  y6  y7 ],
            //   res1 = [ x8  x9  xa  xb  xc  xd  xe  xf  y8  y9  ya  yb  yc  yd  ye  yf ].
            (
                x86_64::_mm512_mask_blend_epi64(EVENS4 as __mmask8, t, x),
                x86_64::_mm512_mask_blend_epi64(EVENS4 as __mmask8, y, t),
            )
        }
    }
}

/// A macro to implement the PackedFieldPow2 trait for PackedFields. The macro assumes that the PackedFields
/// have a `to_vector` and `from_vector` method, which convert between the PackedField and a packed vector.
///
/// # Arguments:
/// - `$type`: The type of the PackedField.
/// - `($type_param, $param_name)`: Optional type parameter if one is needed and a name for it.
/// - `; [ ($block_len, $func), ... ]`: A list of block lengths and their corresponding interleaving functions.
/// - `$width`: The width of the PackedField, corresponding to the largest possible block length.
///
/// For example, calling this macro with:
/// ```rust,ignore
/// impl_packed_field_pow_2!(
///    PackedMontyField31Neon, (FieldParameters, FP);
///    [
///        (1, interleave_u32),
///        (2, interleave_u64),
///   ],
///    4
/// );
/// ```
/// crates the code:
/// ```rust,ignore
/// impl<FP: FieldParameters> PackedFieldPow2 for PackedMontyField31Neon<FP> {
///     #[inline]
///     fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
///         let (v0, v1) = (self.to_vector(), other.to_vector());
///         let (res0, res1) = match block_len {
///             1 => interleave_u32(v0, v1),
///             2 => interleave_u64(v0, v1),
///             4 => (v0, v1),
///             _ => panic!("unsupported block_len"),
///         };
///         unsafe {
///             // Safety: We haven't changed any values, just moved data around
///             // so all entries still represent valid field elements.
///             (Self::from_vector(res0), Self::from_vector(res1))
///         }
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_packed_field_pow_2 {
    // Accepts: type, block sizes as (block_len, function), and optional type param
    (
        $type:ty
        $(, ($type_param:ty, $param_name:ty))?
        ; [ $( ($block_len:expr, $func:ident) ),* $(,)? ],
        $width:expr
    ) => {
        paste::paste! {
            unsafe impl$(<$param_name: $type_param>)? PackedFieldPow2 for $type$(<$param_name>)? {
                #[inline]
                fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
                    let (v0, v1) = (self.to_vector(), other.to_vector());
                    let (res0, res1) = match block_len {
                        $(
                            $block_len => $func(v0, v1),
                        )*
                        $width => (v0, v1),
                        _ => panic!("unsupported block_len"),
                    };
                    unsafe {
                        // Safety: We haven't changed any values, just moved data around
                        // so all entries still represent valid field elements.
                        (Self::from_vector(res0), Self::from_vector(res1))
                    }
                }
            }
        }
    };
}

pub use impl_packed_field_pow_2;
