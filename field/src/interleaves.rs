//! A file containing a collection of architecture-specific interleaving functions.
//! Used for PackedFields to implement interleaving operations.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(feature = "nightly-features", target_feature = "avx512f"))
))]
pub mod interleave_avx2 {
    use core::arch::x86_64::{self, __m256i};

    #[inline]
    #[must_use]
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
