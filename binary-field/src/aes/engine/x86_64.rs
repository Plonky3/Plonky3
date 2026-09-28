//! The byte-wise backend over the Galois-field new instructions.
//!
//! Three instructions cover everything the engine needs, each acting on every byte at once:
//! the AES-field product, an arbitrary `F_2`-linear map, and inversion followed by such a map.
//!
//! The inverting form is what the block cipher's substitution box is built from.
//!
//! Composed with the identity map it is a bare field inversion.
//!
//! An inverse here therefore costs one instruction rather than an exponentiation chain.

use core::arch::x86_64::{
    __m128i, _mm_gf2p8affine_epi64_epi8, _mm_gf2p8affineinv_epi64_epi8, _mm_gf2p8mul_epi8,
    _mm_loadu_si128, _mm_set1_epi64x, _mm_storeu_si128,
};

use super::ByteLanes;

/// The narrower register the kernels sweep with, holding sixteen bytes.
pub(super) type Narrow = __m128i;

// SAFETY: the register is sixteen contiguous bytes with no invalid bit pattern.
unsafe impl ByteLanes for __m128i {
    const WIDTH: usize = 16;

    type Matrix = Self;

    #[inline(always)]
    fn matrix(quadword: u64) -> Self::Matrix {
        // SAFETY: the broadcast is `sse2`, always present on this architecture.
        //
        // The matrix operand is read per quadword, so one copy covers the register.
        unsafe { _mm_set1_epi64x(quadword as i64) }
    }

    #[inline(always)]
    unsafe fn load(from: *const u8) -> Self {
        // SAFETY: readability is the caller's obligation, and the load is the unaligned form.
        unsafe { _mm_loadu_si128(from.cast()) }
    }

    #[inline(always)]
    unsafe fn store(to: *mut u8, value: Self) {
        // SAFETY: writability is the caller's obligation, and the store is the unaligned form.
        unsafe { _mm_storeu_si128(to.cast(), value) }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // SAFETY: this module is compiled only with the byte-wise field instructions enabled.
        unsafe { _mm_gf2p8mul_epi8(self, other) }
    }

    #[inline(always)]
    fn affine(self, matrix: Self::Matrix) -> Self {
        // SAFETY: as above.
        unsafe { _mm_gf2p8affine_epi64_epi8::<0>(self, matrix) }
    }

    #[inline(always)]
    fn inverse_then_affine(self, matrix: Self::Matrix) -> Self {
        // SAFETY: as above.
        unsafe { _mm_gf2p8affineinv_epi64_epi8::<0>(self, matrix) }
    }
}

/// The widest register the kernels sweep with, holding sixteen bytes.
///
/// A 256-bit form exists without the half-kilobit registers, but no test leg here runs it.
#[cfg(not(all(target_feature = "avx512f", target_feature = "avx512bw")))]
pub(super) type Wide = __m128i;

/// The widest register the kernels sweep with, holding sixty-four bytes.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
pub(super) type Wide = core::arch::x86_64::__m512i;

#[cfg(all(target_feature = "avx512f", target_feature = "avx512bw"))]
mod wide {
    use core::arch::x86_64::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_gf2p8affineinv_epi64_epi8,
        _mm512_gf2p8mul_epi8, _mm512_loadu_si512, _mm512_set1_epi64, _mm512_storeu_si512,
    };

    use super::ByteLanes;

    // SAFETY: the register is sixty-four contiguous bytes with no invalid bit pattern.
    unsafe impl ByteLanes for __m512i {
        const WIDTH: usize = 64;

        type Matrix = Self;

        #[inline(always)]
        fn matrix(quadword: u64) -> Self::Matrix {
            // SAFETY: this module is compiled only with the wide registers enabled.
            unsafe { _mm512_set1_epi64(quadword as i64) }
        }

        #[inline(always)]
        unsafe fn load(from: *const u8) -> Self {
            // SAFETY: readability is the caller's obligation, and the load is unaligned.
            unsafe { _mm512_loadu_si512(from.cast()) }
        }

        #[inline(always)]
        unsafe fn store(to: *mut u8, value: Self) {
            // SAFETY: writability is the caller's obligation, and the store is unaligned.
            unsafe { _mm512_storeu_si512(to.cast(), value) }
        }

        #[inline(always)]
        fn mul(self, other: Self) -> Self {
            // SAFETY: this module is compiled only with the wide form of the instruction.
            unsafe { _mm512_gf2p8mul_epi8(self, other) }
        }

        #[inline(always)]
        fn affine(self, matrix: Self::Matrix) -> Self {
            // SAFETY: as above.
            unsafe { _mm512_gf2p8affine_epi64_epi8::<0>(self, matrix) }
        }

        #[inline(always)]
        fn inverse_then_affine(self, matrix: Self::Matrix) -> Self {
            // SAFETY: as above.
            unsafe { _mm512_gf2p8affineinv_epi64_epi8::<0>(self, matrix) }
        }
    }
}
