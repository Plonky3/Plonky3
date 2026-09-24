//! The 128-bit register as a backend of the shared `GF(2^64)` algebra.

#[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
use core::arch::x86_64::_mm_ternarylogic_epi64;
use core::arch::x86_64::{
    __m128i, _mm_clmulepi64_si128, _mm_setzero_si128, _mm_slli_epi64, _mm_srli_epi64,
    _mm_unpackhi_epi64, _mm_unpacklo_epi64, _mm_xor_si128,
};
#[cfg(target_feature = "ssse3")]
use core::arch::x86_64::{_mm_loadu_si128, _mm_shuffle_epi8};

use crate::clmul::wide::Lanes64;
#[cfg(target_feature = "ssse3")]
use crate::clmul::wide::TOP_NIBBLE_FOLD;

/// The truth table of `a ^ b ^ c` for a ternary logic instruction.
#[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
const XOR3: i32 = 0x96;

// SAFETY for every method below: this module is compiled only when `pclmulqdq` is enabled.
//
// `sse2` is part of the `x86_64` baseline.
//
// The shuffle and ternary arms are compiled only under the features they require.
impl Lanes64 for __m128i {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { _mm_xor_si128(self, other) }
    }

    #[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        unsafe { _mm_ternarylogic_epi64::<XOR3>(self, b, c) }
    }

    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        unsafe { _mm_clmulepi64_si128::<IMM>(self, other) }
    }

    #[inline(always)]
    fn unpack_low(self, other: Self) -> Self {
        unsafe { _mm_unpacklo_epi64(self, other) }
    }

    #[inline(always)]
    fn unpack_high(self, other: Self) -> Self {
        unsafe { _mm_unpackhi_epi64(self, other) }
    }

    #[inline(always)]
    fn shl<const N: i32>(self) -> Self {
        unsafe { _mm_slli_epi64::<N>(self) }
    }

    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        unsafe { _mm_srli_epi64::<N>(self) }
    }

    #[cfg(target_feature = "ssse3")]
    #[inline(always)]
    fn fold_top_nibble(self) -> Self {
        // The nibble lands in the low byte of each quadword, and every other byte is zero.
        //
        // Entry zero of the table is zero, so those bytes look up nothing.
        unsafe {
            let table = _mm_loadu_si128(TOP_NIBBLE_FOLD.as_ptr().cast());
            _mm_shuffle_epi8(table, self.shr::<60>())
        }
    }
}
