//! The 128-bit register as a backend of the shared `GF(2^64)` algebra.

#[cfg(target_feature = "ssse3")]
use core::arch::x86_64::_mm_shuffle_epi8;
#[cfg(all(target_feature = "avx512f", target_feature = "avx512vl"))]
use core::arch::x86_64::_mm_ternarylogic_epi64;
use core::arch::x86_64::{
    __m128i, _mm_clmulepi64_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_loadl_epi64,
    _mm_loadu_si128, _mm_setzero_si128, _mm_shuffle_epi32, _mm_slli_epi64, _mm_srli_epi64,
    _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi64, _mm_unpacklo_epi64, _mm_xor_si128,
};
use core::ptr;

use crate::clmul::register::Register128;
use crate::clmul::wide::Lanes64;
#[cfg(target_feature = "ssse3")]
use crate::clmul::wide::TOP_NIBBLE_FOLD;

/// Swaps the two quadwords of a register.
const SWAP_QUADWORDS: i32 = 0x4e;

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

// SAFETY for every method below: `sse2` is part of the `x86_64` baseline.
//
// Every load and store stays inside the array or the reference it is handed, in unaligned form.
impl Register128 for __m128i {
    // The carryless multiplier is the scarce unit on the cores this backend targets.
    const CHEAP_MULTIPLY: bool = false;

    #[inline(always)]
    fn reduce_product(product: Self) -> u64 {
        // Two carryless folds keep both product halves in the vector register file.
        super::super::wide::reduce_by_multiply(
            product,
            Self::lift(super::super::wide::TAIL_64).swap(),
        )
        .lower()
    }

    #[inline(always)]
    fn lift(value: u64) -> Self {
        unsafe { _mm_cvtsi64_si128(value as i64) }
    }

    #[inline(always)]
    fn lower(self) -> u64 {
        unsafe { _mm_cvtsi128_si64(self) as u64 }
    }

    #[inline(always)]
    fn swap(self) -> Self {
        unsafe { _mm_shuffle_epi32::<SWAP_QUADWORDS>(self) }
    }

    #[inline(always)]
    fn load(a: &[u64; 3]) -> (Self, Self) {
        // Bytes 0 to 15, then bytes 16 to 23.
        unsafe {
            (
                _mm_loadu_si128(a.as_ptr().cast()),
                _mm_loadl_epi64(a[2..].as_ptr().cast()),
            )
        }
    }

    #[inline(always)]
    fn store(pair: Self, last: Self) -> [u64; 3] {
        let mut out = [0u64; 3];
        unsafe {
            _mm_storeu_si128(out.as_mut_ptr().cast(), pair);
            _mm_storel_epi64(out[2..].as_mut_ptr().cast(), last);
        }
        out
    }

    #[inline(always)]
    fn load_scalar(k: &u64) -> Self {
        unsafe { _mm_loadl_epi64(ptr::from_ref(k).cast()) }
    }
}
