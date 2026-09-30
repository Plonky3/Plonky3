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

/// The 512-bit register as a backend of the same algebra.
///
/// On a core with a full 512-bit datapath, a carryless multiply costs the same at every width.
///
/// So a kernel with four products to give runs them as one instruction.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
mod zmm {
    #[cfg(target_feature = "avx512bw")]
    use core::arch::x86_64::_mm_loadu_si128;
    use core::arch::x86_64::{
        __m512i, _mm512_clmulepi64_epi128, _mm512_setzero_si512, _mm512_unpackhi_epi64,
        _mm512_unpacklo_epi64, _mm512_xor_si512,
    };
    #[cfg(target_feature = "avx512bw")]
    use core::arch::x86_64::{_mm512_broadcast_i32x4, _mm512_shuffle_epi8};
    #[cfg(not(miri))]
    use core::arch::x86_64::{_mm512_set1_epi64, _mm512_sllv_epi64, _mm512_srlv_epi64};
    #[cfg(miri)]
    use core::mem::transmute;

    use crate::clmul::wide::Lanes64;
    #[cfg(target_feature = "avx512bw")]
    use crate::clmul::wide::TOP_NIBBLE_FOLD;

    /// Applies `f` to each of the eight quadwords of `v`.
    ///
    /// Miri has no shim for the 512-bit per-lane shifts, so the interpreter shifts one lane at a time.
    ///
    /// The immediate shift would be shimmed, but it takes an unsigned count the trait's signed one cannot be cast to.
    #[cfg(miri)]
    #[inline(always)]
    fn map_lanes(v: __m512i, f: impl Fn(u64) -> u64) -> __m512i {
        // SAFETY: both types are 64 bytes of plain integer data, so every bit pattern is valid in each.
        let lanes: [u64; 8] = unsafe { transmute(v) };
        unsafe { transmute(lanes.map(f)) }
    }

    // SAFETY for every method below: this module is compiled only with `avx512f` and `vpclmulqdq`.
    //
    // The byte shuffle arm is compiled only with `avx512bw`, which it requires.
    //
    // Three-way sums keep the default two exclusive ors.
    //
    // The compiler already fuses them into one ternary logic op, and the interpreter the tests run under has no shim for it.
    impl Lanes64 for __m512i {
        #[inline(always)]
        fn zero() -> Self {
            unsafe { _mm512_setzero_si512() }
        }

        #[inline(always)]
        fn xor(self, other: Self) -> Self {
            unsafe { _mm512_xor_si512(self, other) }
        }

        #[inline(always)]
        fn clmul<const IMM: i32>(self, other: Self) -> Self {
            unsafe { _mm512_clmulepi64_epi128::<IMM>(self, other) }
        }

        #[inline(always)]
        fn unpack_low(self, other: Self) -> Self {
            unsafe { _mm512_unpacklo_epi64(self, other) }
        }

        #[inline(always)]
        fn unpack_high(self, other: Self) -> Self {
            unsafe { _mm512_unpackhi_epi64(self, other) }
        }

        #[inline(always)]
        fn shl<const N: i32>(self) -> Self {
            // A constant count lowers to the immediate form.
            #[cfg(not(miri))]
            let out = unsafe { _mm512_sllv_epi64(self, _mm512_set1_epi64(i64::from(N))) };
            #[cfg(miri)]
            let out = map_lanes(self, |lane| lane << N);
            out
        }

        #[inline(always)]
        fn shr<const N: i32>(self) -> Self {
            // A constant count lowers to the immediate form.
            #[cfg(not(miri))]
            let out = unsafe { _mm512_srlv_epi64(self, _mm512_set1_epi64(i64::from(N))) };
            #[cfg(miri)]
            let out = map_lanes(self, |lane| lane >> N);
            out
        }

        #[cfg(target_feature = "avx512bw")]
        #[inline(always)]
        fn fold_top_nibble(self) -> Self {
            // The nibble lands in the low byte of each quadword, and every other byte is zero.
            //
            // Entry zero of the table is zero, so those bytes look up nothing.
            unsafe {
                let table =
                    _mm512_broadcast_i32x4(_mm_loadu_si128(TOP_NIBBLE_FOLD.as_ptr().cast()));
                _mm512_shuffle_epi8(table, self.shr::<60>())
            }
        }
    }
}
