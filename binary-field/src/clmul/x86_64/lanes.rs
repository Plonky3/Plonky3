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

    #[cfg(target_feature = "gfni")]
    #[inline(always)]
    fn reduce_product(product: Self) -> u64 {
        use core::arch::x86_64::{_mm_gf2p8affine_epi64_epi8, _mm_set1_epi64x};

        use crate::ByteMatrix;
        // Each byte of high * 0x1b contributes a low byte and a carry into
        // the next byte. The last carry folds into byte zero modulo x^64 + 0x1b.
        const LOW: u64 =
            ByteMatrix::from_images([0x1b, 0x36, 0x6c, 0xd8, 0xb0, 0x60, 0xc0, 0x80]).to_quadword();
        const CARRY: u64 = ByteMatrix::from_images([0, 0, 0, 0, 1, 3, 6, 13]).to_quadword();
        // SAFETY: the module requires PCLMULQDQ and this arm additionally requires GFNI.
        unsafe {
            let high = _mm_unpackhi_epi64(product, product);
            let low = _mm_gf2p8affine_epi64_epi8::<0>(high, _mm_set1_epi64x(LOW as i64));
            let carry = _mm_gf2p8affine_epi64_epi8::<0>(high, _mm_set1_epi64x(CARRY as i64));
            let top = _mm_gf2p8affine_epi64_epi8::<0>(
                _mm_srli_epi64::<56>(carry),
                _mm_set1_epi64x(LOW as i64),
            );
            _mm_cvtsi128_si64(product.xor3(low, _mm_slli_epi64::<8>(carry)).xor(top)) as u64
        }
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

#[cfg(all(test, target_feature = "gfni"))]
mod tests {
    use core::arch::x86_64::__m128i;

    use proptest::prelude::*;

    use crate::clmul::reduce_64;
    use crate::clmul::register::Register128;

    /// The vector register represents exactly two unrestricted coefficient words.
    fn reduce(words: [u64; 2]) -> u64 {
        // SAFETY: every bit pattern is valid in both sixteen-byte representations.
        let register = unsafe { core::mem::transmute::<[u64; 2], __m128i>(words) };
        __m128i::reduce_product(register)
    }

    #[test]
    fn affine_reduction_matches_every_polynomial_basis_vector() {
        for bit in 0..128 {
            let value = 1u128 << bit;
            assert_eq!(
                reduce([value as u64, (value >> 64) as u64]),
                reduce_64(value)
            );
        }
        assert_eq!(reduce([u64::MAX; 2]), reduce_64(u128::MAX));
        assert_eq!(reduce([0; 2]), 0);
    }

    proptest! {
        #[test]
        fn affine_reduction_matches_arbitrary_polynomials(words in any::<[u64;2]>()) {
            prop_assert_eq!(reduce(words), reduce_64(u128::from(words[0]) | (u128::from(words[1]) << 64)));
        }
    }
}
