//! The `PCLMULQDQ` backend.

use core::arch::x86_64::{
    _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_unpackhi_epi64,
};

/// Selects the low quadword of both operands: bit 0 picks the half of the first argument and
/// bit 4 the half of the second, so `0x00` is `a.low × b.low`.
const LOW_BY_LOW: i32 = 0x00;

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// `PCLMULQDQ` accumulates `b << i` for every set bit `i` of `a`, so bit `j` of the result is
/// the coefficient of `x^j` exactly as the rest of this module assumes.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "pclmulqdq"` is enabled for
    // the crate, which is what `_mm_clmulepi64_si128` requires; the remaining intrinsics are
    // `sse2`, which is unconditionally available on `x86_64`.
    unsafe {
        // `_mm_set_epi64x` takes its arguments from highest lane to lowest, so the operand goes
        // in the second position and the unused high lane in the first.
        let a = _mm_set_epi64x(0, a as i64);
        let b = _mm_set_epi64x(0, b as i64);
        let product = _mm_clmulepi64_si128::<LOW_BY_LOW>(a, b);

        // `_mm_extract_epi64` would need `sse4.1`, so the high lane is moved down instead.
        let low = _mm_cvtsi128_si64(product) as u64;
        let high = _mm_cvtsi128_si64(_mm_unpackhi_epi64(product, product)) as u64;
        u128::from(low) | (u128::from(high) << 64)
    }
}
