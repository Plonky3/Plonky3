//! The `PCLMULQDQ` backend.

use core::arch::x86_64::{
    _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_unpackhi_epi64,
};

mod gf192;
mod gf64;
mod ghash;
mod lanes;

// Repeated squaring as one bit-matrix product, where the byte-affine instruction is there.
#[cfg(all(
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
mod gfni;

pub(crate) use gf64::{poly_dot_64, poly_mul_64, poly_square_64};
pub(crate) use gf192::{
    poly_dot_192, poly_dot_192_by_64, poly_mul_192, poly_mul_192_by_64, poly_square_192,
};
#[cfg(all(
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
pub(crate) use gfni::square_times;
pub(crate) use ghash::{poly_dot_128, poly_mul_128, poly_mul_128_by_64, poly_square_128};

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// The instruction accumulates `b << i` for every set bit `i` of `a`.
///
/// Bit `j` of the result is therefore the coefficient of `x^j`.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module compiles only with `pclmulqdq`, which the carryless multiply needs.
    //
    // The remaining intrinsics are `sse2`, always available on `x86_64`.
    unsafe {
        // Arguments run from the highest lane down.
        //
        // The operand goes second and the unused high lane first.
        let a = _mm_set_epi64x(0, a as i64);
        let b = _mm_set_epi64x(0, b as i64);
        let product = _mm_clmulepi64_si128::<0x00>(a, b);

        // Extracting the high lane directly would need `sse4.1`, so it is moved down instead.
        let low = _mm_cvtsi128_si64(product) as u64;
        let high = _mm_cvtsi128_si64(_mm_unpackhi_epi64(product, product)) as u64;
        u128::from(low) | (u128::from(high) << 64)
    }
}
