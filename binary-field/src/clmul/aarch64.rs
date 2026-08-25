//! The `PMULL` backend, which lives under the `aes` target feature.

use core::arch::aarch64::vmull_p64;

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// `PMULL` accumulates `b << i` for every set bit `i` of `a`, so bit `j` of the result is the
/// coefficient of `x^j` exactly as the rest of this module assumes.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "aes"` is enabled for the
    // crate, and `aes` implies `neon`; together those are what `vmull_p64` requires.
    unsafe { vmull_p64(a, b) }
}
