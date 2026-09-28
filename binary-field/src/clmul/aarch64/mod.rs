//! The `PMULL` backend, which lives under the `aes` target feature on little-endian AArch64.

use core::arch::aarch64::vmull_p64;

mod ghash;
mod lanes;
mod square;

pub(crate) use ghash::{
    SplitMultiplier, poly_add_128, poly_dot_128, poly_mul_128, poly_mul_128_by_64, poly_square_128,
};
pub(crate) use square::square_times;

// Every transmute between a `u128` and a vector relies on its halves being the lanes.
//
// On big-endian AArch64 they run in opposite orders.
//
// The gates that select this module name `target_endian = "little"`, and this holds them to it.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "the halves of a `u128` are its vector lanes only on little-endian targets"
);

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// `PMULL` accumulates `b << i` for every set bit `i` of `a`.
///
/// Bit `j` of the result is therefore the coefficient of `x^j`.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "aes"` is enabled.
    //
    // `aes` implies `neon`, and together they are what the carryless multiply requires.
    unsafe { vmull_p64(a, b) }
}
