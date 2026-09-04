//! The SIMD packing of the polynomial-basis `GF(2^128)`.
//!
//! A packing exists only where the multiply reaches more than one 128-bit lane.
//! Only the widest such register is used, so there is one packing per build.
//!
//! The tower representation has no packing.
//! A product there is table lookups, which no vector unit widens.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
mod x86_64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
pub use x86_64::PackedGhash128;
