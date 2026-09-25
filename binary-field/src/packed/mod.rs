//! The SIMD packings of the polynomial-basis fields.
//!
//! A packing exists only where the multiply reaches more than one 128-bit lane.
//!
//! Only the widest such register is used, so there is one packing per field per build.
//!
//! The tower representation has no packing.
//!
//! A product there is table lookups, which no vector unit widens.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
mod x86_64;

// The algebra the wide kernels share, plus the scalar model that keeps it honest.
//
// Compiled under `test` on every target, so no leg can miss the model.
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    )
))]
pub(crate) mod split;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
// Which type each field packs into, decided once here rather than at each use.
mod selected {
    // The lane wrappers, shared with the polynomial-basis slice kernels.
    pub(crate) use super::x86_64::lanes;
    use super::x86_64::{PackedGhash128, PackedPoly64, PackedPoly192};

    /// The packing of `GF(2^128)` in the GHASH basis.
    pub(crate) type Ghash128Packing = PackedGhash128;
    /// The packing of `GF(2^64)`.
    pub(crate) type Poly64Packing = PackedPoly64;
    /// The packing of `GF(2^192)` over the packing of `GF(2^64)`.
    pub(crate) type Poly192Packing = PackedPoly192;
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
)))]
// Without a wide carryless multiply, every field is its own packing.
mod selected {
    /// One element per packing, where no register widens the multiply.
    pub(crate) type Ghash128Packing = crate::Ghash128;
    /// One element per packing.
    pub(crate) type Poly64Packing = crate::Poly64;
    /// One element per packing.
    pub(crate) type Poly192Packing = crate::Poly192;
}

pub(crate) use selected::*;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
pub use x86_64::{PackedGhash128, PackedPoly64, PackedPoly192};
