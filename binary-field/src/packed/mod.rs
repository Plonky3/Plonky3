//! The SIMD packings of the polynomial-basis fields.
//!
//! On `x86_64` a packing exists where the multiply reaches more than one 128-bit lane.
//!
//! There, `GF(2^128)` takes the widest such register, and `GF(2^64)` and `GF(2^192)` default to 256 bits.
//!
//! The `wide-poly` feature selects 512-bit polynomial packings with AVX-512F/BW and VPCLMULQDQ.
//!
//! On AArch64 the multiply reaches one 128-bit lane.
//!
//! That lane still holds two `GF(2^64)` elements, so those fields pack two per register.
//!
//! A packing of two lanes groups `GF(2^128)` rows instead.
//!
//! Every other field on every other target is its own packing.
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

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "aes"
))]
mod aarch64;

// The packings of `GF(2^64)` and `GF(2^192)`, written once over either register.
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ),
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    )
))]
pub(crate) mod poly192;
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ),
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    )
))]
mod poly64;

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
    use super::x86_64::PackedGhash128;
    pub(crate) use super::x86_64::lanes;
    use super::{PackedPoly64, PackedPoly192};

    /// The packing of `GF(2^128)` in the GHASH basis.
    pub(crate) type Ghash128Packing = PackedGhash128;
    /// The packing of `GF(2^64)`.
    pub(crate) type Poly64Packing = PackedPoly64;
    /// The packing of `GF(2^192)` over the packing of `GF(2^64)`.
    pub(crate) type Poly192Packing = PackedPoly192;
}

#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "aes"
))]
// Two `GF(2^128)` rows per packing, and two `GF(2^64)` elements per register.
mod selected {
    use super::{PackedPoly64, PackedPoly192};

    /// The packing of `GF(2^128)` in the GHASH basis, two rows side by side.
    pub(crate) type Ghash128Packing = super::aarch64::PackedGhash128;
    /// The packing of `GF(2^64)`.
    pub(crate) type Poly64Packing = PackedPoly64;
    /// The packing of `GF(2^192)` over the packing of `GF(2^64)`.
    pub(crate) type Poly192Packing = PackedPoly192;
}

#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ),
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    )
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

// The register behind the `GF(2^64)` packings, and its moves.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "aes"
))]
pub use aarch64::PackedGhash128;
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_feature = "aes"
))]
use aarch64::gf64;
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ),
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    )
))]
pub use poly64::PackedPoly64;
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "vpclmulqdq",
        any(target_feature = "avx2", target_feature = "avx512f")
    ),
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    )
))]
pub use poly192::PackedPoly192;
pub(crate) use selected::*;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
pub use x86_64::PackedGhash128;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
#[cfg(not(all(
    feature = "wide-poly",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
use x86_64::lanes::gf64;

#[cfg(all(
    feature = "wide-poly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
mod avx512_poly;
#[cfg(all(
    feature = "wide-poly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq"
))]
use avx512_poly::gf64;
