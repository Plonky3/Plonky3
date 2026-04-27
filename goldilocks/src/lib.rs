//! The prime field known as Goldilocks, defined as `F_p` where `p = 2^64 - 2^32 + 1`.

#![no_std]

extern crate alloc;

mod extension;
mod goldilocks;
mod mds;
mod poseidon2;

pub use goldilocks::*;
pub use mds::*;
pub use poseidon2::*;

pub mod poseidon1;

#[cfg(target_arch = "aarch64")]
mod aarch64_neon;

#[cfg(target_arch = "aarch64")]
pub use aarch64_neon::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use x86_64_avx2::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod x86_64_avx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use x86_64_avx512::*;

/// Packed scalar type intended for Merkle trees and sponge hashes (e.g. Poseidon2 leaf batching).
///
/// On **aarch64** this is always [`PackedGoldilocksNeon`], even if [`Goldilocks::Packing`] is
/// set to [`Goldilocks`] for arithmetic.
///
/// On **other targets** this is [`Goldilocks::Packing`] (AVX2, AVX-512, or scalar).
#[cfg(target_arch = "aarch64")]
pub type HashPackedGoldilocks = PackedGoldilocksNeon;

#[cfg(not(target_arch = "aarch64"))]
pub type HashPackedGoldilocks = <Goldilocks as p3_field::Field>::Packing;
