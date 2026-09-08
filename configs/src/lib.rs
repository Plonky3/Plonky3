#![no_std]
#![doc = include_str!("../README.md")]

#[cfg(feature = "binary")]
extern crate alloc;

#[cfg(any(feature = "baby-bear", feature = "koala-bear", feature = "goldilocks"))]
mod prime;

/// BabyBear, quartic challenges, Poseidon2 and two-adic FRI.
#[cfg(feature = "baby-bear")]
pub mod baby_bear;
/// GF(2^128), Keccak and the additive-domain multilinear PCS.
#[cfg(feature = "binary")]
pub mod binary;
/// Goldilocks, quadratic challenges, Poseidon2 and two-adic FRI.
#[cfg(feature = "goldilocks")]
pub mod goldilocks;
/// KoalaBear, quartic challenges, Poseidon2 and two-adic FRI.
#[cfg(feature = "koala-bear")]
pub mod koala_bear;

/// Multilinear setup, proving and verification APIs.
#[cfg(feature = "binary")]
pub use p3_multi_stark as multi_stark;
/// Univariate proving, verification, preprocessing and security analysis APIs.
#[cfg(any(feature = "baby-bear", feature = "koala-bear", feature = "goldilocks"))]
pub use p3_uni_stark as uni_stark;
