//! A framework for finite fields.

#![no_std]

mod data_traits;
mod extension;
mod monty_31;
mod poseidon2;
mod utils;
mod mds;

pub use data_traits::*;
pub use monty_31::*;
pub use poseidon2::*;
pub use utils::*;
pub use mds::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use x86_64_avx2::*;

#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
mod x86_64_avx512;
#[cfg(all(
    feature = "nightly-features",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub use x86_64_avx512::*;
