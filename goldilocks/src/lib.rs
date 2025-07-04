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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
mod x86_64_avx512;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
pub use x86_64_avx512::*;
