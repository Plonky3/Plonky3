//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]

extern crate alloc;

mod complex;
mod dft;
mod extension;
mod mds;
mod mersenne_31;
mod non_canonical;
mod radix_2_dit;

pub use complex::*;
pub use dft::Mersenne31Dft;
pub use mds::*;
pub use mersenne_31::*;
pub use non_canonical::*;
pub use radix_2_dit::Mersenne31ComplexRadix2Dit;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;
