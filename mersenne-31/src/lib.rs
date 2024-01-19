//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]

extern crate alloc;

mod complex;
mod dft;
mod extension;
mod mersenne_31;
mod radix_2_dit;

pub use complex::*;
pub use dft::Mersenne31Dft;
pub use mersenne_31::*;
pub use radix_2_dit::Mersenne31ComplexRadix2Dit;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use x86_64_avx2::*;
