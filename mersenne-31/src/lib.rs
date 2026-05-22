//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]
extern crate alloc;

mod complex;
mod dft;
mod extension;
mod mds;
mod mersenne_31;
mod poseidon2;
mod radix_2_dit;

pub use dft::Mersenne31Dft;
pub use mds::*;
pub use mersenne_31::*;
pub use poseidon2::*;
pub use radix_2_dit::Mersenne31ComplexRadix2Dit;

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        mod aarch64_neon;
        pub use aarch64_neon::*;
    }
}

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", rustc_version_1_89_or_later))] {
        mod x86_64_avx512;
        pub use x86_64_avx512::*;
    }
}

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        mod x86_64_avx2;
        pub use x86_64_avx2::*;
    }
}
