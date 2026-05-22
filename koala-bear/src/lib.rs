#![no_std]

extern crate alloc;

mod extension;
mod koala_bear;
mod poseidon2;

pub use koala_bear::*;
pub use poseidon2::*;

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
