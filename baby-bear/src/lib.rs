#![no_std]

mod baby_bear;
pub use baby_bear::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod aarch64_neon;
