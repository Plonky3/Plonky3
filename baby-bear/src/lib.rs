#![no_std]

extern crate alloc;

mod baby_bear;
mod extension;
mod mds;
mod non_canonical;

pub use baby_bear::*;
pub use non_canonical::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;
