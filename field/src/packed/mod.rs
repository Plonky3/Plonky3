pub mod interleaves;
mod packed_traits;

#[allow(unused_imports)]
pub use interleaves::*; // Only used when vectorizations are available
pub use packed_traits::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2",))]
mod x86_64_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2",))]
pub use x86_64_avx::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64_neon;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64_neon::*;

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2",)
)))]
mod no_packing;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2",)
)))]
pub use no_packing::*;
