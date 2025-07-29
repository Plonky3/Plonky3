mod packed_traits;
pub use packed_traits::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2",))]
mod x86_64_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2",))]
pub use x86_64_avx::*;
