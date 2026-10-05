//! The packings over the wide carryless multiply, `VPCLMULQDQ` on 256- and 512-bit registers.

mod ghash128;
pub(crate) mod lanes;
// Sums of products pair their terms where a 512-bit multiply exists.
#[cfg(target_feature = "avx512f")]
pub(crate) mod pairs;

pub use ghash128::PackedGhash128;
