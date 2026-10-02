//! The packings over the wide carryless multiply, `VPCLMULQDQ` on 256- and 512-bit registers.

mod ghash128;
pub(crate) mod lanes;

pub use ghash128::PackedGhash128;
