//! The packings over the wide carryless multiply, `VPCLMULQDQ` on 256- and 512-bit registers.

mod ghash128;
pub(crate) mod lanes;
mod poly192;
mod poly64;

pub use ghash128::PackedGhash128;
pub use poly64::PackedPoly64;
pub use poly192::PackedPoly192;
