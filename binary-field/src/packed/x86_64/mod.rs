//! The packings over the wide carryless multiply, `VPCLMULQDQ` on 256- and 512-bit registers.
//!
//! ```text
//!     lanes/       the register wrappers, one file per width
//!     ghash128.rs  GF(2^128), one element per 128-bit lane
//!     poly64.rs    GF(2^64), one element per quadword
//!     poly192.rs   its cubic extension, one register per coordinate
//! ```

mod ghash128;
pub(crate) mod lanes;
mod poly192;
mod poly64;

pub use ghash128::PackedGhash128;
pub use poly64::PackedPoly64;
pub use poly192::PackedPoly192;
