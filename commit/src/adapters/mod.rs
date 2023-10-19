//! Adapters for converting between different types of commitment schemes.

mod extension_mmcs;
mod multi_from_uni_pcs;
mod uni_from_multi_pcs;

pub use extension_mmcs::*;
pub use multi_from_uni_pcs::*;
pub use uni_from_multi_pcs::*;
