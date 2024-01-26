//! The Monolith permutation, and hash functions built from it.

#![no_std]

extern crate alloc;

mod monolith;
mod monolith_mds;
mod monolith_mds_u64;
mod monolith_u64;
mod util;

pub use monolith::MonolithMersenne31;
pub use monolith_mds::MonolithMdsMatrixMersenne31;
pub use monolith_mds_u64::MonolithMdsMatrixMersenne31U64Width16;
pub use monolith_u64::MonolithMersenne31U64Width16;
