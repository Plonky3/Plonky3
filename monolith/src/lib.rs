//! The Monolith permutation, and hash functions built from it.

#![no_std]

extern crate alloc;

mod monolith;
mod monolith_mds;
mod monolith_mds_width16;
mod monolith_width16;
mod util;

pub use monolith::MonolithM31;
pub use monolith_mds::MonolithMdsMatrixM31;
pub use monolith_mds_width16::MonolithMdsMatrixM31Width16;
pub use monolith_width16::MonolithM31Width16;
