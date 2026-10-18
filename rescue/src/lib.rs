//! The Rescue permutation and its Rescue-Prime Optimized (RPO) variant.

#![no_std]

extern crate alloc;

mod rescue;
mod rpo;
mod util;

pub use rescue::*;
pub use rpo::*;
