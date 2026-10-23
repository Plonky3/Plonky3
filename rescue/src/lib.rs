//! **Deprecated since 0.9.0.** This crate is planned for removal in a future release.
//! Avoid using it in new projects.
//!
//! The Rescue permutation and its Rescue-Prime Optimized (RPO) variant.

#![deprecated(
    since = "0.9.0",
    note = "p3-rescue is planned for removal in a future release."
)]
#![no_std]

extern crate alloc;

mod rescue;
mod rpo;
mod util;

pub use rescue::*;
pub use rpo::*;
