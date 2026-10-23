//! **Deprecated since 0.9.0.** This crate is planned for removal in a future release.
//! Avoid using it in new projects.
//!
//! And AIR for the Poseidon2 permutation.

#![deprecated(
    since = "0.9.0",
    note = "p3-poseidon2-air is planned for removal in a future release."
)]
#![no_std]

extern crate alloc;

mod air;
mod columns;
mod constants;
mod generation;
mod vectorized;

pub use air::*;
pub use columns::*;
pub use constants::*;
pub use generation::*;
pub use vectorized::*;
