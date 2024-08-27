//! And AIR for the Poseidon2 permutation.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod generation;

pub use air::*;
pub use columns::*;
pub use generation::*;
