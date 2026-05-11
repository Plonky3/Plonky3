//! An AIR for the Poseidon2 table for recursion. Handles sponge operations and compressions.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod public_types;

pub use air::*;
pub use columns::*;
pub use public_types::*;
