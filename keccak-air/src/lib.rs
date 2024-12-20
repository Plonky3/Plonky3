//! And AIR for the Keccak-f permutation. Assumes the field size is between 2^16 and 2^32.

#![no_std]

extern crate alloc;

mod air;
mod columns;
pub mod constants;
pub mod generation;
pub mod logic;
mod round_flags;

pub use air::*;
pub use columns::*;
pub use generation::*;

pub const NUM_ROUNDS: usize = 24;
pub const BITS_PER_LIMB: usize = 8;
pub const U64_LIMBS: usize = 64 / BITS_PER_LIMB;
const RATE_BITS: usize = 1088;
const RATE_LIMBS: usize = RATE_BITS / BITS_PER_LIMB;
