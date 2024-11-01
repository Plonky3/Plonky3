//! And AIR for the Keccak-f permutation. Assumes the field size is between 2^16 and 2^32.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod constants;
// mod generation;
// mod logic;
// mod round_flags;

pub use air::*;
pub use columns::*;
// pub use generation::*;

pub const NUM_ROUNDS: usize = 7;
const BITS_PER_LIMB: usize = 16;
pub const U32_LIMBS: usize = 32 / BITS_PER_LIMB;
// const RATE_BITS: usize = 1088;
// const RATE_LIMBS: usize = RATE_BITS / BITS_PER_LIMB;
