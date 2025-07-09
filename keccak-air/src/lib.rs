//! An AIR for the Keccak-f permutation. Assumes the field size is between 2^16 and 2^32.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod constants;
mod generation;
mod round_flags;

pub use air::*;
pub use columns::*;
pub use constants::*;
pub use generation::*;

/// Total number of Keccak-f rounds.
pub const NUM_ROUNDS: usize = 24;

/// Number of Keccak-f rounds minus one.
pub const NUM_ROUNDS_MIN_1: usize = NUM_ROUNDS - 1;

/// Number of bits in each limb used to represent 64-bit words.
const BITS_PER_LIMB: usize = 16;

/// Number of limbs needed to represent a 64-bit word.
///
/// Computed as 64 divided by the number of bits per limb.
pub const U64_LIMBS: usize = 64 / BITS_PER_LIMB;

/// Number of rate bits in Keccak-f.
///
/// In Keccak-f[1600], the "rate" parameter for absorbing and squeezing is 1088 bits.
const RATE_BITS: usize = 1088;

/// Number of limbs needed to represent the rate portion of the state.
///
/// Computed as rate bits divided by bits per limb.
const RATE_LIMBS: usize = RATE_BITS / BITS_PER_LIMB;
