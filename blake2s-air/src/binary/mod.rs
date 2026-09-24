//! A BLAKE2s compression AIR over fields of characteristic 2.
//!
//! Every word is a vector of 32 bits, so XOR is field addition and AND of two bits is
//! field multiplication. The state `v[0..16]` is split into four rows of four words:
//! `a = v[0..4]`, `b = v[4..8]`, `c = v[8..12]` and `d = v[12..16]`.
//!
//! The mixing function is BLAKE3's, with the same rotations, so the witness layout of one G
//! step is the same. What differs is around it: ten rounds rather than seven, a named message
//! schedule rather than a permutation, and a starting state that XORs the counter and the
//! finalization flags into the last four initialization words.

mod air;
mod columns;
mod generation;
#[cfg(test)]
mod tests;

pub use air::*;
pub use columns::*;
pub use generation::*;

/// Number of rounds in one compression.
const NUM_ROUNDS: usize = 10;

/// Number of G steps in one round.
const G_PER_ROUND: usize = 8;

/// Indices of the `a`, `b`, `c`, `d` words within their state rows, for each G step of a round.
///
/// Step `g` mixes `v[a]`, `v[4 + b]`, `v[8 + c]`, `v[12 + d]` with two message words:
/// four column steps followed by four diagonal steps.
const G_SCHEDULE: [[usize; 4]; G_PER_ROUND] = [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
];
