//! A Blake-3 compression AIR over fields of characteristic 2.
//!
//! Every word is a vector of 32 bits, so XOR is field addition and AND of two bits is
//! field multiplication. The state `v[0..16]` is split into four rows of four words:
//! `a = v[0..4]`, `b = v[4..8]`, `c = v[8..12]` and `d = v[12..16]`.

mod air;
mod columns;
mod generation;
#[cfg(test)]
mod tests;

pub use air::*;
pub use columns::*;
pub use generation::*;

use crate::constants::IV;

/// Number of rounds in one compression.
const NUM_ROUNDS: usize = 7;

/// Number of G steps in one round.
const G_PER_ROUND: usize = 8;

/// Indices of the `a`, `b`, `c`, `d` words within their state rows, for each G step of a round.
///
/// Step `g` mixes `v[a]`, `v[4 + b]`, `v[8 + c]`, `v[12 + d]` with the message words
/// `m[2g]` and `m[2g + 1]`: four column steps followed by four diagonal steps.
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

/// The initialization vector word at `index` as a `u32`.
const fn iv_word(index: usize) -> u32 {
    IV[index][0] as u32 | ((IV[index][1] as u32) << 16)
}
