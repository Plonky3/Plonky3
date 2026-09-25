//! A BLAKE2s compression AIR over fields of characteristic 2.
//!
//! Every 32-bit word is stored as 32 bit columns.
//!
//! In characteristic 2:
//! - XOR of two bits is field addition.
//! - AND of two bits is field multiplication.
//!
//! The working state `v[0..16]` is viewed as four rows of four words:
//!
//! ```text
//!     a = v[0..4]      starts as the chaining value, first half
//!     b = v[4..8]      starts as the chaining value, second half
//!     c = v[8..12]     starts as IV[0..4]
//!     d = v[12..15]    starts as IV[4..7] ^ (counter_low, counter_high, last_block)
//!     v[15]            starts as IV[7], which section 3.2 never touches
//! ```

mod air;
mod columns;
mod generation;
#[cfg(test)]
mod tests;

pub use air::*;
pub use columns::*;
pub use generation::*;

/// Number of rounds in one compression, from RFC 7693 section 2.1.
const NUM_ROUNDS: usize = 10;

/// Number of G steps in one round: four on the columns, then four on the diagonals.
const G_PER_ROUND: usize = 8;

/// Positions of the four words each G step mixes, one entry per step of a round.
///
/// Entry `[i_a, i_b, i_c, i_d]` mixes `v[i_a]`, `v[4 + i_b]`, `v[8 + i_c]` and `v[12 + i_d]`.
///
/// ```text
///     steps 0..4 (columns)      steps 4..8 (diagonals)
///     v[0] v[4] v[8]  v[12]     v[0] v[5] v[10] v[15]
///     v[1] v[5] v[9]  v[13]     v[1] v[6] v[11] v[12]
///     v[2] v[6] v[10] v[14]     v[2] v[7] v[8]  v[13]
///     v[3] v[7] v[11] v[15]     v[3] v[4] v[9]  v[14]
/// ```
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
