//! A SHA-256 compression AIR over fields of characteristic 2.
//!
//! Every word is a vector of 32 bits, so XOR is field addition and AND of two bits is
//! field multiplication. Each row proves one compression.
//!
//! The working state is stored the same way as in the prime-field AIR: a round writes only
//! the new `a` and `e` words, so `b, c, d` and `f, g, h` are earlier entries of the
//! `a` and `e` chains and cost no columns of their own. See [`Sha256BinaryCols`].

mod air;
mod columns;
mod generation;
#[cfg(test)]
mod tests;

pub use air::*;
pub use columns::*;
pub use generation::*;

/// Number of stored-carry additions in `T1 = h + Σ1(e) + Ch(e, f, g) + K[t] + W[t]`.
///
/// Adding five words takes four additions. The first three store their carries; the last
/// reads its carries off the stored `T1`.
const T1_CARRIES: usize = 3;

/// Number of stored-carry additions in `W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]`.
///
/// Adding four words takes three additions, of which the last reads its carries off `W[t]`.
const SCHEDULE_CARRIES: usize = 2;

/// `x >>> n` on a word stored least significant bit first.
const fn rotr_index(i: usize, n: usize) -> usize {
    (i + n) % 32
}
