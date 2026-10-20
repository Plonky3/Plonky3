//! Points of a batch's quirky extension: one univariate coordinate for the six low bits, then multilinear ones.

use alloc::vec::Vec;

use crate::Challenge;

/// A point: `skip` on the skip domain for the six low bits, `rest` multilinear, most significant first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyPoint {
    /// The univariate coordinate.
    pub skip: Challenge,
    /// The multilinear coordinates, most significant first: instance bits, then inner bits.
    pub rest: Vec<Challenge>,
}
