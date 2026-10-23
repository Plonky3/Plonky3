//! Signatures.

use alloc::vec::Vec;

/// A signature for one epoch: the encoding randomness, one value per chain, and the Merkle path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature<D, R> {
    /// The randomness under which the message's digits reach the target sum.
    pub randomness: R,
    /// The value of each chain at the position its digit names.
    pub chains: Vec<D>,
    /// The siblings from the epoch's leaf up to the root.
    pub path: Vec<D>,
}
