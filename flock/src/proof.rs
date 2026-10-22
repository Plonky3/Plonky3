//! The proof of one circuit batch.

use core::fmt;

use p3_keccak::Keccak256Hash;
use serde::{Deserialize, Serialize};

use crate::config::FlockHash;
use crate::opening::{Commitment, OpeningProof};
use crate::piop::PiopProof;

/// A proof that a committed witness satisfies every instance of a batch's circuit and holds its public port words.
///
/// The proof is not zero-knowledge.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FlockProof<H: FlockHash = Keccak256Hash> {
    /// Commitment to the packed witness.
    pub commitment: Commitment,
    /// The reductions to one witness claim.
    pub piop: PiopProof,
    /// Opening of that claim.
    pub opening: OpeningProof<H>,
}

impl<H: FlockHash> fmt::Debug for FlockProof<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The opening proof has no `Debug` form.
        f.debug_struct("FlockProof")
            .field("commitment", &self.commitment)
            .field("piop", &self.piop)
            .finish_non_exhaustive()
    }
}
