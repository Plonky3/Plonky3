//! The proof of one circuit batch.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::lincheck::LincheckProof;
use crate::opening::{Commitment, OpeningProof};
use crate::zerocheck::ZerocheckProof;

/// A proof that a committed witness satisfies every instance of a batch's circuit and holds its public port words.
///
/// The proof is not zero-knowledge.
#[derive(Clone, Serialize, Deserialize)]
pub struct FlockProof {
    /// Commitment to the packed witness.
    pub commitment: Commitment,
    /// Zerocheck of `A z o B z = z`.
    pub zerocheck: ZerocheckProof,
    /// Reduction of the zerocheck claims and the public words to one witness claim.
    pub lincheck: LincheckProof,
    /// Opening of that claim.
    pub opening: OpeningProof,
}

impl fmt::Debug for FlockProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The opening proof has no `Debug` form.
        f.debug_struct("FlockProof")
            .field("commitment", &self.commitment)
            .field("zerocheck", &self.zerocheck)
            .field("lincheck", &self.lincheck)
            .finish_non_exhaustive()
    }
}
