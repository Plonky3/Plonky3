use alloc::vec;
use alloc::vec::Vec;

use crate::Target;

/// Trait for extracting commitment targets for Fiat-Shamir observation.
///
/// This trait allows commitments to be observed in the Fiat-Shamir transcript
/// without the verifier needing to know the specific commitment structure.
pub trait ObservableCommitment {
    /// Extract the targets that should be observed in the Fiat-Shamir transcript.
    ///
    /// These targets represent the commitment in a form suitable for hashing
    /// into the challenger state.
    ///
    /// # Returns
    /// A vector of targets representing the commitment
    fn to_observation_targets(&self) -> Vec<Target>;
}

/// Implementation for a single target (used in tests where commitment is a placeholder).
impl ObservableCommitment for Target {
    fn to_observation_targets(&self) -> Vec<Target> {
        vec![*self]
    }
}
