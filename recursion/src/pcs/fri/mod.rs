//! FRI for recursive verification.

mod params;
mod targets;
mod verifier;

pub use params::FriVerifierParams;
pub use targets::{
    BatchOpeningTargets, CommitPhaseProofStepTargets, FriProofTargets, HashProofTargets,
    HidingFriProofTargets, HidingOpenedValuesTargets, InputProofTargets, MerkleCapTargets,
    QueryProofTargets, RecExtensionValMmcs, RecValMmcs, TwoAdicFriProofTargets, Witness,
};
pub use verifier::verify_fri_circuit;
