//! Core type definitions for recursive verification.

mod challenges;
mod proof;
mod selectors;

pub(crate) use challenges::StarkChallengeParams;
pub use challenges::StarkChallenges;
pub use proof::{
    BatchProofTargets, CommitmentTargets, CommonDataTargets, OpenedValuesTargets,
    OpenedValuesTargetsWithLookups, ProofTargets,
};
pub use selectors::RecursiveLagrangeSelectors;

/// Canonical circuit target type used across recursive components.
///
/// This is an alias representing a node in the circuit expression graph.
pub type Target = p3_circuit::ExprId;
