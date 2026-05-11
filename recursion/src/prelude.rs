//! Prelude module for common imports.
//!
//! This module re-exports the most commonly used items from the crate,
//! allowing users to write:
//!
//! ```ignore
//! use p3_recursion::prelude::*;
//! ```
//!
//! Instead of importing each item individually.

pub use crate::Target;
pub use crate::backend::{WhirRecursionBackend, WhirRecursionBackendForExt};
pub use crate::challenger::CircuitChallenger;
pub use crate::generation::{GenerationError, PcsGeneration};
pub use crate::pcs::fri::FriVerifierParams;
pub use crate::pcs::whir::{
    WhirBatchedInitialOracleTargets, WhirConstraintTargets, WhirEqStatementTargets,
    WhirMultilinearRecursionConfig, WhirOpeningClaimTargets, WhirParsedCommitmentTargets,
    WhirProofTargets, WhirProofVerificationInput, WhirProofVerificationTargets,
    WhirQueryOpeningTargets, WhirRecursionCompatibility, WhirRecursiveVerifierParams,
    WhirRoundProofTargets, WhirSelectStatementTargets, WhirSumcheckDataTargets,
    eval_eq_poly_circuit, eval_prefix_constraints_poly_circuit, eval_select_poly_circuit,
    expand_from_univariate_circuit, extrapolate_01inf_circuit,
    observe_whir_commitment_and_ood_circuit, verify_whir_final_sumcheck_rounds_circuit,
    verify_whir_sumcheck_rounds_circuit,
};
pub use crate::public_inputs::{
    CommitmentOpening, FriVerifierInputs, PublicInputBuilder, StarkVerifierInputs,
    StarkVerifierInputsBuilder,
};
pub use crate::traits::{
    ComsWithOpeningsTargets, Recursive, RecursiveAir, RecursiveChallenger, RecursiveExtensionMmcs,
    RecursiveMmcs, RecursivePcs,
};
pub use crate::types::{
    CommitmentTargets, OpenedValuesTargets, ProofTargets, RecursiveLagrangeSelectors,
    StarkChallenges,
};
pub use crate::verifier::{ObservableCommitment, VerificationError, verify_p3_uni_proof_circuit};
