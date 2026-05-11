//! Recursive proof verification for Plonky3 STARKs.

#![no_std]

extern crate alloc;

pub mod backend;
pub mod challenger;
pub mod challenger_perm;
pub mod generation;
pub mod pcs;
pub mod prelude;
pub mod public_inputs;
pub mod recursion;
pub mod traits;
pub mod types;
pub mod verifier;

/// Implement for your StarkConfig (or a wrapper holding FRI verifier params) to use [`FriRecursionBackend`].
pub use backend::fri::FriRecursionConfig;
/// FRI PCS backend for the unified recursion API. Use with [`prove_next_layer`] and a config implementing [`FriRecursionConfig`].
pub use backend::{FriRecursionBackend, FriRecursionBackendD5, FriRecursionBackendForExt};
/// WHIR backend state for native recursive WHIR verifier circuits.
pub use backend::{WhirRecursionBackend, WhirRecursionBackendForExt};
pub use challenger::CircuitChallenger;
pub use challenger_perm::ChallengerPermConfig;
pub use generation::{GenerationError, PcsGeneration, generate_batch_challenges};
pub use p3_circuit::ops;
pub use p3_circuit::ops::Poseidon2Config;
pub use pcs::fri::FriVerifierParams;
pub use pcs::whir::{
    WhirBatchedInitialOracleTargets, WhirConstraintTargets, WhirEqStatementTargets,
    WhirMultilinearRecursionConfig, WhirOpeningClaimTargets, WhirParsedCommitmentTargets,
    WhirProofTargets, WhirProofVerificationInput, WhirProofVerificationTargets,
    WhirQueryOpeningTargets, WhirRecursionCompatibility, WhirRecursiveVerifierParams,
    WhirRoundProofTargets, WhirSelectStatementTargets, WhirSumcheckDataTargets,
    eval_eq_poly_circuit, eval_prefix_constraints_poly_circuit, eval_select_poly_circuit,
    expand_from_univariate_circuit, extrapolate_01inf_circuit,
    observe_whir_commitment_and_ood_circuit, verify_native_whir_proof_circuit,
    verify_whir_final_sumcheck_rounds_circuit, verify_whir_sumcheck_rounds_circuit,
};
pub use public_inputs::{
    BatchStarkVerifierInputsBuilder, CommitmentOpening, FriVerifierInputs, PublicInputBuilder,
    StarkVerifierInputs, StarkVerifierInputsBuilder, construct_batch_stark_verifier_inputs,
};
/// Unified recursion API: single entry point for proving the next layer over a uni-stark or batch-stark proof.
pub use recursion::{
    AggregationCircuitFingerprint, AggregationPrepCache, BatchOnly, NextLayerPrepCache,
    PcsRecursionBackend, ProveNextLayerParams, RecursionInput, RecursionOutput,
    VerifierCircuitResult, build_and_prove_aggregation_layer, build_and_prove_next_layer,
    build_next_layer_circuit, build_next_layer_prep, prove_aggregation_layer, prove_next_layer,
};
pub use traits::{
    Recursive, RecursiveAir, RecursiveChallenger, RecursiveExtensionMmcs, RecursiveMmcs,
    RecursivePcs,
};
pub use types::{
    BatchProofTargets, CommitmentTargets, CommonDataTargets, OpenedValuesTargets, ProofTargets,
    RecursiveLagrangeSelectors, StarkChallenges, Target,
};
pub use verifier::{
    ObservableCommitment, VerificationError, verify_batch_circuit, verify_p3_uni_proof_circuit,
};
