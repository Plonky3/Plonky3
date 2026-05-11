//! WHIR integration boundary for recursive verification.
//!
//! The recursive STARK framework copied from `Plonky3-recursion` is built
//! around [`p3_commit::Pcs`], because `p3-uni-stark` commits univariate trace,
//! quotient, and preprocessing polynomials over two-adic cosets. The WHIR crate
//! in this branch exposes WHIR as [`p3_commit::MultilinearPcs`]: it commits
//! multilinear evaluations and verifies multilinear opening claims.
//!
//! This module keeps that distinction explicit. FRI remains the backend for the
//! copied generic uni-STARK recursion path. Native WHIR recursion is a separate
//! circuit shape: it verifies `p3_whir::pcs::WhirProof` objects directly instead
//! of adapting those proofs through the FRI-specific recursive PCS traits.

use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_whir::parameters::WhirConfig;

mod targets;
mod verifier;

pub use targets::{
    WhirBatchedInitialOracleTargets, WhirOpeningClaimTargets, WhirProofTargets,
    WhirProofVerificationInput, WhirProofVerificationTargets, WhirQueryOpeningTargets,
    WhirRoundProofTargets, WhirSumcheckDataTargets,
};
pub use verifier::{
    WhirConstraintTargets, WhirEqStatementTargets, WhirParsedCommitmentTargets,
    WhirSelectStatementTargets, eval_eq_poly_circuit, eval_multilinear_poly_circuit,
    eval_prefix_constraints_poly_circuit, eval_select_poly_circuit, expand_from_univariate_circuit,
    extrapolate_01inf_circuit, observe_whir_commitment_and_ood_circuit,
    observe_whir_domain_separator_circuit, verify_native_whir_proof_circuit,
    verify_whir_final_sumcheck_rounds_circuit, verify_whir_sumcheck_rounds_circuit,
};

/// Public verifier parameters needed to replay a WHIR multilinear verifier in a
/// recursive circuit.
///
/// The native WHIR parameter object also stores the concrete MMCS instance.
/// Recursive verification should not allocate that object directly in-circuit;
/// it only needs the transcript-bound shape parameters and the commitment/hash
/// gadgets supplied by the recursion backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirRecursiveVerifierParams {
    /// Number of variables in the committed multilinear polynomial.
    pub num_variables: usize,
    /// Logarithm of the inverse initial Reed-Solomon rate.
    pub starting_log_inv_rate: usize,
    /// PoW bits for the initial folding sumcheck.
    pub starting_folding_pow_bits: usize,
    /// Variables folded by each WHIR round.
    pub folding_factors: Vec<usize>,
    /// Number of STIR queries in each intermediate round.
    pub round_queries: Vec<usize>,
    /// OOD samples in each intermediate round.
    pub round_ood_samples: Vec<usize>,
    /// Variables remaining in each intermediate round.
    pub round_num_variables: Vec<usize>,
    /// Evaluation-domain size before each intermediate fold.
    pub round_domain_sizes: Vec<usize>,
    /// STIR PoW bits in each intermediate round.
    pub round_pow_bits: Vec<usize>,
    /// Folding-sumcheck PoW bits in each intermediate round.
    pub round_folding_pow_bits: Vec<usize>,
    /// Number of final STIR queries.
    pub final_queries: usize,
    /// PoW bits for the final STIR query phase.
    pub final_pow_bits: usize,
    /// Number of direct final sumcheck rounds.
    pub final_sumcheck_rounds: usize,
    /// PoW bits for the direct final sumcheck.
    pub final_folding_pow_bits: usize,
    /// Commitment-phase out-of-domain samples.
    pub commitment_ood_samples: usize,
}

impl WhirRecursiveVerifierParams {
    /// Extract the recursive verifier schedule from native WHIR parameters.
    ///
    /// This should be preferred over manually constructing the struct. WHIR's
    /// soundness bounds depend on the exact OOD/query/PoW schedule derived by
    /// `WhirConfig::new`; replaying that schedule in the circuit keeps the
    /// recursive verifier bound to the same transcript as the native verifier.
    pub fn from_whir_config<EF, F, MT, Challenger>(
        config: &WhirConfig<EF, F, MT, Challenger>,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let n_rounds = config.round_parameters.len();

        Self {
            num_variables: config.num_variables,
            starting_log_inv_rate: config.starting_log_inv_rate,
            starting_folding_pow_bits: config.starting_folding_pow_bits,
            folding_factors: (0..=n_rounds)
                .map(|round| config.folding_factor.at_round(round))
                .collect(),
            round_queries: config
                .round_parameters
                .iter()
                .map(|round| round.num_queries)
                .collect(),
            round_ood_samples: config
                .round_parameters
                .iter()
                .map(|round| round.ood_samples)
                .collect(),
            round_num_variables: config
                .round_parameters
                .iter()
                .map(|round| round.num_variables)
                .collect(),
            round_domain_sizes: config
                .round_parameters
                .iter()
                .map(|round| round.domain_size)
                .collect(),
            round_pow_bits: config
                .round_parameters
                .iter()
                .map(|round| round.pow_bits)
                .collect(),
            round_folding_pow_bits: config
                .round_parameters
                .iter()
                .map(|round| round.folding_pow_bits)
                .collect(),
            final_queries: config.final_queries,
            final_pow_bits: config.final_pow_bits,
            final_sumcheck_rounds: config.final_sumcheck_rounds,
            final_folding_pow_bits: config.final_folding_pow_bits,
            commitment_ood_samples: config.commitment_ood_samples,
        }
    }
}

/// Configuration trait for recursive verifiers that consume native WHIR
/// multilinear proofs.
///
/// This is intentionally separate from `FriRecursionConfig`. The copied
/// uni-STARK recursion code is parameterized by a univariate `Pcs`, while the
/// native WHIR path verifies multilinear `WhirProof` transcripts.
pub trait WhirMultilinearRecursionConfig: Clone {
    /// Base field of the committed multilinear messages.
    type Val: Field;

    /// Extension field used by WHIR challenges.
    type Challenge: ExtensionField<Self::Val>;

    /// Fiat-Shamir challenger used by the native WHIR proof.
    type Challenger: FieldChallenger<Self::Val>;

    /// Native WHIR PCS instance.
    type WhirPcs: MultilinearPcs<Self::Challenge, Self::Challenger, Val = Self::Val>;

    /// Return the native WHIR PCS used by this recursion layer.
    fn whir_pcs(&self) -> &Self::WhirPcs;

    /// Return the recursive verifier parameters extracted from the WHIR config.
    fn whir_verifier_params(&self) -> &WhirRecursiveVerifierParams;

    /// Initialize a fresh challenger for replaying the WHIR transcript.
    fn initialise_whir_challenger(&self) -> Self::Challenger;
}

/// Compatibility classification between the copied recursive STARK framework
/// and the WHIR PCS available in this repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhirRecursionCompatibility {
    /// The backend verifies a native multilinear WHIR proof.
    NativeMultilinearWhirProof,
    /// A future adapter implements the univariate `p3_commit::Pcs` trait for
    /// WHIR, allowing direct use in the generic uni-STARK recursion path.
    UnivariateStarkPcsAdapter,
}
