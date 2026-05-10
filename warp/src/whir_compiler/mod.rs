//! Native WARP-to-WHIR compiler scaffolding.
//!
//! This module is the RS-only compiler boundary described in
//! `warp/docs/native-whir-compiler.md`. It starts at the algebraic statement
//! level: WARP evaluation obligations are converted into WHIR linear
//! Sigma-IOP constraints rather than into PCS opening calls.
//!
//! The main invariant is that WARP and WHIR refer to the same Reed-Solomon
//! oracle. WARP's root IOP records obligations such as "open `u = C(w)` at
//! index `i`" or "open the accumulator MLE at `alpha`". This compiler rewrites
//! those obligations as WHIR linear-Sigma constraints over the committed RS
//! message whenever possible:
//!
//! - fresh base WARP inputs use `w = C^{-1}(u)` directly, so WHIR does not
//!   re-encode the already encoded codeword `u`;
//! - accumulator codeword-MLE claims use the adjoint weights of the same RS
//!   encoder;
//! - all touched oracle commitments are absorbed before reduction challenges
//!   are sampled.
//!
//! The proof-system half of this module proves the whole root IOP with one
//! compact batched WHIR opening. Older codeword-domain and limb fallback paths
//! were removed so every root proof uses the same WARP/WHIR RS code.
//!
//! Soundness note: "one compact batched WHIR opening" means one WHIR proof
//! object after WARP has batched its claims. That proof still executes WHIR's
//! ordinary constrained-RS protocol internally: initial folding, every
//! configured intermediate STIR/proximity round, OOD/query-combination checks,
//! and the final folding phase. The WARP root compiler relies on those WHIR
//! round-by-round errors for proximity/opening soundness; the WARP-specific
//! sumcheck only reduces the recorded `VACC`/`DACC` claims to the grouped
//! residual statement proved by WHIR.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use p3_whir::constraints::statement::{
    BatchedLinearSigmaOpeningClaim, BatchedLinearSigmaReductionProof, EqStatement,
    LinearSigmaConstraint, LinearSigmaReductionError, LinearSigmaReductionProof,
    LinearSigmaStatement,
};
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::verifier::errors::VerifierError as WhirVerifierError;
use p3_whir::pcs::{
    WhirBatchedDeferredProverOracle, WhirBatchedDeferredVerifierOracle, WhirDeferredProverData,
    WhirExtensionDeferredProverData, WhirPcs, WhirSharedBaseDeferredProverData,
};
use p3_whir::sumcheck::lagrange::extrapolate_01inf;
use p3_whir::sumcheck::strategy::VariableOrder;
use p3_whir::sumcheck::{SumcheckData, SumcheckError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::code::ReedSolomonCode;
use crate::root_iop::{
    RootIopBoundCommitment, RootIopBoundTranscript, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleField, RootIopOracleValues,
};

mod domain;

mod types;
pub use types::{
    NativeWarpWhirClaimCompileError, NativeWarpWhirRootBaseProverData,
    NativeWarpWhirRootBatchedOpeningProof, NativeWarpWhirRootCommitment,
    NativeWarpWhirRootExtensionProverData, NativeWarpWhirRootOracleProverData,
    NativeWarpWhirRootProof, NativeWarpWhirRootProofError, NativeWarpWhirRootProverData,
    NativeWarpWhirRootReductionError, NativeWarpWhirRootSharedBaseProverData,
};

/// Native WARP root proof system using WHIR over the WARP RS message.
pub struct NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// WHIR PCS configured for the RS-message-domain openings.
    message_pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    /// Algebraic compiler from WARP root claims to WHIR linear-Sigma claims.
    compiler: NativeWarpWhirCompiler<'a, F, Dft>,
    /// Seed challenger used to derive independent role-separated WHIR transcripts.
    challenger_seed: Challenger,
}

mod statement;
pub use statement::{NativeWarpWhirEvalClaim, NativeWarpWhirOracleStatement};

/// Compiler helper for WARP over Plonky3's Reed-Solomon code.
///
/// WARP's RS specialization and WHIR's initial oracle are both statements
/// about one smooth Reed-Solomon code. This compiler therefore works in the
/// message coordinates of `C^{-1}` and uses [`ReedSolomonCode`] to express
/// every WARP codeword query as an RS query over that same initial polynomial.
/// In coefficient layout those weights are WHIR select/monomial weights; in
/// systematic layout they are the corresponding Lagrange weights on the
/// message subgroup. No second code is introduced.
pub struct NativeWarpWhirCompiler<'a, F, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    code: &'a ReedSolomonCode<F, Dft>,
}

impl<'a, F, Dft> NativeWarpWhirCompiler<'a, F, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a compiler for one WARP RS code.
    pub const fn new(code: &'a ReedSolomonCode<F, Dft>) -> Self {
        Self { code }
    }

    /// Return the RS code this compiler targets.
    pub const fn code(&self) -> &'a ReedSolomonCode<F, Dft> {
        self.code
    }

    /// Convert a folded-codeword evaluation claim into a WHIR linear-Sigma
    /// constraint.
    ///
    /// This is the WHIR paper's basic evaluation-as-Sigma-query identity:
    /// `f_hat(z) = v` is
    /// `sum_b eq(z, b) * f_hat(b) = v`.
    ///
    /// # Panics
    ///
    /// Panics if `claim.point` is not a point in the codeword hypercube.
    pub fn eval_claim_constraint<EF>(
        &self,
        claim: &NativeWarpWhirEvalClaim<EF>,
    ) -> LinearSigmaConstraint<EF>
    where
        EF: ExtensionField<F>,
    {
        assert_eq!(
            claim.point.num_variables(),
            self.code.log_codeword_len(),
            "WARP/WHIR evaluation point must have log_n variables",
        );
        let mut eq = EqStatement::initialize(self.code.log_codeword_len());
        eq.add_evaluated_constraint(claim.point.clone(), claim.value);
        LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE)
    }

    /// Convert multiple folded-codeword evaluation claims into separate WHIR
    /// linear-Sigma constraints.
    ///
    /// WHIR's multi-constrained RS layer is responsible for the later random
    /// batching. Keeping these claims separate here preserves the binding
    /// point required by Construction 5.5.
    pub fn eval_claim_statement<EF>(
        &self,
        claims: &[NativeWarpWhirEvalClaim<EF>],
    ) -> NativeWarpWhirOracleStatement<EF>
    where
        EF: ExtensionField<F>,
    {
        let mut statement = LinearSigmaStatement::initialize(self.code.log_codeword_len());
        for claim in claims {
            statement.add_constraint(self.eval_claim_constraint(claim));
        }
        NativeWarpWhirOracleStatement::new(statement)
    }

    /// Convert a systematic witness-MLE evaluation claim into a codeword
    /// linear-Sigma constraint.
    ///
    /// In systematic mode, the witness/message MLE point `y` is lifted to the
    /// codeword point `(y, 0, ..., 0)`. This is the bridge needed for WARP's
    /// final `Pb(beta, C^{-1}(f)) = eta` constraint.
    ///
    /// # Panics
    ///
    /// Panics if the RS code is not systematic or if `message_point` has the
    /// wrong arity.
    pub fn systematic_message_eval_constraint<EF>(
        &self,
        message_point: &[EF],
        value: EF,
    ) -> LinearSigmaConstraint<EF>
    where
        EF: ExtensionField<F>,
    {
        let point = self.code.systematic_message_point(message_point);
        self.eval_claim_constraint(&NativeWarpWhirEvalClaim { point, value })
    }

    fn check_claim_oracles_bound<EF, Comm>(
        &self,
        oracles: &[(RootIopBoundCommitment<Comm>, RootIopOracleValues<F, EF>)],
        claims: &[RootIopOpeningClaim<F, EF>],
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        for claim in claims {
            if !oracles
                .iter()
                .any(|(commitment, _)| commitment.oracle_id == claim.oracle_id)
            {
                return Err(NativeWarpWhirRootReductionError::UnknownOracle(
                    claim.oracle_id,
                ));
            }
        }
        Ok(())
    }

    fn check_unique_bound_oracle_ids<EF, Comm>(
        &self,
        oracles: &[(RootIopBoundCommitment<Comm>, RootIopOracleValues<F, EF>)],
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        let mut seen = Vec::new();
        for (commitment, _) in oracles {
            if seen.contains(&commitment.oracle_id) {
                return Err(NativeWarpWhirRootReductionError::DuplicateOracle(
                    commitment.oracle_id,
                ));
            }
            seen.push(commitment.oracle_id);
        }
        Ok(())
    }

    fn check_unique_public_oracle_ids<Comm>(
        &self,
        commitments: &[RootIopBoundCommitment<Comm>],
    ) -> Result<(), NativeWarpWhirRootReductionError> {
        let mut seen = Vec::new();
        for commitment in commitments {
            if seen.contains(&commitment.oracle_id) {
                return Err(NativeWarpWhirRootReductionError::DuplicateOracle(
                    commitment.oracle_id,
                ));
            }
            seen.push(commitment.oracle_id);
        }
        Ok(())
    }

    fn check_claim_oracles_public<EF, Comm>(
        &self,
        commitments: &[RootIopBoundCommitment<Comm>],
        claims: &[RootIopOpeningClaim<F, EF>],
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        for claim in claims {
            if !commitments
                .iter()
                .any(|commitment| commitment.oracle_id == claim.oracle_id)
            {
                return Err(NativeWarpWhirRootReductionError::UnknownOracle(
                    claim.oracle_id,
                ));
            }
        }
        Ok(())
    }

    fn check_bound_oracle_shape<EF, Comm>(
        &self,
        commitment: &RootIopBoundCommitment<Comm>,
        values: Option<&RootIopOracleValues<F, EF>>,
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        if commitment.log_len != self.code.log_codeword_len() {
            return Err(NativeWarpWhirRootReductionError::OracleLogLengthMismatch {
                oracle_id: commitment.oracle_id,
                expected: self.code.log_codeword_len(),
                actual: commitment.log_len,
            });
        }

        match values {
            Some(RootIopOracleValues::Base(values)) => {
                if commitment.field != RootIopOracleField::Base {
                    return Err(NativeWarpWhirRootReductionError::OracleValueFieldMismatch(
                        commitment.oracle_id,
                    ));
                }
                if values.len() != self.code.codeword_len() {
                    return Err(
                        NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                            oracle_id: commitment.oracle_id,
                            expected: self.code.codeword_len(),
                            actual: values.len(),
                        },
                    );
                }
            }
            Some(RootIopOracleValues::Extension(values)) => {
                if commitment.field != RootIopOracleField::Extension {
                    return Err(NativeWarpWhirRootReductionError::OracleValueFieldMismatch(
                        commitment.oracle_id,
                    ));
                }
                if values.len() != self.code.codeword_len() {
                    return Err(
                        NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                            oracle_id: commitment.oracle_id,
                            expected: self.code.codeword_len(),
                            actual: values.len(),
                        },
                    );
                }
            }
            None => {}
        }

        Ok(())
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a native WARP root proof system.
    ///
    /// `message_pcs` must be configured with `code.log_msg_len()` variables
    /// and the same RS rate/security settings as the WARP code. WHIR commits to
    /// `C^{-1}(u)`, and WARP codeword openings are compiled using the same
    /// [`ReedSolomonCode`] generator.
    pub fn new(
        message_pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        code: &'a ReedSolomonCode<F, Dft>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            message_pcs,
            compiler: NativeWarpWhirCompiler::new(code),
            challenger_seed,
        }
    }

    /// Commit a base-field fresh WARP input without RS double-encoding.
    ///
    /// The WARP verifier still sees and checks openings of `codeword = C(w)`,
    /// but the WHIR commitment is to `w = C^{-1}(codeword)`. During proof
    /// generation those codeword-index claims are transformed into
    /// constrained-RS claims over the same message coordinates by
    /// [`ReedSolomonCode::codeword_index_weights`].
    pub fn commit_base_message_oracle(
        &self,
        oracle_id: usize,
        codeword: Vec<F>,
        message: Vec<F>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if codeword.len() != self.compiler.code().codeword_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().codeword_len(),
                    actual: codeword.len(),
                }
                .into(),
            );
        }
        if message.len() != self.compiler.code().msg_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().msg_len(),
                    actual: message.len(),
                }
                .into(),
            );
        }
        let mut challenger = self.base_oracle_challenger(oracle_id);
        let (commitment, prover_data) = self
            .message_pcs
            .commit_deferred(RowMajorMatrix::new(message.clone(), 1), &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Base,
                commitment: NativeWarpWhirRootCommitment::BaseMessage(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::Base(NativeWarpWhirRootBaseProverData {
                    prover_data,
                    challenger,
                    message,
                }),
            },
        ))
    }

    /// Commit several base-field fresh WARP inputs under one WHIR/MMCS root.
    ///
    /// Each returned root-IOP commitment carries the same Merkle root plus a
    /// distinct column index. The column index is part of the Fiat-Shamir
    /// payload, so swapping columns changes the transcript.
    pub fn commit_shared_base_message_oracles(
        &self,
        inputs: Vec<(usize, Vec<F>, Vec<F>)>,
    ) -> Result<
        Vec<(
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        )>,
        NativeWarpWhirRootProofError,
    > {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let width = inputs.len();
        let mut matrices = Vec::with_capacity(width);
        for (oracle_id, codeword, message) in &inputs {
            if codeword.len() != self.compiler.code().codeword_len() {
                return Err(
                    NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                        oracle_id: *oracle_id,
                        expected: self.compiler.code().codeword_len(),
                        actual: codeword.len(),
                    }
                    .into(),
                );
            }
            if message.len() != self.compiler.code().msg_len() {
                return Err(
                    NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                        oracle_id: *oracle_id,
                        expected: self.compiler.code().msg_len(),
                        actual: message.len(),
                    }
                    .into(),
                );
            }
            matrices.push(RowMajorMatrix::new(message.clone(), 1));
        }

        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(domain::ROOT_WHIR_BASE_ORACLE));
        challenger.observe(F::from_usize(width));
        for (oracle_id, _, _) in &inputs {
            challenger.observe(F::from_usize(*oracle_id));
        }
        // This commitment is shared across columns. The root transcript binds
        // the role tag, width, and ordered oracle ids before WHIR samples any
        // commitment-dependent randomness, and each public commitment below
        // carries `(root, column, width)` so columns cannot be swapped.
        let encoded = self.message_pcs.encode_base_batch_initial_oracles(matrices);
        let (root, shared) = self
            .message_pcs
            .commit_base_batch_encoded_deferred(encoded, &mut challenger);

        let mut out = Vec::with_capacity(width);
        for (column, (oracle_id, _codeword, message)) in inputs.into_iter().enumerate() {
            let commitment = RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Base,
                commitment: NativeWarpWhirRootCommitment::BaseMessageShared {
                    root: root.clone(),
                    column,
                    width,
                },
            };
            let prover_data = NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::BaseShared(
                    NativeWarpWhirRootSharedBaseProverData {
                        shared: shared.clone(),
                        column,
                        width,
                        message,
                    },
                ),
            };
            out.push((commitment, prover_data));
        }

        Ok(out)
    }

    /// Commit an extension-field accumulator codeword for the WARP root IOP.
    ///
    /// The codeword is decoded back to the RS message before commitment, so
    /// this helper keeps the single-RS invariant even when callers only have
    /// the current accumulator codeword available.
    pub fn commit_extension_oracle(
        &self,
        oracle_id: usize,
        codeword: Vec<EF>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if codeword.len() != self.compiler.code().codeword_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().codeword_len(),
                    actual: codeword.len(),
                }
                .into(),
            );
        }

        let challenger = self.base_oracle_challenger(oracle_id);
        let message = self.compiler.code().message_from_codeword(&codeword);
        self.commit_extension_message_oracle_with_challenger(oracle_id, message, challenger)
    }

    /// Commit an extension-field accumulator through WHIR's initial-message
    /// oracle path when the RS message is already available.
    ///
    /// This commits to the same WHIR polynomial as [`commit_extension_oracle`]
    /// would after `message_from_codeword`, but avoids decoding/extracting the
    /// message again in pipelines that already have the merged WARP witness.
    pub fn commit_extension_message_oracle(
        &self,
        oracle_id: usize,
        message: Vec<EF>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        let challenger = self.base_oracle_challenger(oracle_id);
        self.commit_extension_message_oracle_with_challenger(oracle_id, message, challenger)
    }

    fn commit_extension_message_oracle_with_challenger(
        &self,
        oracle_id: usize,
        message: Vec<EF>,
        mut challenger: Challenger,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if message.len() != self.compiler.code().msg_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().msg_len(),
                    actual: message.len(),
                }
                .into(),
            );
        }

        let encoded = self
            .message_pcs
            .encode_extension_initial_oracle(RowMajorMatrix::new(message.clone(), 1));
        let (commitment, prover_data) = self
            .message_pcs
            .commit_extension_encoded_deferred(encoded, &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Extension,
                commitment: NativeWarpWhirRootCommitment::ExtensionMessage(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::ExtensionMessage(
                    NativeWarpWhirRootExtensionProverData {
                        prover_data,
                        challenger,
                        message,
                    },
                ),
            },
        ))
    }

    /// Prove the recorded WARP root IOP with one WHIR batched opening.
    pub fn prove(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<NativeWarpWhirRootProof<F, EF, MT>, NativeWarpWhirRootProofError> {
        let opening = self.prove_direct_batched_root(
            transcript,
            prover_data,
            challenger,
            reduction_pow_bits,
        )?;
        Ok(NativeWarpWhirRootProof { opening })
    }

    /// Verify the recorded WARP root IOP with one WHIR batched opening.
    pub fn verify(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        self.verify_direct_batched_root(
            expected_commitments,
            expected_claims,
            &proof.opening,
            challenger,
            reduction_pow_bits,
        )
    }

    fn prove_direct_batched_root(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>, NativeWarpWhirRootProofError>
    {
        self.compiler
            .check_unique_bound_oracle_ids(&transcript.oracles)?;
        self.compiler
            .check_claim_oracles_bound(&transcript.oracles, &transcript.claims)?;

        let mut commitments_to_observe = Vec::new();
        let mut statements = Vec::new();
        let mut polys = Vec::new();
        let mut whir_oracles = Vec::new();
        for (commitment, values) in &transcript.oracles {
            if !claims_include_oracle(&transcript.claims, commitment.oracle_id) {
                continue;
            }
            self.compiler
                .check_bound_oracle_shape::<EF, _>(commitment, None)?;

            match (&commitment.commitment, values) {
                (NativeWarpWhirRootCommitment::BaseMessage(_), RootIopOracleValues::Base(_)) => {
                    let message =
                        self.base_message_for_oracle(prover_data, commitment.oracle_id)?;
                    if message.len() != self.compiler.code().msg_len() {
                        return Err(
                            NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                                oracle_id: commitment.oracle_id,
                                expected: self.compiler.code().msg_len(),
                                actual: message.len(),
                            }
                            .into(),
                        );
                    }
                    let oracle_data = prover_data
                        .iter()
                        .find(|data| data.oracle_id == commitment.oracle_id)
                        .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                            commitment.oracle_id,
                        ))?;
                    let NativeWarpWhirRootProverData::Base(data) = &oracle_data.data else {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    };
                    commitments_to_observe.push(commitment);
                    statements.push(self.compact_base_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?);
                    polys.push(NativeWarpDirectBatchedResidualPoly::Base(message));
                    whir_oracles.push(NativeWarpBatchedResidualProverOracle::Base(
                        data.prover_data.clone(),
                    ));
                }
                (
                    NativeWarpWhirRootCommitment::BaseMessageShared { column, width, .. },
                    RootIopOracleValues::Base(_),
                ) => {
                    let message =
                        self.base_message_for_oracle(prover_data, commitment.oracle_id)?;
                    if message.len() != self.compiler.code().msg_len() {
                        return Err(
                            NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                                oracle_id: commitment.oracle_id,
                                expected: self.compiler.code().msg_len(),
                                actual: message.len(),
                            }
                            .into(),
                        );
                    }
                    let oracle_data = prover_data
                        .iter()
                        .find(|data| data.oracle_id == commitment.oracle_id)
                        .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                            commitment.oracle_id,
                        ))?;
                    let NativeWarpWhirRootProverData::BaseShared(data) = &oracle_data.data else {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    };
                    if data.column != *column || data.width != *width {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    }
                    commitments_to_observe.push(commitment);
                    statements.push(self.compact_base_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?);
                    polys.push(NativeWarpDirectBatchedResidualPoly::Base(message));
                    whir_oracles.push(NativeWarpBatchedResidualProverOracle::SharedBase {
                        shared: data.shared.clone(),
                        column: *column,
                        width: *width,
                    });
                }
                (
                    NativeWarpWhirRootCommitment::ExtensionMessage(_),
                    RootIopOracleValues::Extension(_),
                ) => {
                    let message =
                        self.extension_message_for_oracle(prover_data, commitment.oracle_id)?;
                    if message.len() != self.compiler.code().msg_len() {
                        return Err(
                            NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                                oracle_id: commitment.oracle_id,
                                expected: self.compiler.code().msg_len(),
                                actual: message.len(),
                            }
                            .into(),
                        );
                    }
                    let oracle_data = prover_data
                        .iter()
                        .find(|data| data.oracle_id == commitment.oracle_id)
                        .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                            commitment.oracle_id,
                        ))?;
                    let NativeWarpWhirRootProverData::ExtensionMessage(data) = &oracle_data.data
                    else {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    };
                    commitments_to_observe.push(commitment);
                    statements.push(self.compact_extension_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?);
                    polys.push(NativeWarpDirectBatchedResidualPoly::Extension(message));
                    whir_oracles.push(NativeWarpBatchedResidualProverOracle::Extension(
                        data.prover_data.clone(),
                    ));
                }
                _ => {
                    return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                        commitment.oracle_id,
                    ));
                }
            }
        }

        if statements.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(usize::MAX).into());
        }

        // Fiat-Shamir binding point for the direct path. All public WARP root
        // commitments are absorbed before the batching challenge used by
        // `prove_compact_batched_root_reduction` is sampled.
        for commitment in commitments_to_observe {
            observe_native_root_commitment::<F, Challenger, MT::Commitment>(challenger, commitment);
        }

        // The compact reducer converts all per-oracle WARP constraints into a
        // single opening claim `(point, value, coeffs)` against the grouped
        // WHIR oracles. WHIR then proves that grouped opening against the
        // already committed RS messages.
        let (reduction, opening_claim) = prove_compact_batched_root_reduction::<F, EF, Dft, _>(
            self.compiler.code(),
            &statements,
            &polys,
            challenger,
            reduction_pow_bits,
        )?;
        let opening = self.message_pcs.open_grouped_batched_deferred(
            Self::group_prover_oracles(whir_oracles, &opening_claim.coeffs)?,
            opening_claim.point,
            opening_claim.value,
            challenger,
        )?;

        Ok(NativeWarpWhirRootBatchedOpeningProof { reduction, opening })
    }

    fn verify_direct_batched_root(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        self.compiler
            .check_unique_public_oracle_ids(expected_commitments)?;
        self.compiler
            .check_claim_oracles_public(expected_commitments, expected_claims)?;

        let mut statements = Vec::new();
        let mut commitments = Vec::new();
        for commitment in expected_commitments {
            if !claims_include_oracle(expected_claims, commitment.oracle_id) {
                continue;
            }
            self.compiler
                .check_bound_oracle_shape::<EF, _>(commitment, None)?;

            match &commitment.commitment {
                NativeWarpWhirRootCommitment::BaseMessage(commitment_root) => {
                    observe_native_root_commitment::<F, Challenger, MT::Commitment>(
                        challenger, commitment,
                    );
                    statements.push(self.compact_base_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?);
                    commitments.push(NativeWarpBatchedResidualCommitment::Base(
                        commitment_root.clone(),
                    ));
                }
                NativeWarpWhirRootCommitment::BaseMessageShared {
                    root,
                    column,
                    width,
                } => {
                    observe_native_root_commitment::<F, Challenger, MT::Commitment>(
                        challenger, commitment,
                    );
                    statements.push(self.compact_base_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?);
                    commitments.push(NativeWarpBatchedResidualCommitment::SharedBase {
                        root: root.clone(),
                        column: *column,
                        width: *width,
                    });
                }
                NativeWarpWhirRootCommitment::ExtensionMessage(commitment_root) => {
                    observe_native_root_commitment::<F, Challenger, MT::Commitment>(
                        challenger, commitment,
                    );
                    statements.push(self.compact_extension_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?);
                    commitments.push(NativeWarpBatchedResidualCommitment::Extension(
                        commitment_root.clone(),
                    ));
                }
            }
        }

        // Same binding order as the prover: for each touched oracle, absorb
        // the WARP metadata plus the WHIR commitment before deriving the
        // compact batching challenges.
        let opening_claim = verify_compact_batched_root_reduction::<F, EF, Dft, _>(
            self.compiler.code(),
            &statements,
            &proof.reduction,
            challenger,
            reduction_pow_bits,
        )?;
        let whir_oracles = Self::group_verifier_oracles(commitments, &opening_claim.coeffs)?;
        self.message_pcs
            .verify_batched_deferred(
                &whir_oracles,
                opening_claim.point,
                opening_claim.value,
                &proof.opening,
                challenger,
            )
            .map_err(NativeWarpWhirRootProofError::BatchedOpening)
    }

    fn group_prover_oracles(
        oracles: Vec<NativeWarpBatchedResidualProverOracle<F, EF, MT, DIGEST_ELEMS>>,
        coeffs: &[EF],
    ) -> Result<
        Vec<WhirBatchedDeferredProverOracle<F, EF, MT, DIGEST_ELEMS>>,
        LinearSigmaReductionError,
    > {
        if oracles.len() != coeffs.len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: oracles.len(),
                actual: coeffs.len(),
            });
        }

        let mut grouped = Vec::new();
        for (oracle, &coeff) in oracles.into_iter().zip(coeffs) {
            match oracle {
                NativeWarpBatchedResidualProverOracle::Base(data) => {
                    grouped.push(WhirBatchedDeferredProverOracle::Base { coeff, data });
                }
                NativeWarpBatchedResidualProverOracle::Extension(data) => {
                    grouped.push(WhirBatchedDeferredProverOracle::Extension { coeff, data });
                }
                NativeWarpBatchedResidualProverOracle::SharedBase {
                    shared,
                    column,
                    width,
                } => {
                    if column >= width {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: width,
                            actual: column + 1,
                        });
                    }
                    let mut inserted = false;
                    for existing in &mut grouped {
                        if let WhirBatchedDeferredProverOracle::SharedBase { coeffs, data } =
                            existing
                        {
                            if Arc::ptr_eq(data, &shared) {
                                if coeffs.len() != width {
                                    return Err(LinearSigmaReductionError::ArityMismatch {
                                        expected: width,
                                        actual: coeffs.len(),
                                    });
                                }
                                coeffs[column] += coeff;
                                inserted = true;
                                break;
                            }
                        }
                    }
                    if !inserted {
                        let mut coeffs = EF::zero_vec(width);
                        coeffs[column] = coeff;
                        grouped.push(WhirBatchedDeferredProverOracle::SharedBase {
                            coeffs,
                            data: shared,
                        });
                    }
                }
            }
        }

        Ok(grouped)
    }

    fn group_verifier_oracles(
        commitments: Vec<NativeWarpBatchedResidualCommitment<MT::Commitment>>,
        coeffs: &[EF],
    ) -> Result<Vec<WhirBatchedDeferredVerifierOracle<EF, MT::Commitment>>, LinearSigmaReductionError>
    {
        if commitments.len() != coeffs.len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: commitments.len(),
                actual: coeffs.len(),
            });
        }

        let mut grouped = Vec::new();
        for (commitment, &coeff) in commitments.into_iter().zip(coeffs) {
            match commitment {
                NativeWarpBatchedResidualCommitment::Base(commitment) => {
                    grouped.push(WhirBatchedDeferredVerifierOracle::Base { coeff, commitment });
                }
                NativeWarpBatchedResidualCommitment::Extension(commitment) => {
                    grouped
                        .push(WhirBatchedDeferredVerifierOracle::Extension { coeff, commitment });
                }
                NativeWarpBatchedResidualCommitment::SharedBase {
                    root,
                    column,
                    width,
                } => {
                    if column >= width {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: width,
                            actual: column + 1,
                        });
                    }
                    let mut inserted = false;
                    for existing in &mut grouped {
                        if let WhirBatchedDeferredVerifierOracle::SharedBase {
                            coeffs,
                            commitment,
                        } = existing
                        {
                            if *commitment == root {
                                if coeffs.len() != width {
                                    return Err(LinearSigmaReductionError::ArityMismatch {
                                        expected: width,
                                        actual: coeffs.len(),
                                    });
                                }
                                coeffs[column] += coeff;
                                inserted = true;
                                break;
                            }
                        }
                    }
                    if !inserted {
                        let mut coeffs = EF::zero_vec(width);
                        coeffs[column] = coeff;
                        grouped.push(WhirBatchedDeferredVerifierOracle::SharedBase {
                            coeffs,
                            commitment: root,
                        });
                    }
                }
            }
        }

        Ok(grouped)
    }

    fn compact_base_message_claim_statement(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
    ) -> Result<NativeWarpCompactRootStatement<EF>, NativeWarpWhirClaimCompileError> {
        let mut statement =
            NativeWarpCompactRootStatement::initialize(self.compiler.code().log_msg_len());
        for claim in claims.iter().filter(|claim| claim.oracle_id == oracle_id) {
            let value = match &claim.value {
                RootIopOpeningValue::Base(value) => EF::from(*value),
                _ => {
                    return Err(NativeWarpWhirClaimCompileError::OracleFieldMismatch(
                        oracle_id,
                    ));
                }
            };
            match &claim.point {
                RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
                    if *index >= self.compiler.code().codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    statement.add_index(*index, value);
                }
                RootIopOpeningPoint::Mle(_) => {
                    return Err(NativeWarpWhirClaimCompileError::UnsupportedBaseMle(
                        oracle_id,
                    ));
                }
            }
        }

        if statement.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }
        Ok(statement)
    }

    fn compact_extension_message_claim_statement(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
    ) -> Result<NativeWarpCompactRootStatement<EF>, NativeWarpWhirClaimCompileError> {
        let mut statement =
            NativeWarpCompactRootStatement::initialize(self.compiler.code().log_msg_len());
        for claim in claims.iter().filter(|claim| claim.oracle_id == oracle_id) {
            let value = match &claim.value {
                RootIopOpeningValue::Extension(value) => *value,
                _ => {
                    return Err(NativeWarpWhirClaimCompileError::OracleFieldMismatch(
                        oracle_id,
                    ));
                }
            };
            match &claim.point {
                RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
                    if *index >= self.compiler.code().codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    statement.add_index(*index, value);
                }
                RootIopOpeningPoint::Mle(point) => {
                    if point.len() != self.compiler.code().log_codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::PointArityMismatch {
                            oracle_id,
                        });
                    }
                    statement.add_mle(point.clone(), value);
                }
            }
        }

        if statement.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }
        Ok(statement)
    }

    fn base_message_for_oracle<'b>(
        &self,
        prover_data: &'b [NativeWarpWhirRootOracleProverData<
            F,
            EF,
            MT,
            Challenger,
            DIGEST_ELEMS,
        >],
        oracle_id: usize,
    ) -> Result<&'b [F], NativeWarpWhirRootProofError> {
        let oracle_data = prover_data
            .iter()
            .find(|data| data.oracle_id == oracle_id)
            .ok_or(NativeWarpWhirRootProofError::MissingProverData(oracle_id))?;
        match &oracle_data.data {
            NativeWarpWhirRootProverData::Base(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::BaseShared(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::ExtensionMessage(_) => {
                Err(NativeWarpWhirRootProofError::OracleKindMismatch(oracle_id))
            }
        }
    }

    fn extension_message_for_oracle<'b>(
        &self,
        prover_data: &'b [NativeWarpWhirRootOracleProverData<
            F,
            EF,
            MT,
            Challenger,
            DIGEST_ELEMS,
        >],
        oracle_id: usize,
    ) -> Result<&'b [EF], NativeWarpWhirRootProofError> {
        let oracle_data = prover_data
            .iter()
            .find(|data| data.oracle_id == oracle_id)
            .ok_or(NativeWarpWhirRootProofError::MissingProverData(oracle_id))?;
        match &oracle_data.data {
            NativeWarpWhirRootProverData::ExtensionMessage(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::Base(_) | NativeWarpWhirRootProverData::BaseShared(_) => {
                Err(NativeWarpWhirRootProofError::OracleKindMismatch(oracle_id))
            }
        }
    }

    fn base_oracle_challenger(&self, oracle_id: usize) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(domain::ROOT_WHIR_BASE_ORACLE));
        challenger.observe(F::from_usize(oracle_id));
        challenger
    }
}

mod direct;
use direct::{
    NativeWarpBatchedResidualCommitment, NativeWarpBatchedResidualProverOracle,
    NativeWarpCompactRootStatement, NativeWarpDirectBatchedResidualPoly,
    observe_native_root_commitment, prove_compact_batched_root_reduction,
    verify_compact_batched_root_reduction,
};

/// Build claims from parallel point/value lists.
///
/// # Panics
///
/// Panics if the two slices have different lengths.
pub fn eval_claims_from_parts<EF: Field>(
    points: &[Point<EF>],
    values: &[EF],
) -> Vec<NativeWarpWhirEvalClaim<EF>> {
    assert_eq!(
        points.len(),
        values.len(),
        "WARP/WHIR claim point/value count mismatch",
    );
    points
        .iter()
        .cloned()
        .zip(values.iter().copied())
        .map(|(point, value)| NativeWarpWhirEvalClaim { point, value })
        .collect()
}

fn claims_include_oracle<F, EF>(claims: &[RootIopOpeningClaim<F, EF>], oracle_id: usize) -> bool
where
    F: Field,
    EF: ExtensionField<F>,
{
    claims.iter().any(|claim| claim.oracle_id == oracle_id)
}

#[cfg(test)]
mod tests;
