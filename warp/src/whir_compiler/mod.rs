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
//! The proof-system half of this module then either proves the whole root IOP
//! with one compact batched WHIR opening, or falls back to a two-stage path:
//! first reduce each oracle's linear-Sigma claims to residual openings, then
//! authenticate those residual openings with WHIR.
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
use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearOpenedValues};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use p3_whir::constraints::statement::{
    BatchedLinearSigmaOpeningClaim, BatchedLinearSigmaProverOracle,
    BatchedLinearSigmaReductionProof, EqStatement, LinearSigmaConstraint, LinearSigmaOpeningClaim,
    LinearSigmaReductionError, LinearSigmaReductionProof, LinearSigmaStatement,
    prove_batched_linear_sigma_reduction, verify_batched_linear_sigma_reduction,
};
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::verifier::errors::VerifierError as WhirVerifierError;
use p3_whir::pcs::{
    WhirBatchedDeferredProverData, WhirBatchedDeferredProverOracle,
    WhirBatchedDeferredVerifierOracle, WhirDeferredProverData, WhirExtensionDeferredProverData,
    WhirLinearSigmaError, WhirLinearSigmaProof, WhirPcs, WhirSharedBaseDeferredProverData,
};
use p3_whir::sumcheck::lagrange::extrapolate_01inf;
use p3_whir::sumcheck::strategy::VariableOrder;
use p3_whir::sumcheck::{SumcheckData, SumcheckError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::code::ReedSolomonCode;
use crate::finalize::{
    AccumulatorPointOpeningBackend, ExtensionLimbPcsError, WhirLimbAccumulatorBackend,
    WhirLimbAccumulatorOpeningProof, WhirLimbAccumulatorProverData,
};
use crate::root_iop::{
    RootIopBoundCommitment, RootIopBoundTranscript, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleField, RootIopOracleValues,
};

mod domain;

mod types;
pub use types::{
    NativeWarpWhirClaimCompileError, NativeWarpWhirCompilerError, NativeWarpWhirPointProof,
    NativeWarpWhirRootBaseProverData, NativeWarpWhirRootBatchedOpeningProof,
    NativeWarpWhirRootCommitment, NativeWarpWhirRootExtensionProverData,
    NativeWarpWhirRootOracleOpeningProof, NativeWarpWhirRootOracleProverData,
    NativeWarpWhirRootOracleReductionProof, NativeWarpWhirRootProof, NativeWarpWhirRootProofError,
    NativeWarpWhirRootProverData, NativeWarpWhirRootReductionError,
    NativeWarpWhirRootReductionProof, NativeWarpWhirRootResidualClaim,
    NativeWarpWhirRootSharedBaseProverData,
};

/// Native WARP root proof system using WHIR for every residual opening.
///
/// `pcs` is the legacy codeword-domain WHIR PCS used when an oracle has already
/// been committed as a codeword. `base_message_pcs`, when present, is the
/// preferred single-RS path: it commits to the RS message and lets the compiler
/// express WARP codeword queries as linear claims over that message. The two
/// handles may point to differently sized WHIR instances because codeword
/// oracles have `log_n` variables while message oracles have `log_k`.
pub struct NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// WHIR PCS configured for codeword-domain openings.
    pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    /// Optional WHIR PCS configured for RS-message-domain openings.
    base_message_pcs: Option<&'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>>,
    /// Compatibility backend for extension accumulators committed limb-by-limb.
    limb_backend: WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>,
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

    /// Compile all recorded root-IOP claims for one oracle into a WARP/WHIR
    /// linear-Sigma statement.
    ///
    /// The root IOP recorder is the point where ordinary WARP `VACC` and `DACC`
    /// checks expose their oracle obligations. This method turns those typed
    /// obligations into the WHIR Section 7 linear form:
    ///
    /// - Boolean index openings become MLE evaluation claims at the corresponding
    ///   Boolean point.
    /// - MLE openings are kept as-is.
    /// - Base-field opened values are embedded into `EF`.
    ///
    /// Claims for other oracle ids are ignored. Returning `EmptyOracle` is
    /// intentional: an empty statement has no binding value and must not be
    /// proved as if it authenticated an oracle.
    pub fn root_iop_claim_statement<EF>(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
        oracle_field: RootIopOracleField,
    ) -> Result<NativeWarpWhirOracleStatement<EF>, NativeWarpWhirClaimCompileError>
    where
        EF: ExtensionField<F>,
    {
        let mut eval_claims = Vec::new();
        for claim in claims.iter().filter(|claim| claim.oracle_id == oracle_id) {
            let point = match &claim.point {
                RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
                    if *index >= self.code.codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    // Legacy codeword-domain path: treat a row opening as the
                    // MLE evaluation at the Boolean point encoding that row.
                    // The single-RS message path below avoids this for base
                    // inputs by using RS evaluation weights instead.
                    Point::new(boolean_index_point::<EF>(
                        *index,
                        self.code.log_codeword_len(),
                    ))
                }
                RootIopOpeningPoint::Mle(point) => {
                    if point.len() != self.code.log_codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::PointArityMismatch {
                            oracle_id,
                        });
                    }
                    Point::new(point.clone())
                }
            };

            let value = match (oracle_field, &claim.value) {
                (RootIopOracleField::Base, RootIopOpeningValue::Base(value)) => EF::from(*value),
                (RootIopOracleField::Extension, RootIopOpeningValue::Extension(value)) => *value,
                _ => {
                    return Err(NativeWarpWhirClaimCompileError::OracleFieldMismatch(
                        oracle_id,
                    ));
                }
            };
            eval_claims.push(NativeWarpWhirEvalClaim::new(point, value));
        }

        if eval_claims.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }

        Ok(self.eval_claim_statement(&eval_claims))
    }

    /// Compile base-field WARP codeword index claims into a constrained-RS
    /// statement over the original RS message.
    ///
    /// This is the single-RS path for fresh WARP inputs. WHIR commits to
    /// `C^{-1}(u)` in its usual initial-polynomial form. Each WARP shift query
    /// `u[i] = C(w)[i] = v` is converted into
    ///
    /// ```text
    ///     sum_j lambda_j(omega_n^i) * w[j] = v
    /// ```
    ///
    /// where `lambda_j` are determined by the selected RS coordinates:
    /// monomial/select weights in coefficient form, or subgroup Lagrange
    /// weights in systematic form. This removes the previous
    /// `C(w)`-then-WHIR-encode-`C(w)` double encoding while keeping the WARP
    /// verifier transcript bound to the same RS codeword claim.
    pub fn root_iop_base_message_claim_statement<EF>(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
    ) -> Result<NativeWarpWhirOracleStatement<EF>, NativeWarpWhirClaimCompileError>
    where
        EF: ExtensionField<F>,
    {
        let mut statement = LinearSigmaStatement::initialize(self.code.log_msg_len());
        let mut saw_claim = false;
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
                    if *index >= self.code.codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    // Single-RS path: `u[i]` is not proved by committing to
                    // `u` again. It is the linear form obtained by evaluating
                    // the RS generator row `i` against the original message.
                    let weights = self.code.codeword_index_weights::<EF>(*index);
                    statement.add_constraint(LinearSigmaConstraint::new(Poly::new(weights), value));
                    saw_claim = true;
                }
                RootIopOpeningPoint::Mle(_) => {
                    return Err(NativeWarpWhirClaimCompileError::UnsupportedBaseMle(
                        oracle_id,
                    ));
                }
            }
        }

        if !saw_claim {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }

        Ok(NativeWarpWhirOracleStatement::new(statement))
    }

    /// Compile extension-field accumulator codeword claims into a constrained-RS
    /// statement over the original accumulator message.
    ///
    /// This is the extension-field analogue of
    /// [`Self::root_iop_base_message_claim_statement`]. Index openings use the
    /// RS evaluation weights for the chosen coordinates, while arbitrary
    /// codeword-MLE openings use the adjoint of the same RS encoder. This is
    /// the WARP paper's `u in C` and `u_hat(alpha)=mu` relation expressed in
    /// WHIR's constrained-RS/SumIOP form.
    pub fn root_iop_extension_message_claim_statement<EF>(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
    ) -> Result<NativeWarpWhirOracleStatement<EF>, NativeWarpWhirClaimCompileError>
    where
        EF: ExtensionField<F> + TwoAdicField + Send + Sync,
    {
        let mut statement = LinearSigmaStatement::initialize(self.code.log_msg_len());
        let mut saw_claim = false;
        for claim in claims.iter().filter(|claim| claim.oracle_id == oracle_id) {
            let value = match &claim.value {
                RootIopOpeningValue::Extension(value) => *value,
                _ => {
                    return Err(NativeWarpWhirClaimCompileError::OracleFieldMismatch(
                        oracle_id,
                    ));
                }
            };
            let weights = match &claim.point {
                RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
                    if *index >= self.code.codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    // Same generator-row identity as the base path, but with
                    // extension-field coefficients because accumulator entries
                    // live in `EF`.
                    self.code.codeword_index_weights::<EF>(*index)
                }
                RootIopOpeningPoint::Mle(point) => {
                    if point.len() != self.code.log_codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::PointArityMismatch {
                            oracle_id,
                        });
                    }
                    // WARP's final accumulator check is an MLE claim on the
                    // codeword. The verifier needs the corresponding linear
                    // form on the RS message, i.e. the adjoint of the same
                    // encoder used to produce the codeword.
                    self.code.codeword_mle_weights::<EF>(point)
                }
            };
            statement.add_constraint(LinearSigmaConstraint::new(Poly::new(weights), value));
            saw_claim = true;
        }

        if !saw_claim {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }

        Ok(NativeWarpWhirOracleStatement::new(statement))
    }

    /// Prove WHIR linear-Sigma reductions for every touched oracle in a
    /// commitment-bound WARP root-IOP transcript.
    ///
    /// This is the first native compiler stage for WARP Construction 10.4:
    /// WARP's `VACC`/`DACC` verifier records concrete oracle obligations, and
    /// this method reduces those obligations to one residual opening per
    /// touched oracle using WHIR's linear Sigma-IOP compiler. Each real oracle
    /// commitment is absorbed before its reduction challenges are sampled, so
    /// the later residual-opening backend is bound to the same oracle metadata
    /// and commitment.
    ///
    /// The returned residuals are not optional bookkeeping. They are the claims
    /// the next layer must authenticate against the same commitments.
    pub fn prove_root_iop_reductions<EF, Comm, Challenger>(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, Comm>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        (
            Vec<NativeWarpWhirRootResidualClaim<EF>>,
            NativeWarpWhirRootReductionProof<F, EF>,
        ),
        NativeWarpWhirRootReductionError,
    >
    where
        F: PrimeCharacteristicRing,
        EF: ExtensionField<F> + TwoAdicField,
        Comm: Clone,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Comm>,
    {
        self.prove_root_iop_reductions_with_observer(
            transcript,
            challenger,
            reduction_pow_bits,
            |challenger, commitment| commitment.observe_into::<F, _>(challenger),
        )
    }

    /// Prove root-IOP reductions with a caller-provided commitment observer.
    ///
    /// This variant is used by mixed commitment schemes, such as the native
    /// WARP root proof where base oracles carry one WHIR root and extension
    /// oracles carry one root per extension limb.
    pub fn prove_root_iop_reductions_with_observer<EF, Comm, Challenger, Observe>(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, Comm>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
        mut observe_commitment: Observe,
    ) -> Result<
        (
            Vec<NativeWarpWhirRootResidualClaim<EF>>,
            NativeWarpWhirRootReductionProof<F, EF>,
        ),
        NativeWarpWhirRootReductionError,
    >
    where
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        Observe: FnMut(&mut Challenger, &RootIopBoundCommitment<Comm>),
    {
        transcript
            .verify_witnessed_claim_values()
            .map_err(NativeWarpWhirRootReductionError::RootIop)?;
        self.check_unique_bound_oracle_ids(&transcript.oracles)?;
        self.check_claim_oracles_bound(&transcript.oracles, &transcript.claims)?;

        let mut residuals = Vec::new();
        let mut reductions = Vec::new();
        for (commitment, values) in &transcript.oracles {
            if !claims_include_oracle(&transcript.claims, commitment.oracle_id) {
                continue;
            }
            self.check_bound_oracle_shape(commitment, Some(values))?;

            let statement = self.root_iop_claim_statement(
                &transcript.claims,
                commitment.oracle_id,
                commitment.field,
            )?;
            observe_commitment(challenger, commitment);
            let (reduction, opening) = match values {
                RootIopOracleValues::Base(values) => statement.prove_reduction_base::<F, _>(
                    &Poly::new(values.clone()),
                    challenger,
                    reduction_pow_bits,
                )?,
                RootIopOracleValues::Extension(values) => statement.prove_reduction_ext::<F, _>(
                    &Poly::new(values.clone()),
                    challenger,
                    reduction_pow_bits,
                )?,
            };

            residuals.push(NativeWarpWhirRootResidualClaim {
                oracle_id: commitment.oracle_id,
                field: commitment.field,
                opening,
            });
            reductions.push(NativeWarpWhirRootOracleReductionProof {
                oracle_id: commitment.oracle_id,
                field: commitment.field,
                reduction,
            });
        }

        Ok((
            residuals,
            NativeWarpWhirRootReductionProof {
                oracles: reductions,
            },
        ))
    }

    /// Verify the WHIR linear-Sigma reduction stage for a WARP root IOP.
    ///
    /// The caller should pass the commitments and claims produced by replaying
    /// WARP with `RootIopBoundVerifier`. Successful verification returns the
    /// residual openings that must be checked by the residual-opening backend.
    pub fn verify_root_iop_reductions<EF, Comm, Challenger>(
        &self,
        expected_commitments: &[RootIopBoundCommitment<Comm>],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootReductionProof<F, EF>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<Vec<NativeWarpWhirRootResidualClaim<EF>>, NativeWarpWhirRootReductionError>
    where
        F: PrimeCharacteristicRing,
        EF: ExtensionField<F> + TwoAdicField,
        Comm: Clone,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Comm>,
    {
        self.verify_root_iop_reductions_with_observer(
            expected_commitments,
            expected_claims,
            proof,
            challenger,
            reduction_pow_bits,
            |challenger, commitment| commitment.observe_into::<F, _>(challenger),
        )
    }

    /// Verify root-IOP reductions with a caller-provided commitment observer.
    pub fn verify_root_iop_reductions_with_observer<EF, Comm, Challenger, Observe>(
        &self,
        expected_commitments: &[RootIopBoundCommitment<Comm>],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootReductionProof<F, EF>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
        mut observe_commitment: Observe,
    ) -> Result<Vec<NativeWarpWhirRootResidualClaim<EF>>, NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        Observe: FnMut(&mut Challenger, &RootIopBoundCommitment<Comm>),
    {
        self.check_unique_public_oracle_ids(expected_commitments)?;
        self.check_claim_oracles_public(expected_commitments, expected_claims)?;

        let mut proof_iter = proof.oracles.iter();
        let mut residuals = Vec::new();
        for commitment in expected_commitments {
            if !claims_include_oracle(expected_claims, commitment.oracle_id) {
                continue;
            }
            self.check_bound_oracle_shape::<EF, Comm>(commitment, None)?;

            let oracle_proof = proof_iter.next().ok_or(
                NativeWarpWhirRootReductionError::MissingOracleReduction(commitment.oracle_id),
            )?;
            if oracle_proof.oracle_id != commitment.oracle_id {
                return Err(
                    NativeWarpWhirRootReductionError::OracleReductionOrderMismatch {
                        expected: commitment.oracle_id,
                        actual: oracle_proof.oracle_id,
                    },
                );
            }
            if oracle_proof.field != commitment.field {
                return Err(
                    NativeWarpWhirRootReductionError::OracleReductionFieldMismatch(
                        commitment.oracle_id,
                    ),
                );
            }

            let statement = self.root_iop_claim_statement(
                expected_claims,
                commitment.oracle_id,
                commitment.field,
            )?;
            observe_commitment(challenger, commitment);
            let opening = statement.verify_reduction::<F, _>(
                &oracle_proof.reduction,
                challenger,
                reduction_pow_bits,
            )?;
            residuals.push(NativeWarpWhirRootResidualClaim {
                oracle_id: commitment.oracle_id,
                field: commitment.field,
                opening,
            });
        }

        if proof_iter.next().is_some() {
            return Err(NativeWarpWhirRootReductionError::TrailingOracleReductions);
        }

        Ok(residuals)
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
    /// Create a native WARP root proof system backed by WHIR.
    pub fn new(
        pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        code: &'a ReedSolomonCode<F, Dft>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            base_message_pcs: None,
            limb_backend: WhirLimbAccumulatorBackend::new(pcs, challenger_seed.clone()),
            compiler: NativeWarpWhirCompiler::new(code),
            challenger_seed,
        }
    }

    /// Create a native WARP root proof system whose WARP inputs are committed
    /// as WHIR initial RS messages.
    ///
    /// `pcs` remains available for legacy codeword-domain openings.
    /// `base_message_pcs` must be configured with `code.log_msg_len()`
    /// variables and the same RS rate/security settings. This is the sound
    /// single-RS path: WHIR commits to `C^{-1}(u)`, and WARP codeword openings
    /// are compiled using the same [`ReedSolomonCode`] generator.
    pub fn new_with_base_message_pcs(
        pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        base_message_pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        code: &'a ReedSolomonCode<F, Dft>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            base_message_pcs: Some(base_message_pcs),
            limb_backend: WhirLimbAccumulatorBackend::new(pcs, challenger_seed.clone()),
            compiler: NativeWarpWhirCompiler::new(code),
            challenger_seed,
        }
    }

    /// Commit a base-field fresh codeword for the WARP root IOP.
    pub fn commit_base_oracle(
        &self,
        oracle_id: usize,
        codeword: Vec<F>,
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

        let mut challenger = self.base_oracle_challenger(oracle_id);
        let (commitment, prover_data) = self
            .pcs
            .commit_deferred(RowMajorMatrix::new(codeword, 1), &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Base,
                commitment: NativeWarpWhirRootCommitment::Base(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::Base(NativeWarpWhirRootBaseProverData {
                    prover_data,
                    challenger,
                    message: None,
                }),
            },
        ))
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
        let base_message_pcs = self
            .base_message_pcs
            .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;

        let mut challenger = self.base_oracle_challenger(oracle_id);
        let (commitment, prover_data) = base_message_pcs
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
                    message: Some(message),
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
        let base_message_pcs = self
            .base_message_pcs
            .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;
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
        let encoded = base_message_pcs.encode_base_batch_initial_oracles(matrices);
        let (root, shared) =
            base_message_pcs.commit_base_batch_encoded_deferred(encoded, &mut challenger);

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

        let mut challenger = self.base_oracle_challenger(oracle_id);
        if let Some(message_pcs) = self.base_message_pcs {
            let message = self.compiler.code().message_from_codeword(&codeword);
            return self.commit_extension_message_oracle_with_challenger(
                oracle_id,
                message,
                challenger,
                message_pcs,
            );
        }

        let (commitment, prover_data) = self
            .pcs
            .commit_extension_deferred(RowMajorMatrix::new(codeword, 1), &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Extension,
                commitment: NativeWarpWhirRootCommitment::ExtensionNative(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::ExtensionNative(
                    NativeWarpWhirRootExtensionProverData {
                        prover_data,
                        challenger,
                        message: None,
                    },
                ),
            },
        ))
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
        let message_pcs = self
            .base_message_pcs
            .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;
        let challenger = self.base_oracle_challenger(oracle_id);
        self.commit_extension_message_oracle_with_challenger(
            oracle_id,
            message,
            challenger,
            message_pcs,
        )
    }

    fn commit_extension_message_oracle_with_challenger(
        &self,
        oracle_id: usize,
        message: Vec<EF>,
        mut challenger: Challenger,
        message_pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
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

        let encoded =
            message_pcs.encode_extension_initial_oracle(RowMajorMatrix::new(message.clone(), 1));
        let (commitment, prover_data) =
            message_pcs.commit_extension_encoded_deferred(encoded, &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Extension,
                commitment: NativeWarpWhirRootCommitment::ExtensionMessage(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::ExtensionNative(
                    NativeWarpWhirRootExtensionProverData {
                        prover_data,
                        challenger,
                        message: Some(message),
                    },
                ),
            },
        ))
    }

    /// Prove WARP root reductions and bind every residual opening with WHIR.
    pub fn prove(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<NativeWarpWhirRootProof<F, EF, MT>, NativeWarpWhirRootProofError> {
        // Fast path: when every touched oracle is a WHIR initial-message
        // oracle, Section 8.2-style batching can combine all WARP constraints
        // directly into one linear-Sigma reduction and one grouped WHIR
        // opening. This is the benchmarked single-RS path.
        if let Some(direct_batched_opening) = self.try_prove_direct_batched_root(
            transcript,
            prover_data,
            challenger,
            reduction_pow_bits,
        )? {
            return Ok(NativeWarpWhirRootProof {
                reductions: NativeWarpWhirRootReductionProof {
                    oracles: Vec::new(),
                },
                openings: Vec::new(),
                batched_opening: None,
                direct_batched_opening: Some(direct_batched_opening),
            });
        }

        // Compatibility path: reduce each oracle separately to one residual
        // opening claim, then try to batch those residual openings. This is
        // still sound, but it performs more transcript and opening work.
        let (residuals, reductions) = self.prove_native_root_reductions(
            transcript,
            prover_data,
            challenger,
            reduction_pow_bits,
        )?;
        if let Some(batched_opening) = self.try_prove_batched_residual_opening(
            &residuals,
            prover_data,
            challenger,
            reduction_pow_bits,
        )? {
            return Ok(NativeWarpWhirRootProof {
                reductions,
                openings: Vec::new(),
                batched_opening: Some(batched_opening),
                direct_batched_opening: None,
            });
        }

        // Last fallback: open each residual with the appropriate backend. This
        // remains useful for mixed legacy commitments and tests, but should not
        // be the fast path for the native WARP/WHIR benchmark.
        let mut openings = Vec::with_capacity(residuals.len());
        for residual in &residuals {
            let oracle_data = prover_data
                .iter()
                .find(|data| data.oracle_id == residual.oracle_id)
                .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                    residual.oracle_id,
                ))?;
            let opening = self.prove_residual_opening(residual, &oracle_data.data)?;
            openings.push(opening);
        }

        Ok(NativeWarpWhirRootProof {
            reductions,
            openings,
            batched_opening: None,
            direct_batched_opening: None,
        })
    }

    /// Verify WARP root reductions and every WHIR-bound residual opening.
    pub fn verify(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<Vec<NativeWarpWhirRootResidualClaim<EF>>, NativeWarpWhirRootProofError> {
        // Direct proofs contain no separate per-oracle reduction objects. The
        // verifier reconstructs the same statements from replayed WARP root
        // claims, absorbs the same commitments in the same order, then checks
        // one batched WHIR opening.
        if let Some(direct_batched_opening) = &proof.direct_batched_opening {
            if !proof.reductions.oracles.is_empty()
                || !proof.openings.is_empty()
                || proof.batched_opening.is_some()
            {
                return Err(NativeWarpWhirRootProofError::TrailingOpenings);
            }
            self.verify_direct_batched_root(
                expected_commitments,
                expected_claims,
                direct_batched_opening,
                challenger,
                reduction_pow_bits,
            )?;
            return Ok(Vec::new());
        }

        // Compatibility verification mirrors the two-stage prover path: first
        // verify linear-Sigma reductions and derive residual openings, then
        // authenticate those openings with WHIR.
        let residuals = self.verify_native_root_reductions(
            expected_commitments,
            expected_claims,
            &proof.reductions,
            challenger,
            reduction_pow_bits,
        )?;
        if let Some(batched_opening) = &proof.batched_opening {
            if !proof.openings.is_empty() {
                return Err(NativeWarpWhirRootProofError::TrailingOpenings);
            }
            self.verify_batched_residual_opening(
                expected_commitments,
                &residuals,
                batched_opening,
                challenger,
                reduction_pow_bits,
            )?;
            return Ok(residuals);
        }

        if proof.openings.len() != residuals.len() {
            if proof.openings.len() > residuals.len() {
                return Err(NativeWarpWhirRootProofError::TrailingOpenings);
            }
            let missing = residuals
                .get(proof.openings.len())
                .map(|residual| residual.oracle_id)
                .unwrap_or(0);
            return Err(NativeWarpWhirRootProofError::MissingOpening(missing));
        }

        for (residual, opening) in residuals.iter().zip(proof.openings.iter()) {
            let commitment = expected_commitments
                .iter()
                .find(|commitment| commitment.oracle_id == residual.oracle_id)
                .ok_or(NativeWarpWhirRootReductionError::UnknownOracle(
                    residual.oracle_id,
                ))?;
            self.verify_residual_opening(commitment, residual, opening)?;
        }

        Ok(residuals)
    }

    fn try_prove_direct_batched_root(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        Option<NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>>,
        NativeWarpWhirRootProofError,
    > {
        let pcs = match self.base_message_pcs {
            Some(pcs) => pcs,
            None => return Ok(None),
        };

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
                    let NativeWarpWhirRootProverData::ExtensionNative(data) = &oracle_data.data
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
                _ => return Ok(None),
            }
        }

        if statements.is_empty() {
            return Ok(None);
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
        let opening = pcs.open_grouped_batched_deferred(
            Self::group_prover_oracles(whir_oracles, &opening_claim.coeffs)?,
            opening_claim.point,
            opening_claim.value,
            challenger,
        )?;

        Ok(Some(NativeWarpWhirRootBatchedOpeningProof {
            reduction,
            opening,
        }))
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
        let pcs = self
            .base_message_pcs
            .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;
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
                _ => {
                    return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                        commitment.oracle_id,
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
        pcs.verify_batched_deferred(
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

    fn try_prove_batched_residual_opening(
        &self,
        residuals: &[NativeWarpWhirRootResidualClaim<EF>],
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        Option<NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>>,
        NativeWarpWhirRootProofError,
    > {
        if residuals.len() < 2 {
            return Ok(None);
        }
        let pcs = match self.base_message_pcs {
            Some(pcs) => pcs,
            None => return Ok(None),
        };

        let mut statements = Vec::with_capacity(residuals.len());
        let mut polys = Vec::with_capacity(residuals.len());
        let mut whir_oracles = Vec::with_capacity(residuals.len());
        for residual in residuals {
            let oracle_data = prover_data
                .iter()
                .find(|data| data.oracle_id == residual.oracle_id)
                .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                    residual.oracle_id,
                ))?;

            statements.push(residual_eq_statement::<F, EF>(residual));
            match (residual.field, &oracle_data.data) {
                (RootIopOracleField::Base, NativeWarpWhirRootProverData::Base(data))
                    if data.message.is_some() =>
                {
                    let message = data.message.as_ref().unwrap();
                    polys.push(NativeWarpBatchedResidualPoly::Base(Poly::new(
                        message.clone(),
                    )));
                    whir_oracles.push(WhirBatchedDeferredProverData::Base(
                        data.prover_data.clone(),
                    ));
                }
                (
                    RootIopOracleField::Extension,
                    NativeWarpWhirRootProverData::ExtensionNative(data),
                ) if data.message.is_some() => {
                    let message = data.message.as_ref().unwrap();
                    polys.push(NativeWarpBatchedResidualPoly::Extension(Poly::new(
                        message.clone(),
                    )));
                    whir_oracles.push(WhirBatchedDeferredProverData::Extension(
                        data.prover_data.clone(),
                    ));
                }
                _ => return Ok(None),
            }
        }

        let sumcheck_oracles = statements
            .iter()
            .zip(&polys)
            .map(|(statement, poly)| match poly {
                NativeWarpBatchedResidualPoly::Base(poly) => {
                    BatchedLinearSigmaProverOracle::base(statement, poly)
                }
                NativeWarpBatchedResidualPoly::Extension(poly) => {
                    BatchedLinearSigmaProverOracle::extension(statement, poly)
                }
            })
            .collect::<Vec<_>>();
        // Residual batching is only a fallback. It randomly combines the
        // already reduced residual claims, producing one further WHIR opening
        // claim over the residual oracle list.
        let (reduction, opening_claim) = prove_batched_linear_sigma_reduction::<F, EF, _>(
            &sumcheck_oracles,
            challenger,
            reduction_pow_bits,
        )?;
        let opening = pcs.open_batched_deferred(
            whir_oracles,
            &opening_claim.coeffs,
            opening_claim.point,
            opening_claim.value,
            challenger,
        )?;

        Ok(Some(NativeWarpWhirRootBatchedOpeningProof {
            reduction,
            opening,
        }))
    }

    fn verify_batched_residual_opening(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        residuals: &[NativeWarpWhirRootResidualClaim<EF>],
        proof: &NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        let pcs = self
            .base_message_pcs
            .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;
        let mut statements = Vec::with_capacity(residuals.len());
        let mut commitments = Vec::with_capacity(residuals.len());
        for residual in residuals {
            statements.push(residual_eq_statement::<F, EF>(residual));
            let commitment = expected_commitments
                .iter()
                .find(|commitment| commitment.oracle_id == residual.oracle_id)
                .ok_or(NativeWarpWhirRootReductionError::UnknownOracle(
                    residual.oracle_id,
                ))?;
            match (&commitment.commitment, residual.field) {
                (
                    NativeWarpWhirRootCommitment::BaseMessage(commitment),
                    RootIopOracleField::Base,
                ) => commitments.push(NativeWarpBatchedResidualCommitment::Base(
                    commitment.clone(),
                )),
                (
                    NativeWarpWhirRootCommitment::BaseMessageShared {
                        root,
                        column,
                        width,
                    },
                    RootIopOracleField::Base,
                ) => commitments.push(NativeWarpBatchedResidualCommitment::SharedBase {
                    root: root.clone(),
                    column: *column,
                    width: *width,
                }),
                (
                    NativeWarpWhirRootCommitment::ExtensionMessage(commitment),
                    RootIopOracleField::Extension,
                ) => commitments.push(NativeWarpBatchedResidualCommitment::Extension(
                    commitment.clone(),
                )),
                _ => {
                    return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                        residual.oracle_id,
                    ));
                }
            }
        }

        let statement_refs = statements.iter().collect::<Vec<_>>();
        let opening_claim = verify_batched_linear_sigma_reduction::<F, EF, _>(
            &statement_refs,
            &proof.reduction,
            challenger,
            reduction_pow_bits,
        )?;
        let whir_oracles = Self::group_verifier_oracles(commitments, &opening_claim.coeffs)?;
        pcs.verify_batched_deferred(
            &whir_oracles,
            opening_claim.point,
            opening_claim.value,
            &proof.opening,
            challenger,
        )
        .map_err(NativeWarpWhirRootProofError::BatchedOpening)
    }

    fn prove_native_root_reductions(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        (
            Vec<NativeWarpWhirRootResidualClaim<EF>>,
            NativeWarpWhirRootReductionProof<F, EF>,
        ),
        NativeWarpWhirRootProofError,
    > {
        transcript
            .verify_witnessed_claim_values()
            .map_err(NativeWarpWhirRootReductionError::RootIop)?;
        self.compiler
            .check_unique_bound_oracle_ids(&transcript.oracles)?;
        self.compiler
            .check_claim_oracles_bound(&transcript.oracles, &transcript.claims)?;

        let mut residuals = Vec::new();
        let mut reductions = Vec::new();
        for (commitment, values) in &transcript.oracles {
            if !claims_include_oracle(&transcript.claims, commitment.oracle_id) {
                continue;
            }
            self.compiler
                .check_bound_oracle_shape(commitment, Some(values))?;
            // Every per-oracle reduction is bound to its WARP oracle metadata
            // and concrete WHIR commitment before the oracle's reduction
            // randomness is sampled.
            observe_native_root_commitment::<F, Challenger, MT::Commitment>(challenger, commitment);

            let (reduction, opening) = match (&commitment.commitment, values) {
                (NativeWarpWhirRootCommitment::Base(_), RootIopOracleValues::Base(values)) => {
                    let statement = self.compiler.root_iop_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                        RootIopOracleField::Base,
                    )?;
                    statement.prove_reduction_base::<F, _>(
                        &Poly::new(values.clone()),
                        challenger,
                        reduction_pow_bits,
                    )?
                }
                (
                    NativeWarpWhirRootCommitment::BaseMessage(_)
                    | NativeWarpWhirRootCommitment::BaseMessageShared { .. },
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
                    let statement = self.compiler.root_iop_base_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?;
                    statement.prove_reduction_base::<F, _>(
                        &Poly::new(message.to_vec()),
                        challenger,
                        reduction_pow_bits,
                    )?
                }
                (
                    NativeWarpWhirRootCommitment::Extension(_)
                    | NativeWarpWhirRootCommitment::ExtensionNative(_),
                    RootIopOracleValues::Extension(values),
                ) => {
                    let statement = self.compiler.root_iop_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                        RootIopOracleField::Extension,
                    )?;
                    statement.prove_reduction_ext::<F, _>(
                        &Poly::new(values.clone()),
                        challenger,
                        reduction_pow_bits,
                    )?
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
                    let statement = self.compiler.root_iop_extension_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?;
                    statement.prove_reduction_ext::<F, _>(
                        &Poly::new(message.to_vec()),
                        challenger,
                        reduction_pow_bits,
                    )?
                }
                _ => {
                    return Err(NativeWarpWhirRootReductionError::OracleValueFieldMismatch(
                        commitment.oracle_id,
                    )
                    .into());
                }
            };

            residuals.push(NativeWarpWhirRootResidualClaim {
                oracle_id: commitment.oracle_id,
                field: commitment.field,
                opening,
            });
            reductions.push(NativeWarpWhirRootOracleReductionProof {
                oracle_id: commitment.oracle_id,
                field: commitment.field,
                reduction,
            });
        }

        Ok((
            residuals,
            NativeWarpWhirRootReductionProof {
                oracles: reductions,
            },
        ))
    }

    fn verify_native_root_reductions(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootReductionProof<F, EF>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<Vec<NativeWarpWhirRootResidualClaim<EF>>, NativeWarpWhirRootProofError> {
        self.compiler
            .check_unique_public_oracle_ids(expected_commitments)?;
        self.compiler
            .check_claim_oracles_public(expected_commitments, expected_claims)?;

        let mut proof_iter = proof.oracles.iter();
        let mut residuals = Vec::new();
        for commitment in expected_commitments {
            if !claims_include_oracle(expected_claims, commitment.oracle_id) {
                continue;
            }
            self.compiler
                .check_bound_oracle_shape::<EF, _>(commitment, None)?;

            let oracle_proof = proof_iter.next().ok_or(
                NativeWarpWhirRootReductionError::MissingOracleReduction(commitment.oracle_id),
            )?;
            if oracle_proof.oracle_id != commitment.oracle_id {
                return Err(
                    NativeWarpWhirRootReductionError::OracleReductionOrderMismatch {
                        expected: commitment.oracle_id,
                        actual: oracle_proof.oracle_id,
                    }
                    .into(),
                );
            }
            if oracle_proof.field != commitment.field {
                return Err(
                    NativeWarpWhirRootReductionError::OracleReductionFieldMismatch(
                        commitment.oracle_id,
                    )
                    .into(),
                );
            }

            observe_native_root_commitment::<F, Challenger, MT::Commitment>(challenger, commitment);
            let statement = match &commitment.commitment {
                NativeWarpWhirRootCommitment::Base(_) => self.compiler.root_iop_claim_statement(
                    expected_claims,
                    commitment.oracle_id,
                    RootIopOracleField::Base,
                )?,
                NativeWarpWhirRootCommitment::BaseMessage(_)
                | NativeWarpWhirRootCommitment::BaseMessageShared { .. } => self
                    .compiler
                    .root_iop_base_message_claim_statement(expected_claims, commitment.oracle_id)?,
                NativeWarpWhirRootCommitment::Extension(_)
                | NativeWarpWhirRootCommitment::ExtensionNative(_) => {
                    self.compiler.root_iop_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                        RootIopOracleField::Extension,
                    )?
                }
                NativeWarpWhirRootCommitment::ExtensionMessage(_) => {
                    self.compiler.root_iop_extension_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?
                }
            };
            let opening = statement.verify_reduction::<F, _>(
                &oracle_proof.reduction,
                challenger,
                reduction_pow_bits,
            )?;
            residuals.push(NativeWarpWhirRootResidualClaim {
                oracle_id: commitment.oracle_id,
                field: commitment.field,
                opening,
            });
        }

        if proof_iter.next().is_some() {
            return Err(NativeWarpWhirRootReductionError::TrailingOracleReductions.into());
        }

        Ok(residuals)
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
            NativeWarpWhirRootProverData::Base(data) => data
                .message
                .as_deref()
                .ok_or(NativeWarpWhirRootProofError::OracleKindMismatch(oracle_id)),
            NativeWarpWhirRootProverData::BaseShared(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::Extension(_)
            | NativeWarpWhirRootProverData::ExtensionNative(_) => {
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
            NativeWarpWhirRootProverData::ExtensionNative(data) => data
                .message
                .as_deref()
                .ok_or(NativeWarpWhirRootProofError::OracleKindMismatch(oracle_id)),
            NativeWarpWhirRootProverData::Base(_)
            | NativeWarpWhirRootProverData::BaseShared(_)
            | NativeWarpWhirRootProverData::Extension(_) => {
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

    fn prove_residual_opening(
        &self,
        residual: &NativeWarpWhirRootResidualClaim<EF>,
        prover_data: &NativeWarpWhirRootProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
    ) -> Result<NativeWarpWhirRootOracleOpeningProof<F, EF, MT>, NativeWarpWhirRootProofError> {
        let opening_points = [vec![residual.opening.point.clone()]];
        match (residual.field, prover_data) {
            (RootIopOracleField::Base, NativeWarpWhirRootProverData::Base(base_prover_data)) => {
                let mut challenger = base_prover_data.challenger.clone();
                let pcs = if base_prover_data.message.is_some() {
                    self.base_message_pcs
                        .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?
                } else {
                    self.pcs
                };
                let (opened_values, proof) = pcs.open_deferred(
                    base_prover_data.prover_data.clone(),
                    &opening_points,
                    &mut challenger,
                );
                self.check_opened_residual(residual, &opened_values)?;
                Ok(NativeWarpWhirRootOracleOpeningProof::Base(proof))
            }
            (
                RootIopOracleField::Extension,
                NativeWarpWhirRootProverData::Extension(extension_prover_data),
            ) => {
                let (opened_values, proof) = self
                    .limb_backend
                    .prove_points(extension_prover_data, &opening_points)
                    .map_err(|error| NativeWarpWhirRootProofError::ExtensionOpening {
                        oracle_id: residual.oracle_id,
                        error,
                    })?;
                self.check_opened_residual(residual, &opened_values)?;
                Ok(NativeWarpWhirRootOracleOpeningProof::Extension(proof))
            }
            (
                RootIopOracleField::Extension,
                NativeWarpWhirRootProverData::ExtensionNative(extension_prover_data),
            ) => {
                let mut challenger = extension_prover_data.challenger.clone();
                let pcs = if extension_prover_data.message.is_some() {
                    self.base_message_pcs
                        .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?
                } else {
                    self.pcs
                };
                let (opened_values, proof) = pcs.open_extension_deferred(
                    extension_prover_data.prover_data.clone(),
                    &opening_points,
                    &mut challenger,
                );
                self.check_opened_residual(residual, &opened_values)?;
                if extension_prover_data.message.is_some() {
                    Ok(NativeWarpWhirRootOracleOpeningProof::ExtensionMessage(
                        proof,
                    ))
                } else {
                    Ok(NativeWarpWhirRootOracleOpeningProof::ExtensionNative(proof))
                }
            }
            _ => Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                residual.oracle_id,
            )),
        }
    }

    fn verify_residual_opening(
        &self,
        commitment: &RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
        residual: &NativeWarpWhirRootResidualClaim<EF>,
        proof: &NativeWarpWhirRootOracleOpeningProof<F, EF, MT>,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        let opening_claims = [vec![(
            residual.opening.point.clone(),
            residual.opening.value,
        )]];
        match (&commitment.commitment, residual.field, proof) {
            (
                NativeWarpWhirRootCommitment::Base(commitment),
                RootIopOracleField::Base,
                NativeWarpWhirRootOracleOpeningProof::Base(proof),
            ) => {
                let mut challenger = self.base_oracle_challenger(residual.oracle_id);
                self.pcs
                    .verify_deferred(commitment, &opening_claims, proof, &mut challenger)
                    .map_err(|error| NativeWarpWhirRootProofError::BaseOpening {
                        oracle_id: residual.oracle_id,
                        error,
                    })
            }
            (
                NativeWarpWhirRootCommitment::BaseMessage(commitment),
                RootIopOracleField::Base,
                NativeWarpWhirRootOracleOpeningProof::Base(proof),
            ) => {
                let base_message_pcs = self
                    .base_message_pcs
                    .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;
                let mut challenger = self.base_oracle_challenger(residual.oracle_id);
                base_message_pcs
                    .verify_deferred(commitment, &opening_claims, proof, &mut challenger)
                    .map_err(|error| NativeWarpWhirRootProofError::BaseOpening {
                        oracle_id: residual.oracle_id,
                        error,
                    })
            }
            (
                NativeWarpWhirRootCommitment::Extension(commitment),
                RootIopOracleField::Extension,
                NativeWarpWhirRootOracleOpeningProof::Extension(proof),
            ) => self
                .limb_backend
                .verify_points(commitment, &opening_claims, proof)
                .map_err(|error| NativeWarpWhirRootProofError::ExtensionOpening {
                    oracle_id: residual.oracle_id,
                    error,
                }),
            (
                NativeWarpWhirRootCommitment::ExtensionNative(commitment),
                RootIopOracleField::Extension,
                NativeWarpWhirRootOracleOpeningProof::ExtensionNative(proof),
            ) => {
                let mut challenger = self.base_oracle_challenger(residual.oracle_id);
                self.pcs
                    .verify_extension_deferred(commitment, &opening_claims, proof, &mut challenger)
                    .map_err(|error| NativeWarpWhirRootProofError::BaseOpening {
                        oracle_id: residual.oracle_id,
                        error,
                    })
            }
            (
                NativeWarpWhirRootCommitment::ExtensionMessage(commitment),
                RootIopOracleField::Extension,
                NativeWarpWhirRootOracleOpeningProof::ExtensionMessage(proof),
            ) => {
                let message_pcs = self
                    .base_message_pcs
                    .ok_or(NativeWarpWhirRootProofError::BaseMessagePcsRequired)?;
                let mut challenger = self.base_oracle_challenger(residual.oracle_id);
                message_pcs
                    .verify_extension_deferred(commitment, &opening_claims, proof, &mut challenger)
                    .map_err(|error| NativeWarpWhirRootProofError::BaseOpening {
                        oracle_id: residual.oracle_id,
                        error,
                    })
            }
            _ => Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                residual.oracle_id,
            )),
        }
    }

    fn check_opened_residual(
        &self,
        residual: &NativeWarpWhirRootResidualClaim<EF>,
        opened_values: &MultilinearOpenedValues<EF>,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        let opened = opened_values
            .first()
            .and_then(|values| values.first())
            .copied()
            .ok_or(NativeWarpWhirRootProofError::OpeningShape(
                residual.oracle_id,
            ))?;
        if opened != residual.opening.value {
            return Err(NativeWarpWhirRootProofError::ResidualOpeningMismatch(
                residual.oracle_id,
            ));
        }
        Ok(())
    }
}

mod direct;
use direct::{
    NativeWarpBatchedResidualCommitment, NativeWarpBatchedResidualPoly,
    NativeWarpBatchedResidualProverOracle, NativeWarpCompactRootStatement,
    NativeWarpDirectBatchedResidualPoly, observe_native_root_commitment,
    prove_compact_batched_root_reduction, residual_eq_statement,
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

fn boolean_index_point<EF: Field>(index: usize, num_variables: usize) -> Vec<EF> {
    (0..num_variables)
        .map(|bit| {
            if (index >> (num_variables - 1 - bit)) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            }
        })
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
