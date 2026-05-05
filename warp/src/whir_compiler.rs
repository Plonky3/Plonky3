//! Native WARP-to-WHIR compiler scaffolding.
//!
//! This module is the RS-only compiler boundary described in
//! `warp/docs/native-whir-compiler.md`. It starts at the algebraic statement
//! level: WARP evaluation obligations are converted into WHIR linear
//! Sigma-IOP constraints rather than into PCS opening calls.

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

const ROOT_WHIR_BASE_ORACLE_TAG: u64 = 0x5741_5250_5242_4153;

/// Proof that a WARP linear-Sigma statement over an extension-field oracle was
/// reduced to one residual opening and that the residual opening is
/// authenticated by the caller's point-opening backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, PointProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, PointProof: Deserialize<'de>"
))]
pub struct NativeWarpWhirPointProof<F, EF, PointProof> {
    /// Sumcheck reducing all public linear-Sigma constraints to one oracle
    /// opening.
    pub reduction: LinearSigmaReductionProof<F, EF>,
    /// Backend proof for the residual opening against the original committed
    /// oracle.
    pub opening: PointProof,
}

/// Errors produced by the native WARP-to-WHIR compiler bridge.
#[derive(Debug, Error)]
pub enum NativeWarpWhirCompilerError<PointError> {
    /// The linear-Sigma reduction failed.
    #[error(transparent)]
    Reduction(#[from] LinearSigmaReductionError),

    /// The point-opening backend rejected or failed to produce an opening.
    #[error("point-opening backend failed: {0:?}")]
    PointOpening(PointError),

    /// The point-opening backend returned a malformed opened-value shape.
    #[error("point-opening backend returned malformed opened values")]
    OpeningShape,

    /// The residual value opened by the backend did not match the linear-Sigma
    /// reduction output.
    #[error("residual opening value does not match linear-Sigma reduction")]
    ResidualOpeningMismatch,
}

/// Errors produced while compiling recorded root-IOP claims into WHIR
/// linear-Sigma statements.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum NativeWarpWhirClaimCompileError {
    /// No claims matched the requested oracle id.
    #[error("no root-IOP claims for oracle {0}")]
    EmptyOracle(usize),

    /// An index opening was outside the compiler's codeword hypercube.
    #[error("root-IOP claim index {index} is outside oracle {oracle_id}")]
    IndexOutOfBounds { oracle_id: usize, index: usize },

    /// A multilinear point had the wrong arity.
    #[error("root-IOP claim point arity mismatch for oracle {oracle_id}")]
    PointArityMismatch { oracle_id: usize },

    /// The recorded value type does not match the oracle field.
    #[error("root-IOP claim value field mismatch for oracle {0}")]
    OracleFieldMismatch(usize),

    /// A base-field MLE claim cannot be compiled against a message-domain
    /// commitment without an explicit RS-adjoint weight transform.
    #[error("root-IOP base MLE claim for oracle {0} is unsupported by the message-domain compiler")]
    UnsupportedBaseMle(usize),
}

/// One per-oracle WHIR linear-Sigma reduction emitted from a WARP root IOP.
///
/// This is the compiler proof layer from WARP's recorded `VACC`/`DACC`
/// obligations to WHIR Section 7 linear Sigma claims. It is not sufficient by
/// itself: successful verification returns residual openings that must still be
/// authenticated against the same committed oracles by the caller's WHIR
/// opening/proximity backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"
))]
pub struct NativeWarpWhirRootOracleReductionProof<F, EF> {
    /// Oracle id assigned by the root-IOP recorder.
    pub oracle_id: usize,
    /// Field of the oracle being reduced.
    pub field: RootIopOracleField,
    /// WHIR linear-Sigma sumcheck reduction for all claims on this oracle.
    pub reduction: LinearSigmaReductionProof<F, EF>,
}

/// WHIR linear-Sigma reduction proof for all touched WARP root-IOP oracles.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"
))]
pub struct NativeWarpWhirRootReductionProof<F, EF> {
    /// Per-oracle reductions in root-IOP commitment order.
    pub oracles: Vec<NativeWarpWhirRootOracleReductionProof<F, EF>>,
}

/// Residual opening returned by the root compiler after a successful reduction.
///
/// The residual is intentionally not a proof. It is the exact opening claim that
/// the next layer must prove against the oracle commitment observed before this
/// reduction was sampled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeWarpWhirRootResidualClaim<EF> {
    /// Oracle id assigned by the root-IOP recorder.
    pub oracle_id: usize,
    /// Field of the oracle being opened.
    pub field: RootIopOracleField,
    /// Residual multilinear opening claim.
    pub opening: LinearSigmaOpeningClaim<EF>,
}

/// Errors from compiling a full root-IOP transcript into WHIR reductions.
#[derive(Debug, Error)]
pub enum NativeWarpWhirRootReductionError {
    /// The recorded root-IOP transcript was internally inconsistent.
    #[error("root-IOP transcript error: {0:?}")]
    RootIop(RootIopError),

    /// A root-IOP claim could not be converted into a WHIR linear-Sigma claim.
    #[error(transparent)]
    ClaimCompile(#[from] NativeWarpWhirClaimCompileError),

    /// A linear-Sigma reduction failed.
    #[error(transparent)]
    Reduction(#[from] LinearSigmaReductionError),

    /// A claim references an oracle that is absent from the public commitments.
    #[error("root-IOP claim references unknown oracle {0}")]
    UnknownOracle(usize),

    /// The root-IOP commitment list repeated an oracle id.
    #[error("root-IOP commitment list repeats oracle {0}")]
    DuplicateOracle(usize),

    /// The oracle has the wrong length for this WARP/WHIR compiler.
    #[error("root-IOP oracle {oracle_id} has log length {actual}, expected {expected}")]
    OracleLogLengthMismatch {
        /// Oracle id assigned by the root-IOP recorder.
        oracle_id: usize,
        /// Expected log length for this compiler's RS code.
        expected: usize,
        /// Actual log length recorded with the commitment.
        actual: usize,
    },

    /// The oracle values do not match the recorded oracle metadata.
    #[error("root-IOP oracle {0} values do not match its committed field")]
    OracleValueFieldMismatch(usize),

    /// The oracle values have the wrong length for this compiler.
    #[error("root-IOP oracle {oracle_id} has {actual} values, expected {expected}")]
    OracleValueLengthMismatch {
        /// Oracle id assigned by the root-IOP recorder.
        oracle_id: usize,
        /// Expected codeword length for this compiler's RS code.
        expected: usize,
        /// Actual witness value count.
        actual: usize,
    },

    /// The proof did not contain the next expected oracle reduction.
    #[error("root-IOP proof is missing reduction for oracle {0}")]
    MissingOracleReduction(usize),

    /// The proof's oracle ordering did not match root-IOP commitment order.
    #[error("root-IOP proof order mismatch: expected oracle {expected}, got {actual}")]
    OracleReductionOrderMismatch {
        /// Expected oracle id.
        expected: usize,
        /// Actual oracle id found in the proof.
        actual: usize,
    },

    /// The proof's field tag disagreed with the public commitment metadata.
    #[error("root-IOP proof field mismatch for oracle {0}")]
    OracleReductionFieldMismatch(usize),

    /// The proof contained reductions for oracles not requested by the WARP
    /// verifier transcript.
    #[error("root-IOP proof contains trailing oracle reductions")]
    TrailingOracleReductions,
}

/// Mixed WHIR commitment used by the native WARP root proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "Comm: Serialize", deserialize = "Comm: Deserialize<'de>"))]
pub enum NativeWarpWhirRootCommitment<Comm> {
    /// Base-field fresh RS codeword committed by one WHIR PCS instance.
    Base(Comm),
    /// Base-field fresh message committed by WHIR; WARP codeword openings are
    /// compiled into linear Sigma claims over this message to avoid RS
    /// double-encoding.
    BaseMessage(Comm),
    /// Base-field fresh message committed as one column of a shared WHIR/MMCS
    /// batch root.
    BaseMessageShared {
        root: Comm,
        column: usize,
        width: usize,
    },
    /// Extension-field accumulator codeword committed limb-by-limb by WHIR.
    Extension(Vec<Comm>),
    /// Extension-field accumulator committed as one WHIR extension oracle.
    ExtensionNative(Comm),
    /// Extension-field accumulator message committed by WHIR; WARP codeword
    /// openings are compiled into linear Sigma claims over this message to
    /// avoid RS double-encoding.
    ExtensionMessage(Comm),
}

impl<Comm> NativeWarpWhirRootCommitment<Comm> {
    fn observe_payload_into<F, Challenger>(&self, challenger: &mut Challenger)
    where
        F: Field + PrimeCharacteristicRing,
        Comm: Clone,
        Challenger: FieldChallenger<F> + CanObserve<Comm>,
    {
        match self {
            Self::Base(commitment) => {
                challenger.observe(F::ZERO);
                challenger.observe(commitment.clone());
            }
            Self::BaseMessage(commitment) => {
                challenger.observe(F::from_u8(2));
                challenger.observe(commitment.clone());
            }
            Self::BaseMessageShared {
                root,
                column,
                width,
            } => {
                challenger.observe(F::from_u8(5));
                challenger.observe(root.clone());
                challenger.observe(F::from_usize(*column));
                challenger.observe(F::from_usize(*width));
            }
            Self::Extension(commitments) => {
                challenger.observe(F::ONE);
                challenger.observe(F::from_usize(commitments.len()));
                for commitment in commitments {
                    challenger.observe(commitment.clone());
                }
            }
            Self::ExtensionNative(commitment) => {
                challenger.observe(F::from_u8(3));
                challenger.observe(commitment.clone());
            }
            Self::ExtensionMessage(commitment) => {
                challenger.observe(F::from_u8(4));
                challenger.observe(commitment.clone());
            }
        }
    }
}

/// Prover data for a base oracle committed by the native WARP root proof.
pub struct NativeWarpWhirRootBaseProverData<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// WHIR deferred prover data for the base-field oracle.
    pub prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    /// Challenger state immediately after the deferred commitment phase.
    pub challenger: Challenger,
    /// Message committed by WHIR when the base oracle uses the single-RS path.
    ///
    /// `None` means the legacy path committed the already encoded WARP
    /// codeword as the WHIR message.
    pub message: Option<Vec<F>>,
}

/// Prover data for one base oracle committed as a column of a shared WHIR
/// batch root.
pub struct NativeWarpWhirRootSharedBaseProverData<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Shared WHIR root and Merkle data for all columns in the batch.
    pub shared: Arc<WhirSharedBaseDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
    /// Column of this oracle inside the shared batch.
    pub column: usize,
    /// Total number of columns in the shared batch.
    pub width: usize,
    /// Message committed in this column.
    pub message: Vec<F>,
}

/// Prover data for one extension oracle committed by the native WHIR
/// extension-initial path.
pub struct NativeWarpWhirRootExtensionProverData<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// WHIR deferred prover data for the extension-field oracle.
    pub prover_data: WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    /// Challenger state immediately after the deferred commitment phase.
    pub challenger: Challenger,
    /// Systematic accumulator message committed by WHIR when using the
    /// single-RS path.
    ///
    /// `None` means the legacy path committed the already encoded accumulator
    /// codeword as the WHIR message.
    pub message: Option<Vec<EF>>,
}

impl<F, EF, MT, Challenger, const DIGEST_ELEMS: usize> Clone
    for NativeWarpWhirRootExtensionProverData<F, EF, MT, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: Clone,
{
    fn clone(&self) -> Self {
        Self {
            prover_data: self.prover_data.clone(),
            challenger: self.challenger.clone(),
            message: self.message.clone(),
        }
    }
}

/// Prover data for one WARP root oracle.
pub enum NativeWarpWhirRootProverData<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Base-field fresh oracle.
    Base(NativeWarpWhirRootBaseProverData<F, EF, MT, Challenger, DIGEST_ELEMS>),
    /// Base-field fresh oracle committed as a shared-root message column.
    BaseShared(NativeWarpWhirRootSharedBaseProverData<F, EF, MT, DIGEST_ELEMS>),
    /// Extension-field accumulator oracle, decomposed into WHIR limbs.
    Extension(WhirLimbAccumulatorProverData<F, EF, MT, Challenger, DIGEST_ELEMS>),
    /// Extension-field accumulator oracle committed through WHIR's extension
    /// initial-oracle path.
    ExtensionNative(NativeWarpWhirRootExtensionProverData<F, EF, MT, Challenger, DIGEST_ELEMS>),
}

/// Prover data tagged with the root-IOP oracle id it belongs to.
pub struct NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Oracle id assigned by the root-IOP recorder.
    pub oracle_id: usize,
    /// Oracle-specific WHIR prover data.
    pub data: NativeWarpWhirRootProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
}

/// Residual-opening proof for one root-IOP oracle.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned",
    deserialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned"
))]
pub enum NativeWarpWhirRootOracleOpeningProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// WHIR opening proof for a base-field oracle.
    Base(WhirProof<F, EF, MT>),
    /// Limb-WHIR opening proof for an extension-field oracle.
    Extension(WhirLimbAccumulatorOpeningProof<F, EF, MT>),
    /// WHIR opening proof for a native extension-field initial oracle.
    ExtensionNative(WhirProof<F, EF, MT>),
    /// WHIR opening proof for an extension-field accumulator message.
    ExtensionMessage(WhirProof<F, EF, MT>),
}

/// One WHIR proof authenticating a batched root-IOP opening at once.
///
/// In the direct path, `reduction` combines all WARP root claims over all
/// message-domain oracles to one virtual same-point opening. In the fallback
/// residual path, it combines the residual openings returned by the per-oracle
/// root reductions. The `opening` proof is WHIR's batched-initial proof
/// against the original per-oracle roots.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned",
    deserialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned"
))]
pub struct NativeWarpWhirRootBatchedOpeningProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Multi-oracle linear-Sigma reduction of residual claims to one virtual
    /// opening.
    pub reduction: BatchedLinearSigmaReductionProof<F, EF>,
    /// Batched WHIR proof for the virtual opening against all original roots.
    pub opening: WhirProof<F, EF, MT>,
}

/// Complete native WARP root proof backed by WHIR residual openings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned",
    deserialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned"
))]
pub struct NativeWarpWhirRootProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Linear-Sigma reductions for every touched oracle.
    pub reductions: NativeWarpWhirRootReductionProof<F, EF>,
    /// Residual openings in the same order as `reductions.oracles`.
    pub openings: Vec<NativeWarpWhirRootOracleOpeningProof<F, EF, MT>>,
    /// Optional single batched opening replacing `openings` when all residuals
    /// are over message-domain WHIR roots.
    #[serde(default)]
    pub batched_opening: Option<NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>>,
    /// Optional direct batched root proof for message-domain WARP roots.
    ///
    /// This is the preferred single-RS path. It compiles all WARP root claims
    /// directly into one multi-oracle linear-Sigma reduction and one WHIR
    /// batched opening, avoiding the older per-oracle reduction followed by a
    /// second residual batching reduction.
    #[serde(default)]
    pub direct_batched_opening: Option<NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>>,
}

/// Errors from the complete native WARP root proof.
#[derive(Debug, Error)]
pub enum NativeWarpWhirRootProofError {
    /// The root reduction layer failed.
    #[error(transparent)]
    Reduction(#[from] NativeWarpWhirRootReductionError),

    /// Prover data for an expected oracle was missing.
    #[error("missing WHIR prover data for root oracle {0}")]
    MissingProverData(usize),

    /// A proof omitted the residual-opening proof for an expected oracle.
    #[error("missing residual-opening proof for root oracle {0}")]
    MissingOpening(usize),

    /// Prover data or proof kind did not match the oracle field.
    #[error("root oracle {0} has mismatched WHIR proof data kind")]
    OracleKindMismatch(usize),

    /// The WHIR opening backend returned malformed opened values.
    #[error("root oracle {0} returned malformed WHIR opened values")]
    OpeningShape(usize),

    /// The opened residual did not match the linear-Sigma residual value.
    #[error("root oracle {0} residual WHIR opening mismatch")]
    ResidualOpeningMismatch(usize),

    /// Base-oracle WHIR verification failed.
    #[error("base-oracle WHIR verifier failed for oracle {oracle_id}: {error:?}")]
    BaseOpening {
        /// Oracle id assigned by the root-IOP recorder.
        oracle_id: usize,
        /// WHIR verifier error.
        error: WhirVerifierError,
    },

    /// Batched residual WHIR verification failed.
    #[error("batched residual WHIR verifier failed: {0:?}")]
    BatchedOpening(WhirVerifierError),

    /// The caller requested message-domain base commitments without providing
    /// a WHIR PCS configured for the WARP message arity.
    #[error("base-message WHIR PCS is required for single-RS WARP base commitments")]
    BaseMessagePcsRequired,

    /// Extension-oracle limb-WHIR commitment failed.
    #[error("extension-oracle WHIR commit failed for oracle {oracle_id}: {error:?}")]
    ExtensionCommit {
        /// Oracle id assigned by the root-IOP recorder.
        oracle_id: usize,
        /// Limb PCS error.
        error: ExtensionLimbPcsError<WhirVerifierError>,
    },

    /// Extension-oracle limb-WHIR verification failed.
    #[error("extension-oracle WHIR verifier failed for oracle {oracle_id}: {error:?}")]
    ExtensionOpening {
        /// Oracle id assigned by the root-IOP recorder.
        oracle_id: usize,
        /// Limb PCS error.
        error: ExtensionLimbPcsError<WhirVerifierError>,
    },

    /// The proof contains extra residual-opening proofs.
    #[error("root proof contains trailing residual-opening proofs")]
    TrailingOpenings,
}

impl From<NativeWarpWhirClaimCompileError> for NativeWarpWhirRootProofError {
    fn from(error: NativeWarpWhirClaimCompileError) -> Self {
        Self::Reduction(NativeWarpWhirRootReductionError::ClaimCompile(error))
    }
}

impl From<LinearSigmaReductionError> for NativeWarpWhirRootProofError {
    fn from(error: LinearSigmaReductionError) -> Self {
        Self::Reduction(NativeWarpWhirRootReductionError::Reduction(error))
    }
}

/// Native WARP root proof system using WHIR for every residual opening.
pub struct NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    base_message_pcs: Option<&'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>>,
    limb_backend: WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>,
    compiler: NativeWarpWhirCompiler<'a, F, Dft>,
    challenger_seed: Challenger,
}

/// One WARP evaluation claim against the folded RS codeword oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeWarpWhirEvalClaim<EF> {
    /// Multilinear point in the codeword hypercube.
    pub point: Point<EF>,
    /// Claimed value of the folded codeword MLE at `point`.
    pub value: EF,
}

impl<EF> NativeWarpWhirEvalClaim<EF> {
    /// Create a new evaluation claim.
    pub const fn new(point: Point<EF>, value: EF) -> Self {
        Self { point, value }
    }
}

/// Statement emitted by the WARP layer for one folded oracle.
///
/// The constraints are linear Sigma constraints of the form
/// `sum_b a(b) * f_hat(b) = sigma`, ready for WHIR's constrained-RS compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeWarpWhirOracleStatement<EF> {
    /// Linear Sigma constraints over the folded codeword oracle.
    pub constraints: LinearSigmaStatement<EF>,
}

impl<EF> NativeWarpWhirOracleStatement<EF> {
    /// Create a statement from constraints.
    pub const fn new(constraints: LinearSigmaStatement<EF>) -> Self {
        Self { constraints }
    }
}

impl<EF: Field> NativeWarpWhirOracleStatement<EF> {
    /// Prove the compiled WARP linear-Sigma statement against a base-field
    /// folded RS oracle, reducing it to one residual opening claim.
    pub fn prove_reduction_base<F, Challenger>(
        &self,
        oracle: &Poly<F>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.constraints
            .prove_reduction_base(oracle, challenger, pow_bits)
    }

    /// Prove the compiled WARP linear-Sigma statement against an
    /// extension-field folded RS oracle.
    pub fn prove_reduction_ext<F, Challenger>(
        &self,
        oracle: &Poly<EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.constraints
            .prove_reduction_ext::<F, _>(oracle, challenger, pow_bits)
    }

    /// Verify the compiled WARP linear-Sigma reduction and return the residual
    /// opening claim to be checked against the committed RS oracle.
    pub fn verify_reduction<F, Challenger>(
        &self,
        proof: &LinearSigmaReductionProof<F, EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, LinearSigmaReductionError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.constraints
            .verify_reduction(proof, challenger, pow_bits)
    }

    /// Prove this compiled WARP statement against an existing deferred WHIR
    /// commitment, including the residual WHIR opening that binds the
    /// sumcheck reduction to the committed oracle.
    pub fn prove_bound_deferred<F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>(
        &self,
        pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        (LinearSigmaOpeningClaim<EF>, WhirLinearSigmaProof<F, EF, MT>),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField + Ord,
        EF: ExtensionField<F> + TwoAdicField,
        MT: Mmcs<F>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        pcs.open_linear_sigma_deferred(
            prover_data,
            &self.constraints,
            challenger,
            reduction_pow_bits,
        )
    }

    /// Verify a bound WARP/WHIR proof against the supplied commitment.
    pub fn verify_bound_deferred<F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>(
        &self,
        pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        commitment: &MT::Commitment,
        proof: &WhirLinearSigmaProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, WhirLinearSigmaError>
    where
        F: TwoAdicField + Ord,
        EF: ExtensionField<F> + TwoAdicField,
        MT: Mmcs<F>,
        MT::Commitment: PartialEq,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        pcs.verify_linear_sigma_deferred(
            commitment,
            &self.constraints,
            proof,
            challenger,
            reduction_pow_bits,
        )
    }

    /// Prove this compiled statement against an already committed
    /// extension-field WARP oracle.
    ///
    /// This is the extension-field counterpart to
    /// [`prove_bound_deferred`](Self::prove_bound_deferred). It uses the
    /// linear-Sigma sumcheck over the EF oracle itself, then asks the existing
    /// WARP accumulator point-opening backend to authenticate the single
    /// residual opening against the original commitment. With
    /// [`WhirLimbAccumulatorBackend`](crate::WhirLimbAccumulatorBackend), that
    /// backend is WHIR over base-field limbs; no verifier logic is reimplemented
    /// here.
    ///
    /// Fiat-Shamir order in this helper is:
    ///
    /// 1. observe the committed oracle through `backend.observe_commitment`,
    /// 2. bind the public linear-Sigma statement and run its reduction,
    /// 3. prove the residual opening against the same commitment.
    pub fn prove_bound_extension_points<F, Backend, Challenger>(
        &self,
        backend: &Backend,
        commitment: &Backend::Commitment,
        prover_data: &Backend::ProverData,
        oracle_values: &[EF],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaOpeningClaim<EF>,
            NativeWarpWhirPointProof<F, EF, Backend::PointProof>,
        ),
        NativeWarpWhirCompilerError<Backend::PointError>,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        backend.observe_commitment(challenger, commitment);

        let oracle = Poly::new(oracle_values.to_vec());
        let (reduction, residual_claim) = self.constraints.prove_reduction_ext::<F, _>(
            &oracle,
            challenger,
            reduction_pow_bits,
        )?;

        let opening_points = [vec![residual_claim.point.clone()]];
        let (opened_values, opening) = backend
            .prove_points(prover_data, &opening_points)
            .map_err(NativeWarpWhirCompilerError::PointOpening)?;
        let opened = opened_values
            .first()
            .and_then(|values| values.first())
            .copied()
            .ok_or(NativeWarpWhirCompilerError::OpeningShape)?;
        if opened != residual_claim.value {
            return Err(NativeWarpWhirCompilerError::ResidualOpeningMismatch);
        }

        Ok((
            residual_claim,
            NativeWarpWhirPointProof { reduction, opening },
        ))
    }

    /// Verify a compiled extension-field statement against an existing WARP
    /// accumulator commitment.
    ///
    /// Verification mirrors [`prove_bound_extension_points`](Self::prove_bound_extension_points):
    /// the commitment is observed before the linear-Sigma challenges are sampled,
    /// and the residual opening is checked by the caller-provided backend.
    pub fn verify_bound_extension_points<F, Backend, Challenger>(
        &self,
        backend: &Backend,
        commitment: &Backend::Commitment,
        proof: &NativeWarpWhirPointProof<F, EF, Backend::PointProof>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, NativeWarpWhirCompilerError<Backend::PointError>>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        backend.observe_commitment(challenger, commitment);

        let residual_claim = self.constraints.verify_reduction::<F, _>(
            &proof.reduction,
            challenger,
            reduction_pow_bits,
        )?;
        let opening_claims = [vec![(residual_claim.point.clone(), residual_claim.value)]];
        backend
            .verify_points(commitment, &opening_claims, &proof.opening)
            .map_err(NativeWarpWhirCompilerError::PointOpening)?;

        Ok(residual_claim)
    }
}

/// Compiler helper for WARP over Plonky3's systematic Reed-Solomon code.
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
                RootIopOpeningPoint::Index(index) => {
                    if *index >= self.code.codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
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

    /// Compile base-field WARP codeword index claims into a linear-Sigma
    /// statement over the original systematic RS message.
    ///
    /// This is the single-RS path for fresh WARP inputs. WHIR commits to the
    /// message `w` in its usual multilinear/RS form. Each WARP shift query
    /// `C(w)[i] = v` is converted into
    ///
    /// ```text
    ///     sum_j lambda_j(omega_n^i) * w[j] = v
    /// ```
    ///
    /// where `lambda_j` are the systematic RS Lagrange weights. This removes
    /// the previous `C(w)`-then-WHIR-encode-`C(w)` double encoding while keeping
    /// the WARP verifier transcript bound to the same codeword claim.
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
                RootIopOpeningPoint::Index(index) => {
                    if *index >= self.code.codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    let weights = self.code.systematic_codeword_index_weights::<EF>(*index);
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

    /// Compile extension-field accumulator codeword claims into a linear-Sigma
    /// statement over the original systematic accumulator message.
    ///
    /// This is the extension-field analogue of
    /// [`Self::root_iop_base_message_claim_statement`]. Index openings use the
    /// usual systematic RS Lagrange weights, while arbitrary codeword-MLE
    /// openings use the adjoint RS map from
    /// [`ReedSolomonCode::systematic_codeword_mle_weights`].
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
                RootIopOpeningPoint::Index(index) => {
                    if *index >= self.code.codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    self.code.systematic_codeword_index_weights::<EF>(*index)
                }
                RootIopOpeningPoint::Mle(point) => {
                    if point.len() != self.code.log_codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::PointArityMismatch {
                            oracle_id,
                        });
                    }
                    self.code.systematic_codeword_mle_weights::<EF>(point)
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

    /// Create a native WARP root proof system whose fresh base inputs are
    /// committed in the WHIR message domain.
    ///
    /// `pcs` remains the codeword-domain PCS used for extension accumulator
    /// limbs. `base_message_pcs` must be configured with
    /// `code.log_msg_len()` variables and the same RS rate/security settings.
    /// This is the sound single-RS path for systematic WARP: WHIR commits to
    /// the original fresh witness, and WARP codeword-index openings are compiled
    /// into linear Sigma claims using the systematic RS Lagrange weights.
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
    /// but the WHIR commitment is to `w`. During proof generation those
    /// codeword-index claims are transformed into linear Sigma claims over the
    /// message by [`ReedSolomonCode::systematic_codeword_index_weights`].
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
        challenger.observe(F::from_u64(ROOT_WHIR_BASE_ORACLE_TAG));
        challenger.observe(F::from_usize(width));
        for (oracle_id, _, _) in &inputs {
            challenger.observe(F::from_usize(*oracle_id));
        }
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
            if self.compiler.code().is_systematic() {
                let message = self
                    .compiler
                    .code()
                    .systematic_message_from_codeword(&codeword);
                let encoded = message_pcs
                    .encode_extension_initial_oracle(RowMajorMatrix::new(message.clone(), 1));
                let (commitment, prover_data) =
                    message_pcs.commit_extension_encoded_deferred(encoded, &mut challenger);
                return Ok((
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
                ));
            }
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

    /// Prove WARP root reductions and bind every residual opening with WHIR.
    pub fn prove(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<NativeWarpWhirRootProof<F, EF, MT>, NativeWarpWhirRootProofError> {
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

        transcript
            .verify_witnessed_claim_values()
            .map_err(NativeWarpWhirRootReductionError::RootIop)?;
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
                .check_bound_oracle_shape(commitment, Some(values))?;

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

        for commitment in commitments_to_observe {
            observe_native_root_commitment::<F, Challenger, MT::Commitment>(challenger, commitment);
        }

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
                RootIopOpeningPoint::Index(index) => {
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
                RootIopOpeningPoint::Index(index) => {
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
        challenger.observe(F::from_u64(ROOT_WHIR_BASE_ORACLE_TAG));
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

enum NativeWarpBatchedResidualPoly<F, EF> {
    Base(Poly<F>),
    Extension(Poly<EF>),
}

enum NativeWarpDirectBatchedResidualPoly<'a, F, EF> {
    Base(&'a [F]),
    Extension(&'a [EF]),
}

const COMPACT_ROOT_EVAL_PAR_THRESHOLD: usize = 1 << 14;

enum NativeWarpCompactEvalPoly<'a, F, EF> {
    Base(&'a [F]),
    ExtensionBorrowed(&'a [EF]),
    ExtensionOwned(Poly<EF>),
}

impl<'a, F, EF> NativeWarpCompactEvalPoly<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
{
    fn from_direct(poly: &'a NativeWarpDirectBatchedResidualPoly<'a, F, EF>) -> Self {
        match poly {
            NativeWarpDirectBatchedResidualPoly::Base(values) => Self::Base(values),
            NativeWarpDirectBatchedResidualPoly::Extension(values) => {
                Self::ExtensionBorrowed(values)
            }
        }
    }

    fn sumcheck_coefficients(&self, weights: &[EF]) -> (EF, EF) {
        match self {
            Self::Base(values) => VariableOrder::Prefix.sumcheck_coefficients(*values, weights),
            Self::ExtensionBorrowed(values) => {
                VariableOrder::Prefix.sumcheck_coefficients(*values, weights)
            }
            Self::ExtensionOwned(poly) => {
                VariableOrder::Prefix.sumcheck_coefficients(poly.as_slice(), weights)
            }
        }
    }

    fn fix_prefix_var_mut(&mut self, r: EF) {
        match self {
            Self::Base(values) => {
                *self = Self::ExtensionOwned(fix_base_prefix_to_extension::<F, EF>(values, r));
            }
            Self::ExtensionBorrowed(values) => {
                *self = Self::ExtensionOwned(fix_extension_prefix_to_owned::<EF>(values, r));
            }
            Self::ExtensionOwned(poly) => poly.fix_prefix_var_mut(r),
        }
    }

    fn final_value(&self) -> EF {
        match self {
            Self::Base(values) => EF::from(values[0]),
            Self::ExtensionBorrowed(values) => values[0],
            Self::ExtensionOwned(poly) => poly.as_slice()[0],
        }
    }
}

fn fix_base_prefix_to_extension<F, EF>(values: &[F], r: EF) -> Poly<EF>
where
    F: Field,
    EF: ExtensionField<F> + Send + Sync,
{
    let mid = values.len() / 2;
    let (lo, hi) = values.split_at(mid);
    let folded = if values.len() >= COMPACT_ROOT_EVAL_PAR_THRESHOLD {
        lo.par_iter()
            .zip(hi.par_iter())
            .map(|(&a0, &a1)| EF::from(a0) + r * EF::from(a1 - a0))
            .collect()
    } else {
        lo.iter()
            .zip(hi.iter())
            .map(|(&a0, &a1)| EF::from(a0) + r * EF::from(a1 - a0))
            .collect()
    };
    Poly::new(folded)
}

fn fix_extension_prefix_to_owned<EF>(values: &[EF], r: EF) -> Poly<EF>
where
    EF: Field,
{
    let mid = values.len() / 2;
    let (lo, hi) = values.split_at(mid);
    let folded = if values.len() >= COMPACT_ROOT_EVAL_PAR_THRESHOLD {
        lo.par_iter()
            .zip(hi.par_iter())
            .map(|(&a0, &a1)| a0 + r * (a1 - a0))
            .collect()
    } else {
        lo.iter()
            .zip(hi.iter())
            .map(|(&a0, &a1)| a0 + r * (a1 - a0))
            .collect()
    };
    Poly::new(folded)
}

enum NativeWarpBatchedResidualCommitment<Comm> {
    Base(Comm),
    Extension(Comm),
    SharedBase {
        root: Comm,
        column: usize,
        width: usize,
    },
}

enum NativeWarpBatchedResidualProverOracle<F, EF, MT, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    Base(WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>),
    Extension(WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>),
    SharedBase {
        shared: Arc<WhirSharedBaseDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
        column: usize,
        width: usize,
    },
}

#[derive(Clone, Debug)]
struct NativeWarpCompactRootStatement<EF> {
    num_variables: usize,
    constraints: Vec<NativeWarpCompactRootConstraint<EF>>,
}

#[derive(Clone, Debug)]
struct NativeWarpCompactRootConstraint<EF> {
    query: NativeWarpCompactRootQuery<EF>,
    target: EF,
}

#[derive(Clone, Debug)]
enum NativeWarpCompactRootQuery<EF> {
    Index(usize),
    Mle(Vec<EF>),
}

impl<EF> NativeWarpCompactRootStatement<EF>
where
    EF: Field,
{
    fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    fn add_index(&mut self, index: usize, target: EF) {
        self.constraints.push(NativeWarpCompactRootConstraint {
            query: NativeWarpCompactRootQuery::Index(index),
            target,
        });
    }

    fn add_mle(&mut self, point: Vec<EF>, target: EF) {
        self.constraints.push(NativeWarpCompactRootConstraint {
            query: NativeWarpCompactRootQuery::Mle(point),
            target,
        });
    }

    fn observe_and_sample_gamma<F, Challenger>(&self, challenger: &mut Challenger) -> EF
    where
        F: Field + PrimeCharacteristicRing,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + CanObserve<F>,
    {
        challenger.observe(F::from_usize(self.num_variables));
        challenger.observe(F::from_usize(self.constraints.len()));
        for constraint in &self.constraints {
            match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    challenger.observe(F::ZERO);
                    challenger.observe(F::from_usize(*index));
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    challenger.observe(F::ONE);
                    challenger.observe(F::from_usize(point.len()));
                    for &coord in point {
                        challenger.observe_algebra_element(coord);
                    }
                }
            }
            challenger.observe_algebra_element(constraint.target);
        }
        challenger.sample_algebra_element()
    }

    fn batched_target(&self, gamma: EF) -> EF {
        let mut scale = EF::ONE;
        let mut target = EF::ZERO;
        for constraint in &self.constraints {
            target += scale * constraint.target;
            scale *= gamma;
        }
        target
    }

    fn batched_weight_eval_from_encoded_eq<F>(
        &self,
        encoded_message_eq: &[EF],
        encoded_message_eq_poly: &Poly<EF>,
        gamma: EF,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let mut scale = EF::ONE;
        let mut value = EF::ZERO;
        for constraint in &self.constraints {
            let local = match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => encoded_message_eq[*index],
                NativeWarpCompactRootQuery::Mle(point) => {
                    encoded_message_eq_poly.eval_ext::<F>(&Point::new(point.clone()))
                }
            };
            value += scale * local;
            scale *= gamma;
        }
        value
    }

    fn batched_weight_eval_from_message_eq_point<F, Dft>(
        &self,
        code: &ReedSolomonCode<F, Dft>,
        message_point: &[EF],
        gamma: EF,
    ) -> Option<EF>
    where
        F: TwoAdicField,
        Dft: TwoAdicSubgroupDft<F>,
    {
        let stride = 1 << code.log_inv_rate();
        let mut scale = EF::ONE;
        let mut value = EF::ZERO;
        for constraint in &self.constraints {
            let local = match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    if !index.is_multiple_of(stride) {
                        return None;
                    }
                    eval_eq_at_hypercube_index(message_point, index / stride)
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    let (prefix, suffix) = point.split_at(code.log_msg_len());
                    if point.len() != code.log_codeword_len()
                        || suffix.iter().any(|&coord| coord != EF::ZERO)
                    {
                        return None;
                    }
                    eval_eq_point(message_point, prefix)
                }
            };
            value += scale * local;
            scale *= gamma;
        }
        Some(value)
    }
}

fn compact_batched_root_weights<F, EF, Dft>(
    code: &ReedSolomonCode<F, Dft>,
    statements: &[NativeWarpCompactRootStatement<EF>],
    gammas: &[EF],
) -> Result<Vec<Poly<EF>>, LinearSigmaReductionError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
{
    if statements.len() != gammas.len() {
        return Err(LinearSigmaReductionError::ArityMismatch {
            expected: statements.len(),
            actual: gammas.len(),
        });
    }

    let stride = 1 << code.log_inv_rate();
    let mut weights = (0..statements.len()).map(|_| None).collect::<Vec<_>>();
    let mut dense_statement_indices = Vec::new();
    let mut dense_query_columns = Vec::new();

    for (statement_index, (statement, &gamma)) in statements.iter().zip(gammas).enumerate() {
        if statement.constraints.is_empty() {
            return Err(LinearSigmaReductionError::EmptyStatement);
        }
        if statement.num_variables != code.log_msg_len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: statement.num_variables,
                actual: code.log_msg_len(),
            });
        }

        let all_systematic_indices = statement.constraints.iter().all(|constraint| {
            matches!(
                constraint.query,
                NativeWarpCompactRootQuery::Index(index) if index.is_multiple_of(stride)
            )
        });
        if all_systematic_indices {
            let mut message_weights = EF::zero_vec(code.msg_len());
            let mut scale = EF::ONE;
            for constraint in &statement.constraints {
                let NativeWarpCompactRootQuery::Index(index) = constraint.query else {
                    unreachable!("all queries are systematic indices");
                };
                message_weights[index / stride] += scale;
                scale *= gamma;
            }
            weights[statement_index] = Some(Poly::new(message_weights));
            continue;
        }

        let mut codeword_query = EF::zero_vec(code.codeword_len());
        let mut scale = EF::ONE;
        for constraint in &statement.constraints {
            match &constraint.query {
                NativeWarpCompactRootQuery::Index(index) => {
                    codeword_query[*index] += scale;
                }
                NativeWarpCompactRootQuery::Mle(point) => {
                    if point.len() != code.log_codeword_len() {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: code.log_codeword_len(),
                            actual: point.len(),
                        });
                    }
                    let eq = Poly::<EF>::new_from_point(point, scale);
                    for (slot, &value) in codeword_query.iter_mut().zip(eq.as_slice()) {
                        *slot += value;
                    }
                }
            }
            scale *= gamma;
        }
        dense_statement_indices.push(statement_index);
        dense_query_columns.push(codeword_query);
    }

    if !dense_query_columns.is_empty() {
        let width = dense_query_columns.len();
        let mut matrix_values = EF::zero_vec(code.codeword_len() * width);
        for (col, query) in dense_query_columns.iter().enumerate() {
            for (row, &value) in query.iter().enumerate() {
                matrix_values[row * width + col] = value;
            }
        }
        let message_weights =
            code.systematic_codeword_query_weights_batch(RowMajorMatrix::new(matrix_values, width));
        for (col, &statement_index) in dense_statement_indices.iter().enumerate() {
            let column = (0..code.msg_len())
                .map(|row| message_weights.values[row * width + col])
                .collect::<Vec<_>>();
            weights[statement_index] = Some(Poly::new(column));
        }
    }

    weights
        .into_iter()
        .map(|weight| weight.ok_or(LinearSigmaReductionError::EmptyStatement))
        .collect()
}

fn prove_compact_batched_root_reduction<F, EF, Dft, Challenger>(
    code: &ReedSolomonCode<F, Dft>,
    statements: &[NativeWarpCompactRootStatement<EF>],
    polys: &[NativeWarpDirectBatchedResidualPoly<'_, F, EF>],
    challenger: &mut Challenger,
    pow_bits: usize,
) -> Result<
    (
        BatchedLinearSigmaReductionProof<F, EF>,
        BatchedLinearSigmaOpeningClaim<EF>,
    ),
    LinearSigmaReductionError,
>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if statements.is_empty() {
        return Err(LinearSigmaReductionError::EmptyStatement);
    }
    if statements.len() != polys.len() {
        return Err(LinearSigmaReductionError::ArityMismatch {
            expected: statements.len(),
            actual: polys.len(),
        });
    }

    challenger.observe(F::from_usize(statements.len()));
    let mut gammas = Vec::with_capacity(statements.len());
    let mut targets = Vec::with_capacity(statements.len());
    for statement in statements {
        let gamma = statement.observe_and_sample_gamma::<F, _>(challenger);
        gammas.push(gamma);
        targets.push(statement.batched_target(gamma));
    }

    let mut evals = Vec::with_capacity(statements.len());
    let mut weights = compact_batched_root_weights(code, statements, &gammas)?;
    for ((poly, weight), &target) in polys.iter().zip(&weights).zip(&targets) {
        match poly {
            NativeWarpDirectBatchedResidualPoly::Base(values) => {
                if values.len() != code.msg_len() {
                    return Err(LinearSigmaReductionError::ArityMismatch {
                        expected: code.log_msg_len(),
                        actual: log2_strict_usize(values.len()),
                    });
                }
                let actual = weight
                    .as_slice()
                    .iter()
                    .zip(values.iter())
                    .map(|(&weight, &value)| weight * value)
                    .sum::<EF>();
                if actual != target {
                    return Err(LinearSigmaReductionError::UnsatisfiedStatement);
                }
                evals.push(NativeWarpCompactEvalPoly::from_direct(poly));
            }
            NativeWarpDirectBatchedResidualPoly::Extension(values) => {
                if values.len() != code.msg_len() {
                    return Err(LinearSigmaReductionError::ArityMismatch {
                        expected: code.log_msg_len(),
                        actual: log2_strict_usize(values.len()),
                    });
                }
                let actual = weight
                    .as_slice()
                    .iter()
                    .zip(values.iter())
                    .map(|(&weight, &value)| weight * value)
                    .sum::<EF>();
                if actual != target {
                    return Err(LinearSigmaReductionError::UnsatisfiedStatement);
                }
                evals.push(NativeWarpCompactEvalPoly::from_direct(poly));
            }
        }
    }

    let rho: EF = challenger.sample_algebra_element();
    let scales = rho.powers().take(statements.len()).collect();
    let mut claimed_sum = scales
        .iter()
        .zip(&targets)
        .map(|(&scale, &target)| scale * target)
        .sum::<EF>();

    let mut sumcheck = SumcheckData::default();
    let mut point = Vec::with_capacity(code.log_msg_len());
    for _ in 0..code.log_msg_len() {
        let mut c0 = EF::ZERO;
        let mut c_inf = EF::ZERO;
        for ((evals, weights), &scale) in evals.iter().zip(weights.iter()).zip(&scales) {
            let (local_c0, local_c_inf) = evals.sumcheck_coefficients(weights.as_slice());
            c0 += scale * local_c0;
            c_inf += scale * local_c_inf;
        }

        let r = sumcheck.observe_and_sample(challenger, c0, c_inf, pow_bits);
        for (evals, weights) in evals.iter_mut().zip(weights.iter_mut()) {
            evals.fix_prefix_var_mut(r);
            weights.fix_prefix_var_mut(r);
        }
        claimed_sum = extrapolate_01inf(c0, claimed_sum - c0, c_inf, r);
        point.push(r);
    }

    let coeffs = scales
        .iter()
        .zip(&weights)
        .map(|(&scale, weights)| scale * weights.as_slice()[0])
        .collect::<Vec<_>>();
    let virtual_eval = coeffs
        .iter()
        .zip(&evals)
        .map(|(&coeff, evals)| coeff * evals.final_value())
        .sum::<EF>();
    if claimed_sum != virtual_eval {
        return Err(LinearSigmaReductionError::FinalCheckFailed);
    }

    challenger.observe_algebra_element(virtual_eval);

    Ok((
        BatchedLinearSigmaReductionProof {
            sumcheck,
            virtual_eval,
        },
        BatchedLinearSigmaOpeningClaim {
            point: Point::new(point),
            coeffs,
            value: virtual_eval,
        },
    ))
}

fn verify_compact_batched_root_reduction<F, EF, Dft, Challenger>(
    code: &ReedSolomonCode<F, Dft>,
    statements: &[NativeWarpCompactRootStatement<EF>],
    proof: &BatchedLinearSigmaReductionProof<F, EF>,
    challenger: &mut Challenger,
    pow_bits: usize,
) -> Result<BatchedLinearSigmaOpeningClaim<EF>, LinearSigmaReductionError>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField + Send + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if statements.is_empty() {
        return Err(LinearSigmaReductionError::EmptyStatement);
    }
    if proof.sumcheck.num_rounds() != code.log_msg_len() {
        return Err(SumcheckError::RoundCountMismatch {
            expected: code.log_msg_len(),
            actual: proof.sumcheck.num_rounds(),
        }
        .into());
    }

    challenger.observe(F::from_usize(statements.len()));
    let mut gammas = Vec::with_capacity(statements.len());
    let mut targets = Vec::with_capacity(statements.len());
    for statement in statements {
        let gamma = statement.observe_and_sample_gamma::<F, _>(challenger);
        gammas.push(gamma);
        targets.push(statement.batched_target(gamma));
    }

    let rho: EF = challenger.sample_algebra_element();
    let scales = rho.powers().take(statements.len()).collect();
    let mut claimed_sum = scales
        .iter()
        .zip(&targets)
        .map(|(&scale, &target)| scale * target)
        .sum::<EF>();

    let point = proof
        .sumcheck
        .verify_rounds(challenger, &mut claimed_sum, pow_bits)?;
    challenger.observe_algebra_element(proof.virtual_eval);
    if claimed_sum != proof.virtual_eval {
        return Err(LinearSigmaReductionError::FinalCheckFailed);
    }

    let mut encoded_message_eq = None;
    let mut encoded_message_eq_poly = None;
    let coeffs = scales
        .iter()
        .zip(&gammas)
        .zip(statements)
        .map(|((&scale, &gamma), statement)| {
            let local = statement
                .batched_weight_eval_from_message_eq_point::<F, Dft>(code, point.as_slice(), gamma)
                .unwrap_or_else(|| {
                    let encoded = encoded_message_eq.get_or_insert_with(|| {
                        let message_eq = Poly::<EF>::new_from_point(point.as_slice(), EF::ONE);
                        code.encode_algebra(message_eq.as_slice())
                    });
                    let poly =
                        encoded_message_eq_poly.get_or_insert_with(|| Poly::new(encoded.clone()));
                    statement.batched_weight_eval_from_encoded_eq::<F>(encoded, poly, gamma)
                });
            scale * local
        })
        .collect::<Vec<_>>();

    Ok(BatchedLinearSigmaOpeningClaim {
        point,
        coeffs,
        value: proof.virtual_eval,
    })
}

fn residual_eq_statement<F, EF>(
    residual: &NativeWarpWhirRootResidualClaim<EF>,
) -> LinearSigmaStatement<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut eq = EqStatement::initialize(residual.opening.point.num_variables());
    eq.add_evaluated_constraint(residual.opening.point.clone(), residual.opening.value);
    let mut statement = LinearSigmaStatement::initialize(residual.opening.point.num_variables());
    statement.add_constraint(LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE));
    statement
}

fn observe_native_root_commitment<F, Challenger, Comm>(
    challenger: &mut Challenger,
    commitment: &RootIopBoundCommitment<NativeWarpWhirRootCommitment<Comm>>,
) where
    F: Field + PrimeCharacteristicRing,
    Comm: Clone,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    challenger.observe(F::from_usize(commitment.oracle_id));
    challenger.observe(F::from_usize(commitment.log_len));
    challenger.observe(F::from_usize(match commitment.field {
        RootIopOracleField::Base => 0,
        RootIopOracleField::Extension => 1,
    }));
    commitment
        .commitment
        .observe_payload_into::<F, _>(challenger);
}

fn eval_eq_point<EF: Field>(lhs: &[EF], rhs: &[EF]) -> EF {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(&l, &r)| l * r + (EF::ONE - l) * (EF::ONE - r))
        .product()
}

fn eval_eq_at_hypercube_index<EF: Field>(point: &[EF], index: usize) -> EF {
    let num_variables = point.len();
    point
        .iter()
        .enumerate()
        .map(|(bit, &coord)| {
            if (index >> (num_variables - 1 - bit)) & 1 == 1 {
                coord
            } else {
                EF::ONE - coord
            }
        })
        .product()
}

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
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_whir::parameters::{
        FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::code::ReedSolomonCode;
    use crate::finalize::WhirLimbAccumulatorBackend;
    use crate::protocol::AccumulatorCommitmentBackend;
    use crate::root_iop::{
        RootIopOpeningClaim, RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleField,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Dft = Radix2DFTSmallBatch<F>;
    type Perm = Poseidon2BabyBear<16>;
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type TestWhirPcs = WhirPcs<EF, F, MyMmcs, TestChallenger, Dft, 8>;

    fn challenger() -> TestChallenger {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(2));
        DuplexChallenger::new(perm)
    }

    fn systematic_code() -> ReedSolomonCode<F, Dft> {
        ReedSolomonCode::new_systematic(2, 1, Dft::default())
    }

    fn whir_pcs(num_variables: usize) -> TestWhirPcs {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(3));
        let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(2),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        TestWhirPcs::new(
            num_variables,
            params,
            Dft::default(),
            SumcheckStrategy::Classic,
        )
    }

    #[test]
    fn folded_oracle_eval_claim_becomes_linear_sigma() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let codeword = code.encode(&witness);
        let codeword_poly = Poly::new(codeword);
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let value = codeword_poly.eval_base(&point);

        let claim = NativeWarpWhirEvalClaim::new(point, value);
        let constraint = compiler.eval_claim_constraint(&claim);

        assert!(constraint.verify_base(&codeword_poly));
    }

    #[test]
    fn compiled_eval_claims_reduce_to_one_opening() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let codeword_poly = Poly::new(code.encode(&witness));
        let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let point1 = Point::new(vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]);
        let claims = eval_claims_from_parts(
            &[point0.clone(), point1.clone()],
            &[
                codeword_poly.eval_base(&point0),
                codeword_poly.eval_base(&point1),
            ],
        );
        let statement = compiler.eval_claim_statement(&claims);

        let mut prover_challenger = challenger();
        let mut verifier_challenger = challenger();
        let (proof, opening) = statement
            .prove_reduction_base::<F, _>(&codeword_poly, &mut prover_challenger, 0)
            .expect("honest WARP/WHIR reduction");
        let verified_opening = statement
            .verify_reduction::<F, _>(&proof, &mut verifier_challenger, 0)
            .expect("WARP/WHIR reduction verification");

        assert_eq!(opening, verified_opening);
        assert_eq!(opening.value, codeword_poly.eval_base(&opening.point));
    }

    #[test]
    fn compiled_wrong_eval_claim_does_not_prove() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let codeword_poly = Poly::new(code.encode(&witness));
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let bad_claim =
            NativeWarpWhirEvalClaim::new(point.clone(), codeword_poly.eval_base(&point) + EF::ONE);
        let statement = compiler.eval_claim_statement(&[bad_claim]);

        let err = statement
            .prove_reduction_base::<F, _>(&codeword_poly, &mut challenger(), 0)
            .expect_err("wrong claim should not produce an honest proof");
        assert!(matches!(
            err,
            LinearSigmaReductionError::UnsatisfiedStatement
        ));
    }

    #[test]
    fn compiled_eval_claims_bind_to_whir_commitment() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let codeword = code.encode(&witness);
        let codeword_poly = Poly::new(codeword.clone());
        let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let point1 = Point::new(vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]);
        let claims = eval_claims_from_parts(
            &[point0.clone(), point1.clone()],
            &[
                codeword_poly.eval_base(&point0),
                codeword_poly.eval_base(&point1),
            ],
        );
        let statement = compiler.eval_claim_statement(&claims);
        let pcs = whir_pcs(code.log_codeword_len());

        let mut prover_challenger = challenger();
        let (commitment, prover_data) =
            pcs.commit_deferred(RowMajorMatrix::new(codeword, 1), &mut prover_challenger);
        let (opening, proof) = statement
            .prove_bound_deferred(&pcs, prover_data, &mut prover_challenger, 0)
            .expect("bound WARP/WHIR proof");

        let mut verifier_challenger = challenger();
        let verified_opening = statement
            .verify_bound_deferred(&pcs, &commitment, &proof, &mut verifier_challenger, 0)
            .expect("bound WARP/WHIR verification");

        assert_eq!(opening, verified_opening);
        assert_eq!(opening.value, codeword_poly.eval_base(&opening.point));
    }

    #[test]
    fn root_iop_index_claim_compiles_with_warp_bit_order() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let codeword = code.encode(&witness);
        let index = 5;
        let claims: Vec<RootIopOpeningClaim<F, EF>> = vec![RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 9,
            point: RootIopOpeningPoint::<EF>::Index(index),
            value: RootIopOpeningValue::Base(codeword[index]),
        }];

        let statement = compiler
            .root_iop_claim_statement(&claims, 9, RootIopOracleField::Base)
            .expect("root-IOP base claim statement");

        assert!(statement.constraints.verify_base(&Poly::new(codeword)));
    }

    #[test]
    fn root_iop_extension_mle_claim_compiles() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let oracle = (0..code.codeword_len())
            .map(|i| EF::from_u64((3 * i + 5) as u64))
            .collect::<Vec<_>>();
        let poly = Poly::new(oracle.clone());
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let value = poly.eval_ext::<F>(&point);
        let claims: Vec<RootIopOpeningClaim<F, EF>> = vec![RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 3,
            point: RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
            value: RootIopOpeningValue::Extension(value),
        }];

        let statement = compiler
            .root_iop_claim_statement(&claims, 3, RootIopOracleField::Extension)
            .expect("root-IOP extension claim statement");

        assert!(statement.constraints.verify_ext(&poly));
    }

    #[test]
    fn compiled_extension_claims_bind_to_whir_limb_backend() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let oracle = (0..code.codeword_len())
            .map(|i| EF::from_u64((11 * i + 7) as u64))
            .collect::<Vec<_>>();
        let poly = Poly::new(oracle.clone());
        let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let point1 = Point::new(vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]);
        let claims = eval_claims_from_parts(
            &[point0.clone(), point1.clone()],
            &[poly.eval_ext::<F>(&point0), poly.eval_ext::<F>(&point1)],
        );
        let statement = compiler.eval_claim_statement(&claims);
        let pcs = whir_pcs(code.log_codeword_len());
        let backend =
            WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger, Dft, 8>::new(&pcs, challenger());
        let (commitment, prover_data) = backend
            .commit(oracle.clone())
            .expect("WHIR limb accumulator commit");

        let mut prover_challenger = challenger();
        let (opening, proof) = statement
            .prove_bound_extension_points::<F, _, _>(
                &backend,
                &commitment,
                &prover_data,
                &oracle,
                &mut prover_challenger,
                0,
            )
            .expect("bound extension WARP/WHIR proof");

        let mut verifier_challenger = challenger();
        let verified_opening = statement
            .verify_bound_extension_points::<F, _, _>(
                &backend,
                &commitment,
                &proof,
                &mut verifier_challenger,
                0,
            )
            .expect("bound extension WARP/WHIR verification");

        assert_eq!(opening, verified_opening);
        assert_eq!(opening.value, poly.eval_ext::<F>(&opening.point));
    }

    #[test]
    fn compiled_extension_claims_reject_tampered_reduction() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let oracle = (0..code.codeword_len())
            .map(|i| EF::from_u64((13 * i + 17) as u64))
            .collect::<Vec<_>>();
        let poly = Poly::new(oracle.clone());
        let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
        let statement = compiler.eval_claim_statement(&[NativeWarpWhirEvalClaim::new(
            point.clone(),
            poly.eval_ext::<F>(&point),
        )]);
        let pcs = whir_pcs(code.log_codeword_len());
        let backend =
            WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger, Dft, 8>::new(&pcs, challenger());
        let (commitment, prover_data) = backend
            .commit(oracle.clone())
            .expect("WHIR limb accumulator commit");
        let (_, mut proof) = statement
            .prove_bound_extension_points::<F, _, _>(
                &backend,
                &commitment,
                &prover_data,
                &oracle,
                &mut challenger(),
                0,
            )
            .expect("bound extension proof");
        proof.reduction.oracle_eval += EF::ONE;

        let err = statement
            .verify_bound_extension_points::<F, _, _>(
                &backend,
                &commitment,
                &proof,
                &mut challenger(),
                0,
            )
            .expect_err("tampered reduction should be rejected");
        assert!(matches!(
            err,
            NativeWarpWhirCompilerError::Reduction(LinearSigmaReductionError::FinalCheckFailed)
        ));
    }

    #[test]
    fn root_iop_transcript_claims_reduce_in_commitment_order() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let base_witness = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let base_oracle = code.encode(&base_witness);
        let ext_oracle = (0..code.codeword_len())
            .map(|i| EF::from_u64((19 * i + 23) as u64))
            .collect::<Vec<_>>();
        let ext_poly = Poly::new(ext_oracle.clone());
        let ext_point = Point::new(vec![EF::from_u64(2), EF::from_u64(5), EF::from_u64(7)]);
        let commitments = vec![
            RootIopBoundCommitment {
                oracle_id: 0,
                log_len: code.log_codeword_len(),
                field: RootIopOracleField::Base,
                commitment: F::from_u64(101),
            },
            RootIopBoundCommitment {
                oracle_id: 1,
                log_len: code.log_codeword_len(),
                field: RootIopOracleField::Extension,
                commitment: F::from_u64(202),
            },
        ];
        let claims = vec![
            RootIopOpeningClaim {
                claim_id: 0,
                oracle_id: 0,
                point: RootIopOpeningPoint::Index(5),
                value: RootIopOpeningValue::Base(base_oracle[5]),
            },
            RootIopOpeningClaim {
                claim_id: 1,
                oracle_id: 1,
                point: RootIopOpeningPoint::Mle(ext_point.as_slice().to_vec()),
                value: RootIopOpeningValue::Extension(ext_poly.eval_ext::<F>(&ext_point)),
            },
        ];
        let transcript = RootIopBoundTranscript {
            oracles: vec![
                (
                    commitments[0].clone(),
                    RootIopOracleValues::Base(base_oracle),
                ),
                (
                    commitments[1].clone(),
                    RootIopOracleValues::Extension(ext_oracle),
                ),
            ],
            claims: claims.clone(),
        };

        let (prover_residuals, proof) = compiler
            .prove_root_iop_reductions(&transcript, &mut challenger(), 0)
            .expect("honest root-IOP reductions");
        let verifier_residuals = compiler
            .verify_root_iop_reductions(&commitments, &claims, &proof, &mut challenger(), 0)
            .expect("root-IOP reduction verification");

        assert_eq!(prover_residuals, verifier_residuals);
        assert_eq!(proof.oracles.len(), 2);
        assert_eq!(proof.oracles[0].oracle_id, 0);
        assert_eq!(proof.oracles[1].oracle_id, 1);
    }

    #[test]
    fn root_iop_reductions_reject_tampered_public_claim() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let oracle = code.encode(&witness);
        let commitment = RootIopBoundCommitment {
            oracle_id: 0,
            log_len: code.log_codeword_len(),
            field: RootIopOracleField::Base,
            commitment: F::from_u64(303),
        };
        let claims = vec![RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(6),
            value: RootIopOpeningValue::Base(oracle[6]),
        }];
        let transcript = RootIopBoundTranscript {
            oracles: vec![(commitment.clone(), RootIopOracleValues::Base(oracle))],
            claims: claims.clone(),
        };
        let (_, proof) = compiler
            .prove_root_iop_reductions(&transcript, &mut challenger(), 0)
            .expect("honest root-IOP reduction");
        let mut tampered_claims = claims;
        tampered_claims[0].value = RootIopOpeningValue::Base(F::from_u64(999));

        assert!(
            compiler
                .verify_root_iop_reductions(
                    &[commitment],
                    &tampered_claims,
                    &proof,
                    &mut challenger(),
                    0
                )
                .is_err()
        );
    }

    #[test]
    fn root_iop_residuals_bind_to_whir_openings() {
        let code = systematic_code();
        let pcs = whir_pcs(code.log_codeword_len());
        let root_system = NativeWarpWhirRootProofSystem::new(&pcs, &code, challenger());
        let base_witness = vec![
            F::from_u64(1),
            F::from_u64(4),
            F::from_u64(9),
            F::from_u64(16),
        ];
        let base_oracle = code.encode(&base_witness);
        let ext_oracle = (0..code.codeword_len())
            .map(|i| EF::from_u64((29 * i + 31) as u64))
            .collect::<Vec<_>>();
        let ext_poly = Poly::new(ext_oracle.clone());
        let ext_point = Point::new(vec![EF::from_u64(3), EF::from_u64(5), EF::from_u64(11)]);
        let (base_commitment, base_prover_data) = root_system
            .commit_base_oracle(0, base_oracle.clone())
            .expect("base root oracle commit");
        let (extension_commitment, extension_prover_data) = root_system
            .commit_extension_oracle(1, ext_oracle.clone())
            .expect("extension root oracle commit");
        let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
        let claims = vec![
            RootIopOpeningClaim {
                claim_id: 0,
                oracle_id: 0,
                point: RootIopOpeningPoint::<EF>::Index(3),
                value: RootIopOpeningValue::Base(base_oracle[3]),
            },
            RootIopOpeningClaim {
                claim_id: 1,
                oracle_id: 1,
                point: RootIopOpeningPoint::Mle(ext_point.as_slice().to_vec()),
                value: RootIopOpeningValue::Extension(ext_poly.eval_ext::<F>(&ext_point)),
            },
        ];
        let transcript = RootIopBoundTranscript {
            oracles: vec![
                (base_commitment, RootIopOracleValues::Base(base_oracle)),
                (
                    extension_commitment,
                    RootIopOracleValues::Extension(ext_oracle),
                ),
            ],
            claims: claims.clone(),
        };
        let proof = root_system
            .prove(
                &transcript,
                &[base_prover_data, extension_prover_data],
                &mut challenger(),
                0,
            )
            .expect("WHIR-bound root proof");

        let residuals = root_system
            .verify(&commitments, &claims, &proof, &mut challenger(), 0)
            .expect("WHIR-bound root proof verification");

        assert_eq!(residuals.len(), 2);
        assert_eq!(proof.reductions.oracles.len(), 2);
        assert_eq!(proof.openings.len(), 2);
    }

    #[test]
    fn root_iop_whir_bound_proof_rejects_tampered_claim() {
        let code = systematic_code();
        let pcs = whir_pcs(code.log_codeword_len());
        let root_system = NativeWarpWhirRootProofSystem::new(&pcs, &code, challenger());
        let witness = vec![
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
        ];
        let oracle = code.encode(&witness);
        let (commitment, prover_data) = root_system
            .commit_base_oracle(0, oracle.clone())
            .expect("base root oracle commit");
        let claims = vec![RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(4),
            value: RootIopOpeningValue::Base(oracle[4]),
        }];
        let transcript = RootIopBoundTranscript {
            oracles: vec![(commitment.clone(), RootIopOracleValues::Base(oracle))],
            claims: claims.clone(),
        };
        let proof = root_system
            .prove(&transcript, &[prover_data], &mut challenger(), 0)
            .expect("WHIR-bound root proof");
        let mut tampered_claims = claims;
        tampered_claims[0].value = RootIopOpeningValue::Base(F::from_u64(1234));

        assert!(
            root_system
                .verify(
                    &[commitment],
                    &tampered_claims,
                    &proof,
                    &mut challenger(),
                    0
                )
                .is_err()
        );
    }

    #[test]
    fn message_domain_root_proof_batches_residual_openings() {
        let code = systematic_code();
        let pcs = whir_pcs(code.log_codeword_len());
        let message_pcs = whir_pcs(code.log_msg_len());
        let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
            &pcs,
            &message_pcs,
            &code,
            challenger(),
        );
        let base_message = vec![
            F::from_u64(2),
            F::from_u64(5),
            F::from_u64(8),
            F::from_u64(13),
        ];
        let base_codeword = code.encode(&base_message);
        let extension_message = (0..code.msg_len())
            .map(|i| EF::from_u64((17 * i + 19) as u64))
            .collect::<Vec<_>>();
        let extension_codeword = code.encode_algebra(&extension_message);
        let extension_poly = Poly::new(extension_codeword.clone());
        let extension_point = Point::new(vec![EF::from_u64(3), EF::from_u64(7), EF::from_u64(11)]);

        let (base_commitment, base_prover_data) = root_system
            .commit_base_message_oracle(0, base_codeword.clone(), base_message)
            .expect("base message root oracle commit");
        let (extension_commitment, extension_prover_data) = root_system
            .commit_extension_oracle(1, extension_codeword.clone())
            .expect("extension message root oracle commit");
        let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
        let claims = vec![
            RootIopOpeningClaim {
                claim_id: 0,
                oracle_id: 0,
                point: RootIopOpeningPoint::<EF>::Index(4),
                value: RootIopOpeningValue::Base(base_codeword[4]),
            },
            RootIopOpeningClaim {
                claim_id: 1,
                oracle_id: 1,
                point: RootIopOpeningPoint::Mle(extension_point.as_slice().to_vec()),
                value: RootIopOpeningValue::Extension(
                    extension_poly.eval_ext::<F>(&extension_point),
                ),
            },
        ];
        let transcript = RootIopBoundTranscript {
            oracles: vec![
                (base_commitment, RootIopOracleValues::Base(base_codeword)),
                (
                    extension_commitment,
                    RootIopOracleValues::Extension(extension_codeword),
                ),
            ],
            claims: claims.clone(),
        };
        let proof = root_system
            .prove(
                &transcript,
                &[base_prover_data, extension_prover_data],
                &mut challenger(),
                0,
            )
            .expect("batched WHIR-bound root proof");

        assert!(proof.openings.is_empty());
        assert!(proof.batched_opening.is_none());
        assert!(proof.direct_batched_opening.is_some());
        let residuals = root_system
            .verify(&commitments, &claims, &proof, &mut challenger(), 0)
            .expect("batched WHIR-bound root proof verification");
        assert!(residuals.is_empty());
    }

    #[test]
    fn shared_message_root_proof_binds_columns() {
        let code = systematic_code();
        let pcs = whir_pcs(code.log_codeword_len());
        let message_pcs = whir_pcs(code.log_msg_len());
        let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
            &pcs,
            &message_pcs,
            &code,
            challenger(),
        );
        let message0 = vec![
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
        ];
        let message1 = vec![
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ];
        let codeword0 = code.encode(&message0);
        let codeword1 = code.encode(&message1);
        let committed = root_system
            .commit_shared_base_message_oracles(vec![
                (0, codeword0.clone(), message0),
                (1, codeword1.clone(), message1),
            ])
            .expect("shared base message root oracle commit");
        let mut commitments = Vec::new();
        let mut prover_data = Vec::new();
        for (commitment, data) in committed {
            commitments.push(commitment);
            prover_data.push(data);
        }
        let claims = vec![
            RootIopOpeningClaim {
                claim_id: 0,
                oracle_id: 0,
                point: RootIopOpeningPoint::<EF>::Index(2),
                value: RootIopOpeningValue::Base(codeword0[2]),
            },
            RootIopOpeningClaim {
                claim_id: 1,
                oracle_id: 1,
                point: RootIopOpeningPoint::<EF>::Index(5),
                value: RootIopOpeningValue::Base(codeword1[5]),
            },
        ];
        let transcript = RootIopBoundTranscript {
            oracles: vec![
                (
                    commitments[0].clone(),
                    RootIopOracleValues::Base(codeword0.clone()),
                ),
                (
                    commitments[1].clone(),
                    RootIopOracleValues::Base(codeword1.clone()),
                ),
            ],
            claims: claims.clone(),
        };
        let proof = root_system
            .prove(&transcript, &prover_data, &mut challenger(), 0)
            .expect("shared WHIR-bound root proof");

        assert!(proof.openings.is_empty());
        assert!(proof.batched_opening.is_none());
        assert!(proof.direct_batched_opening.is_some());
        root_system
            .verify(&commitments, &claims, &proof, &mut challenger(), 0)
            .expect("shared WHIR-bound root proof verification");

        let mut malformed_commitments = commitments;
        if let NativeWarpWhirRootCommitment::BaseMessageShared { column, width, .. } =
            &mut malformed_commitments[1].commitment
        {
            *column = *width;
        } else {
            panic!("expected shared base commitment");
        }
        assert!(
            root_system
                .verify(
                    &malformed_commitments,
                    &claims,
                    &proof,
                    &mut challenger(),
                    0
                )
                .is_err()
        );
    }

    #[test]
    fn message_domain_batched_root_proof_rejects_tampered_virtual_eval() {
        let code = systematic_code();
        let pcs = whir_pcs(code.log_codeword_len());
        let message_pcs = whir_pcs(code.log_msg_len());
        let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
            &pcs,
            &message_pcs,
            &code,
            challenger(),
        );
        let base_message = vec![
            F::from_u64(3),
            F::from_u64(6),
            F::from_u64(10),
            F::from_u64(15),
        ];
        let base_codeword = code.encode(&base_message);
        let extension_message = (0..code.msg_len())
            .map(|i| EF::from_u64((23 * i + 29) as u64))
            .collect::<Vec<_>>();
        let extension_codeword = code.encode_algebra(&extension_message);
        let (base_commitment, base_prover_data) = root_system
            .commit_base_message_oracle(0, base_codeword.clone(), base_message)
            .expect("base message root oracle commit");
        let (extension_commitment, extension_prover_data) = root_system
            .commit_extension_oracle(1, extension_codeword.clone())
            .expect("extension message root oracle commit");
        let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
        let claims = vec![
            RootIopOpeningClaim {
                claim_id: 0,
                oracle_id: 0,
                point: RootIopOpeningPoint::<EF>::Index(2),
                value: RootIopOpeningValue::Base(base_codeword[2]),
            },
            RootIopOpeningClaim {
                claim_id: 1,
                oracle_id: 1,
                point: RootIopOpeningPoint::<EF>::Index(5),
                value: RootIopOpeningValue::Extension(extension_codeword[5]),
            },
        ];
        let transcript = RootIopBoundTranscript {
            oracles: vec![
                (base_commitment, RootIopOracleValues::Base(base_codeword)),
                (
                    extension_commitment,
                    RootIopOracleValues::Extension(extension_codeword),
                ),
            ],
            claims: claims.clone(),
        };
        let mut proof = root_system
            .prove(
                &transcript,
                &[base_prover_data, extension_prover_data],
                &mut challenger(),
                0,
            )
            .expect("batched WHIR-bound root proof");
        proof
            .direct_batched_opening
            .as_mut()
            .expect("batched opening")
            .reduction
            .virtual_eval += EF::ONE;

        assert!(
            root_system
                .verify(&commitments, &claims, &proof, &mut challenger(), 0)
                .is_err()
        );
    }

    #[test]
    fn systematic_message_claim_lifts_to_codeword_subspace() {
        let code = systematic_code();
        let compiler = NativeWarpWhirCompiler::new(&code);
        let witness = vec![
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let codeword = code.encode(&witness);
        let witness_poly = Poly::new(witness);
        let codeword_poly = Poly::new(codeword);
        let message_point = vec![EF::from_u64(17), EF::from_u64(19)];
        let value = witness_poly.eval_base(&Point::new(message_point.clone()));

        let constraint = compiler.systematic_message_eval_constraint(&message_point, value);

        assert!(constraint.verify_base(&codeword_poly));
    }
}
