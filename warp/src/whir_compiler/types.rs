use super::*;

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
    pub(super) fn observe_payload_into<F, Challenger>(&self, challenger: &mut Challenger)
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
