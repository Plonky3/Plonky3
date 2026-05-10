use super::*;

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
}

/// WHIR commitment used by the native WARP root proof.
///
/// All variants are message-domain commitments for the same WARP/WHIR
/// Reed-Solomon code. WARP may record codeword openings, but those openings are
/// compiled into linear claims over the committed message before WHIR proves
/// them. There is intentionally no codeword-domain or limb commitment variant:
/// keeping only these variants prevents the old double-RS and extension-limb
/// fallback paths from re-entering the root proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "Comm: Serialize", deserialize = "Comm: Deserialize<'de>"))]
pub enum NativeWarpWhirRootCommitment<Comm> {
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
    /// RS message committed by WHIR.
    pub message: Vec<F>,
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
    /// Systematic accumulator message committed by WHIR.
    pub message: Vec<EF>,
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
    /// Extension-field accumulator message committed through WHIR's extension
    /// initial-oracle path.
    ExtensionMessage(NativeWarpWhirRootExtensionProverData<F, EF, MT, Challenger, DIGEST_ELEMS>),
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

/// One WHIR proof authenticating the batched root-IOP opening.
///
/// `reduction` combines all WARP root claims over all message-domain oracles to
/// one virtual same-point opening. The `opening` proof is WHIR's grouped
/// batched-initial proof against the original per-oracle roots.
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

/// Complete native WARP root proof backed by one WHIR batched opening.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned",
    deserialize = "F: Serialize + serde::de::DeserializeOwned + Send + Sync + Clone, EF: Serialize + serde::de::DeserializeOwned, MT::Commitment: Serialize + serde::de::DeserializeOwned, MT::Proof: Serialize + serde::de::DeserializeOwned"
))]
pub struct NativeWarpWhirRootProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Direct batched root proof for message-domain WARP roots.
    pub opening: NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>,
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

    /// Prover data or proof kind did not match the oracle field.
    #[error("root oracle {0} has mismatched WHIR proof data kind")]
    OracleKindMismatch(usize),

    /// Batched residual WHIR verification failed.
    #[error("batched residual WHIR verifier failed: {0:?}")]
    BatchedOpening(WhirVerifierError),
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
