//! Error types for STIR verification.

use alloc::format;
use alloc::string::String;
use core::fmt::{Display, Formatter, Result as FmtResult};

use thiserror::Error;

use crate::config::StirConfigError;

/// Which round an error refers to.
///
/// In a batch, a proof element carries its instance's own round index.
/// A shared grind carries the global lockstep round.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoundLabel {
    /// An intermediate folding round, indexed from 0.
    Round(usize),
    /// The final round, checked against the sent polynomial rather than queried again.
    Final,
}

impl Display for RoundLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Round(round) => write!(f, "round {round}"),
            Self::Final => f.write_str("final round"),
        }
    }
}

/// Which of a round's two proof-of-work grinds a witness belongs to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GrindStage {
    /// The grind before the folding challenge `gamma`.
    Folding,
    /// The grind before the combination challenge `r_comb` and the query indices.
    Query,
}

impl Display for GrindStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(match self {
            Self::Folding => "folding",
            Self::Query => "query",
        })
    }
}

/// Why a proof's shape does not match what the configuration and the public input pin.
///
/// `expected` is the value the verifier derived.
/// `got` is what the proof carried.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum ProofShapeError {
    /// A batch takes one proof per configured instance.
    #[error("expected {expected} proofs, got {got}")]
    InstanceCount { expected: usize, got: usize },

    /// A proof carries one round proof per configured round.
    #[error(
        "{}expected {expected} round proofs, got {got}",
        instance.map_or_else(String::new, |i| format!("instance {i}: "))
    )]
    RoundCount {
        /// Batch index of the offending proof, `None` outside a batch.
        instance: Option<usize>,
        expected: usize,
        got: usize,
    },

    /// STIR commits the initial oracle itself, so the proof must carry that commitment.
    #[error("missing initial oracle commitment")]
    MissingInitialCommitment,

    /// The initial oracle is bound by the caller, so the proof must carry no commitment.
    #[error("unexpected initial oracle commitment for an externally bound oracle")]
    UnexpectedInitialCommitment,

    /// The final degree bound pins the final polynomial's coefficient count exactly.
    #[error("final polynomial has {got} coefficients, expected {expected}")]
    FinalPolynomialLength { expected: usize, got: usize },

    /// The soundness assumption fixes each round's out-of-domain sample count.
    #[error("{round}: {got} OOD answers, expected {expected}")]
    OodAnswerCount {
        round: RoundLabel,
        expected: usize,
        got: usize,
    },

    /// `Ans` interpolates the round's OOD and query points, bounding its degree.
    ///
    /// A shorter polynomial is legitimate: the prover may strip trailing zero coefficients.
    #[error("{round}: ans polynomial has {got} coefficients, expected at most {maximum}")]
    AnsPolynomialTooLong {
        round: RoundLabel,
        maximum: usize,
        got: usize,
    },

    /// A committed oracle is read through a Merkle multi-opening the proof must supply.
    #[error("{round}: missing query openings")]
    MissingQueryOpenings { round: RoundLabel },

    /// An externally bound oracle is answered by the caller, so the proof must open nothing.
    #[error("{round}: unexpected query openings for an externally bound oracle")]
    UnexpectedQueryOpenings { round: RoundLabel },

    /// A committed oracle is authenticated against a commitment the proof must supply.
    #[error("{round}: missing oracle commitment")]
    MissingCommitment { round: RoundLabel },

    /// One opened row per query the round draws, a count the configuration fixes.
    #[error("{round}: {got} opened rows, expected {expected}")]
    QueryOpeningCount {
        round: RoundLabel,
        expected: usize,
        got: usize,
    },

    /// An opened row is one fiber of the round's fold, so it holds `arity` evaluations.
    #[error("{round}, query {query}: opened row has {got} evaluations, expected {expected}")]
    OpenedRowArity {
        round: RoundLabel,
        query: usize,
        expected: usize,
        got: usize,
    },

    /// Instances verified in lockstep share one grind, so each must replicate its witness.
    #[error("{round}: replicated {stage} PoW witnesses disagree across batched instances")]
    ReplicatedWitnessMismatch {
        round: RoundLabel,
        stage: GrindStage,
    },

    /// The opening proof carries one STIR instance per distinct shared LDE height.
    #[error("expected {expected} LDE-height buckets, got {got}")]
    BucketCount { expected: usize, got: usize },

    /// A bucket carries one input-opening slot per public commitment, occupied or not.
    ///
    /// A short list would silently drop trailing commitments from the reduced opening.
    #[error("bucket 2^{log_height}: {got} input opening slots, expected {expected}")]
    InputOpeningCount {
        log_height: usize,
        expected: usize,
        got: usize,
    },

    /// A commitment whose matrices live at this bucket's height must be opened there.
    #[error("bucket 2^{log_height}, commitment {commitment}: missing input opening")]
    MissingInputOpening {
        log_height: usize,
        commitment: usize,
    },

    /// A commitment whose matrices live at another height must not be opened at this bucket.
    #[error("bucket 2^{log_height}, commitment {commitment}: unexpected input opening")]
    UnexpectedInputOpening {
        log_height: usize,
        commitment: usize,
    },

    /// A commitment carries one Merkle root per group of the layout its claims imply.
    #[error("commitment {commitment}: expected {expected} Merkle roots, got {got}")]
    CommitmentRootCount {
        commitment: usize,
        expected: usize,
        got: usize,
    },

    /// A bucket that ran no `Combine` holds a single native-height class.
    #[error("bucket 2^{log_height}: {got} native-height classes, expected {expected}")]
    HeightClassCount {
        log_height: usize,
        expected: usize,
        got: usize,
    },

    /// Every class merged by `Combine` needs its degree-correction coefficient.
    #[error(
        "bucket 2^{log_height}: no Combine coefficient for native height 2^{log_native_height}"
    )]
    MissingCombineCoefficient {
        log_height: usize,
        log_native_height: usize,
    },

    /// The opened-row count is the bucket's deduplicated first-round query count.
    #[error(
        "bucket 2^{log_height}, commitment {commitment}: {got} opened rows, expected {expected}"
    )]
    InputOpenedRowCount {
        log_height: usize,
        commitment: usize,
        expected: usize,
        got: usize,
    },
}

/// Why the caller's external initial-oracle source did not answer as asked.
///
/// The source is a closure, so these report a caller bug rather than a malformed proof.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum ExternalSourceError {
    /// A batch takes one fiber source per instance.
    #[error("expected {expected} fiber sources, got {got}")]
    SourceCount { expected: usize, got: usize },

    /// The source must answer one fiber per requested index.
    #[error("{round}: source returned {got} fibers, expected {expected}")]
    FiberCount {
        round: RoundLabel,
        expected: usize,
        got: usize,
    },

    /// Each fiber holds the round's `arity` evaluations.
    #[error("{round}, fiber {fiber}: source returned {got} evaluations, expected {expected}")]
    FiberArity {
        round: RoundLabel,
        /// Position in the sorted, deduplicated index list handed to the source.
        fiber: usize,
        expected: usize,
        got: usize,
    },
}

/// Errors returned by [`crate::verifier::verify_stir`].
#[derive(Debug, Error, PartialEq)]
pub enum StirError<MmcsError, InputError = ()> {
    /// A proof-of-work witness failed verification.
    #[error("{round}: invalid proof-of-work witness")]
    InvalidPowWitness { round: RoundLabel },

    /// A Merkle multi-opening proof failed for a round's queries.
    #[error("{round}: invalid MMCS opening proof")]
    InvalidMmcsProof {
        round: RoundLabel,
        #[source]
        source: MmcsError,
    },

    /// `Ans` did not interpolate the round's claimed values at the random evaluation point.
    #[error("{round}: ans polynomial consistency check failed")]
    InvalidAnsConsistency { round: RoundLabel },

    /// A virtual-oracle evaluation landed in the prior round's challenge set.
    #[error("{round}, query {query}: invalid virtual-oracle query")]
    InvalidRoundConsistency { round: RoundLabel, query: usize },

    /// The final polynomial does not evaluate consistently with the last committed codeword.
    #[error("final polynomial evaluation mismatch")]
    FinalPolyMismatch,

    /// The proof's shape disagrees with the configuration or the public input.
    #[error(transparent)]
    InvalidProofShape(#[from] ProofShapeError),

    /// The caller's external initial-oracle source misbehaved.
    #[error("external initial oracle: {0}")]
    ExternalSource(#[from] ExternalSourceError),

    /// A matrix in `commitment`'s batch was opened at zero points, so its width cannot be
    /// pinned from the claimed evaluations. Every input matrix must be opened at >= 1 point.
    #[error("commitment {commitment}, matrix {matrix}: opened at zero points")]
    MatrixWithoutOpeningPoints { commitment: usize, matrix: usize },

    /// A claimed opening point coincides with a queried fiber lane.
    ///
    /// The quotient `(f(z) - f(x)) / (z - x)` is undefined there.
    #[error(
        "commitment {commitment}, matrix {matrix}, point {point}: opening point coincides with a query point"
    )]
    OpeningPointMatchesQueryPoint {
        commitment: usize,
        matrix: usize,
        point: usize,
    },

    /// The requested STIR parameters cannot reach `security_level` at some LDE-height bucket.
    #[error("STIR config error: {0}")]
    Config(#[source] StirConfigError),

    /// An error propagated from the input polynomial commitment scheme.
    #[error("input error")]
    InputError(InputError),
}

impl<E, IE> StirError<E, IE> {
    /// Map the `InputError` variant to a different type.
    pub fn map_input_err<IE2>(self, f: impl FnOnce(IE) -> IE2) -> StirError<E, IE2> {
        match self {
            Self::InvalidPowWitness { round } => StirError::InvalidPowWitness { round },
            Self::InvalidMmcsProof { round, source } => {
                StirError::InvalidMmcsProof { round, source }
            }
            Self::InvalidAnsConsistency { round } => StirError::InvalidAnsConsistency { round },
            Self::InvalidRoundConsistency { round, query } => {
                StirError::InvalidRoundConsistency { round, query }
            }
            Self::FinalPolyMismatch => StirError::FinalPolyMismatch,
            Self::InvalidProofShape(e) => StirError::InvalidProofShape(e),
            Self::ExternalSource(e) => StirError::ExternalSource(e),
            Self::MatrixWithoutOpeningPoints { commitment, matrix } => {
                StirError::MatrixWithoutOpeningPoints { commitment, matrix }
            }
            Self::OpeningPointMatchesQueryPoint {
                commitment,
                matrix,
                point,
            } => StirError::OpeningPointMatchesQueryPoint {
                commitment,
                matrix,
                point,
            },
            Self::Config(e) => StirError::Config(e),
            Self::InputError(e) => StirError::InputError(f(e)),
        }
    }
}
