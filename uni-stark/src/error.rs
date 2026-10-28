//! Error types for STARK proving and verification.

use alloc::format;
use alloc::string::String;

use p3_air::symbolic::NextRowTrace;
pub use p3_commit::PeriodicColumnShapeError;
use thiserror::Error;

use crate::StarkTranscriptFailure;

/// A recoverable failure during proof generation.
///
/// The prover returns it instead of emitting a partial proof.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProvingError<E> {
    /// The backend error, annotated with the failing proving phase.
    #[error("PCS {phase} failed: {source:?}")]
    Pcs { phase: &'static str, source: E },
    /// A batch prover was given no instances.
    ///
    /// A batch commitment needs at least one matrix, so there is nothing to prove.
    #[error("cannot prove an empty batch: no instances were given")]
    EmptyBatch,
    /// A constraint reads a next-row column that the AIR does not declare.
    ///
    /// Provers and verifiers open the next row only when the declared set is nonempty.
    /// A read outside that set breaks the contract they rely on.
    #[error(
        "{}a constraint reads column {column} of the next {trace} row, which the AIR does not declare",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    UndeclaredNextRowRead {
        /// The instance index, for a batch of AIRs.
        air: Option<usize>,
        /// The trace the read comes from.
        trace: NextRowTrace,
        /// The column index within that trace.
        column: usize,
    },
}

/// Specific reasons why a proof's shape is invalid.
///
/// New reasons appear whenever the verifier learns to reject another malformed shape.
/// Matching on this list from another crate therefore requires a catch-all arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum InvalidProofShapeError {
    /// Instance arrays (airs, opened_values, public_values, degree_bits) have different lengths.
    #[error("instance count mismatch")]
    InstanceCountMismatch,
    /// Trace local width doesn't match the AIR width.
    #[error("air {air}: trace local width mismatch: expected {expected}, got {got}")]
    TraceLocalWidthMismatch {
        air: usize,
        expected: usize,
        got: usize,
    },
    /// Trace next values have wrong width or are unexpectedly missing.
    #[error(
        "{}trace next width mismatch or missing",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    TraceNextMismatch { air: Option<usize> },
    /// Trace next values present when AIR doesn't use next row.
    #[error(
        "{}unexpected trace next values",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    UnexpectedTraceNext { air: Option<usize> },
    /// Preprocessed next values present when the AIR doesn't read the next preprocessed row.
    #[error(
        "{}unexpected preprocessed next values",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    UnexpectedPreprocessedNext { air: Option<usize> },
    /// Quotient chunks count doesn't match expected.
    #[error("air {air}: quotient chunks count mismatch: expected {expected}, got {got}")]
    QuotientChunksCountMismatch {
        air: usize,
        expected: usize,
        got: usize,
    },
    /// Quotient chunk has wrong dimension.
    #[error("air {air}: quotient chunk dimension mismatch")]
    QuotientChunkDimensionMismatch { air: usize },
    /// Quotient opened values count doesn't match domain count.
    #[error("air {air}: quotient domains count mismatch")]
    QuotientDomainsCountMismatch { air: usize },
    /// Preprocessed trace opened values width doesn't match expected.
    #[error(
        "preprocessed trace width mismatch: expected local={expected_local}, next={expected_next}, got local={got_local}, next={got_next}"
    )]
    PreprocessedTraceWidthMismatch {
        expected_local: usize,
        expected_next: usize,
        got_local: usize,
        got_next: usize,
    },
    /// The preprocessed verifier key does not match the preprocessed width the AIR declares.
    ///
    /// This covers a missing key, an unexpected key and a key of another width.
    #[error(
        "preprocessed verifier key inconsistency: air width {air_width}, key width {key_width:?}"
    )]
    PreprocessedVerifierKeyInconsistency {
        /// Preprocessed column count the AIR declares.
        air_width: usize,
        /// Width of the supplied verifier key, when the check knows it.
        key_width: Option<usize>,
    },
    /// Preprocessed and main trace have different heights.
    #[error(
        "preprocessed degree mismatch: vk degree_bits={vk_degree_bits}, proof degree_bits={proof_degree_bits}"
    )]
    PreprocessedDegreeMismatch {
        vk_degree_bits: usize,
        proof_degree_bits: usize,
    },
    /// Preprocessed width mismatch for a specific AIR.
    #[error("air {air}: preprocessed width mismatch")]
    PreprocessedWidthMismatch { air: usize },
    /// Preprocessed values present when preprocessed width is zero.
    #[error(
        "{}unexpected preprocessed values",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    UnexpectedPreprocessedValues { air: Option<usize> },
    /// Proof degree bits fall below the smallest trace height the commitment scheme accepts.
    #[error(
        "{}degree_bits too small for the pcs: expected at least {minimum}, got {got}",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    DegreeBitsTooSmall {
        air: Option<usize>,
        minimum: usize,
        got: usize,
    },
    /// Proof degree bits are too large to safely construct verifier domains.
    #[error(
        "{}degree_bits too large for domain construction: expected at most {maximum}, got {got}",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    DegreeBitsTooLarge {
        air: Option<usize>,
        maximum: usize,
        got: usize,
    },
    /// The quotient domain log-size overflows after adding degree bits and quotient chunk bits.
    #[error(
        "{}quotient domain too large: log-size {got} exceeds maximum {maximum}",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    QuotientDomainTooLarge {
        air: Option<usize>,
        maximum: usize,
        got: usize,
    },
    /// Missing preprocessed local or next values.
    #[error("air {air}: missing preprocessed values")]
    MissingPreprocessedValues { air: usize },
    /// Preprocessed metadata missing or mismatched.
    #[error("air {air}: preprocessed metadata mismatch")]
    PreprocessedMetadataMismatch { air: usize },
    /// Public values length doesn't match what the AIR expects.
    #[error("public values length mismatch: expected {expected}, got {got}")]
    PublicValuesLengthMismatch { expected: usize, got: usize },
    /// Opened values (trace, quotient, random) don't match expected dimensions.
    #[error("opened values do not match expected dimensions")]
    OpenedValuesDimensionMismatch,
    /// The out-of-domain grinding witness is not the value a zero difficulty admits.
    ///
    /// Checked with the other proof-shape rejections, before any transcript work.
    //
    // Why: at `ood_pow_bits = 0` neither side touches the sponge.
    //
    //     prover  : grind(0)            -> returns zero, absorbs nothing
    //     verifier: the Pow step is elided, so no witness is read at all
    //
    // `ood_pow_witness` is then bound to nothing: any value rides along and still verifies.
    // Zero is the only value an honest prover emits, so zero is the only value accepted.
    #[error("out-of-domain grinding witness is nonzero at zero difficulty, expected zero")]
    NonCanonicalOodPowWitness,
}

/// Top-level verification error.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerificationError<PcsErr>
where
    PcsErr: core::fmt::Debug,
{
    /// The proof shape is invalid.
    #[error(transparent)]
    InvalidProofShape(#[from] InvalidProofShapeError),
    /// A periodic column declared by the AIR cannot be evaluated.
    #[error(transparent)]
    PeriodicColumn(#[from] PeriodicColumnShapeError),
    /// An error occurred while verifying the claimed openings.
    #[error("invalid opening argument: {0:?}")]
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    #[error("out-of-domain evaluation mismatch{}", .index.map(|i| format!(" at index {}", i)).unwrap_or_default())]
    OodEvaluationMismatch { index: Option<usize> },
    /// The FRI batch randomization does not correspond to the ZK setting.
    #[error("randomization error: FRI batch randomization does not match ZK setting")]
    RandomizationError,
    /// The domain does not support computing the next point algebraically.
    #[error(
        "next point unavailable: domain does not support computing the next point algebraically"
    )]
    NextPointUnavailable,
    /// The out-of-domain point coincides with a trace-domain point.
    ///
    /// Selector inversion is undefined there.
    #[error("out-of-domain point lies inside the trace domain")]
    OodPointInDomain,
    /// The proof of work guarding the out-of-domain point is invalid.
    ///
    /// Either the witness was forged, or the prover and verifier disagree on
    /// the configured number of out-of-domain grinding bits.
    #[error("invalid proof-of-work witness for the out-of-domain point")]
    InvalidOodPowWitness,
    /// The out-of-domain grinding difficulty cannot be sampled from the base field.
    ///
    /// A grind of b bits needs b below the word size and 2^b below the field order p.
    #[error("{bits}-bit out-of-domain grinding difficulty does not fit the field order")]
    UnsupportedOodPowBits {
        /// The configured difficulty.
        bits: usize,
    },
    /// A constraint reads a next-row column that the AIR does not declare.
    ///
    /// Provers and verifiers open the next row only when the declared set is nonempty.
    /// A read outside that set breaks the contract they rely on.
    #[error(
        "{}a constraint reads column {column} of the next {trace} row, which the AIR does not declare",
        air.map_or_else(String::new, |air| format!("air {air}: "))
    )]
    UndeclaredNextRowRead {
        /// The instance index, for a batch of AIRs.
        air: Option<usize>,
        /// The trace the read comes from.
        trace: NextRowTrace,
        /// The column index within that trace.
        column: usize,
    },
}

impl<PcsErr> From<StarkTranscriptFailure> for VerificationError<PcsErr>
where
    PcsErr: core::fmt::Debug,
{
    fn from(failure: StarkTranscriptFailure) -> Self {
        match failure {
            // A preprocessed commitment that disagrees with the width in force.
            //
            // The transcript knows the described width, but not the key's.
            StarkTranscriptFailure::MissingPreprocessedCommitment { width } => {
                InvalidProofShapeError::PreprocessedVerifierKeyInconsistency {
                    air_width: width,
                    key_width: None,
                }
                .into()
            }
            StarkTranscriptFailure::UnexpectedPreprocessedCommitment => {
                InvalidProofShapeError::PreprocessedVerifierKeyInconsistency {
                    air_width: 0,
                    key_width: None,
                }
                .into()
            }
            // A randomization commitment that disagrees with the PCS's zero-knowledge setting.
            StarkTranscriptFailure::MissingRandomCommitment
            | StarkTranscriptFailure::UnexpectedRandomCommitment => Self::RandomizationError,
            // The grind guarding the out-of-domain point.
            StarkTranscriptFailure::OodPowWitness { .. } => Self::InvalidOodPowWitness,
        }
    }
}
