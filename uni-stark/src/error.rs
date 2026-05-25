//! Error types for STARK verification.

use alloc::format;
use alloc::string::String;

use thiserror::Error;

/// Specific reasons why a proof's shape is invalid.
#[derive(Debug, Error)]
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
    #[error("air {air}: trace next width mismatch or missing")]
    TraceNextMismatch { air: usize },
    /// Trace next values present when AIR doesn't use next row.
    #[error("air {air}: unexpected trace next values")]
    UnexpectedTraceNext { air: usize },
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
    /// Preprocessed verifier key is inconsistent with width.
    #[error("preprocessed verifier key inconsistency")]
    PreprocessedVerifierKeyInconsistency,
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
    #[error("air {air}: unexpected preprocessed values")]
    UnexpectedPreprocessedValues { air: usize },
    /// Proof degree bits are too small for the PCS ZK setting.
    #[error(
        "{}degree_bits too small for zk setting: expected at least {minimum}, got {got}",
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
    /// Lookup commitment presence doesn't match lookup configuration.
    #[error("lookup commitment presence does not match lookup configuration")]
    LookupCommitmentMismatch,
    /// Global lookup data count doesn't match the number of global lookups for an AIR.
    #[error("air {air}: global lookup data count mismatch: expected {expected}, got {got}")]
    GlobalLookupDataCountMismatch {
        air: usize,
        expected: usize,
        got: usize,
    },
    /// Global lookup proof metadata doesn't match the AIR's declared interactions.
    #[error(
        "air {air}: global lookup data metadata mismatch at index {lookup}: expected name={expected_name}, aux_column={expected_aux_column}; got name={got_name}, aux_column={got_aux_column}"
    )]
    GlobalLookupDataMetadataMismatch {
        air: usize,
        lookup: usize,
        expected_name: String,
        got_name: String,
        expected_aux_column: usize,
        got_aux_column: usize,
    },
    /// Permutation local and next have different lengths.
    #[error("air {air}: permutation local/next length mismatch")]
    PermutationLengthMismatch { air: usize },
    /// Permutation width doesn't match expected.
    #[error("air {air}: permutation width mismatch: expected {expected}")]
    PermutationWidthMismatch { air: usize, expected: usize },
    /// Opened values (trace, quotient, random) don't match expected dimensions.
    #[error("opened values do not match expected dimensions")]
    OpenedValuesDimensionMismatch,
}

/// Top-level verification error.
#[derive(Debug, Error)]
pub enum VerificationError<PcsErr>
where
    PcsErr: core::fmt::Debug,
{
    /// The proof shape is invalid.
    #[error(transparent)]
    InvalidProofShape(#[from] InvalidProofShapeError),
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
    /// Lookup related error.
    #[error("lookup error: {0}")]
    LookupError(String),
}
