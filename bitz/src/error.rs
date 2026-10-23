//! Why parameters, a prover or a verifier refuse.

use p3_binary_pcs::whir::{CommitmentError, Region};
use p3_gkr::ProductGkrError;
use p3_sumcheck::SumcheckError;
use thiserror::Error;

/// Parameters that admit no sound reduction.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ParamsError {
    /// A row must hold whole words.
    #[error("rows of 2^{log_rows} bits, minimum is 2^{minimum}")]
    RowsTooShort {
        /// The rejected row variables.
        log_rows: usize,
        /// The fewest row variables.
        minimum: usize,
    },
    /// Carry-free fold accumulation needs fewer than `2^32` rows.
    #[error("rows of 2^{log_rows} bits, maximum is 2^{maximum}")]
    RowsTooLong {
        /// The rejected row variables.
        log_rows: usize,
        /// The most row variables.
        maximum: usize,
    },
    /// The committed bits cannot be addressed by one machine word.
    #[error("2^{num_variables} committed bits, maximum is 2^{maximum}")]
    TooManyVariables {
        /// The rejected variables.
        num_variables: usize,
        /// The most variables.
        maximum: usize,
    },
    /// A fold can reach the order of the generator, so `(p - 1) 2^r1 < 2^192 - 1` fails.
    #[error("a {modulus_bits}-bit field over 2^{log_rows} rows overflows the exponent group")]
    FoldOverflow {
        /// Bits of the field order.
        modulus_bits: u32,
        /// Row variables.
        log_rows: usize,
    },
    /// The commitment admits no schedule.
    #[error(transparent)]
    Commitment(#[from] CommitmentError),
    /// The composed soundness error misses the target.
    #[error("the composed error reaches {achieved} bits, target is {target}")]
    Security {
        /// Bits asked for.
        target: usize,
        /// Bits the union bound reaches.
        achieved: f64,
    },
}

/// A claim that does not fit its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ClaimError {
    /// A reduction needs at least one claim.
    #[error("no claim to reduce")]
    NoClaims,
    /// The row weights do not cover one row.
    #[error("claim {index} weighs {actual} rows, expected {expected}")]
    RowLength {
        /// The claim.
        index: usize,
        /// Rows of the shape.
        expected: usize,
        /// Rows the claim weighs.
        actual: usize,
    },
    /// The column weights do not cover every column.
    #[error("claim {index} weighs {actual} columns, expected {expected}")]
    ColumnLength {
        /// The claim.
        index: usize,
        /// Columns, or column variables for an equality weight.
        expected: usize,
        /// What the claim supplies.
        actual: usize,
    },
    /// The region does not hold the shape.
    #[error(transparent)]
    Region(#[from] RegionError),
}

/// A region that does not hold the shape's bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("{region:?} is not an aligned run of 2^{log_bits} bits inside 2^{num_variables}")]
pub struct RegionError {
    /// The region asked for.
    pub region: Region,
    /// Variables of the shape.
    pub log_bits: usize,
    /// Variables of the commitment.
    pub num_variables: usize,
}

/// Why the prover refuses.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProvingError {
    /// A claim does not fit the shape.
    #[error(transparent)]
    Claim(#[from] ClaimError),
    /// The committed bits do not satisfy a claim.
    #[error("the committed bits do not satisfy claim {index}")]
    FalseClaim {
        /// The claim.
        index: usize,
    },
    /// Committing or opening failed.
    #[error(transparent)]
    Commitment(#[from] CommitmentError),
}

/// Why the verifier rejects.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VerificationError {
    /// A claim does not fit the shape.
    #[error(transparent)]
    Claim(#[from] ClaimError),
    /// The folds have another length than the shape fixes.
    #[error("{actual} fold words, expected {expected}")]
    FoldLength {
        /// Words the shape fixes.
        expected: usize,
        /// Words in the proof.
        actual: usize,
    },
    /// Bits past the last fold are set.
    #[error("the fold padding is not zero")]
    FoldPadding,
    /// A fold exceeds `(p - 1) 2^r1` (Construction 4.4, Round 2).
    #[error("fold {index} is out of range")]
    FoldRange {
        /// Position of the fold, claim major.
        index: usize,
    },
    /// The folds do not sum to a claim's value (Construction 4.4, Round 2, eq. 38).
    #[error("the folds do not sum to claim {index}")]
    ClaimSum {
        /// The claim.
        index: usize,
    },
    /// The grand product rejects.
    #[error(transparent)]
    Gkr(#[from] ProductGkrError),
    /// A product root is not the generator raised to its fold.
    #[error("root {index} is not the power of its fold")]
    RootMismatch {
        /// Position of the root, claim major.
        index: usize,
    },
    /// The inner-product sumcheck rejects.
    #[error(transparent)]
    Inner(#[from] SumcheckError),
    /// The last inner-product round disagrees with the claimed bit evaluation.
    #[error("the inner product does not close at the claimed evaluation")]
    InnerEval,
    /// The opening rejects.
    #[error(transparent)]
    Commitment(#[from] CommitmentError),
}
