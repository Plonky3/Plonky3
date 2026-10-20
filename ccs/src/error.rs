//! Why a system, a witness or a proof is rejected.

use p3_sumcheck::generic_degree::GenericDegreeError;
use thiserror::Error;

/// Why matrices and constraints do not form a constraint system.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum StructureError {
    /// No matrix.
    #[error("a constraint system needs at least one matrix")]
    NoOperators,
    /// No constraint.
    #[error("a constraint system needs at least one constraint")]
    NoConstraints,
    /// Matrices differ in shape.
    #[error(
        "matrix {operator} has shape 2^{log_rows} x 2^{log_cols}, expected 2^{expected_log_rows} x 2^{expected_log_cols}"
    )]
    ShapeMismatch {
        /// Position of the offending matrix.
        operator: usize,
        /// Its number of row variables.
        log_rows: usize,
        /// Its number of column variables.
        log_cols: usize,
        /// Number of row variables of the first matrix.
        expected_log_rows: usize,
        /// Number of column variables of the first matrix.
        expected_log_cols: usize,
    },
    /// A constraint has no terms.
    #[error("constraint {constraint} has no terms")]
    EmptyConstraint {
        /// Position of the offending constraint.
        constraint: usize,
    },
    /// A term has no factors.
    #[error("constraint {constraint}, term {term} has no factors")]
    EmptyTerm {
        /// Position of the constraint.
        constraint: usize,
        /// Position of the term within it.
        term: usize,
    },
    /// A factor names no matrix.
    #[error(
        "constraint {constraint}, term {term} names matrix {factor}, but there are only {operators}"
    )]
    FactorOutOfRange {
        /// Position of the constraint.
        constraint: usize,
        /// Position of the term within it.
        term: usize,
        /// Factor index.
        factor: usize,
        /// Number of matrices in the system.
        operators: usize,
    },
}

/// Why a witness does not satisfy a constraint system.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum WitnessError {
    /// Wrong witness length.
    #[error("witness has {actual} entries, expected {expected}")]
    LengthMismatch {
        /// One entry per matrix column.
        expected: usize,
        /// Entries supplied.
        actual: usize,
    },
    /// A constraint does not vanish on a selected row.
    #[error("constraint {constraint} does not vanish on row {row}")]
    Unsatisfied {
        /// First failing row.
        row: usize,
        /// First failing constraint on that row.
        constraint: usize,
    },
}

/// Why the verifier rejects a reduction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum VerificationError {
    /// The zerocheck claims a nonzero sum.
    #[error("the zerocheck claims a nonzero sum")]
    NonzeroZerocheckClaim,
    /// The zerocheck sumcheck failed.
    #[error("zerocheck sumcheck: {0}")]
    Zerocheck(GenericDegreeError),
    /// Wrong number of matrix outputs.
    #[error("the proof carries {actual} matrix outputs, expected {expected}")]
    OperandCount {
        /// One per matrix.
        expected: usize,
        /// Carried by the proof.
        actual: usize,
    },
    /// The matrix outputs do not match the zerocheck final value.
    #[error("the matrix outputs do not satisfy the constraint terms at the zerocheck point")]
    ZerocheckMismatch,
    /// The linear sumcheck claim is not the batched matrix outputs.
    #[error("the linear sumcheck does not start from the batched matrix outputs")]
    LinearClaimMismatch,
    /// The linear sumcheck failed.
    #[error("linear sumcheck: {0}")]
    Linear(GenericDegreeError),
    /// The witness value does not match the linear sumcheck final value.
    #[error("the witness value does not close the linear sumcheck")]
    LinearMismatch,
}

/// Why the verifier rejects a proof over a committed witness.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CommittedVerificationError<E> {
    /// The reduction itself is rejected.
    #[error("reduction: {0}")]
    Reduction(#[from] VerificationError),
    /// A public region lies outside the witness.
    #[error("public region {region} lies outside a witness of 2^{log_cols} cells")]
    PublicOutOfRange {
        /// Position of the region.
        region: usize,
        /// Witness variables.
        log_cols: usize,
    },
    /// The commitment opening is rejected.
    #[error("opening: {0:?}")]
    Opening(E),
    /// The opened witness value differs from the one the reduction claims.
    #[error("the opened witness value differs from the claimed one")]
    WitnessMismatch,
    /// The committed cells of a public region differ from its values.
    #[error("the committed cells of public region {region} differ from its values")]
    PublicMismatch {
        /// Position of the region.
        region: usize,
    },
}
