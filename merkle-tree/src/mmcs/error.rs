//! Error types raised while committing, opening, or verifying a Merkle MMCS.

use thiserror::Error;

/// Errors that may arise during Merkle tree commitment, opening, or verification.
#[derive(Debug, Error)]
pub enum MerkleTreeError {
    /// The number of openings provided does not match the expected number.
    #[error("wrong batch size: number of openings does not match expected")]
    WrongBatchSize,

    /// An opened row's length does not match the width of its matrix.
    #[error("wrong width: matrix {matrix} expected {expected} values, got {got}")]
    WrongWidth {
        /// Index of the offending matrix in the batch.
        matrix: usize,
        /// Width of the matrix according to the caller-supplied dimensions.
        expected: usize,
        /// Length of the opened row provided in the proof.
        got: usize,
    },

    /// The number of proof nodes does not match the expected tree height.
    #[error("wrong height: expected {expected_proof_len} siblings, got {num_siblings}")]
    WrongHeight {
        /// Expected number of sibling hashes in the proof.
        expected_proof_len: usize,

        /// Actual number of sibling hashes provided in the proof.
        num_siblings: usize,
    },

    /// Matrix heights are incompatible; they cannot share a common binary Merkle tree.
    #[error(
        "matrix height {height} incompatible with tallest height {max_height}; \
         expected ceil_div({max_height}, 2^{bits_reduced}) = {expected_height} \
         so every global index maps to a row at depth {bits_reduced}"
    )]
    IncompatibleHeights {
        /// Height that was provided.
        height: usize,
        /// Height of the tallest matrix in the batch.
        max_height: usize,
        /// The only height that covers every global index at this depth.
        expected_height: usize,
        /// Power-of-two reduction depth from the tallest matrix.
        bits_reduced: usize,
    },

    /// The queried row index exceeds the maximum height.
    #[error("index out of bounds: index {index} exceeds max height {max_height}")]
    IndexOutOfBounds {
        /// Maximum admissible height.
        max_height: usize,
        /// Row index that was provided.
        index: usize,
    },

    /// Attempted to open an empty batch (no committed matrices).
    #[error("empty batch: attempted to open an empty batch with no committed matrices")]
    EmptyBatch,

    /// The computed Merkle digest does not match any entry in the cap.
    #[error("cap mismatch: computed digest does not match any entry in the Merkle cap")]
    CapMismatch,

    /// A pruned batch opening could not be restored or validated.
    #[error("malformed pruned proof: {0}")]
    MalformedPrunedProof(#[from] PrunedProofError),
}

/// Why a pruned batch opening was rejected.
///
/// Each variant pins one failure mode with its diagnostic fields.
#[derive(Debug, Error)]
pub enum PrunedProofError {
    /// More unique paths than the tree can hold.
    #[error("too many unique paths: {got} exceeds tree height {max_height}")]
    TooManyUniquePaths {
        /// Number of unique paths claimed by the proof.
        got: usize,
        /// Maximum admissible height (the tallest committed matrix).
        max_height: usize,
    },

    /// A restored path has the wrong number of siblings for the tree geometry.
    #[error("restored path has {got} siblings, expected {expected}")]
    SiblingCountMismatch {
        /// Sibling count implied by the verifier-known arity schedule.
        expected: usize,
        /// Sibling count produced while restoring the path.
        got: usize,
    },

    /// An `original_order` entry references a unique-path slot that does not exist.
    #[error("original-order references missing path slot {slot} of {num_paths}")]
    OriginalOrderOutOfRange {
        /// Slot index referenced by the entry.
        slot: usize,
        /// Number of unique paths actually present.
        num_paths: usize,
    },

    /// Two queries map to one path but disagree on opened values.
    #[error("duplicate queries for path slot {slot} disagree on opened values")]
    InconsistentDuplicateOpenings {
        /// Unique-path slot the conflicting queries map to.
        slot: usize,
    },

    /// A unique path is never referenced by any query.
    #[error("unique path slot {slot} is never referenced by a query")]
    UnreferencedPath {
        /// Slot that no `original_order` entry points to.
        slot: usize,
    },

    /// Sorted leaf indices are not strictly ascending.
    #[error("leaf indices not strictly ascending at position {position}: index {index}")]
    NonAscendingLeaves {
        /// Position of the offending path in the sorted list.
        position: usize,
        /// Leaf index that fails to exceed its predecessor.
        index: usize,
    },

    /// The path and query counts are inconsistent (e.g. queries present but no paths).
    #[error("path/query count mismatch: {num_paths} paths but {num_queries} queries")]
    PathQueryCountMismatch {
        /// Number of unique paths.
        num_paths: usize,
        /// Number of original queries.
        num_queries: usize,
    },
}
