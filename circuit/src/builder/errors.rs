use alloc::string::String;

use thiserror::Error;

use crate::ExprId;
use crate::ops::NpoTypeId;
use crate::types::NonPrimitiveOpId;

/// Errors that can occur during circuit building/lowering.
#[derive(Debug, Error)]
pub enum CircuitBuilderError {
    /// Expression not found in the witness mapping during lowering.
    #[error("Expression {expr_id:?} not found in witness mapping: {context}")]
    MissingExprMapping { expr_id: ExprId, context: String },

    /// Non-primitive op received an unexpected number of input expressions.
    #[error("{op} expects exactly {expected} witness expressions, got {got}")]
    NonPrimitiveOpArity {
        op: &'static str,
        expected: String,
        got: usize,
    },

    /// Non-primitive operation referenced by id was not found.
    #[error("Non-primitive operation id {op_id:?} not found")]
    MissingNonPrimitiveOp { op_id: NonPrimitiveOpId },

    /// Non-primitive output indices for an op are malformed (duplicates or gaps).
    #[error("Non-primitive output indices malformed for op {op_id:?}: {details}")]
    MalformedNonPrimitiveOutputs {
        op_id: NonPrimitiveOpId,
        details: String,
    },

    /// Non-primitive operation exists in the builder but was never anchored in the expression DAG,
    /// so the lowerer cannot place it in a well-defined execution order.
    #[error("Non-primitive operation {op_id:?} is not anchored in the expression DAG")]
    UnanchoredNonPrimitiveOp { op_id: NonPrimitiveOpId },

    /// Non-primitive operation rejected by the active policy/profile.
    #[error("Operation {op:?} is not allowed by the current profile")]
    OpNotAllowed { op: NpoTypeId },

    /// Non-primitive operation is recognized but not implemented in lowering.
    #[error("Operation {op:?} is not implemented in lowering")]
    UnsupportedNonPrimitiveOp { op: NpoTypeId },

    /// Mismatched non-primitive operation configuration
    #[error("Invalid configuration for operation {op:?}")]
    InvalidNonPrimitiveOpConfiguration { op: NpoTypeId },

    /// Merkle-path Poseidon2 rows require a direction bit.
    #[error("Poseidon2Perm merkle_path=true requires mmcs_bit")]
    Poseidon2MerkleMissingMmcsBit,

    /// Non-merkle Poseidon2 rows should not have mmcs_bit set.
    #[error("Poseidon2Perm merkle_path=false must not have mmcs_bit (it has no effect)")]
    Poseidon2NonMerkleWithMmcsBit,

    /// Poseidon2 configuration mismatch.
    #[error("Poseidon2 config mismatch: expected {expected}, got {got}")]
    Poseidon2ConfigMismatch { expected: String, got: String },

    /// Requested bit length exceeds the maximum allowed for binary decomposition.
    #[error("Too many bits for binary decomposition: expected at most {expected}, got {n_bits}")]
    BinaryDecompositionTooManyBits { expected: usize, n_bits: usize },

    /// Missing output
    #[error("An output was expected but none was given")]
    MissingOutput,

    /// Duplicate tag: a tag with this name was already registered.
    #[error("Duplicate tag: '{tag}' is already registered")]
    DuplicateTag { tag: String },

    /// Wrong batch size passed to recursive MMCS verifier: expected one length, got another.
    #[error("Wrong batch size: expected {expected}, got {got}")]
    WrongBatchSize { expected: usize, got: usize },

    /// Failed to format openings for MMCS preprocessing; preserves some context.
    #[error("Failed to format openings for operation {op:?}: {details}")]
    FormatOpeningsFailed { op: NpoTypeId, details: String },

    /// Invalid dimension: expected a specific number of elements.
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
}
