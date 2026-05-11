use alloc::string::String;
use alloc::vec::Vec;

use thiserror::Error;

use crate::ops::NpoTypeId;
use crate::types::NonPrimitiveOpId;
use crate::{CircuitBuilderError, ExprId, WitnessId};

/// Errors that can occur during circuit execution and trace generation.
#[derive(Debug, Error)]
pub enum CircuitError {
    /// Public input length mismatch.
    #[error("Public input length mismatch: expected {expected}, got {got}")]
    PublicInputLengthMismatch { expected: usize, got: usize },

    /// Circuit missing public_rows mapping.
    #[error("Circuit missing public_rows mapping")]
    MissingPublicRowsMapping,

    /// Private input length mismatch.
    #[error("Private input length mismatch: expected {expected}, got {got}")]
    PrivateInputLengthMismatch { expected: usize, got: usize },

    /// Circuit missing private_input_rows mapping.
    #[error("Circuit missing private_input_rows mapping")]
    MissingPrivateRowsMapping,

    /// NonPrimitiveOpId out of range.
    #[error("NonPrimitiveOpId {op_id} out of range (circuit has {max_ops} complex ops)")]
    NonPrimitiveOpIdOutOfRange { op_id: u32, max_ops: usize },

    /// Public input not set for a WitnessId.
    #[error("Public input not set for WitnessId({witness_id})")]
    PublicInputNotSet { witness_id: WitnessId },

    /// Witness not set for a WitnessId.
    #[error("Witness not set for WitnessId({witness_id})")]
    WitnessNotSet { witness_id: WitnessId },

    /// WitnessId out of bounds.
    #[error("WitnessId({witness_id}) out of bounds")]
    WitnessIdOutOfBounds { witness_id: WitnessId },

    /// Witness conflict: trying to reassign to a different value.
    #[error(
        "Witness conflict: WitnessId({witness_id}) already set to {existing}, cannot reassign to {new}; corresponding ExprIds: {expr_ids:?}"
    )]
    WitnessConflict {
        witness_id: WitnessId,
        existing: String,
        new: String,
        expr_ids: Vec<ExprId>,
    },

    /// Witness not set for an index during trace generation.
    #[error("Witness not set for index {index}")]
    WitnessNotSetForIndex { index: usize },

    /// Non-primitive op attempted to read a witness value that was not set.
    #[error("Witness value not set for non-primitive operation {operation_index}")]
    NonPrimitiveOpWitnessNotSet { operation_index: NonPrimitiveOpId },

    /// Missing private data for a non-primitive operation.
    #[error("Missing private data for non-primitive operation {operation_index}")]
    NonPrimitiveOpMissingPrivateData { operation_index: NonPrimitiveOpId },

    /// Division by zero encountered.
    #[error("Division by zero encountered")]
    DivisionByZero,

    /// Invalid bit value in SampleBits bit decomposition (must be 0 or 1).
    #[error(
        "Invalid bit value in SampleBits bit decomposition for WitnessId({input_witness_id}): {bit_value} (must be 0 or 1)"
    )]
    InvalidBitValue {
        input_witness_id: WitnessId,
        bit_value: String,
    },

    /// Bit decomposition doesn't reconstruct to the input value.
    #[error(
        "Bit decomposition for WitnessId({input_witness_id}) doesn't match input: expected {expected}, reconstructed {reconstructed}"
    )]
    BitDecompositionMismatch {
        input_witness_id: WitnessId,
        expected: String,
        reconstructed: String,
    },

    /// Mismatched non-primitive operation configuration
    #[error("Invalid configuration for operation {op:?}")]
    InvalidNonPrimitiveOpConfiguration { op: NpoTypeId },

    /// Non-primitive operation has incorrect input/output layout.
    #[error("Incorrect layout for operation {op:?}: expected {expected}, got {got}")]
    NonPrimitiveOpLayoutMismatch {
        op: NpoTypeId,
        expected: String,
        got: usize,
    },

    /// Incorrect size of private data provided for a non-primitive operation.
    #[error(
        "Incorrect size of private data provided for operation {op:?}: expected {expected}, got {got}"
    )]
    IncorrectNonPrimitiveOpPrivateDataSize {
        op: NpoTypeId,
        expected: String,
        got: usize,
    },

    /// Incorrect input size provided for a non-primitive operation.
    #[error("Incorrect input size provided for operation {op:?}: expected {expected}, got {got}")]
    IncorrectNonPrimitiveOpInputSize {
        op: NpoTypeId,
        expected: String,
        got: usize,
    },

    /// Non primitive private data is not correct
    #[error(
        "Incorrect private data provided for op {op:?} (operation {operation_index}): expected {expected}, got {got}"
    )]
    IncorrectNonPrimitiveOpPrivateData {
        op: NpoTypeId,
        operation_index: NonPrimitiveOpId,
        expected: String,
        got: String,
    },

    /// ExprId not found.
    #[error("ExprId {expr_id} not found")]
    ExprIdNotFound { expr_id: ExprId },

    /// Invalid Circuit
    #[error("Failed to build circuit: {error}")]
    InvalidCircuit { error: CircuitBuilderError },

    /// Unconstrained operation is given an incorrect number of inputs.
    #[error("Unconstrained operation input length mismatch: expected {op} {expected}, got {got}")]
    UnconstrainedOpInputLengthMismatch {
        op: String,
        expected: usize,
        got: usize,
    },

    /// Requested bit length exceeds the maximum allowed for binary decomposition.
    #[error("Too many bits for binary decomposition: expected at most {expected}, got {n_bits}")]
    BinaryDecompositionTooManyBits { expected: usize, n_bits: usize },

    /// Invalid preprocessed values
    #[error("Preprocessed values should be base field elements")]
    InvalidPreprocessedValues,

    /// Invalid preprocessing operation
    #[error("Invalid preprocessing: {reason}")]
    InvalidPreprocessing { reason: &'static str },

    /// Inconsistent matrix heights when formatting openings: heights that round up
    /// to the same power of two must be equal.
    #[error("Inconsistent matrix heights: {details}")]
    InconsistentMatrixHeights { details: String },

    /// Poseidon2 chaining requires previous state but none was available.
    #[error(
        "Poseidon2 chain missing previous state for operation {operation_index} (new_start=false but no previous permutation)"
    )]
    Poseidon2ChainMissingPreviousState { operation_index: NonPrimitiveOpId },

    /// Poseidon2 merkle path mode requires a sibling input limb (which limbs are required depends on `mmcs_bit`).
    #[error(
        "Poseidon2 merkle path missing sibling input for operation {operation_index}, limb {limb}"
    )]
    Poseidon2MerkleMissingSiblingInput {
        operation_index: NonPrimitiveOpId,
        limb: usize,
    },

    /// Poseidon2 operation is missing required input limb.
    #[error(
        "Poseidon2 operation {operation_index} missing input for limb {limb} (new_start=true requires all inputs)"
    )]
    Poseidon2MissingInput {
        operation_index: NonPrimitiveOpId,
        limb: usize,
    },

    /// Unknown tag: the tag was not registered during circuit construction.
    #[error("Unknown tag: '{tag}'")]
    UnknownTag { tag: String },
}

impl From<CircuitBuilderError> for CircuitError {
    fn from(error: CircuitBuilderError) -> Self {
        Self::InvalidCircuit { error }
    }
}
