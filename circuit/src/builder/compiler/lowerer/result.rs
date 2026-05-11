//! Lowering pipeline output type.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;

use super::state::LoweringState;
use crate::ops::Op;
use crate::types::{ExprId, WitnessId};

/// Complete output of the expression lowering pipeline.
pub struct LoweringResult<F> {
    /// Flat sequence of primitive operations in emission order.
    pub ops: Vec<Op<F>>,
    /// Witness slot for each public input, indexed by declaration position.
    pub public_rows: Vec<WitnessId>,
    /// Witness slot for each private input, indexed by declaration position.
    pub private_input_rows: Vec<WitnessId>,
    /// Mapping from every lowered expression to its witness slot.
    pub expr_to_widx: HashMap<ExprId, WitnessId>,
    /// Subset of the expression-to-witness map containing only public inputs.
    pub public_mappings: HashMap<ExprId, WitnessId>,
    /// Total number of witness slots allocated during lowering.
    pub witness_count: u32,
}

/// Consume the mutable lowering state into an immutable result.
///
/// Extracts all accumulated outputs and queries the allocator for the final witness count.
impl<F: Field> From<LoweringState<'_, F>> for LoweringResult<F> {
    fn from(state: LoweringState<'_, F>) -> Self {
        Self {
            // Move all accumulated vectors and maps out of the mutable state.
            ops: state.ops,
            public_rows: state.public_rows,
            private_input_rows: state.private_input_rows,
            expr_to_widx: state.expr_to_widx,
            public_mappings: state.public_mappings,
            // Query the allocator for the total number of slots used.
            witness_count: state.witness_alloc.witness_count(),
        }
    }
}
