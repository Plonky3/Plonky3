//! Expression graph lowering pipeline.

mod connect_dsu;
mod result;
pub(super) mod state;

#[cfg(test)]
mod tests;

use alloc::sync::Arc;

use hashbrown::HashMap;
use p3_field::Field;
pub use result::LoweringResult;
use state::LoweringState;

use crate::builder::CircuitBuilderError;
use crate::builder::npo::{NonPrimitiveOperationData, NpoCircuitPlugin};
use crate::expr::ExpressionGraph;
use crate::ops::NpoTypeId;
use crate::types::{ExprId, WitnessAllocator};

/// Front-end for the expression lowering pipeline.
///
/// Holds all immutable context needed to lower an expression DAG into
/// a flat sequence of primitive operations with witness allocation.
pub struct ExpressionLowerer<'a, F: Field> {
    /// The expression DAG to lower.
    pub(super) graph: &'a ExpressionGraph<F>,
    /// Registered non-primitive operations (hints, table-backed ops).
    pub(super) non_primitive_ops: &'a [NonPrimitiveOperationData<F>],
    /// Pairs of expressions that must share the same witness slot.
    pub(super) pending_connects: &'a [(ExprId, ExprId)],
    /// Number of declared public inputs.
    pub(super) public_input_count: usize,
    /// Number of declared private inputs.
    pub(super) private_input_count: usize,
    /// Monotonic witness slot allocator.
    pub(super) witness_alloc: WitnessAllocator,
    /// Plugin registry for table-backed non-primitive operations.
    pub(super) npo_registry: &'a HashMap<NpoTypeId, Arc<dyn NpoCircuitPlugin<F>>>,
}

impl<'a, F: Field> ExpressionLowerer<'a, F> {
    /// Create a new lowerer with all required context.
    pub const fn new(
        graph: &'a ExpressionGraph<F>,
        non_primitive_ops: &'a [NonPrimitiveOperationData<F>],
        pending_connects: &'a [(ExprId, ExprId)],
        public_input_count: usize,
        private_input_count: usize,
        witness_alloc: WitnessAllocator,
        npo_registry: &'a HashMap<NpoTypeId, Arc<dyn NpoCircuitPlugin<F>>>,
    ) -> Self {
        Self {
            graph,
            non_primitive_ops,
            pending_connects,
            public_input_count,
            private_input_count,
            witness_alloc,
            npo_registry,
        }
    }

    /// Run the full lowering pipeline and return the result.
    pub fn lower(self) -> Result<LoweringResult<F>, CircuitBuilderError> {
        // Initialise mutable state (builds the DSU and validates the NPO output map).
        let mut state = LoweringState::new(self)?;
        // Emit all constant nodes first so they are available as operands.
        state.emit_constants();
        // Emit public and private inputs, recording their witness slots.
        state.emit_publics();
        state.emit_privates();
        // Emit arithmetic and non-primitive operations in DAG order.
        state.emit_operations()?;
        // Verify that no registered non-primitive operation was left unreachable.
        state.validate_all_npo_emitted()?;
        // Fill in witness mappings for connect-class members not directly visited.
        state.backfill_connect_mappings();
        // Convert the accumulated mutable state into the immutable result.
        Ok(state.into())
    }
}
