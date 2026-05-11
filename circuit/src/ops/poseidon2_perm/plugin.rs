//! Poseidon2 circuit plugin — [`NpoCircuitPlugin`] implementation.

use alloc::boxed::Box;

use p3_field::Field;

use crate::CircuitBuilderError;
use crate::builder::{NpoCircuitPlugin, NpoLoweringContext};
use crate::ops::poseidon2_perm::config::{
    Poseidon2Config, Poseidon2PermConfigData, Poseidon2PermExec,
};
use crate::ops::poseidon2_perm::executor::Poseidon2PermExecutor;
use crate::ops::{NpoConfig, NpoTypeId, Op};
use crate::tables::TraceGeneratorFn;
use crate::types::ExprId;

/// Circuit-layer plugin for Poseidon2 non-primitive operations.
pub struct Poseidon2CircuitPlugin<F: Field> {
    type_id: NpoTypeId,
    poseidon2_config: Poseidon2Config,
    npo_config: NpoConfig,
    trace_gen: TraceGeneratorFn<F>,
}

impl<F: Field> Poseidon2CircuitPlugin<F> {
    pub fn new(
        poseidon2_config: Poseidon2Config,
        exec: Poseidon2PermExec<F>,
        trace_gen: TraceGeneratorFn<F>,
    ) -> Self {
        Self {
            type_id: NpoTypeId::poseidon2_perm(poseidon2_config),
            poseidon2_config,
            npo_config: NpoConfig::new(Poseidon2PermConfigData { exec }),
            trace_gen,
        }
    }
}

impl<F: Field> NpoCircuitPlugin<F> for Poseidon2CircuitPlugin<F> {
    fn type_id(&self) -> NpoTypeId {
        self.type_id.clone()
    }

    /// Lower a high-level Poseidon2 permutation operation into a concrete executor op.
    ///
    /// # Algorithm
    ///
    /// 1. Allocate witness slots for all output expressions.
    /// 2. Extract per-row flags (new_start, merkle_path) from the operation params.
    /// 3. Validate input/output counts against the config layout.
    /// 4. Lower input and output expressions to witness indices.
    /// 5. Return an executor-backed operation ready for runtime dispatch.
    fn lower(
        &self,
        data: &crate::builder::NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError> {
        // Ensure every output expression has a witness slot before lowering.
        for (_output_idx, expr_id) in output_exprs {
            ctx.ensure_witness_id(*expr_id);
        }

        // Extract the Poseidon2-specific per-row flags from the operation params.
        let (new_start, merkle_path) = data
            .params
            .as_ref()
            .and_then(|p| p.as_poseidon2_perm())
            .ok_or_else(|| CircuitBuilderError::InvalidNonPrimitiveOpConfiguration {
                op: data.op_type.clone(),
            })?;

        // Validate that the builder provided the expected number of inputs/outputs.
        let config = self.poseidon2_config;
        config.validate_io_counts(data.input_exprs.len(), data.output_exprs.len(), merkle_path)?;

        // Convert expression-level slots to witness-level slots.
        let inputs_widx = config.lower_inputs(&data.input_exprs, ctx, merkle_path)?;
        let outputs_widx = ctx.lower_expr_slots(&data.output_exprs, "Poseidon2Perm", "output")?;

        Ok(Op::NonPrimitiveOpWithExecutor {
            inputs: inputs_widx,
            outputs: outputs_widx,
            executor: Box::new(Poseidon2PermExecutor::new(
                data.op_type.clone(),
                config,
                new_start,
                merkle_path,
            )),
            op_id: data.op_id,
        })
    }

    fn trace_generator(&self) -> TraceGeneratorFn<F> {
        self.trace_gen
    }

    fn config(&self) -> crate::ops::NpoConfig {
        self.npo_config.clone()
    }
}
