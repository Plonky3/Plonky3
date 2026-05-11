use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use hashbrown::HashMap;
use p3_field::{Dup, Field};
use tracing::instrument;

use super::alu::{AluOpRecord, AluTrace};
use super::constant::ConstTraceBuilder;
use super::public::PublicTraceBuilder;
use super::witness::WitnessTrace;
use super::{NonPrimitiveTrace, Traces};
use crate::circuit::Circuit;
use crate::ops::{ExecutionContext, NpoPrivateData, NpoTypeId, Op, OpStateMap};
use crate::types::{NonPrimitiveOpId, WitnessId};
use crate::{AluOpKind, CircuitError};

/// Circuit execution engine.
pub struct CircuitRunner<'a, F> {
    /// Borrowed circuit specification.
    circuit: &'a Circuit<F>,
    /// Witness values (None = unset, Some = computed).
    witness: Vec<Option<F>>,
    /// ALU deduplication rewrite map.
    witness_rewrite: Option<HashMap<WitnessId, WitnessId>>,
    /// Private data for non-primitive operations (not on witness bus)
    non_primitive_op_private_data: Vec<Option<NpoPrivateData>>,
    /// Map from NonPrimitiveOpId -> index in `circuit.ops` for type checks.
    non_primitive_op_index_by_id: Vec<Option<usize>>,
    /// Operation-specific execution state (e.g., Poseidon chaining, row records).
    op_states: OpStateMap,
}

impl<'a, F: Field> CircuitRunner<'a, F> {
    /// Creates circuit runner with empty witness storage.
    pub fn new(circuit: &'a Circuit<F>) -> Self {
        let witness = vec![None; circuit.witness_count as usize];
        let mut max_op_id: Option<u32> = None;
        for op in &circuit.ops {
            if let Op::NonPrimitiveOpWithExecutor { op_id, .. } = op {
                max_op_id = Some(max_op_id.map_or(op_id.0, |cur| cur.max(op_id.0)));
            }
        }
        let non_primitive_op_count = max_op_id.map_or(0, |m| m as usize + 1);

        let mut non_primitive_op_index_by_id = vec![None; non_primitive_op_count];
        for (idx, op) in circuit.ops.iter().enumerate() {
            if let Op::NonPrimitiveOpWithExecutor { op_id, .. } = op
                && let Some(slot) = non_primitive_op_index_by_id.get_mut(op_id.0 as usize)
            {
                #[cfg(debug_assertions)]
                debug_assert!(
                    slot.is_none(),
                    "duplicate NonPrimitiveOpId({}) in circuit.ops",
                    op_id.0
                );
                // Keep the first occurrence if duplicates exist (release builds).
                if slot.is_none() {
                    *slot = Some(idx);
                }
            }
        }

        let mut non_primitive_op_private_data: Vec<Option<NpoPrivateData>> =
            Vec::with_capacity(non_primitive_op_count);
        non_primitive_op_private_data.resize_with(non_primitive_op_count, || None);
        let witness_rewrite = circuit.witness_rewrite.clone();
        let op_states = BTreeMap::new();
        Self {
            circuit,
            witness,
            witness_rewrite,
            non_primitive_op_private_data,
            non_primitive_op_index_by_id,
            op_states,
        }
    }

    /// Sets public input values into witness table.
    pub fn set_public_inputs(&mut self, public_values: &[F]) -> Result<(), CircuitError> {
        if public_values.len() != self.circuit.public_flat_len {
            return Err(CircuitError::PublicInputLengthMismatch {
                expected: self.circuit.public_flat_len,
                got: public_values.len(),
            });
        }
        if self.circuit.public_rows.len() != self.circuit.public_flat_len {
            return Err(CircuitError::MissingPublicRowsMapping);
        }

        for (i, value) in public_values.iter().enumerate() {
            let widx = self.circuit.public_rows[i];
            self.set_witness(widx, *value)?;
        }

        Ok(())
    }

    /// Sets private input values into witness table.
    ///
    /// Private inputs do not create Public table rows, and should be constrained by downstream use.
    pub fn set_private_inputs(&mut self, private_values: &[F]) -> Result<(), CircuitError> {
        if private_values.len() != self.circuit.private_flat_len {
            return Err(CircuitError::PrivateInputLengthMismatch {
                expected: self.circuit.private_flat_len,
                got: private_values.len(),
            });
        }
        if self.circuit.private_input_rows.len() != self.circuit.private_flat_len {
            return Err(CircuitError::MissingPrivateRowsMapping);
        }

        for (i, value) in private_values.iter().enumerate() {
            let widx = self.circuit.private_input_rows[i];
            self.set_witness(widx, *value)?;
        }

        Ok(())
    }

    /// Sets private data for a non-primitive operation.
    pub fn set_private_data(
        &mut self,
        op_id: NonPrimitiveOpId,
        private_data: NpoPrivateData,
    ) -> Result<(), CircuitError> {
        // Validate that the op_id exists in the circuit.
        if op_id.0 as usize >= self.non_primitive_op_private_data.len()
            || self
                .non_primitive_op_index_by_id
                .get(op_id.0 as usize)
                .and_then(|x| *x)
                .is_none()
        {
            return Err(CircuitError::NonPrimitiveOpIdOutOfRange {
                op_id: op_id.0,
                max_ops: self.non_primitive_op_private_data.len(),
            });
        }

        // Validate that the private data matches the operation type
        let op_idx = self
            .non_primitive_op_index_by_id
            .get(op_id.0 as usize)
            .and_then(|x| *x)
            .ok_or(CircuitError::NonPrimitiveOpIdOutOfRange {
                op_id: op_id.0,
                max_ops: self.non_primitive_op_private_data.len(),
            })?;
        let Op::NonPrimitiveOpWithExecutor { executor, .. } = &self.circuit.ops[op_idx] else {
            return Err(CircuitError::NonPrimitiveOpIdOutOfRange {
                op_id: op_id.0,
                max_ops: self.non_primitive_op_private_data.len(),
            });
        };
        // Disallow double-setting private data
        if self.non_primitive_op_private_data[op_id.0 as usize].is_some() {
            return Err(CircuitError::IncorrectNonPrimitiveOpPrivateData {
                op: executor.op_type().clone(),
                operation_index: op_id,
                expected: "private data not previously set".to_string(),
                got: "already set".to_string(),
            });
        }

        // Store private data for this operation
        self.non_primitive_op_private_data[op_id.0 as usize] = Some(private_data);
        Ok(())
    }

    /// Sets private data for a non-primitive operation by its tag.
    ///
    /// The tag must have been registered during circuit construction via `builder.tag_op()`.
    ///
    /// # Errors
    /// Returns `CircuitError::UnknownTag` if the tag was not registered.
    pub fn set_private_data_by_tag(
        &mut self,
        tag: &str,
        private_data: NpoPrivateData,
    ) -> Result<(), CircuitError> {
        let op_id = self.circuit.tag_to_op_id.get(tag).copied().ok_or_else(|| {
            CircuitError::UnknownTag {
                tag: tag.to_string(),
            }
        })?;
        self.set_private_data(op_id, private_data)
    }

    /// Run the circuit and generate traces
    #[instrument(skip_all)]
    pub fn run(mut self) -> Result<Traces<F>, CircuitError> {
        let alu_records = self.execute_all()?;

        if let Some(rewrite) = self.witness_rewrite.take() {
            let mut resolved: HashMap<WitnessId, WitnessId> = HashMap::with_capacity(rewrite.len());
            let mut root = |canon: WitnessId| {
                *resolved.entry(canon).or_insert_with(|| {
                    let mut cur = canon;
                    while let Some(&next) = rewrite.get(&cur) {
                        cur = next;
                    }
                    cur
                })
            };
            for (dup, canon) in &rewrite {
                let r = root(*canon);
                if let Some(ref val) = self.witness[r.0 as usize] {
                    self.set_witness(*dup, *val)?;
                }
            }
        }

        // Build witness trace directly from the populated witness table.
        let mut witness_values = Vec::with_capacity(self.witness.len());
        for (i, value) in self.witness.iter().enumerate() {
            witness_values.push((*value).ok_or(CircuitError::WitnessNotSetForIndex { index: i })?);
        }
        let witness_trace = WitnessTrace::new(witness_values);

        let const_trace = ConstTraceBuilder::new(&self.circuit.ops).build()?;
        let public_trace = PublicTraceBuilder::new(&self.circuit.ops, &self.witness).build()?;
        let alu_trace = AluTrace::from_records(alu_records);

        let mut non_primitive_traces: HashMap<NpoTypeId, Box<dyn NonPrimitiveTrace<F>>> =
            HashMap::with_capacity(self.circuit.non_primitive_trace_generator_order.len());
        // Iterate over generators in deterministic order (sorted by key)
        let _scope = tracing::debug_span!("generators").entered();

        for op_type in &self.circuit.non_primitive_trace_generator_order {
            let generator = &self.circuit.non_primitive_trace_generators[op_type];
            if let Some(trace) = generator(&self.op_states)? {
                let trace_op_type = trace.op_type();
                non_primitive_traces.insert(trace_op_type, trace);
            }
        }
        _scope.exit();

        Ok(Traces {
            witness_trace,
            const_trace,
            public_trace,
            alu_trace,
            tag_to_witness: self.circuit.tag_to_witness.clone(),
            non_primitive_traces,
        })
    }

    /// Executes the full circuit operation list to populate witness table.
    ///
    /// The circuit is already lowered into a valid execution order, so this function
    /// can blindly execute from index 0 to end.
    #[instrument(skip_all, level = "debug")]
    pub fn execute_all(&mut self) -> Result<Vec<AluOpRecord<F>>, CircuitError> {
        let mut alu_records = Vec::with_capacity(self.circuit.ops.len());

        for op in &self.circuit.ops {
            match op {
                Op::Const { out, val } => {
                    self.set_witness(*out, *val)?;
                }
                Op::Public { out, public_pos: _ } => {
                    if self.witness[out.0 as usize].is_none() {
                        return Err(CircuitError::PublicInputNotSet { witness_id: *out });
                    }
                }
                Op::Alu {
                    kind,
                    a,
                    b,
                    c,
                    out,
                    intermediate_out,
                } => {
                    let kind = *kind;
                    let a = *a;
                    let b = *b;
                    let c_index = c.unwrap_or(WitnessId(0));
                    let out = *out;
                    let intermediate_out = *intermediate_out;
                    match kind {
                        AluOpKind::Add => {
                            let a_val = self.get_witness(a)?;
                            if let Some(b_val) = self.witness_value(b) {
                                let result = a_val + b_val;
                                self.set_witness(out, result)?;
                                alu_records.push(AluOpRecord {
                                    kind,
                                    a_index: a,
                                    b_index: b,
                                    c_index,
                                    out_index: out,
                                    a_val,
                                    b_val,
                                    c_val: F::ZERO,
                                    out_val: result,
                                });
                            } else {
                                let out_val = self.get_witness(out)?;
                                let b_val = out_val - a_val;
                                self.set_witness(b, b_val)?;
                                alu_records.push(AluOpRecord {
                                    kind,
                                    a_index: a,
                                    b_index: b,
                                    c_index,
                                    out_index: out,
                                    a_val,
                                    b_val,
                                    c_val: F::ZERO,
                                    out_val,
                                });
                            }
                        }
                        AluOpKind::Mul => {
                            let a_val = self.get_witness(a)?;
                            if let Some(b_val) = self.witness_value(b) {
                                let result = a_val * b_val;
                                self.set_witness(out, result)?;
                                alu_records.push(AluOpRecord {
                                    kind,
                                    a_index: a,
                                    b_index: b,
                                    c_index,
                                    out_index: out,
                                    a_val,
                                    b_val,
                                    c_val: F::ZERO,
                                    out_val: result,
                                });
                            } else {
                                let result_val = self.get_witness(out)?;
                                let a_inv =
                                    a_val.try_inverse().ok_or(CircuitError::DivisionByZero)?;
                                let b_val = result_val * a_inv;
                                self.set_witness(b, b_val)?;
                                alu_records.push(AluOpRecord {
                                    kind,
                                    a_index: a,
                                    b_index: b,
                                    c_index,
                                    out_index: out,
                                    a_val,
                                    b_val,
                                    c_val: F::ZERO,
                                    out_val: result_val,
                                });
                            }
                        }
                        AluOpKind::BoolCheck => {
                            let a_val = self.get_witness(a)?;
                            self.set_witness(out, a_val)?;
                            alu_records.push(AluOpRecord {
                                kind,
                                a_index: a,
                                b_index: b,
                                c_index,
                                out_index: out,
                                a_val,
                                b_val: F::ZERO,
                                c_val: a_val,
                                out_val: a_val,
                            });
                        }
                        AluOpKind::MulAdd => {
                            let a_val = self.get_witness(a)?;
                            let b_val = self.get_witness(b)?;
                            let ab_product = a_val * b_val;
                            let c_id_opt = *c;

                            if let Some(io) = intermediate_out {
                                self.set_witness(io, ab_product)?;
                            }

                            let c_val = if let Some(c_id) = c_id_opt {
                                self.get_witness(c_id)?
                            } else {
                                F::ZERO
                            };
                            let out_val = ab_product + c_val;
                            self.set_witness(out, out_val)?;
                            alu_records.push(AluOpRecord {
                                kind,
                                a_index: a,
                                b_index: b,
                                c_index,
                                out_index: out,
                                a_val,
                                b_val,
                                c_val,
                                out_val,
                            });
                        }
                        AluOpKind::HornerAcc => {
                            let acc_id = intermediate_out
                                .expect("HornerAcc requires acc in intermediate_out");
                            let c_id = c.expect("HornerAcc requires c operand");
                            let acc_val = self.get_witness(acc_id)?;
                            let a_val = self.get_witness(a)?;
                            let b_val = self.get_witness(b)?;
                            let c_val = self.get_witness(c_id)?;
                            let result = acc_val * b_val + c_val - a_val;
                            self.set_witness(out, result)?;
                            alu_records.push(AluOpRecord {
                                kind,
                                a_index: a,
                                b_index: b,
                                c_index,
                                out_index: out,
                                a_val,
                                b_val,
                                c_val,
                                out_val: result,
                            });
                        }
                    }
                }
                Op::Hint {
                    inputs,
                    outputs,
                    executor,
                } => {
                    executor.execute(inputs, outputs, &mut self.witness)?;
                }
                Op::NonPrimitiveOpWithExecutor {
                    inputs,
                    outputs,
                    executor,
                    op_id,
                } => {
                    let mut ctx = ExecutionContext::new(
                        &mut self.witness,
                        &self.non_primitive_op_private_data,
                        &self.circuit.enabled_ops,
                        *op_id,
                        &mut self.op_states,
                    );

                    executor.execute(inputs, outputs, &mut ctx)?;
                }
            }
        }

        Ok(alu_records)
    }

    /// Witness value if the slot exists and is set (`None` = unset or out of range).
    #[inline(always)]
    fn witness_value(&self, widx: WitnessId) -> Option<F> {
        self.witness
            .get(widx.0 as usize)
            .and_then(|opt| opt.as_ref().map(Dup::dup))
    }

    /// Gets witness value by ID.
    #[inline(always)]
    fn get_witness(&self, widx: WitnessId) -> Result<F, CircuitError> {
        self.witness_value(widx)
            .ok_or(CircuitError::WitnessNotSet { witness_id: widx })
    }

    /// Sets witness value by ID.
    #[inline(always)]
    fn set_witness(&mut self, widx: WitnessId, value: F) -> Result<(), CircuitError> {
        if widx.0 as usize >= self.witness.len() {
            return Err(CircuitError::WitnessIdOutOfBounds { witness_id: widx });
        }

        let slot = &mut self.witness[widx.0 as usize];

        // Check for conflicting reassignment
        if let Some(existing_value) = slot.as_ref() {
            if *existing_value == value {
                return Ok(());
            }
            #[cfg(feature = "debugging")]
            let expr_ids = self
                .circuit
                .expr_to_widx
                .iter()
                .filter_map(|(expr_id, &witness_id)| {
                    if witness_id == widx {
                        Some(*expr_id)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            #[cfg(not(feature = "debugging"))]
            let expr_ids = vec![];

            return Err(CircuitError::WitnessConflict {
                witness_id: widx,
                existing: format!("{existing_value:?}"),
                new: format!("{value:?}"),
                expr_ids,
            });
        }

        *slot = Some(value);
        Ok(())
    }

    /// Reference to the witness slice (for benchmarking trace builders after `execute_all`).
    pub fn witness(&self) -> &[Option<F>] {
        &self.witness
    }

    /// Reference to the circuit ops (for benchmarking trace builders after `execute_all`).
    pub fn ops(&self) -> &[Op<F>] {
        &self.circuit.ops
    }
}

#[cfg(test)]
mod tests {
    use p3_test_utils::baby_bear_params::{
        BabyBear, BasedVectorSpace, BinomialExtensionField, Field, PrimeCharacteristicRing,
    };
    use tracing_forest::ForestLayer;
    use tracing_forest::util::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Registry};

    use super::*;
    use crate::builder::CircuitBuilder;
    use crate::ops::HintExecutor;
    use crate::tables::{ConstTrace, PublicTrace};
    use crate::types::WitnessId;

    /// Initializes a global logger with default parameters.
    fn init_logger() {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }

    #[test]
    fn test_table_generation_basic() {
        let mut builder = CircuitBuilder::new();

        // Simple test: x + 5 = result
        let x = builder.public_input();
        let c5 = builder.define_const(BabyBear::from_u64(5));
        let _result = builder.add(x, c5);

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();

        // Set public input: x = 3
        runner.set_public_inputs(&[BabyBear::from_u64(3)]).unwrap();

        let traces = runner.run().unwrap();

        let f = BabyBear::from_u64;
        assert_eq!(
            traces,
            Traces {
                witness_trace: WitnessTrace::new(vec![f(0), f(5), f(3), f(8)]),
                const_trace: ConstTrace {
                    index: vec![WitnessId(0), WitnessId(1)],
                    values: vec![f(0), f(5)],
                },
                public_trace: PublicTrace {
                    index: vec![WitnessId(2)],
                    values: vec![f(3)],
                },
                alu_trace: AluTrace {
                    op_kind: vec![AluOpKind::Add],
                    values: vec![[f(3), f(5), f(0), f(8)]],
                    indices: vec![[WitnessId(2), WitnessId(1), WitnessId(0), WitnessId(3)]],
                },
                non_primitive_traces: HashMap::new(),
                tag_to_witness: HashMap::new(),
            }
        );
    }

    #[derive(Debug, Clone)]
    /// The hint defined by x in an equation a*x - b = 0
    struct XHint;

    impl XHint {
        pub fn new() -> Self {
            Self
        }
    }

    impl<F: Field> HintExecutor<F> for XHint {
        fn execute(
            &self,
            inputs: &[WitnessId],
            outputs: &[WitnessId],
            witness: &mut [Option<F>],
        ) -> Result<(), CircuitError> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(CircuitError::UnconstrainedOpInputLengthMismatch {
                    op: "XHint".to_string(),
                    expected: 2,
                    got: inputs.len(),
                });
            }

            let a_idx = inputs[0].0 as usize;
            let b_idx = inputs[1].0 as usize;

            let a = witness
                .get(a_idx)
                .and_then(|opt| opt.as_ref())
                .copied()
                .ok_or(CircuitError::WitnessNotSet {
                    witness_id: inputs[0],
                })?;
            let b = witness
                .get(b_idx)
                .and_then(|opt| opt.as_ref())
                .copied()
                .ok_or(CircuitError::WitnessNotSet {
                    witness_id: inputs[1],
                })?;

            let inv_a = a.try_inverse().ok_or(CircuitError::DivisionByZero)?;
            let x = b * inv_a;

            let out_wid = outputs[0];
            let out_idx = out_wid.0 as usize;
            if out_idx >= witness.len() {
                return Err(CircuitError::WitnessIdOutOfBounds {
                    witness_id: out_wid,
                });
            }
            let slot = &mut witness[out_idx];
            if let Some(existing) = slot.as_ref() {
                if *existing != x {
                    return Err(CircuitError::WitnessConflict {
                        witness_id: out_wid,
                        existing: format!("{existing:?}"),
                        new: format!("{x:?}"),
                        expr_ids: vec![],
                    });
                }
            } else {
                *slot = Some(x);
            }

            Ok(())
        }

        fn boxed(&self) -> Box<dyn HintExecutor<F>> {
            Box::new(self.clone())
        }
    }

    #[test]
    // Proves that we know x such that 37 * x - 111 = 0
    fn test_toy_example_37_times_x_minus_111() {
        init_logger();

        let mut builder = CircuitBuilder::new();

        let c37 = builder.define_const(BabyBear::from_u64(37));
        let c111 = builder.define_const(BabyBear::from_u64(111));
        let x_hint = XHint::new();
        let x = builder
            .push_unconstrained_op(vec![vec![c37, c111]], 1, x_hint, "x")
            .2[0]
            .unwrap();

        let mul_result = builder.mul(c37, x);
        let sub_result = builder.sub(mul_result, c111);
        builder.assert_zero(sub_result);

        let circuit = builder.build().unwrap();

        let runner = circuit.runner();

        let traces = runner.run().unwrap();

        let f = BabyBear::from_u64;
        let neg_111 = -f(111);
        assert_eq!(
            traces,
            Traces {
                witness_trace: WitnessTrace::new(vec![f(0), f(37), f(111), f(3), f(111), neg_111]),
                const_trace: ConstTrace {
                    index: vec![WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(5)],
                    values: vec![f(0), f(37), f(111), neg_111],
                },
                public_trace: PublicTrace {
                    index: vec![],
                    values: vec![],
                },
                alu_trace: AluTrace {
                    op_kind: vec![AluOpKind::Mul, AluOpKind::Add],
                    values: vec![[f(37), f(3), f(0), f(111)], [f(111), neg_111, f(0), f(0)]],
                    indices: vec![
                        [WitnessId(1), WitnessId(3), WitnessId(0), WitnessId(4)],
                        [WitnessId(4), WitnessId(5), WitnessId(0), WitnessId(0)],
                    ],
                },
                non_primitive_traces: HashMap::new(),
                tag_to_witness: HashMap::new(),
            }
        );
    }

    #[test]
    fn test_extension_field_support() {
        type ExtField = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::new();

        // Test extension field operations: y * z + x (fused multiply-add)
        let x = builder.public_input();
        let y = builder.public_input();
        let z = builder.public_input();

        let _result = builder.mul_add(y, z, x);

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();

        // Set public inputs to genuine extension field values with ALL non-zero coefficients
        let x_val = ExtField::from_basis_coefficients_slice(&[
            BabyBear::from_u64(1), // a0
            BabyBear::from_u64(2), // a1
            BabyBear::from_u64(3), // a2
            BabyBear::from_u64(4), // a3
        ])
        .unwrap();
        let y_val = ExtField::from_basis_coefficients_slice(&[
            BabyBear::from_u64(5), // b0
            BabyBear::from_u64(6), // b1
            BabyBear::from_u64(7), // b2
            BabyBear::from_u64(8), // b3
        ])
        .unwrap();
        let z_val = ExtField::from_basis_coefficients_slice(&[
            BabyBear::from_u64(9),  // c0
            BabyBear::from_u64(10), // c1
            BabyBear::from_u64(11), // c2
            BabyBear::from_u64(12), // c3
        ])
        .unwrap();

        runner.set_public_inputs(&[x_val, y_val, z_val]).unwrap();
        let traces = runner.run().unwrap();

        let result = y_val * z_val + x_val;
        assert_eq!(
            traces,
            Traces {
                witness_trace: WitnessTrace::new(
                    vec![ExtField::ZERO, x_val, y_val, z_val, result,]
                ),
                const_trace: ConstTrace {
                    index: vec![WitnessId(0)],
                    values: vec![ExtField::ZERO],
                },
                public_trace: PublicTrace {
                    index: vec![WitnessId(1), WitnessId(2), WitnessId(3)],
                    values: vec![x_val, y_val, z_val],
                },
                alu_trace: AluTrace {
                    op_kind: vec![AluOpKind::MulAdd],
                    values: vec![[y_val, z_val, x_val, result]],
                    indices: vec![[WitnessId(2), WitnessId(3), WitnessId(1), WitnessId(4)]],
                },
                non_primitive_traces: HashMap::new(),
                tag_to_witness: HashMap::new(),
            }
        );
    }

    #[test]
    fn test_set_private_data_by_unknown_tag_error() {
        use crate::ops::poseidon2_perm::Poseidon2PermPrivateData;

        let builder = CircuitBuilder::<BabyBear>::new();
        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();

        let private_data = Poseidon2PermPrivateData {
            sibling: vec![BabyBear::ZERO, BabyBear::ZERO],
        };

        let result = runner.set_private_data_by_tag(
            "nonexistent-tag",
            crate::ops::NpoPrivateData::new(private_data),
        );

        assert!(matches!(
            result,
            Err(CircuitError::UnknownTag { tag }) if tag == "nonexistent-tag"
        ));
    }
}
