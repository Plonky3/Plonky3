use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;
use strum::EnumCount;

use crate::ops::{
    NonPrimitivePreprocessedMap, NpoConfig, NpoTypeId, Op, PreprocessedWriter, PrimitiveOpType,
};
use crate::tables::{CircuitRunner, TraceGeneratorFn};
use crate::types::{ExprId, NonPrimitiveOpId, WitnessId};
use crate::{AluOpKind, CircuitError};

/// Preprocessed data for primitive and non-primitive operation tables.
///
/// The const generic `D` is the extension degree used for base-field index scaling.
/// A `WitnessId(n)` is stored as base-field index `n * D` in CTL lookup tuples.
#[derive(Debug)]
pub struct PreprocessedColumns<F, const D: usize> {
    pub primitive: Vec<Vec<F>>,
    pub non_primitive: NonPrimitivePreprocessedMap<F>,
    /// Ext-field read counts per witness index (indexed by `WitnessId.0`).
    ///
    /// `ext_reads[i]` is the number of times `WitnessId(i)` is read by any table
    /// as an extension-field value. This is used by creator tables to set their
    /// signed multiplicity on the `WitnessChecks` bus.
    pub ext_reads: Vec<u32>,
    /// Per-NPO duplicate-output flags: `dup_npo_outputs[op_type][wid] == true` means
    /// `WitnessId(wid)` was already defined by an earlier op and this NPO occurrence is
    /// a reader, not the creator. Populated by `generate_preprocessed_columns`.
    pub dup_npo_outputs: HashMap<NpoTypeId, Vec<bool>>,
    /// WitnessId.0 values for all `Op::Hint` outputs in the circuit.
    ///
    /// Used by prover preprocessors (e.g. `recompose_preprocess_impl`) to distinguish
    /// hint-derived witnesses (which need to be created on the WitnessChecks bus by the
    /// owning NPO table) from already-defined witnesses (e.g. Poseidon2 rate outputs that
    /// are also passed as recompose coefficients via `sample_ext`).
    pub hint_output_wids: hashbrown::HashSet<u32>,
}

impl<F: PartialEq, const D: usize> PartialEq for PreprocessedColumns<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.primitive == other.primitive
            && self.ext_reads == other.ext_reads
            && self.non_primitive == other.non_primitive
            && self.dup_npo_outputs == other.dup_npo_outputs
            && self.hint_output_wids == other.hint_output_wids
    }
}

impl<F: Eq, const D: usize> Eq for PreprocessedColumns<F, D> {}

impl<F: Field + Clone, const D: usize> Clone for PreprocessedColumns<F, D> {
    fn clone(&self) -> Self {
        Self {
            primitive: self.primitive.clone(),
            non_primitive: self.non_primitive.clone(),
            ext_reads: self.ext_reads.clone(),
            dup_npo_outputs: self.dup_npo_outputs.clone(),
            hint_output_wids: self.hint_output_wids.clone(),
        }
    }
}

impl<F: Field, const D: usize> PreprocessedColumns<F, D> {
    /// Creates an empty [`PreprocessedColumns`] with one primitive entry per [`PrimitiveOpType`].
    pub fn new() -> Self {
        const { assert!(D >= 1, "extension degree must be at least 1") };
        Self {
            primitive: vec![vec![]; PrimitiveOpType::COUNT],
            non_primitive: NonPrimitivePreprocessedMap::new(),
            ext_reads: Vec::new(),
            dup_npo_outputs: HashMap::new(),
            hint_output_wids: hashbrown::HashSet::new(),
        }
    }
}

impl<F: Field, const D: usize> PreprocessedWriter<F> for PreprocessedColumns<F, D> {
    fn witness_index_as_field(&self, wid: WitnessId) -> F {
        wid.base_field_index::<F, D>()
    }

    /// Increments the ext-field read count for each of the given witness indices.
    fn increment_ext_reads(&mut self, wids: &[WitnessId]) {
        for wid in wids {
            let idx = wid.0 as usize;
            if idx >= self.ext_reads.len() {
                self.ext_reads.resize(idx + 1, 0);
            }
            self.ext_reads[idx] += 1;
        }
    }

    /// Extends the preprocessed data of `op_type`'s non-primitive operation
    /// with `wids`'s witness indices (D-scaled). Does NOT increment ext-field read counts.
    ///
    /// Use this for non-primitive OUTPUTS: the table creates these witnesses on the
    /// `WitnessChecks` bus, so they are not readers. The `out_ctl` multiplicity is
    /// set separately by `get_airs_and_degrees_with_prep` based on `ext_reads`.
    fn register_non_primitive_output_index(&mut self, op_type: &NpoTypeId, wids: &[WitnessId]) {
        let entry = self.non_primitive.entry(op_type.clone()).or_default();
        let wids_field = wids.iter().map(|wid| wid.base_field_index::<F, D>());
        entry.extend(wids_field);
    }

    /// Extends the preprocessed data of `op_type`'s non-primitive operation
    /// with `wids`'s witness indices (D-scaled), and increments their ext-field read counts.
    ///
    /// Use this for non-primitive inputs that the table reads from the `WitnessChecks` bus.
    fn register_non_primitive_witness_reads(
        &mut self,
        op_type: &NpoTypeId,
        wids: &[WitnessId],
    ) -> Result<(), CircuitError> {
        let entry = self.non_primitive.entry(op_type.clone()).or_default();
        let wids_field = wids.iter().map(|wid| wid.base_field_index::<F, D>());
        entry.extend(wids_field);
        self.increment_ext_reads(wids);
        Ok(())
    }

    /// Extends the preprocessed data of `op_type`'s non-primitive operation with `values`.
    /// Does not update read counts.
    fn register_non_primitive_preprocessed_no_read(&mut self, op_type: &NpoTypeId, values: &[F]) {
        let entry = self.non_primitive.entry(op_type.clone()).or_default();
        entry.extend(values);
    }

    fn witness_extension_degree_slots(&self) -> usize {
        D
    }
}

impl<F: Field, const D: usize> Default for PreprocessedColumns<F, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Static circuit specification containing constraint system and metadata
///
/// This represents the compiled output of a `CircuitBuilder`. It contains:
/// - Primitive operations (add, multiply, subtract, constants, public inputs)
/// - Non-primitive operations (complex operations like MMCS verification)
/// - Public input metadata and witness table structure
///
/// The circuit is static and serializable. Use `.runner()` to create
/// a `CircuitRunner` for execution with specific input values.
#[derive(Debug)]
pub struct Circuit<F> {
    /// Number of witness table rows
    pub witness_count: u32,
    /// Operations in execution order (primitive + non-primitive).
    pub ops: Vec<Op<F>>,
    /// Public input witness indices
    pub public_rows: Vec<WitnessId>,
    /// Total number of public field elements
    pub public_flat_len: usize,
    /// Private input witness indices
    pub private_input_rows: Vec<WitnessId>,
    /// Total number of private input field elements
    pub private_flat_len: usize,
    /// Enabled non-primitive operation types with their respective configuration
    pub enabled_ops: HashMap<NpoTypeId, NpoConfig>,
    /// Expression to witness index map
    pub expr_to_widx: HashMap<ExprId, WitnessId>,
    /// Registered non-primitive trace generators.
    pub non_primitive_trace_generators: HashMap<NpoTypeId, TraceGeneratorFn<F>>,
    /// Sorted keys of `non_primitive_trace_generators` for deterministic iteration without sorting each run.
    pub non_primitive_trace_generator_order: Vec<NpoTypeId>,
    /// Tag to witness index mapping for probing values by name.
    pub tag_to_witness: HashMap<String, WitnessId>,
    /// Tag to non-primitive operation ID mapping.
    pub tag_to_op_id: HashMap<String, NonPrimitiveOpId>,
    /// After ALU deduplication, duplicate outputs are rewritten to canonical.
    /// This map is used by the runner to fill those slots.
    pub witness_rewrite: Option<HashMap<WitnessId, WitnessId>>,
}

impl<F: Field + Clone> Clone for Circuit<F> {
    fn clone(&self) -> Self {
        Self {
            witness_count: self.witness_count,
            ops: self.ops.clone(),
            public_rows: self.public_rows.clone(),
            public_flat_len: self.public_flat_len,
            private_input_rows: self.private_input_rows.clone(),
            private_flat_len: self.private_flat_len,
            enabled_ops: self.enabled_ops.clone(),
            expr_to_widx: self.expr_to_widx.clone(),
            non_primitive_trace_generators: self.non_primitive_trace_generators.clone(),
            non_primitive_trace_generator_order: self.non_primitive_trace_generator_order.clone(),
            tag_to_witness: self.tag_to_witness.clone(),
            tag_to_op_id: self.tag_to_op_id.clone(),
            witness_rewrite: self.witness_rewrite.clone(),
        }
    }
}

impl<F: Field> Circuit<F> {
    /// Create a new circuit with the given witness count and expression to witness index map.
    pub fn new(witness_count: u32, expr_to_widx: HashMap<ExprId, WitnessId>) -> Self {
        Self {
            witness_count,
            ops: Vec::new(),
            public_rows: Vec::new(),
            public_flat_len: 0,
            private_input_rows: Vec::new(),
            private_flat_len: 0,
            enabled_ops: HashMap::new(),
            expr_to_widx,
            non_primitive_trace_generators: HashMap::new(),
            non_primitive_trace_generator_order: Vec::new(),
            tag_to_witness: HashMap::new(),
            tag_to_op_id: HashMap::new(),
            witness_rewrite: None,
        }
    }

    /// Generates preprocessed columns for all primitive operation types.
    ///
    /// Returns a [`PreprocessedColumns`] with one primitive entry per [`PrimitiveOpType`]:
    ///
    /// | Index | Operation | Column Layout                                                              | Width (per op) |
    /// |-------|-----------|----------------------------------------------------------------------------|----------------|
    /// | 0     | Const     | `[out_0, out_1, ...]` (D-scaled indices)                                   | 1              |
    /// | 1     | Public    | `[out_0, out_1, ...]` (D-scaled indices)                                   | 1              |
    /// | 2     | Alu       | `[sel_add_vs_mul, sel_bool, sel_muladd, sel_horner, a_idx, b_idx, c_idx, out_idx, a_state, b_is_creator, c_state, out_is_creator]` | 12 |
    ///
    /// Signed multiplicities are not stored here; they are computed in `get_airs_and_degrees_with_prep`
    /// using the `ext_reads` field, which tracks how many times each witness is read.
    ///
    /// Indices in CTL lookups are stored as `WitnessId(n) * D`; use `D = EF::DIMENSION` for extension field.
    pub fn generate_preprocessed_columns<const D: usize>(
        &self,
    ) -> Result<PreprocessedColumns<F, D>, CircuitError> {
        let mut preprocessed = PreprocessedColumns::<F, D>::new();

        // Track which witnesses have been defined (first-occurrence = creator).
        // Const and Public define their outputs first. ALU ops define their output (forward)
        // or their `b` operand (backward/sub encoding where `out` was already defined).
        let mut defined = vec![false; self.witness_count as usize];

        // Private input witness IDs: these get their bus creator role from the first
        // ALU op that uses them, rather than from a Public table row.
        let private_input_wids: hashbrown::HashSet<u32> =
            self.private_input_rows.iter().map(|w| w.0).collect();

        // Hint output witness IDs: like private inputs, they are not emitted by any AIR
        // table and must have their bus creator role assigned by the first ALU op that
        // uses them.  Collect them in a pre-pass so the main loop can check membership.
        // Also stored in `preprocessed.hint_output_wids` for use by prover preprocessors
        // (e.g. `recompose_preprocess_impl`) that need to distinguish hint-derived witnesses
        // from already-defined ones (e.g. Poseidon2 outputs used as recompose coefficients).
        //
        // Important: when `assert_zero(x)` connects a hint output to `ExprId::ZERO`, the
        // hint output alias gets WitnessId(0), which is the Const-defined zero. We must
        // exclude such Const/Public-defined WitnessIds so they are not treated as creators
        // by the Recompose table (which would double-create them alongside ConstAir).
        let const_public_wids: hashbrown::HashSet<u32> = self
            .ops
            .iter()
            .filter_map(|op| match op {
                Op::Const { out, .. } => Some(out.0),
                Op::Public { out, .. } => Some(out.0),
                _ => None,
            })
            .collect();
        preprocessed.hint_output_wids = self
            .ops
            .iter()
            .filter_map(|op| {
                if let Op::Hint { outputs, .. } = op {
                    Some(outputs.iter().map(|w| w.0))
                } else {
                    None
                }
            })
            .flatten()
            .filter(|wid| !const_public_wids.contains(wid))
            .collect();
        // Clone for use in the loop below; the original stays in `preprocessed` for the prover.
        let hint_output_wids = preprocessed.hint_output_wids.clone();

        // Process each primitive operation.
        for op in &self.ops {
            match op {
                // Const: creates the output witness value. Store D-scaled out index.
                // No ext_reads increment: Const is a creator, not a reader.
                Op::Const { out, .. } => {
                    let idx = out.base_field_index::<F, D>();
                    preprocessed.primitive[PrimitiveOpType::Const as usize].push(idx);
                    let out_idx = out.0 as usize;
                    if out_idx >= defined.len() {
                        defined.resize(out_idx + 1, false);
                    }
                    defined[out_idx] = true;
                }
                // Public: creates the output witness value. Store D-scaled out index.
                // No ext_reads increment: Public is a creator, not a reader.
                Op::Public { out, .. } => {
                    let idx = out.base_field_index::<F, D>();
                    preprocessed.primitive[PrimitiveOpType::Public as usize].push(idx);
                    let out_idx = out.0 as usize;
                    if out_idx >= defined.len() {
                        defined.resize(out_idx + 1, false);
                    }
                    defined[out_idx] = true;
                }
                // Unified ALU operations with selectors for operation kind.
                //
                // Preprocessed per op (12 values, no multiplicities):
                // [sel_add_vs_mul, sel_bool, sel_muladd, sel_horner, a_idx, b_idx, c_idx, out_idx,
                //  a_state, b_is_creator, c_state, out_is_creator]
                //
                // a_state / c_state (3-valued):
                //   0 = skip (unconstrained, no bus contribution)
                //   1 = reader (defined by Const/Public/ALU/Poseidon2, bus receive)
                //   2 = creator (private input or hint output, first ALU use → bus send)
                //
                // b_is_creator / out_is_creator can both be set simultaneously when
                // b is a private input in the forward case.
                Op::Alu {
                    kind, a, b, c, out, ..
                } => {
                    let (sel_add_vs_mul, sel_bool, sel_muladd, sel_horner) = match kind {
                        AluOpKind::Add => (F::ONE, F::ZERO, F::ZERO, F::ZERO),
                        AluOpKind::Mul => (F::ZERO, F::ZERO, F::ZERO, F::ZERO),
                        AluOpKind::BoolCheck => (F::ZERO, F::ONE, F::ZERO, F::ZERO),
                        AluOpKind::MulAdd => (F::ZERO, F::ZERO, F::ONE, F::ZERO),
                        AluOpKind::HornerAcc => (F::ZERO, F::ZERO, F::ZERO, F::ONE),
                    };

                    let out_already_defined =
                        (out.0 as usize) < defined.len() && defined[out.0 as usize];
                    let b_already_defined = (b.0 as usize) < defined.len() && defined[b.0 as usize];

                    // 3-state for a and c:
                    //   0 = skip (not defined, not private/hint → unconstrained)
                    //   1 = reader (already defined by another table)
                    //   2 = creator (private input or hint output, first use → this row creates it)
                    //
                    // Guard: if the `out` slot is the creator for this op (!out_already_defined)
                    // and a (or c) aliases `out` (same WitnessId), skip the a/c creator role to
                    // avoid double-creation.  This happens in e.g. BoolCheck where a = c = out.
                    let a_defined = (a.0 as usize) < defined.len() && defined[a.0 as usize];
                    let a_aliased_by_out = !out_already_defined && a.0 == out.0;
                    let a_state: F = if a_defined {
                        F::ONE // reader
                    } else if (private_input_wids.contains(&a.0) || hint_output_wids.contains(&a.0))
                        && !a_aliased_by_out
                    {
                        F::TWO // creator (private input or hint output)
                    } else {
                        F::ZERO // skip
                    };

                    // `c` is absent for Add/Mul/BoolCheck. Do not use WitnessId(0) as a fake c:
                    // witness 0 may hold Const(0); treating it as c would set c_state = reader and
                    // duplicate WitnessChecks reads with b when assert_zero connects the sub result
                    // to ExprId::ZERO (b aliases witness 0).
                    let (c_wid, c_state) = c.as_ref().map_or((WitnessId(0), F::ZERO), |w| {
                        let c_defined = (w.0 as usize) < defined.len() && defined[w.0 as usize];
                        let c_aliased_by_out = !out_already_defined && w.0 == out.0;
                        let c_state = if c_defined {
                            F::ONE // reader
                        } else if (private_input_wids.contains(&w.0)
                            || hint_output_wids.contains(&w.0))
                            && !c_aliased_by_out
                        {
                            F::TWO // creator (private input or hint output)
                        } else {
                            F::ZERO // skip
                        };
                        (*w, c_state)
                    });

                    // b and out creator flags (now independent).
                    // Private inputs can be b-creators even in the forward case.
                    let b_is_private_creator =
                        !b_already_defined && private_input_wids.contains(&b.0);
                    let out_is_creator = F::from_bool(!out_already_defined);
                    let b_is_creator = F::from_bool(
                        b_is_private_creator || out_already_defined && !b_already_defined,
                    );

                    preprocessed.primitive[PrimitiveOpType::Alu as usize].extend([
                        sel_add_vs_mul,
                        sel_bool,
                        sel_muladd,
                        sel_horner,
                        a.base_field_index::<F, D>(),
                        b.base_field_index::<F, D>(),
                        c_wid.base_field_index::<F, D>(),
                        out.base_field_index::<F, D>(),
                        a_state,
                        b_is_creator,
                        c_state,
                        out_is_creator,
                    ]);

                    // Build readers list — creators are excluded.
                    let mut readers = Vec::new();

                    // b: reader unless it's a creator (private or backward)
                    if b_is_creator == F::ZERO {
                        readers.push(*b);
                    }
                    // out: reader unless it's a creator
                    if out_is_creator == F::ZERO {
                        readers.push(*out);
                    }
                    if a_state == F::ONE {
                        readers.push(*a);
                    }
                    if c_state == F::ONE {
                        readers.push(c_wid);
                    }
                    preprocessed.increment_ext_reads(&readers);

                    // Mark new creators as defined.
                    if out_is_creator == F::ONE {
                        let out_idx = out.0 as usize;
                        if out_idx >= defined.len() {
                            defined.resize(out_idx + 1, false);
                        }
                        defined[out_idx] = true;
                    }
                    if b_is_creator == F::ONE {
                        let b_idx = b.0 as usize;
                        if b_idx >= defined.len() {
                            defined.resize(b_idx + 1, false);
                        }
                        defined[b_idx] = true;
                    }
                    if a_state == F::TWO {
                        let a_idx = a.0 as usize;
                        if a_idx >= defined.len() {
                            defined.resize(a_idx + 1, false);
                        }
                        defined[a_idx] = true;
                    }
                    if c_state == F::TWO {
                        let c_idx = c_wid.0 as usize;
                        if c_idx >= defined.len() {
                            defined.resize(c_idx + 1, false);
                        }
                        defined[c_idx] = true;
                    }
                }
                Op::NonPrimitiveOpWithExecutor {
                    executor,
                    inputs,
                    outputs,
                    ..
                } => {
                    executor.preprocess(inputs, outputs, &mut preprocessed)?;

                    // Track duplicate non-primitive outputs: first occurrence is a creator,
                    // subsequent occurrences are treated as readers on WitnessChecks.
                    let op_type = executor.op_type();
                    let n_exposed = executor.num_exposed_outputs().unwrap_or(outputs.len());
                    for out_limb in outputs.iter().take(n_exposed) {
                        for wid in out_limb {
                            let wid_idx = wid.0 as usize;
                            if wid_idx < defined.len() && defined[wid_idx] {
                                let dup = preprocessed
                                    .dup_npo_outputs
                                    .entry(op_type.clone())
                                    .or_default();
                                if wid_idx >= dup.len() {
                                    dup.resize(wid_idx + 1, false);
                                }
                                dup[wid_idx] = true;
                                preprocessed.increment_ext_reads(&[*wid]);
                            } else {
                                if wid_idx >= defined.len() {
                                    defined.resize(wid_idx + 1, false);
                                }
                                defined[wid_idx] = true;
                            }
                        }
                    }
                }
                Op::Hint { .. } => {
                    // Hints do not participate in preprocessed columns or table-backed ops.
                }
            }
        }

        // Ensure ext_reads covers at least all witnesses.
        let size = self.witness_count as usize;
        if preprocessed.ext_reads.len() < size {
            preprocessed.ext_reads.resize(size, 0);
        }

        // Safety: every private input must have been claimed as a creator by some ALU op,
        // or the WitnessChecks bus would be unbalanced.
        for &wid in &self.private_input_rows {
            debug_assert!(
                defined[wid.0 as usize],
                "Private input WitnessId({}) was never used as an ALU operand — \
                 cannot assign a bus creator. All private inputs must appear in \
                 at least one ALU op.",
                wid.0
            );
        }

        Ok(preprocessed)
    }
}

impl<F: Field> Circuit<F> {
    /// Create a circuit runner for execution and trace generation.
    pub fn runner(&self) -> CircuitRunner<'_, F> {
        CircuitRunner::new(self)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use hashbrown::HashMap;
    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};
    use strum::EnumCount;

    use super::*;
    use crate::ops::PrimitiveOpType;
    use crate::types::WitnessId;

    type F = BabyBear;

    fn make_circuit(ops: Vec<Op<F>>) -> Circuit<F> {
        let mut circuit = Circuit::new(0, HashMap::new());
        circuit.ops = ops;
        circuit
    }

    #[test]
    fn test_empty_circuit() {
        let mut circuit: Circuit<F> = make_circuit(vec![]);
        circuit.witness_count = 1;
        let result = circuit.generate_preprocessed_columns::<1>().unwrap();

        assert_eq!(
            result,
            PreprocessedColumns {
                primitive: vec![vec![]; PrimitiveOpType::COUNT],
                non_primitive: HashMap::new(),
                ext_reads: vec![0],
                dup_npo_outputs: HashMap::new(),
                hint_output_wids: hashbrown::HashSet::new(),
            }
        );
    }

    #[test]
    fn test_mixed_operations() {
        // Test covering various operation types and behaviors:
        // - Each operation type populates its correct column
        // - ext_reads tracks ALU input reads correctly
        // - Column data preserves operation order
        let ops = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::from_u64(100),
            },
            Op::Public {
                out: WitnessId(1),
                public_pos: 0,
            },
            Op::Const {
                out: WitnessId(2),
                val: F::from_u64(200),
            },
            Op::add(WitnessId(0), WitnessId(1), WitnessId(3)),
            Op::add(WitnessId(3), WitnessId(2), WitnessId(4)),
            Op::mul(WitnessId(4), WitnessId(2), WitnessId(5)),
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns::<1>().unwrap();

        let f = F::from_u32;
        assert_eq!(
            result,
            PreprocessedColumns {
                primitive: vec![
                    // Const: D-scaled output indices
                    vec![F::ZERO, f(2)],
                    // Public: D-scaled output index
                    vec![f(1)],
                    // ALU: [sel_add_vs_mul, sel_bool, sel_muladd, sel_horner, a, b, c, out,
                    //       a_state, b_is_creator, c_state, out_is_creator] per op
                    vec![
                        // add(0, 1, 3): forward, a=0(defined), no c limb
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        f(1),
                        F::ZERO,
                        f(3),
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        F::ONE,
                        // add(3, 2, 4): forward, a=3(defined), no c limb
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        f(3),
                        f(2),
                        F::ZERO,
                        f(4),
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        F::ONE,
                        // mul(4, 2, 5): forward, a=4(defined), no c limb
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        f(4),
                        f(2),
                        F::ZERO,
                        f(5),
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        F::ONE,
                    ],
                ],
                non_primitive: HashMap::new(),
                // ext_reads: op1 reads a=0,b=1; op2 reads a=3,b=2; op3 reads a=4,b=2
                ext_reads: vec![1, 1, 2, 1, 1],
                dup_npo_outputs: HashMap::new(),
                hint_output_wids: hashbrown::HashSet::new(),
            }
        );
    }

    #[test]
    fn test_input_indices_contribute_to_ext_reads() {
        // Ensures input indices are tracked for ext_reads
        // add(0, 15, 5): a=0 (undefined, not private/hint → a_state=skip), no c limb
        // Only b=15 is counted (always a reader in forward case).
        let ops = vec![Op::add(
            WitnessId(0),
            WitnessId(15), // Highest index is an input, not output
            WitnessId(5),
        )];

        let mut circuit = make_circuit(ops);
        circuit.witness_count = 16;
        let result = circuit.generate_preprocessed_columns::<1>().unwrap();

        let f = F::from_u32;
        assert_eq!(
            result,
            PreprocessedColumns {
                primitive: vec![
                    vec![],
                    vec![],
                    // add(0, 15, 5): forward, a undefined, c undefined
                    vec![
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        f(15),
                        F::ZERO,
                        f(5),
                        F::ZERO,
                        F::ZERO,
                        F::ZERO,
                        F::ONE,
                    ],
                ],
                non_primitive: HashMap::new(),
                //                    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                ext_reads: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                dup_npo_outputs: HashMap::new(),
                hint_output_wids: hashbrown::HashSet::new(),
            }
        );
    }

    #[test]
    fn test_muladd_operation() {
        // Test the MulAdd operation preprocessed format: 3*5+7=22
        let ops = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::from_u64(3),
            },
            Op::Const {
                out: WitnessId(1),
                val: F::from_u64(5),
            },
            Op::Const {
                out: WitnessId(2),
                val: F::from_u64(7),
            },
            Op::mul_add(WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3)),
        ];

        let circuit = make_circuit(ops);
        let result = circuit.generate_preprocessed_columns::<1>().unwrap();

        let f = F::from_u32;
        assert_eq!(
            result,
            PreprocessedColumns {
                primitive: vec![
                    // Const: D-scaled output indices
                    vec![F::ZERO, f(1), f(2)],
                    // Public: none
                    vec![],
                    // ALU: mul_add(0, 1, 2, 3) forward, a=0(defined), c=2(defined)
                    vec![
                        F::ZERO,
                        F::ZERO,
                        F::ONE,
                        F::ZERO,
                        F::ZERO,
                        f(1),
                        f(2),
                        f(3),
                        F::ONE,
                        F::ZERO,
                        F::ONE,
                        F::ONE,
                    ],
                ],
                non_primitive: HashMap::new(),
                // ext_reads: 0(a)=1, 1(b)=1, 2(c)=1
                ext_reads: vec![1, 1, 1],
                dup_npo_outputs: HashMap::new(),
                hint_output_wids: hashbrown::HashSet::new(),
            }
        );
    }
}
