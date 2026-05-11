//! Poseidon2 permutation executor.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use p3_field::Field;

use crate::CircuitError;
use crate::ops::poseidon2_perm::config::{
    Poseidon2Config, Poseidon2PermConfigData, Poseidon2PermExec,
};
use crate::ops::poseidon2_perm::state::{Poseidon2ExecutionState, Poseidon2PermPrivateData};
use crate::ops::poseidon2_perm::trace::Poseidon2CircuitRow;
use crate::ops::{ExecutionContext, NonPrimitiveExecutor, NpoTypeId, PreprocessedWriter};
use crate::types::WitnessId;

/// Runtime executor for a single Poseidon2 permutation row.
///
/// Handles both D=1 (base field) and D>1 (extension field) modes,
/// as well as standard hashing and Merkle-path verification.
#[derive(Debug, Clone)]
pub(crate) struct Poseidon2PermExecutor {
    /// Operation type identifier for config/state lookups.
    op_type: NpoTypeId,
    /// Poseidon2 parameters (width, rate, extension degree, etc.).
    config: Poseidon2Config,
    /// When true, this row starts a fresh chain instead of continuing
    /// from the previous permutation output.
    pub(crate) new_start: bool,
    /// When true, the executor arranges inputs for Merkle-path verification
    /// and conditionally swaps left/right halves based on the direction bit.
    pub(crate) merkle_path: bool,
}

impl Poseidon2PermExecutor {
    pub const fn new(
        op_type: NpoTypeId,
        config: Poseidon2Config,
        new_start: bool,
        merkle_path: bool,
    ) -> Self {
        Self {
            op_type,
            config,
            new_start,
            merkle_path,
        }
    }

    #[inline]
    const fn compact_d1_preprocessed_layout(&self) -> bool {
        self.config.d() == 1 && self.config.width_ext() == 16 && self.config.rate_ext() == 8
    }

    #[inline]
    const fn limb_ctl_enabled(slot: &[WitnessId]) -> bool {
        !slot.is_empty()
    }

    /// Build the initial permutation state vector.
    ///
    /// - New chain: returns a zero vector of `width_ext` elements.
    /// - Continuation: copies previous output into the state.
    ///   In Merkle mode only the rate portion is carried forward;
    ///   in normal mode the full width is carried.
    ///
    /// # Errors
    ///
    /// Returns `Poseidon2ChainMissingPreviousState` when chaining is
    /// requested but no prior output exists.
    fn init_chain_state<F: Field>(
        &self,
        last_output: Option<&[F]>,
        ctx: &ExecutionContext<'_, F>,
    ) -> Result<Vec<F>, CircuitError> {
        let width_ext = self.config.width_ext();
        let mut resolved = F::zero_vec(width_ext);

        if self.new_start {
            return Ok(resolved);
        }

        let prev = last_output.ok_or_else(|| CircuitError::Poseidon2ChainMissingPreviousState {
            operation_index: ctx.operation_id(),
        })?;

        if self.merkle_path {
            let n = self.config.rate_ext().min(prev.len());
            resolved[..n].copy_from_slice(&prev[..n]);
        } else {
            let n = width_ext.min(prev.len());
            resolved[..n].copy_from_slice(&prev[..n]);
        }

        Ok(resolved)
    }

    /// Copy sibling hash limbs into the capacity portion of the state.
    ///
    /// Only active when both Merkle mode is enabled and private data is provided.
    /// Writes up to `capacity_ext` elements starting at index `rate_ext`.
    fn fill_sibling_data<F: Field>(&self, state: &mut [F], private: Option<&[F]>) {
        if let Some(private) = private
            && self.merkle_path
        {
            let rate_ext = self.config.rate_ext();
            let n = private.len().min(self.config.capacity_ext());
            state[rate_ext..rate_ext + n].copy_from_slice(&private[..n]);
        }
    }

    /// Overwrite state elements with witness values from CTL-exposed input slots.
    ///
    /// Only slots containing exactly one witness identifier are read;
    /// empty slots leave the state element unchanged.
    fn apply_witness_values<F: Field>(
        &self,
        state: &mut [F],
        inputs: &[Vec<WitnessId>],
        ctx: &ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError> {
        for (slot, inp) in state[..self.config.width_ext()].iter_mut().zip(inputs) {
            if let [wid] = inp.as_slice() {
                *slot = ctx.get_witness(*wid)?;
            }
        }
        Ok(())
    }

    /// Swap the two rate halves of the state in-place.
    ///
    /// Active only when both Merkle mode is enabled and the direction bit is set.
    /// This places the computed hash on the correct side (left vs right) before
    /// the permutation.
    fn apply_merkle_swap<F: Field>(&self, state: &mut [F], mmcs_bit: bool) {
        if self.merkle_path && mmcs_bit {
            let rate_ext = self.config.rate_ext();
            for i in 0..rate_ext {
                state.swap(i, rate_ext + i);
            }
        }
    }

    /// Extract Merkle sibling data from the operation's private payload.
    ///
    /// Returns `None` when no private data is attached or the payload is
    /// not the expected sibling type.
    ///
    /// # Errors
    ///
    /// Returns an error if sibling data is present but the executor is
    /// not in Merkle mode, which indicates a caller configuration mistake.
    fn resolve_private_data<'a, F: Field + 'static>(
        &self,
        ctx: &'a ExecutionContext<'_, F>,
    ) -> Result<Option<&'a [F]>, CircuitError> {
        let Ok(private_data) = ctx.get_private_data() else {
            return Ok(None);
        };
        let Some(data) = private_data.downcast_ref::<Poseidon2PermPrivateData<F>>() else {
            return Ok(None);
        };
        if !self.merkle_path {
            return Err(CircuitError::IncorrectNonPrimitiveOpPrivateData {
                op: self.op_type.clone(),
                operation_index: ctx.operation_id(),
                expected: "no private data (only Merkle mode accepts private data)".to_string(),
                got: "private data provided for non-Merkle operation".to_string(),
            });
        }
        Ok(Some(data.sibling.as_slice()))
    }

    /// Read the MMCS direction bit from the witness table.
    ///
    /// The value must be boolean (0 or 1).
    /// When the slot is empty and Merkle mode is off, defaults to false.
    ///
    /// # Errors
    ///
    /// - Non-boolean witness value.
    /// - Missing direction bit when Merkle mode requires it.
    fn resolve_mmcs_bit<F: Field>(
        &self,
        inputs: &[Vec<WitnessId>],
        ctx: &ExecutionContext<'_, F>,
    ) -> Result<bool, CircuitError> {
        let width_ext = self.config.width_ext();
        if let Some(&wid) = inputs[width_ext + 1].first() {
            let val = ctx.get_witness(wid)?;
            match val {
                v if v == F::ZERO => Ok(false),
                v if v == F::ONE => Ok(true),
                v => Err(CircuitError::IncorrectNonPrimitiveOpPrivateData {
                    op: self.op_type.clone(),
                    operation_index: ctx.operation_id(),
                    expected: "boolean mmcs_bit (0 or 1)".into(),
                    got: format!("{v:?}"),
                }),
            }
        } else if self.merkle_path {
            Err(CircuitError::IncorrectNonPrimitiveOpPrivateData {
                op: self.op_type.clone(),
                operation_index: ctx.operation_id(),
                expected: "mmcs_bit must be provided when merkle_path=true".into(),
                got: "missing mmcs_bit".into(),
            })
        } else {
            Ok(false)
        }
    }

    /// Return the previous permutation output from chain state, if any.
    ///
    /// Selects the Merkle or normal output depending on this executor's mode.
    fn get_chain_output<'a, F: Field + 'static>(
        &self,
        ctx: &'a ExecutionContext<'_, F>,
    ) -> Option<&'a Vec<F>> {
        ctx.get_op_state::<Poseidon2ExecutionState<F>>(&self.op_type)
            .and_then(|s| {
                if self.merkle_path {
                    s.last_output_merkle.as_ref()
                } else {
                    s.last_output_normal.as_ref()
                }
            })
    }

    /// Construct the circuit trace row for D>1 (extension field) mode.
    ///
    /// Scans input/output slots to determine which limbs are CTL-exposed
    /// and records their witness indices.
    /// Also resolves the MMCS index accumulator if present.
    fn build_trace_row<F: Field>(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        mmcs_bit: bool,
        input_values: Vec<F>,
        ctx: &ExecutionContext<'_, F>,
    ) -> Result<Poseidon2CircuitRow<F>, CircuitError> {
        let width_ext = self.config.width_ext();
        let rate_ext = self.config.rate_ext();

        let mut in_ctl = vec![false; width_ext];
        let mut input_indices = vec![0u32; width_ext];
        for (i, inp) in inputs[..width_ext].iter().enumerate() {
            if let Some(&wid) = inp.first() {
                in_ctl[i] = true;
                input_indices[i] = wid.0;
            }
        }

        let mut out_ctl = vec![false; rate_ext];
        let mut output_indices = vec![0u32; rate_ext];
        for (i, out_slot) in outputs.iter().take(rate_ext).enumerate() {
            if let Some(&wid) = out_slot.first() {
                out_ctl[i] = true;
                output_indices[i] = wid.0;
            }
        }

        let (mmcs_index_sum, mmcs_index_sum_idx, mmcs_ctl_enabled) = if inputs[width_ext].len() == 1
        {
            let wid = inputs[width_ext][0];
            let val = ctx.get_witness(wid)?;
            (val, wid.0, true)
        } else {
            (F::ZERO, 0, false)
        };

        debug_assert_eq!(
            input_values.len(),
            width_ext,
            "Execution row must have width_ext input limbs"
        );

        Ok(Poseidon2CircuitRow {
            new_start: self.new_start,
            merkle_path: self.merkle_path,
            mmcs_bit,
            mmcs_index_sum,
            input_values,
            in_ctl,
            input_indices,
            out_ctl,
            output_indices,
            mmcs_index_sum_idx,
            mmcs_ctl_enabled,
        })
    }

    /// Construct the circuit trace row for D=1 (base field) mode.
    ///
    /// One CTL flag and witness index per physical input slot (`WIDTH` = 16) and per
    /// rate output slot (`RATE` = 8), matching the preprocessed column layout for
    /// width-16 / rate-8 Poseidon2 AIR.
    fn build_base_trace_row<F: Field>(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        input_values: &[F],
    ) -> Poseidon2CircuitRow<F> {
        let width = self.config.width();
        let rate_ext = self.config.rate_ext();
        let mut in_ctl = vec![false; width];
        let mut input_indices = vec![0u32; width];
        for i in 0..width {
            if let Some(inp) = inputs.get(i)
                && let [wid] = inp.as_slice()
            {
                in_ctl[i] = true;
                input_indices[i] = wid.0;
            }
        }
        if self.compact_d1_preprocessed_layout() {
            for c in in_ctl.iter_mut().take(width).skip(rate_ext) {
                *c = false;
            }
        }

        let mut out_ctl = vec![false; rate_ext];
        let mut output_indices = vec![0u32; rate_ext];
        for i in 0..rate_ext {
            if let Some(out_slot) = outputs.get(i)
                && let [wid] = out_slot.as_slice()
            {
                out_ctl[i] = true;
                output_indices[i] = wid.0;
            }
        }

        Poseidon2CircuitRow {
            new_start: self.new_start,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: F::ZERO,
            input_values: input_values.to_vec(),
            in_ctl,
            input_indices,
            out_ctl,
            output_indices,
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        }
    }

    /// Store the permutation output in chain state and append the trace row.
    ///
    /// Writes to the Merkle or normal output slot depending on this executor's mode.
    fn update_chain_state<F: Field + 'static>(
        &self,
        ctx: &mut ExecutionContext<'_, F>,
        output: Vec<F>,
        row: Poseidon2CircuitRow<F>,
    ) {
        let op_id = ctx.operation_id();
        let state = ctx.get_op_state_mut::<Poseidon2ExecutionState<F>>(&self.op_type);
        let slot = if self.merkle_path {
            &mut state.last_output_merkle
        } else {
            &mut state.last_output_normal
        };
        let kind = if self.merkle_path { "merkle" } else { "normal" };
        tracing::trace!(
            "Poseidon2 op {op_id:?}: updating last_output_{kind} from {prev:?} to {output:?}",
            prev = slot.as_deref(),
        );
        *slot = Some(output);
        state.rows.push(row);
    }

    /// Validate the input layout for D>1 (extension field) mode.
    ///
    /// Expects exactly `width_ext + 2` input vectors:
    /// - `width_ext` limb slots, each with 0 or 1 witness.
    /// - 1 MMCS index accumulator slot (0 or 1 element).
    /// - 1 MMCS direction bit slot (0 or 1 element).
    fn validate_ext_inputs(&self, inputs: &[Vec<WitnessId>]) -> Result<(), CircuitError> {
        let width_ext = self.config.width_ext();
        let expected_inputs = width_ext + 2;
        if inputs.len() != expected_inputs {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: format!("{expected_inputs} input vectors"),
                got: inputs.len(),
            });
        }
        for limb_inputs in inputs[..width_ext].iter() {
            if limb_inputs.len() > 1 {
                return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                    op: self.op_type.clone(),
                    expected: "0 or 1 witness per input limb (extension-only)".to_string(),
                    got: limb_inputs.len(),
                });
            }
        }
        if inputs[width_ext].len() > 1 {
            return Err(CircuitError::IncorrectNonPrimitiveOpPrivateDataSize {
                op: self.op_type.clone(),
                expected: "0 or 1 element for mmcs_index_sum".to_string(),
                got: inputs[width_ext].len(),
            });
        }
        if inputs[width_ext + 1].len() > 1 {
            return Err(CircuitError::IncorrectNonPrimitiveOpPrivateDataSize {
                op: self.op_type.clone(),
                expected: "0 or 1 element for mmcs_bit".to_string(),
                got: inputs[width_ext + 1].len(),
            });
        }
        Ok(())
    }

    /// Validate the output layout for D>1 (extension field) mode.
    ///
    /// Accepts either `rate_ext` outputs (rate-only) or `width_ext` outputs (full state).
    fn validate_ext_outputs(&self, outputs: &[Vec<WitnessId>]) -> Result<(), CircuitError> {
        let rate_ext = self.config.rate_ext();
        let width_ext = self.config.width_ext();
        if outputs.len() != rate_ext && outputs.len() != width_ext {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: format!("{rate_ext} or {width_ext} output vectors"),
                got: outputs.len(),
            });
        }
        Ok(())
    }

    /// Write permutation output values to witness slots.
    ///
    /// Empty slots are skipped.
    /// Slots with exactly one witness get the corresponding output value.
    /// Slots with more than one witness are rejected.
    fn write_outputs<F: Field>(
        &self,
        outputs: &[Vec<WitnessId>],
        output_values: &[F],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError> {
        for (out_slot, &val) in outputs.iter().zip(output_values) {
            match out_slot.as_slice() {
                [] => {}
                [wid] => ctx.set_witness(*wid, val)?,
                _ => {
                    return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                        op: self.op_type.clone(),
                        expected: "0 or 1 witness per output limb".to_string(),
                        got: out_slot.len(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Execute the D=1 (base field) permutation.
    ///
    /// Validates a 16-input / 8-or-16-output layout, resolves witness values,
    /// runs the permutation closure, records a trace row, and writes outputs.
    fn execute_base<F: Field + Send + Sync + 'static>(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
        exec: &dyn Fn(&[F]) -> Vec<F>,
    ) -> Result<(), CircuitError> {
        let width = self.config.width();
        let width_ext = self.config.width_ext();
        let limbs: &[Vec<WitnessId>] = match inputs.len() {
            n if n == width => inputs,
            n if n == width_ext + 2 => {
                for (i, slot) in inputs[width_ext..].iter().enumerate() {
                    if !slot.is_empty() {
                        return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                            op: self.op_type.clone(),
                            expected: format!(
                                "empty mmcs slots for D=1 non-Merkle (tail slot {i})"
                            ),
                            got: slot.len(),
                        });
                    }
                }
                &inputs[..width]
            }
            got => {
                return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                    op: self.op_type.clone(),
                    expected: format!("{width} or {} input vectors for D=1 mode", width_ext + 2),
                    got,
                });
            }
        };

        for (i, inp) in limbs.iter().enumerate() {
            if inp.len() > 1 {
                return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                    op: self.op_type.clone(),
                    expected: format!("0 or 1 witness per input element {}", i),
                    got: inp.len(),
                });
            }
        }
        if outputs.len() != 8 && outputs.len() != 16 {
            return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                op: self.op_type.clone(),
                expected: "8 or 16 output vectors for D=1 mode".to_string(),
                got: outputs.len(),
            });
        }

        // Initialize from previous chain output (or zeros for new_start).
        // This matches native PaddingFreeSponge overwrite mode, which preserves the full
        // state (including capacity) between chunks of a multi-chunk sponge absorption.
        let chain_output = self.get_chain_output(ctx);
        let mut resolved_inputs = self.init_chain_state(chain_output.map(|v| v.as_slice()), ctx)?;
        for (slot, inp) in resolved_inputs.iter_mut().zip(limbs) {
            if let [wid] = inp.as_slice() {
                *slot = ctx.get_witness(*wid)?;
            }
        }

        let output = exec(&resolved_inputs);
        let row = self.build_base_trace_row(limbs, outputs, &resolved_inputs);

        self.write_outputs(outputs, &output, ctx)?;
        self.update_chain_state(ctx, output, row);

        Ok(())
    }

    /// Emit preprocessed columns for input limbs.
    ///
    /// For each limb, registers:
    /// - A witness read index (or zero if the limb is chain-inherited).
    /// - A CTL-enabled flag.
    /// - A normal-chain selector (1 when the limb inherits from the previous normal output).
    /// - A Merkle-chain selector (1 when the limb inherits from the previous Merkle output).
    fn preprocess_inputs<F: Field>(
        &self,
        inputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        let width_ext = self.config.width_ext();
        let rate_ext = self.config.rate_ext();

        if self.compact_d1_preprocessed_layout() {
            // Sponge rows: capacity is never witness-fed (AIR zero-assert on new_start; else chain).
            // Merkle rows: siblings live in capacity slots and use witness indices without input CTL.
            if !self.merkle_path {
                for input in inputs.iter().skip(rate_ext).take(width_ext - rate_ext) {
                    if Self::limb_ctl_enabled(input) {
                        return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                            op: self.op_type.clone(),
                            expected:
                                "capacity input slots must be empty on compact D=1 sponge rows"
                                    .into(),
                            got: input.len(),
                        });
                    }
                }
            }

            let cap_chain_enable = !self.new_start;
            let mut hdr = Vec::with_capacity(rate_ext + 2 + rate_ext + rate_ext);
            for inp in inputs.iter().take(rate_ext) {
                hdr.push(F::from_bool(Self::limb_ctl_enabled(inp)));
            }
            hdr.push(F::ZERO);
            hdr.push(F::from_bool(cap_chain_enable));
            for inp in inputs.iter().take(rate_ext) {
                let ctl = Self::limb_ctl_enabled(inp);
                hdr.push(F::from_bool(!self.new_start && !self.merkle_path && !ctl));
            }
            for inp in inputs.iter().take(rate_ext) {
                let ctl = Self::limb_ctl_enabled(inp);
                hdr.push(F::from_bool(!self.new_start && self.merkle_path && !ctl));
            }
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &hdr);

            for inp in inputs[0..width_ext].iter() {
                if inp.is_empty() {
                    preprocessed
                        .register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
                } else if let [wid] = inp.as_slice() {
                    if self.merkle_path {
                        preprocessed.register_non_primitive_preprocessed_no_read(
                            &self.op_type,
                            &[preprocessed.witness_index_as_field(*wid)],
                        );
                    } else {
                        preprocessed.register_non_primitive_witness_reads(&self.op_type, inp)?;
                    }
                } else {
                    return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                        op: self.op_type.clone(),
                        expected: "0 or 1 witness per input limb".into(),
                        got: inp.len(),
                    });
                }
            }
            return Ok(());
        }

        for inp in inputs[0..width_ext].iter() {
            if inp.is_empty() {
                preprocessed.register_non_primitive_preprocessed_no_read(
                    &self.op_type,
                    &[F::ZERO, F::ZERO],
                );
            } else if self.merkle_path {
                preprocessed.register_non_primitive_preprocessed_no_read(
                    &self.op_type,
                    &[preprocessed.witness_index_as_field(inp[0])],
                );
                preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
            } else {
                preprocessed.register_non_primitive_witness_reads(&self.op_type, inp)?;
                preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
            }
            let normal_chain_sel =
                F::from_bool(!self.new_start && !self.merkle_path && inp.is_empty());

            preprocessed
                .register_non_primitive_preprocessed_no_read(&self.op_type, &[normal_chain_sel]);

            let merkle_chain_sel =
                F::from_bool(!self.new_start && self.merkle_path && inp.is_empty());
            preprocessed
                .register_non_primitive_preprocessed_no_read(&self.op_type, &[merkle_chain_sel]);
        }
        Ok(())
    }

    /// Emit preprocessed columns for rate-portion output limbs.
    ///
    /// Each exposed output gets a witness output index and a CTL-enabled flag.
    fn preprocess_outputs<F: Field>(
        &self,
        outputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        let rate_ext = self.config.rate_ext();

        if self.compact_d1_preprocessed_layout() {
            for out in outputs.iter().take(rate_ext) {
                if out.is_empty() {
                    preprocessed
                        .register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
                } else if let [_] = out.as_slice() {
                    preprocessed.register_non_primitive_output_index(&self.op_type, out);
                } else {
                    return Err(CircuitError::NonPrimitiveOpLayoutMismatch {
                        op: self.op_type.clone(),
                        expected: "0 or 1 witness per output limb".into(),
                        got: out.len(),
                    });
                }
            }

            for out in outputs.iter().take(rate_ext) {
                preprocessed.register_non_primitive_preprocessed_no_read(
                    &self.op_type,
                    &[F::from_bool(Self::limb_ctl_enabled(out))],
                );
            }
            return Ok(());
        }

        for out in outputs.iter().take(rate_ext) {
            if out.is_empty() {
                preprocessed.register_non_primitive_preprocessed_no_read(
                    &self.op_type,
                    &[F::ZERO, F::ZERO],
                );
            } else {
                preprocessed.register_non_primitive_output_index(&self.op_type, out);
                preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ONE]);
            }
        }
        Ok(())
    }

    /// Emit preprocessed columns for control flags.
    ///
    /// Registers the MMCS index accumulator witness index, the Merkle CTL flag,
    /// and the `new_start` / `merkle_path` boolean selectors.
    fn preprocess_flags<F: Field>(
        &self,
        inputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        let width_ext = self.config.width_ext();
        // D=1 non-Merkle: keep the compact flag layout expected by the D=1 Poseidon2 AIR (covers
        // both `add_poseidon2_perm_base` and `add_poseidon2_perm` empty MMCS slots).
        if self.config.d() == 1 && !self.merkle_path {
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
            let new_start_val = if self.new_start { F::ONE } else { F::ZERO };
            let merkle_path_val = if self.merkle_path { F::ONE } else { F::ZERO };
            preprocessed.register_non_primitive_preprocessed_no_read(
                &self.op_type,
                &[new_start_val, merkle_path_val],
            );
            return Ok(());
        }

        if inputs[width_ext].is_empty() {
            preprocessed.register_non_primitive_preprocessed_no_read(&self.op_type, &[F::ZERO]);
        } else {
            preprocessed.register_non_primitive_preprocessed_no_read(
                &self.op_type,
                &[preprocessed.witness_index_as_field(inputs[width_ext][0])],
            );
        }

        let mmcs_ctl_enabled = !inputs[width_ext].is_empty();
        let mmcs_merkle_flag = F::from_bool(mmcs_ctl_enabled && self.merkle_path);
        preprocessed
            .register_non_primitive_preprocessed_no_read(&self.op_type, &[mmcs_merkle_flag]);

        let new_start_val = F::from_bool(self.new_start);
        let merkle_path_val = F::from_bool(self.merkle_path);
        preprocessed.register_non_primitive_preprocessed_no_read(
            &self.op_type,
            &[new_start_val, merkle_path_val],
        );

        Ok(())
    }
}

impl<F: Field + Send + Sync + 'static> NonPrimitiveExecutor<F> for Poseidon2PermExecutor {
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError> {
        let exec: Poseidon2PermExec<F> = ctx
            .get_config(&self.op_type)?
            .downcast_ref::<Poseidon2PermConfigData<F>>()
            .map(|cfg| cfg.exec.clone())
            .ok_or_else(|| CircuitError::InvalidNonPrimitiveOpConfiguration {
                op: self.op_type.clone(),
            })?;

        // D=1 non-Merkle: base trace row (`width` limbs, or `width_ext+2` with empty MMCS tail).
        if self.config.d() == 1 && !self.merkle_path {
            return self.execute_base(inputs, outputs, ctx, exec.as_ref());
        }

        // Validate extension-field input/output shapes before touching any state.
        self.validate_ext_inputs(inputs)?;
        self.validate_ext_outputs(outputs)?;

        // Gather all auxiliary data needed to assemble the pre-permutation state.
        let private_inputs = self.resolve_private_data(ctx)?;
        let mmcs_bit = self.resolve_mmcs_bit(inputs, ctx)?;
        let chain_output = self.get_chain_output(ctx);

        // Build the permutation input state:
        // 1. Start from zeros (new chain) or the previous output (continuation).
        let mut state = self.init_chain_state(chain_output.map(|v| v.as_slice()), ctx)?;
        // 2. In Merkle mode, place sibling limbs in the capacity portion.
        self.fill_sibling_data(&mut state, private_inputs);
        // 3. Overwrite with any CTL-exposed witness values.
        self.apply_witness_values(&mut state, inputs, ctx)?;
        // 4. Conditionally swap rate halves for the Merkle direction bit.
        self.apply_merkle_swap(&mut state, mmcs_bit);

        // Run the permutation and record the result.
        let output = exec(&state);

        let row = self.build_trace_row(inputs, outputs, mmcs_bit, state, ctx)?;
        self.write_outputs(outputs, &output, ctx)?;
        self.update_chain_state(ctx, output, row);

        Ok(())
    }

    fn op_type(&self) -> &NpoTypeId {
        &self.op_type
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }

    fn num_exposed_outputs(&self) -> Option<usize> {
        Some(self.config.rate_ext())
    }

    fn preprocess(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        self.preprocess_inputs(inputs, preprocessed)?;
        self.preprocess_outputs(outputs, preprocessed)?;
        self.preprocess_flags(inputs, preprocessed)?;

        Ok(())
    }

    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<F>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use hashbrown::HashMap;
    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::ops::npo::NpoPrivateData;
    use crate::types::NonPrimitiveOpId;

    type F = BabyBear;

    const CONFIG_D4_W16: Poseidon2Config = Poseidon2Config::BabyBearD4Width16;
    const CONFIG_D1_W16: Poseidon2Config = Poseidon2Config::BabyBearD1Width16;

    fn op_type_d4() -> NpoTypeId {
        NpoTypeId::poseidon2_perm(CONFIG_D4_W16)
    }

    fn executor(
        config: Poseidon2Config,
        new_start: bool,
        merkle_path: bool,
    ) -> Poseidon2PermExecutor {
        Poseidon2PermExecutor::new(
            NpoTypeId::poseidon2_perm(config),
            config,
            new_start,
            merkle_path,
        )
    }

    fn make_ctx<'a>(
        witness: &'a mut [Option<F>],
        private_data: &'a [Option<NpoPrivateData>],
        configs: &'a HashMap<NpoTypeId, crate::ops::NpoConfig>,
        op_states: &'a mut BTreeMap<NpoTypeId, Box<dyn crate::ops::OpExecutionState>>,
    ) -> ExecutionContext<'a, F> {
        ExecutionContext::new(
            witness,
            private_data,
            configs,
            NonPrimitiveOpId(0),
            op_states,
        )
    }

    #[test]
    fn init_chain_state_new_start_returns_zeros() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let mut witness = vec![];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let state = exec.init_chain_state(Some(&[F::ONE; 4]), &ctx).unwrap();
        assert_eq!(state, F::zero_vec(CONFIG_D4_W16.width_ext()));
    }

    #[test]
    fn init_chain_state_chain_normal_copies_full_width() {
        let exec = executor(CONFIG_D4_W16, false, false);
        let width_ext = CONFIG_D4_W16.width_ext(); // 4
        let prev: Vec<F> = (1..=width_ext).map(|i| F::from_u64(i as u64)).collect();
        let mut witness = vec![];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let state = exec.init_chain_state(Some(&prev), &ctx).unwrap();
        assert_eq!(state, prev);
    }

    #[test]
    fn init_chain_state_chain_merkle_copies_rate_only() {
        let exec = executor(CONFIG_D4_W16, false, true);
        let prev: Vec<F> = (1..=CONFIG_D4_W16.width_ext())
            .map(|i| F::from_u64(i as u64))
            .collect();
        let mut witness = vec![];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let state = exec.init_chain_state(Some(&prev), &ctx).unwrap();
        assert_eq!(
            state,
            vec![F::from_u64(1), F::from_u64(2), F::ZERO, F::ZERO]
        );
    }

    #[test]
    fn init_chain_state_missing_previous_errors() {
        let exec = executor(CONFIG_D4_W16, false, false);
        let mut witness = vec![];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let Err(CircuitError::Poseidon2ChainMissingPreviousState { operation_index }) =
            exec.init_chain_state(None, &ctx)
        else {
            panic!("expected Poseidon2ChainMissingPreviousState");
        };
        assert_eq!(operation_index, NonPrimitiveOpId(0));
    }

    #[test]
    fn fill_sibling_data_merkle_mode_writes_capacity_slots() {
        let exec = executor(CONFIG_D4_W16, true, true);
        let mut state = F::zero_vec(CONFIG_D4_W16.width_ext());
        let sibling = [F::from_u64(10), F::from_u64(20)];

        exec.fill_sibling_data(&mut state, Some(&sibling));

        assert_eq!(
            state,
            vec![F::ZERO, F::ZERO, F::from_u64(10), F::from_u64(20)]
        );
    }

    #[test]
    fn fill_sibling_data_non_merkle_is_noop() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut state = F::zero_vec(width_ext);

        exec.fill_sibling_data(&mut state, Some(&[F::ONE, F::ONE]));

        assert!(state.iter().all(|&v| v == F::ZERO));
    }

    #[test]
    fn fill_sibling_data_none_private_is_noop() {
        let exec = executor(CONFIG_D4_W16, true, true);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut state = F::zero_vec(width_ext);

        exec.fill_sibling_data(&mut state, None);

        assert!(state.iter().all(|&v| v == F::ZERO));
    }

    #[test]
    fn apply_witness_values_reads_single_element_slots() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let mut witness = vec![Some(F::from_u64(10)), None, Some(F::from_u64(30)), None];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let mut state = F::zero_vec(CONFIG_D4_W16.width_ext());
        let inputs: Vec<Vec<WitnessId>> = vec![
            vec![WitnessId(0)], // slot 0: has witness
            vec![],             // slot 1: empty
            vec![WitnessId(2)], // slot 2: has witness
            vec![],             // slot 3: empty
        ];

        exec.apply_witness_values(&mut state, &inputs, &ctx)
            .unwrap();

        assert_eq!(
            state,
            vec![F::from_u64(10), F::ZERO, F::from_u64(30), F::ZERO]
        );
    }

    #[test]
    fn apply_merkle_swap_swaps_when_merkle_and_bit_set() {
        let exec = executor(CONFIG_D4_W16, true, true);
        let mut state = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];

        exec.apply_merkle_swap(&mut state, true);

        // rate_ext=2, so [0..2] and [2..4] should swap.
        assert_eq!(
            state,
            vec![
                F::from_u64(3),
                F::from_u64(4),
                F::from_u64(1),
                F::from_u64(2)
            ]
        );
    }

    #[test]
    fn apply_merkle_swap_noop_when_bit_false() {
        let exec = executor(CONFIG_D4_W16, true, true);
        let original = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let mut state = original.clone();

        exec.apply_merkle_swap(&mut state, false);

        assert_eq!(state, original);
    }

    #[test]
    fn apply_merkle_swap_noop_when_not_merkle() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let original = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let mut state = original.clone();

        exec.apply_merkle_swap(&mut state, true);

        assert_eq!(state, original);
    }

    #[test]
    fn resolve_mmcs_bit_zero() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        // inputs[width_ext+1] has a witness with value 0
        let mut witness = vec![None; width_ext + 2];
        witness[5] = Some(F::ZERO);
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[width_ext + 1] = vec![WitnessId(5)];

        assert!(!exec.resolve_mmcs_bit(&inputs, &ctx).unwrap());
    }

    #[test]
    fn resolve_mmcs_bit_one() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut witness = vec![None; width_ext + 2];
        witness[5] = Some(F::ONE);
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[width_ext + 1] = vec![WitnessId(5)];

        assert!(exec.resolve_mmcs_bit(&inputs, &ctx).unwrap());
    }

    #[test]
    fn resolve_mmcs_bit_invalid_value_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut witness = vec![None; width_ext + 2];
        witness[5] = Some(F::from_u64(7));
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[width_ext + 1] = vec![WitnessId(5)];

        let Err(CircuitError::IncorrectNonPrimitiveOpPrivateData {
            op,
            operation_index,
            expected,
            got,
        }) = exec.resolve_mmcs_bit(&inputs, &ctx)
        else {
            panic!("expected IncorrectNonPrimitiveOpPrivateData");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(operation_index, NonPrimitiveOpId(0));
        assert_eq!(expected, "boolean mmcs_bit (0 or 1)");
        assert_eq!(got, format!("{:?}", F::from_u64(7)));
    }

    #[test]
    fn resolve_mmcs_bit_missing_in_merkle_mode_errors() {
        let exec = executor(CONFIG_D4_W16, true, true);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut witness = vec![None; width_ext + 2];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];

        let Err(CircuitError::IncorrectNonPrimitiveOpPrivateData {
            op,
            operation_index,
            expected,
            got,
        }) = exec.resolve_mmcs_bit(&inputs, &ctx)
        else {
            panic!("expected IncorrectNonPrimitiveOpPrivateData");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(operation_index, NonPrimitiveOpId(0));
        assert_eq!(expected, "mmcs_bit must be provided when merkle_path=true");
        assert_eq!(got, "missing mmcs_bit");
    }

    #[test]
    fn resolve_mmcs_bit_missing_non_merkle_defaults_false() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut witness = vec![None; width_ext + 2];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];

        assert!(!exec.resolve_mmcs_bit(&inputs, &ctx).unwrap());
    }

    #[test]
    fn validate_ext_inputs_correct_layout() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext(); // 4
        // width_ext + 2 = 6 input vectors, each with 0 or 1 element
        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        assert!(exec.validate_ext_inputs(&inputs).is_ok());
    }

    #[test]
    fn validate_ext_inputs_wrong_count_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; 3]; // too few
        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.validate_ext_inputs(&inputs)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(expected, "6 input vectors");
        assert_eq!(got, 3);
    }

    #[test]
    fn validate_ext_inputs_multi_witness_limb_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[0] = vec![WitnessId(0), WitnessId(1)]; // 2 witnesses = invalid
        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.validate_ext_inputs(&inputs)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(expected, "0 or 1 witness per input limb (extension-only)");
        assert_eq!(got, 2);
    }

    #[test]
    fn validate_ext_inputs_multi_mmcs_index_sum_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[width_ext] = vec![WitnessId(0), WitnessId(1)];
        let Err(CircuitError::IncorrectNonPrimitiveOpPrivateDataSize { op, expected, got }) =
            exec.validate_ext_inputs(&inputs)
        else {
            panic!("expected IncorrectNonPrimitiveOpPrivateDataSize");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(expected, "0 or 1 element for mmcs_index_sum");
        assert_eq!(got, 2);
    }

    #[test]
    fn validate_ext_inputs_multi_mmcs_bit_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();
        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[width_ext + 1] = vec![WitnessId(0), WitnessId(1)];
        let Err(CircuitError::IncorrectNonPrimitiveOpPrivateDataSize { op, expected, got }) =
            exec.validate_ext_inputs(&inputs)
        else {
            panic!("expected IncorrectNonPrimitiveOpPrivateDataSize");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(expected, "0 or 1 element for mmcs_bit");
        assert_eq!(got, 2);
    }

    #[test]
    fn validate_ext_outputs_rate_count_ok() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; CONFIG_D4_W16.rate_ext()];
        assert!(exec.validate_ext_outputs(&outputs).is_ok());
    }

    #[test]
    fn validate_ext_outputs_width_count_ok() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; CONFIG_D4_W16.width_ext()];
        assert!(exec.validate_ext_outputs(&outputs).is_ok());
    }

    #[test]
    fn validate_ext_outputs_wrong_count_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; 1]; // neither rate nor width
        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.validate_ext_outputs(&outputs)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(expected, "2 or 4 output vectors");
        assert_eq!(got, 1);
    }

    #[test]
    fn write_outputs_writes_single_witness_slots() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let mut witness: Vec<Option<F>> = vec![None; 4];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let outputs = vec![vec![WitnessId(0)], vec![], vec![WitnessId(2)]];
        let values = [F::from_u64(10), F::from_u64(20), F::from_u64(30)];

        exec.write_outputs(&outputs, &values, &mut ctx).unwrap();

        assert_eq!(
            witness,
            [Some(F::from_u64(10)), None, Some(F::from_u64(30)), None]
        );
    }

    #[test]
    fn write_outputs_multi_witness_slot_errors() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let mut witness: Vec<Option<F>> = vec![None; 4];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let outputs = vec![vec![WitnessId(0), WitnessId(1)]]; // 2 witnesses = invalid
        let values = [F::ONE];

        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.write_outputs(&outputs, &values, &mut ctx)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d4());
        assert_eq!(expected, "0 or 1 witness per output limb");
        assert_eq!(got, 2);
    }

    #[test]
    fn build_trace_row_sets_ctl_flags() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();

        let mut witness: Vec<Option<F>> = vec![None; 10];
        witness[7] = Some(F::from_u64(42));
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        inputs[0] = vec![WitnessId(1)];
        inputs[2] = vec![WitnessId(3)];
        inputs[width_ext] = vec![WitnessId(7)];

        let outputs: Vec<Vec<WitnessId>> = vec![
            vec![WitnessId(5)], // out 0 exposed
            vec![],             // out 1 not exposed
        ];

        let input_values: Vec<F> = (0..width_ext).map(|i| F::from_u64(i as u64)).collect();

        let row = exec
            .build_trace_row(&inputs, &outputs, false, input_values, &ctx)
            .unwrap();

        assert_eq!(row.in_ctl, vec![true, false, true, false]);
        assert_eq!(row.input_indices, vec![1, 0, 3, 0]);
        assert_eq!(row.out_ctl, vec![true, false]);
        assert_eq!(row.output_indices, vec![5, 0]);
        assert!(row.mmcs_ctl_enabled);
        assert_eq!(row.mmcs_index_sum, F::from_u64(42));
    }

    #[test]
    fn build_trace_row_no_mmcs_index() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let width_ext = CONFIG_D4_W16.width_ext();

        let mut witness: Vec<Option<F>> = vec![None; 10];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; width_ext + 2];
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; CONFIG_D4_W16.rate_ext()];
        let input_values = F::zero_vec(width_ext);

        let row = exec
            .build_trace_row(&inputs, &outputs, false, input_values, &ctx)
            .unwrap();

        assert!(!row.mmcs_ctl_enabled);
        assert_eq!(row.mmcs_index_sum, F::ZERO);
    }

    #[test]
    fn build_base_trace_row_one_ctl_flag_per_input_slot() {
        let exec = executor(CONFIG_D1_W16, true, false);

        // 16 inputs: make slot 0 and slot 4 nonempty
        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; 16];
        inputs[0] = vec![WitnessId(10)];
        inputs[4] = vec![WitnessId(20)];

        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; 8];
        let input_values = F::zero_vec(16);

        let row = exec.build_base_trace_row(&inputs, &outputs, &input_values);

        let mut exp_ctl = vec![false; 16];
        exp_ctl[0] = true;
        exp_ctl[4] = true;
        let mut exp_idx = vec![0u32; 16];
        exp_idx[0] = 10;
        exp_idx[4] = 20;
        assert_eq!(row.in_ctl, exp_ctl);
        assert_eq!(row.input_indices, exp_idx);
    }

    #[test]
    fn update_chain_state_normal_mode() {
        let exec = executor(CONFIG_D4_W16, true, false);
        let mut witness: Vec<Option<F>> = vec![];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let output = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let row = Poseidon2CircuitRow {
            new_start: true,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: F::ZERO,
            input_values: vec![],
            in_ctl: vec![],
            input_indices: vec![],
            out_ctl: vec![],
            output_indices: vec![],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        exec.update_chain_state(&mut ctx, output.clone(), row);

        let state = ctx
            .get_op_state::<Poseidon2ExecutionState<F>>(&exec.op_type)
            .unwrap();
        assert_eq!(state.last_output_normal, Some(output));
        assert_eq!(state.last_output_merkle, None);
        assert_eq!(state.rows.len(), 1);
    }

    #[test]
    fn update_chain_state_merkle_mode() {
        let exec = executor(CONFIG_D4_W16, true, true);
        let mut witness: Vec<Option<F>> = vec![];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let output = vec![
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ];
        let row = Poseidon2CircuitRow {
            new_start: true,
            merkle_path: true,
            mmcs_bit: false,
            mmcs_index_sum: F::ZERO,
            input_values: vec![],
            in_ctl: vec![],
            input_indices: vec![],
            out_ctl: vec![],
            output_indices: vec![],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };

        exec.update_chain_state(&mut ctx, output.clone(), row);

        let state = ctx
            .get_op_state::<Poseidon2ExecutionState<F>>(&exec.op_type)
            .unwrap();
        assert_eq!(state.last_output_normal, None);
        assert_eq!(state.last_output_merkle, Some(output));
    }

    #[test]
    fn execute_base_wrong_input_count_errors() {
        let exec = executor(CONFIG_D1_W16, true, false);
        let op_type = NpoTypeId::poseidon2_perm(CONFIG_D1_W16);
        let npo_config = crate::ops::NpoConfig::new(Poseidon2PermConfigData {
            exec: alloc::sync::Arc::new(|_: &[F]| vec![F::ZERO; 16]),
        });
        let mut configs = HashMap::new();
        configs.insert(op_type, npo_config);
        let mut witness: Vec<Option<F>> = vec![None; 32];
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; 10]; // not 16
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; 8];

        let op_type_d1 = NpoTypeId::poseidon2_perm(CONFIG_D1_W16);

        let identity = |s: &[F]| s.to_vec();
        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.execute_base(&inputs, &outputs, &mut ctx, &identity)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d1);
        assert_eq!(expected, "16 or 18 input vectors for D=1 mode");
        assert_eq!(got, 10);
    }

    #[test]
    fn execute_base_wrong_output_count_errors() {
        let exec = executor(CONFIG_D1_W16, true, false);
        let op_type_d1 = NpoTypeId::poseidon2_perm(CONFIG_D1_W16);
        let mut witness: Vec<Option<F>> = vec![None; 32];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let inputs: Vec<Vec<WitnessId>> = vec![vec![]; 16];
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; 5]; // neither 8 nor 16

        let identity = |s: &[F]| s.to_vec();
        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.execute_base(&inputs, &outputs, &mut ctx, &identity)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d1);
        assert_eq!(expected, "8 or 16 output vectors for D=1 mode");
        assert_eq!(got, 5);
    }

    #[test]
    fn execute_base_multi_witness_per_input_errors() {
        let exec = executor(CONFIG_D1_W16, true, false);
        let op_type_d1 = NpoTypeId::poseidon2_perm(CONFIG_D1_W16);
        let mut witness: Vec<Option<F>> = vec![None; 32];
        let configs = HashMap::new();
        let mut op_states = BTreeMap::new();
        let mut ctx = make_ctx(&mut witness, &[], &configs, &mut op_states);

        let mut inputs: Vec<Vec<WitnessId>> = vec![vec![]; 16];
        inputs[3] = vec![WitnessId(0), WitnessId(1)];
        let outputs: Vec<Vec<WitnessId>> = vec![vec![]; 8];

        let identity = |s: &[F]| s.to_vec();
        let Err(CircuitError::NonPrimitiveOpLayoutMismatch { op, expected, got }) =
            exec.execute_base(&inputs, &outputs, &mut ctx, &identity)
        else {
            panic!("expected NonPrimitiveOpLayoutMismatch");
        };
        assert_eq!(op, op_type_d1);
        assert_eq!(expected, "0 or 1 witness per input element 3");
        assert_eq!(got, 2);
    }
}
