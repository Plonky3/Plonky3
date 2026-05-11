//! Inherent `CircuitBuilder` methods for adding Poseidon2 permutation rows.

use alloc::string::ToString;
use alloc::vec::Vec;
use alloc::{format, vec};

use p3_field::Field;

use crate::CircuitBuilderError;
use crate::builder::{CircuitBuilder, NonPrimitiveOpParams};
use crate::ops::NpoTypeId;
use crate::ops::poseidon2_perm::call::{Poseidon2PermCall, Poseidon2PermCallBase};
use crate::types::{ExprId, NonPrimitiveOpId};

impl<F: Field> CircuitBuilder<F> {
    /// Add a Poseidon2 perm row (one permutation) for D>=2 extension field.
    ///
    /// Returns `(op_id, outputs)` where outputs has length `width_ext`:
    /// - `outputs[0..rate_ext]`: present if `out_ctl[i]` is true (CTL-verified)
    /// - `outputs[rate_ext..]`: present if `return_all_outputs` is true (capacity, not CTL-verified)
    pub fn add_poseidon2_perm(
        &mut self,
        call: &Poseidon2PermCall,
    ) -> Result<(NonPrimitiveOpId, Vec<Option<ExprId>>), CircuitBuilderError> {
        let op_type = NpoTypeId::poseidon2_perm(call.config);
        self.ensure_op_enabled(&op_type)?;

        if call.merkle_path && call.mmcs_bit.is_none() {
            return Err(CircuitBuilderError::Poseidon2MerkleMissingMmcsBit);
        }
        if !call.merkle_path && call.mmcs_bit.is_some() {
            return Err(CircuitBuilderError::Poseidon2NonMerkleWithMmcsBit);
        }

        let width_ext = call.config.width_ext();
        let rate_ext = call.config.rate_ext();

        let mut input_exprs: Vec<Vec<ExprId>> = Vec::with_capacity(width_ext + 2);
        for limb in &call.inputs {
            input_exprs.push(limb.map_or_else(Vec::new, |v| vec![v]));
        }
        input_exprs.push(call.mmcs_index_sum.map_or_else(Vec::new, |v| vec![v]));
        input_exprs.push(call.mmcs_bit.map_or_else(Vec::new, |v| vec![v]));

        let mut output_labels: Vec<Option<&'static str>> = Vec::with_capacity(width_ext);
        for i in 0..rate_ext {
            let expose = call.out_ctl.get(i).copied().unwrap_or(false);
            output_labels.push(expose.then_some("poseidon2_perm_out"));
        }
        for _ in rate_ext..width_ext {
            output_labels.push(
                call.return_all_outputs
                    .then_some("poseidon2_perm_out_capacity"),
            );
        }

        let (op_id, _call_expr_id, outputs) = self.push_non_primitive_op_with_outputs(
            op_type,
            input_exprs,
            output_labels,
            Some(NonPrimitiveOpParams::Poseidon2Perm {
                new_start: call.new_start,
                merkle_path: call.merkle_path,
            }),
            "poseidon2_perm",
        );
        Ok((op_id, outputs))
    }

    /// Add a Poseidon2 perm row (one permutation) for D=1 base field.
    ///
    /// Returns `(op_id, outputs)` where outputs is `[Option<ExprId>; 16]`:
    /// - `outputs[0..8]`: present if `out_ctl[i]` is true (CTL-verified, rate elements)
    /// - `outputs[8..16]`: present if `return_all_outputs` is true (capacity, not CTL-verified)
    pub fn add_poseidon2_perm_base(
        &mut self,
        call: &Poseidon2PermCallBase,
    ) -> Result<(NonPrimitiveOpId, [Option<ExprId>; 16]), CircuitBuilderError> {
        let op_type = NpoTypeId::poseidon2_perm(call.config);
        self.ensure_op_enabled(&op_type)?;

        if call.config.d() != 1 {
            return Err(CircuitBuilderError::Poseidon2ConfigMismatch {
                expected: "D=1 configuration".to_string(),
                got: format!("D={} configuration", call.config.d()),
            });
        }

        let input_exprs: [Vec<ExprId>; 16] = call
            .inputs
            .map(|opt| opt.map_or_else(Vec::new, |v| vec![v]));

        let output_labels: [Option<&'static str>; 16] = core::array::from_fn(|i| match i {
            0..8 if call.out_ctl[i] => Some("poseidon2_perm_base_out"),
            8..16 if call.return_all_outputs => Some("poseidon2_perm_base_out_capacity"),
            _ => None,
        });

        let (op_id, _call_expr_id, outputs) = self.push_non_primitive_op_with_outputs(
            op_type,
            input_exprs.into(),
            output_labels.into(),
            Some(NonPrimitiveOpParams::Poseidon2Perm {
                new_start: call.new_start,
                merkle_path: false,
            }),
            "poseidon2_perm_base",
        );

        let outputs: [Option<ExprId>; 16] = outputs
            .try_into()
            .expect("push_non_primitive_op_with_outputs must return exactly 16 outputs");
        Ok((op_id, outputs))
    }
}
