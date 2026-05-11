use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use p3_field::Field;

use crate::ops::{Poseidon2Config, Poseidon2PermCall};
use crate::{CircuitBuilder, CircuitBuilderError, ExprId, NonPrimitiveOpId};

impl<F: Field> CircuitBuilder<F> {
    pub fn add_hash_slice(
        &mut self,
        poseidon2_config: &Poseidon2Config,
        inputs: &[ExprId],
        reset: bool,
    ) -> Result<Vec<ExprId>, CircuitBuilderError> {
        let width_ext = poseidon2_config.width_ext();
        let rate_ext = poseidon2_config.rate_ext();
        let chunks = inputs.chunks(rate_ext);
        let last_idx = chunks.len() - 1;
        let mut outputs = vec![None; width_ext];
        let mut last_op_id = NonPrimitiveOpId(0);
        for (i, input) in chunks.enumerate() {
            let is_first = i == 0;
            let is_last = i == last_idx;
            let call_inputs: Vec<Option<ExprId>> = input
                .iter()
                .copied()
                .map(Some)
                .chain(iter::repeat(None))
                .take(width_ext)
                .collect();
            let (op_id, maybe_outputs) = self.add_poseidon2_perm(&Poseidon2PermCall {
                config: *poseidon2_config,
                new_start: is_first && reset,
                merkle_path: false,
                mmcs_bit: None,
                inputs: call_inputs,
                out_ctl: vec![is_last; rate_ext],
                return_all_outputs: false,
                mmcs_index_sum: None,
            })?;
            outputs = maybe_outputs;
            last_op_id = op_id;
        }

        outputs
            .into_iter()
            .take(rate_ext)
            .map(|o| {
                o.ok_or_else(|| CircuitBuilderError::MalformedNonPrimitiveOutputs {
                    op_id: last_op_id,
                    details: "".to_string(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::iter;

    use itertools::Itertools;
    use p3_symmetric::CryptographicHasher;
    use p3_test_utils::baby_bear_params::*;

    use crate::ops::{
        Poseidon2Config, Poseidon2Params, generate_poseidon2_trace, generate_recompose_trace,
    };
    use crate::{CircuitBuilder, ExprId};

    type CF = Challenge;

    struct DummyParams;

    impl Poseidon2Params for DummyParams {
        type BaseField = F;
        const CONFIG: Poseidon2Config = Poseidon2Config::BabyBearD4Width16;
    }

    #[test]
    fn test_hash_squeeze() {
        let perm = default_babybear_poseidon2_16();
        let hasher = MyHash::new(perm.clone());

        // Test only lengths that are multiples of 4 (aligned to extension degree)
        // Non-aligned lengths (like 9, 10, 11) have different behavior in overwrite-mode
        // sponge vs zero-padded extension packing - see test_hash_non_aligned for details
        for len in [4, 8, 12, 16, 32, 64] {
            let base_inputs = (0..len)
                .map(|i| F::from_u64(i as u64 + 1))
                .collect::<Vec<_>>();
            let expected = hasher.hash_iter(base_inputs.clone());

            let mut builder = CircuitBuilder::<CF>::new();
            builder.enable_poseidon2_perm::<DummyParams, _>(
                generate_poseidon2_trace::<CF, DummyParams>,
                perm.clone(),
            );
            builder.enable_recompose::<F>(generate_recompose_trace::<F, CF>);

            let input_exprs: Vec<ExprId> = (0..base_inputs.len())
                .chunks(<CF as BasedVectorSpace<F>>::DIMENSION)
                .into_iter()
                .map(|_| builder.public_input())
                .collect();

            let outputs = builder
                .add_hash_slice(&Poseidon2Config::BabyBearD4Width16, &input_exprs, true)
                .unwrap();

            let out0_pi = builder.public_input();
            let out1_pi = builder.public_input();
            builder.connect(outputs[0], out0_pi);
            builder.connect(outputs[1], out1_pi);

            let circuit = builder.build().unwrap();
            let mut runner = circuit.runner();
            let mut public_inputs = base_inputs // Pad to multiple of 4
                .chunks(<CF as BasedVectorSpace<F>>::DIMENSION)
                .map(|chunk| {
                    let chunk: Vec<F> = chunk
                        .iter()
                        .copied()
                        .chain(iter::repeat(F::ZERO))
                        .take(<CF as BasedVectorSpace<F>>::DIMENSION)
                        .collect();
                    CF::from_basis_coefficients_slice(&chunk).unwrap()
                })
                .collect::<Vec<_>>();
            let expected_limb0 = CF::from_basis_coefficients_slice(&expected[0..4]).unwrap();
            let expected_limb1 = CF::from_basis_coefficients_slice(&expected[4..8]).unwrap();
            public_inputs.push(expected_limb0);
            public_inputs.push(expected_limb1);
            runner.set_public_inputs(&public_inputs).unwrap();

            runner.run().unwrap();
        }
    }

    /// Test that exposes the mismatch between circuit hashing and native PaddingFreeSponge
    /// for non-aligned input lengths.
    ///
    /// Native PaddingFreeSponge uses "overwrite mode": when absorbing a partial chunk,
    /// only the absorbed positions are overwritten; the rest keep their previous values.
    ///
    /// The circuit's add_hash_slice + repack_base_to_ext uses zero-padding: partial
    /// extension elements are padded with zeros, which overwrites all rate positions.
    ///
    /// This test demonstrates the mismatch for 9 base field elements (not a multiple of 4).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "WitnessConflict")]
    fn test_hash_non_aligned_shows_mismatch() {
        let perm = default_babybear_poseidon2_16();
        let hasher = MyHash::new(perm.clone());

        // 9 elements: NOT aligned to extension degree (4)
        // Circuit will pack as 3 extension elements: [[v0,v1,v2,v3], [v4,v5,v6,v7], [v8,0,0,0]]
        // Native sponge will absorb: [v0..v7], permute, then [v8] (keeping positions 1-7 from permutation)
        let len = 9;
        let base_inputs = (0..len)
            .map(|i| F::from_u64(i as u64 + 1))
            .collect::<Vec<_>>();
        let expected = hasher.hash_iter(base_inputs.clone());

        let mut builder = CircuitBuilder::<CF>::new();
        builder.enable_poseidon2_perm::<DummyParams, _>(
            generate_poseidon2_trace::<CF, DummyParams>,
            perm,
        );
        builder.enable_recompose::<F>(generate_recompose_trace::<F, CF>);

        // Pack base inputs into extension elements (will zero-pad the last one)
        let input_exprs: Vec<ExprId> = base_inputs
            .chunks(<CF as BasedVectorSpace<F>>::DIMENSION)
            .map(|_| builder.public_input())
            .collect();

        let outputs = builder
            .add_hash_slice(&Poseidon2Config::BabyBearD4Width16, &input_exprs, true)
            .unwrap();

        let out0_pi = builder.public_input();
        let out1_pi = builder.public_input();
        builder.connect(outputs[0], out0_pi);
        builder.connect(outputs[1], out1_pi);

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();

        // Pack base inputs with zero-padding for the last chunk
        let mut public_inputs = base_inputs
            .chunks(<CF as BasedVectorSpace<F>>::DIMENSION)
            .map(|chunk| {
                let chunk: Vec<F> = chunk
                    .iter()
                    .copied()
                    .chain(iter::repeat(F::ZERO))
                    .take(<CF as BasedVectorSpace<F>>::DIMENSION)
                    .collect();
                CF::from_basis_coefficients_slice(&chunk).unwrap()
            })
            .collect::<Vec<_>>();

        // Native hash expects different result due to overwrite mode
        let expected_limb0 = CF::from_basis_coefficients_slice(&expected[0..4]).unwrap();
        let expected_limb1 = CF::from_basis_coefficients_slice(&expected[4..8]).unwrap();
        public_inputs.push(expected_limb0);
        public_inputs.push(expected_limb1);
        runner.set_public_inputs(&public_inputs).unwrap();

        // This will panic with WitnessConflict because circuit hash != native hash
        runner.run().unwrap();
    }
}
