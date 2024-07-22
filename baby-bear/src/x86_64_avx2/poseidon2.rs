use p3_poseidon2::{internal_permute_state, matmul_internal, InternalLayer};

use crate::{
    BabyBear, DiffusionMatrixBabyBear, PackedBabyBearAVX2, MONTY_INVERSE,
    POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY,
    POSEIDON2_INTERNAL_MATRIX_DIAG_24_BABYBEAR_MONTY,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl InternalLayer<PackedBabyBearAVX2, 16, 7> for DiffusionMatrixBabyBear {
    type InternalState = [PackedBabyBearAVX2; 16];

    type InternalConstantsType = BabyBear;

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Self::InternalConstantsType],
    ) {
        internal_permute_state::<PackedBabyBearAVX2, 16, 7>(
            state,
            |state| {
                matmul_internal::<BabyBear, PackedBabyBearAVX2, 16>(
                    state,
                    POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY,
                );
                state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
            },
            internal_constants,
        )
    }
}

impl InternalLayer<PackedBabyBearAVX2, 24, 7> for DiffusionMatrixBabyBear {
    type InternalState = [PackedBabyBearAVX2; 24];

    type InternalConstantsType = BabyBear;

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Self::InternalConstantsType],
    ) {
        internal_permute_state::<PackedBabyBearAVX2, 24, 7>(
            state,
            |state| {
                matmul_internal::<BabyBear, PackedBabyBearAVX2, 24>(
                    state,
                    POSEIDON2_INTERNAL_MATRIX_DIAG_24_BABYBEAR_MONTY,
                );
                state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
            },
            internal_constants,
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{
        BabyBear, DiffusionMatrixBabyBear, MDSLightPermutationBabyBear, PackedBabyBearAVX2,
    };

    type F = BabyBear;
    const D: u64 = 7;
    type Perm16 = Poseidon2<F, MDSLightPermutationBabyBear, DiffusionMatrixBabyBear, 16, D>;
    type Perm24 = Poseidon2<F, MDSLightPermutationBabyBear, DiffusionMatrixBabyBear, 24, D>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            MDSLightPermutationBabyBear,
            DiffusionMatrixBabyBear,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedBabyBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            MDSLightPermutationBabyBear,
            DiffusionMatrixBabyBear,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedBabyBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
