// use core::arch::x86_64::{self, __m256i};
// use core::mem::transmute;

use p3_poseidon2::{
    external_final_permute_state, external_initial_permute_state, internal_permute_state,
    matmul_internal, ExternalLayer, InternalLayer, MDSMat4,
};

use crate::{
    Mersenne31, PackedMersenne31AVX2, Poseidon2ExternalLayerMersenne31,
    Poseidon2InternalLayerMersenne31, POSEIDON2_INTERNAL_MATRIX_DIAG_16,
    POSEIDON2_INTERNAL_MATRIX_DIAG_24,
};

impl InternalLayer<PackedMersenne31AVX2, 16, 5> for Poseidon2InternalLayerMersenne31 {
    type InternalState = [PackedMersenne31AVX2; 16];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Mersenne31],
        _packed_internal_constants: &[()],
    ) {
        internal_permute_state::<PackedMersenne31AVX2, 16, 5>(
            state,
            |x| matmul_internal(x, POSEIDON2_INTERNAL_MATRIX_DIAG_16),
            internal_constants,
        )
    }
}

impl InternalLayer<PackedMersenne31AVX2, 24, 5> for Poseidon2InternalLayerMersenne31 {
    type InternalState = [PackedMersenne31AVX2; 24];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Mersenne31],
        _packed_internal_constants: &[()],
    ) {
        internal_permute_state::<PackedMersenne31AVX2, 24, 5>(
            state,
            |x| matmul_internal(x, POSEIDON2_INTERNAL_MATRIX_DIAG_24),
            internal_constants,
        )
    }
}

impl<const WIDTH: usize> ExternalLayer<PackedMersenne31AVX2, WIDTH, 5>
    for Poseidon2ExternalLayerMersenne31
{
    type InternalState = [PackedMersenne31AVX2; WIDTH];

    fn permute_state_initial(
        &self,
        mut state: Self::InternalState,
        initial_external_constants: &[[Mersenne31; WIDTH]],
        _packed_initial_external_constants: &[()],
    ) -> [PackedMersenne31AVX2; WIDTH] {
        external_initial_permute_state::<_, _, WIDTH, 5>(
            &mut state,
            initial_external_constants,
            &MDSMat4,
        );
        state
    }

    fn permute_state_final(
        &self,
        mut state: Self::InternalState,
        final_external_constants: &[[Mersenne31; WIDTH]],
        _packed_final_external_constants: &[()],
    ) -> [PackedMersenne31AVX2; WIDTH] {
        external_final_permute_state::<_, _, WIDTH, 5>(
            &mut state,
            final_external_constants,
            &MDSMat4,
        );
        state
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use super::*;

    type F = Mersenne31;
    const D: u64 = 5;
    type Perm16 =
        Poseidon2<F, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31, 16, D>;
    type Perm24 =
        Poseidon2<F, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31, 24, D>;

    /// Test that the output is the same as the scalar version on a random input of length 16.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedMersenne31AVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input of length 24.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedMersenne31AVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
