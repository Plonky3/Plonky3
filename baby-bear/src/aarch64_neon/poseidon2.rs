use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    BabyBear, DiffusionMatrixBabybear, PackedBabyBearNeon, POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY,
    POSEIDON2_INTERNAL_MATRIX_DIAG_24_BABYBEAR_MONTY, MONTY_INVERSE,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl Permutation<[PackedBabyBearNeon; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearNeon; 16]) {
        matmul_internal::<BabyBear, PackedBabyBearNeon, 16>(state, POSEIDON2_INTERNAL_MATRIX_DIAG_16_BABYBEAR_MONTY);
        state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
    }
}

impl DiffusionPermutation<PackedBabyBearNeon, 16> for DiffusionMatrixBabybear {}

impl Permutation<[PackedBabyBearNeon; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearNeon; 24]) {
        matmul_internal::<BabyBear, PackedBabyBearNeon, 24>(state, POSEIDON2_INTERNAL_MATRIX_DIAG_24_BABYBEAR_MONTY);
        state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
    }
}

impl DiffusionPermutation<PackedBabyBearNeon, 24> for DiffusionMatrixBabybear {}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{DiffusionMatrixBabybear, BabyBear, PackedBabyBearNeon};

    type F = BabyBear;
    const D: u64 = 7;
    type Perm16 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 16, D>;
    type Perm24 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 24, D>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_neon_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabybear,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut neon_input = input.map(PackedBabyBearNeon::from_f);
        poseidon2.permute_mut(&mut neon_input);

        let neon_output = neon_input.map(|x| x.0[0]);

        assert_eq!(neon_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_neon_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabybear,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut neon_input = input.map(PackedBabyBearNeon::from_f);
        poseidon2.permute_mut(&mut neon_input);

        let neon_output = neon_input.map(|x| x.0[0]);

        assert_eq!(neon_output, expected);
    }
}