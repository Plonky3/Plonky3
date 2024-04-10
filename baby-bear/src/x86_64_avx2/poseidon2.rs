use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    BabyBear, DiffusionMatrixBabybear, PackedBabyBearAVX2, MATRIX_DIAG_16_BABYBEAR_MONTY,
    MATRIX_DIAG_24_BABYBEAR_MONTY, MONTY_INVERSE,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl Permutation<[PackedBabyBearAVX2; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearAVX2; 16]) {
        matmul_internal::<BabyBear, PackedBabyBearAVX2, 16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY);
        state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
    }
}

impl DiffusionPermutation<PackedBabyBearAVX2, 16> for DiffusionMatrixBabybear {}

impl Permutation<[PackedBabyBearAVX2; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [PackedBabyBearAVX2; 24]) {
        matmul_internal::<BabyBear, PackedBabyBearAVX2, 24>(state, MATRIX_DIAG_24_BABYBEAR_MONTY);
        state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
    }
}

impl DiffusionPermutation<PackedBabyBearAVX2, 24> for DiffusionMatrixBabybear {}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{DiffusionMatrixBabybear, BabyBear, PackedBabyBearAVX2};

    type F = BabyBear;
    const D: u64 = 7;
    type Perm16 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 16, D>;
    type Perm24 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 24, D>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
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
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabybear,
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
