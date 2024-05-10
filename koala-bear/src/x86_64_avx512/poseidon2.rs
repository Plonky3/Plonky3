use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{
    DiffusionMatrixKoalaBear, KoalaBear, PackedKoalaBearAVX512, MONTY_INVERSE,
    POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY,
    POSEIDON2_INTERNAL_MATRIX_DIAG_24_KOALABEAR_MONTY,
};

// We need to change from the standard implementation as we are interpreting the matrix (1 + D(v)) as the monty form of the matrix not the raw form.
// matmul_internal internal performs a standard matrix multiplication so we need to additional rescale by the inverse monty constant.
// These will be removed once we have architecture specific implementations.

impl Permutation<[PackedKoalaBearAVX512; 16]> for DiffusionMatrixKoalaBear {
    fn permute_mut(&self, state: &mut [PackedKoalaBearAVX512; 16]) {
        matmul_internal::<KoalaBear, PackedKoalaBearAVX512, 16>(
            state,
            POSEIDON2_INTERNAL_MATRIX_DIAG_16_KOALABEAR_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
    }
}

impl DiffusionPermutation<PackedKoalaBearAVX512, 16> for DiffusionMatrixKoalaBear {}

impl Permutation<[PackedKoalaBearAVX512; 24]> for DiffusionMatrixKoalaBear {
    fn permute_mut(&self, state: &mut [PackedKoalaBearAVX512; 24]) {
        matmul_internal::<KoalaBear, PackedKoalaBearAVX512, 24>(
            state,
            POSEIDON2_INTERNAL_MATRIX_DIAG_24_KOALABEAR_MONTY,
        );
        state.iter_mut().for_each(|i| *i *= MONTY_INVERSE);
    }
}

impl DiffusionPermutation<PackedKoalaBearAVX512, 24> for DiffusionMatrixKoalaBear {}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{DiffusionMatrixKoalaBear, KoalaBear, PackedKoalaBearAVX512};

    type F = KoalaBear;
    const D: u64 = 7;
    type Perm16 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, D>;
    type Perm24 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, D>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixKoalaBear,
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(PackedKoalaBearAVX512::from_f);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixKoalaBear,
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(PackedKoalaBearAVX512::from_f);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }
}
