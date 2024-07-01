#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{DiffusionMatrixKoalaBear, KoalaBear, PackedKoalaBearAVX2};

    type F = KoalaBear;
    const D: u64 = 7;
    type Perm16 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, D>;
    type Perm24 = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, D>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixKoalaBear::default(),
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
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
            DiffusionMatrixKoalaBear::default(),
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
