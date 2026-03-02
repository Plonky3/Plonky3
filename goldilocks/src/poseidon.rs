//! Tests for the Poseidon permutation over the Goldilocks field.
//!
//! Verifies that scalar and packed (SIMD) implementations produce identical
//! results for widths 16 and 24.

use p3_poseidon::{Poseidon, PoseidonExternalLayerGeneric, PoseidonInternalLayerGeneric};
use p3_symmetric::Permutation;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::{Goldilocks, MdsMatrixGoldilocks};

type PoseidonGoldilocks<const WIDTH: usize> = Poseidon<
    Goldilocks,
    PoseidonExternalLayerGeneric<Goldilocks, MdsMatrixGoldilocks, WIDTH>,
    PoseidonInternalLayerGeneric<Goldilocks, WIDTH>,
    WIDTH,
    7,
>;

#[test]
fn test_poseidon_goldilocks_width_16() {
    let mut rng = SmallRng::seed_from_u64(1);
    let poseidon = PoseidonGoldilocks::<16>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
    let input: [Goldilocks; 16] = rng.random();
    let output = poseidon.permute(input);
    assert_ne!(output, input);
}

#[test]
fn test_poseidon_goldilocks_width_24() {
    let mut rng = SmallRng::seed_from_u64(1);
    let poseidon = PoseidonGoldilocks::<24>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
    let input: [Goldilocks; 24] = rng.random();
    let output = poseidon.permute(input);
    assert_ne!(output, input);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512 {
    use super::*;
    use crate::PackedGoldilocksAVX512;

    #[test]
    fn test_avx512_poseidon_width_16() {
        let mut rng = SmallRng::seed_from_u64(1);
        let poseidon =
            PoseidonGoldilocks::<16>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
        let input: [Goldilocks; 16] = rng.random();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedGoldilocksAVX512>::into);
        poseidon.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);
        assert_eq!(avx512_output, expected);
    }

    #[test]
    fn test_avx512_poseidon_width_24() {
        let mut rng = SmallRng::seed_from_u64(1);
        let poseidon =
            PoseidonGoldilocks::<24>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
        let input: [Goldilocks; 24] = rng.random();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedGoldilocksAVX512>::into);
        poseidon.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);
        assert_eq!(avx512_output, expected);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod avx2 {
    use super::*;
    use crate::PackedGoldilocksAVX2;

    #[test]
    fn test_avx2_poseidon_width_16() {
        let mut rng = SmallRng::seed_from_u64(1);
        let poseidon =
            PoseidonGoldilocks::<16>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
        let input: [Goldilocks; 16] = rng.random();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx2_input = input.map(Into::<PackedGoldilocksAVX2>::into);
        poseidon.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }

    #[test]
    fn test_avx2_poseidon_width_24() {
        let mut rng = SmallRng::seed_from_u64(1);
        let poseidon =
            PoseidonGoldilocks::<24>::new_from_rng(4, 22, &MdsMatrixGoldilocks, &mut rng);
        let input: [Goldilocks; 24] = rng.random();

        let mut expected = input;
        poseidon.permute_mut(&mut expected);

        let mut avx2_input = input.map(Into::<PackedGoldilocksAVX2>::into);
        poseidon.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);
        assert_eq!(avx2_output, expected);
    }
}
