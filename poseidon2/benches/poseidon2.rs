use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
// use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
// use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
use p3_field::{AbstractField, Field};
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_goldilocks::{
    Goldilocks, Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks,
};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_koala_bear::{KoalaBear, Poseidon2ExternalLayerKoalaBear, Poseidon2InternalLayerKoalaBear};
use p3_mersenne_31::{
    Mersenne31, Poseidon2ExternalLayerMersenne31, Poseidon2InternalLayerMersenne31,
};
use p3_poseidon2::{
    ExternalLayer, InternalLayer, Poseidon2, Poseidon2ExternalPackedConstants,
    Poseidon2InternalPackedConstants,
};
// use p3_koala_bear::{DiffusionMatrixKoalaBear, KoalaBear};
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
use p3_symmetric::Permutation;
use rand::thread_rng;

fn bench_poseidon2(c: &mut Criterion) {
    let mut rng = thread_rng();
    // poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>(c);
    // poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 24, 7>(c);

    // poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 3>(c);
    // poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, 3>(c);

    let poseidon2_m31_16 = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
    poseidon2::<Mersenne31, Poseidon2Mersenne31<16>, 16, 5>(c, poseidon2_m31_16);
    let poseidon2_m31_24 = Poseidon2Mersenne31::<24>::new_from_rng_128(&mut rng);
    poseidon2::<Mersenne31, Poseidon2Mersenne31<24>, 24, 5>(c, poseidon2_m31_24);

    let poseidon2_gold_8 = Poseidon2Goldilocks::<8>::new_from_rng_128(&mut rng);
    poseidon2::<Goldilocks, Poseidon2Goldilocks<8>, 8, 5>(c, poseidon2_gold_8);
    let poseidon2_gold_12 = Poseidon2Goldilocks::<12>::new_from_rng_128(&mut rng);
    poseidon2::<Goldilocks, Poseidon2Goldilocks<12>, 12, 5>(c, poseidon2_gold_12);
    let poseidon2_gold_16 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    poseidon2::<Goldilocks, Poseidon2Goldilocks<16>, 16, 5>(c, poseidon2_gold_16);

    // poseidon2::<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>(c, 8, 22);
}

fn poseidon2<F, Perm, const WIDTH: usize, const D: u64>(c: &mut Criterion, poseidon2: Perm)
where
    F: Field,
    Perm: Permutation<[F::Packing; WIDTH]>,
{
    let input = [F::Packing::zero(); WIDTH];
    let name = format!(
        "poseidon2::<{}, {}, {}>",
        type_name::<F::Packing>(),
        D,
        WIDTH
    );
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon2.permute(input)));
}

criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
