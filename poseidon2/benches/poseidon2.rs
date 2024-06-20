use std::any::type_name;
use std::mem::transmute;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
use p3_field::{PackedValue, PrimeField, PrimeField32, PrimeField64};
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_koala_bear::{DiffusionMatrixKoalaBear, KoalaBear};
use p3_mersenne_31::{
    final_external_rounds, initial_external_rounds, internal_rounds, DiffusionMatrixMersenne31,
    Mersenne31, Packed64bitM31Matrix, PackedMersenne31AVX2,
};
use p3_poseidon2::{
    DiffusionPermutation, MdsLightPermutation, Poseidon2, Poseidon2ExternalMatrixGeneral,
    Poseidon2Fast,
};
use p3_symmetric::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_poseidon2(c: &mut Criterion) {
    poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>(c);
    poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 24, 7>(c);

    poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 3>(c);
    poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 5>(c);
    poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 7>(c);
    poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, 3>(c);
    poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, 5>(c);
    poseidon2_p64::<KoalaBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, 7>(c);

    poseidon2_p64::<Mersenne31, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 16, 5>(
        c,
    );
    poseidon2_p64::<Mersenne31, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 24, 5>(
        c,
    );
    poseidon2_avx2_m31(c);

    poseidon2_p64::<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 8, 7>(c);
    poseidon2_p64::<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 12, 7>(
        c,
    );
    poseidon2_p64::<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 16, 7>(
        c,
    );

    poseidon2::<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>(c, 8, 22);
}

fn poseidon2<F, MdsLight, Diffusion, const WIDTH: usize, const D: u64>(
    c: &mut Criterion,
    rounds_f: usize,
    rounds_p: usize,
) where
    F: PrimeField,
    Standard: Distribution<F>,
    MdsLight: MdsLightPermutation<F, WIDTH> + Default,
    Diffusion: DiffusionPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let external_linear_layer = MdsLight::default();
    let internal_linear_layer = Diffusion::default();

    let poseidon = Poseidon2::<F, MdsLight, Diffusion, WIDTH, D>::new_from_rng(
        rounds_f,
        external_linear_layer,
        rounds_p,
        internal_linear_layer,
        &mut rng,
    );
    let input = [F::zero(); WIDTH];
    let name = format!(
        "poseidon2::<{}, {}, {}, {}>",
        type_name::<F>(),
        D,
        rounds_f,
        rounds_p
    );
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

// For fields implementing PrimeField64 we should benchmark using the optimal round constants.
fn poseidon2_p64<F, MdsLight, Diffusion, const WIDTH: usize, const D: u64>(c: &mut Criterion)
where
    F: PrimeField64,
    Standard: Distribution<F>,
    MdsLight: MdsLightPermutation<F, WIDTH> + Default,
    Diffusion: DiffusionPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let external_linear_layer = MdsLight::default();
    let internal_linear_layer = Diffusion::default();

    let poseidon = Poseidon2::<F, MdsLight, Diffusion, WIDTH, D>::new_from_rng_128(
        external_linear_layer,
        internal_linear_layer,
        &mut rng,
    );
    let input = [F::zero(); WIDTH];
    let name = format!("poseidon2::<{}, {}>", type_name::<F>(), D);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

fn poseidon2_avx2_m31(c: &mut Criterion) {
    const PACKED_WIDTH: usize = 2;
    const FULL_WIDTH: usize = PACKED_WIDTH * PackedMersenne31AVX2::WIDTH;

    let mut rng = thread_rng();

    let poseidon_2: Poseidon2Fast<PackedMersenne31AVX2, Packed64bitM31Matrix, u32, PACKED_WIDTH> =
        Poseidon2Fast::new_from_rng_128::<_, 5>(
            Packed64bitM31Matrix::from_packed_field_array,
            Packed64bitM31Matrix::to_packed_field_array,
            |x| x.as_canonical_u32(),
            initial_external_rounds,
            internal_rounds,
            final_external_rounds,
            &mut rng,
        );

    let avx2_input: [PackedMersenne31AVX2; PACKED_WIDTH] =
        unsafe { transmute([0_u32; FULL_WIDTH]) };
    let name = "poseidon2_avx2_Mersenne31";
    let id = BenchmarkId::new(name, 16);
    c.bench_with_input(id, &avx2_input, |b, &avx2_input| {
        b.iter(|| poseidon_2.permute(avx2_input))
    });
}
criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
