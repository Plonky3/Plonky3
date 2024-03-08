use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabybear};
use p3_bn254::{DiffusionMatrixBN254, BN254};
use p3_field::PrimeField;
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_poseidon2::{DiffusionPermutation, Poseidon2};
use p3_symmetric::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_poseidon2(c: &mut Criterion) {
    poseidon2::<BabyBear, DiffusionMatrixBabybear, 16, 7>(c);
    poseidon2::<BabyBear, DiffusionMatrixBabybear, 24, 7>(c);

    poseidon2::<Goldilocks, DiffusionMatrixGoldilocks, 8, 7>(c);
    poseidon2::<Goldilocks, DiffusionMatrixGoldilocks, 12, 7>(c);
    poseidon2::<Goldilocks, DiffusionMatrixGoldilocks, 16, 7>(c);

    poseidon2::<BN254, DiffusionMatrixBN254, 3, 5>(c);
}

fn poseidon2<F, Diffusion, const WIDTH: usize, const D: u64>(c: &mut Criterion)
where
    F: PrimeField,
    Standard: Distribution<F>,
    Diffusion: DiffusionPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let internal_mds = Diffusion::default();

    // TODO: Should be calculated for the particular field, width and ALPHA.
    let rounds_f = 8;
    let rounds_p = 22;

    let poseidon = Poseidon2::<F, Diffusion, WIDTH, D>::new_from_rng(
        rounds_f,
        rounds_p,
        internal_mds,
        &mut rng,
    );
    let input = [F::zero(); WIDTH];
    let name = format!("poseidon2::<{}, {}>", type_name::<F>(), D);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
