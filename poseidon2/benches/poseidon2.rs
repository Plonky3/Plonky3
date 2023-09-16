use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use p3_mds::babybear::MdsMatrixBabyBear;
use p3_mds::goldilocks::MdsMatrixGoldilocks;
use p3_mds::mersenne31::MdsMatrixMersenne31;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_poseidon2::Poseidon2;
use p3_symmetric::permutation::CryptographicPermutation;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_poseidon2(c: &mut Criterion) {
    poseidon2::<BabyBear, MdsMatrixBabyBear, 16, 7>(c);
    poseidon2::<BabyBear, MdsMatrixBabyBear, 24, 7>(c);

    poseidon2::<Goldilocks, MdsMatrixGoldilocks, 8, 7>(c);
    poseidon2::<Goldilocks, MdsMatrixGoldilocks, 12, 7>(c);
    poseidon2::<Goldilocks, MdsMatrixGoldilocks, 16, 7>(c);

    poseidon2::<Mersenne31, MdsMatrixMersenne31, 16, 5>(c);
    poseidon2::<Mersenne31, MdsMatrixMersenne31, 32, 5>(c);
}

fn poseidon2<F, Mds, const WIDTH: usize, const D: usize>(c: &mut Criterion)
where
    F: PrimeField64,
    Standard: Distribution<F>,
    Mds: MdsPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let mds = Mds::default();

    // TODO: Should be calculated for the particular field, width and ALPHA.
    let rounds_f = 8;
    let rounds_p = 22;

    let poseidon = Poseidon2::<F, Mds, WIDTH, D>::new_from_rng(rounds_f, rounds_p, mds, &mut rng);
    let input = [F::ZERO; WIDTH];
    let name = format!("poseidon2::<{}, {}>", type_name::<F>(), D);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
