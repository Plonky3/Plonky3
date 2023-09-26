use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_mds::babybear::MdsMatrixBabyBear;
use p3_mds::goldilocks::MdsMatrixGoldilocks;
use p3_mds::integrated_coset_mds::IntegratedCosetMds;
use p3_mds::mersenne31::MdsMatrixMersenne31;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_rescue::{BasicSboxLayer, Rescue, SboxLayers};
use p3_symmetric::permutation::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_rescue(c: &mut Criterion) {
    rescue::<BabyBear, IntegratedCosetMds<_, 16>, BasicSboxLayer<_>, 16, 7>(c);
    rescue::<<BabyBear as Field>::Packing, IntegratedCosetMds<_, 16>, BasicSboxLayer<_>, 16, 7>(c);
    rescue::<BabyBear, MdsMatrixBabyBear, BasicSboxLayer<_>, 24, 7>(c);
    rescue::<BabyBear, MdsMatrixBabyBear, BasicSboxLayer<_>, 32, 7>(c);

    rescue::<Goldilocks, MdsMatrixGoldilocks, BasicSboxLayer<_>, 8, 7>(c);
    rescue::<Goldilocks, MdsMatrixGoldilocks, BasicSboxLayer<_>, 12, 7>(c);
    rescue::<Goldilocks, MdsMatrixGoldilocks, BasicSboxLayer<_>, 16, 7>(c);

    rescue::<Mersenne31, MdsMatrixMersenne31, BasicSboxLayer<_>, 16, 5>(c);
    rescue::<Mersenne31, MdsMatrixMersenne31, BasicSboxLayer<_>, 32, 5>(c);
}

fn rescue<F, Mds, Sbox, const WIDTH: usize, const ALPHA: u64>(c: &mut Criterion)
where
    F: AbstractField,
    F::F: PrimeField64,
    Standard: Distribution<F::F>,
    Mds: MdsPermutation<F, WIDTH> + Default,
    Sbox: SboxLayers<F, WIDTH> + Default,
{
    // 8 rounds seems to work for the configs we use in practice. For benchmarking purposes we will
    // assume it suffices; for real usage the Sage calculation in the paper should be used.
    const NUM_ROUNDS: usize = 8;

    let rng = thread_rng();
    let num_constants = 2 * WIDTH * NUM_ROUNDS;
    let round_constants = rng.sample_iter(Standard).take(num_constants).collect();
    let mds = Mds::default();
    let isl = Sbox::default();
    let rescue = Rescue::<F, Mds, Sbox, WIDTH>::new(NUM_ROUNDS, round_constants, mds, isl);
    let input = [F::ZERO; WIDTH];
    let name = format!("rescue::<{}, {}>", type_name::<F>(), ALPHA);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| rescue.permute(input.clone()))
    });
}

criterion_group!(benches, bench_rescue);
criterion_main!(benches);
