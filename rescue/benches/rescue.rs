use std::any::type_name;
use std::array;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::{AbstractField, Field, PrimeField64};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_mds::integrated_coset_mds::IntegratedCosetMds;
use p3_mds::MdsPermutation;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_rescue::{BasicSboxLayer, Rescue};
use p3_symmetric::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_rescue(c: &mut Criterion) {
    rescue::<BabyBear, IntegratedCosetMds<_, 16>, 16, 7>(c);
    rescue::<<BabyBear as Field>::Packing, IntegratedCosetMds<_, 16>, 16, 7>(c);
    rescue::<BabyBear, MdsMatrixBabyBear, 24, 7>(c);
    rescue::<BabyBear, MdsMatrixBabyBear, 32, 7>(c);

    rescue::<Goldilocks, MdsMatrixGoldilocks, 8, 7>(c);
    rescue::<Goldilocks, MdsMatrixGoldilocks, 12, 7>(c);
    rescue::<Goldilocks, MdsMatrixGoldilocks, 16, 7>(c);

    rescue::<Mersenne31, MdsMatrixMersenne31, 16, 5>(c);
    rescue::<Mersenne31, MdsMatrixMersenne31, 32, 5>(c);
}

fn rescue<AF, Mds, const WIDTH: usize, const ALPHA: u64>(c: &mut Criterion)
where
    AF: AbstractField,
    AF::F: PrimeField64,
    Standard: Distribution<AF::F>,
    Mds: MdsPermutation<AF, WIDTH> + Default,
{
    // 8 rounds seems to work for the configs we use in practice. For benchmarking purposes we will
    // assume it suffices; for real usage the Sage calculation in the paper should be used.
    const NUM_ROUNDS: usize = 8;

    let rng = thread_rng();
    let num_constants = 2 * WIDTH * NUM_ROUNDS;
    let round_constants = rng.sample_iter(Standard).take(num_constants).collect();
    let mds = Mds::default();
    let sbox = BasicSboxLayer::for_alpha(ALPHA);
    let rescue = Rescue::<AF::F, Mds, _, WIDTH>::new(NUM_ROUNDS, round_constants, mds, sbox);
    let input: [AF; WIDTH] = array::from_fn(|_| AF::zero());
    let name = format!("rescue::<{}, {}>", type_name::<AF>(), ALPHA);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| rescue.permute(input.clone()))
    });
}

criterion_group!(benches, bench_rescue);
criterion_main!(benches);
