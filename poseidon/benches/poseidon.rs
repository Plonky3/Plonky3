use std::any::type_name;
use std::array;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::{AbstractField, Field, PrimeField};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_mds::coset_mds::CosetMds;
use p3_mds::MdsPermutation;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_poseidon::Poseidon;
use p3_symmetric::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_poseidon(c: &mut Criterion) {
    poseidon::<BabyBear, MdsMatrixBabyBear, 16, 7>(c);
    poseidon::<BabyBear, MdsMatrixBabyBear, 24, 7>(c);
    poseidon::<BabyBear, CosetMds<_, 32>, 32, 7>(c);
    poseidon::<<BabyBear as Field>::Packing, CosetMds<_, 32>, 32, 7>(c);

    poseidon::<Goldilocks, MdsMatrixGoldilocks, 8, 7>(c);
    poseidon::<Goldilocks, MdsMatrixGoldilocks, 12, 7>(c);
    poseidon::<Goldilocks, MdsMatrixGoldilocks, 16, 7>(c);

    poseidon::<Mersenne31, MdsMatrixMersenne31, 16, 5>(c);
    poseidon::<Mersenne31, MdsMatrixMersenne31, 32, 5>(c);
}

fn poseidon<AF, Mds, const WIDTH: usize, const ALPHA: u64>(c: &mut Criterion)
where
    AF: AbstractField,
    AF::F: PrimeField,
    Standard: Distribution<AF::F>,
    Mds: MdsPermutation<AF, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let mds = Mds::default();

    // TODO: Should be calculated for the particular field, width and ALPHA.
    let half_num_full_rounds = 4;
    let num_partial_rounds = 22;

    let poseidon = Poseidon::<AF::F, Mds, WIDTH, ALPHA>::new_from_rng(
        half_num_full_rounds,
        num_partial_rounds,
        mds,
        &mut rng,
    );
    let input: [AF; WIDTH] = array::from_fn(|_| AF::zero());
    let name = format!("poseidon::<{}, {}>", type_name::<AF>(), ALPHA);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| poseidon.permute(input.clone()))
    });
}

criterion_group!(benches, bench_poseidon);
criterion_main!(benches);
