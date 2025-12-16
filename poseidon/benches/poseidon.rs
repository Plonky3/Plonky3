use core::any::type_name;
use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::{Algebra, Field, InjectiveMonomial, PrimeField64};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_mds::MdsPermutation;
use p3_mds::coset_mds::CosetMds;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_poseidon::{poseidon_round_numbers_128, Poseidon};
use p3_symmetric::Permutation;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn bench_poseidon(c: &mut Criterion) {
    poseidon::<BabyBear, BabyBear, MdsMatrixBabyBear, 16, 7>(c);
    poseidon::<BabyBear, BabyBear, MdsMatrixBabyBear, 24, 7>(c);
    poseidon::<BabyBear, BabyBear, CosetMds<_, 32>, 32, 7>(c);
    poseidon::<BabyBear, <BabyBear as Field>::Packing, CosetMds<BabyBear, 32>, 32, 7>(c);

    poseidon::<Goldilocks, Goldilocks, MdsMatrixGoldilocks, 8, 7>(c);
    poseidon::<Goldilocks, Goldilocks, MdsMatrixGoldilocks, 12, 7>(c);
    poseidon::<Goldilocks, Goldilocks, MdsMatrixGoldilocks, 16, 7>(c);

    poseidon::<Mersenne31, Mersenne31, MdsMatrixMersenne31, 16, 5>(c);
    poseidon::<Mersenne31, Mersenne31, MdsMatrixMersenne31, 32, 5>(c);
}

fn poseidon<F, A, Mds, const WIDTH: usize, const ALPHA: u64>(c: &mut Criterion)
where
    F: PrimeField64 + InjectiveMonomial<ALPHA>,
    A: Algebra<F> + InjectiveMonomial<ALPHA>,
    StandardUniform: Distribution<F>,
    Mds: MdsPermutation<A, WIDTH> + Default,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let mds = Mds::default();

    // Calculate round numbers for the particular field, width and ALPHA.
    let (half_num_full_rounds, num_partial_rounds) = poseidon_round_numbers_128::<F>(WIDTH, ALPHA)
        .unwrap_or_else(|e| {
            panic!("Failed to get round numbers for width={}, alpha={}: {}", WIDTH, ALPHA, e)
        });

    let poseidon = Poseidon::<F, Mds, WIDTH, ALPHA>::new_from_rng(
        half_num_full_rounds,
        num_partial_rounds,
        mds,
        &mut rng,
    );
    let input: [A; WIDTH] = array::from_fn(|_| A::ZERO);
    let name = format!("poseidon::<{}, {}>", type_name::<A>(), ALPHA);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| poseidon.permute(input.clone()));
    });
}

criterion_group!(benches, bench_poseidon);
criterion_main!(benches);
