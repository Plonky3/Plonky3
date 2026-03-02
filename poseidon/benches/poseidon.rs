use core::any::type_name;
use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::{Algebra, InjectiveMonomial, PrimeField};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_poseidon::{Poseidon, PoseidonExternalLayerGeneric, PoseidonInternalLayerGeneric};
use p3_symmetric::Permutation;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn bench_poseidon(c: &mut Criterion) {
    poseidon::<BabyBear, BabyBear, MdsMatrixBabyBear, 16, 7>(c);
    poseidon::<BabyBear, BabyBear, MdsMatrixBabyBear, 24, 7>(c);

    poseidon::<Goldilocks, Goldilocks, MdsMatrixGoldilocks, 8, 7>(c);
    poseidon::<Goldilocks, Goldilocks, MdsMatrixGoldilocks, 12, 7>(c);
    poseidon::<Goldilocks, Goldilocks, MdsMatrixGoldilocks, 16, 7>(c);

    poseidon::<Mersenne31, Mersenne31, MdsMatrixMersenne31, 16, 5>(c);
    poseidon::<Mersenne31, Mersenne31, MdsMatrixMersenne31, 32, 5>(c);
}

fn poseidon<F, A, Mds, const WIDTH: usize, const ALPHA: u64>(c: &mut Criterion)
where
    F: PrimeField + InjectiveMonomial<ALPHA>,
    A: Algebra<F> + Sync + InjectiveMonomial<ALPHA>,
    StandardUniform: Distribution<F>,
    Mds: Permutation<[F; WIDTH]> + Permutation<[A; WIDTH]> + Default + Sync + Clone,
{
    let mut rng = SmallRng::seed_from_u64(1);

    // TODO: Should be calculated for the particular field, width and ALPHA.
    let half_num_full_rounds = 4;
    let num_partial_rounds = 22;

    type External<F, Mds, const W: usize> = PoseidonExternalLayerGeneric<F, Mds, W>;
    type Internal<F, const W: usize> = PoseidonInternalLayerGeneric<F, W>;

    let poseidon =
        Poseidon::<F, External<F, Mds, WIDTH>, Internal<F, WIDTH>, WIDTH, ALPHA>::new_from_rng(
            half_num_full_rounds,
            num_partial_rounds,
            &Mds::default(),
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
