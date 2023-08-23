use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_field::PrimeField64;
use p3_mds::mersenne31::MdsMatrixMersenne31;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_poseidon::Poseidon;
use p3_symmetric::permutation::CryptographicPermutation;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_poseidon(c: &mut Criterion) {
    poseidon::<Mersenne31, MdsMatrixMersenne31, 12, 5>(c);
}

fn poseidon<F, Mds, const WIDTH: usize, const ALPHA: u64>(c: &mut Criterion)
where
    F: PrimeField64,
    Standard: Distribution<F>,
    Mds: MdsPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let mds = Mds::default();

    // TODO: Should be field specific
    let half_num_full_rounds = 4;
    let num_partial_rounds = 22;

    let poseidon = Poseidon::<F, Mds, WIDTH, ALPHA>::new_from_rng(
        half_num_full_rounds,
        num_partial_rounds,
        mds,
        &mut rng,
    );
    let input = [F::ZERO; WIDTH];
    let name = format!("poseidon::<{}, {}>", type_name::<F>(), ALPHA,);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

criterion_group!(benches, bench_poseidon);
criterion_main!(benches);
