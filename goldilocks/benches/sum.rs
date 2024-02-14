use criterion::BenchmarkId;
use criterion::{criterion_group, criterion_main, Criterion};
use p3_field_testing::bench_func::{benchmark_sum};

use p3_goldilocks::{Goldilocks, sum_i128};

use rand::Rng;

type F = Goldilocks;

fn bench_sum(c: &mut Criterion) {
    let name = "Goldilocks";

    benchmark_sum::<F, 1>(c, name);
    sum_delayed::<1>(c);

    benchmark_sum::<F, 2>(c, name);
    sum_delayed::<2>(c);

    benchmark_sum::<F, 4>(c, name);
    sum_delayed::<4>(c);

    benchmark_sum::<F, 8>(c, name);
    sum_delayed::<8>(c);

    benchmark_sum::<F, 16>(c, name);
    sum_delayed::<16>(c);

    benchmark_sum::<F, 32>(c, name);
    sum_delayed::<32>(c);
}

fn sum_delayed<const N: usize>(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..N {
        input.push(rng.gen::<F>())
    }
    let id = BenchmarkId::new("Goldilocks sum_delayed", N);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| sum_i128(&input));
    });
}

criterion_group!(goldilocks_arithmetic, bench_sum);
criterion_main!(goldilocks_arithmetic);
