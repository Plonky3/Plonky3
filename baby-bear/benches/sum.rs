use p3_field_testing::benchmark_sum;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, sum_i64};
use rand::Rng;

type F = BabyBear;

fn bench_sum(c: &mut Criterion) {
    let name = "BabyBear";

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
    let id = BenchmarkId::new("BabyBear sum_delayed", N);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| sum_i64(&input));
    });
}

criterion_group!(baby_bear_arithmetic, bench_sum);
criterion_main!(baby_bear_arithmetic);
