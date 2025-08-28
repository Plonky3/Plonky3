use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_multilinear_util::point::MultilinearPoint;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type F = BabyBear;

fn expand_from_univariate_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("expand_from_univariate");

    // Benchmarking various sizes
    for num_variables in [4, 8, 16, 32, 64, 128].iter() {
        let mut rng = SmallRng::seed_from_u64(1);
        let point: F = rng.random();

        group.bench_with_input(
            BenchmarkId::new("expand", num_variables),
            num_variables,
            |b, &n| {
                b.iter(|| MultilinearPoint::expand_from_univariate(black_box(point), black_box(n)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, expand_from_univariate_benchmark,);
criterion_main!(benches);
