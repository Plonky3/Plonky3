use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_multilinear_util::point::MultilinearPoint;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type F = BabyBear;

fn generate_random_test_case(num_variables: usize) -> (MultilinearPoint<F>, usize) {
    let mut rng = SmallRng::seed_from_u64(1);
    let point = MultilinearPoint::<F>::rand(&mut rng, num_variables);
    let ternary_point_val = rng.random_range(0..3usize.pow(num_variables as u32));
    (point, ternary_point_val)
}

fn eq_poly3_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("eq_poly3_comparison");

    // Define the input sizes to test
    let num_variables_sizes = vec![5, 10, 15, 20, 25];

    for num_variables in num_variables_sizes {
        let (point, value) = generate_random_test_case(num_variables);
        let name_suffix = format!("_{num_variables}vars");

        group.bench_function(format!("eq_poly3{name_suffix}"), move |b| {
            b.iter(|| black_box(point.eq_poly3(black_box(value))));
        });
    }

    group.finish();
}

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

criterion_group!(
    benches,
    eq_poly3_benchmark,
    expand_from_univariate_benchmark,
);
criterion_main!(benches);
