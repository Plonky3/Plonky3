use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
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

fn criterion_benchmark(c: &mut Criterion) {
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
