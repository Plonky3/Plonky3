use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrixView;
use p3_multilinear_util::eq::{eval_eq, eval_eq_base};
use p3_multilinear_util::eq_batch::{eval_eq_base_batch, eval_eq_batch};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Base field type.
type F = BabyBear;

/// Extension field: a degree-4 binomial extension of the base field.
type EF4 = BinomialExtensionField<F, 4>;

/// Generates random batch inputs for benchmarking:
/// - A matrix of evaluation points: rows are variables, columns are batch points
/// - A vector of random extension field scalars, one for each batch point
fn generate_batch_input(num_vars: usize, batch_size: usize) -> (Vec<EF4>, Vec<EF4>) {
    let mut rng = SmallRng::seed_from_u64(987654321);

    // Generate batch_size random evaluation points, each with num_vars coordinates
    let mut eval_points_data = Vec::with_capacity(num_vars * batch_size);
    for _var_idx in 0..num_vars {
        for _point_idx in 0..batch_size {
            eval_points_data.push(rng.random());
        }
    }

    // Generate batch_size random scalars
    let scalars: Vec<EF4> = (0..batch_size).map(|_| rng.random()).collect();

    (eval_points_data, scalars)
}

/// Generate base field batch inputs (for eval_eq_base_batch benchmarking)
fn generate_base_batch_input(num_vars: usize, batch_size: usize) -> (Vec<F>, Vec<EF4>) {
    let mut rng = SmallRng::seed_from_u64(987654321);

    // Generate batch_size random base field evaluation points
    let mut eval_points_data = Vec::with_capacity(num_vars * batch_size);
    for _var_idx in 0..num_vars {
        for _point_idx in 0..batch_size {
            eval_points_data.push(rng.random());
        }
    }

    // Generate batch_size random extension field scalars
    let scalars: Vec<EF4> = (0..batch_size).map(|_| rng.random()).collect();

    (eval_points_data, scalars)
}

fn bench_eval_eq_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_eq_batch");

    // Test different combinations of variable count and batch size
    let test_cases = [
        // (num_vars, batch_size)
        (10, 1),
        (10, 4),
        (10, 16),
        (10, 64),
        (15, 1),
        (15, 4),
        (15, 16),
        (15, 64),
        (20, 1),
        (20, 4),
        (20, 16),
        (20, 64),
    ];

    for &(num_vars, batch_size) in &test_cases {
        let (eval_points_data, scalars) = generate_batch_input(num_vars, batch_size);
        let eval_points = RowMajorMatrixView::new(&eval_points_data, batch_size);
        let out = vec![EF4::ZERO; 1 << num_vars];

        let bench_name = format!("n{}_b{}", num_vars, batch_size);

        // Benchmark when output buffer is uninitialized
        group.bench_with_input(
            BenchmarkId::new("eval_eq_batch_false", bench_name.clone()),
            &(num_vars, batch_size),
            |b, _| {
                b.iter(|| eval_eq_batch::<F, EF4, false>(eval_points, &scalars, &mut out.clone()))
            },
        );

        // Benchmark when output buffer is initialized (accumulation mode)
        group.bench_with_input(
            BenchmarkId::new("eval_eq_batch_true", bench_name),
            &(num_vars, batch_size),
            |b, _| {
                b.iter(|| eval_eq_batch::<F, EF4, true>(eval_points, &scalars, &mut out.clone()))
            },
        );
    }

    group.finish();
}

fn bench_eval_eq_base_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_eq_base_batch");

    // Test different combinations optimized for base field operations
    let test_cases = [
        // (num_vars, batch_size)
        (10, 1),
        (10, 4),
        (10, 16),
        (10, 64),
        (15, 1),
        (15, 4),
        (15, 16),
        (15, 64),
        (20, 1),
        (20, 4),
        (20, 16),
        (20, 64),
    ];

    for &(num_vars, batch_size) in &test_cases {
        let (eval_points_data, scalars) = generate_base_batch_input(num_vars, batch_size);
        let eval_points = RowMajorMatrixView::new(&eval_points_data, batch_size);
        let out = vec![EF4::ZERO; 1 << num_vars];

        let bench_name = format!("n{}_b{}", num_vars, batch_size);

        // Benchmark the optimized base field batched evaluation
        group.bench_with_input(
            BenchmarkId::new("eval_eq_base_batch_false", bench_name.clone()),
            &(num_vars, batch_size),
            |b, _| {
                b.iter(|| {
                    eval_eq_base_batch::<F, EF4, false>(eval_points, &scalars, &mut out.clone())
                })
            },
        );

        // Benchmark with accumulation
        group.bench_with_input(
            BenchmarkId::new("eval_eq_base_batch_true", bench_name),
            &(num_vars, batch_size),
            |b, _| {
                b.iter(|| {
                    eval_eq_base_batch::<F, EF4, true>(eval_points, &scalars, &mut out.clone())
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_vs_single_extension(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_single_extension");

    // Test with different variable counts to see how the overhead scales
    let variable_counts = [20];

    for &num_vars in &variable_counts {
        let mut rng = SmallRng::seed_from_u64(987654321);

        // Generate single evaluation point for both approaches
        let single_eval_point: Vec<EF4> = (0..num_vars).map(|_| rng.random()).collect();
        let single_scalar: EF4 = rng.random();

        // Create batch format (1x1 matrix) for batch function
        let batch_eval_data = single_eval_point.clone();
        let batch_eval_points = RowMajorMatrixView::new(&batch_eval_data, 1);
        let batch_scalars = vec![single_scalar];

        let out = vec![EF4::ZERO; 1 << num_vars];

        // Benchmark single eval_eq
        group.bench_with_input(BenchmarkId::new("single", num_vars), &num_vars, |b, _| {
            b.iter(|| eval_eq::<F, EF4, false>(&single_eval_point, &mut out.clone(), single_scalar))
        });

        // Benchmark batch eval_eq_batch with batch_size = 1
        group.bench_with_input(
            BenchmarkId::new("batch_size_1", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    eval_eq_batch::<F, EF4, false>(
                        batch_eval_points,
                        &batch_scalars,
                        &mut out.clone(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_vs_single_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_single_base");

    // Test with different variable counts to see how the overhead scales
    let variable_counts = [20];

    for &num_vars in &variable_counts {
        let mut rng = SmallRng::seed_from_u64(987654321);

        // Generate single base field evaluation point
        let single_eval_point: Vec<F> = (0..num_vars).map(|_| rng.random()).collect();
        let single_scalar: EF4 = rng.random();

        // Create batch format (1x1 matrix) for batch function
        let batch_eval_data = single_eval_point.clone();
        let batch_eval_points = RowMajorMatrixView::new(&batch_eval_data, 1);
        let batch_scalars = vec![single_scalar];

        let out = vec![EF4::ZERO; 1 << num_vars];

        // Benchmark single eval_eq_base
        group.bench_with_input(BenchmarkId::new("single", num_vars), &num_vars, |b, _| {
            b.iter(|| {
                eval_eq_base::<F, EF4, false>(&single_eval_point, &mut out.clone(), single_scalar)
            })
        });

        // Benchmark batch eval_eq_base_batch with batch_size = 1
        group.bench_with_input(
            BenchmarkId::new("batch_size_1", num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    eval_eq_base_batch::<F, EF4, false>(
                        batch_eval_points,
                        &batch_scalars,
                        &mut out.clone(),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    // bench_eval_eq_batch,
    // bench_eval_eq_base_batch,
    bench_batch_vs_single_extension,
    bench_batch_vs_single_base
);
criterion_main!(benches);
