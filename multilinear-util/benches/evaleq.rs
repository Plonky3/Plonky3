use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_multilinear_util::eq::{eval_eq, eval_eq_base};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Base field type.
type F = BabyBear;

/// Extension field: a degree-4 binomial extension of the base field.
type EF4 = BinomialExtensionField<F, 4>;

/// Generates a random input for benchmarking:
/// - A vector of `n` extension field elements representing the evaluation point `z`
/// - A random extension field scalar `α`
fn generate_input(n: usize) -> (Vec<EF4>, EF4) {
    let mut rng = SmallRng::seed_from_u64(123456789);

    // Random evaluation point z ∈ EF4^n
    let eval = (0..n).map(|_| rng.random()).collect();

    // Random scalar α ∈ EF4
    let scalar = rng.random();

    (eval, scalar)
}

fn bench_eval_eq(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_eq");

    for &n in &[10, 15, 20, 25] {
        let (eval, scalar) = generate_input(n);

        // Output vector of size 2^n, initialized to zero
        let out = vec![EF4::ZERO; 1 << n];

        // Benchmark when the output buffer is uninitialized
        group.bench_with_input(BenchmarkId::new("eval_eq_false", n), &n, |b, _| {
            b.iter(|| eval_eq::<F, EF4, false>(&eval, &mut out.clone(), scalar))
        });

        // Benchmark when the output buffer is assumed to be initialized
        group.bench_with_input(BenchmarkId::new("eval_eq_true", n), &n, |b, _| {
            b.iter(|| eval_eq::<F, EF4, true>(&eval, &mut out.clone(), scalar))
        });
    }

    group.finish();
}

fn bench_eval_eq_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_eq_base");

    for &n in &[10, 15, 20, 25] {
        let (eval_ext, scalar) = generate_input(n);

        // Convert the extension field input to base field
        let eval_base: Vec<F> = eval_ext
            .iter()
            .map(|&x| x.as_basis_coefficients_slice()[0])
            .collect();

        // Output buffer
        let out = vec![EF4::ZERO; 1 << n];

        // Benchmark with INITIALIZED = false
        group.bench_with_input(BenchmarkId::new("eval_eq_base_false", n), &n, |b, _| {
            b.iter(|| eval_eq_base::<F, EF4, false>(&eval_base, &mut out.clone(), scalar))
        });

        // Benchmark with INITIALIZED = true
        group.bench_with_input(BenchmarkId::new("eval_eq_base_true", n), &n, |b, _| {
            b.iter(|| eval_eq_base::<F, EF4, true>(&eval_base, &mut out.clone(), scalar))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_eval_eq, bench_eval_eq_base);
criterion_main!(benches);
