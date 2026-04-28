//! Benchmarks for barycentric Lagrange interpolation over two-adic cosets.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

// Cover all four corners (tall/short) × (narrow/wide) plus a balanced midpoint.
//
// This way, each shape's bottleneck can be observed independently:
//   tall narrow    → per-element weight work dominates
//   short wide     → SIMD dot product dominates
//   tall wide      → both axes large, stresses the full pipeline
//   short narrow   → both axes small, baseline
//   balanced       → weights and dot product share the cost
const CONFIGS: &[(usize, usize)] = &[
    (10, 1),   // short, narrow
    (10, 128), // short, wide
    (18, 1),   // tall, narrow
    (18, 128), // tall, wide
    (14, 16),  // balanced
];

fn interpolate_coset(c: &mut Criterion) {
    // Tier 1: full end-to-end.
    // Timed: coset enumeration + batch inversion + evaluation.
    let mut group = c.benchmark_group("interpolate_coset");
    group.sample_size(20);

    let shift = F::GENERATOR;

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(0);
        let m = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let point = EF::from_u32(123456789);

        let param = format!("2^{log_rows}x{width}");
        group.bench_with_input(BenchmarkId::new("full", &param), &(), |b, _| {
            b.iter(|| m.interpolate_coset(shift, point));
        });
    }

    group.finish();
}

fn interpolate_coset_with_precomputation(c: &mut Criterion) {
    // Tier 2: adjusted weights precomputed by the caller.
    // Timed: dot product + scaling only (zero inner allocation).
    let mut group = c.benchmark_group("interpolate_coset_with_precomputation");
    group.sample_size(20);

    let shift = F::GENERATOR;

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(0);
        let m = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let point = EF::from_u32(123456789);

        // Setup (not timed): build coset, batch-invert, subtract z^{-1}.
        let subgroup_gen = F::two_adic_generator(log_rows);
        let coset: Vec<F> = subgroup_gen.shifted_powers(shift).collect_n(rows);
        let diffs: Vec<EF> = coset.iter().map(|&g| point - g).collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);
        let adjusted = compute_adjusted_weights(point, &diff_invs);

        let param = format!("2^{log_rows}x{width}");
        group.bench_with_input(BenchmarkId::new("precomputed", &param), &(), |b, _| {
            b.iter(|| m.interpolate_coset_with_precomputation(shift, point, &adjusted));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    interpolate_coset,
    interpolate_coset_with_precomputation,
);
criterion_main!(benches);
