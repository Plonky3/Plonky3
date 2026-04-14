//! Benchmarks for barycentric Lagrange interpolation over two-adic cosets.
//!
//! Measures three interpolation tiers at increasing matrix sizes:
//!
//! ```text
//!   Tier 1 (full)        — builds coset + batch-inverts + evaluates.
//!   Tier 2 (precomputed) — caller supplies 1/(z - x_i), internally adjusts.
//!   Tier 3 (adjusted)    — caller supplies adjusted weights, zero inner alloc.
//! ```
//!
//! Tier 3 is the FRI PCS prover hot path.

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

// Narrow  → per-element weight work dominates.
// Balanced → weights and dot product share the cost.
// Wide    → SIMD dot product dominates, weight cost is noise.
const CONFIGS: &[(usize, usize)] = &[(10, 1), (14, 16), (18, 128)];

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
    // Tier 2: inverse denominators precomputed by the caller.
    // Timed: weight adjustment + dot product + scaling.
    let mut group = c.benchmark_group("interpolate_coset_with_precomputation");
    group.sample_size(20);

    let shift = F::GENERATOR;

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(0);
        let m = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let point = EF::from_u32(123456789);

        // Setup (not timed): build coset and 1/(z - x_i) via batch inversion.
        let subgroup_gen = F::two_adic_generator(log_rows);
        let coset: Vec<F> = subgroup_gen.shifted_powers(shift).collect_n(rows);
        let diffs: Vec<EF> = coset.iter().map(|&g| point - g).collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);

        let param = format!("2^{log_rows}x{width}");
        group.bench_with_input(BenchmarkId::new("precomputed", &param), &(), |b, _| {
            b.iter(|| m.interpolate_coset_with_precomputation(shift, point, &coset, &diff_invs));
        });
    }

    group.finish();
}

fn interpolate_coset_with_adjusted_weights(c: &mut Criterion) {
    // Tier 3: adjusted weights precomputed by the caller.
    // Timed: dot product + scaling only (zero inner allocation).
    let mut group = c.benchmark_group("interpolate_coset_with_adjusted_weights");
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
        group.bench_with_input(BenchmarkId::new("adjusted", &param), &(), |b, _| {
            b.iter(|| m.interpolate_coset_with_adjusted_weights(shift, point, &adjusted));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    interpolate_coset,
    interpolate_coset_with_precomputation,
    interpolate_coset_with_adjusted_weights
);
criterion_main!(benches);
