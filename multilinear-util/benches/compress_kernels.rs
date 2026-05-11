//! Per-kernel benches for the split-eq compress and dot kernels.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

/// Deterministic RNG seeded from a bench label and a variable count.
///
/// Different shapes get different seeds so cache state of one bench
/// does not leak into another, while a single shape stays reproducible
/// across runs.
fn rng_for(label: &str, k: usize) -> SmallRng {
    // FNV-style mix: combine the variable count with each label byte.
    let mut h: u64 = k as u64;
    for b in label.bytes() {
        h = h.wrapping_mul(0x100000001b3).wrapping_add(b as u64);
    }
    SmallRng::seed_from_u64(h)
}

/// Suffix-variable dot product at production shapes.
///
/// # Role
///
/// - Regression guard for the SIMD kernel introduced in PR #1574.
/// - Comparison anchor for the new prefix-side kernels.
fn bench_compress_suffix_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_eq/compress_suffix_dot");

    // Variable counts that mirror the SvoClaim midpoint shapes.
    let cases = [(12usize, "k12"), (16, "k16"), (20, "k20")];

    for &(k_total, label) in &cases {
        // Build inputs deterministically so reruns are stable.
        let mut rng = rng_for(label, k_total);
        let poly = Poly::<F>::rand(&mut rng, k_total);
        let point = Point::<EF>::rand(&mut rng, k_total);

        // Pre-build the split eq table outside the timing loop:
        // construction cost is not part of the kernel we measure.
        let split_unpacked = SplitEq::<F, EF>::new_unpacked(&point, EF::ONE);
        let split_packed = SplitEq::<F, EF>::new_packed(&point, EF::ONE);

        // Packed-eq path: hits the SIMD kernel from PR #1574.
        group.bench_with_input(BenchmarkId::new("packed", label), &label, |b, _| {
            let mut out = EF::zero_vec(1 << (k_total - split_packed.num_variables()));
            b.iter(|| split_packed.compress_suffix_into(&mut out, &poly));
        });

        // Unpacked-eq path: scalar reference for the same operation.
        group.bench_with_input(BenchmarkId::new("unpacked", label), &label, |b, _| {
            let mut out = EF::zero_vec(1 << (k_total - split_unpacked.num_variables()));
            b.iter(|| split_unpacked.compress_suffix_into(&mut out, &poly));
        });
    }

    group.finish();
}

/// Prefix-variable fold into a scalar extension-field output.
///
/// Reaches the scalar-output prefix kernel; serves as a regression
/// guard for the unmerged Phase 3 work.
fn bench_compress_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_eq/compress_prefix");

    // Each tuple is (k_total, eq_k, label):
    //
    // - midpoint cases (k_total / 2): mirror SVO claim build.
    // - eq_k = 1 cases: mirror the per-round sumcheck shape.
    let cases = [
        (12usize, 6usize, "k12_eq6"),
        (16, 8, "k16_eq8"),
        (20, 10, "k20_eq10"),
        (16, 1, "k16_eq1"),
        (20, 1, "k20_eq1"),
    ];

    for &(k_total, eq_k, label) in &cases {
        let mut rng = rng_for(label, k_total);
        let poly = Poly::<F>::rand(&mut rng, k_total);
        let point = Point::<EF>::rand(&mut rng, eq_k);

        let split_packed = SplitEq::<F, EF>::new_packed(&point, EF::ONE);

        group.bench_with_input(BenchmarkId::new("packed", label), &label, |b, _| {
            b.iter(|| split_packed.compress_prefix(&poly));
        });
    }

    group.finish();
}

/// Prefix-variable fold into a packed extension-field output.
///
/// # Role
///
/// - Hits the SIMD kernel introduced in this PR (Phase 2 target).
/// - Production-shape caller in WHIR's per-round folding sumcheck.
fn bench_compress_prefix_to_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_eq/compress_prefix_to_packed");

    // Same shape grid as the scalar-output bench above. Every shape
    // here yields a packed output of at least one packed element.
    let cases = [
        (12usize, 6usize, "k12_eq6"),
        (16, 8, "k16_eq8"),
        (20, 10, "k20_eq10"),
        (16, 1, "k16_eq1"),
        (20, 1, "k20_eq1"),
    ];

    for &(k_total, eq_k, label) in &cases {
        let mut rng = rng_for(label, k_total);
        let poly = Poly::<F>::rand(&mut rng, k_total);
        let point = Point::<EF>::rand(&mut rng, eq_k);

        let split_packed = SplitEq::<F, EF>::new_packed(&point, EF::ONE);

        group.bench_with_input(BenchmarkId::new("packed", label), &label, |b, _| {
            b.iter(|| split_packed.compress_prefix_to_packed(&poly));
        });
    }

    group.finish();
}

/// Single-row dot product of the eq table against a base-field row.
///
/// Reaches the `dot_with_base` kernel through the SplitEq evaluation pipeline.
fn bench_dot_with_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_eq/dot_with_base");

    let cases = [(12usize, "k12"), (14, "k14"), (16, "k16")];

    for &(k_total, label) in &cases {
        let mut rng = rng_for(label, k_total);
        let poly = Poly::<F>::rand(&mut rng, k_total);
        let point = Point::<EF>::rand(&mut rng, k_total);

        let split_packed = SplitEq::<F, EF>::new_packed(&point, EF::ONE);

        group.bench_with_input(BenchmarkId::new("packed", label), &label, |b, _| {
            b.iter(|| split_packed.eval_base(&poly));
        });
    }

    group.finish();
}

/// Single-row dot product against a pre-packed extension-field row.
///
/// Both sides of the dot are packed, so this is purely SIMD work.
fn bench_dot_with_ext_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_eq/dot_with_ext_packed");

    let cases = [(12usize, "k12"), (14, "k14"), (16, "k16")];

    for &(k_total, label) in &cases {
        let mut rng = rng_for(label, k_total);
        let point = Point::<EF>::rand(&mut rng, k_total);
        // Build a packed polynomial of the right shape for the packed-eval API.
        let packed_poly = Poly::<EF>::rand(&mut rng, k_total).pack::<F, EF>();

        let split_packed = SplitEq::<F, EF>::new_packed(&point, EF::ONE);

        group.bench_with_input(BenchmarkId::new("packed", label), &label, |b, _| {
            b.iter(|| split_packed.eval_packed(&packed_poly));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compress_suffix_dot,
    bench_compress_prefix,
    bench_compress_prefix_to_packed,
    bench_dot_with_base,
    bench_dot_with_ext_packed,
);
criterion_main!(benches);
