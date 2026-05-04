use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_whir::constraints::statement::initial::InitialStatement;
use p3_whir::parameters::SumcheckStrategy;
use p3_whir::sumcheck::SumcheckData;
use p3_whir::sumcheck::single::SingleSumcheck;
use p3_whir::sumcheck::strategy::sumcheck_coefficients_prefix;
use p3_whir::sumcheck::svo::SvoClaim;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type FP = <F as Field>::Packing;
type EFPacked = <EF as ExtensionField<F>>::ExtensionPacking;

fn make_challenger() -> Challenger {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(42));
    DuplexChallenger::new(perm)
}

/// Build an initial statement with `num_constraints` random evaluation points.
fn make_statement(
    num_variables: usize,
    folding_factor: usize,
    num_constraints: usize,
    mode: SumcheckStrategy,
) -> InitialStatement<F, EF> {
    let mut rng = SmallRng::seed_from_u64(
        (num_variables as u64) ^ ((folding_factor as u64) << 16) ^ ((num_constraints as u64) << 32),
    );
    let poly = Poly::new((0..1 << num_variables).map(|_| rng.random()).collect());
    let mut stmt = InitialStatement::<F, EF>::new(poly, folding_factor, mode);
    for _ in 0..num_constraints {
        let pt = Point::<EF>::rand(&mut rng, num_variables);
        let _ = stmt.evaluate(&pt);
    }
    stmt
}

/// Random packed input pair sized to `2^num_variables` evaluations.
///
/// Mirrors the layout consumed by the round-coefficient kernel inside
/// `ProductPolynomial::round`:
///
/// ```text
///     evals_packed   : &[F::Packing]              base-field
///     weights_packed : &[EF::ExtensionPacking]    extension-field accumulator
/// ```
fn make_packed_inputs(num_variables: usize) -> (Vec<FP>, Vec<EFPacked>) {
    // Per-size deterministic seed so successive bench runs are reproducible.
    let mut rng = SmallRng::seed_from_u64(0xc0ffee ^ num_variables as u64);
    let n = 1 << num_variables;

    // Scalar buffers first; the packing layer demands a contiguous slice.
    let evals: Vec<F> = (0..n).map(|_| F::from_u32(rng.random::<u32>())).collect();
    let weights: Vec<EF> = (0..n).map(|_| rng.random()).collect();

    // Repack base-field evaluations into SIMD lanes of width FP::WIDTH.
    let evals_packed = FP::pack_slice(&evals).to_vec();
    // Repack extension weights similarly: one EFPacked per WIDTH consecutive elements.
    let weights_packed = weights
        .chunks(FP::WIDTH)
        .map(EFPacked::from_ext_slice)
        .collect();

    (evals_packed, weights_packed)
}

/// End-to-end sumcheck prover: Classic vs SVO.
fn bench_sumcheck_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir/sumcheck_prover");

    let cases = [
        (14, 4, 4, "small"),
        (18, 4, 4, "medium"),
        (22, 4, 4, "large"),
    ];

    for &(num_variables, folding_factor, num_constraints, label) in &cases {
        let classic_stmt = make_statement(
            num_variables,
            folding_factor,
            num_constraints,
            SumcheckStrategy::Classic,
        );
        let svo_stmt = make_statement(
            num_variables,
            folding_factor,
            num_constraints,
            SumcheckStrategy::Svo,
        );

        group.bench_with_input(BenchmarkId::new("classic", label), &label, |b, _| {
            b.iter(|| {
                let mut data = SumcheckData::default();
                let mut challenger = make_challenger();
                SingleSumcheck::new(&mut data, &mut challenger, folding_factor, 0, &classic_stmt)
            });
        });

        group.bench_with_input(BenchmarkId::new("svo", label), &label, |b, _| {
            b.iter(|| {
                let mut data = SumcheckData::default();
                let mut challenger = make_challenger();
                SingleSumcheck::new(&mut data, &mut challenger, folding_factor, 0, &svo_stmt)
            });
        });
    }

    group.finish();
}

/// SVO claim building: this is where the grid-expansion optimization applies.
fn bench_svo_claim_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir/svo_claim_build");

    // l=1,2,3 are the straightline-specialized cases; l=4 is the general-path control.
    //
    // k12_* shapes:  N in 1..4, below the SIMD path on NEON (W = 4).
    // k16_l4, k18_l4, k20_l4: production-shape N = 16 / 32 / 64.
    //   These exercise the SIMD path on every supported width.
    let cases = [
        (12, 1, "k12_l1"),
        (12, 2, "k12_l2"),
        (12, 3, "k12_l3"),
        (12, 4, "k12_l4"),
        (16, 4, "k16_l4"),
        (18, 4, "k18_l4"),
        (20, 4, "k20_l4"),
    ];

    for &(num_variables, l, label) in &cases {
        let mut rng = SmallRng::seed_from_u64(((num_variables as u64) << 32) | l as u64);
        let poly = Poly::new((0..1 << num_variables).map(|_| rng.random()).collect());
        let point = Point::<EF>::rand(&mut rng, num_variables);

        group.bench_with_input(BenchmarkId::new("claim_build", label), &label, |b, _| {
            b.iter(|| SvoClaim::<F, EF>::new(&point, l, &poly));
        });
    }

    group.finish();
}

/// Round-coefficient kernel measured in isolation.
///
/// # Scope
///
/// Excludes:
///
/// - challenger sampling
/// - transcript bookkeeping
/// - equality polynomial construction
/// - binding
fn bench_round_coefficients(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir/kernel");
    let n = 20;
    let (evals, weights) = make_packed_inputs(n);

    // Prefix-binding: split halves of the buffer feed (acc0, acc_inf).
    group.bench_with_input(
        BenchmarkId::new("round_coefficients_prefix", n),
        &n,
        |b, _| {
            b.iter(|| {
                let (c0, c_inf) =
                    sumcheck_coefficients_prefix(black_box(&evals), black_box(&weights));
                black_box((c0, c_inf))
            });
        },
    );

    group.finish();
}

/// In-place binding kernel measured in isolation.
///
/// # Operation
///
/// Each round of sumcheck binds the active variable by overwriting the
/// first half of the buffer with a linear interpolation against the
/// verifier challenge `r`:
///
/// ```text
///     p[i] <- p[i] + (p[i + mid] - p[i]) * r       for i in 0..mid
/// ```
///
/// where `mid = len / 2`. The pass runs twice per round in the prover —
/// once for the evaluation polynomial and once for the weight polynomial.
fn bench_fix_prefix_var_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir/kernel");
    let n = 20;
    let (_, weights) = make_packed_inputs(n);

    // Scalar challenge; broadcast onto the packed lanes inside the kernel.
    let mut rng = SmallRng::seed_from_u64(0xdeadbeef ^ n as u64);
    let r: EF = rng.random();
    let template = Poly::<EFPacked>::new(weights);

    group.bench_with_input(BenchmarkId::new("fix_prefix_var_mut", n), &n, |b, _| {
        // Buffer is mutated in place; a fresh clone per iteration restores it.
        b.iter_batched_ref(
            || template.clone(),
            |poly| {
                poly.fix_prefix_var_mut(black_box(r));
                black_box(&*poly);
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sumcheck_prover,
    bench_svo_claim_build,
    bench_round_coefficients,
    bench_fix_prefix_var_mut,
);
criterion_main!(benches);
