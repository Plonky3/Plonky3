use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_multilinear_util::poly::Poly;
use p3_whir::sumcheck::SumcheckData;
use p3_whir::sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
use p3_whir::sumcheck::strategy::sumcheck_coefficients_prefix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type FP = <F as Field>::Packing;
type EFPacked = <EF as ExtensionField<F>>::ExtensionPacking;

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

fn make_challenger() -> Challenger {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(42));
    DuplexChallenger::new(perm)
}

fn make_table(num_variables: usize) -> Table<F> {
    let mut rng = SmallRng::seed_from_u64(num_variables as u64);
    Table::new(vec![Poly::new(
        (0..1 << num_variables).map(|_| rng.random()).collect(),
    )])
}

fn setup_prefix(table: &Table<F>, folding: usize) -> (PrefixProver<F, EF>, Challenger) {
    let witness = PrefixProver::<F, EF>::new_witness(vec![table.clone()], folding);
    let mut prover = PrefixProver::<F, EF>::from_witness(witness);
    let mut challenger = make_challenger();
    let evals = prover.eval(0, &[0], &mut challenger);
    assert_eq!(evals.len(), 1);
    (prover, challenger)
}

fn setup_suffix(table: &Table<F>, folding: usize) -> (SuffixProver<F, EF>, Challenger) {
    let witness = SuffixProver::<F, EF>::new_witness(vec![table.clone()], folding);
    let mut prover = SuffixProver::<F, EF>::from_witness(witness);
    let mut challenger = make_challenger();
    let evals = prover.eval(0, &[0], &mut challenger);
    assert_eq!(evals.len(), 1);
    (prover, challenger)
}

fn run_sumcheck<L: Layout<F, EF>>(prover: L, challenger: &mut Challenger, folding: usize) {
    let mut data = SumcheckData::default();
    let (residual, randomness) = prover.into_sumcheck(&mut data, 0, challenger);
    assert_eq!(data.num_rounds(), folding);
    assert_eq!(randomness.num_variables(), folding);
    assert!(residual.num_variables() > 0);
    black_box((data, residual, randomness));
}

fn bench_sumcheck_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir/layout_sumcheck");

    let cases = [(14, 4, "log14"), (18, 4, "log18"), (22, 4, "log22")];

    for &(num_variables, folding, label) in &cases {
        let table = make_table(num_variables);

        group.bench_with_input(BenchmarkId::new("prefix", label), &table, |b, table| {
            b.iter_batched(
                || setup_prefix(table, folding),
                |(prover, mut challenger)| run_sumcheck(prover, &mut challenger, folding),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("suffix", label), &table, |b, table| {
            b.iter_batched(
                || setup_suffix(table, folding),
                |(prover, mut challenger)| run_sumcheck(prover, &mut challenger, folding),
                BatchSize::SmallInput,
            );
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
    bench_round_coefficients,
    bench_fix_prefix_var_mut,
);
criterion_main!(benches);
