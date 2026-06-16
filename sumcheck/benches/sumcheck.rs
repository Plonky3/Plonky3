//! Criterion benches for the quadratic sumcheck engine.
//!
//! The benches walk the protocol from the innermost kernel outward:
//!
//! - The per-round coefficient kernel that produces `(h(0), h(inf))`.
//! - The in-place pass that binds one variable per round.
//! - The running-sum consistency dot product.
//! - One complete protocol round end to end.
//! - The multi-round driver that folds every variable to a constant.
//! - The stacked-layout preprocessing handoff.
//!
//! Three axes are configurable:
//!
//! - Field: instantiated for two 31-bit primes through a small bundling trait.
//! - Binding order: every prover-shaped bench runs prefix-first and suffix-first.
//! - Sizes and folding: the `*_SIZES` and folding constants below.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{
    Algebra, ExtensionField, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField,
};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::constraints::statement::EqStatement;
use p3_sumcheck::constraints::{Constraint, Statements};
use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
use p3_sumcheck::product_polynomial::ProductPolynomial;
use p3_sumcheck::strategy::{
    SumcheckProver, VariableOrder, sumcheck_coefficients_prefix, sumcheck_coefficients_suffix,
};
use p3_sumcheck::{OpeningBatch, SumcheckData};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Variable counts for the single-pass packed kernels.
///
/// The kernel switches from serial to parallel at `2^14` paired elements.
///
/// The grid straddles that crossover and climbs to a production-scale `2^22`.
const KERNEL_SIZES: &[usize] = &[16, 18, 20, 22];

/// Variable counts for the scalar coefficient kernel.
///
/// Scalar arithmetic only runs once the polynomial has folded below the SIMD width.
///
/// The grid therefore stays small.
const SCALAR_SIZES: &[usize] = &[8, 10, 12];

/// Variable counts for the in-place binding pass.
const FIX_VAR_SIZES: &[usize] = &[16, 18, 20, 22];

/// Variable counts for the running-sum dot product.
const DOT_SIZES: &[usize] = &[16, 18, 20];

/// Variable counts for one complete protocol round.
const ROUND_SIZES: &[usize] = &[16, 18, 20];

/// Variable counts for the multi-round driver.
///
/// Each case folds all the way down to a constant, so the grid stays modest.
const PROVER_SIZES: &[usize] = &[12, 16, 20];

/// Variable counts for the stacked-layout preprocessing handoff.
const LAYOUT_SIZES: &[usize] = &[16, 18, 20];

/// Variables consumed by the layout's packed or accumulator-driven first phase.
///
/// Matches the folding depth used by the crate's roundtrip tests.
const FOLDING: usize = 4;

/// Binding orders paired with the label each one prints under.
const ORDERS: [(VariableOrder, &str); 2] = [
    (VariableOrder::Prefix, "prefix"),
    (VariableOrder::Suffix, "suffix"),
];

/// A base field, its extension, and a matching Fiat-Shamir transcript.
///
/// One implementor pins all three so the field-generic bench bodies have a
/// concrete transcript constructor to call.
trait BenchField: 'static {
    /// Base field carrying the committed evaluations.
    type F: TwoAdicField;
    /// Extension field used for challenges and accumulators.
    type EF: ExtensionField<Self::F>;
    /// Fiat-Shamir transcript paired with this field.
    type Challenger: FieldChallenger<Self::F> + GrindingChallenger<Witness = Self::F> + Clone;

    /// Short label embedded in every benchmark group name.
    const NAME: &'static str;

    /// Builds a deterministic transcript so reruns are reproducible.
    fn challenger() -> Self::Challenger;
}

/// BabyBear with its degree-4 binomial extension.
struct BabyBear4;

impl BenchField for BabyBear4 {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Challenger = DuplexChallenger<BabyBear, Poseidon2BabyBear<16>, 16, 8>;

    const NAME: &'static str = "babybear";

    fn challenger() -> Self::Challenger {
        // Fixed seed: the permutation, and thus every sampled challenge, is reproducible.
        let perm = Poseidon2BabyBear::new_from_rng_128(&mut SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }
}

/// KoalaBear with its degree-4 binomial extension.
struct KoalaBear4;

impl BenchField for KoalaBear4 {
    type F = KoalaBear;
    type EF = BinomialExtensionField<KoalaBear, 4>;
    type Challenger = DuplexChallenger<KoalaBear, Poseidon2KoalaBear<16>, 16, 8>;

    const NAME: &'static str = "koalabear";

    fn challenger() -> Self::Challenger {
        // Fixed seed: identical role to the BabyBear constructor above.
        let perm = Poseidon2KoalaBear::new_from_rng_128(&mut SmallRng::seed_from_u64(42));
        DuplexChallenger::new(perm)
    }
}

/// SIMD-packed base field for a chosen bundle.
type Packed<B> = <<B as BenchField>::F as Field>::Packing;
/// SIMD-packed extension field for a chosen bundle.
type ExtPacked<B> =
    <<B as BenchField>::EF as ExtensionField<<B as BenchField>::F>>::ExtensionPacking;

/// Deterministic generator keyed by a category tag and a variable count.
///
/// Distinct shapes get distinct streams, so one shape's cache state never
/// bleeds into another, while a single shape stays stable across reruns.
fn rng_for(tag: u64, k: usize) -> SmallRng {
    // Multiply the variable count by an odd constant before mixing, so adjacent
    // sizes land far apart in the seed space.
    SmallRng::seed_from_u64(tag ^ ((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

/// Random evaluation polynomial over the base field with `2^k` entries.
fn rand_base<B: BenchField>(rng: &mut SmallRng, k: usize) -> Poly<B::F>
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    // The field's own uniform generator fills all `2^k` hypercube points.
    Poly::<B::F>::rand(rng, k)
}

/// Random evaluation polynomial over the extension field with `2^k` entries.
fn rand_ext<B: BenchField>(rng: &mut SmallRng, k: usize) -> Poly<B::EF>
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    // Same uniform fill, one extension element per point.
    Poly::<B::EF>::rand(rng, k)
}

/// Random extension polynomial regrouped into SIMD lanes.
fn rand_ext_packed<B: BenchField>(rng: &mut SmallRng, k: usize) -> Poly<ExtPacked<B>>
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    // Build the scalar polynomial first, then fold consecutive elements into lanes.
    rand_ext::<B>(rng, k).pack::<B::F, B::EF>()
}

/// Packed product polynomial with independent random evals and weights.
fn rand_product_poly<B: BenchField>(
    order: VariableOrder,
    rng: &mut SmallRng,
    k: usize,
) -> ProductPolynomial<B::F, B::EF>
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    // The two sides are independent, mirroring a real evals-times-weights pair.
    let evals = rand_ext_packed::<B>(rng, k);
    let weights = rand_ext_packed::<B>(rng, k);
    ProductPolynomial::new_packed(order, evals, weights)
}

/// Runs one round-coefficient computation for the requested binding order.
///
/// # Arguments
///
/// - `order` — which hypercube axis the round sums over.
/// - `evals` — multilinear evaluations of the polynomial under sumcheck.
/// - `weights` — multilinear evaluations of the weight polynomial.
///
/// # Returns
///
/// The constant term and the leading coefficient of the round polynomial.
#[inline]
fn coeffs<Base, Acc>(order: VariableOrder, evals: &[Base], weights: &[Acc]) -> (Acc, Acc)
where
    Base: PrimeCharacteristicRing + Copy + Send + Sync,
    Acc: Algebra<Base> + Copy + Send + Sync,
{
    match order {
        // Prefix binding sums over the high half against the low half.
        VariableOrder::Prefix => sumcheck_coefficients_prefix(evals, weights),
        // Suffix binding sums over adjacent even/odd pairs.
        VariableOrder::Suffix => sumcheck_coefficients_suffix(evals, weights),
    }
}

/// Benches the per-round coefficient kernel in its three operand flavours.
///
/// - Round zero multiplies base-field evaluations against extension weights.
/// - Later rounds multiply two packed extension operands.
/// - Final rounds multiply two scalar extension operands.
fn bench_round_coefficients<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/round_coefficients", B::NAME));

    for &k in KERNEL_SIZES {
        // Report element-rate so the curve is comparable across sizes.
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0001, k);

        // Round-zero operands: base evaluations regrouped into lanes, extension weights.
        let base = rand_base::<B>(&mut rng, k);
        let base_packed = Packed::<B>::pack_slice(base.as_slice()).to_vec();
        let round_zero_weights = rand_ext_packed::<B>(&mut rng, k);
        for (order, name) in ORDERS {
            group.bench_with_input(
                BenchmarkId::new(format!("base_ext_{name}"), &label),
                &order,
                |b, &order| {
                    // Time the kernel alone; operands are prebuilt outside the loop.
                    b.iter(|| {
                        black_box(coeffs(order, &base_packed, round_zero_weights.as_slice()))
                    });
                },
            );
        }

        // Later-round operands: both sides are packed extension elements.
        let packed_evals = rand_ext_packed::<B>(&mut rng, k);
        let packed_weights = rand_ext_packed::<B>(&mut rng, k);
        for (order, name) in ORDERS {
            group.bench_with_input(
                BenchmarkId::new(format!("ext_ext_packed_{name}"), &label),
                &order,
                |b, &order| {
                    b.iter(|| {
                        black_box(coeffs(
                            order,
                            packed_evals.as_slice(),
                            packed_weights.as_slice(),
                        ))
                    });
                },
            );
        }
    }

    for &k in SCALAR_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0002, k);

        // Final-round operands: both sides are scalar extension elements.
        let evals = rand_ext::<B>(&mut rng, k);
        let weights = rand_ext::<B>(&mut rng, k);
        for (order, name) in ORDERS {
            group.bench_with_input(
                BenchmarkId::new(format!("ext_ext_scalar_{name}"), &label),
                &order,
                |b, &order| {
                    b.iter(|| black_box(coeffs(order, evals.as_slice(), weights.as_slice())));
                },
            );
        }
    }

    group.finish();
}

/// Benches the in-place pass that binds one variable per round.
///
/// The buffer is overwritten each call, so a clone is restored before every
/// timed iteration.
fn bench_fix_var<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/fix_var", B::NAME));

    for &k in FIX_VAR_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0003, k);

        // Pristine buffer cloned per iteration, plus the challenge it folds against.
        let template = rand_ext_packed::<B>(&mut rng, k);
        let r: B::EF = rng.random();

        for (order, name) in ORDERS {
            group.bench_with_input(BenchmarkId::new(name, &label), &order, |b, &order| {
                b.iter_batched_ref(
                    // Setup (untimed): restore the buffer mutated by the previous run.
                    || template.clone(),
                    // Routine (timed): fold the active variable into the buffer.
                    |poly| {
                        order.fix_var(poly, black_box(r));
                        black_box(&*poly);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

/// Benches the dot product backing the running-sum consistency check.
fn bench_dot_product<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/dot_product", B::NAME));

    for &k in DOT_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0004, k);

        // The binding order does not affect this sum, so one representative suffices.
        let poly = rand_product_poly::<B>(VariableOrder::Prefix, &mut rng, k);

        group.bench_with_input(BenchmarkId::from_parameter(&label), &k, |b, _| {
            b.iter(|| black_box(poly.dot_product()));
        });
    }

    group.finish();
}

/// Benches one complete protocol round.
///
/// A round computes the coefficients, absorbs them into the transcript, samples
/// a challenge, folds both sides, and updates the running sum.
fn bench_product_round<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/product_round", B::NAME));

    for &k in ROUND_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0005, k);

        for (order, name) in ORDERS {
            // Prebuilt round inputs: the polynomial, its matching sum, and a transcript.
            let poly = rand_product_poly::<B>(order, &mut rng, k);
            let sum = poly.dot_product();
            let challenger = B::challenger();

            group.bench_with_input(BenchmarkId::new(name, &label), &order, |b, _| {
                b.iter_batched(
                    // Setup (untimed): a round mutates all three inputs, so clone them.
                    || (poly.clone(), sum, challenger.clone()),
                    // Routine (timed): drive exactly one round with grinding disabled.
                    |(mut poly, mut sum, mut challenger)| {
                        let mut data = SumcheckData::<B::F, B::EF>::default();
                        let r = poly.round(&mut data, &mut challenger, &mut sum, 0);
                        black_box((r, sum, data));
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

/// Builds a random equality constraint over `k` variables.
///
/// The evaluations are arbitrary: only the `2^k`-sized weight-table build inside
/// the absorb path is being measured, and its cost is value-independent.
fn rand_eq_constraint<B: BenchField>(rng: &mut SmallRng, k: usize) -> Constraint<B::F, B::EF>
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    // Empty group sized to the current round's variable count.
    let mut eq = EqStatement::<B::EF>::initialize(k);

    // Two equality points: each records a claimed value at a random point.
    for _ in 0..2 {
        eq.add_evaluated_constraint(Point::<B::EF>::rand(rng, k), rng.random());
    }

    // Bundle the group under a random batching challenge.
    Constraint::new(rng.random(), k, vec![Statements::Eq(eq)])
}

/// Benches the multi-round driver folding every variable to a constant.
///
/// Two arms isolate the cost of absorbing a constraint:
///
/// - One drives rounds only.
/// - One folds a constraint into the weights before the rounds.
fn bench_prover<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/prover", B::NAME));

    // The full fold is the heaviest bench, so keep the sample count low.
    group.sample_size(10);

    for &k in PROVER_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0006, k);

        for (order, name) in ORDERS {
            // Prover seeded with a sum that matches its polynomial pair.
            let poly = rand_product_poly::<B>(order, &mut rng, k);
            let prover = SumcheckProver::new(poly.clone(), poly.dot_product());
            let challenger = B::challenger();

            // Rounds only: fold all `k` variables with no constraint absorbed.
            group.bench_with_input(
                BenchmarkId::new(format!("plain_{name}"), &label),
                &k,
                |b, &k| {
                    b.iter_batched(
                        // Setup (untimed): the driver consumes the prover and transcript.
                        || (prover.clone(), challenger.clone()),
                        // Routine (timed): one batch of `k` rounds, grinding disabled.
                        |(mut prover, mut challenger)| {
                            let mut data = SumcheckData::<B::F, B::EF>::default();
                            let r = prover.compute_sumcheck_polynomials(
                                &mut data,
                                &mut challenger,
                                k,
                                0,
                                None,
                            );
                            black_box((r, data));
                        },
                        BatchSize::LargeInput,
                    );
                },
            );

            // Same drive, but fold one constraint into the weights first.
            let constraint = rand_eq_constraint::<B>(&mut rng, k);
            group.bench_with_input(
                BenchmarkId::new(format!("with_constraint_{name}"), &label),
                &k,
                |b, &k| {
                    b.iter_batched(
                        // Setup (untimed): clone the prover, transcript, and constraint.
                        || (prover.clone(), challenger.clone(), constraint.clone()),
                        // Routine (timed): absorb the constraint, then fold `k` rounds.
                        |(mut prover, mut challenger, constraint)| {
                            let mut data = SumcheckData::<B::F, B::EF>::default();
                            let r = prover.compute_sumcheck_polynomials(
                                &mut data,
                                &mut challenger,
                                k,
                                0,
                                Some(constraint),
                            );
                            black_box((r, data));
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Benches the stacked-layout preprocessing handoff for both prover modes.
///
/// - Prefix-first binding runs the first rounds in SIMD-packed arithmetic.
/// - Suffix-first binding runs them off precomputed small-value accumulators.
///
/// Witness construction and the opening draw sit in untimed setup, so only the
/// folding-depth handoff is measured.
fn bench_layout<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/layout", B::NAME));
    group.sample_size(10);

    for &k in LAYOUT_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0007, k);

        // Single source table, one column of `2^k` base-field evaluations.
        let table = Table::new(vec![rand_base::<B>(&mut rng, k)]);

        group.bench_with_input(BenchmarkId::new("prefix", &label), &table, |b, table| {
            b.iter_batched(
                // Setup (untimed): build the prover and record one opening claim.
                || setup_layout::<B, PrefixProver<B::F, B::EF>>(table),
                // Routine (timed): consume the prover through the handoff.
                |(prover, mut challenger)| run_into_sumcheck::<B, _>(prover, &mut challenger),
                BatchSize::LargeInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("suffix", &label), &table, |b, table| {
            b.iter_batched(
                || setup_layout::<B, SuffixProver<B::F, B::EF>>(table),
                |(prover, mut challenger)| run_into_sumcheck::<B, _>(prover, &mut challenger),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Builds a single-table layout prover and records one direct opening.
fn setup_layout<B, L>(table: &Table<B::F>) -> (L, B::Challenger)
where
    B: BenchField,
    L: Layout<B::F, B::EF>,
{
    // Commit the cloned table at the shared folding depth.
    let witness = L::new_witness(vec![table.clone()], FOLDING);
    let mut prover = L::from_witness(witness);
    let mut challenger = B::challenger();

    // Open the single column directly, matching the protocol's claim setup.
    prover.eval(0, &OpeningBatch::new(vec![0], Vec::new()), &mut challenger);
    (prover, challenger)
}

/// Drives the preprocessing handoff and discards the result through a barrier.
fn run_into_sumcheck<B, L>(prover: L, challenger: &mut B::Challenger)
where
    B: BenchField,
    L: Layout<B::F, B::EF>,
{
    // Fresh proof buffer; grinding disabled to isolate the folding cost.
    let mut data = SumcheckData::<B::F, B::EF>::default();

    // Consume the prover, yielding the residual prover and the sampled challenges.
    let (residual, randomness) = prover.into_sumcheck(&mut data, 0, challenger);
    black_box((data, residual, randomness));
}

/// Coefficient kernel for every field.
fn round_coefficients(c: &mut Criterion) {
    bench_round_coefficients::<BabyBear4>(c);
    bench_round_coefficients::<KoalaBear4>(c);
}

/// Binding pass for every field.
fn fix_var(c: &mut Criterion) {
    bench_fix_var::<BabyBear4>(c);
    bench_fix_var::<KoalaBear4>(c);
}

/// Running-sum dot product for every field.
fn dot_product(c: &mut Criterion) {
    bench_dot_product::<BabyBear4>(c);
    bench_dot_product::<KoalaBear4>(c);
}

/// One complete round for every field.
fn product_round(c: &mut Criterion) {
    bench_product_round::<BabyBear4>(c);
    bench_product_round::<KoalaBear4>(c);
}

/// Multi-round driver for every field.
fn prover(c: &mut Criterion) {
    bench_prover::<BabyBear4>(c);
    bench_prover::<KoalaBear4>(c);
}

/// Stacked-layout handoff for every field.
fn layout(c: &mut Criterion) {
    bench_layout::<BabyBear4>(c);
    bench_layout::<KoalaBear4>(c);
}

criterion_group!(
    benches,
    round_coefficients,
    fix_var,
    dot_product,
    product_round,
    prover,
    layout,
);
criterion_main!(benches);
