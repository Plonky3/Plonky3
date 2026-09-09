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
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{
    Algebra, ExtensionField, Field, PackedValue, PrimeCharacteristicRing, TwoAdicField,
};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::constraints::statement::{EqStatement, SelectStatement};
use p3_sumcheck::constraints::{Constraint, Statements};
use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
use p3_sumcheck::product_polynomial::ProductPolynomial;
use p3_sumcheck::strategy::{
    RoundMessage, SumcheckProver, VariableOrder, sumcheck_coefficients_prefix,
    sumcheck_coefficients_prefix_projective, sumcheck_coefficients_suffix,
};
use p3_sumcheck::zk::ZkSumcheckData;
use p3_sumcheck::{OpeningBatch, SumcheckData};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_zk_codes::reed_solomon::ReedSolomonZkEncoding;
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

/// Variable counts for the one-round-at-a-time drive.
///
/// The binary PCS asks for a single round between each codeword fold and commitment.
///
/// The grid spans the whole ladder it drives, from a table below the parallel
/// threshold up to a production-scale one.
const SINGLE_ROUND_SIZES: &[usize] = &[8, 12, 16, 20];

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

/// Product polynomial with independent random evals and weights.
///
/// The pair is handed to `new_packed`, but only prefix binding actually keeps
/// packed storage: `ProductPolynomial` unpacks a suffix-bound pair up front,
/// because the suffix variable lives inside the SIMD lanes. The suffix rows of
/// the prover benches therefore measure the scalar kernel.
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
fn coeffs<Base, Acc>(order: VariableOrder, evals: &[Base], weights: &[Acc]) -> RoundMessage<Acc>
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
///
/// The two packed flavours time prefix binding only. `ProductPolynomial` serves
/// suffix binding from scalar storage, so a suffix round never meets a packed
/// operand; the scalar flavour is the one that carries both orders.
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
        group.bench_with_input(
            BenchmarkId::new("base_ext_prefix", &label),
            &label,
            |b, _| {
                // Time the kernel alone; operands are prebuilt outside the loop.
                b.iter(|| {
                    black_box(coeffs(
                        VariableOrder::Prefix,
                        &base_packed,
                        round_zero_weights.as_slice(),
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("base_ext_prefix_projective", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    black_box(sumcheck_coefficients_prefix_projective(
                        &base_packed,
                        round_zero_weights.as_slice(),
                    ))
                });
            },
        );

        // Later-round operands: both sides are packed extension elements.
        let packed_evals = rand_ext_packed::<B>(&mut rng, k);
        let packed_weights = rand_ext_packed::<B>(&mut rng, k);
        group.bench_with_input(
            BenchmarkId::new("ext_ext_packed_prefix", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    black_box(coeffs(
                        VariableOrder::Prefix,
                        packed_evals.as_slice(),
                        packed_weights.as_slice(),
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ext_ext_packed_prefix_projective", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    black_box(sumcheck_coefficients_prefix_projective(
                        packed_evals.as_slice(),
                        packed_weights.as_slice(),
                    ))
                });
            },
        );
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

        group.bench_with_input(
            BenchmarkId::new("ext_ext_scalar_prefix_projective", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    black_box(sumcheck_coefficients_prefix_projective(
                        evals.as_slice(),
                        weights.as_slice(),
                    ))
                });
            },
        );
    }

    group.finish();
}

/// Benches the in-place pass that binds one variable per round.
///
/// The buffer is overwritten each call, so a clone is restored before every
/// timed iteration.
///
/// The buffer is packed, so prefix is the only order timed here: a suffix round
/// binds a variable that lives inside the SIMD lanes, which `ProductPolynomial`
/// serves from scalar storage instead.
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

        group.bench_with_input(BenchmarkId::new("prefix", &label), &label, |b, _| {
            b.iter_batched_ref(
                // Setup (untimed): restore the buffer mutated by the previous run.
                || template.clone(),
                // Routine (timed): fold the active variable into the buffer.
                |poly| {
                    VariableOrder::Prefix.fix_var(poly, black_box(r));
                    black_box(&*poly);
                },
                BatchSize::LargeInput,
            );
        });

        // Projective (monomial-basis) binding: the subtraction-free
        // `a0 + a1 * r` of eprint 2026/762, prefix only.
        //
        // This is where the projective basis is expected to pay. The
        // round-coefficient kernel only trades a subtraction pass for an
        // addition pass, so it cannot come out ahead; binding drops a pass.
        group.bench_with_input(
            BenchmarkId::new("prefix_projective", &label),
            &(),
            |b, ()| {
                b.iter_batched_ref(
                    || template.clone(),
                    |poly| {
                        poly.fix_prefix_var_mut_monomial(black_box(r));
                        black_box(&*poly);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
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

/// Benches folding a large `SelectStatement` (STIR-query-shaped) into an
/// unpacked weight accumulator, isolated from any round-folding cost.
fn bench_combine<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/combine", B::NAME));
    group.sample_size(10);

    let k = 20;
    let n = 80;
    let mut rng = rng_for(0x000C, k);

    let evals = rand_ext::<B>(&mut rng, k);
    let weights = Poly::<B::EF>::new(vec![B::EF::ZERO; 1 << k]);
    let poly =
        ProductPolynomial::<B::F, B::EF>::new_unpacked(VariableOrder::Prefix, evals, weights);

    let mut statement = SelectStatement::<B::F, B::EF>::initialize(k);
    for _ in 0..n {
        statement.add_constraint(rng.random(), rng.random());
    }
    let constraint = Constraint::new(rng.random(), k, vec![Statements::Select(statement)]);

    group.throughput(Throughput::Elements(1 << k));
    group.bench_function("select", |b| {
        b.iter_batched(
            || (poly.clone(), B::EF::ZERO, constraint.clone()),
            |(mut poly, mut sum, constraint)| {
                poly.combine(&mut sum, &constraint);
                black_box((poly, sum));
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
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

/// Benches a sumcheck driven one round per call, fused against binding each round.
///
/// This is the shape the binary PCS uses: it interleaves a codeword fold and a Merkle
/// commitment between rounds, so it can never ask for several rounds at once.
///
/// Two arms over the identical ladder:
///
/// ```text
///     fused           : the challenge is held and the next round's pass absorbs it
///     bind_each_round : the challenge is applied on the spot, so the next round
///                       measures in a pass of its own
/// ```
///
/// The difference between them is one pass over both tables per round.
fn bench_single_round_drive<B: BenchField>(c: &mut Criterion)
where
    StandardUniform: Distribution<B::F> + Distribution<B::EF>,
{
    let mut group = c.benchmark_group(format!("sumcheck/{}/single_round_drive", B::NAME));

    // A full ladder per iteration, so keep the sample count low.
    group.sample_size(10);

    for &k in SINGLE_ROUND_SIZES {
        group.throughput(Throughput::Elements(1 << k));
        let label = format!("k{k}");
        let mut rng = rng_for(0x0007, k);

        for (order, name) in ORDERS {
            // Prover seeded with a sum that matches its polynomial pair.
            let poly = rand_product_poly::<B>(order, &mut rng, k);
            let prover = SumcheckProver::new(poly.clone(), poly.dot_product());
            let challenger = B::challenger();

            // Fused arm: every round after the first absorbs the previous challenge.
            group.bench_with_input(
                BenchmarkId::new(format!("fused_{name}"), &label),
                &k,
                |b, &k| {
                    b.iter_batched(
                        || (prover.clone(), challenger.clone()),
                        |(mut prover, mut challenger)| {
                            let mut data = SumcheckData::<B::F, B::EF>::default();
                            // One round per call, exactly as the binary PCS asks for them.
                            for _ in 0..k {
                                let r = prover.compute_sumcheck_polynomials(
                                    &mut data,
                                    &mut challenger,
                                    1,
                                    0,
                                    None,
                                );
                                black_box(r);
                            }
                            black_box(data);
                        },
                        BatchSize::LargeInput,
                    );
                },
            );

            // Reference arm: settling after each round leaves the next one a plain measure.
            group.bench_with_input(
                BenchmarkId::new(format!("bind_each_round_{name}"), &label),
                &k,
                |b, &k| {
                    b.iter_batched(
                        || (prover.clone(), challenger.clone()),
                        |(mut prover, mut challenger)| {
                            let mut data = SumcheckData::<B::F, B::EF>::default();
                            for _ in 0..k {
                                let r = prover.compute_sumcheck_polynomials(
                                    &mut data,
                                    &mut challenger,
                                    1,
                                    0,
                                    None,
                                );
                                // Applying the challenge now is the second pass per round.
                                prover.settle();
                                black_box(r);
                            }
                            black_box(data);
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
        let table = Table::rand(&mut rng, 1, k);

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

/// Variable count for the hiding residual driver.
///
/// Matches the largest plain-prover case, so the two curves are read side by side.
const ZK_RESIDUAL_SIZE: usize = 20;

/// Message length of the mask code.
///
/// Section 2.7 of eprint 2026/391 sizes masks at `O(lambda / log log lambda)`.
/// Sixteen is the value the hiding WHIR benches use, so the mask cost is realistic.
const ZK_ELL: usize = 16;

/// Randomness symbols appended to each mask before encoding.
const ZK_T: usize = 2;

/// Benches the hiding residual driver folding one batch of rounds.
///
/// The hiding prover reaches this driver once per WHIR round.
/// Its plain counterpart is the multi-round driver benched above.
fn bench_zk_residual(c: &mut Criterion) {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type BaseMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
    type Mmcs = ExtensionMmcs<F, EF, BaseMmcs>;
    type Dft = Radix2DFTSmallBatch<EF>;
    type Enc = ReedSolomonZkEncoding<EF, Dft>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    let mut group = c.benchmark_group("sumcheck/babybear/zk_residual");

    // The hiding path commits a mask oracle per batch, so keep the sample count low.
    group.sample_size(10);

    let mut rng = rng_for(0x000D, ZK_RESIDUAL_SIZE);
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(42));

    // Mask commitment scheme and code, sized as the hiding WHIR prover sizes them.
    let mmcs = Mmcs::new(BaseMmcs::new(
        Hash::new(perm.clone()),
        Compress::new(perm.clone()),
        0,
    ));
    let encoding = Enc::new(
        ZK_T,
        ZK_ELL,
        (ZK_ELL + ZK_T).next_power_of_two(),
        Dft::default(),
    );

    // The pair the driver folds, with the claim it starts from.
    let evals = Poly::<EF>::rand(&mut rng, ZK_RESIDUAL_SIZE);
    let weights = Poly::<EF>::rand(&mut rng, ZK_RESIDUAL_SIZE);
    let poly = ProductPolynomial::<F, EF>::new_packed(
        VariableOrder::Prefix,
        evals.pack::<F, EF>(),
        weights.pack::<F, EF>(),
    );
    let sum = poly.dot_product();

    group.throughput(Throughput::Elements(1 << ZK_RESIDUAL_SIZE));
    group.bench_function(
        BenchmarkId::from_parameter(format!("k{ZK_RESIDUAL_SIZE}")),
        |b| {
            b.iter_batched(
                // Setup (untimed): the driver consumes the prover and the transcript.
                || {
                    (
                        SumcheckProver::new(poly.clone(), sum),
                        Challenger::new(perm.clone()),
                        SmallRng::seed_from_u64(7),
                    )
                },
                // Routine (timed): one batch of rounds, grinding disabled.
                |(prover, mut challenger, mut mask_rng)| {
                    let mut data = ZkSumcheckData::<F, EF>::default();
                    let handoff = prover.into_zk_sumcheck(
                        &mut data,
                        &encoding,
                        &mmcs,
                        FOLDING,
                        0,
                        EF::ZERO,
                        &mut challenger,
                        &mut mask_rng,
                    );
                    black_box((data, handoff));
                },
                BatchSize::LargeInput,
            );
        },
    );

    group.finish();
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

/// Select-statement combine for every field.
fn combine(c: &mut Criterion) {
    bench_combine::<BabyBear4>(c);
    bench_combine::<KoalaBear4>(c);
}

/// Stacked-layout handoff for every field.
fn layout(c: &mut Criterion) {
    bench_layout::<BabyBear4>(c);
    bench_layout::<KoalaBear4>(c);
}

fn single_round_drive(c: &mut Criterion) {
    bench_single_round_drive::<BabyBear4>(c);
    bench_single_round_drive::<KoalaBear4>(c);
}

criterion_group!(
    benches,
    round_coefficients,
    fix_var,
    dot_product,
    product_round,
    prover,
    single_round_drive,
    combine,
    layout,
    bench_zk_residual,
);
criterion_main!(benches);
