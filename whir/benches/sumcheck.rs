use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_whir::constraints::statement::initial::InitialStatement;
use p3_whir::parameters::SumcheckStrategy;
use p3_whir::sumcheck::SumcheckData;
use p3_whir::sumcheck::single::SingleSumcheck;
use p3_whir::sumcheck::svo::SvoClaim;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;

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

criterion_group!(benches, bench_sumcheck_prover, bench_svo_claim_build);
criterion_main!(benches);
