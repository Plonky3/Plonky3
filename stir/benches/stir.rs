//! Benchmarks for the STIR polynomial commitment scheme.
//!
//! Covers commit (codeword evaluation + Merkle commit), prove, and verify phases
//! for polynomial degrees 2^14 through 2^22 over BabyBear with its quartic extension.
//!
//! # Benchmark parameters
//!
//! `max_pow_bits = 0` is used throughout so benchmarks measure the cryptographic core
//! (DFTs, Merkle trees, Lagrange folds, shake polynomial arithmetic) without PoW
//! grinding noise.  Real deployments add PoW overhead on top of these numbers.
//!
//! Three configurations are benchmarked:
//!   - `fold1`: arity 2 per round (k=2); no rate improvement, FRI-like behaviour
//!   - `fold2`: arity 4 per round (k=4); rate improves by 1 bit per round
//!   - `fold3`: arity 8 per round (k=8); rate improves by 2 bits per round (fewest queries)

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, SecurityAssumption};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::config::{StirConfig, StirParameters};
use p3_stir::prover::prove_stir;
use p3_stir::verifier::verify_stir;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type MyMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

fn make_rng() -> SmallRng {
    SmallRng::seed_from_u64(0xdeadbeef)
}

/// Build a STIR environment (params, DFT, challenger template) without PoW.
fn make_stir_env(log_folding_factor: usize) -> (StirParameters<MyMmcs>, Dft, Challenger) {
    let mut rng = make_rng();
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let mmcs = MyMmcs::new(val_mmcs);

    let params = StirParameters {
        log_blowup: 1,
        log_folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        // Use a reasonable security level, disable PoW.
        security_level: 100,
        max_pow_bits: 0,
        mmcs,
    };
    (params, Dft::default(), Challenger::new(perm))
}

/// Generate a random polynomial in coefficient form of degree `2^log_degree`.
fn random_poly<EF>(log_degree: usize) -> Vec<EF>
where
    EF: Copy,
    StandardUniform: Distribution<EF>,
{
    let mut rng = make_rng();
    let degree = 1usize << log_degree;
    (0..degree).map(|_| rng.random::<EF>()).collect()
}

/// Benchmark the full prove path for each degree.
fn bench_prove<F, EF, M, D, C>(
    c: &mut Criterion,
    params: &StirParameters<M>,
    dft: &D,
    challenger_template: &C,
    log_degrees: &[usize],
    group_name: &str,
) where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF> + Clone,
    D: TwoAdicSubgroupDft<F>,
    C: FieldChallenger<F> + CanObserve<M::Commitment> + GrindingChallenger<Witness = F> + Clone,
    StandardUniform: Distribution<EF>,
{
    let mut group = c.benchmark_group(format!("{group_name}/prove"));
    group.sample_size(10);

    for &log_degree in log_degrees {
        let poly = random_poly::<EF>(log_degree);
        let config = StirConfig::<F, EF, M, C>::new(log_degree, params.clone());

        group.bench_with_input(
            BenchmarkId::from_parameter(1usize << log_degree),
            &log_degree,
            |b, _| {
                b.iter(|| {
                    let mut ch = challenger_template.clone();
                    prove_stir(&config, poly.clone(), dft, &mut ch)
                });
            },
        );
    }
}

/// Benchmark the verify path only.
fn bench_verify<F, EF, M, D, C>(
    c: &mut Criterion,
    params: &StirParameters<M>,
    dft: &D,
    challenger_template: &C,
    log_degrees: &[usize],
    group_name: &str,
) where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF> + Clone,
    D: TwoAdicSubgroupDft<F>,
    C: FieldChallenger<F> + CanObserve<M::Commitment> + GrindingChallenger<Witness = F> + Clone,
    StandardUniform: Distribution<EF>,
{
    let mut group = c.benchmark_group(format!("{group_name}/verify"));
    group.sample_size(10);

    for &log_degree in log_degrees {
        let poly = random_poly::<EF>(log_degree);
        let config = StirConfig::<F, EF, M, C>::new(log_degree, params.clone());

        group.bench_with_input(
            BenchmarkId::from_parameter(1usize << log_degree),
            &log_degree,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut p_ch = challenger_template.clone();
                        prove_stir(&config, poly.clone(), dft, &mut p_ch)
                    },
                    |proof| {
                        let mut v_ch = challenger_template.clone();
                        verify_stir::<F, EF, M, C>(&config, &proof, &mut v_ch)
                            .expect("verification failed in benchmark");
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

fn bench_stir_babybear_fold1(c: &mut Criterion) {
    let log_degrees: Vec<usize> = (14..=22).collect();
    let (params, dft, challenger) = make_stir_env(1);

    bench_prove::<Val, Challenge, MyMmcs, Dft, Challenger>(
        c,
        &params,
        &dft,
        &challenger,
        &log_degrees,
        "stir_babybear_blowup1_fold1",
    );
    bench_verify::<Val, Challenge, MyMmcs, Dft, Challenger>(
        c,
        &params,
        &dft,
        &challenger,
        &log_degrees,
        "stir_babybear_blowup1_fold1",
    );
}

fn bench_stir_babybear_fold2(c: &mut Criterion) {
    let log_degrees: Vec<usize> = (14..=22).collect();
    let (params, dft, challenger) = make_stir_env(2);

    bench_prove::<Val, Challenge, MyMmcs, Dft, Challenger>(
        c,
        &params,
        &dft,
        &challenger,
        &log_degrees,
        "stir_babybear_blowup1_fold2",
    );
    bench_verify::<Val, Challenge, MyMmcs, Dft, Challenger>(
        c,
        &params,
        &dft,
        &challenger,
        &log_degrees,
        "stir_babybear_blowup1_fold2",
    );
}

fn bench_stir_babybear_fold3(c: &mut Criterion) {
    let log_degrees: Vec<usize> = (14..=22).collect();
    let (params, dft, challenger) = make_stir_env(3);

    bench_prove::<Val, Challenge, MyMmcs, Dft, Challenger>(
        c,
        &params,
        &dft,
        &challenger,
        &log_degrees,
        "stir_babybear_blowup1_fold3",
    );
    bench_verify::<Val, Challenge, MyMmcs, Dft, Challenger>(
        c,
        &params,
        &dft,
        &challenger,
        &log_degrees,
        "stir_babybear_blowup1_fold3",
    );
}

criterion_group!(
    benches,
    bench_stir_babybear_fold1,
    bench_stir_babybear_fold2,
    bench_stir_babybear_fold3,
);
criterion_main!(benches);
