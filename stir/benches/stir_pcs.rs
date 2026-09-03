//! Benchmarks for [`TwoAdicStirPcs`] across LDE-height buckets.
//!
//! Matrices sharing an LDE height form one bucket. `max_pow_bits` is set high so grind-sharing
//! dominates `open()`.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::QuinticTrinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::{SecurityAssumption, StirParameters, TwoAdicStirPcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type Val = KoalaBear;
type Challenge = QuinticTrinomialExtensionField<Val>;
type Perm = Poseidon2KoalaBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicStirPcs<Val, Dft, ValMmcs, ChallengeMmcs, Challenge, Challenger>;

/// Columns per committed matrix.
const WIDTH: usize = 4;

/// Bucket layouts under test: log-degrees of the matrices in one commitment.
const LAYOUTS: &[(&str, &[usize])] = &[
    ("1bucket", &[18]),
    ("2buckets_ratio2", &[18, 17]),
    ("2buckets_ratio4", &[18, 16]),
    ("3buckets_ratio4", &[18, 17, 16]),
];

fn seeded_rng() -> SmallRng {
    SmallRng::seed_from_u64(0xdeadbeef)
}

fn make_pcs() -> (MyPcs, Challenger) {
    let perm = Perm::new_from_rng_128(&mut seeded_rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let stir_params = StirParameters {
        log_blowup: 1,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type: SecurityAssumption::CapacityBound,
        security_level: 100,
        max_pow_bits: 20,
        mmcs: challenge_mmcs,
    };

    (
        MyPcs::new(Dft::default(), val_mmcs, stir_params),
        Challenger::new(perm),
    )
}

type DomainsAndPolys = Vec<(
    <MyPcs as Pcs<Challenge, Challenger>>::Domain,
    RowMajorMatrix<Val>,
)>;

fn make_inputs(pcs: &MyPcs, log_degrees: &[usize]) -> DomainsAndPolys {
    let mut rng = seeded_rng();
    log_degrees
        .iter()
        .map(|&log_d| {
            let d = 1 << log_d;
            (
                <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(pcs, d),
                RowMajorMatrix::<Val>::rand(&mut rng, d, WIDTH),
            )
        })
        .collect()
}

fn bench_commit(c: &mut Criterion) {
    let (pcs, _) = make_pcs();
    let mut group = c.benchmark_group("stir_pcs/commit");
    group.sample_size(10);
    for &(name, log_degrees) in LAYOUTS {
        let inputs = make_inputs(&pcs, log_degrees);
        group.bench_with_input(BenchmarkId::from_parameter(name), &inputs, |b, inputs| {
            b.iter_batched(
                || inputs.clone(),
                |inputs| <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, inputs),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let (pcs, challenger) = make_pcs();
    let mut group = c.benchmark_group("stir_pcs/open");
    group.sample_size(10);
    for &(name, log_degrees) in LAYOUTS {
        let inputs = make_inputs(&pcs, log_degrees);
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, inputs.iter().cloned());
        let mut base = challenger.clone();
        commit.iter().for_each(|root| base.observe(root.clone()));
        let zeta: Challenge = base.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = inputs.iter().map(|_| vec![zeta]).collect();

        let mut nonce_rng = seeded_rng();
        group.bench_with_input(BenchmarkId::from_parameter(name), &points, |b, points| {
            b.iter_batched(
                || {
                    let mut challenger = base.clone();
                    let nonce: Val = nonce_rng.random();
                    challenger.observe(nonce);
                    (challenger, points.clone())
                },
                |(mut challenger, points)| {
                    <MyPcs as Pcs<Challenge, Challenger>>::open(
                        &pcs,
                        vec![(&data, points)],
                        &mut challenger,
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let (pcs, challenger) = make_pcs();
    let mut group = c.benchmark_group("stir_pcs/verify");
    group.sample_size(10);
    for &(name, log_degrees) in LAYOUTS {
        let inputs = make_inputs(&pcs, log_degrees);
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, inputs.iter().cloned());
        let mut p_challenger = challenger.clone();
        commit
            .iter()
            .for_each(|root| p_challenger.observe(root.clone()));
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = inputs.iter().map(|_| vec![zeta]).collect();
        let (opened, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![(&data, points)],
            &mut p_challenger,
        );

        let claims: Vec<_> = inputs
            .iter()
            .zip(opened.first().unwrap().iter())
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect();
        let proof_bytes = postcard::to_allocvec(&proof).unwrap().len();
        eprintln!("stir_pcs/proof_size/{name}: {proof_bytes} bytes");

        // Same transcript state `open` reached: commitment observed, opening point drawn.
        let mut v_base = challenger.clone();
        commit.iter().for_each(|root| v_base.observe(root.clone()));
        let _: Challenge = v_base.sample_algebra_element();

        group.bench_with_input(BenchmarkId::from_parameter(name), &claims, |b, claims| {
            b.iter_batched(
                || v_base.clone(),
                |mut challenger| {
                    <MyPcs as Pcs<Challenge, Challenger>>::verify(
                        &pcs,
                        vec![(commit.clone(), claims.clone())],
                        &proof,
                        &mut challenger,
                    )
                    .expect("verification failed in benchmark");
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_commit, bench_open, bench_verify);
criterion_main!(benches);
