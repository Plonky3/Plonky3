//! End-to-end benchmarks for the WHIR polynomial commitment scheme.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::constraints::statement::initial::InitialStatement;
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    WhirConfig,
};
use p3_whir::pcs::committer::reader::CommitmentReader;
use p3_whir::pcs::committer::writer::CommitmentWriter;
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::prover::WhirProver;
use p3_whir::pcs::verifier::WhirVerifier;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type PackedF = <F as Field>::Packing;
type Mmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;

// Polynomial sizes (log_2 of coefficient count).
const SMALL: usize = 14;
const MEDIUM: usize = 18;
const LARGE: usize = 20;

// Default knobs used by every benchmark unless overridden.
const FOLDING: usize = 4;
const LOG_INV_RATE: usize = 1;
const SOUNDNESS: SecurityAssumption = SecurityAssumption::CapacityBound;
const SUMCHECK: SumcheckStrategy = SumcheckStrategy::Svo;
const SECURITY_LEVEL: usize = 128;
const NUM_EVALUATIONS: usize = 1;
const RS_INITIAL_REDUCTION: usize = 3;

/// One benchmark configuration.
#[derive(Clone, Copy)]
struct Options {
    /// Log-2 of the polynomial coefficient count.
    num_variables: usize,
    /// Variables eliminated per WHIR round.
    folding: usize,
    /// Log-2 inverse rate of the starting Reed-Solomon code.
    log_inv_rate: usize,
    /// Soundness assumption driving query counts and PoW.
    soundness: SecurityAssumption,
    /// Initial-round sumcheck variant.
    sumcheck: SumcheckStrategy,
}

impl Options {
    /// Default configuration sized to a polynomial of the given variable count.
    const fn sized(num_variables: usize) -> Self {
        Self {
            num_variables,
            folding: FOLDING,
            log_inv_rate: LOG_INV_RATE,
            soundness: SOUNDNESS,
            sumcheck: SUMCHECK,
        }
    }

    /// Override the soundness assumption.
    const fn with_soundness(mut self, soundness: SecurityAssumption) -> Self {
        self.soundness = soundness;
        self
    }

    /// Override the sumcheck strategy.
    const fn with_sumcheck(mut self, sumcheck: SumcheckStrategy) -> Self {
        self.sumcheck = sumcheck;
        self
    }

    /// Override the folding factor.
    const fn with_folding(mut self, folding: usize) -> Self {
        self.folding = folding;
        self
    }
}

/// Pre-built benchmark fixture.
struct Bench {
    /// Derived per-round protocol configuration.
    config: WhirConfig<EF, F, Mmcs, Challenger>,
    /// User-facing protocol parameters retained for proof allocation.
    proto: ProtocolParameters<Mmcs>,
    /// DFT engine pre-loaded with twiddles up to the largest commit FFT.
    dft: Radix2DFTSmallBatch<F>,
    /// Random multilinear polynomial committed by every iteration.
    polynomial: Poly<F>,
    /// Evaluation points whose claims drive the prover's sumcheck rounds.
    eval_points: Vec<Point<EF>>,
    /// Fiat-Shamir domain separator binding the protocol structure.
    domain_separator: DomainSeparator<EF, F>,
    /// Pristine challenger cloned at the start of each iteration.
    base_challenger: Challenger,
    /// Initial-round sumcheck variant under test.
    sumcheck: SumcheckStrategy,
}

impl Bench {
    /// Build a fresh fixture from user-facing options.
    fn new(opts: Options) -> Self {
        // Deterministic Poseidon2 instances;
        //
        // The seed only fixes the round constants and never the protocol cost.
        let mut perm_rng = SmallRng::seed_from_u64(1);
        let poseidon16 = Poseidon16::new_from_rng_128(&mut perm_rng);
        let poseidon24 = Poseidon24::new_from_rng_128(&mut perm_rng);

        // Wire the Merkle hash (24-wide) and 2-to-1 compress (16-wide).
        let mmcs = Mmcs::new(
            MerkleHash::new(poseidon24),
            MerkleCompress::new(poseidon16.clone()),
            0,
        );

        // Translate user options into the protocol's parameter struct.
        let proto = ProtocolParameters {
            security_level: SECURITY_LEVEL,
            pow_bits: DEFAULT_MAX_POW,
            folding_factor: FoldingFactor::Constant(opts.folding),
            mmcs,
            soundness_type: opts.soundness,
            starting_log_inv_rate: opts.log_inv_rate,
            rs_domain_initial_reduction_factor: RS_INITIAL_REDUCTION,
        };

        // Derive the per-round configuration.
        let config = WhirConfig::<EF, F, Mmcs, Challenger>::new(opts.num_variables, proto.clone());

        // Pre-allocate twiddles up to the largest FFT performed at commit.
        let dft = Radix2DFTSmallBatch::<F>::new(1 << config.max_fft_size());

        // Random polynomial of 2^m coefficients; same seed across runs.
        let mut data_rng = SmallRng::seed_from_u64(0xD157A1B);
        let polynomial = Poly::<F>::new(
            (0..1 << opts.num_variables)
                .map(|_| data_rng.random())
                .collect(),
        );

        // Evaluation claims to prove.
        //
        // One claim is enough to exercise the full pipeline.
        //
        // We can add more if we want to stress test sumcheck things.
        let eval_points = (0..NUM_EVALUATIONS)
            .map(|_| Point::rand(&mut data_rng, opts.num_variables))
            .collect();

        // Bind the protocol structure into the Fiat-Shamir transcript.
        let mut domain_separator = DomainSeparator::<EF, F>::new(vec![]);
        domain_separator.commit_statement::<_, _, 32>(&config);
        domain_separator.add_whir_proof::<_, _, 32>(&config);

        Self {
            config,
            proto,
            dft,
            polynomial,
            eval_points,
            domain_separator,
            base_challenger: Challenger::new(poseidon16),
            sumcheck: opts.sumcheck,
        }
    }

    /// Initial statement seeded with the cached evaluation claims.
    ///
    /// A new statement is allocated per iteration:
    /// - commit mutates it by appending OOD constraints,
    /// - the verifier-side claim must come from a pre-OOD copy.
    fn statement(&self) -> InitialStatement<F, EF> {
        let mut statement = self
            .config
            .initial_statement(self.polynomial.clone(), self.sumcheck);
        for point in &self.eval_points {
            let _ = statement.evaluate(point);
        }
        statement
    }

    /// Empty proof container shaped for this configuration.
    fn proof(&self) -> WhirProof<F, EF, Mmcs> {
        WhirProof::<F, EF, Mmcs>::from_protocol_parameters(&self.proto, self.config.num_variables)
    }

    /// Pristine challenger with the domain separator already absorbed.
    fn challenger(&self) -> Challenger {
        let mut challenger = self.base_challenger.clone();
        self.domain_separator
            .observe_domain_separator(&mut challenger);
        challenger
    }

    /// Run the commit phase and return the per-iteration mutable inputs.
    ///
    /// Used to seed both the prove benchmark and the verify-side proof construction.
    fn commit(
        &self,
    ) -> (
        WhirProof<F, EF, Mmcs>,
        InitialStatement<F, EF>,
        Challenger,
        <Mmcs as p3_commit::Mmcs<F>>::ProverData<p3_matrix::dense::DenseMatrix<F>>,
    ) {
        let mut proof = self.proof();
        let mut statement = self.statement();
        let mut challenger = self.challenger();
        let prover_data = CommitmentWriter::new(&self.config)
            .commit(&self.dft, &mut proof, &mut challenger, &mut statement)
            .unwrap();
        (proof, statement, challenger, prover_data)
    }

    /// Build a complete proof outside any timed region.
    fn full_proof(&self) -> WhirProof<F, EF, Mmcs> {
        let (mut proof, statement, mut challenger, prover_data) = self.commit();
        WhirProver(&self.config)
            .prove(
                &self.dft,
                &mut proof,
                &mut challenger,
                &statement,
                prover_data,
            )
            .unwrap();
        proof
    }

    /// Time the commit phase under the given criterion group.
    fn bench_commit(&self, group: &mut BenchmarkGroup<'_, WallTime>, label: &str) {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || (self.proof(), self.statement(), self.challenger()),
                |(mut proof, mut statement, mut challenger)| {
                    CommitmentWriter::new(&self.config)
                        .commit(&self.dft, &mut proof, &mut challenger, &mut statement)
                        .unwrap()
                },
                BatchSize::LargeInput,
            );
        });
    }

    /// Time the prove phase.
    ///
    /// Commit is run during setup and excluded from the measurement window.
    fn bench_prove(&self, group: &mut BenchmarkGroup<'_, WallTime>, label: &str) {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || self.commit(),
                |(mut proof, statement, mut challenger, prover_data)| {
                    WhirProver(&self.config)
                        .prove(
                            &self.dft,
                            &mut proof,
                            &mut challenger,
                            &statement,
                            prover_data,
                        )
                        .unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }

    /// Time the verify phase.
    ///
    /// The proof is built once outside the measurement window; only verification is timed.
    fn bench_verify(&self, group: &mut BenchmarkGroup<'_, WallTime>, label: &str) {
        let proof = self.full_proof();
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || {
                    // Re-derive verifier-side state per iteration.
                    let verifier_statement = self.statement().normalize();
                    let mut challenger = self.challenger();
                    let parsed = CommitmentReader::new(&self.config)
                        .parse_commitment::<F, 8>(&proof, &mut challenger);
                    (challenger, parsed, verifier_statement)
                },
                |(mut challenger, parsed, verifier_statement)| {
                    WhirVerifier::new(&self.config)
                        .verify(&proof, &mut challenger, &parsed, verifier_statement)
                        .unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }
}

/// Apply a uniform sample-size policy to heavy benchmark groups.
///
/// 10 samples and a 20s measurement window keep the large-size cases
/// from blowing past the suite's overall time budget.
fn configure_heavy(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(2));
}

/// Scaling sweep across small / medium / large at the default options.
fn bench_scaling(c: &mut Criterion) {
    let cases: [(&str, Bench); 3] = [
        ("small", Bench::new(Options::sized(SMALL))),
        ("medium", Bench::new(Options::sized(MEDIUM))),
        ("large", Bench::new(Options::sized(LARGE))),
    ];

    // Commit phase across all three sizes.
    let mut group = c.benchmark_group("whir_pcs/scaling/commit");
    configure_heavy(&mut group);
    for (label, bench) in &cases {
        bench.bench_commit(&mut group, label);
    }
    group.finish();

    // Prove phase across all three sizes.
    let mut group = c.benchmark_group("whir_pcs/scaling/prove");
    configure_heavy(&mut group);
    for (label, bench) in &cases {
        bench.bench_prove(&mut group, label);
    }
    group.finish();

    // Verify is fast enough to keep a higher sample count.
    let mut group = c.benchmark_group("whir_pcs/scaling/verify");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    for (label, bench) in &cases {
        bench.bench_verify(&mut group, label);
    }
    group.finish();
}

/// Options sweep at the medium size, prove phase only.
fn bench_options(c: &mut Criterion) {
    let base = Options::sized(MEDIUM);

    // Sumcheck strategy: classic vs SVO. The user-requested axis.
    let mut group = c.benchmark_group("whir_pcs/options/sumcheck");
    configure_heavy(&mut group);
    for (label, mode) in [
        ("classic", SumcheckStrategy::Classic),
        ("svo", SumcheckStrategy::Svo),
    ] {
        Bench::new(base.with_sumcheck(mode)).bench_prove(&mut group, label);
    }
    group.finish();

    // Soundness assumption: drives query counts, OOD samples, and PoW.
    let mut group = c.benchmark_group("whir_pcs/options/soundness");
    configure_heavy(&mut group);
    for (label, assumption) in [
        ("ud", SecurityAssumption::UniqueDecoding),
        ("jb", SecurityAssumption::JohnsonBound),
        ("cb", SecurityAssumption::CapacityBound),
    ] {
        Bench::new(base.with_soundness(assumption)).bench_prove(&mut group, label);
    }
    group.finish();

    // Folding factor: trades round count against per-round work.
    //
    // The lower bound k = 3 matches `RS_INITIAL_REDUCTION`. A smaller k
    // would violate the protocol invariant that the first-round domain
    // reduction never exceeds the folding factor.
    let mut group = c.benchmark_group("whir_pcs/options/folding");
    configure_heavy(&mut group);
    for k in [3_usize, 4, 5] {
        Bench::new(base.with_folding(k)).bench_prove(&mut group, &format!("k{k}"));
    }
    group.finish();
}

criterion_group!(benches, bench_scaling, bench_options);
criterion_main!(benches);
