//! End-to-end benchmarks for the WHIR polynomial commitment scheme.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::QuinticTrinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PointSchedule, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = KoalaBear;
type EF = QuinticTrinomialExtensionField<F>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type PackedF = <F as Field>::Packing;
type Mmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type Dft = Radix2DFTSmallBatch<F>;

/// Concrete PCS instantiation parameterized by the sumcheck layout mode.
type Pcs<L> = WhirProver<EF, F, Dft, Mmcs, Challenger, L>;

// Polynomial sizes (log_2 of coefficient count).
const SMALL: usize = 14;
const MEDIUM: usize = 18;
const LARGE: usize = 20;

// Default knobs used by every benchmark unless overridden.
const FOLDING: usize = 4;
const LOG_INV_RATE: usize = 1;
const SOUNDNESS: SecurityAssumption = SecurityAssumption::CapacityBound;
const SECURITY_LEVEL: usize = 128;
// One opening claim is enough to exercise the full pipeline.
//
// We can add more if we want to stress test sumcheck things.
const NUM_EVALUATIONS: usize = 1;

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
}

impl Options {
    /// Default configuration sized to a polynomial of the given variable count.
    const fn sized(num_variables: usize) -> Self {
        Self {
            num_variables,
            folding: FOLDING,
            log_inv_rate: LOG_INV_RATE,
            soundness: SOUNDNESS,
        }
    }

    /// Override the soundness assumption.
    const fn with_soundness(mut self, soundness: SecurityAssumption) -> Self {
        self.soundness = soundness;
        self
    }

    /// Override the folding factor.
    const fn with_folding(mut self, folding: usize) -> Self {
        self.folding = folding;
        self
    }
}

/// Per-round inverse-rate schedule matching the default protocol parameters.
///
/// Each entry is the log-2 inverse rate of the Reed-Solomon code used in that
/// round. Round 0 starts at rate 1 and each subsequent round absorbs one fewer
/// rate halving per folded variable.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let (num_rounds, _) = folding_factor
        .compute_number_of_rounds(num_variables)
        .expect("valid folding schedule");
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for round in 0..num_rounds {
        rate += folding_factor.at_round(round) - 1;
        rates.push(rate);
    }
    rates
}

/// Pre-built benchmark fixture parameterized by the sumcheck layout mode.
struct Bench<L: Layout<F, EF>> {
    /// Fully-instantiated WHIR PCS.
    pcs: Pcs<L>,
    /// Pre-built witness; cloned at the start of each iteration.
    witness: <Pcs<L> as MultilinearPcs<EF, Challenger>>::Witness,
    /// Public opening protocol matching the witness shape.
    protocol: OpeningProtocol,
    /// Fiat-Shamir domain separator binding the protocol structure.
    domain_separator: DomainSeparator<EF, F>,
    /// Pristine challenger cloned at the start of each iteration.
    base_challenger: Challenger,
}

impl<L: Layout<F, EF>> Bench<L> {
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
        let folding_factor = FoldingFactor::Constant(opts.folding);
        let params = ProtocolParameters {
            security_level: SECURITY_LEVEL,
            pow_bits: DEFAULT_MAX_POW,
            round_log_inv_rates: default_round_log_inv_rates(opts.num_variables, &folding_factor),
            folding_factor,
            soundness_type: opts.soundness,
            starting_log_inv_rate: opts.log_inv_rate,
        };

        // Derive the per-round configuration and pre-allocate FFT twiddles.
        let config = WhirConfig::<EF, F, Challenger>::new(opts.num_variables, params).unwrap();
        let dft = Dft::new(1 << config.max_fft_size());
        let pcs = Pcs::<L>::new(config, dft, mmcs);

        // Single random table of one column committed by every iteration.
        let mut data_rng = SmallRng::seed_from_u64(0xD157A1B);
        let table = Table::new(vec![Poly::<F>::rand(&mut data_rng, opts.num_variables)]);
        let witness = L::new_witness(vec![table], opts.folding);

        // Open the single column NUM_EVALUATIONS times at fresh sampled points.
        let point_schedule: PointSchedule = (0..NUM_EVALUATIONS)
            .map(|_| OpeningBatch::new(vec![0], Vec::new()))
            .collect();
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(opts.num_variables, 1),
            point_schedule,
        )]);

        // Bind the protocol structure into the Fiat-Shamir transcript.
        let mut domain_separator = DomainSeparator::<EF, F>::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);

        Self {
            pcs,
            witness,
            protocol,
            domain_separator,
            base_challenger: Challenger::new(poseidon16),
        }
    }

    /// Pristine challenger with the domain separator already absorbed.
    fn challenger(&self) -> Challenger {
        let mut challenger = self.base_challenger.clone();
        self.domain_separator
            .observe_domain_separator(&mut challenger);
        challenger
    }

    /// Time the commit phase under the given criterion group.
    fn bench_commit(&self, group: &mut BenchmarkGroup<'_, WallTime>, label: &str) {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || (self.witness.clone(), self.challenger()),
                |(witness, mut challenger)| {
                    <Pcs<L> as MultilinearPcs<EF, Challenger>>::commit(
                        &self.pcs,
                        witness,
                        &mut challenger,
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    /// Time the open phase.
    ///
    /// Commit runs in setup and is excluded from the measurement window.
    fn bench_prove(&self, group: &mut BenchmarkGroup<'_, WallTime>, label: &str) {
        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || {
                    let mut challenger = self.challenger();
                    let (_, prover_data) = <Pcs<L> as MultilinearPcs<EF, Challenger>>::commit(
                        &self.pcs,
                        self.witness.clone(),
                        &mut challenger,
                    );
                    (prover_data, challenger)
                },
                |(prover_data, mut challenger)| {
                    <Pcs<L> as MultilinearPcs<EF, Challenger>>::open(
                        &self.pcs,
                        prover_data,
                        self.protocol.clone(),
                        &mut challenger,
                    )
                },
                BatchSize::PerIteration,
            );
        });
    }

    /// Time the verify phase.
    ///
    /// The proof is built once outside the measurement window; only verification is timed.
    fn bench_verify(&self, group: &mut BenchmarkGroup<'_, WallTime>, label: &str) {
        let mut challenger = self.challenger();
        let (commitment, prover_data) = <Pcs<L> as MultilinearPcs<EF, Challenger>>::commit(
            &self.pcs,
            self.witness.clone(),
            &mut challenger,
        );
        let proof = <Pcs<L> as MultilinearPcs<EF, Challenger>>::open(
            &self.pcs,
            prover_data,
            self.protocol.clone(),
            &mut challenger,
        );

        group.bench_function(BenchmarkId::from_parameter(label), |b| {
            b.iter_batched(
                || self.challenger(),
                |mut challenger| {
                    <Pcs<L> as MultilinearPcs<EF, Challenger>>::verify(
                        &self.pcs,
                        &commitment,
                        &proof,
                        &mut challenger,
                        self.protocol.clone(),
                    )
                    .unwrap();
                },
                BatchSize::PerIteration,
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
    type L = SuffixProver<F, EF>;
    let cases: [(&str, Bench<L>); 3] = [
        ("small", Bench::new(Options::sized(SMALL))),
        ("medium", Bench::new(Options::sized(MEDIUM))),
        ("large", Bench::new(Options::sized(LARGE))),
    ];

    // Commit phase across all three sizes.
    {
        let mut group = c.benchmark_group("whir_pcs/scaling/commit");
        configure_heavy(&mut group);
        for (label, bench) in &cases {
            bench.bench_commit(&mut group, label);
        }
        group.finish();
    }

    // Prove phase across all three sizes.
    {
        let mut group = c.benchmark_group("whir_pcs/scaling/prove");
        configure_heavy(&mut group);
        for (label, bench) in &cases {
            bench.bench_prove(&mut group, label);
        }
        group.finish();
    }

    // Verify is fast enough to keep a higher sample count.
    {
        let mut group = c.benchmark_group("whir_pcs/scaling/verify");
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(10));
        for (label, bench) in &cases {
            bench.bench_verify(&mut group, label);
        }
        group.finish();
    }
}

/// Options sweep at the medium size, prove phase only.
fn bench_options(c: &mut Criterion) {
    let base = Options::sized(MEDIUM);

    // Layout: SVO suffix vs prefix binding order.
    //
    // Replaces the legacy SumcheckStrategy axis after the sumcheck refactor.
    {
        let mut group = c.benchmark_group("whir_pcs/options/layout");
        configure_heavy(&mut group);
        Bench::<SuffixProver<F, EF>>::new(base).bench_prove(&mut group, "suffix");
        Bench::<PrefixProver<F, EF>>::new(base).bench_prove(&mut group, "prefix");
        group.finish();
    }

    // Soundness assumption: drives query counts, OOD samples, and PoW.
    {
        let mut group = c.benchmark_group("whir_pcs/options/soundness");
        configure_heavy(&mut group);
        for (label, assumption) in [
            ("ud", SecurityAssumption::UniqueDecoding),
            ("jb", SecurityAssumption::JohnsonBound),
            ("cb", SecurityAssumption::CapacityBound),
        ] {
            Bench::<SuffixProver<F, EF>>::new(base.with_soundness(assumption))
                .bench_prove(&mut group, label);
        }
        group.finish();
    }

    // Folding factor: trades round count against per-round work.
    {
        let mut group = c.benchmark_group("whir_pcs/options/folding");
        configure_heavy(&mut group);
        for k in [3_usize, 4, 5] {
            Bench::<SuffixProver<F, EF>>::new(base.with_folding(k))
                .bench_prove(&mut group, &format!("k{k}"));
        }
        group.finish();
    }
}

criterion_group!(benches, bench_scaling, bench_options);
criterion_main!(benches);
