//! WHIR multilinear polynomial commitment scheme benchmark.
//!
//! - Demonstrates the full commit -> prove -> verify cycle,
//! - Prints timing and proof size statistics.
//!
//! Run with: `cargo run --example whir -- --num-variables 20 --evaluations 2`

use std::time::Instant;

use clap::Parser;
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use p3_whir::sumcheck::layout::{Layout as _, SuffixProver, Table};
use p3_whir::sumcheck::{OpeningProtocol, PointSchedule, TableShape, TableSpec};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use tracing::{info, warn};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

// Base field: 31-bit prime with large 2-adic subgroup.
type F = KoalaBear;
// Degree-4 extension for constraint batching and OOD sampling.
type EF = BinomialExtensionField<F, 4>;

// Poseidon2 permutations: width 16 for the challenger, width 24 for leaf hashing.
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

// Merkle tree: Poseidon2-based sponge for leaves, truncated permutation for 2-to-1 compression.
type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
// Duplex sponge challenger for Fiat-Shamir.
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type Layout = SuffixProver<F, EF>;
type MyPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, Layout>;

/// Command-line arguments for the WHIR benchmark.
#[derive(Parser, Debug)]
#[command(author, version, about = "WHIR multilinear PCS benchmark")]
struct Args {
    /// Target security level in bits (e.g. 90 or 128).
    #[arg(short = 'l', long, default_value = "90")]
    security_level: usize,

    /// Proof-of-work grinding difficulty. Defaults to the protocol maximum.
    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    /// Number of variables m in the multilinear polynomial (degree = 2^m).
    #[arg(short = 'd', long, default_value = "25")]
    num_variables: usize,

    /// Number of opening points for the single committed polynomial.
    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    /// Starting log-inverse rate: log_2(1/rho) where rho = 2^m / |L|.
    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    /// Folding parameter k: variables eliminated per WHIR round.
    #[arg(short = 'k', long = "fold", default_value = "5")]
    folding_factor: usize,

    /// Soundness assumption: UniqueDecoding, JohnsonBound, or CapacityBound.
    #[arg(long = "sec", default_value = "CapacityBound")]
    soundness_type: SecurityAssumption,

    /// Initial Reed-Solomon domain reduction factor before the first fold.
    #[arg(long = "initial-rs-reduction", default_value = "3")]
    rs_domain_initial_reduction_factor: usize,
}

fn main() {
    // Initialize structured logging with tracing-forest.
    // Respects RUST_LOG env var; defaults to INFO level.
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    // Parse command-line arguments.
    let mut args = Args::parse();

    // Default PoW to the protocol's recommended maximum if not specified.
    if args.pow_bits.is_none() {
        args.pow_bits = Some(DEFAULT_MAX_POW);
    }

    // Unpack arguments into local bindings.
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let folding_factor = FoldingFactor::Constant(args.folding_factor);
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;

    // Initialize Poseidon2 permutations from a deterministic seed.
    let mut rng = SmallRng::seed_from_u64(1);
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);

    // Build the Merkle tree commitment scheme from the hash primitives.
    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;
    // Assemble the WHIR protocol parameters.
    let whir_params = ProtocolParameters {
        security_level,
        pow_bits,
        folding_factor,
        soundness_type,
        starting_log_inv_rate: starting_rate,
        rs_domain_initial_reduction_factor,
    };

    info!(
        num_variables,
        num_evaluations,
        %soundness_type,
        pow_bits,
        starting_rate,
        "WHIR PCS"
    );

    // Generate one random multilinear polynomial f: {0,1}^m -> F and expose it
    // as a single-column table.
    let mut rng = SmallRng::seed_from_u64(0);
    let polynomial = Poly::<F>::new((0..1 << num_variables).map(|_| rng.random()).collect());
    let table = Table::new(vec![polynomial]);
    let witness = Layout::new_witness(vec![table], folding_factor.at_round(0));

    let point_schedule: PointSchedule = (0..num_evaluations).map(|_| vec![0]).collect();
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        point_schedule,
    )])
    .pad_to_min_num_variables(folding_factor.at_round(0));
    assert_eq!(witness.table_shapes(), protocol.table_shapes());

    // Derive the full round-by-round configuration from the committed witness.
    let config = WhirConfig::<EF, F, MyChallenger>::new(witness.num_variables(), whir_params);
    if !config.check_pow_bits() {
        warn!("more PoW bits required than what was specified");
    }

    // Create the base challenger from the Poseidon2 permutation.
    let challenger = MyChallenger::new(poseidon16);

    // Pre-allocate DFT twiddle factors up to the maximum FFT size.
    let dft = Radix2DFTSmallBatch::<F>::new(1 << config.max_fft_size());
    let pcs = MyPcs::new(config, dft, mmcs);

    // Phase 1: Commitment (DFT encoding + Merkle tree + OOD sampling).
    let mut prover_challenger = challenger.clone();
    let mut domainsep = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domainsep);
    domainsep.observe_domain_separator(&mut prover_challenger);
    let time = Instant::now();
    let (commitment, prover_data) =
        <MyPcs as MultilinearPcs<EF, MyChallenger>>::commit(&pcs, witness, &mut prover_challenger);
    let commit_time = time.elapsed();

    // Phase 2: Opening proof (multi-round sumcheck + STIR queries + PoW).
    let time = Instant::now();
    let proof = <MyPcs as MultilinearPcs<EF, MyChallenger>>::open(
        &pcs,
        prover_data,
        protocol.clone(),
        &mut prover_challenger,
    );
    let opening_time = time.elapsed();

    // Verifier: independent transcript from the same domain separator.
    let mut verifier_challenger = challenger;
    let mut domainsep = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domainsep);
    domainsep.observe_domain_separator(&mut verifier_challenger);

    // Phase 3: Verification (sumcheck checks + Merkle proof verification).
    let verif_time = Instant::now();
    <MyPcs as MultilinearPcs<EF, MyChallenger>>::verify(
        &pcs,
        &commitment,
        &proof,
        &mut verifier_challenger,
        protocol,
    )
    .expect("verification failed");
    let verify_time = verif_time.elapsed();

    // Report results via structured logging.
    let total_proving_ms = commit_time.as_millis() + opening_time.as_millis();
    info!(
        total_proving_ms,
        commit_ms = commit_time.as_millis(),
        opening_ms = opening_time.as_millis(),
        verification_us = verify_time.as_micros(),
        "done"
    );
}
