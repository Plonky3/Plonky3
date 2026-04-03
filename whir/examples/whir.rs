//! WHIR multilinear polynomial commitment scheme benchmark.
//!
//! - Demonstrates the full commit -> prove -> verify cycle,
//! - Prints timing and proof size statistics.
//!
//! Run with: `cargo run --example whir -- --num-variables 20 --evaluations 2`

use std::time::Instant;

use clap::Parser;
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    WhirConfig,
};
use p3_whir::pcs::committer::reader::CommitmentReader;
use p3_whir::pcs::committer::writer::CommitmentWriter;
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::prover::Prover;
use p3_whir::pcs::verifier::Verifier;
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

    /// Number of evaluation claims to prove (0 = proximity test only).
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

    if num_evaluations == 0 {
        warn!("running as proximity test only — no evaluation claims specified");
    }

    // Initialize Poseidon2 permutations from a deterministic seed.
    let mut rng = SmallRng::seed_from_u64(1);
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);

    // Build the Merkle tree commitment scheme from the hash primitives.
    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    let rs_domain_initial_reduction_factor = args.rs_domain_initial_reduction_factor;
    // Polynomial has 2^m evaluations on the Boolean hypercube.
    let num_coeffs = 1 << num_variables;

    // Assemble the WHIR protocol parameters.
    let whir_params = ProtocolParameters {
        security_level,
        pow_bits,
        folding_factor,
        mmcs,
        soundness_type,
        starting_log_inv_rate: starting_rate,
        rs_domain_initial_reduction_factor,
    };

    // Derive the full round-by-round configuration.
    let params = WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

    info!(
        num_variables,
        num_evaluations,
        %soundness_type,
        pow_bits,
        starting_rate,
        "WHIR PCS"
    );

    if !params.check_pow_bits() {
        warn!("more PoW bits required than what was specified");
    }

    // Generate a random multilinear polynomial f: {0,1}^m -> F.
    let mut rng = SmallRng::seed_from_u64(0);
    let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());

    // Build the initial statement and add evaluation constraints.
    // Each constraint asserts f(z_i) = v_i for a random point z_i.
    let mut initial_statement = params.initial_statement(polynomial, SumcheckStrategy::default());
    (0..num_evaluations).for_each(|_| {
        let _ = initial_statement.evaluate(&Point::rand(&mut rng, num_variables));
    });

    // Normalize the statement into an equality form for the verifier.
    let verifier_statement = initial_statement.normalize();

    // Build the Fiat-Shamir domain separator.
    // This encodes the protocol parameters and proof structure into the
    // transcript so challenges are bound to this specific configuration.
    let mut domainsep = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, 32>(&params);
    domainsep.add_whir_proof::<_, _, 32>(&params);

    // Create the base challenger from the Poseidon2 permutation.
    let challenger = MyChallenger::new(poseidon16);

    // Prover: initialize transcript and commit.
    let mut prover_challenger = challenger.clone();
    domainsep.observe_domain_separator(&mut prover_challenger);

    let committer = CommitmentWriter::new(&params);
    // Pre-allocate DFT twiddle factors up to the maximum FFT size.
    let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());

    let mut proof =
        WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

    // Phase 1: Commitment (DFT encoding + Merkle tree + OOD sampling).
    let time = Instant::now();
    let prover_data = committer
        .commit(
            &dft,
            &mut proof,
            &mut prover_challenger,
            &mut initial_statement,
        )
        .unwrap();
    let commit_time = time.elapsed();

    // Phase 2: Opening proof (multi-round sumcheck + STIR queries + PoW).
    let prover = Prover(&params);
    let time = Instant::now();
    prover
        .prove(
            &dft,
            &mut proof,
            &mut prover_challenger,
            &initial_statement,
            prover_data,
        )
        .unwrap();
    let opening_time = time.elapsed();

    // Verifier: independent transcript from the same domain separator.
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    let mut verifier_challenger = challenger;
    domainsep.observe_domain_separator(&mut verifier_challenger);

    // Parse the commitment from the proof (replays OOD transcript interactions).
    let parsed_commitment =
        commitment_reader.parse_commitment::<F, 8>(&proof, &mut verifier_challenger);

    // Phase 3: Verification (sumcheck checks + Merkle proof verification).
    let verif_time = Instant::now();
    verifier
        .verify(
            &proof,
            &mut verifier_challenger,
            &parsed_commitment,
            verifier_statement,
        )
        .unwrap();
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
