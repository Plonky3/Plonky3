//! WHIR PCS profiling example with a Blake3 Merkle backend.
//!
//! Identical protocol flow to the `whir` example, but with byte-oriented
//! Blake3 hashing for the Merkle tree AND the Fiat-Shamir challenger, so both
//! commitment hashing and proof-of-work grinding run on the cheap hash. This
//! is the most sum-check-favorable configuration: it maximizes the fraction of
//! prover time spent in basis-affected arithmetic.
//!
//! Run with: `cargo run --release --example whir_blake3 -- --num-variables 22 -p 20`

use std::time::Instant;

use clap::Parser;
use p3_blake3::Blake3;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::KoalaBear;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout as _, PrefixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PointSchedule, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig,
};
use p3_whir::pcs::prover::WhirProver;
use rand::SeedableRng;
use rand::rngs::SmallRng;
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

// Blake3-backed Merkle: serialize field rows to bytes, hash into 32-byte digests.
type ByteHash = Blake3;
type FieldHash = SerializingHasher<Blake3>;
type MerkleCompress = CompressionFunctionFromHasher<Blake3, 2, 32>;
type MyMmcs = MerkleTreeMmcs<F, u8, FieldHash, MerkleCompress, 2, 32>;
// Byte-stream challenger: Blake3 sponge feeding the Fiat-Shamir transcript.
type MyChallenger = SerializingChallenger32<F, HashChallenger<u8, Blake3, 32>>;
type MyDft = Radix2DFTSmallBatch<F>;
type Layout = PrefixProver<F, EF>;
type MyPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, Layout>;

/// Command-line arguments for the Blake3 WHIR profile.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "WHIR multilinear PCS profile (Blake3 Merkle)"
)]
struct Args {
    /// Target security level in bits (e.g. 90 or 128).
    #[arg(short = 'l', long, default_value = "90")]
    security_level: usize,

    /// Proof-of-work grinding difficulty. Defaults to the protocol maximum.
    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    /// Number of variables m in the multilinear polynomial (degree = 2^m).
    #[arg(short = 'd', long, default_value = "22")]
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
}

fn main() {
    // Structured logging with tracing-forest; respects RUST_LOG, defaults to INFO.
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let mut args = Args::parse();
    if args.pow_bits.is_none() {
        args.pow_bits = Some(DEFAULT_MAX_POW);
    }

    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let folding_factor = FoldingFactor::Constant(args.folding_factor);
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;

    // Blake3 is stateless: no RNG or round constants needed.
    let field_hash = FieldHash::new(ByteHash {});
    let merkle_compress = MerkleCompress::new(ByteHash {});
    let mmcs = MyMmcs::new(field_hash, merkle_compress, 0);

    let round_log_inv_rates = {
        let (num_rounds, _) = folding_factor
            .compute_number_of_rounds(num_variables)
            .expect("valid folding schedule");
        let mut rates = Vec::with_capacity(num_rounds);
        let mut rate = starting_rate;
        for round in 0..num_rounds {
            rate += folding_factor.at_round(round) - 1;
            rates.push(rate);
        }
        rates
    };

    let whir_params = ProtocolParameters {
        security_level,
        pow_bits,
        folding_factor: folding_factor.clone(),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        round_log_inv_rates,
    };

    info!(
        num_variables,
        num_evaluations,
        %soundness_type,
        pow_bits,
        starting_rate,
        "WHIR PCS (Blake3 Merkle + challenger)"
    );

    // One random multilinear as a single-column table.
    let mut rng = SmallRng::seed_from_u64(0);
    let table = Table::rand(&mut rng, 1, num_variables);
    let witness = Layout::new_witness(vec![table], folding_factor.at_round(0));

    let point_schedule: PointSchedule = (0..num_evaluations)
        .map(|_| OpeningBatch::new(vec![0], Vec::new()))
        .collect();
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        point_schedule,
    )])
    .pad_to_min_num_variables(folding_factor.at_round(0));
    assert_eq!(witness.table_shapes(), protocol.table_shapes());

    let config =
        WhirConfig::<EF, F, MyChallenger>::new(witness.num_variables(), whir_params).unwrap();
    if !config.check_pow_bits() {
        warn!("more PoW bits required than what was specified");
    }

    let challenger = MyChallenger::new(HashChallenger::new(vec![], ByteHash {}));

    let dft = MyDft::new(1 << config.max_fft_size());
    let pcs = MyPcs::new(config, dft, mmcs);

    // Phase 1: Commitment (DFT encoding + Merkle tree + OOD sampling).
    let mut prover_challenger = challenger.clone();
    let mut domainsep = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<32>(&mut domainsep);
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
    pcs.add_domain_separator::<32>(&mut domainsep);
    domainsep.observe_domain_separator(&mut verifier_challenger);

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

    let total_proving_ms = commit_time.as_millis() + opening_time.as_millis();
    info!(
        total_proving_ms,
        commit_ms = commit_time.as_millis(),
        opening_ms = opening_time.as_millis(),
        verification_us = verify_time.as_micros(),
        "done"
    );
}
