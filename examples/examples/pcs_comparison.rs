//! Head-to-head comparison of the FRI, STIR, and WHIR polynomial commitment schemes.
//!
//! Commits, opens, and verifies the same-shaped claim under all three protocols at a
//! shared target security level, then reports prover/verifier wall-clock time and
//! postcard-serialized proof size side by side.
//!
//! # Claim shape
//!
//! FRI and STIR are both univariate PCS: they commit a `2^log-message-size` x `width`
//! matrix (one polynomial of degree `< 2^log-message-size / width` per column) and open
//! every column at one common out-of-domain point `z`.
//!
//! WHIR is multilinear: it commits the same data reshaped as `width` stacked
//! multilinears in `log-message-size - log(width)` variables, and opens every column at
//! one common point via the univariate/multilinear bridge
//! `z_multilinear = (z, z^2, z^4, ..., z^(2^(k-1)))`, so all three protocols answer the
//! same query about the same committed data.
//!
//! # Soundness
//!
//! All three are configured under the capacity bound (conjectured RS proximity gap) at
//! the same `security-level` / `pow-bits` budget. FRI's query count follows a closed
//! form; STIR and WHIR each derive their own per-round query/PoW schedule from that
//! budget, so query counts differ across protocols even at matched security.
//!
//! STIR's schedule derivation also enforces a per-round validity ceiling on its `eta`
//! parameter (see `p3_stir::config`); some `(log-message-size, rate, stir-log-fold)`
//! combinations are not valid under that ceiling and `StirConfig::new` panics with a
//! description of the violated bound. Lower `stir-log-fold` or raise `rate` if that
//! happens.
//!
//! # Run
//!
//! Each protocol's internal tracing spans log at INFO by default; set `RUST_LOG=warn`
//! to see only the final comparison table.
//!
//! ```bash
//! cargo run --release --example pcs_comparison -- --log-message-size 20 --log-width 8
//! ```

use std::time::Instant;

use clap::Parser;
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, MultilinearPcs, Pcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::QuinticTrinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::{StirConfig, StirParameters, TwoAdicStirPcs};
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tracing::{info, warn};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// Base field shared by all three protocols.
type F = KoalaBear;
/// Challenge field used for Fiat-Shamir sampling and out-of-domain evaluation.
///
/// STIR's capacity-bound schedule bounds its per-round `eta` against the challenge field
/// size, so a quintic extension is used to leave headroom for larger folding arities.
type EF = QuinticTrinomialExtensionField<F>;
/// DFT backend shared by all three protocols.
type Dft = Radix2DFTSmallBatch<F>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;
type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type PackedF = <F as Field>::Packing;
type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;

type FriPcsTy = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>;
type StirPcsTy = TwoAdicStirPcs<F, Dft, ValMmcs, ChallengeMmcs>;
type WhirLayout = SuffixProver<F, EF>;
type WhirPcsTy = WhirProver<EF, F, Dft, ValMmcs, Challenger, WhirLayout>;

/// Number of base-field elements per Merkle digest for this hash backend.
const DIGEST_ELEMS: usize = 8;

/// Command-line arguments for the PCS comparison.
#[derive(Parser, Debug)]
#[command(author, version, about = "FRI vs STIR vs WHIR PCS comparison")]
struct Args {
    /// Target security level in bits, shared by all three protocols.
    #[arg(short = 'l', long, default_value = "100")]
    security_level: usize,

    /// Proof-of-work grinding budget in bits, shared by all three protocols.
    #[arg(short = 'p', long, default_value = "20")]
    pow_bits: usize,

    /// Log_2 of the total number of base-field elements committed.
    #[arg(short = 'm', long, default_value = "20")]
    log_message_size: usize,

    /// Log_2 of the number of polynomials/columns opened at the common point.
    #[arg(short = 'w', long, default_value = "0")]
    log_width: usize,

    /// Log_2 of the inverse rate of the starting Reed-Solomon code.
    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    /// WHIR folding factor k: variables eliminated per round.
    #[arg(long, default_value = "4")]
    whir_fold: usize,

    /// STIR log_2 folding arity per round (must be >= 2).
    ///
    /// Matches `whir-fold` by default so both protocols fold at arity 16.
    #[arg(long, default_value = "4")]
    stir_log_fold: usize,

    /// FRI log_2 folding arity per round.
    #[arg(long, default_value = "1")]
    fri_log_arity: usize,
}

/// Timing and size summary for one protocol's commit -> open -> verify run.
struct ProtocolReport {
    label: &'static str,
    commit_ms: u128,
    open_ms: u128,
    verify_us: u128,
    proof_bytes: usize,
    queries: String,
}

/// Per-round inverse-rate schedule matching WHIR's default protocol parameters.
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

/// Run one full commit -> open -> verify cycle for a univariate PCS (FRI or STIR) and
/// report its timing, proof size, and query count.
fn run_univariate_pcs<P>(
    label: &'static str,
    pcs: &P,
    domain: TwoAdicMultiplicativeCoset<F>,
    message: RowMajorMatrix<F>,
    base_challenger: &Challenger,
    queries: String,
    observe: impl Fn(&mut Challenger, &P::Commitment),
) -> ProtocolReport
where
    P: Pcs<EF, Challenger, Domain = TwoAdicMultiplicativeCoset<F>>,
{
    let mut prover_challenger = base_challenger.clone();

    let t = Instant::now();
    let (commit, prover_data) = pcs.commit([(domain, message)]);
    let commit_ms = t.elapsed().as_millis();

    observe(&mut prover_challenger, &commit);
    // UFCS spelling pins the challenger's field generic so trait selection is unambiguous
    // in the presence of the generic `P::Commitment` bound above.
    let zeta: EF =
        <Challenger as FieldChallenger<F>>::sample_algebra_element(&mut prover_challenger);

    let t = Instant::now();
    let (openings, proof) = pcs.open(
        vec![(&prover_data, vec![vec![zeta]])],
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();
    let values = openings[0][0][0].clone();

    let mut verifier_challenger = base_challenger.clone();
    observe(&mut verifier_challenger, &commit);
    let derived: EF =
        <Challenger as FieldChallenger<F>>::sample_algebra_element(&mut verifier_challenger);
    assert_eq!(derived, zeta, "verifier challenger drifted from prover");

    let t = Instant::now();
    pcs.verify(
        vec![(commit, vec![(domain, vec![(zeta, values)])])],
        &proof,
        &mut verifier_challenger,
    )
    .unwrap_or_else(|_| panic!("{label} verify failed"));
    let verify_us = t.elapsed().as_micros();

    let proof_bytes = postcard::to_allocvec(&proof)
        .unwrap_or_else(|_| panic!("{label} proof failed to serialize"))
        .len();

    ProtocolReport {
        label,
        commit_ms,
        open_ms,
        verify_us,
        proof_bytes,
        queries,
    }
}

/// Run one full commit -> open -> verify cycle for WHIR and report its timing, proof
/// size, and per-round query counts.
fn run_whir(
    pcs: &WhirPcsTy,
    witness: <WhirPcsTy as MultilinearPcs<EF, Challenger>>::Witness,
    protocol: OpeningProtocol,
    domain_separator: &DomainSeparator<EF, F>,
    base_challenger: &Challenger,
) -> ProtocolReport {
    let mut prover_challenger = base_challenger.clone();
    domain_separator.observe_domain_separator(&mut prover_challenger);

    let t = Instant::now();
    let (commitment, prover_data) =
        <WhirPcsTy as MultilinearPcs<EF, Challenger>>::commit(pcs, witness, &mut prover_challenger);
    let commit_ms = t.elapsed().as_millis();

    let t = Instant::now();
    let proof = <WhirPcsTy as MultilinearPcs<EF, Challenger>>::open(
        pcs,
        prover_data,
        protocol.clone(),
        &mut prover_challenger,
    );
    let open_ms = t.elapsed().as_millis();

    let mut verifier_challenger = base_challenger.clone();
    domain_separator.observe_domain_separator(&mut verifier_challenger);

    let t = Instant::now();
    <WhirPcsTy as MultilinearPcs<EF, Challenger>>::verify(
        pcs,
        &commitment,
        &proof,
        &mut verifier_challenger,
        protocol,
    )
    .expect("whir verify failed");
    let verify_us = t.elapsed().as_micros();

    let proof_bytes = postcard::to_allocvec(&proof)
        .expect("whir proof failed to serialize")
        .len();

    let queries = pcs
        .config
        .round_parameters
        .iter()
        .map(|r| r.num_queries.to_string())
        .collect::<Vec<_>>()
        .join(",");

    ProtocolReport {
        label: "whir",
        commit_ms,
        open_ms,
        verify_us,
        proof_bytes,
        queries,
    }
}

fn print_report(args: &Args, reports: &[ProtocolReport]) {
    println!();
    println!(
        "=== FRI vs STIR vs WHIR ({}-bit security, rho = 2^-{}, m = {}, width = 2^{}) ===",
        args.security_level, args.rate, args.log_message_size, args.log_width
    );
    println!(
        "  protocol | commit ms | open ms | total ms | verify us | proof bytes | proof KiB | queries"
    );
    println!(
        "-----------+-----------+---------+----------+-----------+-------------+-----------+--------"
    );
    for r in reports {
        println!(
            "  {:<8} | {:>9} | {:>7} | {:>8} | {:>9} | {:>11} | {:>9.2} | [{}]",
            r.label,
            r.commit_ms,
            r.open_ms,
            r.commit_ms + r.open_ms,
            r.verify_us,
            r.proof_bytes,
            r.proof_bytes as f64 / 1024.0,
            r.queries,
        );
    }
    println!();
}

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let args = Args::parse();
    assert!(
        args.pow_bits < args.security_level,
        "pow-bits must be strictly less than security-level"
    );
    assert!(
        args.log_width <= args.log_message_size,
        "log-width cannot exceed log-message-size"
    );

    let log_height = args.log_message_size - args.log_width;
    let width = 1usize << args.log_width;

    let mut perm_rng = SmallRng::seed_from_u64(1);
    let poseidon16 = Poseidon16::new_from_rng_128(&mut perm_rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut perm_rng);
    let val_mmcs = ValMmcs::new(
        MerkleHash::new(poseidon24),
        MerkleCompress::new(poseidon16.clone()),
        0,
    );
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let base_challenger = Challenger::new(poseidon16);

    info!(
        security_level = args.security_level,
        pow_bits = args.pow_bits,
        log_message_size = args.log_message_size,
        log_width = args.log_width,
        rate = args.rate,
        "PCS comparison"
    );

    let mut reports = Vec::with_capacity(3);

    // FRI: closed-form query count, num_queries * log_blowup + pow_bits >= security_level.
    {
        let num_queries = (args.security_level - args.pow_bits)
            .div_ceil(args.rate)
            .max(1);
        let fri_params = FriParameters {
            log_blowup: args.rate,
            log_final_poly_len: 0,
            max_log_arity: args.fri_log_arity,
            num_queries,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: args.pow_bits,
            mmcs: challenge_mmcs.clone(),
        };
        let dft = Dft::new(1 << (log_height + args.rate));
        let pcs = FriPcsTy::new(dft, val_mmcs.clone(), fri_params);
        let domain =
            <FriPcsTy as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_height);
        let mut rng = SmallRng::seed_from_u64(0xF12);
        let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, width);
        reports.push(run_univariate_pcs(
            "fri",
            &pcs,
            domain,
            message,
            &base_challenger,
            num_queries.to_string(),
            |ch, commit| ch.observe(commit.clone()),
        ));
    }

    // STIR: query/PoW schedule is derived per round from the same security budget.
    {
        let stir_params = StirParameters {
            log_blowup: args.rate,
            log_folding_factor: args.stir_log_fold,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: args.security_level,
            max_pow_bits: args.pow_bits,
            mmcs: challenge_mmcs,
        };
        let config =
            StirConfig::<F, EF, ChallengeMmcs, Challenger>::new(log_height, stir_params.clone());
        let queries = config
            .round_configs
            .iter()
            .map(|rc| rc.num_queries.to_string())
            .chain(std::iter::once(config.final_queries.to_string()))
            .collect::<Vec<_>>()
            .join(",");

        let dft = Dft::new(1 << (log_height + args.rate));
        let pcs = StirPcsTy::new(dft, val_mmcs.clone(), stir_params);
        let domain =
            <StirPcsTy as Pcs<EF, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_height);
        let mut rng = SmallRng::seed_from_u64(0x57113);
        let message = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, width);
        reports.push(run_univariate_pcs(
            "stir",
            &pcs,
            domain,
            message,
            &base_challenger,
            queries,
            // STIR commits one Merkle tree per distinct LDE height.
            |ch, commit| commit.iter().for_each(|c| ch.observe(c.clone())),
        ));
    }

    // WHIR: same target security budget, multilinear analogue of the same claim shape.
    {
        let folding_factor = FoldingFactor::Constant(args.whir_fold);
        let params = ProtocolParameters {
            security_level: args.security_level,
            pow_bits: args.pow_bits,
            round_log_inv_rates: default_round_log_inv_rates(
                args.log_message_size,
                &folding_factor,
            ),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: args.rate,
        };
        let config = WhirConfig::<EF, F, Challenger>::new(args.log_message_size, params).unwrap();
        if !config.check_pow_bits() {
            warn!("WHIR requires more PoW bits than the configured budget");
        }

        let mut rng = SmallRng::seed_from_u64(0x41112);
        let table = Table::rand(&mut rng, width, log_height);
        let witness = WhirLayout::new_witness(vec![table], args.whir_fold);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(log_height, width),
            vec![OpeningBatch::new((0..width).collect(), Vec::new())],
        )])
        .pad_to_min_num_variables(args.whir_fold);

        let dft = Dft::new(1 << config.max_fft_size());
        let pcs = WhirPcsTy::new(config, dft, val_mmcs);

        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<DIGEST_ELEMS>(&mut domain_separator);

        reports.push(run_whir(
            &pcs,
            witness,
            protocol,
            &domain_separator,
            &base_challenger,
        ));
    }

    print_report(&args, &reports);
}
