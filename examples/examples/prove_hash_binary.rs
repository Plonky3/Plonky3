//! Prove a batch of hashes, each one instance of its Flock circuit inside one multi-STARK circuit table.

use std::io;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use p3_binary_pcs::whir::BinaryWhirBudget;
use p3_examples::binary::{
    BinaryProofOptions, BinaryProofReport, HashFamily, WhirOptions, WhirRegime, preflight_circuit,
    prove_circuit,
};
use p3_examples::parsers::{BinaryCommitmentHashOptions, BinaryHashOptions, OutputFormat};
use p3_flock::Gadget;
use p3_flock::gadgets::blake2s::Blake2sCompress;
use p3_flock::gadgets::blake3::Blake3Compress;
use p3_flock::gadgets::keccak::KeccakF1600;
use p3_flock::gadgets::sha256::Sha256Compress;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use tracing_forest::util::LevelFilter;
use tracing_forest::{ForestLayer, PrettyPrinter};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhirRegimeOptions {
    #[value(alias = "unique")]
    UniqueDecoding,
    Johnson,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// What we are trying to prove.
    #[arg(short, long, ignore_case = true, value_enum)]
    objective: BinaryHashOptions,

    /// The log base 2 of the number of hashes.
    #[arg(short, long, visible_alias = "log-trace-length")]
    log_hashes: u8,

    /// How to print the measurements.
    ///
    /// In `json` mode, standard output carries only the JSON report.
    ///
    /// Progress lines and tracing spans move to standard error.
    #[arg(long, ignore_case = true, value_enum, default_value_t = OutputFormat::Human)]
    format: OutputFormat,

    /// Log of the inverse code rate.
    #[arg(long, default_value_t = 1)]
    log_inv_rate: usize,

    /// Composed security target of the whole proof, in bits; the 32-byte hash caps it at 128.
    #[arg(long, default_value_t = 100)]
    security_bits: usize,

    /// Variables each WHIR round after the first folds.
    #[arg(long, default_value_t = 4)]
    folding: usize,

    /// The byte hash the Merkle trees and the Fiat-Shamir transcript share.
    #[arg(long, ignore_case = true, value_enum, default_value_t = BinaryCommitmentHashOptions::Keccak256)]
    hash: BinaryCommitmentHashOptions,

    /// WHIR proximity regime.
    #[arg(long = "whir-regime", ignore_case = true, value_enum, default_value_t = WhirRegimeOptions::Johnson)]
    whir_regime: WhirRegimeOptions,

    /// Per-term WHIR security target, twelve bits above the composed one unless set.
    #[arg(long = "whir-term-security-bits")]
    whir_term_security_bits: Option<usize>,

    /// Grinding bits before each WHIR query phase.
    ///
    /// Twelve, not Flock's sixteen: past twelve, each bit doubles the grinding for a few queries less.
    #[arg(long = "whir-query-grinding-bits", default_value_t = 12)]
    whir_query_grinding_bits: usize,

    /// Variables the first WHIR round folds.
    #[arg(long = "whir-first-folding", default_value_t = 6)]
    whir_first_folding: usize,

    /// Domain bits the first WHIR round drops.
    #[arg(long = "whir-first-reduction", default_value_t = 4)]
    whir_first_reduction: usize,

    /// Optional WHIR query ceiling.
    #[arg(long = "whir-max-queries")]
    whir_max_queries: Option<usize>,

    /// Optional serialized proof byte ceiling.
    #[arg(long = "whir-max-proof-bytes")]
    whir_max_proof_bytes: Option<usize>,

    /// Optional ceiling on any one grinding difficulty.
    #[arg(long = "whir-max-grinding-bits")]
    whir_max_grinding_bits: Option<usize>,

    /// Assess the batch's security and WHIR schedule, then exit without a witness or a proof.
    #[arg(long)]
    preflight: bool,
}

impl Args {
    /// The proof options: Flock's own commitment schedule, but for query grinding, unless the caller changes it.
    ///
    /// A ceiling the caller does not set is left open, so the report shows what the schedule costs.
    fn options(&self) -> BinaryProofOptions {
        let production = BinaryWhirBudget::PRODUCTION;
        BinaryProofOptions {
            whir: WhirOptions {
                regime: match self.whir_regime {
                    WhirRegimeOptions::UniqueDecoding => WhirRegime::UniqueDecoding,
                    WhirRegimeOptions::Johnson => WhirRegime::Johnson,
                },
                term_security_bits: self
                    .whir_term_security_bits
                    .unwrap_or(self.security_bits + 12),
                budget: BinaryWhirBudget {
                    max_stir_queries: self.whir_max_queries.unwrap_or(usize::MAX),
                    max_proof_bytes: self.whir_max_proof_bytes.unwrap_or(usize::MAX),
                    max_grinding_bits: self
                        .whir_max_grinding_bits
                        .unwrap_or(production.max_grinding_bits),
                },
                query_grinding: self.whir_query_grinding_bits,
                first_round: Some((self.whir_first_folding, self.whir_first_reduction)),
            },
            log_inv_rate: self.log_inv_rate,
            security_bits: self.security_bits,
            folding: self.folding,
            sumcheck_pow_bits: 0,
            hash: match self.hash {
                BinaryCommitmentHashOptions::Keccak256 => HashFamily::Keccak256,
                BinaryCommitmentHashOptions::Blake3 => HashFamily::Blake3,
            },
        }
    }
}

/// Prove `2^log_hashes` random instances of a gadget, the witness timed apart.
fn prove_hashes<G: Gadget + Sync>(
    log_hashes: usize,
    options: BinaryProofOptions,
    random: impl Fn(&mut SmallRng) -> G,
) -> Result<BinaryProofReport, String> {
    let mut rng = SmallRng::seed_from_u64(0xF10C);
    let rows = (0..1usize << log_hashes)
        .map(|_| random(&mut rng))
        .collect::<Vec<_>>();
    let circuit = G::circuit();
    let witness_start = Instant::now();
    let witness = G::witness(&circuit, log_hashes, &rows, &rows[0]);
    let witness_seconds = witness_start.elapsed().as_secs_f64();
    prove_circuit(&circuit, witness, options)
        .map(|report| report.with_witness_seconds(witness_seconds))
        .map_err(|error| format!("proof failed: {error}"))
}

/// Prints a status line where it cannot corrupt the JSON report on standard output.
fn status(format: OutputFormat, line: &str) {
    match format {
        OutputFormat::Human => println!("{line}"),
        OutputFormat::Json => eprintln!("{line}"),
    }
}

fn run(args: &Args) -> Result<(), String> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    // Exactly one of these two layers is installed.
    //
    // Both render the same span tree, only to a different stream.
    let json = args.format == OutputFormat::Json;
    Registry::default()
        .with(env_filter)
        .with((!json).then(ForestLayer::default))
        .with(json.then(|| ForestLayer::from(PrettyPrinter::new().writer(io::stderr))))
        .init();

    let log_hashes = usize::from(args.log_hashes);
    let options = args.options();
    if args.preflight {
        let circuit = match args.objective {
            BinaryHashOptions::Blake3Compressions => Blake3Compress::circuit(),
            BinaryHashOptions::Blake2sCompressions => Blake2sCompress::circuit(),
            BinaryHashOptions::Sha256Compressions => Sha256Compress::circuit(),
            BinaryHashOptions::KeccakFPermutations => KeccakF1600::circuit(),
        };
        let (security_bits, summary) = preflight_circuit(&circuit, log_hashes, options)
            .map_err(|error| format!("preflight failed: {error}"))?;
        status(
            args.format,
            &format!(
                "Preflight accepted: composed security {security_bits:.2} bits, schedule={summary:?}"
            ),
        );
        return Ok(());
    }
    status(
        args.format,
        &format!("Proving 2^{log_hashes} {:?}", args.objective),
    );
    let report = match args.objective {
        BinaryHashOptions::Blake3Compressions => {
            prove_hashes(log_hashes, options, |rng| Blake3Compress {
                cv: rng.random(),
                block: rng.random(),
                counter: rng.next_u64(),
                block_len: 64,
                flags: 11,
            })
        }
        BinaryHashOptions::Blake2sCompressions => {
            prove_hashes(log_hashes, options, |rng| Blake2sCompress {
                h: rng.random(),
                m: rng.random(),
                t: rng.next_u64(),
                f0: u32::MAX,
            })
        }
        BinaryHashOptions::Sha256Compressions => {
            prove_hashes(log_hashes, options, |rng| Sha256Compress {
                state: rng.random(),
                block: rng.random(),
            })
        }
        BinaryHashOptions::KeccakFPermutations => prove_hashes(log_hashes, options, |rng| {
            KeccakF1600(core::array::from_fn(|_| rng.next_u64()))
        }),
    }?;

    match args.format {
        OutputFormat::Human => {
            println!("{report}");
            println!("Proof Verified Successfully");
        }
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string(&report).expect("the report serializes")
        ),
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(error) = run(&args) {
        panic!("{error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_flocks_commitment_schedule() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-hashes",
            "2",
        ])
        .expect("minimal arguments parse");
        let options = args.options();

        // Johnson regime, twelve grinding bits, a first round folding six and dropping four.
        assert_eq!(options.whir.regime, WhirRegime::Johnson);
        assert_eq!(options.whir.query_grinding, 12);
        assert_eq!(options.whir.first_round, Some((6, 4)));
        assert_eq!(options.whir.term_security_bits, 112);
        assert_eq!(options.log_inv_rate, 1);
        assert_eq!(options.hash, HashFamily::Keccak256);
    }

    #[test]
    fn a_preflight_assesses_the_batch_and_stops_before_the_witness() {
        let preflight = |extra: &[&str]| {
            let args = Args::try_parse_from(
                [
                    "prove_hash_binary",
                    "--objective",
                    "blake-3-compressions",
                    "--log-hashes",
                    "4",
                    "--preflight",
                ]
                .iter()
                .chain(extra),
            )
            .expect("preflight arguments parse");
            let circuit = Blake3Compress::circuit();
            preflight_circuit(&circuit, usize::from(args.log_hashes), args.options())
        };

        // An attainable target is assessed and met.
        let (bits, _) = preflight(&["--security-bits", "100"]).expect("100 bits is attainable");
        assert!(bits >= 100.0);

        // Past the 32-byte hash's 128-bit cap, the assessment refuses the target.
        assert!(preflight(&["--security-bits", "129"]).is_err());

        // The opening carries two weighted claims, not one per port.
        //
        // This budget fits two claims and would refuse one per port.
        assert!(preflight(&["--whir-max-proof-bytes", "100000"]).is_ok());
    }

    #[test]
    fn the_old_size_flag_still_parses() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "keccak-f-permutations",
            "--log-trace-length",
            "5",
        ])
        .expect("the alias parses");
        assert_eq!(args.log_hashes, 5);
    }
}
