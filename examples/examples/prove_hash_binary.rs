use clap::{Parser, ValueEnum};
use p3_binary_field::{BinaryField128, Gf2};
use p3_blake3_air::Blake3BinaryAir;
use p3_examples::binary::{
    Backend, BinaryProofOptions, BinaryWhirBudget, BooleanPcsChoice, HashFamily, WhirOptions,
    WhirRegime, WhirSummary, preflight_boolean_air_with_summary, prove_boolean_air_with_backend,
};
use p3_examples::parsers::{BinaryCommitmentHashOptions, BinaryHashOptions, RepresentationOptions};
use p3_keccak_air::{KECCAK_BINARY_ROWS_PER_PERM, KeccakBinaryAir, NUM_KECCAK_BINARY_COLS};
use p3_matrix::Matrix;
use p3_sha256_air::{NUM_SHA256_BINARY_COLS, Sha256BinaryAir};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::Table;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PcsOptions {
    #[value(alias = "fold")]
    Folding,
    Whir,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhirRegimeOptions {
    #[value(alias = "unique")]
    UniqueDecoding,
    Johnson,
    Capacity,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// What we are trying to prove.
    #[arg(short, long, ignore_case = true, value_enum)]
    objective: BinaryHashOptions,

    /// The log base 2 of the desired trace length.
    #[arg(short, long)]
    log_trace_length: u8,

    /// The field representation the zerocheck prover runs its later rounds in.
    ///
    /// Every choice proves and verifies the same statement and emits a byte-identical proof;
    /// this only trades off performance. `auto` picks polynomial basis when the build has a
    /// hardware carryless multiply, and subfield-tower basis otherwise.
    #[arg(short, long, ignore_case = true, value_enum, default_value_t = RepresentationOptions::Auto)]
    representation: RepresentationOptions,

    /// Log of the inverse code rate for the binary PCS.
    #[arg(long, default_value_t = 1)]
    log_inv_rate: usize,

    /// Grinding bits the binary PCS demands once, before its query phase.
    #[arg(long, default_value_t = 0)]
    pcs_pow_bits: usize,

    /// Composed security target of the whole proof, in bits.
    ///
    /// The Boolean commitment packs the trace's bits into `BinaryField128` elements, which caps
    /// it at roughly 125 - (log-trace-length + ceil(log2(width)) - 7 + log-inv-rate) on the
    /// binary PCS over those packed elements; PCS grinding does not raise that cap.
    #[arg(long, default_value_t = 100)]
    security_bits: usize,

    /// Sequential variable folds batched between binary-PCS commitments.
    #[arg(long, default_value_t = 4)]
    folding: usize,

    /// Number of children each Merkle-tree node compresses: 2 or 4.
    ///
    /// 4 trades larger authentication paths in the proof for fewer compressions per tree.
    #[arg(long, default_value_t = 4, value_parser = parse_merkle_arity)]
    merkle_arity: usize,

    /// The byte hash the Merkle trees and the Fiat-Shamir transcript share.
    ///
    /// Both emit a 32-byte digest and are capped at the same collision resistance, so the
    /// composed security does not move between them. The proof bytes do.
    #[arg(long, ignore_case = true, value_enum, default_value_t = BinaryCommitmentHashOptions::Keccak256)]
    hash: BinaryCommitmentHashOptions,

    /// Field elements each Merkle leaf packs; defaults to one fold batch's coset.
    ///
    /// A wider leaf shortens the tree and hands the hash longer messages, and pays for it in
    /// proof bytes: a query then authenticates symbols it did not ask for, and those symbols
    /// travel in the opening.
    #[arg(long, value_parser = parse_leaf_elements)]
    leaf_elements: Option<usize>,

    /// Boolean PCS used for the trace commitment.
    #[arg(long, value_enum, default_value_t = PcsOptions::Folding)]
    pcs: PcsOptions,

    /// WHIR proximity regime. Required when `--pcs whir` is selected.
    #[arg(long = "whir-regime", value_enum, visible_alias = "regime")]
    whir_regime: Option<WhirRegimeOptions>,

    /// Per-term WHIR security target. Required when `--pcs whir` is selected.
    #[arg(long = "whir-term-security-bits", visible_alias = "term-security-bits")]
    whir_term_security_bits: Option<usize>,

    /// Optional WHIR verifier query ceiling.
    #[arg(
        long = "whir-max-queries",
        visible_aliases = ["whir-max-stir-queries", "max-queries", "max-stir-queries"]
    )]
    whir_max_stir_queries: Option<usize>,

    /// Optional serialized WHIR proof byte ceiling.
    #[arg(long = "whir-max-proof-bytes", visible_alias = "max-proof-bytes")]
    whir_max_proof_bytes: Option<usize>,

    /// Optional WHIR maximum grinding difficulty, including zero.
    #[arg(long = "whir-max-grinding-bits", visible_alias = "max-grinding-bits")]
    whir_max_grinding_bits: Option<usize>,

    /// Run configuration/security preflight and exit without generating a witness.
    #[arg(long)]
    preflight: bool,
}

impl Args {
    fn proof_options(&self) -> Result<BinaryProofOptions, String> {
        let common = BinaryProofOptions {
            log_inv_rate: self.log_inv_rate,
            pcs_pow_bits: self.pcs_pow_bits,
            security_bits: self.security_bits,
            folding: self.folding,
            merkle_arity: self.merkle_arity,
            hash: match self.hash {
                BinaryCommitmentHashOptions::Keccak256 => HashFamily::Keccak256,
                BinaryCommitmentHashOptions::Blake3 => HashFamily::Blake3,
            },
            leaf_elements: self.leaf_elements,
            ..BinaryProofOptions::default()
        };
        let has_whir_fields = self.whir_regime.is_some()
            || self.whir_term_security_bits.is_some()
            || self.whir_max_stir_queries.is_some()
            || self.whir_max_proof_bytes.is_some()
            || self.whir_max_grinding_bits.is_some();

        match self.pcs {
            PcsOptions::Folding if has_whir_fields => {
                Err("WHIR options require --pcs whir".to_string())
            }
            PcsOptions::Folding => Ok(common),
            PcsOptions::Whir => {
                if self.merkle_arity != 2 {
                    return Err("WHIR requires --merkle-arity 2".to_string());
                }
                let regime = match self.whir_regime {
                    None => return Err("--whir-regime is required with --pcs whir".to_string()),
                    Some(WhirRegimeOptions::Capacity) => {
                        return Err("WHIR capacity regime is unsupported".to_string());
                    }
                    Some(WhirRegimeOptions::UniqueDecoding) => WhirRegime::UniqueDecoding,
                    Some(WhirRegimeOptions::Johnson) => WhirRegime::Johnson,
                };
                let term_security_bits = self.whir_term_security_bits.ok_or_else(|| {
                    "--whir-term-security-bits is required with --pcs whir".to_string()
                })?;
                let production = BinaryWhirBudget::PRODUCTION;
                let budget = BinaryWhirBudget {
                    max_stir_queries: self
                        .whir_max_stir_queries
                        .unwrap_or(production.max_stir_queries),
                    max_proof_bytes: self
                        .whir_max_proof_bytes
                        .unwrap_or(production.max_proof_bytes),
                    max_grinding_bits: self
                        .whir_max_grinding_bits
                        .unwrap_or(production.max_grinding_bits),
                };
                Ok(BinaryProofOptions {
                    merkle_arity: self.merkle_arity,
                    pcs: BooleanPcsChoice::Whir(WhirOptions {
                        regime,
                        term_security_bits,
                        budget,
                    }),
                    ..common
                })
            }
        }
    }
}

/// Parses a `--merkle-arity` value, rejecting anything but 2 or 4.
fn parse_merkle_arity(arg: &str) -> Result<usize, String> {
    match arg.parse::<usize>() {
        Ok(arity @ (2 | 4)) => Ok(arity),
        Ok(arity) => Err(format!("merkle arity must be 2 or 4, got {arity}")),
        Err(_) => Err(format!("invalid merkle arity: {arg}")),
    }
}

/// Parses a `--leaf-elements` value, rejecting anything but a power of two.
fn parse_leaf_elements(arg: &str) -> Result<usize, String> {
    match arg.parse::<usize>() {
        Ok(elements) if elements.is_power_of_two() => Ok(elements),
        Ok(elements) => Err(format!(
            "leaf elements must be a power of two, got {elements}"
        )),
        Err(_) => Err(format!("invalid leaf element count: {arg}")),
    }
}

fn requested_shape(
    objective: BinaryHashOptions,
    log_trace_length: u8,
) -> Result<(usize, TableShape), String> {
    let log_height = log_trace_length as usize;
    let trace_height = 1usize
        .checked_shl(log_trace_length as u32)
        .ok_or_else(|| format!("log trace length {log_trace_length} does not fit this platform"))?;
    let width = match objective {
        BinaryHashOptions::Blake3Compressions => p3_blake3_air::NUM_BLAKE3_BINARY_COLS,
        BinaryHashOptions::KeccakFPermutations => {
            if trace_height < KECCAK_BINARY_ROWS_PER_PERM {
                return Err(format!(
                    "2^{log_trace_length} = {trace_height} rows does not fit one {KECCAK_BINARY_ROWS_PER_PERM}-row Keccak-f permutation; raise --log-trace-length"
                ));
            }
            NUM_KECCAK_BINARY_COLS
        }
        BinaryHashOptions::Sha256Compressions => NUM_SHA256_BINARY_COLS,
    };
    Ok((trace_height, TableShape::new(log_height, width)))
}

fn preflight_then_maybe_prove<A, Generate>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
    backend: Backend,
    preflight_only: bool,
    generate: Generate,
) -> Result<Option<p3_examples::binary::BinaryProofReport>, String>
where
    A: p3_examples::binary::BinaryAir,
    Generate: FnOnce() -> Table<BinaryField128>,
{
    let selected_whir = matches!(options.pcs, BooleanPcsChoice::Whir(_));
    if selected_whir || preflight_only {
        let (security_bits, summary) = preflight_boolean_air_with_summary(air, shape, options)
            .map_err(|error| format!("Boolean PCS preflight failed: {error}"))?;
        print_preflight_result(options, security_bits, summary);
        if preflight_only {
            return Ok(None);
        }
    }
    let trace = generate();
    prove_boolean_air_with_backend(air, trace, options, backend)
        .map(Some)
        .map_err(|error| format!("proof failed: {error}"))
}

fn print_preflight_result(
    options: BinaryProofOptions,
    security_bits: f64,
    summary: Option<WhirSummary>,
) {
    match options.pcs {
        BooleanPcsChoice::Folding => {
            println!("Preflight accepted: folding, composed security {security_bits:.2} bits");
        }
        BooleanPcsChoice::Whir(whir) => {
            let summary = summary.expect("WHIR preflight must return its schedule summary");
            println!(
                "Preflight accepted: WHIR {:?}, composed security {security_bits:.2} bits, schedule={summary:?}",
                whir.regime
            );
        }
    }
}

fn run(args: &Args) -> Result<(), String> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let options = args.proof_options()?;
    let (trace_height, shape) = requested_shape(args.objective, args.log_trace_length)?;

    let backend = match args.representation {
        RepresentationOptions::Auto => Backend::preferred(),
        RepresentationOptions::Subfield => Backend::Subfield,
        RepresentationOptions::PolyBasis => Backend::PolyBasis,
    };

    // The Boolean commitment cannot represent a cell outside `{0, 1}`, so booleanity constraints
    // add nothing to soundness under it. Keccak-f drops them, as they are half of its constraints.
    // Blake-3 and SHA-256 constrain only their input cells, a small share of their constraints,
    // and keep them.
    let result = match args.objective {
        BinaryHashOptions::KeccakFPermutations => {
            let num_hashes = trace_height / KECCAK_BINARY_ROWS_PER_PERM;
            let air = KeccakBinaryAir::assuming_boolean_trace();
            preflight_then_maybe_prove(&air, shape, options, backend, args.preflight, || {
                println!("Proving {num_hashes} Keccak-f permutations");
                let words = air.generate_random_trace_packed::<Gf2>(num_hashes);
                assert_eq!(
                    words.height(),
                    trace_height.div_ceil(64),
                    "generated trace height must match the requested log-trace-length {}",
                    args.log_trace_length
                );
                Table::<BinaryField128>::from_packed_bits(words, args.log_trace_length as usize)
            })
        }
        BinaryHashOptions::Blake3Compressions => {
            let air = Blake3BinaryAir::default();
            preflight_then_maybe_prove(&air, shape, options, backend, args.preflight, || {
                println!("Proving {trace_height} Blake-3 compressions");
                let words = air.generate_random_trace_packed::<Gf2>(trace_height);
                let trace = Table::<BinaryField128>::from_packed_bits(
                    words,
                    args.log_trace_length as usize,
                );
                assert_eq!(
                    trace.num_variables(),
                    args.log_trace_length as usize,
                    "generated trace height must match the requested log-trace-length"
                );
                trace
            })
        }
        BinaryHashOptions::Sha256Compressions => {
            let air = Sha256BinaryAir::default();
            preflight_then_maybe_prove(&air, shape, options, backend, args.preflight, || {
                println!("Proving {trace_height} SHA-256 compressions");
                let words = air.generate_random_trace_packed::<Gf2>(trace_height);
                let trace = Table::<BinaryField128>::from_packed_bits(
                    words,
                    args.log_trace_length as usize,
                );
                assert_eq!(
                    trace.num_variables(),
                    args.log_trace_length as usize,
                    "generated trace height must match the requested log-trace-length"
                );
                trace
            })
        }
    };

    match result {
        Ok(Some(report)) => {
            println!("{report}");
            println!("Proof Verified Successfully");
        }
        Ok(None) => {}
        Err(error) => return Err(error),
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
    fn cli_defaults_select_the_fast_binary_pcs_parameters() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
        ])
        .expect("minimal CLI arguments parse");
        assert_eq!(args.log_inv_rate, 1);
        assert_eq!(args.folding, 4);
        assert_eq!(args.merkle_arity, 4);
        assert_eq!(args.hash, BinaryCommitmentHashOptions::Keccak256);
        assert_eq!(args.leaf_elements, None);
        assert_eq!(args.representation, RepresentationOptions::Auto);
        assert_eq!(args.pcs, PcsOptions::Folding);
        assert_eq!(args.pcs_pow_bits, 0);
        assert_eq!(args.security_bits, 100);
        assert!(!args.preflight);
        assert!(args.whir_regime.is_none());
        assert!(args.whir_term_security_bits.is_none());
        assert_eq!(args.proof_options().unwrap().merkle_arity, 4);
    }

    #[test]
    fn cli_selects_the_hash_and_the_leaf_geometry() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--hash",
            "blake-3",
            "--leaf-elements",
            "64",
        ])
        .expect("the hash and leaf-geometry arguments parse");
        assert_eq!(args.hash, BinaryCommitmentHashOptions::Blake3);
        assert_eq!(args.leaf_elements, Some(64));
    }

    #[test]
    fn cli_rejects_a_leaf_size_that_is_not_a_power_of_two() {
        assert!(
            Args::try_parse_from([
                "prove_hash_binary",
                "--objective",
                "blake-3-compressions",
                "--log-trace-length",
                "2",
                "--leaf-elements",
                "48",
            ])
            .is_err()
        );
    }

    #[test]
    fn cli_selects_whir_unique_and_johnson_regimes() {
        for regime in ["unique-decoding", "johnson"] {
            let args = Args::try_parse_from([
                "prove_hash_binary",
                "--objective",
                "blake-3-compressions",
                "--log-trace-length",
                "2",
                "--pcs",
                "whir",
                "--merkle-arity",
                "2",
                "--whir-regime",
                regime,
                "--whir-term-security-bits",
                "98",
            ])
            .expect("explicit WHIR regime parses");
            let options = args.proof_options().expect("explicit WHIR options convert");
            let BooleanPcsChoice::Whir(whir) = options.pcs else {
                panic!("WHIR selection must produce WHIR options")
            };
            assert_eq!(whir.term_security_bits, 98);
            assert_eq!(
                whir.regime,
                if regime == "unique-decoding" {
                    WhirRegime::UniqueDecoding
                } else {
                    WhirRegime::Johnson
                }
            );
            assert_eq!(whir.budget, BinaryWhirBudget::PRODUCTION);
        }
    }

    #[test]
    fn cli_requires_whir_regime_and_term_target() {
        let missing_regime = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--merkle-arity",
            "2",
            "--whir-term-security-bits",
            "98",
        ])
        .expect("missing regime remains a conversion error");
        assert!(missing_regime.proof_options().is_err());

        let missing_term = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--merkle-arity",
            "2",
            "--whir-regime",
            "unique",
        ])
        .expect("missing term target remains a conversion error");
        assert!(missing_term.proof_options().is_err());
    }

    #[test]
    fn cli_rejects_unsupported_capacity_and_whir_flags_with_folding() {
        let capacity = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--merkle-arity",
            "2",
            "--whir-regime",
            "capacity",
            "--whir-term-security-bits",
            "98",
        ])
        .expect("capacity parses as an explicit unsupported mode");
        assert!(capacity.proof_options().is_err());

        let folding_flags = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--whir-regime",
            "unique",
            "--whir-term-security-bits",
            "98",
        ])
        .expect("WHIR flags parse before PCS combination validation");
        assert!(folding_flags.proof_options().is_err());
    }

    #[test]
    fn cli_rejects_whir_default_merkle_arity_before_preflight() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--whir-regime",
            "unique-decoding",
            "--whir-term-security-bits",
            "102",
        ])
        .expect("omitted arity parses with the folding default");
        let error = args
            .proof_options()
            .expect_err("WHIR must reject the omitted arity before preflight");
        assert!(error.contains("--merkle-arity 2"));
    }

    #[test]
    fn cli_accepts_zero_grinding_cap_and_rejects_invalid_geometry() {
        let zero_grind = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--merkle-arity",
            "2",
            "--whir-regime",
            "unique",
            "--whir-term-security-bits",
            "98",
            "--whir-max-grinding-bits",
            "0",
        ])
        .expect("zero grinding is a valid budget override");
        let options = zero_grind.proof_options().expect("zero grinding converts");
        let BooleanPcsChoice::Whir(whir) = options.pcs else {
            panic!("WHIR selection must produce WHIR options")
        };
        assert_eq!(whir.budget.max_grinding_bits, 0);

        assert!(
            Args::try_parse_from([
                "prove_hash_binary",
                "--objective",
                "blake-3-compressions",
                "--log-trace-length",
                "2",
                "--merkle-arity",
                "3",
            ])
            .is_err()
        );
    }

    #[test]
    fn cli_validates_trace_height_before_generation() {
        assert!(requested_shape(BinaryHashOptions::Blake3Compressions, u8::MAX).is_err());
        let error = requested_shape(BinaryHashOptions::KeccakFPermutations, 4)
            .expect_err("Keccak must reject a height smaller than one permutation");
        assert!(error.contains("does not fit one"));
    }

    #[test]
    fn cli_preflight_never_invokes_the_witness_generator() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--merkle-arity",
            "2",
            "--whir-regime",
            "unique",
            "--whir-term-security-bits",
            "102",
            "--preflight",
        ])
        .expect("preflight arguments parse");
        let options = args.proof_options().expect("preflight options convert");
        let air = Blake3BinaryAir::default();
        let shape = TableShape::new(2, p3_blake3_air::NUM_BLAKE3_BINARY_COLS);
        let result =
            preflight_then_maybe_prove(&air, shape, options, Backend::preferred(), true, || {
                panic!("preflight must not invoke the witness generator")
            })
            .expect("accepted preflight returns successfully");
        assert!(result.is_none());
    }

    #[test]
    fn cli_rejected_preflight_budget_never_invokes_the_witness_generator() {
        let args = Args::try_parse_from([
            "prove_hash_binary",
            "--objective",
            "blake-3-compressions",
            "--log-trace-length",
            "2",
            "--pcs",
            "whir",
            "--merkle-arity",
            "2",
            "--whir-regime",
            "unique",
            "--whir-term-security-bits",
            "102",
            "--whir-max-queries",
            "0",
        ])
        .expect("budget arguments parse");
        let options = args.proof_options().expect("budget options convert");
        let air = Blake3BinaryAir::default();
        let shape = TableShape::new(2, p3_blake3_air::NUM_BLAKE3_BINARY_COLS);
        let error =
            preflight_then_maybe_prove(&air, shape, options, Backend::preferred(), false, || {
                panic!("rejected preflight must not invoke the witness generator")
            })
            .expect_err("the zero-query budget must be rejected before generation");
        assert!(error.contains("preflight"));
    }
}
