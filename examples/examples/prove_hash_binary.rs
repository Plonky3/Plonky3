use clap::Parser;
use p3_binary_dft::{LchNtt, NaiveAdditiveNtt, PolyBasisNtt};
use p3_binary_field::BinaryField128;
use p3_blake3_air::Blake3BinaryAir;
use p3_examples::binary::{
    AdditiveNttChoice, Backend, BinaryProofOptions, prove_binary_air_with_ntt_and_backend,
    prove_boolean_air_with_backend,
};
use p3_examples::parsers::{BinaryHashOptions, NttOptions, RepresentationOptions};
use p3_keccak_air::{KECCAK_BINARY_ROWS_PER_PERM, KeccakBinaryAir};
use p3_matrix::Matrix;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// What we are trying to prove.
    #[arg(short, long, ignore_case = true, value_enum)]
    objective: BinaryHashOptions,

    /// The log base 2 of the desired trace length.
    #[arg(short, long)]
    log_trace_length: u8,

    /// The additive NTT used to encode the binary-PCS codeword.
    ///
    /// Keccak-f commits one field element per trace cell and accepts every choice. BLAKE3 commits
    /// its trace as bits through the Boolean commitment, which encodes through the
    /// polynomial-basis NTT only.
    #[arg(short, long, ignore_case = true, value_enum, default_value_t = NttOptions::PolyBasis)]
    ntt: NttOptions,

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
    /// Keccak-f commits every cell as a `BinaryField128` element, which caps it at roughly
    /// 128 - (log-trace-length + ceil(log2(width)) + log-inv-rate + 3); PCS grinding does not
    /// raise that cap.
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
}

/// Parses a `--merkle-arity` value, rejecting anything but 2 or 4.
fn parse_merkle_arity(arg: &str) -> Result<usize, String> {
    match arg.parse::<usize>() {
        Ok(arity @ (2 | 4)) => Ok(arity),
        Ok(arity) => Err(format!("merkle arity must be 2 or 4, got {arity}")),
        Err(_) => Err(format!("invalid merkle arity: {arg}")),
    }
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
    let trace_height = 1usize << args.log_trace_length;

    let ntt = match args.ntt {
        NttOptions::PolyBasis => AdditiveNttChoice::PolyBasis(PolyBasisNtt::default()),
        NttOptions::Lch => AdditiveNttChoice::Lch(LchNtt::default()),
        NttOptions::Naive => AdditiveNttChoice::Naive(NaiveAdditiveNtt::default()),
    };

    let backend = match args.representation {
        RepresentationOptions::Auto => Backend::preferred(),
        RepresentationOptions::Subfield => Backend::Subfield,
        RepresentationOptions::PolyBasis => Backend::PolyBasis,
    };

    let options = BinaryProofOptions {
        log_inv_rate: args.log_inv_rate,
        pcs_pow_bits: args.pcs_pow_bits,
        security_bits: args.security_bits,
        folding: args.folding,
        merkle_arity: args.merkle_arity,
        ..BinaryProofOptions::default()
    };

    let result = match args.objective {
        BinaryHashOptions::KeccakFPermutations => {
            assert!(
                trace_height >= KECCAK_BINARY_ROWS_PER_PERM,
                "2^{} = {trace_height} rows does not fit one {KECCAK_BINARY_ROWS_PER_PERM}-row \
                 Keccak-f permutation; raise --log-trace-length",
                args.log_trace_length,
            );
            let num_hashes = trace_height / KECCAK_BINARY_ROWS_PER_PERM;
            println!("Proving {num_hashes} Keccak-f permutations");

            let air = KeccakBinaryAir {};
            let trace = air.generate_random_trace_rows::<BinaryField128>(num_hashes, 0);
            assert_eq!(
                trace.height(),
                trace_height,
                "generated trace height must match the requested log-trace-length"
            );
            prove_binary_air_with_ntt_and_backend(&air, trace, options, ntt, backend)
        }
        BinaryHashOptions::Blake3Compressions => {
            assert_eq!(
                args.ntt,
                NttOptions::PolyBasis,
                "BLAKE3 is committed as bits, whose codeword encodes through the polynomial-basis \
                 NTT only; drop --ntt"
            );
            println!("Proving {trace_height} Blake-3 compressions");

            // Every cell is a bit and no constraint reads the next row, so the trace commits as
            // bits.
            let air = Blake3BinaryAir {};
            let trace = air.generate_random_trace_rows::<BinaryField128>(trace_height, 0);
            assert_eq!(
                trace.height(),
                trace_height,
                "generated trace height must match the requested log-trace-length"
            );
            prove_boolean_air_with_backend(&air, trace, options, backend)
        }
    };

    match result {
        Ok(report) => {
            println!("{report}");
            println!("Proof Verified Successfully");
        }
        Err(error) => panic!("{error}"),
    }
}
