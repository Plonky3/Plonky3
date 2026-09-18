use clap::Parser;
use p3_binary_field::BinaryField128;
use p3_blake3_air::Blake3BinaryAir;
use p3_examples::binary::{BinaryProofOptions, prove_binary_air};
use p3_matrix::Matrix;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The log base 2 of the desired trace length.
    #[arg(short, long)]
    log_trace_length: u8,

    /// Log of the inverse code rate for the binary PCS.
    #[arg(long, default_value_t = 2)]
    log_inv_rate: usize,

    /// Grinding bits the binary PCS demands once, before its query phase.
    #[arg(long, default_value_t = 0)]
    pcs_pow_bits: usize,

    /// Composed security target of the whole proof, in bits.
    ///
    /// Committing every cell as a `BinaryField128` element caps it at roughly
    /// 128 - (log-trace-length + ceil(log2(width)) + log-inv-rate + 3); PCS grinding does not
    /// raise that cap.
    #[arg(long, default_value_t = 100)]
    security_bits: usize,

    /// Sequential variable folds batched between binary-PCS commitments.
    #[arg(long, default_value_t = 3)]
    folding: usize,

    /// Number of children each Merkle-tree node compresses: 2 or 4.
    ///
    /// 4 trades larger authentication paths in the proof for fewer compressions per tree.
    #[arg(long, default_value_t = 2, value_parser = parse_merkle_arity)]
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
    let num_hashes = 1usize << args.log_trace_length;
    println!("Proving {num_hashes} Blake-3 compressions");

    let air = Blake3BinaryAir {};
    let trace = air.generate_random_trace_rows::<BinaryField128>(num_hashes, 0);
    assert_eq!(
        trace.height(),
        num_hashes,
        "generated trace height must match the requested log-trace-length"
    );

    let options = BinaryProofOptions {
        log_inv_rate: args.log_inv_rate,
        pcs_pow_bits: args.pcs_pow_bits,
        security_bits: args.security_bits,
        folding: args.folding,
        merkle_arity: args.merkle_arity,
        ..BinaryProofOptions::default()
    };

    match prove_binary_air(&air, trace, options) {
        Ok(report) => {
            println!("{report}");
            println!("Proof Verified Successfully");
        }
        Err(error) => panic!("{error:?}"),
    }
}
