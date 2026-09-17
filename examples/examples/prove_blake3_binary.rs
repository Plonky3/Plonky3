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

    /// Target for the union of every reduction and opening error, in bits.
    #[arg(long, default_value_t = 100)]
    security_bits: usize,

    /// Sequential variable folds batched between binary-PCS commitments.
    #[arg(long, default_value_t = 3)]
    folding: usize,
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
