use std::env;
use std::fmt::Debug;

use p3_blake3_air::Blake3Air;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_keccak_air::KeccakAir;
use p3_koala_bear::KoalaBear;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_monty_31::dft::{FastDFTs, RecursiveDft};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::random;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

fn main() -> Result<(), impl Debug> {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3, "There should be two arguments passed this functions.
    \n The first chooses which statement we are proving. Use \"B\" to prove Blake3 hashes and \"K\" to prove Keccak hashes.
    \n The second gives the choice of fourier transform. Use \"R\" for RecursiveDft and \"P\" for Radix2DitParallel.");

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = KoalaBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher32<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(Keccak256Hash {});

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let trace = match args[1].as_ref() {
        "B" | "Blake3" | "BLAKE3" => {
            const NUM_HASHES: usize = 1 << 13;
            println!("Proving {NUM_HASHES} Blake3 Hashes");
            let inputs = (0..NUM_HASHES).map(|_| random()).collect::<Vec<_>>();
            p3_blake3_air::generate_trace_rows::<Val>(inputs)
        }
        "K" | "Keccak" | "KECCAK" => {
            const NUM_HASHES: usize = 1365;
            println!("Proving {NUM_HASHES} KECCAK Hashes");
            let inputs = (0..NUM_HASHES).map(|_| random()).collect::<Vec<_>>();
            p3_keccak_air::generate_trace_rows::<Val>(inputs)
        }
        _ => {
            panic!("Could not understand the proof goal. Use \"B\" to prove Blake3 Hashes or \"K\" for Keccak hashes.")
        }
    };

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    type FastDft = FastDFTs<Val>;
    let dft = match args[2].as_ref() {
        "R" | "Recursive" => {
            println!("Using the RecursiveDft for our DFT");
            type Dft = RecursiveDft<Val>;
            let dft = Dft::new(trace.height() << fri_config.log_blowup);

            FastDFTs::<Val>::Recursive(dft)
        }
        "P" | "Parallel" => {
            println!("Using the Radix2DitParallel for our DFT");
            type Dft = Radix2DitParallel<Val>;
            let dft = Dft::default();

            FastDFTs::<Val>::Parallel(dft)
        }
        _ => {
            panic!("Could not understand the Fourier transform method. Use \"R\" for the Recursive DFT or \"P\" for the Parallel DFT.")
        }
    };

    type Pcs = TwoAdicFriPcs<Val, FastDft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut proof_challenger = Challenger::from_hasher(vec![], byte_hash);
    let mut verif_challenger = Challenger::from_hasher(vec![], byte_hash);

    match args[1].as_ref() {
        "B" | "Blake3" | "BLAKE3" => {
            let proof = prove(
                &config,
                &Blake3Air {},
                &mut proof_challenger,
                trace,
                &vec![],
            );

            verify(
                &config,
                &Blake3Air {},
                &mut verif_challenger,
                &proof,
                &vec![],
            )
        }
        "K" | "Keccak" | "KECCAK" => {
            let proof = prove(
                &config,
                &KeccakAir {},
                &mut proof_challenger,
                trace,
                &vec![],
            );

            verify(
                &config,
                &KeccakAir {},
                &mut verif_challenger,
                &proof,
                &vec![],
            )
        }
        _ => {
            panic!("Could not understand the proof goal. Use \"B\" to prove Blake3 Hashes or \"K\" for Keccak hashes.")
        }
    }
}
