use std::env;
use std::fmt::Debug;

use p3_baby_bear::BabyBear;
use p3_blake3_air::Blake3Air;
use p3_challenger::SerializingChallenger32;
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_keccak_air::KeccakAir;
use p3_koala_bear::KoalaBear;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_monty_31::dft::RecursiveDft;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::random;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

fn prove_hashes<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<F>,
>(
    args: Vec<String>,
    dft: DFT,
    num_hashes: usize,
) -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    // type Val = KoalaBear;
    // type Challenge = BinomialExtensionField<KoalaBear, 4>;

    // type ByteHash = Keccak256Hash;
    // type FieldHash = SerializingHasher32<ByteHash>;
    let byte_hash = Keccak256Hash {};
    let field_hash = SerializingHasher32::new(Keccak256Hash {});

    // type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = CompressionFunctionFromHasher::<_, 2, 32>::new(byte_hash);

    // type ValMmcs = MerkleTreeMmcs<KoalaBear, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = MerkleTreeMmcs::<F, u8, _, _, 32>::new(field_hash, compress);

    // type ChallengeMmcs = ExtensionMmcs<KoalaBear, BinomialExtensionField<KoalaBear, 4>, ValMmcs>;
    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());

    // type Challenger = SerializingChallenger32<KoalaBear, HashChallenger<u8, ByteHash, 32>>;

    let trace = match args[2].as_ref() {
        "B" | "Blake3" | "BLAKE3" => {
            println!("Proving {num_hashes} Blake3 Hashes");
            let inputs = (0..num_hashes).map(|_| random()).collect::<Vec<_>>();
            p3_blake3_air::generate_trace_rows(inputs)
        }
        "K" | "Keccak" | "KECCAK" => {
            println!("Proving {num_hashes} KECCAK Hashes");
            let inputs = (0..num_hashes).map(|_| random()).collect::<Vec<_>>();
            p3_keccak_air::generate_trace_rows(inputs)
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

    // type Pcs = TwoAdicFriPcs<KoalaBear, FastDft, ValMmcs, ChallengeMmcs>;
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);

    // type MyConfig = StarkConfig<Pcs, BinomialExtensionField<KoalaBear, 4>, Challenger>;
    let config = StarkConfig::new(pcs);

    let mut proof_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
    let mut verif_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);

    match args[2].as_ref() {
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

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 5, "There should be three arguments passed this functions.
    \n The first chooses the field we are using for the proof. Use \"B\" to use the BabyBear field and \"K\" to use the KoalaBear field.
    \n The second chooses which statement we are proving. Use \"B\" to prove Blake3 hashes and \"K\" to prove Keccak hashes.
    \n The third gives the number of Hashes. It should fit in a u8 and will be interpreted as the log base 2 of the trace height. For Blake3 this
    is equal to the number of hashes but for Keccak this is roughly the number of hashes times 24.
    \n The fourth gives the choice of fourier transform. Use \"R\" for RecursiveDft and \"P\" for Radix2DitParallel.");

    let log_2_hashes = args[3]
        .parse::<u8>()
        .expect("The third command line input should be a u8");
    let num_hashes = 1 << log_2_hashes;

    match args[1].as_ref() {
        "K" | "KoalaBear" | "KOALABEAR" => {
            println!("Proving using the KoalaBear Field.");

            match args[4].as_ref() {
                "R" | "Recursive" => {
                    println!("Using the RecursiveDft for our DFT");
                    let dft = RecursiveDft::new(num_hashes << 1);
                    let result = prove_hashes::<KoalaBear, BinomialExtensionField<KoalaBear, 4>, _>(
                        args, dft, num_hashes,
                    );
                    if result.is_ok() {
                        println!("Proof Verified Successfully")
                    } else {
                        println!("{:?}", result.unwrap_err())
                    }
                }
                "P" | "Parallel" => {
                    println!("Using the Radix2DitParallel for our DFT");
                    let dft = Radix2DitParallel::default();
                    let result = prove_hashes::<KoalaBear, BinomialExtensionField<KoalaBear, 4>, _>(
                        args, dft, num_hashes,
                    );
                    if result.is_ok() {
                        println!("Proof Verified Successfully")
                    } else {
                        println!("{:?}", result.unwrap_err())
                    }
                }
                _ => {
                    panic!("Could not understand the Fourier transform method. Use \"R\" for the Recursive DFT or \"P\" for the Parallel DFT.")
                }
            };
        }
        "B" | "BabyBear" | "BABYBEAR" => {
            match args[4].as_ref() {
                "R" | "Recursive" => {
                    println!("Using the RecursiveDft for our DFT");
                    let dft = RecursiveDft::new(num_hashes << 1);
                    let result = prove_hashes::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
                        args, dft, num_hashes,
                    );
                    if result.is_ok() {
                        println!("Proof Verified Successfully")
                    } else {
                        println!("{:?}", result.unwrap_err())
                    }
                }
                "P" | "Parallel" => {
                    println!("Using the Radix2DitParallel for our DFT");
                    let dft = Radix2DitParallel::default();
                    let result = prove_hashes::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
                        args, dft, num_hashes,
                    );
                    if result.is_ok() {
                        println!("Proof Verified Successfully")
                    } else {
                        println!("{:?}", result.unwrap_err())
                    }
                }
                _ => {
                    panic!("Could not understand the Fourier transform method. Use \"R\" for the Recursive DFT or \"P\" for the Parallel DFT.")
                }
            };
        }
        _ => {
            panic!("Could not understand the choice of field. Use \"B\" to prove using the BabyBear Field or \"K\" to use the Keccak Field.")
        }
    }
}
