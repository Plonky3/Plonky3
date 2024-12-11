use std::env;
use std::marker::PhantomData;

use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear, Poseidon2BabyBear};
use p3_blake3_air::Blake3Air;
use p3_dft::Radix2DitParallel;
use p3_examples::airs::ProofGoal;
use p3_examples::dfts::DFTOptions;
use p3_examples::proofs::{prove_hashes_keccak, prove_hashes_poseidon2, report_result};
use p3_field::extension::BinomialExtensionField;
use p3_keccak_air::KeccakAir;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear, Poseidon2KoalaBear};
use p3_monty_31::dft::RecursiveDft;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
use rand::thread_rng;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

// General constants for constructing the Poseidon2 AIR.
const P2_WIDTH: usize = 16;
const P2_HALF_FULL_ROUNDS: usize = 4;
const P2_VECTOR_LEN: usize = 1 << 3;

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len() - 1, 5, "There should be five arguments passed to this function.
    \n The first chooses the field we are using for the proof. Use \"B\" to use the BabyBear field and \"K\" to use the KoalaBear field.
    \n The second chooses which statement we are proving. Use \"B\", \"P\" or \"K\" to prove Blake3, Poseidon2 or Keccak hashes.
    \n The third gives the desired trace height. It should fit in a u8 and will be interpreted as the log base 2 of the trace height. For Blake3 this
    is the number of hashes, for Poseidon2 it's the number of hashes divided by 8 and for Keccak this is roughly the number of hashes times 24.
    \n The fourth gives the choice of fourier transform. Use \"R\" for RecursiveDft and \"P\" for Radix2DitParallel.
    \n The fifth gives the choice of hashing function for the merkle tree. Use \"K\" for Keccak and \"P\" for Poseidon2.");

    match args[1].as_ref() {
        "K" | "KoalaBear" | "KOALABEAR" => {
            println!("Choice of Field: KoalaBear");
        }
        "B" | "BabyBear" | "BABYBEAR" => {
            println!("Choice of Field: BabyBear");
        }
        _ => panic!("Could not understand the choice of field. Use \"B\" to prove using the BabyBear Field or \"K\" to use the Keccak Field.")
    };

    let log_2_trace_height = args[3]
        .parse::<u8>()
        .expect("The third command line input should be a u8");
    let trace_height = 1 << log_2_trace_height;

    let num_hashes = match args[2].as_ref() {
        "B" | "B3" | "Blake3" | "BLAKE3" => {
            println!("Proving 2^{log_2_trace_height} Blake3 Hashes");
            trace_height
        }
        "P" | "P2" | "Poseidon2" | "POSEIDON2" => {
            println!(
                "Proving 2^{} native Poseidon2 Hashes",
                log_2_trace_height + 3
            );
            trace_height << 3
        }
        "K" | "Keccak" | "KECCAK" => {
            let num_hashes = trace_height / 24;
            println!("Proving {num_hashes} KECCAK Hashes");
            num_hashes
        }
        _ => {
            panic!("Could not understand the proof goal. Use \"B\", \"P\" or \"K\" to prove Blake3, Poseidon2 or Keccak hashes.")
        }
    };

    match args[4].as_ref() {
        "R" | "Recursive" => {
            println!("Choice of DFT: RecursiveDft");
        }
        "P" | "Parallel" => {
            println!("Choice of DFT: Radix2DitParallel");
        }
        _ => {
            panic!("Could not understand the Fourier transform method. Use \"R\" for the Recursive DFT or \"P\" for the Parallel DFT.")
        }
    };

    match args[5].as_ref() {
        "K" | "Keccak" | "KECCAK" => {
            println!("Choice of Hash for Merkle Tree: Keccak");
        }
        "P" | "P2" | "Poseidon2" | "POSEIDON2" => {
            println!("Choice of Hash for Merkle Tree: Poseidon2");
        }
        _ => {
            panic!("Could not understand the desired merkle hash function. Use \"K\" or \"P\" to hash using Keccak or Poseidon2")
        }
    };

    match args[1].as_ref() {
        "K" | "KoalaBear" | "KOALABEAR" => {
            type EF = BinomialExtensionField<KoalaBear, 4>;

            let proof_goal = match args[2].as_ref() {
                "B" | "B3" | "Blake3" | "BLAKE3" => ProofGoal::Blake3(Blake3Air {}),
                "P" | "P2" | "Poseidon2" | "POSEIDON2" => {
                    let constants = RoundConstants::from_rng(&mut thread_rng());

                    // Field specific constants for constructing the Poseidon2 AIR.
                    const SBOX_DEGREE: u64 = 3;
                    const SBOX_REGISTERS: usize = 0;
                    const PARTIAL_ROUNDS: usize = 20;

                    let p2_air: VectorizedPoseidon2Air<
                        KoalaBear,
                        GenericPoseidon2LinearLayersKoalaBear,
                        P2_WIDTH,
                        SBOX_DEGREE,
                        SBOX_REGISTERS,
                        P2_HALF_FULL_ROUNDS,
                        PARTIAL_ROUNDS,
                        P2_VECTOR_LEN,
                    > = VectorizedPoseidon2Air::new(constants);
                    ProofGoal::Poseidon2(p2_air)
                }
                "K" | "Keccak" | "KECCAK" => ProofGoal::Keccak(KeccakAir {}),
                _ => unreachable!(),
            };

            let dft = match args[4].as_ref() {
                "R" | "Recursive" => DFTOptions::Recursive(RecursiveDft::new(trace_height << 1)),
                "P" | "Parallel" => DFTOptions::Parallel(Radix2DitParallel::default()),
                _ => unreachable!(),
            };

            match args[5].as_ref() {
                "K" | "Keccak" | "KECCAK" => {
                    let result =
                        prove_hashes_keccak(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                "P" | "P2" | "Poseidon2" | "POSEIDON2" => {
                    let perm16 = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut thread_rng());
                    let perm24 = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut thread_rng());
                    let result = prove_hashes_poseidon2(
                        proof_goal,
                        dft,
                        num_hashes,
                        perm16,
                        perm24,
                        PhantomData::<EF>,
                    );
                    report_result(result);
                }
                _ => unreachable!(),
            };
        }
        "B" | "BabyBear" | "BABYBEAR" => {
            type EF = BinomialExtensionField<BabyBear, 4>;

            let proof_goal = match args[2].as_ref() {
                "B" | "B3" | "Blake3" | "BLAKE3" => ProofGoal::Blake3(Blake3Air {}),
                "P" | "P2" | "Poseidon2" | "POSEIDON2" => {
                    let constants = RoundConstants::from_rng(&mut thread_rng());

                    // Field specific constants for constructing the Poseidon2 AIR.
                    const SBOX_DEGREE: u64 = 7;
                    const SBOX_REGISTERS: usize = 1;
                    const PARTIAL_ROUNDS: usize = 13;

                    let p2_air: VectorizedPoseidon2Air<
                        BabyBear,
                        GenericPoseidon2LinearLayersBabyBear,
                        P2_WIDTH,
                        SBOX_DEGREE,
                        SBOX_REGISTERS,
                        P2_HALF_FULL_ROUNDS,
                        PARTIAL_ROUNDS,
                        P2_VECTOR_LEN,
                    > = VectorizedPoseidon2Air::new(constants);
                    ProofGoal::Poseidon2(p2_air)
                }
                "K" | "Keccak" | "KECCAK" => ProofGoal::Keccak(KeccakAir {}),
                _ => unreachable!(),
            };

            let dft = match args[4].as_ref() {
                "R" | "Recursive" => DFTOptions::Recursive(RecursiveDft::new(trace_height << 2)),
                "P" | "Parallel" => DFTOptions::Parallel(Radix2DitParallel::default()),
                _ => unreachable!(),
            };

            match args[5].as_ref() {
                "K" | "Keccak" | "KECCAK" => {
                    let result =
                        prove_hashes_keccak(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                "P" | "P2" | "Poseidon2" | "POSEIDON2" => {
                    let perm16 = Poseidon2BabyBear::<16>::new_from_rng_128(&mut thread_rng());
                    let perm24 = Poseidon2BabyBear::<24>::new_from_rng_128(&mut thread_rng());
                    let result = prove_hashes_poseidon2(
                        proof_goal,
                        dft,
                        num_hashes,
                        perm16,
                        perm24,
                        PhantomData::<EF>,
                    );
                    report_result(result);
                }
                _ => unreachable!(),
            };
        }
        _ => unreachable!(),
    }
}
