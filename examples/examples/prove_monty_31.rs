use std::marker::PhantomData;

use clap::Parser;
use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear, Poseidon2BabyBear};
use p3_blake3_air::Blake3Air;
use p3_dft::Radix2DitParallel;
use p3_examples::airs::ProofGoal;
use p3_examples::dfts::DFTs;
use p3_examples::parsers::{DFTOptions, FieldOptions, MerkleHashOptions, ProofObjectives};
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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The field to use for our proof.
    #[arg(short, long, ignore_case = true, value_enum)]
    field: FieldOptions,

    /// What we are trying to prove.
    #[arg(short, long, ignore_case = true, value_enum)]
    proof_objective: ProofObjectives,

    /// The log base 2 of the desired trace length.
    #[arg(short, long)]
    log_trace_length: u8,

    /// The discrete fourier transform to use in the proof.
    #[arg(short, long, ignore_case = true, value_enum)]
    discrete_fourier_transform: DFTOptions,

    /// The hash function to use when assembling the Merkle tree.
    #[arg(short, long, ignore_case = true, value_enum)]
    merkle_hash: MerkleHashOptions,
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

    let trace_height = 1 << args.log_trace_length;

    let num_hashes = match args.proof_objective {
        ProofObjectives::Blake3Permutations => {
            println!("Proving 2^{} Blake-3 permutations", {
                args.log_trace_length
            });
            trace_height
        }
        ProofObjectives::Poseidon2Permutations => {
            println!("Proving 2^{} native Poseidon-2 permutations", {
                args.log_trace_length + 3
            });
            trace_height << 3
        }
        ProofObjectives::KeccakFPermutations => {
            let num_hashes = trace_height / 24;
            println!("Proving {num_hashes} Keccak-F permutations");
            num_hashes
        }
    };

    match args.field {
        FieldOptions::KoalaBear => {
            type EF = BinomialExtensionField<KoalaBear, 4>;

            let proof_goal = match args.proof_objective {
                ProofObjectives::Blake3Permutations => ProofGoal::Blake3(Blake3Air {}),
                ProofObjectives::KeccakFPermutations => ProofGoal::Keccak(KeccakAir {}),
                ProofObjectives::Poseidon2Permutations => {
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
            };

            let dft = match args.discrete_fourier_transform {
                DFTOptions::RecursiveDft => DFTs::Recursive(RecursiveDft::new(trace_height << 1)),
                DFTOptions::Radix2DitParallel => DFTs::Parallel(Radix2DitParallel::default()),
            };

            match args.merkle_hash {
                MerkleHashOptions::KeccakF => {
                    let result =
                        prove_hashes_keccak(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                MerkleHashOptions::Poseidon2 => {
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
            };
        }
        FieldOptions::BabyBear => {
            type EF = BinomialExtensionField<BabyBear, 4>;

            let proof_goal = match args.proof_objective {
                ProofObjectives::Blake3Permutations => ProofGoal::Blake3(Blake3Air {}),
                ProofObjectives::KeccakFPermutations => ProofGoal::Keccak(KeccakAir {}),
                ProofObjectives::Poseidon2Permutations => {
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
            };

            let dft = match args.discrete_fourier_transform {
                DFTOptions::RecursiveDft => DFTs::Recursive(RecursiveDft::new(trace_height << 2)),
                DFTOptions::Radix2DitParallel => DFTs::Parallel(Radix2DitParallel::default()),
            };

            match args.merkle_hash {
                MerkleHashOptions::KeccakF => {
                    let result =
                        prove_hashes_keccak(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                MerkleHashOptions::Poseidon2 => {
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
            };
        }
    }
}
