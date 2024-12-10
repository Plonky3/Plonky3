use std::env;
use std::fmt::Debug;
use std::marker::PhantomData;

use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_blake3_air::Blake3Air;
use p3_challenger::SerializingChallenger32;
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeField32, TwoAdicField};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_keccak_air::KeccakAir;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_monty_31::dft::RecursiveDft;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher32To64};
use p3_uni_stark::{prove, verify, StarkConfig, SymbolicExpression};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

// General constants for constructing the Poseidon2 AIR.
const P2_WIDTH: usize = 16;
const P2_HALF_FULL_ROUNDS: usize = 4;
const P2_VECTOR_LEN: usize = 1 << 3;

enum ProofGoal<
    F: Field,
    LinearLayers,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
> {
    Blake3(Blake3Air),
    Keccak(KeccakAir),
    Poseidon2(
        VectorizedPoseidon2Air<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            VECTOR_LEN,
        >,
    ),
}

fn prove_hashes<
    F: PrimeField32 + TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<F>,
    LinearLayers: GenericPoseidon2LinearLayers<F, P2_WIDTH>
        + GenericPoseidon2LinearLayers<SymbolicExpression<F>, P2_WIDTH>
        + GenericPoseidon2LinearLayers<F::Packing, P2_WIDTH>
        + GenericPoseidon2LinearLayers<EF, P2_WIDTH>,
    const P2_SBOX_DEGREE: u64,
    const P2_SBOX_REGISTERS: usize,
    const P2_PARTIAL_ROUNDS: usize,
>(
    proof_goal: ProofGoal<
        F,
        LinearLayers,
        P2_WIDTH,
        P2_SBOX_DEGREE,
        P2_SBOX_REGISTERS,
        P2_HALF_FULL_ROUNDS,
        P2_PARTIAL_ROUNDS,
        P2_VECTOR_LEN,
    >,
    dft: DFT,
    num_hashes: usize,
    _phantom: PhantomData<EF>, // A simple workaround allowing the compiler to determine all generic parameters
) -> Result<(), impl Debug>
where
    Standard: Distribution<F>,
{
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let byte_hash = Keccak256Hash {};
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
    let field_hash = SerializingHasher32To64::new(u64_hash);

    let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

    let val_mmcs =
        MerkleTreeMmcs::<[F; p3_keccak::VECTOR_LEN], [u64; p3_keccak::VECTOR_LEN], _, _, 4>::new(
            field_hash, compress,
        );

    let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());

    let trace = match proof_goal {
        ProofGoal::Blake3(_) => {
            let inputs = (0..num_hashes)
                .map(|_| random::<[u32; 24]>())
                .collect::<Vec<_>>();
            p3_blake3_air::generate_trace_rows(inputs)
        }
        ProofGoal::Poseidon2(ref p2_air) => {
            let inputs = (0..num_hashes)
                .map(|_| random::<[F; P2_WIDTH]>())
                .collect::<Vec<_>>();
            p2_air.generate_vectorized_trace_rows(inputs)
        }
        ProofGoal::Keccak(_) => {
            let inputs = (0..num_hashes)
                .map(|_| random::<[u64; 25]>())
                .collect::<Vec<_>>();
            p3_keccak_air::generate_trace_rows(inputs)
        }
    };

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_config);

    let config = StarkConfig::new(pcs);

    let mut proof_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
    let mut verif_challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);

    match proof_goal {
        ProofGoal::Blake3(b3_air) => {
            let proof = prove(&config, &b3_air, &mut proof_challenger, trace, &vec![]);

            verify(&config, &b3_air, &mut verif_challenger, &proof, &vec![])
        }
        ProofGoal::Poseidon2(p2_air) => {
            let proof = prove(&config, &p2_air, &mut proof_challenger, trace, &vec![]);

            verify(&config, &p2_air, &mut verif_challenger, &proof, &vec![])
        }
        ProofGoal::Keccak(k_air) => {
            let proof = prove(&config, &k_air, &mut proof_challenger, trace, &vec![]);

            verify(&config, &k_air, &mut verif_challenger, &proof, &vec![])
        }
    }
}

fn report_result(result: Result<(), impl Debug>) {
    if result.is_ok() {
        println!("Proof Verified Successfully")
    } else {
        panic!("{:?}", result.unwrap_err())
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len() - 1, 4, "There should be four arguments passed to this function.
    \n The first chooses the field we are using for the proof. Use \"B\" to use the BabyBear field and \"K\" to use the KoalaBear field.
    \n The second chooses which statement we are proving. Use \"B\", \"P\" or \"K\" to prove Blake3, Poseidon2 or Keccak hashes.
    \n The third gives the desired trace height. It should fit in a u8 and will be interpreted as the log base 2 of the trace height. For Blake3 this
    is the number of hashes, for Poseidon2 it's the number of hashes divided by 8 and for Keccak this is roughly the number of hashes times 24.
    \n The fourth gives the choice of fourier transform. Use \"R\" for RecursiveDft and \"P\" for Radix2DitParallel.");

    let log_2_trace_height = args[3]
        .parse::<u8>()
        .expect("The third command line input should be a u8");
    let trace_height = 1 << log_2_trace_height;

    match args[1].as_ref() {
        "K" | "KoalaBear" | "KOALABEAR" => {
            println!("Choice of Field: KoalaBear");

            type EF = BinomialExtensionField<KoalaBear, 4>;

            let (num_hashes, proof_goal) = match args[2].as_ref() {
                "B" | "Blake3" | "BLAKE3" => {
                    println!("Proving 2^{log_2_trace_height} Blake3 Hashes");
                    (trace_height, ProofGoal::Blake3(Blake3Air {}))
                }
                "P" | "Poseidon2" | "POSEIDON2" => {
                    println!(
                        "Proving 2^{} native Poseidon2 Hashes",
                        log_2_trace_height + 3
                    );

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
                    (trace_height << 3, ProofGoal::Poseidon2(p2_air))
                }
                "K" | "Keccak" | "KECCAK" => {
                    let num_hashes = trace_height / 24;
                    println!("Proving {num_hashes} KECCAK Hashes");
                    (num_hashes, ProofGoal::Keccak(KeccakAir {}))
                }
                _ => {
                    panic!("Could not understand the proof goal. Use \"B\", \"P\" or \"K\" to prove Blake3, Poseidon2 or Keccak hashes.")
                }
            };

            match args[4].as_ref() {
                "R" | "Recursive" => {
                    println!("Choice of DFT: RecursiveDft");
                    let dft = RecursiveDft::new(trace_height << 2);
                    let result = prove_hashes(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                "P" | "Parallel" => {
                    println!("Choice of DFT: Radix2DitParallel");
                    let dft = Radix2DitParallel::default();
                    let result = prove_hashes(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                _ => {
                    panic!("Could not understand the Fourier transform method. Use \"R\" for the Recursive DFT or \"P\" for the Parallel DFT.")
                }
            };
        }
        "B" | "BabyBear" | "BABYBEAR" => {
            println!("Choice of Field: BabyBear");

            type EF = BinomialExtensionField<BabyBear, 4>;

            let (num_hashes, proof_goal) = match args[2].as_ref() {
                "B" | "Blake3" | "BLAKE3" => {
                    println!("Proving 2^{log_2_trace_height} Blake3 Hashes");
                    (trace_height, ProofGoal::Blake3(Blake3Air {}))
                }
                "P" | "Poseidon2" | "POSEIDON2" => {
                    println!(
                        "Proving 2^{} native Poseidon2 Hashes",
                        log_2_trace_height + 3
                    );

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
                    (trace_height << 3, ProofGoal::Poseidon2(p2_air))
                }
                "K" | "Keccak" | "KECCAK" => {
                    let num_hashes = trace_height / 24;
                    println!("Proving {num_hashes} KECCAK Hashes");
                    (num_hashes, ProofGoal::Keccak(KeccakAir {}))
                }
                _ => {
                    panic!("Could not understand the proof goal. Use \"B\", \"P\" or \"K\" to prove Blake3, Poseidon2 or Keccak hashes.")
                }
            };

            match args[4].as_ref() {
                "R" | "Recursive" => {
                    println!("Choice of DFT: RecursiveDft");
                    let dft = RecursiveDft::new(trace_height << 2);
                    let result = prove_hashes(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
                }
                "P" | "Parallel" => {
                    println!("Choice of DFT: Radix2DitParallel");
                    let dft = Radix2DitParallel::default();
                    let result = prove_hashes(proof_goal, dft, num_hashes, PhantomData::<EF>);
                    report_result(result);
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
