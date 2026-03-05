//! End-to-end Poseidon1 STARK proof using BabyBear and Keccak.

use core::fmt::Debug;

use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon_air::{RoundConstants, VectorizedPoseidonAir};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;
#[cfg(target_family = "unix")]
use tikv_jemallocator::Jemalloc;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg(target_family = "unix")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// Poseidon1 parameters for BabyBear width 16 (same as in tests).
const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 7;
const SBOX_REGISTERS: usize = 1;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 13;

/// Number of trace rows (must be a power of two).
const NUM_ROWS: usize = 1 << 16;

/// Number of permutations packed per row.
const VECTOR_LEN: usize = 1 << 3;

/// Total permutations computed = NUM_ROWS × VECTOR_LEN.
const NUM_PERMUTATIONS: usize = NUM_ROWS * VECTOR_LEN;

/// Select DFT backend: parallel Radix2 when `parallel` feature is enabled.
#[cfg(feature = "parallel")]
type Dft = p3_dft::Radix2DitParallel<BabyBear>;
#[cfg(not(feature = "parallel"))]
type Dft = p3_dft::Radix2Bowers;

fn main() -> Result<(), impl Debug> {
    // Set up tracing (structured logging) with INFO level by default.
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    // Run multiple prove-verify cycles to warm up and benchmark.
    const PROOFS: usize = 2;
    for _ in 1..PROOFS {
        prove_and_verify()?;
    }
    prove_and_verify()
}

/// Run one complete prove-verify cycle.
fn prove_and_verify() -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;


    type ByteHash = Keccak256Hash;
    let byte_hash = ByteHash {};

    // Sponge-based hash: absorb 17 lanes from KeccakF (rate), squeeze 4.
    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    let u64_hash = U64Hash::new(KeccakF {});

    // Serialize field elements to bytes, then hash.
    type FieldHash = SerializingHasher<U64Hash>;
    let field_hash = FieldHash::new(u64_hash);

    // Merkle tree compression: hash two 4-u64 digests into one.
    type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
    let compress = MyCompress::new(u64_hash);


    // Merkle tree MMCS for committing to the trace polynomial evaluations.
    type ValMmcs = MerkleTreeMmcs<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        FieldHash,
        MyCompress,
        4,
    >;
    let val_mmcs = ValMmcs::new(field_hash, compress, 3);

    // Extension field MMCS for FRI commitments.
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // Fiat-Shamir challenger (serializes field elements to bytes for hashing).
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    let challenger = Challenger::from_hasher(vec![], byte_hash);


    // WARNING: SmallRng is NOT cryptographically secure. Use a real PRNG in production.
    let mut rng = SmallRng::seed_from_u64(1);
    let constants = RoundConstants::from_rng(&mut rng);
    let air: VectorizedPoseidonAir<
        Val,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidonAir::new(constants);


    let fri_params = create_benchmark_fri_params(challenge_mmcs);

    // Generate the execution trace (with extra capacity for LDE blowup).
    let trace = air.generate_vectorized_trace_rows(NUM_PERMUTATIONS, fri_params.log_blowup);

    let dft = Dft::default();

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    // Generate the STARK proof.
    let proof = prove(&config, &air, trace, &[]);

    // Verify the proof.
    verify(&config, &air, &proof, &[])
}
