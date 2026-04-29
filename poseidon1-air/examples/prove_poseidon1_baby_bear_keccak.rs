//! End-to-end Poseidon1 STARK proof using BabyBear and Keccak.

use core::fmt::Debug;

use p3_baby_bear::{
    BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
    BABYBEAR_POSEIDON1_RC_16, BABYBEAR_S_BOX_DEGREE, BabyBear, MDSBabyBearData,
};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_monty_31::MDSUtils;
use p3_poseidon1::Poseidon1Constants;
use p3_poseidon1_air::VectorizedPoseidon1Air;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};
#[cfg(all(target_family = "unix", not(feature = "zk-alloc")))]
use tikv_jemallocator::Jemalloc;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg(feature = "zk-alloc")]
#[global_allocator]
static GLOBAL: p3_zk_alloc::ZkAllocator = p3_zk_alloc::ZkAllocator;

#[cfg(all(target_family = "unix", not(feature = "zk-alloc")))]
#[global_allocator]
static GLOBAL_JE: Jemalloc = Jemalloc;

// Poseidon1 parameters for BabyBear width 16.
const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;
const SBOX_REGISTERS: usize = 1;
const HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS;
const PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16;

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

    prove_and_verify()?;

    #[cfg(feature = "zk-alloc")]
    p3_zk_alloc::begin_phase();
    prove_and_verify()?;

    #[cfg(feature = "zk-alloc")]
    p3_zk_alloc::begin_phase();
    let result = prove_and_verify();
    #[cfg(feature = "zk-alloc")]
    p3_zk_alloc::end_phase();
    result
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
        2,
        4,
    >;
    let val_mmcs = ValMmcs::new(field_hash, compress, 3);

    // Extension field MMCS for FRI commitments.
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // Fiat-Shamir challenger (serializes field elements to bytes for hashing).
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let raw = Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: PARTIAL_ROUNDS,
        mds_circ_col: MDSBabyBearData::MATRIX_CIRC_MDS_16_COL,
        round_constants: BABYBEAR_POSEIDON1_RC_16.to_vec(),
    };
    let (full, partial) = raw.to_optimized();
    let air: VectorizedPoseidon1Air<
        Val,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidon1Air::new(full, partial);

    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

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
