use std::fmt::Debug;
use std::marker::PhantomData;

use p3_challenger::DuplexChallenger;
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::create_benchmark_fri_config;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::{GenericPoseidon2LinearLayersMersenne31, Mersenne31, Poseidon2Mersenne31};
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::thread_rng;
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 5;
const SBOX_REGISTERS: usize = 1;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 14;

const NUM_ROWS: usize = 1 << 16;
const VECTOR_LEN: usize = 1 << 3;
const NUM_PERMUTATIONS: usize = NUM_ROWS * VECTOR_LEN;

fn main() -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;

    type Perm16 = Poseidon2Mersenne31<16>;
    let perm16 = Perm16::new_from_rng_128(&mut thread_rng());

    type Perm24 = Poseidon2Mersenne31<24>;
    let perm24 = Perm24::new_from_rng_128(&mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm24, 24, 16, 8>;
    let hash = MyHash::new(perm24.clone());

    type MyCompress = TruncatedPermutation<Perm16, 2, 8, 16>;
    let compress = MyCompress::new(perm16.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = DuplexChallenger<Val, Perm24, 24, 16>;

    let constants = RoundConstants::from_rng(&mut thread_rng());
    let air: VectorizedPoseidon2Air<
        Val,
        GenericPoseidon2LinearLayersMersenne31,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidon2Air::new(constants);

    let trace = air.generate_vectorized_trace_rows(NUM_PERMUTATIONS);

    let fri_config = create_benchmark_fri_config(challenge_mmcs);
    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_config,
        _phantom: PhantomData,
    };

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut challenger = Challenger::new(perm24.clone());
    let proof = prove(&config, &air, &mut challenger, trace, &vec![]);

    let mut challenger = Challenger::new(perm24.clone());
    verify(&config, &air, &mut challenger, &proof, &vec![])
}
