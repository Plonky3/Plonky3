use std::fmt::Debug;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_blake3_air::Blake3Air;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::thread_rng;
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

const NUM_HASHES: usize = 1 << 13;

fn main() -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm16 = Poseidon2BabyBear<16>;
    let perm16 = Perm16::new_from_rng_128(&mut thread_rng());

    type Perm24 = Poseidon2BabyBear<24>;
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

    let blake3_air = Blake3Air {};
    let trace = blake3_air.generate_trace_rows::<Val>(NUM_HASHES);

    let fri_config = FriConfig {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut challenger = Challenger::new(perm24.clone());
    let proof = prove(&config, &Blake3Air {}, &mut challenger, trace, &vec![]);

    let mut challenger = Challenger::new(perm24.clone());
    verify(&config, &Blake3Air {}, &mut challenger, &proof, &vec![])
}
