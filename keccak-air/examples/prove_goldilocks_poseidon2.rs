use std::fmt::Debug;

use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{create_benchmark_fri_config, TwoAdicFriPcs};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_keccak_air::{generate_trace_rows, KeccakAir};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

const NUM_HASHES: usize = 1365;

fn main() -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = Goldilocks;
    type Challenge = BinomialExtensionField<Val, 2>;

    type Perm = Poseidon2Goldilocks<8>;
    let perm = Perm::new_from_rng_128(&mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, 8, 4, 4>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 4, 8>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 4>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 8, 4>;

    let fri_config = create_benchmark_fri_config(challenge_mmcs);

    let inputs = (0..NUM_HASHES).map(|_| random()).collect::<Vec<_>>();
    let trace = generate_trace_rows::<Val>(inputs, fri_config.log_blowup);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut challenger = Challenger::new(perm.clone());
    let proof = prove(&config, &KeccakAir {}, &mut challenger, trace, &vec![]);

    let mut challenger = Challenger::new(perm);
    verify(&config, &KeccakAir {}, &mut challenger, &proof, &vec![])
}
