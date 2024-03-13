use p3_baby_bear::BabyBear;
use p3_bn254_fr::{DiffusionMatrixBN254, Bn254Fr};
use p3_challenger::MultiFieldChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak_air::{generate_trace_rows, KeccakAir};
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::Poseidon2;
use p3_symmetric::{PaddingFreeSpongeMultiField, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig, VerificationError};
use p3_util::log2_ceil_usize;
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

const NUM_HASHES: usize = 680;

fn main() -> Result<(), VerificationError> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Bn254Fr, DiffusionMatrixBN254, 3, 5>;

    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBN254, &mut thread_rng());

    type MyHash = PaddingFreeSpongeMultiField<Val, Bn254Fr, Perm, 3, 8, 1>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 1, 3>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
        BabyBear,
        Bn254Fr,
        MyHash,
        MyCompress,
        1,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = MultiFieldChallenger<Val, Bn254Fr, Perm, 3>;

    let inputs = (0..NUM_HASHES).map(|_| random()).collect::<Vec<_>>();
    let trace = generate_trace_rows::<Val>(inputs);

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(log2_ceil_usize(trace.height()), dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut challenger = Challenger::new(perm.clone()).unwrap();

    let proof  = prove::<MyConfig, _>(&config, &KeccakAir {}, &mut challenger, trace);
    
    let mut challenger = Challenger::new(perm).unwrap();
    verify(&config, &KeccakAir {}, &mut challenger, &proof)
}
