use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_keccak_air::{generate_trace_rows, KeccakAir};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixGoldilocks, Poseidon2};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher64};
use p3_uni_stark::{prove, verify, PublicRow, StarkConfig, VerificationError};
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};
use p3_field::AbstractField;

const NUM_HASHES: usize = 680;

fn main() -> Result<(), VerificationError> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = Goldilocks;
    type Challenge = BinomialExtensionField<Val, 2>;

    type Perm = Poseidon2<Val, DiffusionMatrixGoldilocks, 8, 7>;
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixGoldilocks, &mut thread_rng());

    type MyHash = SerializingHasher64<Keccak256Hash>;
    let hash = MyHash::new(Keccak256Hash {});
    type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 4>;
    let compress = MyCompress::new(hash);

    type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 4>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 8>;

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    type Pcs =
        TwoAdicFriPcs<TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>>;
    let pcs = Pcs::new(fri_config, dft, val_mmcs);

    type MyConfig = StarkConfig<Val, Challenge, Pcs, Challenger>;
    let config = StarkConfig::new(pcs);

    let mut challenger = Challenger::new(perm.clone());

    let inputs = (0..NUM_HASHES).map(|_| random()).collect::<Vec<_>>();
    let trace = generate_trace_rows::<Val>(inputs);
    let proof = prove::<MyConfig, _, PublicRow<Val>>(
        &config,
        &KeccakAir {},
        &mut challenger,
        trace,
        &PublicRow::default(),
    );

    let mut challenger = Challenger::new(perm);
    verify(
        &config,
        &KeccakAir {},
        &mut challenger,
        &proof,
        &RowMajorMatrix::new(vec![Val::zero()], 1),
    )
}
