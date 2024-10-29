use std::fmt::Debug;

use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_koala_bear::{
    DiffusionMatrixKoalaBear, KoalaBear, KoalaBearDiffusionMatrixParameters, KoalaBearParameters,
};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_monty_31::GenericDiffusionMatrixMontyField31;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_poseidon2_air::{generate_vectorized_trace_rows, RoundConstants, VectorizedPoseidon2Air};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::{random, thread_rng};
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
const SBOX_DEGREE: usize = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

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

    type Val = KoalaBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm16 = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 16, 3>;
    let perm16 = Perm16::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixKoalaBear::default(),
        &mut thread_rng(),
    );

    type Perm24 = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixKoalaBear, 24, 3>;
    let perm24 = Perm24::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixKoalaBear::default(),
        &mut thread_rng(),
    );

    type MyHash = PaddingFreeSponge<Perm24, 24, 16, 8>;
    let hash = MyHash::new(perm24.clone());

    type MyCompress = TruncatedPermutation<Perm16, 2, 8, 16>;
    let compress = MyCompress::new(perm16.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm24, 24, 16>;

    type MdsLight = Poseidon2ExternalMatrixGeneral;
    let external_linear_layer = MdsLight {};

    type Diffusion =
        GenericDiffusionMatrixMontyField31<KoalaBearParameters, KoalaBearDiffusionMatrixParameters>;
    let internal_linear_layer = Diffusion::new();

    let constants = RoundConstants::from_rng(&mut thread_rng());
    let inputs = (0..NUM_PERMUTATIONS).map(|_| random()).collect::<Vec<_>>();
    let trace = generate_vectorized_trace_rows::<
        Val,
        MdsLight,
        Diffusion,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    >(
        inputs,
        &constants,
        &external_linear_layer,
        &internal_linear_layer,
    );

    let air: VectorizedPoseidon2Air<
        Val,
        MdsLight,
        Diffusion,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidon2Air::new(constants, external_linear_layer, internal_linear_layer);

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut challenger = Challenger::new(perm24.clone());
    let proof = prove(&config, &air, &mut challenger, trace, &vec![]);

    let mut challenger = Challenger::new(perm24);
    verify(&config, &air, &mut challenger, &proof, &vec![])
}
