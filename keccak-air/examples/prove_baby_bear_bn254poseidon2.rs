use p3_baby_bear::{BabyBear, PackedBabyBearNeon};
use p3_bn254::{convert_babybear_elements_to_bn254_element, DiffusionMatrixBN254, BN254};
use p3_challenger::BabyBearBN254Challenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_keccak_air::{generate_trace_rows, KeccakAir};
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::Poseidon2;
use p3_symmetric::{CryptographicHasher, CryptographicPermutation, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig, VerificationError};
use p3_util::log2_ceil_usize;
use rand::{random, thread_rng};
use tracing_forest::util::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};
use itertools::Itertools;

const NUM_HASHES: usize = 100;

#[derive(Clone)]
struct PaddingFreeSpongeBabyBearBN254<P> {
    permutation: P,
}

impl<P> PaddingFreeSpongeBabyBearBN254<P>
{
    pub fn new(permutation: P) -> Self {
        Self { permutation }
    }
}

impl<P> CryptographicHasher<BabyBear, [BN254; 1]>
    for PaddingFreeSpongeBabyBearBN254<P>
where
    P: CryptographicPermutation<[BN254; 3]>,
{
    fn hash_iter<I>(&self, input: I) -> [BN254; 1]
    where
        I: IntoIterator<Item = BabyBear>,
    {
        // static_assert(RATE < WIDTH)
        let mut state = [BN254::default(); 3];
        for block_chunk in &input.into_iter().chunks(16) {
            let mut chunk_id = 1;
            for chunk in &block_chunk.into_iter().chunks(8) {
                state[chunk_id] = convert_babybear_elements_to_bn254_element(&chunk.collect::<Vec<_>>());
                chunk_id += 1;
                state = self.permutation.permute(state);
            }
    }
        state[..1].try_into().unwrap()
    }
}

impl<P> CryptographicHasher<PackedBabyBearNeon, [BN254; 1]>
    for PaddingFreeSpongeBabyBearBN254<P>
where
    P: CryptographicPermutation<[BN254; 3]>,
{
    fn hash_iter<I>(&self, input: I) -> [BN254; 1]
    where
        I: IntoIterator<Item = PackedBabyBearNeon>,
    {
        // static_assert(RATE < WIDTH)
        let mut state = [BN254::default(); 3];
        for block_chunk in &input.into_iter().chunks(4) {
            let mut chunk_id = 1;
            for chunk in &block_chunk.into_iter().chunks(2) {
                let mut bb_elements = Vec::new();

                for packed_element in chunk {
                    bb_elements.extend(packed_element.0);
                }

                state[chunk_id] = convert_babybear_elements_to_bn254_element(&bb_elements);
                chunk_id += 1;
                state = self.permutation.permute(state);
            }
    }
        state[..1].try_into().unwrap()
    }
}


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

    type Perm = Poseidon2<BN254, DiffusionMatrixBN254, 3, 5>;

    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBN254, &mut thread_rng());

    type MyHash = PaddingFreeSpongeBabyBearBN254<Perm>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 1, 3>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
        <BabyBear as Field>::Packing,
        <BN254 as Field>::Packing,
        MyHash,
        MyCompress,
        1,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = BabyBearBN254Challenger<Perm>;

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

    let mut challenger = Challenger::new(perm.clone());

    let proof = prove::<MyConfig, _>(&config, &KeccakAir {}, &mut challenger, trace);

    let mut challenger = Challenger::new(perm);
    verify(&config, &KeccakAir {}, &mut challenger, &proof)
}
