//! Deterministic backend configurations shared by the batch contract tests.

use core::marker::PhantomData;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::StarkConfig;
use p3_util::{assert_clone, assert_send, assert_sync};
use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};

pub(super) type Val = BabyBear;
pub(super) type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type PermWide = Poseidon2BabyBear<32>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyCompressWide = TruncatedPermutation<PermWide, 4, 8, 32>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ValMmcsWide =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompressWide, 4, 8>;
type HidingValMmcs = MerkleTreeHidingMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    StdRng,
    2,
    8,
    4,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type ChallengeMmcsWide = ExtensionMmcs<Val, Challenge, ValMmcsWide>;
type HidingChallengeMmcs = ExtensionMmcs<Val, Challenge, HidingValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyPcsWide = TwoAdicFriPcs<Val, Dft, ValMmcsWide, ChallengeMmcsWide>;
type HidingPcs = HidingFriPcs<Val, Dft, HidingValMmcs, HidingChallengeMmcs, StdRng>;
pub(super) type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;
pub(super) type MyConfigWide = StarkConfig<MyPcsWide, Challenge, Challenger>;
pub(super) type MyHidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;

pub(super) fn make_config(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Minimal FRI shape so a tiny trace still completes `prove_batch` / `verify_batch`.
pub(super) fn make_config_allow_tiny_trace(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Same as make_config, but with a different arity.
pub(super) fn make_config_wide(seed: u64) -> MyConfigWide {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let perm_wide = PermWide::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompressWide::new(perm_wide);
    let val_mmcs = ValMmcsWide::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcsWide::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = MyPcsWide::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

pub(super) fn make_two_adic_compat_config(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 1);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 2,
        max_log_arity: 1,
        num_queries: 2,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

pub(super) fn make_config_zk(seed: u64) -> MyHidingConfig {
    assert_clone::<HidingValMmcs>();
    assert_sync::<HidingValMmcs>();
    assert_send::<HidingPcs>();
    assert_sync::<HidingPcs>();
    assert_sync::<MyHidingConfig>();

    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = HidingValMmcs::new(hash, compress, 2, StdRng::seed_from_u64(1));
    let challenge_mmcs = HidingChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = HidingPcs::new(dft, val_mmcs, fri_params, 4, StdRng::seed_from_u64(2));
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

pub(super) type CircleVal = Mersenne31;
type CircleChallenge = BinomialExtensionField<CircleVal, 3>;
type CircleByteHash = Keccak256Hash;
type CircleFieldHash = SerializingHasher<CircleByteHash>;
type CircleCompress = CompressionFunctionFromHasher<CircleByteHash, 2, 32>;
type CircleValMmcs = MerkleTreeMmcs<CircleVal, u8, CircleFieldHash, CircleCompress, 2, 32>;
type CircleChallengeMmcs = ExtensionMmcs<CircleVal, CircleChallenge, CircleValMmcs>;
type CircleChallenger = SerializingChallenger32<CircleVal, HashChallenger<u8, CircleByteHash, 32>>;
type CirclePcsType = CirclePcs<CircleVal, CircleValMmcs, CircleChallengeMmcs>;
pub(super) type CircleConfig = StarkConfig<CirclePcsType, CircleChallenge, CircleChallenger>;

pub(super) fn make_circle_config() -> CircleConfig {
    let byte_hash = CircleByteHash {};
    let field_hash = CircleFieldHash::new(byte_hash);
    let compress = CircleCompress::new(byte_hash);
    let val_mmcs = CircleValMmcs::new(field_hash, compress, 3);
    let challenge_mmcs = CircleChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        batch_proof_of_work_bits: 0,
        commit_proof_of_work_bits: 8,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = CirclePcsType {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = CircleChallenger::from_hasher(vec![], byte_hash);
    CircleConfig::new(pcs, challenger)
}
