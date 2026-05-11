use p3_fri::FriParameters;
use p3_test_utils::baby_bear_params::{
    ChallengeMmcs, Challenger, Dft, MyCompress, MyConfig, MyHash, MyMmcs, MyPcs,
    default_babybear_poseidon2_16,
};

/// Build a test STARK config using the standard BabyBear Poseidon2 setup.
pub fn build_test_config() -> MyConfig {
    let perm = default_babybear_poseidon2_16();
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = MyMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 0);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}
