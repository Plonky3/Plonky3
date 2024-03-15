use rand::thread_rng;
use serde_json;

use fibnonacci_air::FibonacciAir;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabybear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::Poseidon2;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, StarkConfig, VerificationError, verify};
use p3_util::log2_ceil_usize;

fn main() -> Result<(), VerificationError> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, DiffusionMatrixBabybear, 16, 7>;
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    let trace = fibnonacci_air::generate_trace_rows::<Val>(1, 1, (1 << 5) - 1);

    let fri_config = FriConfig {
        log_blowup: 3,
        num_queries: 28,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(log2_ceil_usize(trace.height()), dft, val_mmcs, fri_config);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs);

    let mut challenger = Challenger::new(perm.clone());

    let proof = prove::<MyConfig, _>(&config, &FibonacciAir {}, &mut challenger, trace);
    let proof_json = serde_json::to_string(&proof).expect("failed to serialize");
    println!("proof json {} bytes", proof_json.as_bytes().len());

    let mut challenger = Challenger::new(perm);
    verify(&config, &FibonacciAir {}, &mut challenger, &proof)
}
