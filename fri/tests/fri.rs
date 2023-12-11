use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanSample, DuplexChallenger};
use p3_commit::{DirectMmcs, ExtensionMmcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_fri::{FriConfigImpl, FriLdt};
use p3_ldt::Ldt;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::thread_rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type MyMds = CosetMds<Val, 16>;
type Perm = Poseidon2<Val, MyMds, DiffusionMatrixBabybear, 16, 5>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = FieldMerkleTreeMmcs<<Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16>;
type MyFriConfig = FriConfigImpl<Val, Challenge, ValMmcs, ChallengeMmcs, Challenger>;

fn get_ldt_for_testing() -> (Perm, ValMmcs, FriLdt<MyFriConfig>) {
    let mds = MyMds::default();
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = MyFriConfig::new(1, challenge_mmcs);
    (perm, val_mmcs, FriLdt { config: fri_config })
}

#[test]
fn test_fri_ldt() {
    let (perm, val_mmcs, ldt) = get_ldt_for_testing();
    let dft = Radix2Dit;

    let ldes: Vec<RowMajorMatrix<Val>> = (3..4)
        .map(|deg_bits| {
            let evals = RowMajorMatrix::<Val>::rand_nonzero(&mut thread_rng(), 1 << deg_bits, 4);
            dft.coset_lde_batch(evals, 1, Val::one())
        })
        .collect();
    let dims = ldes.iter().map(|m| m.dimensions()).collect_vec();
    let (comm, data) = val_mmcs.commit(ldes);

    let mut p_challenger = Challenger::new(perm.clone());
    let proof = ldt.prove(&[val_mmcs.clone()], &[&data], &mut p_challenger);

    let mut v_challenger = Challenger::new(perm);
    ldt.verify(&[val_mmcs], &[dims], &[comm], &proof, &mut v_challenger)
        .unwrap();

    assert_eq!(
        p_challenger.sample(),
        v_challenger.sample(),
        "prover and verifier transcript have same state after FRI"
    );
}

/*
#[test]
#[ignore]
fn test_fri_ldt_multiple_degrees() {
    let (perm, challenge_mmcs, ldt) = get_ldt_for_testing();
    let dft = Radix2Dit;

    let ldes: Vec<RowMajorMatrix<Challenge>> = (3..5)
        .map(|deg_bits| {
            let evals = RowMajorMatrix::<Val>::rand_nonzero(&mut thread_rng(), 1 << deg_bits, 4);
            dft.coset_lde_batch(evals, 1, Val::one()).to_ext()
        })
        .collect();
    let dims = ldes.iter().map(|m| m.dimensions()).collect_vec();
    let (comm, data) = challenge_mmcs.commit(ldes);

    let mut challenger = Challenger::new(perm.clone());
    let proof = ldt.prove(&[challenge_mmcs.clone()], &[&data], &mut challenger);

    let mut challenger = Challenger::new(perm);
    ldt.verify(&[challenge_mmcs], &[dims], &[comm], &proof, &mut challenger)
        .unwrap();
}
*/
