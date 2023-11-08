use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::{DirectMmcs, ExtensionMmcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
use p3_fri::{FriConfigImpl, FriLdt};
use p3_ldt::Ldt;
use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::thread_rng;

#[test]
fn test_fri() {
    type Val = BabyBear;
    type Domain = Val;
    // type Challenge = BinomialExtensionField<Val, 4>;
    type Challenge = Val;
    // type PackedChallenge = BinomialExtensionField<<Domain as Field>::Packing, 4>;

    type MyMds = CosetMds<Val, 16>;
    let mds = MyMds::default();

    type Perm = Poseidon2<Val, MyMds, DiffusionMatrixBabybear, 16, 5>;
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<<Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    // type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    // let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    type ChallengeMmcs = ValMmcs;
    let challenge_mmcs = val_mmcs.clone();

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    type MyFriConfig =
        FriConfigImpl<Val, Domain, Challenge, ChallengeMmcs, ChallengeMmcs, Challenger>;
    let fri_config = MyFriConfig::new(1, challenge_mmcs.clone());
    let ldt = FriLdt { config: fri_config };

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

    let mut challenger = Challenger::new(perm.clone());
    ldt.verify(
        &[challenge_mmcs.clone()],
        &[dims],
        &[comm],
        &proof,
        &mut challenger,
    )
    .unwrap();
}
