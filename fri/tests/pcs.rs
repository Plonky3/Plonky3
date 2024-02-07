use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs, UnivariatePcs};
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, Field};
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::thread_rng;

fn make_test_fri_pcs(log_degrees: &[usize]) {
    let mut rng = thread_rng();
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type MyMds = CosetMds<Val, 16>;
    let mds = MyMds::default();

    type Perm = Poseidon2<Val, MyMds, DiffusionMatrixBabybear, 16, 7>;
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = FieldMerkleTreeMmcs<<Val as Field>::Packing, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, 16>;

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    type Pcs =
        TwoAdicFriPcs<TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>>;
    let pcs = Pcs::new(fri_config, dft, val_mmcs);

    let mut challenger = Challenger::new(perm.clone());

    let polynomials = log_degrees
        .iter()
        .map(|d| RowMajorMatrix::rand(&mut rng, 1 << *d, 10))
        .collect::<Vec<_>>();

    let (commit, data) = pcs.commit_batches(polynomials.clone());

    challenger.observe(commit.clone());

    let zeta = challenger.sample_ext_element::<Challenge>();

    let points = polynomials.iter().map(|_| vec![zeta]).collect::<Vec<_>>();

    let (opening, proof) = <Pcs as UnivariatePcs<_, _, RowMajorMatrix<Val>, _>>::open_multi_batches(
        &pcs,
        &[(&data, &points)],
        &mut challenger,
    );

    // verify the proof.
    let mut challenger = Challenger::new(perm);
    challenger.observe(commit.clone());
    let _ = challenger.sample_ext_element::<Challenge>();
    let dims = polynomials
        .iter()
        .map(|p| p.dimensions())
        .collect::<Vec<_>>();
    <Pcs as UnivariatePcs<_, _, RowMajorMatrix<Val>, _>>::verify_multi_batches(
        &pcs,
        &[(commit, &points)],
        &[dims],
        opening,
        &proof,
        &mut challenger,
    )
    .unwrap();
}

#[test]
fn test_fri_pcs_single() {
    make_test_fri_pcs(&[3]);
}

#[test]
fn test_fri_pcs_many_equal() {
    for i in 1..4 {
        make_test_fri_pcs(&[i; 5]);
    }
}

#[test]
fn test_fri_pcs_many_different() {
    for i in 1..4 {
        let degrees = (3..3 + i).collect::<Vec<_>>();
        make_test_fri_pcs(&degrees);
    }
}
