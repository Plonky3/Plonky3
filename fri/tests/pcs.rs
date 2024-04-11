use itertools::{izip, Itertools};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabybear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::thread_rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;

type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

type Dft = Radix2DitParallel;

type Challenger = DuplexChallenger<Val, Perm, 16>;

type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn make_test_fri_pcs(log_degrees_by_round: &[&[usize]]) {
    let num_rounds = log_degrees_by_round.len();
    let mut rng = thread_rng();

    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabybear,
        &mut rng,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());

    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_config = FriConfig {
        log_blowup: 1,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let max_log_n = log_degrees_by_round
        .iter()
        .copied()
        .flatten()
        .copied()
        .max()
        .unwrap();
    let pcs = MyPcs::new(max_log_n, Dft {}, val_mmcs, fri_config);

    let mut challenger = Challenger::new(perm.clone());

    let domains_and_polys_by_round = log_degrees_by_round
        .iter()
        .map(|log_degrees| {
            log_degrees
                .iter()
                .map(|&log_degree| {
                    let d = 1 << log_degree;
                    (
                        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d),
                        RowMajorMatrix::<Val>::rand(&mut rng, d, 10),
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let (commits_by_round, data_by_round): (Vec<_>, Vec<_>) = domains_and_polys_by_round
        .iter()
        .map(|domains_and_polys| {
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domains_and_polys.clone())
        })
        .unzip();
    assert_eq!(commits_by_round.len(), num_rounds);
    assert_eq!(data_by_round.len(), num_rounds);
    challenger.observe_slice(&commits_by_round);

    let zeta: Challenge = challenger.sample_ext_element();

    let points_by_round = log_degrees_by_round
        .iter()
        .map(|log_degrees| vec![vec![zeta]; log_degrees.len()])
        .collect_vec();
    let data_and_points = data_by_round.iter().zip(points_by_round).collect();
    let (opening_by_round, proof) = pcs.open(data_and_points, &mut challenger);
    assert_eq!(opening_by_round.len(), num_rounds);

    // Verify the proof.
    let mut challenger = Challenger::new(perm);
    challenger.observe_slice(&commits_by_round);
    let verifier_zeta: Challenge = challenger.sample_ext_element();
    assert_eq!(verifier_zeta, zeta);

    let commits_and_claims_by_round = izip!(
        commits_by_round,
        domains_and_polys_by_round,
        opening_by_round
    )
    .map(|(commit, domains_and_polys, openings)| {
        let claims = domains_and_polys
            .iter()
            .zip(openings)
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect_vec();
        (commit, claims)
    })
    .collect_vec();
    assert_eq!(commits_and_claims_by_round.len(), num_rounds);

    pcs.verify(commits_and_claims_by_round, &proof, &mut challenger)
        .unwrap()
}

#[test]
fn test_fri_pcs_single() {
    make_test_fri_pcs(&[&[3]]);
}

#[test]
fn test_fri_pcs_many_equal() {
    for i in 1..4 {
        make_test_fri_pcs(&[&[i; 5]]);
    }
}

#[test]
fn test_fri_pcs_many_different() {
    for i in 2..4 {
        let degrees = (3..3 + i).collect::<Vec<_>>();
        make_test_fri_pcs(&[&degrees]);
    }
}

#[test]
fn test_fri_pcs_many_different_rev() {
    for i in 2..4 {
        let degrees = (3..3 + i).rev().collect::<Vec<_>>();
        make_test_fri_pcs(&[&degrees]);
    }
}

#[test]
fn test_fri_pcs_multiple_rounds() {
    make_test_fri_pcs(&[&[3]]);
    make_test_fri_pcs(&[&[3], &[3]]);
    make_test_fri_pcs(&[&[3], &[2]]);
    make_test_fri_pcs(&[&[2], &[3]]);
    make_test_fri_pcs(&[&[3, 4], &[3, 4]]);
    make_test_fri_pcs(&[&[4, 2], &[4, 2]]);
    make_test_fri_pcs(&[&[2, 2], &[3, 3]]);
    make_test_fri_pcs(&[&[3, 3], &[2, 2]]);
    make_test_fri_pcs(&[&[2], &[3, 3]]);
}
