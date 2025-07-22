use core::marker::PhantomData;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, Pcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{
    CommitmentWithOpeningPoints, FriParameters, ProverDataWithOpeningPoints, TwoAdicFriFolding,
    TwoAdicFriPcs, prover, verifier,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyFriParams = FriParameters<ChallengeMmcs>;

fn get_ldt_for_testing<R: Rng>(
    rng: &mut R,
    log_final_poly_len: usize,
) -> (Perm, ValMmcs, MyFriParams) {
    let perm = Perm::new_from_rng_128(rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let input_mmcs = ValMmcs::new(hash.clone(), compress.clone());
    let fri_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs: fri_mmcs,
    };
    (perm, input_mmcs, fri_params)
}

fn do_test_fri_ldt<R: Rng>(rng: &mut R, log_final_poly_len: usize) {
    let (perm, input_mmcs, fri_params) = get_ldt_for_testing(rng, log_final_poly_len);
    let dft = Radix2Dit::default();

    let pcs = TwoAdicFriPcs::<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs>::new(
        dft, input_mmcs, fri_params,
    );

    let sizes = [9, 8, 7, 6, 5];
    let val_sizes = sizes.map(Val::from_u8);

    // Prover World:
    let (commitment, mut p_challenger, opened_values, opening_proof) = {
        let mut challenger = Challenger::new(perm.clone());
        challenger.observe(val_sizes);

        let evaluations: Vec<(TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)> = sizes
            .iter()
            .map(|deg_bits| {
                let deg = 1 << deg_bits;
                (
                    <TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs> as Pcs<
                        Challenge,
                        Challenger,
                    >>::natural_domain_for_degree(&pcs, deg),
                    RowMajorMatrix::<Val>::rand_nonzero(rng, deg, 16),
                )
            })
            .collect();

        let num_evaluations = evaluations.len();

        let (commitment, prover_data) =
            <TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, evaluations);

        challenger.observe(commitment);

        let zeta = challenger.sample_algebra_element();

        let open_data = vec![(&prover_data, vec![vec![zeta]; num_evaluations])]; // open every chunk at zeta

        let (opened_values, opening_proof) = pcs.open(open_data, &mut challenger);
        (commitment, challenger, opened_values, opening_proof)
    };

    // Verifier World
    let mut v_challenger = {
        let mut challenger = Challenger::new(perm.clone());
        challenger.observe(val_sizes);
        challenger.observe(commitment);

        let zeta = challenger.sample_algebra_element();

        let domains = sizes.map(|size| {
            <TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::natural_domain_for_degree(&pcs, 1 << size)
        });

        let commitments_with_opening_points = vec![(
            commitment,
            domains
                .into_iter()
                .zip(opened_values.into_iter().flatten().flatten())
                .map(|(domain, value)| (domain, vec![(zeta, value)]))
                .collect(),
        )];

        let verification = pcs.verify(
            commitments_with_opening_points,
            &opening_proof,
            &mut challenger,
        );
        assert!(verification.is_ok());
        challenger
    };

    assert_eq!(
        p_challenger.sample_bits(8),
        v_challenger.sample_bits(8),
        "prover and verifier transcript have same state after FRI"
    );
}

#[test]
fn test_fri_ldt() {
    // FRI is kind of flaky depending on indexing luck
    for i in 0..4 {
        let mut rng = SmallRng::seed_from_u64(i as u64);
        do_test_fri_ldt(&mut rng, i + 1);
    }
}

// This test is expected to panic because the polynomial degree is less than the final_poly_degree in the parameters.
#[test]
#[should_panic]
fn test_fri_ldt_should_panic() {
    // FRI is kind of flaky depending on indexing luck
    for i in 0..4 {
        let mut rng = SmallRng::seed_from_u64(i);
        do_test_fri_ldt(&mut rng, 5);
    }
}
