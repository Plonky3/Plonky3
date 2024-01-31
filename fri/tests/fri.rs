use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanSample, CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::{DirectMmcs, ExtensionMmcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_fri::two_adic_pcs::PowersReducer;
use p3_fri::{prover, verifier, FriConfigImpl};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::{Matrix, MatrixRows};
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type MyMds = CosetMds<Val, 16>;
type Perm = Poseidon2<Val, MyMds, DiffusionMatrixBabybear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = FieldMerkleTreeMmcs<<Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16>;
type MyFriConfig = FriConfigImpl<Challenge, ChallengeMmcs, Challenger>;

fn get_ldt_for_testing<R: Rng>(rng: &mut R) -> (Perm, ValMmcs, MyFriConfig) {
    let mds = MyMds::default();
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_config = MyFriConfig::new(1, 10, 8, challenge_mmcs);
    (perm, val_mmcs, fri_config)
}

fn do_test_fri_ldt<R: Rng>(rng: &mut R) {
    let (perm, val_mmcs, fc) = get_ldt_for_testing(rng);
    let dft = Radix2Dit::default();

    let shift = Val::generator();

    let ldes: Vec<RowMajorMatrix<Val>> = (3..10)
        .map(|deg_bits| {
            let evals = RowMajorMatrix::<Val>::rand_nonzero(rng, 1 << deg_bits, 16);
            let mut lde = dft.coset_lde_batch(evals, 1, shift);
            reverse_matrix_index_bits(&mut lde);
            lde
        })
        .collect();
    let dims = ldes.iter().map(|m| m.dimensions()).collect_vec();
    let (comm, data) = val_mmcs.commit(ldes.clone());

    let (proof, reduced_openings, p_sample) = {
        // Prover world
        let mut chal = Challenger::new(perm.clone());
        let alpha: Challenge = chal.sample_ext_element();

        let input: [_; 32] = core::array::from_fn(|log_height| {
            let matrices_with_log_height: Vec<&RowMajorMatrix<Val>> = ldes
                .iter()
                .filter(|m| log2_strict_usize(m.height()) == log_height)
                .collect();
            if matrices_with_log_height.is_empty() {
                None
            } else {
                let reduced: Vec<Challenge> = (0..(1 << log_height))
                    .map(|r| {
                        alpha
                            .powers()
                            .zip(matrices_with_log_height.iter().flat_map(|m| m.row(r)))
                            .map(|(alpha_pow, v)| alpha_pow * v)
                            .sum()
                    })
                    .collect();
                Some(reduced)
            }
        });

        let (proof, idxs) = prover::prove(&fc, &input, &mut chal);

        let log_max_height = input.iter().rposition(Option::is_some).unwrap();
        let reduced_openings: Vec<[Challenge; 32]> = idxs
            .into_iter()
            .map(|idx| {
                input
                    .iter()
                    .enumerate()
                    .map(|(log_height, v)| {
                        if let Some(v) = v {
                            v[idx >> (log_max_height - log_height)]
                        } else {
                            Challenge::zero()
                        }
                    })
                    .collect_vec()
                    .try_into()
                    .unwrap()
            })
            .collect();

        (proof, reduced_openings, chal.sample_bits(8))
    };

    /*
    let proof = ldt.prove(&[val_mmcs.clone()], &[&data], &mut p_challenger);
    */

    let mut v_challenger = Challenger::new(perm);
    let alpha: Challenge = v_challenger.sample_ext_element();
    /*
    let v_idxs = verifier::verify(&fc, &proof, &reduced_openings, &mut v_challenger)
        .expect("verification failed");
        */
    let fri_challenges =
        verifier::verify_shape_and_sample_challenges(&fc, &proof, &mut v_challenger)
            .expect("failed verify shape and sample");
    verifier::verify_challenges(&fc, &proof, &fri_challenges, &reduced_openings)
        .expect("failed verify challenges");

    /*
    ldt.verify(&[val_mmcs], &[dims], &[comm], &proof, &mut v_challenger)
        .expect("verification failed");
        */

    assert_eq!(
        p_sample,
        v_challenger.sample_bits(8),
        "prover and verifier transcript have same state after FRI"
    );
}

#[test]
fn test_fri_ldt() {
    // FRI is kind of flaky depending on indexing luck
    for i in 0..20 {
        let mut rng = ChaCha20Rng::seed_from_u64(i);
        do_test_fri_ldt(&mut rng);
    }
}

// You can uncomment this for use with `cargo-show-asm` to investigate `reduce_base`.
/*
#[test]
fn force_monomorphize() {
    let mut rng = thread_rng();
    let alpha: Challenge = rng.gen();
    let r = PowersReducer::new(alpha, 1024);
    let xs: Vec<Val> = (0..1024).map(|_| rng.gen()).collect();
    core::hint::black_box(r.reduce_base(&xs));
}
*/
