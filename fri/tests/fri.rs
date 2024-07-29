use itertools::Itertools;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_fri::{prover, verifier, FriConfig};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyFriConfig = FriConfig<ChallengeMmcs>;

fn get_ldt_for_testing<R: Rng>(rng: &mut R) -> (Perm, MyFriConfig) {
    let perm = Perm::new_from_rng_128(Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let fri_config = FriConfig {
        log_blowup: 1,
        log_arity: 1,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs,
    };
    (perm, fri_config)
}

fn do_test_fri_ldt<R: Rng>(rng: &mut R) {
    let (perm, fc) = get_ldt_for_testing(rng);
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

    let mut v_challenger = Challenger::new(perm);
    let _alpha: Challenge = v_challenger.sample_ext_element();
    let fri_challenges =
        verifier::verify_shape_and_sample_challenges(&fc, &proof, &mut v_challenger)
            .expect("failed verify shape and sample");
    verifier::verify_challenges(&fc, &proof, &fri_challenges, &reduced_openings)
        .expect("failed verify challenges");

    assert_eq!(
        p_sample,
        v_challenger.sample_bits(8),
        "prover and verifier transcript have same state after FRI"
    );
}

#[test]
fn test_fri_ldt() {
    // FRI is kind of flaky depending on indexing luck
    for i in 0..4 {
        let mut rng = ChaCha20Rng::seed_from_u64(i);
        do_test_fri_ldt(&mut rng);
    }
}
