//! End-to-end STIR prove/verify tests.
//!
//! Each test constructs a polynomial, runs `prove_stir`, then `verify_stir`, and checks
//! that the proof verifies. Tests cover BabyBear (quartic extension), KoalaBear (quartic
//! extension), and Goldilocks (quadratic extension).

use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, Pcs, SecurityAssumption};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::config::{StirConfig, StirParameters};
use p3_stir::prover::prove_stir;
use p3_stir::verifier::verify_stir;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn seeded_rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

// ---------------------------------------------------------------------------
// Generic prove/verify harness.
// ---------------------------------------------------------------------------

/// Run prove_stir then verify_stir with a random polynomial of degree `2^log_degree`.
fn do_test_stir_prove_verify<F, EF, Dft, M, Challenger>(
    params: &StirParameters<M>,
    dft: &Dft,
    challenger_template: &Challenger,
    log_degree: usize,
) where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    Dft: TwoAdicSubgroupDft<F>,
    M: Mmcs<EF> + Clone,
    Challenger:
        FieldChallenger<F> + CanObserve<M::Commitment> + GrindingChallenger<Witness = F> + Clone,
    StandardUniform: Distribution<EF>,
{
    let mut rng = seeded_rng();
    let degree = 1usize << log_degree;

    // Random polynomial in coefficient form.
    let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

    let config = StirConfig::<F, EF, M, Challenger>::new(log_degree, params.clone());

    // -----------------------------------------------------------------------
    // Prove
    // -----------------------------------------------------------------------
    let mut p_challenger = challenger_template.clone();

    let proof = prove_stir(&config, poly_coeffs, dft, &mut p_challenger);

    // -----------------------------------------------------------------------
    // Verify
    // -----------------------------------------------------------------------
    let mut v_challenger = challenger_template.clone();

    verify_stir::<F, EF, M, Challenger>(&config, &proof, &mut v_challenger)
        .unwrap_or_else(|e| panic!("Verification failed for log_degree={log_degree}: {e}"));
}

// ---------------------------------------------------------------------------
// BabyBear tests
// ---------------------------------------------------------------------------

mod babybear_stir {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type MyMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Dft = Radix2DitParallel<F>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn make_params(
        log_blowup: usize,
        log_folding_factor: usize,
    ) -> (StirParameters<MyMmcs>, Dft, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let mmcs = MyMmcs::new(val_mmcs);

        let params = StirParameters {
            log_blowup,
            log_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            // Low security + no PoW for fast tests. Real deployments use
            // security_level=128 and appropriate max_pow_bits.
            security_level: 16,
            max_pow_bits: 0,
            mmcs,
        };
        (params, Dft::default(), Challenger::new(perm))
    }

    #[test]
    fn test_prove_verify_blowup1_fold1_degree8() {
        let (params, dft, challenger) = make_params(1, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup2_fold1_degree8() {
        let (params, dft, challenger) = make_params(2, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup1_fold2_degree8() {
        let (params, dft, challenger) = make_params(1, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup1_fold1_degree4() {
        let (params, dft, challenger) = make_params(1, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 4);
    }

    #[test]
    fn test_prove_verify_blowup1_fold1_degree12() {
        let (params, dft, challenger) = make_params(1, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 12);
    }

    #[test]
    fn test_tampered_next_query_opening_fails() {
        let (params, dft, challenger) = make_params(1, 1);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let mut proof = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        proof.round_proofs[0].next_query_proofs[0].row_evals[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err()
        );
    }

    #[test]
    fn test_tampered_fold_polynomial_fails() {
        let (params, dft, challenger) = make_params(1, 1);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let mut proof = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        proof.round_proofs[0].fold_polynomial[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err()
        );
    }
}

// ---------------------------------------------------------------------------
// KoalaBear tests (quartic extension)
// ---------------------------------------------------------------------------

mod koalabear_stir {
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2KoalaBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type MyMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Dft = Radix2DitParallel<F>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn make_params(
        log_blowup: usize,
        log_folding_factor: usize,
    ) -> (StirParameters<MyMmcs>, Dft, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let mmcs = MyMmcs::new(val_mmcs);

        let params = StirParameters {
            log_blowup,
            log_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs,
        };
        (params, Dft::default(), Challenger::new(perm))
    }

    #[test]
    fn test_prove_verify_blowup1_fold1_degree8() {
        let (params, dft, challenger) = make_params(1, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup2_fold1_degree8() {
        let (params, dft, challenger) = make_params(2, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup1_fold2_degree8() {
        let (params, dft, challenger) = make_params(1, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }
}

// ---------------------------------------------------------------------------
// Goldilocks tests (quadratic extension)
// ---------------------------------------------------------------------------

mod goldilocks_stir {
    use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};

    use super::*;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;
    type Perm = Poseidon2Goldilocks<8>;
    type MyHash = PaddingFreeSponge<Perm, 8, 4, 4>;
    type MyCompress = TruncatedPermutation<Perm, 2, 4, 8>;
    type ValMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 4>;
    type MyMmcs = ExtensionMmcs<F, EF, ValMmcs>;
    type Dft = Radix2DitParallel<F>;
    type Challenger = DuplexChallenger<F, Perm, 8, 4>;

    fn make_params(
        log_blowup: usize,
        log_folding_factor: usize,
    ) -> (StirParameters<MyMmcs>, Dft, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let mmcs = MyMmcs::new(val_mmcs);

        let params = StirParameters {
            log_blowup,
            log_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs,
        };
        (params, Dft::default(), Challenger::new(perm))
    }

    #[test]
    fn test_prove_verify_blowup1_fold1_degree8() {
        let (params, dft, challenger) = make_params(1, 1);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup2_fold2_degree8() {
        let (params, dft, challenger) = make_params(2, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }
}

// ---------------------------------------------------------------------------
// PCS layer tests (TwoAdicStirPcs via Pcs trait)
// ---------------------------------------------------------------------------

mod babybear_pcs {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_stir::TwoAdicStirPcs;

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2DitParallel<Val>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicStirPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    fn get_pcs() -> (MyPcs, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 1,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs: challenge_mmcs,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs, stir_params);
        (pcs, Challenger::new(perm))
    }

    fn do_test_pcs(log_degrees: &[usize]) {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let mut p_challenger = challenger_template.clone();

        // Commit: one round with multiple matrices.
        let domains_and_polys: Vec<_> = log_degrees
            .iter()
            .map(|&log_d| {
                let d = 1 << log_d;
                let width = 3;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, width),
                )
            })
            .collect();

        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domains_and_polys.iter().cloned());
        p_challenger.observe(commit.clone());

        let zeta: Challenge = p_challenger.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let data_and_points = vec![(&data, points)];
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_challenger);

        // Verify.
        let mut v_challenger = challenger_template;
        v_challenger.observe(commit.clone());
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims: Vec<_> = domains_and_polys
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![(commit, claims)],
            &proof,
            &mut v_challenger,
        )
        .unwrap_or_else(|e| panic!("PCS verification failed: {e:?}"));
    }

    #[test]
    fn test_pcs_single_degree8() {
        do_test_pcs(&[8]);
    }

    #[test]
    fn test_pcs_single_degree4() {
        do_test_pcs(&[4]);
    }

    #[test]
    fn test_pcs_multiple_equal_degree6() {
        do_test_pcs(&[6, 6, 6]);
    }

    #[test]
    fn test_pcs_multiple_different_degrees() {
        do_test_pcs(&[4, 6, 8]);
    }
}
