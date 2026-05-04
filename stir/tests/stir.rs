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

    let (proof, _query_indices) = prove_stir(&config, poly_coeffs, dft, &mut p_challenger);

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

    fn make_params_full(
        log_blowup: usize,
        log_folding_factor: usize,
        security_level: usize,
        max_pow_bits: usize,
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
            security_level,
            max_pow_bits,
            mmcs,
        };
        (params, Dft::default(), Challenger::new(perm))
    }

    fn make_params(
        log_blowup: usize,
        log_folding_factor: usize,
    ) -> (StirParameters<MyMmcs>, Dft, Challenger) {
        // Low security + no PoW for fast tests. Real deployments use
        // security_level=128 and appropriate max_pow_bits.
        make_params_full(log_blowup, log_folding_factor, 16, 0)
    }

    #[test]
    fn test_prove_verify_blowup1_fold2_degree8() {
        let (params, dft, challenger) = make_params(1, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup2_fold2_degree8() {
        let (params, dft, challenger) = make_params(2, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup1_fold3_degree12() {
        let (params, dft, challenger) = make_params(1, 3);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 12);
    }

    #[test]
    fn test_prove_verify_blowup1_fold2_degree4() {
        let (params, dft, challenger) = make_params(1, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 4);
    }

    #[test]
    fn test_prove_verify_blowup1_fold2_degree12() {
        let (params, dft, challenger) = make_params(1, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 12);
    }

    #[test]
    fn test_prove_verify_zero_intermediate_rounds() {
        // log_starting_degree == log_folding_factor ⇒ total_folds = 1, num_rounds = 0:
        // exercise the final-only path including the `prev_ctx == None` branch in
        // `materialize_virtual_fiber`.
        let (params, dft, challenger) = make_params(1, 3);
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(3, params);
        assert_eq!(config.num_rounds(), 0);
        assert_eq!(config.log_final_degree, 0);

        let mut rng = seeded_rng();
        let degree = 1usize << 3;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let mut p_ch = challenger.clone();
        let (proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_ch);
        assert!(proof.round_proofs.is_empty());

        let mut v_ch = challenger;
        verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect("verification of num_rounds == 0 protocol failed");
    }

    // ---------------------------------------------------------------------------
    // Proof-of-work / grinding tests
    //
    // Default tests use max_pow_bits=0, which short-circuits `grind` and `check_witness`
    // and bypasses the entire grinding code path. These tests configure parameters such
    // that derived `pow_bits` is ≥ 1 (typically ~10 with security_level=32, max_pow_bits=12),
    // exercising the actual grind loop and the `observe(witness) + sample_bits == 0` check.
    // ---------------------------------------------------------------------------

    /// Returns `(params, dft, challenger)` tuned so derived per-round `pow_bits` is positive.
    fn make_pow_params() -> (StirParameters<MyMmcs>, Dft, Challenger) {
        // security_level=32, max_pow_bits=12 gives derived pow_bits ~10 per round on
        // BabyBear quartic — small enough to run in a few ms but exercises the grinding
        // code path end-to-end.
        make_params_full(1, 2, 32, 12)
    }

    /// Fixed log_starting_degree used by the PoW tests.
    const POW_LOG_DEGREE: usize = 8;

    fn pow_proof_setup() -> (
        StirConfig<F, EF, MyMmcs, Challenger>,
        Dft,
        Challenger,
        Vec<EF>,
    ) {
        let (params, dft, challenger) = make_pow_params();
        let mut rng = seeded_rng();
        let degree = 1usize << POW_LOG_DEGREE;
        let poly: Vec<EF> = (0..degree).map(|_| rng.random()).collect();
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(POW_LOG_DEGREE, params);
        (config, dft, challenger, poly)
    }

    #[test]
    fn test_prove_verify_with_grinding() {
        let (config, dft, challenger, poly) = pow_proof_setup();

        // Sanity: the test is only meaningful if at least one round actually grinds.
        let any_query_pow =
            config.round_configs.iter().any(|rc| rc.pow_bits > 0) || config.final_pow_bits > 0;
        assert!(
            any_query_pow,
            "PoW test parameters must produce at least one round with pow_bits > 0"
        );

        let mut p_ch = challenger.clone();
        let (proof, _idx) = prove_stir(&config, poly, &dft, &mut p_ch);

        let mut v_ch = challenger;
        verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect("verification with PoW grinding failed");
    }

    #[test]
    fn test_tampered_round_pow_witness_fails() {
        let (config, dft, challenger, poly) = pow_proof_setup();
        let round_with_pow = config
            .round_configs
            .iter()
            .position(|rc| rc.pow_bits > 0)
            .expect("expected at least one intermediate round with pow_bits > 0");

        let mut p_ch = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly, &dft, &mut p_ch);

        // Corrupt the query-phase PoW witness. After observing the bogus witness, the
        // sampled bits will (with overwhelming probability) not all be zero, so
        // `check_witness` returns false and verify rejects.
        proof.round_proofs[round_with_pow].pow_witness += F::ONE;

        let mut v_ch = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect_err("tampered pow_witness must be rejected");
        assert!(matches!(err, p3_stir::StirError::InvalidPowWitness { .. }));
    }

    #[test]
    fn test_tampered_final_pow_witness_fails() {
        let (config, dft, challenger, poly) = pow_proof_setup();
        assert!(
            config.final_pow_bits > 0,
            "expected final_pow_bits > 0 under PoW test parameters"
        );

        let mut p_ch = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly, &dft, &mut p_ch);

        proof.final_pow_witness += F::ONE;

        let mut v_ch = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect_err("tampered final_pow_witness must be rejected");
        assert!(matches!(err, p3_stir::StirError::InvalidPowWitness { .. }));
    }

    #[test]
    fn test_grinding_proof_verifies_under_replay() {
        // Two prove runs from the same FS state must produce identical proofs even with
        // grinding active (grind is deterministic given the transcript).
        let (config, dft, challenger, poly) = pow_proof_setup();

        let mut p_ch_a = challenger.clone();
        let (proof_a, idx_a) = prove_stir(&config, poly.clone(), &dft, &mut p_ch_a);
        let mut p_ch_b = challenger;
        let (proof_b, idx_b) = prove_stir(&config, poly, &dft, &mut p_ch_b);

        assert_eq!(idx_a, idx_b);
        assert_eq!(proof_a.final_polynomial, proof_b.final_polynomial);
        assert_eq!(proof_a.final_pow_witness, proof_b.final_pow_witness);
        assert_eq!(
            proof_a.final_folding_pow_witness,
            proof_b.final_folding_pow_witness
        );
        assert_eq!(proof_a.round_proofs.len(), proof_b.round_proofs.len());
        for (a, b) in proof_a.round_proofs.iter().zip(proof_b.round_proofs.iter()) {
            assert_eq!(a.pow_witness, b.pow_witness);
            assert_eq!(a.folding_pow_witness, b.folding_pow_witness);
        }
    }

    #[test]
    fn test_prove_is_deterministic() {
        // Cloning the challenger and re-running `prove_stir` must produce the same proof
        // (Fiat-Shamir transcript determinism).
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_ch_a = challenger.clone();
        let mut p_ch_b = challenger;
        let (proof_a, idx_a) = prove_stir(&config, poly_coeffs.clone(), &dft, &mut p_ch_a);
        let (proof_b, idx_b) = prove_stir(&config, poly_coeffs, &dft, &mut p_ch_b);

        assert_eq!(idx_a, idx_b);
        assert_eq!(
            proof_a.final_polynomial, proof_b.final_polynomial,
            "final_polynomial must be deterministic under FS replay"
        );
        assert_eq!(proof_a.round_proofs.len(), proof_b.round_proofs.len());
        for (rp_a, rp_b) in proof_a.round_proofs.iter().zip(proof_b.round_proofs.iter()) {
            assert_eq!(rp_a.ood_answers, rp_b.ood_answers);
            assert_eq!(rp_a.ans_polynomial, rp_b.ans_polynomial);
            assert_eq!(rp_a.shake_polynomial, rp_b.shake_polynomial);
            assert_eq!(rp_a.query_proofs.len(), rp_b.query_proofs.len());
        }
    }

    #[test]
    fn test_tampered_round_query_opening_fails() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        assert!(!proof.round_proofs[0].query_proofs.is_empty());
        proof.round_proofs[0].query_proofs[0].row_evals[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err()
        );
    }

    #[test]
    fn test_tampered_ood_answer_fails() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        assert!(!proof.round_proofs[0].ood_answers.is_empty());
        proof.round_proofs[0].ood_answers[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err()
        );
    }

    #[test]
    fn test_tampered_ans_polynomial_fails() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        // Tamper the prover-supplied answer polynomial. The shake identity at the
        // verifier-sampled rho should catch it with overwhelming probability.
        assert!(!proof.round_proofs[0].ans_polynomial.is_empty());
        proof.round_proofs[0].ans_polynomial[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err()
        );
    }

    #[test]
    fn test_tampered_shake_polynomial_fails() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        assert!(!proof.round_proofs[0].shake_polynomial.is_empty());
        proof.round_proofs[0].shake_polynomial[0] += EF::from(F::ONE);

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
    fn test_prove_verify_blowup1_fold2_degree8() {
        let (params, dft, challenger) = make_params(1, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup2_fold2_degree8() {
        let (params, dft, challenger) = make_params(2, 2);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 8);
    }

    #[test]
    fn test_prove_verify_blowup1_fold3_degree12() {
        let (params, dft, challenger) = make_params(1, 3);
        do_test_stir_prove_verify::<F, EF, _, _, _>(&params, &dft, &challenger, 12);
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
    fn test_prove_verify_blowup1_fold2_degree8() {
        let (params, dft, challenger) = make_params(1, 2);
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
            log_folding_factor: 2,
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

    #[test]
    #[should_panic(expected = "is below the minimum")]
    fn test_pcs_rejects_too_small_matrix() {
        // log_folding_factor = 2 ⇒ minimum natural matrix height = 2^2 = 4. A height-2
        // matrix should be rejected at commit time with a clear error rather than
        // panicking deep inside StirConfig::new.
        let (pcs, _challenger) = get_pcs();
        let mut rng = seeded_rng();
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 2);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 2, 3);
        let _ = <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
    }
}
