//! End-to-end STIR prove/verify tests.
//!
//! Each test constructs a polynomial, runs `prove_stir`, then `verify_stir`, and checks
//! that the proof verifies. Tests cover BabyBear (quartic extension), KoalaBear (quartic
//! extension), and Goldilocks (quadratic extension).

use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{ExtensionMmcs, Mmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::SecurityAssumption;
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
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + Clone,
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
        make_params_with_soundness(
            log_blowup,
            log_folding_factor,
            SecurityAssumption::CapacityBound,
            security_level,
            max_pow_bits,
        )
    }

    fn make_params_with_soundness(
        log_blowup: usize,
        log_folding_factor: usize,
        soundness_type: SecurityAssumption,
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
            log_starting_folding_factor: log_folding_factor,
            soundness_type,
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

    /// `(params, dft, challenger)` with an explicit, possibly different, round-0 folding
    /// factor — the other `make_*` helpers above always set `log_starting_folding_factor`
    /// equal to `log_folding_factor`.
    fn make_two_tier_params(
        log_blowup: usize,
        log_starting_folding_factor: usize,
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
            log_starting_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs,
        };
        (params, Dft::default(), Challenger::new(perm))
    }

    #[test]
    fn test_prove_verify_two_tier_folding_schedule() {
        // Round 0 folds by k0=4 (log=2); every later round, and the final direct-send
        // stage, folds by k=8 (log=3). log_starting_degree=10 gives after_starting_fold=8,
        // extra_folds=2 (floor(8/3)), total_folds=3, num_rounds=2 (round0=k0, round1=k),
        // log_final_degree=2 — exercises both arities across an intermediate round and the
        // final stage.
        let (params, dft, challenger) = make_two_tier_params(1, 2, 3);
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(10, params.clone());
        assert_eq!(config.num_rounds(), 2);
        assert_eq!(config.round_configs[0].log_folding_factor, 2);
        assert_eq!(config.round_configs[1].log_folding_factor, 3);
        assert_eq!(config.final_log_folding_factor(), 3);
        assert_eq!(config.log_final_degree, 2);

        do_test_stir_prove_verify::<F, EF, Dft, MyMmcs, Challenger>(&params, &dft, &challenger, 10);
    }

    #[test]
    fn test_prove_verify_two_tier_zero_intermediate_rounds() {
        // after_starting_fold = 5 - 2 = 3 < 4 = k, so round 0's k0-fold IS the final fold:
        // exercises `final_log_folding_factor()` returning k0 (not the steady-state k) when
        // num_rounds == 0.
        let (params, dft, challenger) = make_two_tier_params(1, 2, 4);
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(5, params);
        assert_eq!(config.num_rounds(), 0);
        assert_eq!(config.final_log_folding_factor(), 2);
        assert_eq!(config.log_final_degree, 3);

        let mut rng = seeded_rng();
        let degree = 1usize << 5;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let mut p_ch = challenger.clone();
        let (proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_ch);
        assert!(proof.round_proofs.is_empty());

        let mut v_ch = challenger;
        verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect("verification of two-tier num_rounds == 0 protocol failed");
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
    fn test_johnson_bound_prove_verify_with_grinding() {
        // Keep the algebraic target fixed at 20 bits while reserving 8 bits for PoW.
        // This exercises the BCSS25-derived Johnson eta and both grinding phases without
        // making the ordinary test suite expensive.
        let (params, dft, challenger) =
            make_params_with_soundness(2, 2, SecurityAssumption::JohnsonBound, 28, 8);
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(POW_LOG_DEGREE, params);

        assert_eq!(config.soundness_type, SecurityAssumption::JohnsonBound);
        assert!(
            config
                .round_configs
                .iter()
                .all(|rc| rc.num_ood_samples == 1 && rc.eta.is_finite() && rc.eta > 0.)
        );
        assert!(config.final_eta.is_finite() && config.final_eta > 0.);
        assert!(
            config.round_configs.iter().any(|rc| rc.pow_bits > 0) || config.final_pow_bits > 0,
            "Johnson-bound test parameters must exercise query grinding"
        );
        assert!(
            config
                .round_configs
                .iter()
                .any(|rc| rc.folding_pow_bits > 0)
                || config.final_folding_pow_bits > 0,
            "Johnson-bound test parameters must exercise folding grinding"
        );

        let mut rng = seeded_rng();
        let degree = 1usize << POW_LOG_DEGREE;
        let poly: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let mut p_ch = challenger.clone();
        let (proof, _idx) = prove_stir(&config, poly, &dft, &mut p_ch);

        let mut v_ch = challenger;
        verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect("Johnson-bound proof with PoW grinding should verify");
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
    fn test_tampered_ood_answer_invalidates_following_pow_witness() {
        let (config, dft, challenger, poly) = pow_proof_setup();
        let round_with_pow = config
            .round_configs
            .iter()
            .position(|rc| rc.pow_bits > 0)
            .expect("expected at least one intermediate round with pow_bits > 0");

        let mut p_ch = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly, &dft, &mut p_ch);

        assert!(!proof.round_proofs[round_with_pow].ood_answers.is_empty());
        proof.round_proofs[round_with_pow].ood_answers[0] += EF::from(F::ONE);

        let mut v_ch = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_ch)
            .expect_err("tampered OOD answer must invalidate the following PoW witness");
        assert!(
            matches!(
                err,
                p3_stir::StirError::InvalidPowWitness { round } if round == round_with_pow
            ),
            "expected InvalidPowWitness in round {round_with_pow}, got {err:?}"
        );
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
        // With `parallel` enabled, grinding may choose any valid witness found by
        // `find_any`, so two proving runs from the same FS state need not produce
        // byte-identical proofs. The invariant STIR needs is that each proof's
        // witness-driven transcript replays through the verifier, including the
        // first-round query indices returned for PCS binding.
        let (config, dft, challenger, poly) = pow_proof_setup();

        let mut p_ch_a = challenger.clone();
        let (proof_a, idx_a) = prove_stir(&config, poly.clone(), &dft, &mut p_ch_a);
        let mut v_ch_a = challenger.clone();
        let outputs_a = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof_a, &mut v_ch_a)
            .expect("first proof should verify under transcript replay");
        assert_eq!(idx_a, outputs_a.first_round_indices);

        let mut p_ch_b = challenger.clone();
        let (proof_b, idx_b) = prove_stir(&config, poly, &dft, &mut p_ch_b);
        let mut v_ch_b = challenger;
        let outputs_b = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof_b, &mut v_ch_b)
            .expect("second proof should verify under transcript replay");
        assert_eq!(idx_b, outputs_b.first_round_indices);
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
            assert_eq!(
                rp_a.query_openings.as_ref().unwrap().row_evals.len(),
                rp_b.query_openings.as_ref().unwrap().row_evals.len()
            );
        }
    }

    #[test]
    fn test_tampered_round_query_opening_fails() {
        // Covers arity 4, 8, and 16 (log_folding_factor 2, 3, 4).
        for log_folding_factor in [2, 3, 4] {
            let (params, dft, challenger) = make_params(1, log_folding_factor);
            let mut rng = seeded_rng();
            let log_degree = 8;
            let degree = 1usize << log_degree;
            let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

            let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
            let mut p_challenger = challenger.clone();
            let (mut proof, _query_indices) =
                prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

            let row_evals = &mut proof.round_proofs[0]
                .query_openings
                .as_mut()
                .unwrap()
                .row_evals;
            assert!(!row_evals.is_empty());
            row_evals[0][0] += EF::ONE;

            let mut v_challenger = challenger;
            assert!(
                verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger)
                    .is_err()
            );
        }
    }

    /// Swapping two `row_evals` entries keeps every length check happy (same count, same
    /// per-row width), so only the positional binding between `row_evals` and the
    /// transcript-derived `query_indices` inside Merkle verification can catch it.
    #[test]
    fn test_permuted_row_evals_rejected() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        let row_evals = &mut proof.round_proofs[0]
            .query_openings
            .as_mut()
            .unwrap()
            .row_evals;
        assert!(row_evals.len() >= 2, "need at least two queries to permute");
        assert_ne!(
            row_evals[0], row_evals[1],
            "rows must differ for the permutation to change anything"
        );
        row_evals.swap(0, 1);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err(),
            "permuted row_evals must be rejected"
        );
    }

    #[test]
    fn test_tampered_pruned_sibling_rejected() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        let sibs = &mut proof.round_proofs[0]
            .query_openings
            .as_mut()
            .unwrap()
            .opening_proof
            .sibling_hashes;
        assert!(!sibs.is_empty());
        sibs[0][0] += F::ONE;

        let mut v_challenger = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger)
            .expect_err("tampered sibling hash must be rejected");
        assert!(matches!(
            err,
            p3_stir::StirError::InvalidMmcsProof { round: 0, .. }
        ));
    }

    #[test]
    fn test_dropped_pruned_sibling_rejected() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _query_indices) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        let sibs = &mut proof.round_proofs[0]
            .query_openings
            .as_mut()
            .unwrap()
            .opening_proof
            .sibling_hashes;
        assert!(!sibs.is_empty());
        sibs.pop();

        let mut v_challenger = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger)
            .expect_err("dropped sibling hash must be rejected");
        assert!(matches!(
            err,
            p3_stir::StirError::InvalidMmcsProof { round: 0, .. }
        ));
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

    #[test]
    fn test_tampered_final_polynomial_fails() {
        // The final polynomial is observed into the FS transcript, so corrupting it both
        // (a) directly breaks the final-fold check at the queried positions and
        // (b) desynchronizes the verifier transcript, causing downstream sample_bits / PoW
        // checks to differ from the prover's. Either way, verification must fail.
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        assert!(!proof.final_polynomial.is_empty());
        proof.final_polynomial[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err(),
            "tampered final_polynomial must be rejected"
        );
    }

    #[test]
    fn test_missing_initial_commitment_rejected() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        // `verify_stir` verifies the initial oracle against its commitment, so a proof that
        // omits the commitment cannot be checked and must be rejected rather than skipped.
        proof.initial_commitment = None;

        let mut v_challenger = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger)
            .expect_err("a missing initial commitment must be rejected");
        assert!(matches!(err, p3_stir::StirError::InvalidProofShape));
    }

    #[test]
    fn test_tampered_final_query_proof_fails() {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let mut p_challenger = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        let row_evals = &mut proof.final_query_openings.as_mut().unwrap().row_evals;
        assert!(!row_evals.is_empty());
        assert!(!row_evals[0].is_empty());
        row_evals[0][0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err(),
            "tampered final_query_openings.row_evals must be rejected"
        );
    }

    #[test]
    fn test_tampered_round_commitment_fails() {
        // Replacing a round commitment with a different value should desynchronize the
        // verifier's transcript and/or fail Merkle verification at the queries.
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let degree = 1usize << log_degree;
        let poly_coeffs: Vec<EF> = (0..degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        assert!(
            config.num_rounds() >= 2,
            "need at least two rounds to swap commitments"
        );

        let mut p_challenger = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        // Swap two distinct round commitments. Both are MMCS commitments so swapping
        // produces a syntactically valid but semantically wrong proof.
        proof.round_proofs.swap(0, 1);

        let mut v_challenger = challenger;
        assert!(
            verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger).is_err(),
            "swapped round commitments must be rejected"
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
            log_starting_folding_factor: log_folding_factor,
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
            log_starting_folding_factor: log_folding_factor,
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
    use p3_fri::{FriParameters, TwoAdicFriPcs};
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
    type MyPcs = TwoAdicStirPcs<Val, Dft, ValMmcs, ChallengeMmcs, Challenge, Challenger>;
    type FriPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

    fn make_mmcs(perm: &Perm) -> (ValMmcs, ChallengeMmcs) {
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        (val_mmcs, challenge_mmcs)
    }

    fn get_pcs() -> (MyPcs, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let (val_mmcs, challenge_mmcs) = make_mmcs(&perm);

        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 2,
            log_starting_folding_factor: 2,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs: challenge_mmcs,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs, stir_params);
        (pcs, Challenger::new(perm))
    }

    #[test]
    fn test_pcs_log_max_lde_height_reserves_blowup_bits() {
        let (pcs, _challenger) = get_pcs();
        assert_eq!(
            <MyPcs as Pcs<Challenge, Challenger>>::log_max_lde_height(&pcs),
            Val::TWO_ADICITY - 1
        );
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
    fn test_pcs_single_degree2_no_intermediate_rounds() {
        // log_stir_degree == log_folding_factor, so STIR runs no intermediate rounds and the
        // final-round queries read the external initial oracle directly.
        do_test_pcs(&[2]);
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
    fn test_pcs_two_tier_multiple_different_degrees() {
        // Round 0 folds by k0=4 (log=2); every later round folds by k=8 (log=3) —
        // exercises the PCS-layer input fiber grouping and reconstruction
        // (`log_starting_folding_factor`) across multiple height buckets under a schedule
        // that changes arity after round 0.
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let (val_mmcs, challenge_mmcs) = make_mmcs(&perm);
        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 3,
            log_starting_folding_factor: 2,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = MyPcs::new(Dft::default(), val_mmcs, stir_params);
        let challenger_template = Challenger::new(perm);

        let log_degrees = [4, 6, 8];
        let mut rng = seeded_rng();
        let mut p_challenger = challenger_template.clone();

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
        .unwrap_or_else(|e| panic!("two-tier PCS verification failed: {e:?}"));
    }

    fn compare_stir_proof_size_with_binary_fri(
        log_degree: usize,
        log_folding_factor: usize,
        width: usize,
    ) -> (usize, usize) {
        const SECURITY_BITS: usize = 32;

        let mut perm_rng = seeded_rng();
        let perm = Perm::new_from_rng_128(&mut perm_rng);

        let (fri_val_mmcs, fri_challenge_mmcs) = make_mmcs(&perm);
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: SECURITY_BITS,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: fri_challenge_mmcs,
        };
        assert_eq!(fri_params.conjectured_soundness_bits(), SECURITY_BITS);
        let fri_pcs = FriPcs::new(Dft::default(), fri_val_mmcs, fri_params);

        let (stir_val_mmcs, stir_challenge_mmcs) = make_mmcs(&perm);
        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor,
            log_starting_folding_factor: log_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: SECURITY_BITS,
            max_pow_bits: 0,
            mmcs: stir_challenge_mmcs,
        };
        let stir_pcs = MyPcs::new(Dft::default(), stir_val_mmcs, stir_params);

        let mut rng = seeded_rng();
        let degree = 1 << log_degree;
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);

        let fri_domain =
            <FriPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&fri_pcs, degree);
        let stir_domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&stir_pcs, degree);
        assert_eq!(fri_domain.size(), stir_domain.size());
        assert_eq!(fri_domain.shift(), stir_domain.shift());

        let mut fri_p_ch = Challenger::new(perm.clone());
        let (fri_commit, fri_data) =
            <FriPcs as Pcs<Challenge, Challenger>>::commit(&fri_pcs, [(fri_domain, mat.clone())]);
        fri_p_ch.observe(fri_commit.clone());
        let zeta: Challenge = fri_p_ch.sample_algebra_element();
        let (fri_openings, fri_proof) = <FriPcs as Pcs<Challenge, Challenger>>::open(
            &fri_pcs,
            vec![(&fri_data, vec![vec![zeta]])],
            &mut fri_p_ch,
        );

        let mut fri_v_ch = Challenger::new(perm.clone());
        fri_v_ch.observe(fri_commit.clone());
        let fri_v_zeta: Challenge = fri_v_ch.sample_algebra_element();
        assert_eq!(fri_v_zeta, zeta);
        let fri_claims = vec![(fri_domain, vec![(zeta, fri_openings[0][0][0].clone())])];
        <FriPcs as Pcs<Challenge, Challenger>>::verify(
            &fri_pcs,
            vec![(fri_commit, fri_claims)],
            &fri_proof,
            &mut fri_v_ch,
        )
        .expect("binary FRI proof should verify");
        let fri_bytes =
            postcard::to_allocvec(&fri_proof).expect("binary FRI proof should serialize");

        let mut stir_p_ch = Challenger::new(perm.clone());
        let (stir_commit, stir_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&stir_pcs, [(stir_domain, mat)]);
        stir_p_ch.observe(stir_commit.clone());
        // STIR's input commitment hashes fiber-grouped leaves, so its root — and hence the
        // point derived from it — differs from FRI's over the same matrix. Proof size does
        // not depend on which point is opened, so the comparison stays like-for-like.
        let stir_zeta: Challenge = stir_p_ch.sample_algebra_element();
        let (stir_openings, stir_proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &stir_pcs,
            vec![(&stir_data, vec![vec![stir_zeta]])],
            &mut stir_p_ch,
        );

        let mut stir_v_ch = Challenger::new(perm);
        stir_v_ch.observe(stir_commit.clone());
        let stir_v_zeta: Challenge = stir_v_ch.sample_algebra_element();
        assert_eq!(stir_v_zeta, stir_zeta);
        let stir_claims = vec![(
            stir_domain,
            vec![(stir_zeta, stir_openings[0][0][0].clone())],
        )];
        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &stir_pcs,
            vec![(stir_commit, stir_claims)],
            &stir_proof,
            &mut stir_v_ch,
        )
        .expect("STIR proof should verify");
        let stir_bytes = postcard::to_allocvec(&stir_proof).expect("STIR proof should serialize");
        let proof_ratio = fri_bytes.len() as f64 / stir_bytes.len() as f64;

        println!(
            "proof-size: log_degree={log_degree}, log_folding_factor={log_folding_factor}, \
             STIR={} bytes, binary FRI={} bytes, FRI/STIR={proof_ratio:.2}x",
            stir_bytes.len(),
            fri_bytes.len()
        );

        // This intentionally measures serialized PCS proof objects only. Opened values are
        // excluded because both proofs open the same point and width. This test reports the
        // current sizes; the regression below asserts the intended ordering.
        (stir_bytes.len(), fri_bytes.len())
    }

    #[test]
    fn test_pcs_proof_size_vs_binary_fri_equivalent_input() {
        const WIDTH: usize = 3;

        for (log_degree, log_folding_factor) in [(14, 2), (16, 2)] {
            compare_stir_proof_size_with_binary_fri(log_degree, log_folding_factor, WIDTH);
        }
    }

    #[test]
    fn assert_stir_proof_smaller_than_binary_fri() {
        const WIDTH: usize = 3;

        for (log_degree, log_folding_factor) in [(14, 2), (16, 2)] {
            let (stir_bytes, fri_bytes) =
                compare_stir_proof_size_with_binary_fri(log_degree, log_folding_factor, WIDTH);
            assert!(
                stir_bytes < fri_bytes,
                "STIR proof ({stir_bytes} bytes) should be smaller than binary FRI proof \
                 ({fri_bytes} bytes) for log_degree={log_degree}, \
                 log_folding_factor={log_folding_factor}"
            );
        }
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

    /// Open and verify across **two independent commitments**: each commitment carries one
    /// matrix, and the opening claims span both. Exercises the multi-`commitments_with_opening_points`
    /// loop in `pcs::open` / `pcs::verify`, which the single-commitment tests miss.
    #[test]
    fn test_pcs_two_commitments() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d_a = 6;
        let log_d_b = 7;
        let width = 3;

        let domain_a =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d_a);
        let mat_a = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d_a, width);
        let domain_b =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d_b);
        let mat_b = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d_b, width);

        let mut p_ch = challenger_template.clone();
        let (commit_a, data_a) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain_a, mat_a)]);
        p_ch.observe(commit_a.clone());
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain_b, mat_b)]);
        p_ch.observe(commit_b.clone());

        let zeta: Challenge = p_ch.sample_algebra_element();

        // Each commitment has one matrix, opened at the same `zeta`.
        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);

        // Verify.
        let mut v_ch = challenger_template;
        v_ch.observe(commit_a.clone());
        v_ch.observe(commit_b.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();

        let commitments_with_claims = vec![
            (commit_a, vec![(domain_a, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain_b, vec![(zeta, opening_b)])]),
        ];

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            commitments_with_claims,
            &proof,
            &mut v_ch,
        )
        .unwrap_or_else(|e| panic!("two-commitment PCS verification failed: {e:?}"));
    }

    /// Two commitments at the **same** LDE height — both contributions land in the same
    /// STIR bucket. The previous verifier compared the per-commit reduced-opening
    /// contribution against STIR's combined fiber evaluation; multi-commit-per-bucket
    /// would slip past the check unless the verifier accumulates across commitments.
    #[test]
    fn test_pcs_two_commitments_same_bucket() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let width = 3;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat_a = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);
        let mat_b = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);

        let mut p_ch = challenger_template.clone();
        let (commit_a, data_a) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_a)]);
        p_ch.observe(commit_a.clone());
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_b)]);
        p_ch.observe(commit_b.clone());

        let zeta: Challenge = p_ch.sample_algebra_element();

        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);

        let mut v_ch = challenger_template;
        v_ch.observe(commit_a.clone());
        v_ch.observe(commit_b.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();
        let claims = vec![
            (commit_a, vec![(domain, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain, vec![(zeta, opening_b)])]),
        ];
        <MyPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_ch)
            .unwrap_or_else(|e| panic!("two-commitment same-bucket verification failed: {e:?}"));
    }

    /// Committing a matrix and then opening it at no points would emit a proof that cannot
    /// verify: the verifier reads native-height class membership off the claimed domains, so
    /// it still counts the matrix as a class even though the prover contributed nothing for
    /// it. Rejecting in `open` turns that into a named prover error rather than a mystery
    /// proof.
    #[test]
    #[should_panic(expected = "was opened at no points")]
    fn test_pcs_open_rejects_matrix_without_opening_points() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat_a = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);
        let mat_b = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);

        let mut p_ch = challenger_template;
        let (commit, data) = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &pcs,
            vec![(domain, mat_a), (domain, mat_b)],
        );
        p_ch.observe(commit);
        let zeta: Challenge = p_ch.sample_algebra_element();

        // `mat_a` is opened at `zeta`; `mat_b` is opened at no points at all.
        let data_and_points = vec![(&data, vec![vec![zeta], vec![]])];
        let _ = <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);
    }

    /// The degenerate extreme of the above: nothing is opened at all, so the prover would
    /// otherwise emit a proof with no STIR instances in it.
    #[test]
    #[should_panic(expected = "was opened at no points")]
    fn test_pcs_open_rejects_a_commitment_opened_at_no_points() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);

        let mut p_ch = challenger_template;
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        p_ch.observe(commit);

        let data_and_points = vec![(&data, vec![vec![]])];
        let _ = <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);
    }

    /// Prove honestly at `prove_log_degrees`, then verify with matrix `emptied`'s claims
    /// stripped to no points. Returns the verifier's result together with a challenger that
    /// was never handed to `verify`, so the caller can check how far the transcript got.
    fn verify_with_matrix_claims_emptied(
        prove_log_degrees: &[usize],
        emptied: usize,
    ) -> (
        Result<(), <MyPcs as Pcs<Challenge, Challenger>>::Error>,
        Challenger,
        Challenger,
    ) {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let domains_and_polys: Vec<_> = prove_log_degrees
            .iter()
            .map(|&log_d| {
                let d = 1 << log_d;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, 3),
                )
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domains_and_polys.iter().cloned());
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = prove_log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&data, points)], &mut p_ch);

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);
        let untouched = v_ch.clone();

        let claims: Vec<_> = domains_and_polys
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .enumerate()
            .map(|(mat_idx, ((domain, _), mat_openings))| {
                let point_claims = if mat_idx == emptied {
                    vec![]
                } else {
                    vec![(zeta, mat_openings[0].clone())]
                };
                (*domain, point_claims)
            })
            .collect();

        let result = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![(commit, claims)],
            &proof,
            &mut v_ch,
        );
        (result, v_ch, untouched)
    }

    #[test]
    fn test_pcs_matrix_without_opening_points_rejected() {
        let (result, _, _) = verify_with_matrix_claims_emptied(&[6, 6], 1);
        let err = result.expect_err("a matrix claimed at no points must be rejected");
        assert!(
            matches!(
                err,
                p3_stir::StirError::MatrixWithoutOpeningPoints {
                    commitment: 0,
                    matrix: 1,
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn test_pcs_matrix_without_opening_points_rejected_before_the_transcript_forks() {
        // With distinct native heights the emptied matrix is the *only* member of its class,
        // so whether it counts decides whether `Combine` runs and therefore whether `r_comb`
        // is drawn. Rejecting before anything reaches the transcript is what keeps the cause
        // legible: were the check left inside the per-bucket work, the config divergence
        // would surface first, as an unrelated-looking failure.
        let (result, mut used, mut untouched) = verify_with_matrix_claims_emptied(&[8, 6], 1);
        let err = result.expect_err("a matrix claimed at no points must be rejected");
        assert!(
            matches!(
                err,
                p3_stir::StirError::MatrixWithoutOpeningPoints {
                    commitment: 0,
                    matrix: 1,
                }
            ),
            "{err:?}"
        );

        // Nothing was observed or sampled, so the two challengers still agree.
        let after: Challenge = used.sample_algebra_element();
        let expected: Challenge = untouched.sample_algebra_element();
        assert_eq!(
            after, expected,
            "verify touched the transcript before rejecting"
        );
    }

    /// A proof with the per-commitment input-openings vector dropped (truncated) must be
    /// rejected. Without the up-front length check, `zip` would silently ignore the missing
    /// commitments, letting a malicious proof omit input bindings entirely.
    #[test]
    fn test_pcs_truncated_input_openings_rejected() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let width = 3;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat_a = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);
        let mat_b = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);

        let mut p_ch = challenger_template.clone();
        let (commit_a, data_a) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_a)]);
        p_ch.observe(commit_a.clone());
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_b)]);
        p_ch.observe(commit_b.clone());

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, mut proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);

        // Drop the second commitment's input-opening vector. The verifier must reject:
        // skipping a commit's openings would let the proof verify against a proper subset
        // of the public input.
        for (_stir_proof, input_openings) in proof.iter_mut() {
            assert_eq!(input_openings.len(), 2);
            input_openings.pop();
        }

        let mut v_ch = challenger_template;
        v_ch.observe(commit_a.clone());
        v_ch.observe(commit_b.clone());
        let _v_zeta: Challenge = v_ch.sample_algebra_element();

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();
        let claims = vec![
            (commit_a, vec![(domain, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain, vec![(zeta, opening_b)])]),
        ];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_ch);
        assert!(
            matches!(res, Err(p3_stir::StirError::InvalidProofShape)),
            "truncated input_openings must be rejected as InvalidProofShape, got {res:?}"
        );
    }

    /// A present-but-should-be-`Some` per-commitment input opening turned into `None` must
    /// be rejected. This is distinct from truncation: the slot still exists at the right
    /// index, only its content is dropped, so only the `has_at_bucket` shape check (not a
    /// length mismatch) can catch it.
    #[test]
    fn test_pcs_input_opening_present_to_none_rejected() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let width = 3;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat_a = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);
        let mat_b = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);

        let mut p_ch = challenger_template.clone();
        let (commit_a, data_a) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_a)]);
        p_ch.observe(commit_a.clone());
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_b)]);
        p_ch.observe(commit_b.clone());

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, mut proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);

        // Both commitments land in the same bucket, so both slots start as `Some`. Blank
        // out the first one in place, keeping the vector's length untouched.
        for (_stir_proof, input_openings) in proof.iter_mut() {
            assert_eq!(input_openings.len(), 2);
            assert!(input_openings[0].is_some());
            input_openings[0] = None;
        }

        let mut v_ch = challenger_template;
        v_ch.observe(commit_a.clone());
        v_ch.observe(commit_b.clone());
        let _v_zeta: Challenge = v_ch.sample_algebra_element();

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();
        let claims = vec![
            (commit_a, vec![(domain, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain, vec![(zeta, opening_b)])]),
        ];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_ch);
        assert!(
            matches!(res, Err(p3_stir::StirError::InvalidProofShape)),
            "Some -> None input opening must be rejected as InvalidProofShape, got {res:?}"
        );
    }

    /// A per-commitment `opened_values` vector truncated to fewer rows than the queried
    /// positions must be rejected before the MMCS multi-batch verification (which expects
    /// matching lengths) is even called.
    #[test]
    fn test_pcs_opened_values_truncated_rejected() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();
        let log_d = 6;
        let width = 3;

        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![(&data, vec![vec![zeta]])],
            &mut p_ch,
        );

        for (_stir_proof, input_openings) in proof.iter_mut() {
            let opening = input_openings[0]
                .as_mut()
                .expect("single commitment must have a present opening");
            assert!(!opening.opened_values.is_empty());
            opening.opened_values.pop();
        }

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let opening = opening_values[0][0][0].clone();
        let claims = vec![(commit, vec![(domain, vec![(zeta, opening)])])];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_ch);
        assert!(
            matches!(res, Err(p3_stir::StirError::InvalidProofShape)),
            "truncated opened_values must be rejected as InvalidProofShape, got {res:?}"
        );
    }

    /// Tampering with the alpha-batched opening value (the claimed `f_i(z)`) should be
    /// rejected by the input-MMCS binding check inside `pcs::verify`.
    /// Commit and open honestly at `prove_log_degrees`, then verify against
    /// `claim_log_degrees`. Every native height in the claims must give the same shared LDE
    /// height as the honest commitment, so the two sides agree on the bucket and disagree
    /// only on how many `Combine` classes it holds.
    fn verify_with_claimed_degrees(
        prove_log_degrees: &[usize],
        claim_log_degrees: &[usize],
    ) -> Result<(), <MyPcs as Pcs<Challenge, Challenger>>::Error> {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let domains_and_polys: Vec<_> = prove_log_degrees
            .iter()
            .map(|&log_d| {
                let d = 1 << log_d;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, 3),
                )
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domains_and_polys.iter().cloned());
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = prove_log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&data, points)], &mut p_ch);

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims: Vec<_> = claim_log_degrees
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .map(|(&log_d, mat_openings)| {
                let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                    &pcs,
                    1 << log_d,
                );
                (domain, vec![(zeta, mat_openings[0].clone())])
            })
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![(commit, claims)],
            &proof,
            &mut v_ch,
        )
    }

    #[test]
    fn test_pcs_verify_rejects_understated_native_height() {
        // Prover sees one class (no `Combine`, no `r_comb` drawn); the verifier is told matrix
        // 1 sits at 2^6, so it sees two classes, draws `r_comb`, and applies degree
        // correction. The degree-correction gap is the only thing binding a short class's
        // degree bound, so this is what stops a prover passing a degree-2^8 polynomial off as
        // a degree-2^6 one.
        let err = verify_with_claimed_degrees(&[8, 8], &[8, 6])
            .expect_err("an understated native height must be rejected");
        assert!(
            matches!(err, p3_stir::StirError::InvalidProofShape),
            "{err:?}"
        );
    }

    #[test]
    fn test_pcs_verify_rejects_overstated_native_height() {
        // The mirror image: the prover ran `Combine` over two classes, the verifier is told
        // there is only one and skips it entirely.
        let err = verify_with_claimed_degrees(&[8, 6], &[8, 8])
            .expect_err("an overstated native height must be rejected");
        assert!(
            matches!(err, p3_stir::StirError::InvalidProofShape),
            "{err:?}"
        );
    }

    #[test]
    fn test_pcs_honest_claims_verify_across_native_height_classes() {
        // Control for the two rejections above: the same shape passes when claimed honestly.
        verify_with_claimed_degrees(&[8, 6], &[8, 6])
            .unwrap_or_else(|e| panic!("honest multi-class proof must verify: {e:?}"));
    }

    #[test]
    fn test_pcs_tampered_opening_in_short_combine_class_fails() {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        // `test_pcs_tampered_opening_value_fails` runs at a single height, so it never
        // exercises a `Combine`d bucket. Perturbing the *short* class specifically is what
        // catches a wrong per-class coefficient or degree-correction gap: a mistake there
        // still yields a low-degree combined codeword, so only a value that must not fit can
        // separate the two.
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();
        let log_degrees = [8usize, 6];

        let domains_and_polys: Vec<_> = log_degrees
            .iter()
            .map(|&log_d| {
                let d = 1 << log_d;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, 3),
                )
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domains_and_polys.iter().cloned());
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&data, points)], &mut p_ch);

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let mut tampered_short = opening_values[0][1][0].clone();
        tampered_short[0] += Challenge::from(Val::ONE);

        let claims: Vec<_> = domains_and_polys
            .iter()
            .enumerate()
            .map(|(mat_idx, (domain, _))| {
                let vals = if mat_idx == 1 {
                    tampered_short.clone()
                } else {
                    opening_values[0][mat_idx][0].clone()
                };
                (*domain, vec![(zeta, vals)])
            })
            .collect();

        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![(commit, claims)],
            &proof,
            &mut v_ch,
        );
        assert!(
            res.is_err(),
            "PCS verify must reject a tampered opening in the short Combine class"
        );
    }

    #[test]
    fn test_pcs_johnson_bound_multiple_native_height_classes() {
        // Every other PCS test here runs under `CapacityBound`. The Johnson regime derives
        // `Combine`'s eta from BCSS25's multiplicity `m`, which grows fast enough in `d*` that
        // this shape is right at the edge of feasibility — so it is the configuration that
        // actually exercises the round-0 eta ceiling rather than passing it comfortably.
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let (val_mmcs, challenge_mmcs) = make_mmcs(&perm);
        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 2,
            log_starting_folding_factor: 2,
            soundness_type: SecurityAssumption::JohnsonBound,
            security_level: 64,
            max_pow_bits: 0,
            mmcs: challenge_mmcs,
        };
        let pcs = MyPcs::new(Dft::default(), val_mmcs, stir_params);
        let challenger_template = Challenger::new(perm);

        let log_degrees = [14usize, 12];
        let mut rng = seeded_rng();
        let domains_and_polys: Vec<_> = log_degrees
            .iter()
            .map(|&log_d| {
                let d = 1 << log_d;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, 1),
                )
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, domains_and_polys.iter().cloned());
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, vec![(&data, points)], &mut p_ch);

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
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
            &mut v_ch,
        )
        .unwrap_or_else(|e| panic!("JohnsonBound PCS verification failed: {e:?}"));
    }

    #[test]
    fn test_pcs_tampered_opening_value_fails() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();
        let log_d = 6;
        let width = 3;

        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![(&data, vec![vec![zeta]])],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        // Tamper one coordinate of the claimed evaluation. The reduced opening the verifier
        // reconstructs then diverges from the codeword STIR actually folded, so the round
        // consistency checks must reject.
        let mut tampered = opening_values[0][0][0].clone();
        tampered[0] += Challenge::from(Val::ONE);

        let claims = vec![(commit, vec![(domain, vec![(zeta, tampered)])])];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_ch);
        assert!(
            res.is_err(),
            "PCS verify must reject a tampered claimed opening"
        );
    }

    #[test]
    fn test_pcs_rejects_stray_initial_commitment() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();
        let log_d = 6;

        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![(&data, vec![vec![zeta]])],
            &mut p_ch,
        );

        // The PCS runs STIR with an external initial oracle, so a proof carrying a commitment
        // to it is malformed: accepting one would let a prover feed the transcript an extra
        // message the verifier never checks.
        proof[0].0.initial_commitment = Some(proof[0].0.round_proofs[0].commitment.clone());

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        let claims = vec![(
            commit,
            vec![(domain, vec![(v_zeta, opening_values[0][0][0].clone())])],
        )];
        let err = <MyPcs as Pcs<Challenge, Challenger>>::verify(&pcs, claims, &proof, &mut v_ch)
            .expect_err("a stray initial commitment must be rejected");
        assert!(matches!(err, p3_stir::StirError::InvalidProofShape));
    }
}

// ---------------------------------------------------------------------------
// Multi-instance lockstep driver: grind sharing across STIR height buckets.
// ---------------------------------------------------------------------------

mod babybear_stir_multi {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_stir::prover::{prove_stir, prove_stir_multi};
    use p3_stir::verifier::verify_stir_multi;

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
            log_starting_folding_factor: log_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level,
            max_pow_bits,
            mmcs,
        };
        (params, Dft::default(), Challenger::new(perm))
    }

    type Instances = (Vec<StirConfig<F, EF, MyMmcs, Challenger>>, Vec<Vec<EF>>);

    /// Build one config per `log_degrees` entry (all sharing `params`) plus a random
    /// polynomial for each.
    fn make_instances(params: &StirParameters<MyMmcs>, log_degrees: &[usize]) -> Instances {
        let mut rng = seeded_rng();
        let configs = log_degrees
            .iter()
            .map(|&log_degree| {
                StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params.clone())
            })
            .collect();
        let polys = log_degrees
            .iter()
            .map(|&log_degree| (0..1usize << log_degree).map(|_| rng.random()).collect())
            .collect();
        (configs, polys)
    }

    /// Run `prove_stir_multi` then `verify_stir_multi` over one bucket per `log_degrees` entry
    /// and assert the proof verifies.
    fn do_test_multi_prove_verify(
        params: &StirParameters<MyMmcs>,
        dft: &Dft,
        challenger_template: &Challenger,
        log_degrees: &[usize],
    ) {
        let (configs, polys) = make_instances(params, log_degrees);
        let config_refs: Vec<&StirConfig<F, EF, MyMmcs, Challenger>> = configs.iter().collect();

        let mut p_ch = challenger_template.clone();
        let results = prove_stir_multi(&config_refs, polys, dft, &mut p_ch);
        assert_eq!(results.len(), log_degrees.len());

        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();
        let mut v_ch = challenger_template.clone();
        verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proofs, &mut v_ch)
            .unwrap_or_else(|e| {
                panic!("multi-bucket verification failed for log_degrees={log_degrees:?}: {e}")
            });
    }

    #[test]
    fn test_multi_one_bucket() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        do_test_multi_prove_verify(&params, &dft, &challenger, &[8]);
    }

    #[test]
    fn test_multi_two_buckets_ratio2() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        do_test_multi_prove_verify(&params, &dft, &challenger, &[9, 8]);
    }

    #[test]
    fn test_multi_two_buckets_ratio8() {
        // log_degree 11 vs 8: LDE-height ratio 2^3 = 8. Grind sharing is ratio-independent,
        // so this must configure and verify exactly like the ratio-2 case.
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        do_test_multi_prove_verify(&params, &dft, &challenger, &[11, 8]);
    }

    #[test]
    fn test_multi_three_buckets_ratio2_steps() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        do_test_multi_prove_verify(&params, &dft, &challenger, &[10, 9, 8]);
    }

    #[test]
    fn test_multi_three_buckets_ratio8_spread() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        do_test_multi_prove_verify(&params, &dft, &challenger, &[12, 9, 8]);
    }

    #[test]
    fn test_multi_one_bucket_matches_single_instance_bytes() {
        // At B=1 the shared-grind schedule degenerates to the single-instance schedule, so
        // the multi-driver must reproduce the exact same transcript and proof bytes.
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        let log_degree = 8;
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);

        let mut rng = seeded_rng();
        let poly: Vec<EF> = (0..1usize << log_degree).map(|_| rng.random()).collect();

        let mut p_ch_single = challenger.clone();
        let (single_proof, single_idx) = prove_stir(&config, poly.clone(), &dft, &mut p_ch_single);

        let config_refs = [&config];
        let mut p_ch_multi = challenger;
        let results = prove_stir_multi(&config_refs, vec![poly], &dft, &mut p_ch_multi);
        assert_eq!(results.len(), 1);
        let (multi_proof, multi_idx) = &results[0];

        assert_eq!(single_idx, *multi_idx);
        let single_bytes = postcard::to_allocvec(&single_proof).expect("serialize");
        let multi_bytes = postcard::to_allocvec(multi_proof).expect("serialize");
        assert_eq!(
            single_bytes, multi_bytes,
            "B=1 multi-driver must be byte-identical to the single-instance path"
        );
    }

    #[test]
    fn test_multi_witness_replayed_across_grind_sites_rejected() {
        let (params, dft, challenger) = make_params(1, 2, 32, 12);
        let log_degree = 8;
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let round_with_pow = config
            .round_configs
            .iter()
            .position(|rc| rc.pow_bits > 0)
            .expect("expected a round with pow_bits > 0");

        let mut rng = seeded_rng();
        let poly: Vec<EF> = (0..1usize << log_degree).map(|_| rng.random()).collect();

        let config_refs = [&config];
        let mut p_ch = challenger.clone();
        let results = prove_stir_multi(&config_refs, vec![poly], &dft, &mut p_ch);
        let (mut proof, _idx) = results.into_iter().next().expect("one instance");

        // Replay the folding-grind witness at the query-grind site of the same round: the two
        // grinds bind different transcript states, so this must be rejected.
        proof.round_proofs[round_with_pow].pow_witness =
            proof.round_proofs[round_with_pow].folding_pow_witness;

        let mut v_ch = challenger;
        let proofs = [&proof];
        let err = verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proofs, &mut v_ch)
            .expect_err("a witness replayed from another grind site must be rejected");
        assert!(matches!(err, p3_stir::StirError::InvalidPowWitness { .. }));
    }

    #[test]
    fn test_multi_disagreeing_replicated_witness_rejected() {
        let (params, dft, challenger) = make_params(1, 2, 32, 12);
        let log_degree = 8;
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        // Two identical-height instances so their round indices line up 1:1 and share every
        // grind site exactly.
        let config_refs = [&config, &config];

        let mut rng = seeded_rng();
        let polys = vec![
            (0..1usize << log_degree).map(|_| rng.random()).collect(),
            (0..1usize << log_degree).map(|_| rng.random()).collect(),
        ];

        let mut p_ch = challenger.clone();
        let mut results = prove_stir_multi(&config_refs, polys, &dft, &mut p_ch).into_iter();
        let proof_a = results.next().expect("bucket a").0;
        let mut proof_b = results.next().expect("bucket b").0;

        // Disagree bucket B's round-0 folding-grind witness from bucket A's replicated copy.
        proof_b.round_proofs[0].folding_pow_witness += F::ONE;

        let mut v_ch = challenger;
        let proofs = [&proof_a, &proof_b];
        let err = verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proofs, &mut v_ch)
            .expect_err("disagreeing replicated witnesses across buckets must be rejected");
        assert!(matches!(err, p3_stir::StirError::InvalidProofShape));
    }
}
