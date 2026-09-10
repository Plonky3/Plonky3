//! End-to-end STIR prove/verify tests.
//!
//! Each test constructs a polynomial, runs `prove_stir`, then `verify_stir`, and checks
//! that the proof verifies. Tests cover BabyBear (quartic extension), KoalaBear (quartic
//! extension), and Goldilocks (quadratic extension).

use core::fmt::Debug;

use p3_challenger::{
    CanObserve, CanSampleUniformBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{ExtensionMmcs, Mmcs, Pcs, UnivariateStarkPcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::config::{StirConfig, StirOptions, StirParameters};
use p3_stir::proof::StirProof;
use p3_stir::prover::{codeword_from_coeffs, prove_stir, prove_stir_from_external_codeword};
use p3_stir::verifier::{verify_stir, verify_stir_with_external_initial};
use p3_stir::{
    ExternalSourceError, GrindStage, ProofShapeError, RoundLabel, SecurityAssumption, StirError,
};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

fn seeded_rng() -> SmallRng {
    SmallRng::seed_from_u64(42)
}

/// The shape error inside a `StirError`, or a panic naming what came instead.
fn shape_of<E: Debug, IE: Debug>(err: StirError<E, IE>) -> ProofShapeError {
    match err {
        StirError::InvalidProofShape(shape) => shape,
        other => panic!("expected a shape error, got {other:?}"),
    }
}

/// `draws`, ascending and deduplicated: the shape the unique-index lists carry.
fn sorted_dedup(draws: &[usize]) -> Vec<usize> {
    let mut unique = draws.to_vec();
    unique.sort_unstable();
    unique.dedup();
    unique
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
    F: TwoAdicField + PrimeField64,
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
    fn early_stop_roundtrips_and_checks_final_coefficients() {
        for (degree, starting_fold, cap, rounds, final_log) in [
            (12, 2, 6, 2, 6),
            (10, 3, 3, 2, 3),
            (10, 3, 5, 1, 5),
            (10, 3, usize::MAX, 0, 7),
        ] {
            let (mut params, dft, challenger) = make_params(1, 2);
            params.log_starting_folding_factor = starting_fold;
            let config = StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
                degree,
                params,
                StirOptions {
                    max_log_final_poly_len: Some(cap),
                    ..Default::default()
                },
            );
            let mut rng = seeded_rng();
            let poly: Vec<EF> = (0..1usize << degree).map(|_| rng.random()).collect();
            let mut p_ch = challenger.clone();
            let (mut proof, _) = prove_stir(&config, poly, &dft, &mut p_ch);
            assert_eq!(proof.round_proofs.len(), rounds);
            assert_eq!(proof.final_polynomial.len(), 1 << final_log);
            let mut v_ch = challenger.clone();
            verify_stir(&config, &proof, &mut v_ch).unwrap();
            assert_eq!(
                p_ch.sample_algebra_element::<EF>(),
                v_ch.sample_algebra_element::<EF>()
            );

            *proof.final_polynomial.last_mut().unwrap() += EF::ONE;
            assert!(verify_stir(&config, &proof, &mut challenger.clone()).is_err());
        }
    }

    #[test]
    fn early_stop_reduces_serialized_proof_size() {
        let (params, dft, mut challenger) = make_params_full(1, 2, 80, 0);
        let full = StirConfig::<F, EF, MyMmcs, Challenger>::new(14, params.clone());
        let early = StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
            14,
            params,
            StirOptions {
                max_log_final_poly_len: Some(6),
                ..Default::default()
            },
        );
        let mut rng = seeded_rng();
        let poly: Vec<EF> = (0..1 << 14).map(|_| rng.random()).collect();
        let (full_proof, _) = prove_stir(&full, poly.clone(), &dft, &mut challenger.clone());
        let (early_proof, _) = prove_stir(&early, poly, &dft, &mut challenger.clone());
        verify_stir(&full, &full_proof, &mut challenger.clone()).unwrap();
        verify_stir(&early, &early_proof, &mut challenger).unwrap();
        assert!(
            postcard::to_allocvec(&early_proof).unwrap().len()
                < postcard::to_allocvec(&full_proof).unwrap().len()
        );
    }

    #[test]
    fn compact_answers_preserve_proofs_and_transcripts() {
        for (degree, kind, cap) in [
            (8, 0, None),
            (8, 1, None),
            (8, 2, None),
            (3, 0, None),
            (10, 0, Some(6)),
        ] {
            let (params, dft, challenger) = make_params(1, 2);
            let options = StirOptions {
                max_log_final_poly_len: cap,
                ..Default::default()
            };
            let full = StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
                degree,
                params.clone(),
                options,
            );
            let compact = StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
                degree,
                params,
                StirOptions {
                    compact_answers: true,
                    ..options
                },
            );
            let mut rng = seeded_rng();
            let mut poly: Vec<EF> = (0..1 << degree).map(|_| rng.random()).collect();
            if kind != 0 {
                poly.fill(EF::ZERO);
                if kind == 2 {
                    poly[0] = EF::from_u64(7);
                }
            }
            let mut full_p = challenger.clone();
            let mut compact_p = challenger.clone();
            let (full_proof, full_queries) = prove_stir(&full, poly.clone(), &dft, &mut full_p);
            let (compact_proof, compact_queries) = prove_stir(&compact, poly, &dft, &mut compact_p);
            assert_eq!(full_queries, compact_queries);
            let mut full_v = challenger.clone();
            let mut compact_v = challenger.clone();
            verify_stir(&full, &full_proof, &mut full_v).unwrap();
            verify_stir(&compact, &compact_proof, &mut compact_v).unwrap();
            let next: EF = full_p.sample_algebra_element();
            assert_eq!(next, compact_p.sample_algebra_element::<EF>());
            assert_eq!(next, full_v.sample_algebra_element::<EF>());
            assert_eq!(next, compact_v.sample_algebra_element::<EF>());

            if kind == 1 {
                assert!(
                    full_proof
                        .round_proofs
                        .iter()
                        .all(|r| r.ans_polynomial == [EF::ZERO])
                );
            }
            let mut omitted = full_proof.clone();
            for round in &mut omitted.round_proofs {
                round.ans_polynomial.clear();
            }
            let full_bytes = postcard::to_allocvec(&full_proof).unwrap();
            let compact_bytes = postcard::to_allocvec(&compact_proof).unwrap();
            assert_eq!(compact_bytes, postcard::to_allocvec(&omitted).unwrap());
            if compact.num_rounds() == 0 {
                assert_eq!(compact_bytes, full_bytes);
                continue;
            }
            assert!(compact_bytes.len() < full_bytes.len());
            for coefficients in [
                full_proof.round_proofs[0].ans_polynomial.clone(),
                vec![EF::ONE],
            ] {
                let mut bad = compact_proof.clone();
                let got = coefficients.len();
                bad.round_proofs[0].ans_polynomial = coefficients;
                let err = verify_stir(&compact, &bad, &mut challenger.clone()).unwrap_err();
                assert_eq!(
                    shape_of(err),
                    ProofShapeError::UnexpectedAnsPolynomial {
                        round: RoundLabel::Round(0),
                        got
                    }
                );
            }
            if degree == 8 && kind == 0 {
                let mut bad = compact_proof.clone();
                bad.round_proofs[0].ood_answers[0] += EF::ONE;
                assert!(verify_stir(&compact, &bad, &mut challenger.clone()).is_err());
                let mut bad = compact_proof;
                bad.round_proofs[0]
                    .query_openings
                    .as_mut()
                    .unwrap()
                    .row_evals[0][0] += EF::ONE;
                assert!(verify_stir(&compact, &bad, &mut challenger.clone()).is_err());
            }
        }
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
        assert!(
            matches!(
                err,
                StirError::InvalidPowWitness { round, .. }
                    if round == RoundLabel::Round(round_with_pow)
            ),
            "{err:?}"
        );
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
                StirError::InvalidPowWitness { round, .. }
                    if round == RoundLabel::Round(round_with_pow)
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
        assert!(
            matches!(
                err,
                StirError::InvalidPowWitness {
                    round: RoundLabel::Final,
                    ..
                }
            ),
            "{err:?}"
        );
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
        assert_eq!(
            sorted_dedup(&outputs_a.first_round_draws),
            outputs_a.first_round_indices
        );

        let mut p_ch_b = challenger.clone();
        let (proof_b, idx_b) = prove_stir(&config, poly, &dft, &mut p_ch_b);
        let mut v_ch_b = challenger;
        let outputs_b = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof_b, &mut v_ch_b)
            .expect("second proof should verify under transcript replay");
        assert_eq!(idx_b, outputs_b.first_round_indices);
        assert_eq!(
            sorted_dedup(&outputs_b.first_round_draws),
            outputs_b.first_round_indices
        );
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
            StirError::InvalidMmcsProof {
                round: RoundLabel::Round(0),
                ..
            }
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
            StirError::InvalidMmcsProof {
                round: RoundLabel::Round(0),
                ..
            }
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

        // Tamper the prover-supplied answer polynomial. The tampered Ans no longer
        // interpolates the values the verifier derives for itself, so the identity at the
        // verifier-sampled rho catches it before any later check runs.
        assert!(!proof.round_proofs[0].ans_polynomial.is_empty());
        proof.round_proofs[0].ans_polynomial[0] += EF::from(F::ONE);

        let mut v_challenger = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger)
            .expect_err("a tampered answer polynomial must be rejected");
        assert!(matches!(
            err,
            p3_stir::StirError::InvalidAnsConsistency {
                round: RoundLabel::Round(0)
            }
        ));
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
        assert_eq!(shape_of(err), ProofShapeError::MissingInitialCommitment);
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

    /// Prove with an uncommitted initial oracle, then verify against a caller-supplied source.
    ///
    /// `mutate` tampers with the proof.
    /// `fibers` rewrites the honest fibers the source would have returned.
    ///
    /// Lane `l` of query `j` sits at natural-order position `j + l * fold_height`.
    fn verify_external_initial(
        mutate: impl FnOnce(&mut StirProof<EF, MyMmcs, F>),
        fibers: impl FnOnce(Vec<Vec<EF>>) -> Vec<Vec<EF>>,
    ) -> Result<(), StirError<<MyMmcs as Mmcs<EF>>::Error>> {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let coeffs: Vec<EF> = (0..1usize << log_degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let log_domain = config.log_starting_domain_size();
        let codeword = codeword_from_coeffs(&dft, coeffs, F::GENERATOR, log_domain);

        // Binding the codeword before proving is the caller's job. Observing its values
        // stands in for the PCS layer's input commitments.
        let mut p_challenger = challenger.clone();
        p_challenger.observe_algebra_slice(&codeword);
        let (mut proof, _idx) =
            prove_stir_from_external_codeword(&config, codeword.clone(), &dft, &mut p_challenger);
        mutate(&mut proof);

        let arity0 = 1usize << config.log_starting_folding_factor;
        let fold_height = (1usize << log_domain) / arity0;

        let mut v_challenger = challenger;
        v_challenger.observe_algebra_slice(&codeword);
        verify_stir_with_external_initial(&config, &proof, &mut v_challenger, |js| {
            let honest = js
                .iter()
                .map(|&j| (0..arity0).map(|l| codeword[j + l * fold_height]).collect())
                .collect();
            Ok(fibers(honest))
        })
        .map(|_| ())
    }

    #[test]
    fn test_external_initial_oracle_verifies() {
        verify_external_initial(|_| {}, |honest| honest)
            .unwrap_or_else(|e| panic!("an honest external oracle must verify: {e:?}"));
    }

    #[test]
    fn test_external_source_returning_too_few_fibers_rejected() {
        let err = verify_external_initial(
            |_| {},
            |mut honest| {
                honest.pop();
                honest
            },
        )
        .expect_err("a short fiber list must be rejected");
        let StirError::ExternalSource(source) = err else {
            panic!("expected an external-source error, got {err:?}");
        };
        assert_eq!(
            source,
            ExternalSourceError::FiberCount {
                round: RoundLabel::Round(0),
                expected: 20,
                got: 19,
            }
        );
    }

    #[test]
    fn test_external_source_returning_short_fiber_rejected() {
        let err = verify_external_initial(
            |_| {},
            |mut honest| {
                honest[0].pop();
                honest
            },
        )
        .expect_err("a fiber below the round's arity must be rejected");
        let StirError::ExternalSource(source) = err else {
            panic!("expected an external-source error, got {err:?}");
        };
        assert_eq!(
            source,
            ExternalSourceError::FiberArity {
                round: RoundLabel::Round(0),
                fiber: 0,
                expected: 4,
                got: 3,
            }
        );
    }

    #[test]
    fn test_external_oracle_carrying_query_openings_rejected() {
        let err = verify_external_initial(
            |proof| {
                // Nothing commits to an externally bound oracle, so rows shipped against it
                // are unauthenticated.
                proof.round_proofs[0].query_openings = proof.final_query_openings.clone();
                assert!(proof.round_proofs[0].query_openings.is_some());
            },
            |honest| honest,
        )
        .expect_err("openings against an external oracle must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::UnexpectedQueryOpenings {
                round: RoundLabel::Round(0),
            }
        );
    }

    /// Prove a fixed instance, apply `mutate`, and return the shape error the verifier reports.
    ///
    /// Every mutation below breaks a length the configuration pins.
    fn shape_error_after(mutate: impl FnOnce(&mut StirProof<EF, MyMmcs, F>)) -> ProofShapeError {
        let (params, dft, challenger) = make_params(1, 2);
        let mut rng = seeded_rng();
        let log_degree = 8;
        let poly_coeffs: Vec<EF> = (0..1usize << log_degree).map(|_| rng.random()).collect();

        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        assert_eq!(config.num_rounds(), 3);
        let mut p_challenger = challenger.clone();
        let (mut proof, _idx) = prove_stir(&config, poly_coeffs, &dft, &mut p_challenger);

        mutate(&mut proof);

        let mut v_challenger = challenger;
        let err = verify_stir::<F, EF, MyMmcs, Challenger>(&config, &proof, &mut v_challenger)
            .expect_err("a shape-mutated proof must be rejected");
        shape_of(err)
    }

    #[test]
    fn test_dropped_round_proof_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs.pop();
        });
        assert_eq!(
            err,
            ProofShapeError::RoundCount {
                instance: None,
                expected: 3,
                got: 2,
            }
        );
    }

    #[test]
    fn test_extra_ood_answer_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs[0].ood_answers.push(EF::ONE);
        });
        assert_eq!(
            err,
            ProofShapeError::OodAnswerCount {
                round: RoundLabel::Round(0),
                expected: 2,
                got: 3,
            }
        );
    }

    #[test]
    fn test_dropped_ood_answer_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs[0].ood_answers.pop();
        });
        assert_eq!(
            err,
            ProofShapeError::OodAnswerCount {
                round: RoundLabel::Round(0),
                expected: 2,
                got: 1,
            }
        );
    }

    /// Mutating round 1 rather than round 0 pins that the round index is threaded through.
    #[test]
    fn test_extra_ood_answer_in_a_later_round_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs[1].ood_answers.push(EF::ONE);
        });
        assert_eq!(
            err,
            ProofShapeError::OodAnswerCount {
                round: RoundLabel::Round(1),
                expected: 2,
                got: 3,
            }
        );
    }

    #[test]
    fn test_overlong_ans_polynomial_rejected() {
        // The transcript describes the answer step with the loosest cap a round can reach:
        // one point per OOD sample plus one per query draw, before any of them collide.
        //
        //     described cap = num_ood_samples + num_queries = 2 + 21 = 23
        //
        // A proof above that cap is rejected before the run is described at all.
        let err = shape_error_after(|proof| {
            proof.round_proofs[0].ans_polynomial.resize(1024, EF::ONE);
        });
        assert_eq!(
            err,
            ProofShapeError::AnsPolynomialTooLong {
                round: RoundLabel::Round(0),
                maximum: 23,
                got: 1024,
            }
        );
    }

    #[test]
    fn test_ans_polynomial_above_the_dedup_bound_rejected() {
        // Below the described cap but above the round's actual point count.
        //
        // The draw settles how many query indices collide.
        //
        // So this bound is only known once the round has run, and the rejection lands
        // mid-transcript, where the verifier has to release its completeness check first.
        let err = shape_error_after(|proof| {
            proof.round_proofs[0].ans_polynomial.resize(22, EF::ONE);
        });
        assert!(
            matches!(
                err,
                ProofShapeError::AnsPolynomialTooLong {
                    round: RoundLabel::Round(0),
                    maximum,
                    got: 22,
                } if maximum < 22
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn test_missing_query_openings_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs[0].query_openings = None;
        });
        assert_eq!(
            err,
            ProofShapeError::MissingQueryOpenings {
                round: RoundLabel::Round(0),
            }
        );
    }

    /// The final round reads its oracle through the same path, under its own label.
    #[test]
    fn test_missing_final_query_openings_rejected() {
        let err = shape_error_after(|proof| {
            proof.final_query_openings = None;
        });
        assert_eq!(
            err,
            ProofShapeError::MissingQueryOpenings {
                round: RoundLabel::Final,
            }
        );
    }

    #[test]
    fn test_dropped_opened_row_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs[0]
                .query_openings
                .as_mut()
                .expect("a committed oracle carries its openings")
                .row_evals
                .pop();
        });
        assert_eq!(
            err,
            ProofShapeError::QueryOpeningCount {
                round: RoundLabel::Round(0),
                expected: 21,
                got: 20,
            }
        );
    }

    #[test]
    fn test_short_opened_row_rejected() {
        let err = shape_error_after(|proof| {
            proof.round_proofs[0]
                .query_openings
                .as_mut()
                .expect("a committed oracle carries its openings")
                .row_evals[0]
                .pop();
        });
        assert_eq!(
            err,
            ProofShapeError::OpenedRowArity {
                round: RoundLabel::Round(0),
                query: 0,
                expected: 4,
                got: 3,
            }
        );
    }

    #[test]
    fn test_overlong_final_polynomial_rejected() {
        let err = shape_error_after(|proof| {
            proof.final_polynomial.push(EF::ONE);
        });
        assert_eq!(
            err,
            ProofShapeError::FinalPolynomialLength {
                expected: 1,
                got: 2,
            }
        );
    }

    /// A prover that truncates rather than pads is the other side of the same `!=`.
    #[test]
    fn test_truncated_final_polynomial_rejected() {
        let err = shape_error_after(|proof| {
            proof.final_polynomial.pop();
        });
        assert_eq!(
            err,
            ProofShapeError::FinalPolynomialLength {
                expected: 1,
                got: 0,
            }
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

    /// Observe a commitment through the single `CanObserve` call `StarkGenericConfig` asks
    /// for: the root count, then one root per shared-domain group.
    fn observe_commitment(
        challenger: &mut Challenger,
        commit: &<MyPcs as Pcs<Challenge, Challenger>>::Commitment,
    ) {
        challenger.observe(commit.clone());
    }

    /// `log_starting_folding_factor` of [`get_pcs_with_spread`]'s STIR parameters: the round-0
    /// fold arity, i.e. the length of each fiber the PCS opens against the committed initial
    /// oracle.
    const LOG_STARTING_FOLDING_FACTOR: usize = 2;

    fn get_pcs() -> (MyPcs, Challenger) {
        get_pcs_with_spread(p3_stir::DEFAULT_MAX_LOG_HEIGHT_SPREAD)
    }

    /// [`get_pcs`] with an explicit cap on how wide a native-height spread may share one LDE
    /// domain.
    fn get_pcs_with_spread(max_log_height_spread: usize) -> (MyPcs, Challenger) {
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let (val_mmcs, challenge_mmcs) = make_mmcs(&perm);

        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 2,
            log_starting_folding_factor: LOG_STARTING_FOLDING_FACTOR,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs: challenge_mmcs,
        };

        let pcs = MyPcs::new(Dft::default(), val_mmcs, stir_params)
            .with_max_log_height_spread(max_log_height_spread);
        (pcs, Challenger::new(perm))
    }

    /// `get_evaluations_on_domain` on a domain taller than the committed LDE — the shape
    /// `uni-stark`'s prover asks for whenever the quotient domain exceeds `log_blowup`, and the
    /// one case the fast slice-of-the-committed-rows path cannot serve, so it forces the general
    /// interpolate-and-reevaluate path.
    #[test]
    fn get_evaluations_on_domain_extrapolates_beyond_the_committed_lde() {
        use p3_field::coset::TwoAdicMultiplicativeCoset;
        use p3_matrix::Matrix;

        let (pcs, _) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 4;
        let d = 1usize << log_d;
        let width = 3;
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, d, width);

        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d);
        let (_, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace.clone())]);

        // Strictly taller than the committed LDE (`d` folded by `log_blowup = 1`), so the fast
        // path's `lde.height() >= domain.size()` guard cannot fire.
        let tall_domain = TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_d + 2).unwrap();
        let evals = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain(
            &pcs,
            &data,
            0,
            tall_domain,
        );
        let evals = evals.to_row_major_matrix();

        let dft = Dft::default();
        let mut coeffs = dft.idft_batch(trace);
        let coeffs_width = coeffs.width();
        coeffs
            .values
            .resize(tall_domain.size() * coeffs_width, Val::ZERO);
        let expected = dft
            .coset_dft_batch(coeffs, tall_domain.shift())
            .to_row_major_matrix();

        assert_eq!(evals, expected);
    }

    /// The general `get_evaluations_on_domain` path must interpolate the committed polynomial
    /// before changing cosets. Exercise both a native-size target and a target taller than the
    /// committed LDE so neither can use the borrowed `GENERATOR`-shift prefix.
    #[test]
    fn get_evaluations_on_changed_shift_matches_direct_evaluation() {
        use p3_field::coset::TwoAdicMultiplicativeCoset;
        use p3_matrix::Matrix;

        let (pcs, _) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 4;
        let d = 1usize << log_d;
        let width = 3;
        let trace = RowMajorMatrix::<Val>::rand(&mut rng, d, width);
        let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, d);
        let (_, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, [(domain, trace.clone())]);

        let dft = Dft::default();
        let native_coeffs = dft.idft_batch(trace);
        for log_target in [log_d, log_d + 2] {
            let target = TwoAdicMultiplicativeCoset::new(Val::from_u64(7), log_target).unwrap();
            let actual =
                <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_evaluations_on_domain(
                    &pcs, &data, 0, target,
                )
                .to_row_major_matrix();

            let mut padded_coeffs = native_coeffs.clone();
            padded_coeffs
                .values
                .resize(target.size() * width, Val::ZERO);
            let expected = dft
                .coset_dft_batch(padded_coeffs, target.shift())
                .to_row_major_matrix();

            assert_eq!(
                actual, expected,
                "changed-shift evaluation disagrees at log_target={log_target}"
            );
        }
    }

    /// Commit `log_degrees`, with `widths[i]` columns in matrix `i`, open every matrix at one
    /// shared point, and verify.
    ///
    /// Returns the commitment so callers can inspect how many groups the layout produced.
    fn round_trip_under(
        pcs: &MyPcs,
        challenger_template: &Challenger,
        log_degrees: &[usize],
        widths: &[usize],
    ) -> <MyPcs as Pcs<Challenge, Challenger>>::Commitment {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let mut rng = seeded_rng();
        let domains_and_polys: Vec<_> = log_degrees
            .iter()
            .zip(widths)
            .map(|(&log_d, &width)| {
                let d = 1 << log_d;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(pcs, d),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, width),
                )
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(pcs, domains_and_polys.iter().cloned());
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points,
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template.clone();
        observe_commitment(&mut v_ch, &commit);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims: Vec<_> = domains_and_polys
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            pcs,
            vec![(commit.clone(), claims).into()],
            &proof,
            &mut v_ch,
        )
        .unwrap_or_else(|e| panic!("verification failed: {e:?}"));

        commit
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(8))]

        #[test]
        fn test_pcs_round_trips_over_random_layouts(
            max_log_height_spread in 0usize..=4,
            log_degrees in proptest::collection::vec(
                LOG_STARTING_FOLDING_FACTOR..=6usize,
                1..=4,
            ),
            widths in proptest::collection::vec(1usize..=4, 1..=4),
        ) {
            // Invariant: an honest opening verifies whatever layout the claims imply.
            //
            // That layout describes the transcript of the commitment scheme entirely.
            //
            // A failure here is the two sides disagreeing on the description itself.
            //
            // No proof-shape check could catch that.
            //
            // Fixture state: 1 to 4 matrices, log-height 2 to 6, width 1 to 4.
            //
            // The spread cap ranges over 0 to 4.
            //
            //     spread 0        -> one bucket per distinct height, nothing merged
            //     spread >= range -> one bucket, every height merged into it
            //
            // Both ends and the mixed middle move the bucket count.
            //
            // They move the merged-class count and every claimed width too.
            //
            // So they move the description the two sides have to agree on.
            let widths: Vec<usize> = log_degrees
                .iter()
                .enumerate()
                .map(|(i, _)| widths[i % widths.len()])
                .collect();

            let (pcs, challenger_template) = get_pcs_with_spread(max_log_height_spread);
            round_trip_under(&pcs, &challenger_template, &log_degrees, &widths);
        }
    }

    #[test]
    fn test_pcs_round_trips_at_every_spread() {
        // The same claim must verify whichever layout the spread cap produces: one STIR
        // instance per distinct height at 0, everything merged onto one domain at 8, and the
        // mixed cases in between.
        let log_degrees = [8usize, 6, 4];
        for (max_log_height_spread, expected_roots) in [(0, 3), (1, 3), (2, 2), (8, 1)] {
            let (pcs, challenger_template) = get_pcs_with_spread(max_log_height_spread);
            let commit = round_trip_under(&pcs, &challenger_template, &log_degrees, &[3, 3, 3]);
            assert_eq!(
                commit.len(),
                expected_roots,
                "spread {max_log_height_spread} should give {expected_roots} groups"
            );
        }
    }

    #[test]
    fn test_pcs_round_trips_with_interleaved_heights_and_widths() {
        // Every other round trip commits descending heights at one uniform width, which makes
        // `group_of_matrix` monotone: a group's claim indices are then a contiguous prefix, so
        // the remap from a group's tree order back to the commitment's claim order is the
        // identity and a mixup would read the right slot by accident. The same goes for
        // per-matrix widths. Ascending, interleaved and repeated heights, each at its own
        // width, make both index sets non-contiguous and the widths distinguishable.
        for spread in [0usize, 1, 2, 3, 8] {
            let (pcs, challenger_template) = get_pcs_with_spread(spread);
            round_trip_under(
                &pcs,
                &challenger_template,
                &[4, 8, 4, 8, 6],
                &[2, 5, 3, 7, 4],
            );
        }

        let (pcs, challenger_template) = get_pcs_with_spread(2);
        round_trip_under(&pcs, &challenger_template, &[6, 8, 2, 4], &[1, 9, 4, 2]);
    }

    #[test]
    fn test_pcs_commit_ldes_groups_like_commit_and_round_trips() {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        // `commit_ldes` takes matrices already extended to their own native blowup and
        // re-extends the short ones onto their group's domain, so it has to reach the same
        // layout `commit` does from the same heights — otherwise batch-stark's quotient
        // commitment would be laid out differently from its trace commitments.
        let log_degrees = [8usize, 6, 4];
        for max_log_height_spread in [0usize, 2, 64] {
            let (pcs, challenger_template) = get_pcs_with_spread(max_log_height_spread);
            let mut rng = seeded_rng();

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

            let ldes = <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::get_quotient_ldes(
                &pcs,
                domains_and_polys.iter().cloned(),
                1,
            );

            let mut p_ch = challenger_template.clone();
            let (commit, data) =
                <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::commit_ldes(&pcs, ldes);

            let (direct_commit, _) = <MyPcs as Pcs<Challenge, Challenger>>::commit(
                &pcs,
                domains_and_polys.iter().cloned(),
            );
            // Equal roots is the whole claim, not just an equal group count: two layouts can
            // agree on how many trees they build and still put different heights on different
            // domains. `[8, 7, 4]` gives `{8,7} | {4}` at spread 1 and `{8} | {7,4}` at spread
            // 3 — two trees each, different LDE heights, different class sets — so a length
            // comparison cannot see the failure this is about. Matching roots pin the
            // dimensions and the extended values together.
            assert_eq!(
                commit, direct_commit,
                "commit_ldes and commit disagree on the layout at spread \
                 {max_log_height_spread}"
            );

            observe_commitment(&mut p_ch, &commit);
            let zeta: Challenge = p_ch.sample_algebra_element();
            let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
            let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
                &pcs,
                vec![p3_commit::OpeningRequest {
                    prover_data: &data,
                    points,
                }],
                &mut p_ch,
            );

            let mut v_ch = challenger_template;
            observe_commitment(&mut v_ch, &commit);
            let v_zeta: Challenge = v_ch.sample_algebra_element();
            assert_eq!(v_zeta, zeta);

            let claims: Vec<_> = domains_and_polys
                .iter()
                .zip(opening_values.first().unwrap().iter())
                .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
                .collect();

            <MyPcs as Pcs<Challenge, Challenger>>::verify(
                &pcs,
                vec![(commit, claims).into()],
                &proof,
                &mut v_ch,
            )
            .unwrap_or_else(|e| {
                panic!("commit_ldes round trip failed at spread {max_log_height_spread}: {e:?}")
            });
        }
    }

    #[test]
    fn test_pcs_merges_classes_pooled_from_several_commitments() {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        // A bucket pools every group topped at the same native height, across commitments, and
        // merges their classes into one `Combine`. Here commitment A holds {2^8, 2^6} and B
        // holds {2^8, 2^7}: both groups sit on the 2^9 domain, so the bucket runs `Combine`
        // over the union {2^8, 2^7, 2^6} — a class set neither commitment holds on its own.
        // This is what the band-width probe is sized against.
        let (pcs, challenger_template) = get_pcs_with_spread(2);
        let mut rng = seeded_rng();

        let domains = |log_ds: &[usize]| -> Vec<_> {
            log_ds
                .iter()
                .map(|&log_d| {
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                        &pcs,
                        1 << log_d,
                    )
                })
                .collect()
        };

        let domains_a: Vec<_> = domains(&[8, 6]);
        let domains_b: Vec<_> = domains(&[8, 7]);
        let mats_a: Vec<_> = domains_a
            .iter()
            .map(|d| (*d, RowMajorMatrix::<Val>::rand(&mut rng, d.size(), 3)))
            .collect();
        let mats_b: Vec<_> = domains_b
            .iter()
            .map(|d| (*d, RowMajorMatrix::<Val>::rand(&mut rng, d.size(), 3)))
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit_a, data_a) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, mats_a.iter().cloned());
        observe_commitment(&mut p_ch, &commit_a);
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, mats_b.iter().cloned());
        observe_commitment(&mut p_ch, &commit_b);

        // Both commitments span two octaves, which the cap admits, so each is a single group.
        assert_eq!(commit_a.len(), 1);
        assert_eq!(commit_b.len(), 1);

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points = vec![
            (&data_a, vec![vec![zeta], vec![zeta]]),
            (&data_b, vec![vec![zeta], vec![zeta]]),
        ];
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );

        // One shared 2^9 domain, so one STIR instance for both commitments.
        assert_eq!(proof.buckets.len(), 1);

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit_a);
        observe_commitment(&mut v_ch, &commit_b);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims = |commit_idx: usize, doms: &[<MyPcs as Pcs<Challenge, Challenger>>::Domain]| {
            doms.iter()
                .enumerate()
                .map(|(mat_idx, domain)| {
                    (
                        *domain,
                        vec![(zeta, opening_values[commit_idx][mat_idx][0].clone())],
                    )
                })
                .collect::<Vec<_>>()
        };

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![
                (commit_a, claims(0, &domains_a)).into(),
                (commit_b, claims(1, &domains_b)).into(),
            ],
            &proof,
            &mut v_ch,
        )
        .unwrap_or_else(|e| panic!("pooled-class verification failed: {e:?}"));
    }

    #[test]
    fn test_pcs_bucket_skips_a_commitment_that_has_no_group_on_its_domain() {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        // Three commitments at spread 1, so that every bucket sees both states a commitment
        // can be in. A and C hold {2^8, 2^4} (C in the opposite caller order) and split into
        // two groups each, on the 2^9 and 2^5 domains; B holds only {2^6}, one group on the
        // 2^7 domain. The three buckets are then 9, 7 and 5, and each one has a commitment
        // that contributes nothing to it — the `None` input-opening slot, which a proof over a
        // single bucket never reaches.
        let (pcs, challenger_template) = get_pcs_with_spread(1);
        let mut rng = seeded_rng();

        let commit_shapes: [&[usize]; 3] = [&[8, 4], &[6], &[4, 8]];
        let mats: Vec<Vec<_>> = commit_shapes
            .iter()
            .map(|log_ds| {
                log_ds
                    .iter()
                    .map(|&log_d| {
                        let domain =
                            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                                &pcs,
                                1 << log_d,
                            );
                        (domain, RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3))
                    })
                    .collect()
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let mut commits = Vec::new();
        let mut datas = Vec::new();
        for per_commit in &mats {
            let (commit, data) =
                <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, per_commit.iter().cloned());
            observe_commitment(&mut p_ch, &commit);
            commits.push(commit);
            datas.push(data);
        }
        assert_eq!(
            commits.iter().map(|c| c.len()).collect::<Vec<_>>(),
            vec![2, 1, 2]
        );

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points: Vec<_> = datas
            .iter()
            .zip(&mats)
            .map(|(data, per_commit)| p3_commit::OpeningRequest {
                prover_data: data,
                points: per_commit.iter().map(|_| vec![zeta]).collect(),
            })
            .collect();
        let (opening_values, proof) =
            <MyPcs as Pcs<Challenge, Challenger>>::open(&pcs, data_and_points, &mut p_ch);

        // Buckets 2^9, 2^7 and 2^5, and B reaches only the middle one.
        assert_eq!(proof.buckets.len(), 3);

        let mut v_ch = challenger_template;
        for commit in &commits {
            observe_commitment(&mut v_ch, commit);
        }
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let commitments_with_claims: Vec<_> = commits
            .into_iter()
            .zip(&mats)
            .enumerate()
            .map(|(commit_idx, (commit, per_commit))| {
                let claims = per_commit
                    .iter()
                    .enumerate()
                    .map(|(mat_idx, (domain, _))| {
                        (
                            *domain,
                            vec![(zeta, opening_values[commit_idx][mat_idx][0].clone())],
                        )
                    })
                    .collect::<Vec<_>>();
                (commit, claims).into()
            })
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            commitments_with_claims,
            &proof,
            &mut v_ch,
        )
        .unwrap_or_else(|e| panic!("partially-present commitment verification failed: {e:?}"));
    }

    /// Prove `log_degrees` at `prover_spread`, verify at `verifier_spread`, and return the
    /// error the mismatch produces.
    fn verify_at_a_different_spread(
        log_degrees: &[usize],
        prover_spread: usize,
        verifier_spread: usize,
    ) -> p3_stir::StirError<<ChallengeMmcs as Mmcs<Challenge>>::Error, <ValMmcs as Mmcs<Val>>::Error>
    {
        #[allow(unused_imports)]
        use p3_commit::Pcs as _;

        let (prover_pcs, challenger_template) = get_pcs_with_spread(prover_spread);
        let (verifier_pcs, _) = get_pcs_with_spread(verifier_spread);

        let mut rng = seeded_rng();
        let domains_and_polys: Vec<_> = log_degrees
            .iter()
            .map(|&log_d| {
                let d = 1 << log_d;
                (
                    <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                        &prover_pcs,
                        d,
                    ),
                    RowMajorMatrix::<Val>::rand(&mut rng, d, 3),
                )
            })
            .collect();

        let mut p_ch = challenger_template.clone();
        let (commit, data) = <MyPcs as Pcs<Challenge, Challenger>>::commit(
            &prover_pcs,
            domains_and_polys.iter().cloned(),
        );
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &prover_pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points,
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims: Vec<_> = domains_and_polys
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &verifier_pcs,
            vec![(commit, claims).into()],
            &proof,
            &mut v_ch,
        )
        .expect_err("a proof laid out at a different spread must be rejected")
    }

    #[test]
    fn test_pcs_verify_rejects_a_proof_built_at_a_different_spread() {
        // The layout is not carried in the proof: both sides derive it from the claimed
        // heights and their own parameters. A verifier configured for a different spread
        // therefore expects a different number of trees, and must not accept.
        //
        // `[8, 6, 4]` is one tree at spread 8 and three at spread 0, so the root-count check
        // that runs before anything else rejects it.
        let err = verify_at_a_different_spread(&[8, 6, 4], 8, 0);
        assert_eq!(
            shape_of(err),
            ProofShapeError::CommitmentRootCount {
                commitment: 0,
                expected: 3,
                got: 1,
            }
        );
    }

    #[test]
    fn test_pcs_verify_rejects_a_relaid_out_proof_with_the_same_root_count() {
        // The case the root-count check cannot see, and the one the layout-agreement claim
        // actually rests on. `[8, 7, 4]` is `{8,7} | {4}` at spread 1 and `{8} | {7,4}` at
        // spread 3: two trees either way, so the root-count check and the bucket-count check
        // both pass. Keep both cases — collapsing this one back into a root-count mismatch
        // stops exercising that path.
        //
        // The two layouts put different native heights in each bucket, so the second bucket
        // derives a different `StirConfig` and its round schedule no longer matches.
        let err = verify_at_a_different_spread(&[8, 7, 4], 1, 3);
        assert_eq!(
            shape_of(err),
            ProofShapeError::RoundCount {
                instance: Some(1),
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn test_pcs_log_max_lde_height_reserves_blowup_bits() {
        let (pcs, _challenger) = get_pcs();
        assert_eq!(
            <MyPcs as UnivariateStarkPcs<Challenge, Challenger>>::log_max_lde_height(&pcs),
            Val::TWO_ADICITY - 1
        );
    }

    fn do_test_pcs(log_degrees: &[usize]) {
        use p3_commit::PolynomialSpace;
        use p3_commit::testing::assert_pcs_opening_contract;

        let (pcs, challenger) = get_pcs();
        if log_degrees.len() == 1 {
            // Preserve the isolated schedules on random full-degree inputs. The shared
            // batched contract needs a nonempty commitment after removing a matrix.
            round_trip_under(&pcs, &challenger, log_degrees, &[3]);
            return;
        }
        let matrices: Vec<_> = log_degrees
            .iter()
            .enumerate()
            .map(|(matrix_index, &log_degree)| {
                let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                    &pcs,
                    1 << log_degree,
                );
                let mut x = domain.first_point();
                let mut values = Vec::new();
                for _ in 0..domain.size() {
                    values.extend([
                        x.square() + Val::from_usize(2 + matrix_index),
                        x + Val::from_usize(7 + matrix_index),
                        Val::from_usize(23 + matrix_index),
                        // Reach the degree bound while keeping the expected value independent.
                        x.exp_u64((domain.size() - 1) as u64),
                    ]);
                    x = domain.next_point(x).unwrap();
                }
                (domain, RowMajorMatrix::new(values, 4))
            })
            .collect();
        assert_pcs_opening_contract(&pcs, &challenger, &[matrices], |_, matrix_index, point| {
            vec![
                point.square() + Challenge::from_usize(2 + matrix_index),
                point + Challenge::from_usize(7 + matrix_index),
                Challenge::from_usize(23 + matrix_index),
                point.exp_u64(((1usize << log_degrees[matrix_index]) - 1) as u64),
            ]
        });
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
        // final-round queries are the ones reading the committed initial oracle.
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
        // exercises the PCS-layer lane sampling and position split
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
        observe_commitment(&mut p_challenger, &commit);

        let zeta: Challenge = p_challenger.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let data_and_points = vec![(&data, points)];
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_challenger,
        );

        let mut v_challenger = challenger_template;
        observe_commitment(&mut v_challenger, &commit);
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims: Vec<_> = domains_and_polys
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![(commit, claims).into()],
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
            batch_proof_of_work_bits: 0,
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
            vec![p3_commit::OpeningRequest {
                prover_data: &fri_data,
                points: vec![vec![zeta]],
            }],
            &mut fri_p_ch,
        );

        let mut fri_v_ch = Challenger::new(perm.clone());
        fri_v_ch.observe(fri_commit.clone());
        let fri_v_zeta: Challenge = fri_v_ch.sample_algebra_element();
        assert_eq!(fri_v_zeta, zeta);
        let fri_claims = vec![(fri_domain, vec![(zeta, fri_openings[0][0][0].clone())])];
        <FriPcs as Pcs<Challenge, Challenger>>::verify(
            &fri_pcs,
            vec![(fri_commit, fri_claims).into()],
            &fri_proof,
            &mut fri_v_ch,
        )
        .expect("binary FRI proof should verify");
        let fri_bytes =
            postcard::to_allocvec(&fri_proof).expect("binary FRI proof should serialize");

        let mut stir_p_ch = Challenger::new(perm.clone());
        let (stir_commit, stir_data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&stir_pcs, [(stir_domain, mat)]);
        observe_commitment(&mut stir_p_ch, &stir_commit);
        // STIR's commitment wraps one root per shared-domain group, so the transcript absorbs
        // a group count before the root and the point derived from it differs from FRI's over
        // the same matrix. Proof size does not depend on which point is opened, so the
        // comparison stays like-for-like.
        let stir_zeta: Challenge = stir_p_ch.sample_algebra_element();
        let (stir_openings, stir_proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &stir_pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &stir_data,
                points: vec![vec![stir_zeta]],
            }],
            &mut stir_p_ch,
        );

        let mut stir_v_ch = Challenger::new(perm);
        observe_commitment(&mut stir_v_ch, &stir_commit);
        let stir_v_zeta: Challenge = stir_v_ch.sample_algebra_element();
        assert_eq!(stir_v_zeta, stir_zeta);
        let stir_claims = vec![(
            stir_domain,
            vec![(stir_zeta, stir_openings[0][0][0].clone())],
        )];
        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &stir_pcs,
            vec![(stir_commit, stir_claims).into()],
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
    #[ignore = "full FRI+STIR prove/verify at two sizes; run from heavy CI"]
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
        observe_commitment(&mut p_ch, &commit_a);
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain_b, mat_b)]);
        observe_commitment(&mut p_ch, &commit_b);

        let zeta: Challenge = p_ch.sample_algebra_element();

        // Each commitment has one matrix, opened at the same `zeta`.
        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );

        // Verify.
        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit_a);
        observe_commitment(&mut v_ch, &commit_b);
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
            commitments_with_claims
                .into_iter()
                .map(Into::into)
                .collect(),
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
        observe_commitment(&mut p_ch, &commit_a);
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_b)]);
        observe_commitment(&mut p_ch, &commit_b);

        let zeta: Challenge = p_ch.sample_algebra_element();

        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit_a);
        observe_commitment(&mut v_ch, &commit_b);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();
        let claims = vec![
            (commit_a, vec![(domain, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain, vec![(zeta, opening_b)])]),
        ];
        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        )
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();

        // `mat_a` is opened at `zeta`; `mat_b` is opened at no points at all.
        let data_and_points = vec![(&data, vec![vec![zeta], vec![]])];
        let _ = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );
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
        observe_commitment(&mut p_ch, &commit);

        let data_and_points = vec![(&data, vec![vec![]])];
        let _ = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = prove_log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points,
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
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
            vec![(commit, claims).into()],
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
        observe_commitment(&mut p_ch, &commit_a);
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_b)]);
        observe_commitment(&mut p_ch, &commit_b);

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );

        // Drop the second commitment's input-opening vector. The verifier must reject:
        // skipping a commit's openings would let the proof verify against a proper subset
        // of the public input.
        for (_stir_proof, input_openings) in proof.buckets.iter_mut() {
            assert_eq!(input_openings.len(), 2);
            input_openings.pop();
        }

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit_a);
        observe_commitment(&mut v_ch, &commit_b);
        let _v_zeta: Challenge = v_ch.sample_algebra_element();

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();
        let claims = vec![
            (commit_a, vec![(domain, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain, vec![(zeta, opening_b)])]),
        ];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        );
        let err = res.expect_err("truncated input_openings must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::InputOpeningCount {
                log_height: log_d + 1,
                expected: 2,
                got: 1,
            }
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
        observe_commitment(&mut p_ch, &commit_a);
        let (commit_b, data_b) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat_b)]);
        observe_commitment(&mut p_ch, &commit_b);

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points = vec![(&data_a, vec![vec![zeta]]), (&data_b, vec![vec![zeta]])];
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );

        // Both commitments land in the same bucket, so both slots start as `Some`. Blank
        // out the first one in place, keeping the vector's length untouched.
        for (_stir_proof, input_openings) in proof.buckets.iter_mut() {
            assert_eq!(input_openings.len(), 2);
            assert!(input_openings[0].is_some());
            input_openings[0] = None;
        }

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit_a);
        observe_commitment(&mut v_ch, &commit_b);
        let _v_zeta: Challenge = v_ch.sample_algebra_element();

        let opening_a = opening_values[0][0][0].clone();
        let opening_b = opening_values[1][0][0].clone();
        let claims = vec![
            (commit_a, vec![(domain, vec![(zeta, opening_a)])]),
            (commit_b, vec![(domain, vec![(zeta, opening_b)])]),
        ];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        );
        let err = res.expect_err("a blanked input opening must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::MissingInputOpening {
                log_height: log_d + 1,
                commitment: 0,
            }
        );
    }

    /// Every bucket's STIR proof carries its own initial-oracle commitment and opens that
    /// commitment's fibers in round 0, while the input openings hold one LDE row per matrix
    /// per queried position.
    #[test]
    fn test_pcs_proof_commits_the_initial_oracle_and_opens_single_input_rows() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let width = 3;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, width);

        let mut p_ch = challenger_template;
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (_, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        assert_eq!(proof.buckets.len(), 1);
        let (stir_proof, input_openings) = &proof.buckets[0];
        assert!(stir_proof.initial_commitment.is_some());
        let round0 = stir_proof
            .round_proofs
            .first()
            .expect("log_d = 6 has intermediate rounds");
        let fibers = round0
            .query_openings
            .as_ref()
            .expect("round 0 opens the committed initial oracle");
        assert!(
            fibers
                .row_evals
                .iter()
                .all(|fiber| fiber.len() == 1 << LOG_STARTING_FOLDING_FACTOR)
        );

        let opening = input_openings[0]
            .as_ref()
            .expect("the commitment sits on this bucket");
        assert!(!opening.opened_values.is_empty());
        assert!(
            opening
                .opened_values
                .iter()
                .all(|per_query| per_query.len() == 1 && per_query[0].len() == width),
            "one row of the single committed matrix per queried position"
        );
    }

    /// The round-0 fibers are authenticated against STIR's own commitment: a changed lane
    /// value fails the Merkle check before any lane comparison runs.
    #[test]
    fn test_pcs_rejects_a_tampered_initial_oracle_fiber() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        proof.buckets[0].0.round_proofs[0]
            .query_openings
            .as_mut()
            .expect("round 0 opens the committed initial oracle")
            .row_evals[0][0] += Challenge::ONE;

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
        let _v_zeta: Challenge = v_ch.sample_algebra_element();
        let claims = vec![(
            commit,
            vec![(domain, vec![(zeta, opening_values[0][0][0].clone())])],
        )];
        let err = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        )
        .expect_err("a tampered initial-oracle fiber must be rejected");
        assert!(
            matches!(
                err,
                StirError::InvalidMmcsProof {
                    round: RoundLabel::Round(0),
                    ..
                }
            ),
            "expected the round-0 Merkle check to fail, got {err:?}"
        );
    }

    /// The single input row per query is authenticated against the input commitment.
    #[test]
    fn test_pcs_rejects_a_tampered_input_row() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        let log_d = 6;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        proof.buckets[0].1[0]
            .as_mut()
            .expect("the commitment sits on this bucket")
            .opened_values[0][0][0] += Val::ONE;

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
        let _v_zeta: Challenge = v_ch.sample_algebra_element();
        let claims = vec![(
            commit,
            vec![(domain, vec![(zeta, opening_values[0][0][0].clone())])],
        )];
        let err = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        )
        .expect_err("a tampered input row must be rejected");
        assert!(
            matches!(err, StirError::InputError(_)),
            "expected the input Merkle check to fail, got {err:?}"
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        for (_stir_proof, input_openings) in proof.buckets.iter_mut() {
            let opening = input_openings[0]
                .as_mut()
                .expect("single commitment must have a present opening");
            assert!(!opening.opened_values.is_empty());
            opening.opened_values.pop();
        }

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let opening = opening_values[0][0][0].clone();
        let claims = vec![(commit, vec![(domain, vec![(zeta, opening)])])];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        );
        let err = res.expect_err("truncated opened_values must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::InputOpenedRowCount {
                log_height: log_d + 1,
                commitment: 0,
                expected: 20,
                got: 19,
            }
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();

        let points: Vec<Vec<Challenge>> = prove_log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points,
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
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
            vec![(commit, claims).into()],
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
        // Two classes rather than one give a different `combine_key`, hence a different
        // `StirConfig`, hence a different first-round query count. STIR runs before the lane
        // check, so the disagreement surfaces on its own round-0 openings.
        assert_eq!(
            shape_of(err),
            ProofShapeError::QueryOpeningCount {
                round: RoundLabel::Round(0),
                expected: 22,
                got: 21,
            }
        );
    }

    #[test]
    fn test_pcs_verify_rejects_overstated_native_height() {
        // The mirror image: the prover ran `Combine` over two classes, the verifier is told
        // there is only one and skips it entirely.
        let err = verify_with_claimed_degrees(&[8, 6], &[8, 8])
            .expect_err("an overstated native height must be rejected");
        // One class rather than two gives a smaller first-round query count.
        //
        // So the point set an answer polynomial may interpolate is smaller too.
        //
        //     verifier accepts at most  2 OOD points + 21 queries = 23 coefficients
        //     the proof carries         2 OOD points + 22 queries = 24 coefficients
        //
        // Two of STIR's shape checks can catch this, and both are correct.
        //
        //     query openings     one opening per distinct queried position
        //     answer polynomial  one coefficient per interpolated point
        //
        // Which one trips first depends on how many of the drawn positions coincided.
        //
        // That is a property of the challenge stream, not of the mutation.
        //
        // So the assertion pins what the mutation controls, not which check won the race.
        //
        // In both readings the proof carries exactly one item too many, at round 0.
        match shape_of(err) {
            ProofShapeError::QueryOpeningCount {
                round,
                expected,
                got,
            }
            | ProofShapeError::AnsPolynomialTooLong {
                round,
                maximum: expected,
                got,
            } => {
                assert_eq!(round, RoundLabel::Round(0));
                assert_eq!(got, expected + 1);
            }
            other => panic!("expected a round-0 shape rejection, got {other:?}"),
        }
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points,
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
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
            vec![(commit, claims).into()],
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let points: Vec<Vec<Challenge>> = log_degrees.iter().map(|_| vec![zeta]).collect();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points,
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        let claims: Vec<_> = domains_and_polys
            .iter()
            .zip(opening_values.first().unwrap().iter())
            .map(|((domain, _), mat_openings)| (*domain, vec![(zeta, mat_openings[0].clone())]))
            .collect();

        <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            vec![(commit, claims).into()],
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
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        let mut v_ch = challenger_template;
        observe_commitment(&mut v_ch, &commit);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        assert_eq!(v_zeta, zeta);

        // Tamper one coordinate of the claimed evaluation. The reduced opening the verifier
        // reconstructs then diverges from the codeword STIR actually folded, so the round
        // consistency checks must reject.
        let mut tampered = opening_values[0][0][0].clone();
        tampered[0] += Challenge::from(Val::ONE);

        let claims = vec![(commit, vec![(domain, vec![(zeta, tampered)])])];
        let res = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        );
        assert!(
            res.is_err(),
            "PCS verify must reject a tampered claimed opening"
        );
    }

    /// An opening point on the LDE coset makes a quotient denominator vanish.
    ///
    /// `batch_multiplicative_inverse` panics on a zero input, so the verifier must reject
    /// first. Matches `FriError::OpeningPointMatchesQueryPoint`.
    #[test]
    fn test_pcs_rejects_opening_point_on_the_evaluation_domain() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        // A degree-4 matrix leaves a 2^3 LDE coset folded into two fibers of four lanes, so
        // the first-round queries cover every coset position.
        let log_d = 2;
        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 1);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        p_ch.observe(commit.clone());
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        // Claim the same value at the coset's first point instead. `open` cannot be asked
        // for one: it would divide by zero building its own denominators.
        let coset_point = Challenge::from(Val::GENERATOR);

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let _v_zeta: Challenge = v_ch.sample_algebra_element();
        let claims = vec![(
            commit,
            vec![(domain, vec![(coset_point, opening_values[0][0][0].clone())])],
        )];
        let err = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        )
        .expect_err("an opening point on the evaluation domain must be rejected");
        assert!(
            matches!(
                err,
                StirError::OpeningPointInDomain {
                    commitment: 0,
                    matrix: 0,
                    point: 0,
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn test_pcs_rejects_dropped_height_bucket() {
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
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        // The claims pin one STIR instance per distinct shared LDE height, so a proof with
        // fewer must be rejected before the transcript is touched.
        assert_eq!(proof.buckets.len(), 1);
        proof.buckets.pop();

        let mut v_ch = challenger_template;
        v_ch.observe(commit.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        let claims = vec![(
            commit,
            vec![(domain, vec![(v_zeta, opening_values[0][0][0].clone())])],
        )];
        let err = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        )
        .expect_err("a missing height bucket must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::BucketCount {
                expected: 1,
                got: 0,
            }
        );
    }

    /// An input opening at a bucket the commitment has no matrices at must be rejected.
    ///
    /// This is the mirror of [`test_pcs_input_opening_present_to_none_rejected`]: the slot
    /// exists and is occupied, but by rows from a different LDE domain, so folding them into
    /// this bucket's reduced opening would bind the wrong codeword.
    #[test]
    fn test_pcs_rejects_input_opening_at_wrong_bucket() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();

        // Two commitments two octaves apart, so each gets its own shared LDE height — hence
        // its own bucket, with the taller one first.
        let domain_tall =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << 8);
        let domain_short =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << 6);
        let mat_tall = RowMajorMatrix::<Val>::rand(&mut rng, 1 << 8, 3);
        let mat_short = RowMajorMatrix::<Val>::rand(&mut rng, 1 << 6, 3);

        let mut p_ch = challenger_template.clone();
        let (commit_tall, data_tall) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain_tall, mat_tall)]);
        p_ch.observe(commit_tall.clone());
        let (commit_short, data_short) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain_short, mat_short)]);
        p_ch.observe(commit_short.clone());

        let zeta: Challenge = p_ch.sample_algebra_element();
        let data_and_points = vec![
            (&data_tall, vec![vec![zeta]]),
            (&data_short, vec![vec![zeta]]),
        ];
        let (opening_values, mut proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            data_and_points.into_iter().map(Into::into).collect(),
            &mut p_ch,
        );
        assert_eq!(proof.buckets.len(), 2, "two heights must give two buckets");

        // Copy the short commitment's own opening into its (rightly empty) slot at the tall
        // bucket.
        let short_opening = proof.buckets[1].1[1].clone();
        assert!(short_opening.is_some());
        assert!(proof.buckets[0].1[1].is_none());
        proof.buckets[0].1[1] = short_opening;

        let mut v_ch = challenger_template;
        v_ch.observe(commit_tall.clone());
        v_ch.observe(commit_short.clone());
        let v_zeta: Challenge = v_ch.sample_algebra_element();

        let claims = vec![
            (
                commit_tall,
                vec![(domain_tall, vec![(v_zeta, opening_values[0][0][0].clone())])],
            ),
            (
                commit_short,
                vec![(
                    domain_short,
                    vec![(v_zeta, opening_values[1][0][0].clone())],
                )],
            ),
        ];
        let err = <MyPcs as Pcs<Challenge, Challenger>>::verify(
            &pcs,
            claims.into_iter().map(Into::into).collect(),
            &proof,
            &mut v_ch,
        )
        .expect_err("an opening at the wrong bucket must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::UnexpectedInputOpening {
                log_height: 9,
                commitment: 1,
            }
        );
    }

    #[test]
    fn test_pcs_rejects_a_missing_or_swapped_initial_commitment() {
        let (pcs, challenger_template) = get_pcs();
        let mut rng = seeded_rng();
        let log_d = 6;

        let domain =
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << log_d);
        let mat = RowMajorMatrix::<Val>::rand(&mut rng, 1 << log_d, 3);

        let mut p_ch = challenger_template.clone();
        let (commit, data) =
            <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, mat)]);
        observe_commitment(&mut p_ch, &commit);
        let zeta: Challenge = p_ch.sample_algebra_element();
        let (opening_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
            &pcs,
            vec![p3_commit::OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        );

        let verify_with = |proof: &<MyPcs as Pcs<Challenge, Challenger>>::Proof| {
            let mut v_ch = challenger_template.clone();
            observe_commitment(&mut v_ch, &commit);
            let v_zeta: Challenge = v_ch.sample_algebra_element();
            let claims = vec![(
                commit.clone(),
                vec![(domain, vec![(v_zeta, opening_values[0][0][0].clone())])],
            )];
            <MyPcs as Pcs<Challenge, Challenger>>::verify(
                &pcs,
                claims.into_iter().map(Into::into).collect(),
                proof,
                &mut v_ch,
            )
        };

        // STIR commits the initial oracle itself, and that commitment is what the round-0
        // fibers — and hence the lane checks — are authenticated against. Dropping it leaves
        // the transcript a message short.
        let mut dropped = proof.clone();
        dropped.buckets[0].0.initial_commitment = None;
        let err = verify_with(&dropped).expect_err("a missing initial commitment is malformed");
        assert_eq!(shape_of(err), ProofShapeError::MissingInitialCommitment);

        // Swapping it for another root the proof already carries must not authenticate the
        // round-0 openings.
        let mut swapped = proof;
        swapped.buckets[0].0.initial_commitment =
            Some(swapped.buckets[0].0.round_proofs[0].commitment.clone());
        let err = verify_with(&swapped).expect_err("a swapped initial commitment must be rejected");
        assert!(
            matches!(err, StirError::InvalidMmcsProof { round, .. } if round == RoundLabel::Round(0)),
            "expected a round-0 MMCS failure, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-instance lockstep driver: grind sharing across STIR height buckets.
// ---------------------------------------------------------------------------

mod babybear_stir_multi {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_stir::prover::{prove_stir, prove_stir_multi, prove_stir_multi_from_external_codewords};
    use p3_stir::verifier::{verify_stir_multi, verify_stir_multi_with_external_initial};

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

    /// The round-0 draws the prover reports are exactly the ones the verifier samples, one
    /// per configured query, and both sides' unique lists are their sorted deduplication.
    #[test]
    fn test_multi_first_round_draws_agree_between_prover_and_verifier() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        let log_degrees = [8usize, 6];
        let (configs, polys) = make_instances(&params, &log_degrees);
        let config_refs: Vec<&StirConfig<F, EF, MyMmcs, Challenger>> = configs.iter().collect();

        let mut p_ch = challenger.clone();
        let results = prove_stir_multi(&config_refs, polys, &dft, &mut p_ch);
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();

        let mut v_ch = challenger;
        let outputs =
            verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proofs, &mut v_ch)
                .expect("honest multi-instance proof verifies");

        for ((config, (_, first_round)), output) in configs.iter().zip(&results).zip(&outputs) {
            let expected_draws = if config.num_rounds() == 0 {
                config.final_queries
            } else {
                config.round_configs[0].num_queries
            };
            assert_eq!(first_round.draws.len(), expected_draws);
            assert_eq!(first_round.draws, output.first_round_draws);
            assert_eq!(sorted_dedup(&first_round.draws), first_round.unique_sorted);
            assert_eq!(output.first_round_indices, first_round.unique_sorted);
        }
    }

    #[test]
    fn a_batch_mixing_ground_and_ungrounded_instances_still_verifies() {
        // A shared grind runs at the largest difficulty among the instances active at its site.
        //
        //     instance A  max_pow_bits 0   -> derives 0 bits
        //     instance B  max_pow_bits 12  -> derives > 0 bits
        //     shared site                  -> max(0, >0) = > 0 bits
        //
        // So instance A legitimately carries a nonzero witness for a site it asked no work at.
        // Pinning a witness against its own instance's bits would reject this honest prover.
        // The canonical check reads the shared shape instead, and skips a site the batch grinds.
        let (idle_params, dft, challenger) = make_params(1, 2, 32, 0);
        let (ground_params, _, _) = make_params(1, 2, 32, 12);

        let log_degree = 8usize;
        let mut rng = seeded_rng();
        let configs = [&idle_params, &ground_params]
            .map(|params| StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params.clone()))
            .to_vec();

        // The grinding instance must actually grind, or the batch proves nothing.
        assert!(
            configs[1].round_configs.iter().any(|rc| rc.pow_bits > 0)
                || configs[1].final_pow_bits > 0,
            "the second instance must derive a positive difficulty for this batch to be mixed",
        );
        assert!(
            configs[0].round_configs.iter().all(|rc| rc.pow_bits == 0)
                && configs[0].final_pow_bits == 0,
            "the first instance must derive no difficulty for this batch to be mixed",
        );

        let polys: Vec<Vec<EF>> = (0..2)
            .map(|_| (0..1usize << log_degree).map(|_| rng.random()).collect())
            .collect();
        let config_refs: Vec<&StirConfig<F, EF, MyMmcs, Challenger>> = configs.iter().collect();

        let mut p_ch = challenger.clone();
        let results = prove_stir_multi(&config_refs, polys, &dft, &mut p_ch);
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();

        // Completeness: the honest mixed batch verifies, canonical check and all.
        let mut v_ch = challenger;
        verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proofs, &mut v_ch)
            .expect("an honest batch mixing ground and ungrounded instances must verify");
    }

    #[test]
    fn test_multi_noncanonical_replicated_pow_witnesses_are_rejected() {
        // Why: `check_replicated_witnesses` compares the copies of a shared site only
        // against each other, and at zero difficulty nothing else reads them at all.
        //
        //     agree, both wrong -> the agreement check passes  -> unbound
        //     pinned to zero    -> every rewrite is caught     -> bound
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        let log_degrees = [8usize, 6];
        let (configs, polys) = make_instances(&params, &log_degrees);
        let config_refs: Vec<&StirConfig<F, EF, MyMmcs, Challenger>> = configs.iter().collect();

        let mut p_ch = challenger.clone();
        let results = prove_stir_multi(&config_refs, polys, &dft, &mut p_ch);
        let mut proofs: Vec<_> = results.into_iter().map(|(proof, _)| proof).collect();

        // Every site asks for no work, so the honest prover wrote zero into all of them.
        for proof in &proofs {
            assert!(
                proof
                    .round_proofs
                    .iter()
                    .all(|rp| rp.folding_pow_witness == F::ZERO && rp.pow_witness == F::ZERO)
            );
            assert_eq!(proof.final_folding_pow_witness, F::ZERO);
            assert_eq!(proof.final_pow_witness, F::ZERO);
        }

        // Rewrite every replicated copy to one shared wrong value, so the instances still
        // agree and the replication check has nothing to catch.
        for proof in &mut proofs {
            for rp in &mut proof.round_proofs {
                rp.folding_pow_witness = F::ONE;
                rp.pow_witness = F::ONE;
            }
            proof.final_folding_pow_witness = F::ONE;
            proof.final_pow_witness = F::ONE;
        }

        let proof_refs: Vec<_> = proofs.iter().collect();
        let mut v_ch = challenger;
        let err =
            verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proof_refs, &mut v_ch)
                .expect_err("rewritten replicated witnesses must be rejected");
        assert!(
            matches!(
                err,
                StirError::InvalidProofShape(ProofShapeError::NonCanonicalPowWitness { .. })
            ),
            "expected NonCanonicalPowWitness, got {err:?}"
        );
    }

    /// The result an external fiber source returns.
    type FiberResult = Result<Vec<Vec<EF>>, StirError<<MyMmcs as Mmcs<EF>>::Error>>;

    /// Fiber source for one instance's external codeword, honest for the indices STIR queries
    /// in the round that reads it: lane `l` of query `j` sits at natural-order position
    /// `j + l * fold_height`, mirroring `verify_external_initial`'s single-instance source.
    fn external_fiber_source(
        codeword: Vec<EF>,
        arity: usize,
        fold_height: usize,
    ) -> impl FnOnce(&[usize]) -> FiberResult {
        move |js: &[usize]| {
            Ok(js
                .iter()
                .map(|&j| (0..arity).map(|l| codeword[j + l * fold_height]).collect())
                .collect())
        }
    }

    #[test]
    fn early_stop_mixed_schedules_verify_with_committed_and_external_oracles() {
        let (mut params, dft, challenger) = make_params(1, 2, 16, 0);
        params.log_starting_folding_factor = 3;
        let configs: Vec<_> = [(8, Some(4)), (6, Some(usize::MAX))]
            .into_iter()
            .map(|(degree, cap)| {
                StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
                    degree,
                    params.clone(),
                    StirOptions {
                        max_log_final_poly_len: cap,
                        ..Default::default()
                    },
                )
            })
            .collect();
        let config_refs: Vec<_> = configs.iter().collect();
        assert_eq!(
            configs.iter().map(|c| c.num_rounds()).collect::<Vec<_>>(),
            vec![1, 0]
        );
        let mut rng = seeded_rng();
        let polys: Vec<Vec<EF>> = [8, 6]
            .map(|degree| (0..1 << degree).map(|_| rng.random()).collect())
            .into();
        let mut p_ch = challenger.clone();
        let results = prove_stir_multi(&config_refs, polys.clone(), &dft, &mut p_ch);
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();
        let mut v_ch = challenger.clone();
        verify_stir_multi(&config_refs, &proofs, &mut v_ch).unwrap();
        assert_eq!(
            p_ch.sample_algebra_element::<EF>(),
            v_ch.sample_algebra_element::<EF>()
        );

        let codewords: Vec<_> = configs
            .iter()
            .zip(polys)
            .map(|(config, poly)| {
                codeword_from_coeffs(&dft, poly, F::GENERATOR, config.log_starting_domain_size())
            })
            .collect();
        let mut base = challenger;
        for codeword in &codewords {
            base.observe_algebra_slice(codeword);
        }
        let mut p_ch = base.clone();
        let results = prove_stir_multi_from_external_codewords(
            &config_refs,
            codewords.clone(),
            &dft,
            &mut p_ch,
        );
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();
        let sources: Vec<_> = configs
            .iter()
            .zip(codewords)
            .map(|(config, codeword)| {
                let arity = 1 << config.log_starting_folding_factor;
                let height = codeword.len() / arity;
                external_fiber_source(codeword, arity, height)
            })
            .collect();
        let outputs = verify_stir_multi_with_external_initial::<F, EF, MyMmcs, Challenger, (), _>(
            &config_refs,
            &proofs,
            &mut base,
            sources,
        )
        .unwrap();
        assert_eq!(
            p_ch.sample_algebra_element::<EF>(),
            base.sample_algebra_element::<EF>()
        );
        for ((_, first), output) in results.iter().zip(outputs) {
            assert_eq!(first.draws, output.first_round_draws);
            assert_eq!(first.unique_sorted, output.first_round_indices);
        }
    }

    #[test]
    fn compact_answers_support_mixed_configs_and_external_oracles() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        let full_configs: Vec<_> = [(10, None), (8, Some(4)), (6, Some(usize::MAX))]
            .into_iter()
            .map(|(degree, cap)| {
                StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
                    degree,
                    params.clone(),
                    StirOptions {
                        max_log_final_poly_len: cap,
                        ..Default::default()
                    },
                )
            })
            .collect();
        let configs: Vec<_> = full_configs
            .iter()
            .zip([true, false, true])
            .map(|(full, compact_answers)| {
                StirConfig::<F, EF, MyMmcs, Challenger>::new_with_options(
                    full.log_starting_degree,
                    params.clone(),
                    StirOptions {
                        compact_answers,
                        ..full.options()
                    },
                )
            })
            .collect();
        let full_refs: Vec<_> = full_configs.iter().collect();
        let config_refs: Vec<_> = configs.iter().collect();
        let mut rng = seeded_rng();
        let polys: Vec<Vec<EF>> = configs
            .iter()
            .map(|config| {
                (0..1 << config.log_starting_degree)
                    .map(|_| rng.random())
                    .collect()
            })
            .collect();
        let codewords: Vec<_> = configs
            .iter()
            .zip(&polys)
            .map(|(config, poly)| {
                codeword_from_coeffs(
                    &dft,
                    poly.clone(),
                    F::GENERATOR,
                    config.log_starting_domain_size(),
                )
            })
            .collect();

        for external in [false, true] {
            let mut base = challenger.clone();
            if external {
                for codeword in &codewords {
                    base.observe_algebra_slice(codeword);
                }
            }
            let prove = |refs: &[&StirConfig<F, EF, MyMmcs, Challenger>], ch: &mut Challenger| {
                if external {
                    prove_stir_multi_from_external_codewords(refs, codewords.clone(), &dft, ch)
                } else {
                    prove_stir_multi(refs, polys.clone(), &dft, ch)
                }
            };
            let verify = |refs: &[&StirConfig<F, EF, MyMmcs, Challenger>],
                          proofs: &[&StirProof<EF, MyMmcs, F>],
                          ch: &mut Challenger| {
                if external {
                    let sources: Vec<_> = configs
                        .iter()
                        .zip(&codewords)
                        .map(|(config, codeword)| {
                            let arity = 1 << config.log_starting_folding_factor;
                            external_fiber_source(codeword.clone(), arity, codeword.len() / arity)
                        })
                        .collect();
                    verify_stir_multi_with_external_initial::<F, EF, MyMmcs, Challenger, (), _>(
                        refs, proofs, ch, sources,
                    )
                } else {
                    verify_stir_multi(refs, proofs, ch)
                }
            };
            let mut full_p = base.clone();
            let mut compact_p = base.clone();
            let full = prove(&full_refs, &mut full_p);
            let mixed = prove(&config_refs, &mut compact_p);
            let mut full_v = base.clone();
            let mut compact_v = base.clone();
            let full_outputs = verify(
                &full_refs,
                &full.iter().map(|(p, _)| p).collect::<Vec<_>>(),
                &mut full_v,
            )
            .unwrap();
            let mixed_outputs = verify(
                &config_refs,
                &mixed.iter().map(|(p, _)| p).collect::<Vec<_>>(),
                &mut compact_v,
            )
            .unwrap();
            let next: EF = full_p.sample_algebra_element();
            assert_eq!(next, compact_p.sample_algebra_element::<EF>());
            assert_eq!(next, full_v.sample_algebra_element::<EF>());
            assert_eq!(next, compact_v.sample_algebra_element::<EF>());
            for (i, ((full_proof, full_queries), (mixed_proof, mixed_queries))) in
                full.iter().zip(&mixed).enumerate()
            {
                assert_eq!(full_queries, mixed_queries);
                assert_eq!(
                    full_outputs[i].first_round_draws,
                    mixed_outputs[i].first_round_draws
                );
                assert_eq!(mixed_queries.draws, mixed_outputs[i].first_round_draws);
                let mut expected = full_proof.clone();
                if configs[i].options().compact_answers {
                    for round in &mut expected.round_proofs {
                        round.ans_polynomial.clear();
                    }
                }
                assert_eq!(
                    postcard::to_allocvec(&expected).unwrap(),
                    postcard::to_allocvec(mixed_proof).unwrap()
                );
            }
            let mut bad: Vec<_> = mixed.into_iter().map(|(proof, _)| proof).collect();
            bad[0].round_proofs[1].ans_polynomial = vec![EF::ONE];
            let err = verify(&config_refs, &bad.iter().collect::<Vec<_>>(), &mut base).unwrap_err();
            assert_eq!(
                shape_of(err),
                ProofShapeError::UnexpectedAnsPolynomial {
                    round: RoundLabel::Round(1),
                    got: 1
                }
            );
        }
    }

    /// Prove and verify two different-degree instances whose initial oracles are external,
    /// exercising `verify_stir_multi_inner`'s per-instance external-oracle wiring: distinct
    /// `external_fibers[i]` sources, `is_external` flags, and right-aligned instance offsets,
    /// none of which any other test drives with `initial_is_external == true`.
    #[test]
    fn test_multi_external_initial_oracle_verifies() {
        let (params, dft, challenger) = make_params(1, 2, 16, 0);
        let log_degrees = [8usize, 6];
        let (configs, polys) = make_instances(&params, &log_degrees);
        let config_refs: Vec<&StirConfig<F, EF, MyMmcs, Challenger>> = configs.iter().collect();

        let codewords: Vec<Vec<EF>> = configs
            .iter()
            .zip(polys)
            .map(|(config, coeffs)| {
                codeword_from_coeffs(
                    &dft,
                    coeffs,
                    F::GENERATOR,
                    config.log_starting_domain_size(),
                )
            })
            .collect();

        // Binding each codeword before proving is the caller's job. Observing its values
        // stands in for the PCS layer's input commitments, as in `verify_external_initial`.
        let mut p_ch = challenger.clone();
        for codeword in &codewords {
            p_ch.observe_algebra_slice(codeword);
        }
        let results = prove_stir_multi_from_external_codewords(
            &config_refs,
            codewords.clone(),
            &dft,
            &mut p_ch,
        );
        assert_eq!(results.len(), log_degrees.len());
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();

        let mut v_ch = challenger;
        for codeword in &codewords {
            v_ch.observe_algebra_slice(codeword);
        }

        let initial_fibers: Vec<_> = configs
            .iter()
            .zip(&codewords)
            .map(|(config, codeword)| {
                let arity = 1usize << config.log_starting_folding_factor;
                let fold_height = (1usize << config.log_starting_domain_size()) / arity;
                external_fiber_source(codeword.clone(), arity, fold_height)
            })
            .collect();

        let outputs = verify_stir_multi_with_external_initial::<F, EF, MyMmcs, Challenger, (), _>(
            &config_refs,
            &proofs,
            &mut v_ch,
            initial_fibers,
        )
        .expect("honest multi-instance external-oracle proof verifies");

        for (((config, codeword), (_, first_round)), output) in
            configs.iter().zip(&codewords).zip(&results).zip(&outputs)
        {
            let arity = 1usize << config.log_starting_folding_factor;
            let fold_height = (1usize << config.log_starting_domain_size()) / arity;

            assert_eq!(sorted_dedup(&first_round.draws), first_round.unique_sorted);
            assert_eq!(output.first_round_indices, first_round.unique_sorted);

            for (&j, evals) in output
                .first_round_indices
                .iter()
                .zip(&output.first_round_fiber_evals)
            {
                let expected: Vec<EF> = (0..arity).map(|l| codeword[j + l * fold_height]).collect();
                assert_eq!(evals, &expected);
            }
        }
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
    fn test_multi_shared_folding_pow_with_late_and_final_only_instances() {
        let (params, dft, challenger) = make_params(1, 2, 100, 16);
        let (configs, polys) = make_instances(&params, &[10, 6, 2]);
        assert!(configs[0].num_rounds() > configs[1].num_rounds());
        assert!(configs[1].num_rounds() > 0);
        assert_eq!(configs[2].num_rounds(), 0);
        assert!(configs[1].round_configs[0].folding_pow_bits > 0);
        assert!(configs[2].final_folding_pow_bits > 0);
        let config_refs: Vec<_> = configs.iter().collect();

        let mut prover_challenger = challenger.clone();
        let results = prove_stir_multi(&config_refs, polys, &dft, &mut prover_challenger);
        let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();
        let mut verifier_challenger = challenger;
        let outputs = verify_stir_multi::<F, EF, MyMmcs, Challenger>(
            &config_refs,
            &proofs,
            &mut verifier_challenger,
        )
        .expect("shared folding grinds must cover every right-aligned instance");

        for ((_, queries), output) in results.iter().zip(outputs) {
            assert_eq!(queries.draws, output.first_round_draws);
        }
        assert_eq!(
            prover_challenger.sample_algebra_element::<EF>(),
            verifier_challenger.sample_algebra_element::<EF>()
        );
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

        assert_eq!(single_idx, multi_idx.unique_sorted);
        assert_eq!(sorted_dedup(&multi_idx.draws), multi_idx.unique_sorted);
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
        assert!(
            matches!(
                err,
                StirError::InvalidPowWitness { round, .. }
                    if round == RoundLabel::Round(round_with_pow)
            ),
            "{err:?}"
        );
    }

    /// Prove two identical-height instances, apply `mutate` to the second, and verify.
    ///
    /// Equal heights line the round indices up 1:1, so every grind site is shared exactly.
    fn multi_shape_error_after(
        mutate: impl FnOnce(&mut StirProof<EF, MyMmcs, F>),
    ) -> ProofShapeError {
        let (params, dft, challenger) = make_params(1, 2, 32, 12);
        let log_degree = 8;
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
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
        mutate(&mut proof_b);

        let mut v_ch = challenger;
        let proofs = [&proof_a, &proof_b];
        let err = verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &proofs, &mut v_ch)
            .expect_err("a shape-mutated batch must be rejected");
        shape_of(err)
    }

    #[test]
    fn test_multi_disagreeing_folding_witness_rejected() {
        let err = multi_shape_error_after(|proof| {
            proof.round_proofs[0].folding_pow_witness += F::ONE;
        });
        assert_eq!(
            err,
            ProofShapeError::ReplicatedWitnessMismatch {
                round: RoundLabel::Round(0),
                stage: GrindStage::Folding,
            }
        );
    }

    /// The query grind sits after the OOD absorb, so its witness is checked separately.
    #[test]
    fn test_multi_disagreeing_query_witness_rejected() {
        let err = multi_shape_error_after(|proof| {
            proof.round_proofs[0].pow_witness += F::ONE;
        });
        assert_eq!(
            err,
            ProofShapeError::ReplicatedWitnessMismatch {
                round: RoundLabel::Round(0),
                stage: GrindStage::Query,
            }
        );
    }

    /// A batched round-count mismatch names the instance that carries it.
    #[test]
    fn test_multi_round_count_names_its_instance() {
        let err = multi_shape_error_after(|proof| {
            proof.round_proofs.pop();
        });
        assert_eq!(
            err,
            ProofShapeError::RoundCount {
                instance: Some(1),
                expected: 3,
                got: 2,
            }
        );
    }

    #[test]
    fn test_multi_proof_count_mismatch_rejected() {
        let (params, dft, challenger) = make_params(1, 2, 32, 12);
        let log_degree = 8;
        let config = StirConfig::<F, EF, MyMmcs, Challenger>::new(log_degree, params);
        let config_refs = [&config, &config];

        let mut rng = seeded_rng();
        let poly: Vec<EF> = (0..1usize << log_degree).map(|_| rng.random()).collect();

        let mut p_ch = challenger.clone();
        let results = prove_stir_multi(&[&config], vec![poly], &dft, &mut p_ch);
        let proof = &results[0].0;

        let mut v_ch = challenger;
        let err = verify_stir_multi::<F, EF, MyMmcs, Challenger>(&config_refs, &[proof], &mut v_ch)
            .expect_err("one proof for two configs must be rejected");
        assert_eq!(
            shape_of(err),
            ProofShapeError::InstanceCount {
                expected: 2,
                got: 1,
            }
        );
    }

    /// An empty batch has no transcript operations, matching the prover, so it verifies.
    #[test]
    fn test_multi_empty_batch_verifies() {
        let (_params, _dft, challenger) = make_params(1, 2, 32, 12);
        let mut v_ch = challenger;
        let outputs = verify_stir_multi::<F, EF, MyMmcs, Challenger>(&[], &[], &mut v_ch)
            .expect("an empty batch must verify");
        assert!(outputs.is_empty());
    }
}

// ---------------------------------------------------------------------------
// `TwoAdicStirPcs` as a `StarkGenericConfig`'s PCS
// ---------------------------------------------------------------------------

mod uni_stark_with_stir_pcs {
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_stir::TwoAdicStirPcs;
    use p3_uni_stark::{StarkConfig, prove, verify};

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
    type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

    /// Two columns holding `(i, i + 1)`: `b = a + 1` on every row, and `a` advances to the
    /// previous row's `b` on every transition.
    struct StepAir;

    impl<F> BaseAir<F> for StepAir {
        fn width(&self) -> usize {
            2
        }

        fn max_constraint_degree(&self) -> Option<usize> {
            Some(2)
        }
    }

    impl<AB: AirBuilder> Air<AB> for StepAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let (a, b, next_a) = {
                let local = main.current_slice();
                let next = main.next_slice();
                (local[0], local[1], next[0])
            };
            builder.assert_eq(b, a + AB::Expr::ONE);
            builder.when_transition().assert_eq(next_a, b);
        }
    }

    fn step_trace(log_n: usize) -> RowMajorMatrix<Val> {
        RowMajorMatrix::new(
            (0..(1u64 << log_n))
                .flat_map(|i| [Val::from_u64(i), Val::from_u64(i + 1)])
                .collect(),
            2,
        )
    }

    #[test]
    fn test_stir_pcs_drives_a_uni_stark_proof() {
        // `StarkGenericConfig` requires `Challenger: CanObserve<Pcs::Commitment>`, so a
        // commitment carrying one root per shared-domain group has to be observable as a
        // single value. Proving through `p3-uni-stark` is what checks that end to end: it
        // commits a trace and a quotient (through `commit_ldes`, at heights the layout may
        // well split differently), observes both commitments, and opens them together.
        // `p3-batch-stark` re-exports this same trait, so it is covered by the same bound.
        let perm = Perm::new_from_rng_128(&mut seeded_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress, 0);
        let stir_params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 2,
            log_starting_folding_factor: 2,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 16,
            max_pow_bits: 0,
            mmcs: ChallengeMmcs::new(val_mmcs.clone()),
        };
        let pcs = MyPcs::new(Dft::default(), val_mmcs, stir_params);
        let config = MyConfig::new(pcs, Challenger::new(perm));

        let proof = prove(&config, &StepAir, step_trace(5), &[]);
        verify(&config, &StepAir, &proof, &[]).expect("verification failed");
    }
}
