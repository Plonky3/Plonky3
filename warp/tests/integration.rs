//! End-to-end WARP tests over the direct Boolean PESAT relation.
//!
//! The relation has one quadratic constraint per witness coordinate:
//! `cell · (cell − 1) = 0`. So a satisfying witness is any vector where
//! every cell is in `{0, 1}`.
//!
//! Coverage:
//!
//! - **Happy path**: prove → verify → decide on a satisfying witness.
//! - **Multi-step accumulation**: produce `acc_1`, then accumulate again
//!   into `acc_2`, then run the decider on `acc_2`.
//! - **Soundness**: malicious-prover tampering tests where we mutate one
//!   field of the proof or the new accumulator and confirm that the
//!   verifier or decider rejects with a specific error.

use std::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::Mmcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_warp::error::{DeciderError, FinalizerError, VerifierError, WarpError};
use p3_warp::finalize::{Finalizer, LocalDeciderFinalizer, WitnessFinalizer};
use p3_warp::{
    BooleanPesat, CommittedCodeword, MmcsExternalOpeningVerifier, ReedSolomonCode, WarpDecider,
    WarpParams, WarpProof, WarpProver, WarpRootProver, WarpRootVerifier, WarpVerifier,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

// -----------------------------------------------------------------------------
// Plonky3 type setup.
// -----------------------------------------------------------------------------

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type MyComm = <MyMmcs as Mmcs<F>>::Commitment;
type MyMtProof = <MyMmcs as Mmcs<F>>::Proof;

const MAIN_WIDTH: usize = 4;
const LOG_HEIGHT: usize = 3;
const LOG_WITNESS: usize = 5;
const LOG_INV_RATE: usize = 1;
const NUM_FRESH: usize = 4;

fn make_components() -> (
    MyMmcs,
    MyChallenger,
    BooleanPesat<F, EF>,
    ReedSolomonCode<F, MyDft>,
) {
    let perm = make_permutation();
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm.clone()), 0);
    let dft = Radix2DFTSmallBatch::<F>::default();
    let challenger = MyChallenger::new(perm);
    let pesat = BooleanPesat::<F, EF>::new(LOG_WITNESS, b"BooleanPesat/v1".to_vec());
    // log_msg = log_main_width + log_height = 2 + 3 = 5; n = 2^(5+1) = 64.
    let code = ReedSolomonCode::<F, MyDft>::new_systematic(LOG_WITNESS, LOG_INV_RATE, dft);
    (mmcs, challenger, pesat, code)
}

fn make_permutation() -> Perm {
    default_babybear_poseidon2_16()
}

fn make_satisfying_witness(seed: u64) -> Vec<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let k = MAIN_WIDTH * (1 << LOG_HEIGHT);
    (0..k)
        .map(|_| {
            // Random Boolean from the LSB of a u64.
            let bit: u64 = rng.random::<u64>() & 1;
            F::from_u64(bit)
        })
        .collect()
}

fn make_params() -> WarpParams {
    // 1 + s + t must be a power of two. Use s = 1, t = 2 → r = 4.
    WarpParams::new(1, 2)
}

// -----------------------------------------------------------------------------
// 1. Honest happy-path tests.
// -----------------------------------------------------------------------------

#[test]
fn honest_first_step_accepts() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(100 + i as u64))
        .collect();

    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (acc, proof) = prover.prove(&mut p_ch, &witnesses, &[]);

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc.instance, &proof)
        .expect("honest verify should accept");

    let decider = WarpDecider::new(&mmcs, &code, &pesat);
    decider
        .decide(&acc.instance, &acc.witness)
        .expect("honest decide should accept");
}

#[test]
fn honest_two_step_accumulation_accepts() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let decider = WarpDecider::new(&mmcs, &code, &pesat);

    // Step 1: ℓ_1 = 4 fresh, ℓ_2 = 0.
    let witnesses1: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(200 + i as u64))
        .collect();
    let mut p_ch1 = base_challenger.clone();
    let (acc1, proof1) = prover.prove(&mut p_ch1, &witnesses1, &[]);
    let mut v_ch1 = base_challenger.clone();
    verifier
        .verify(&mut v_ch1, NUM_FRESH, &[], &acc1.instance, &proof1)
        .expect("step-1 verify");

    // Step 2: ℓ_1 = 3 fresh, ℓ_2 = 1 → ℓ = 4.
    let witnesses2: Vec<Vec<F>> = (0..3)
        .map(|i| make_satisfying_witness(300 + i as u64))
        .collect();
    let prior_inst1 = acc1.instance.clone();
    let priors = std::vec![acc1];
    let mut p_ch2 = base_challenger.clone();
    let (acc2, proof2) = prover.prove(&mut p_ch2, &witnesses2, &priors);
    let mut v_ch2 = base_challenger.clone();
    verifier
        .verify(
            &mut v_ch2,
            3,
            std::slice::from_ref(&prior_inst1),
            &acc2.instance,
            &proof2,
        )
        .expect("step-2 verify");

    decider
        .decide(&acc2.instance, &acc2.witness)
        .expect("step-2 decide");
}

// -----------------------------------------------------------------------------
// 2. Soundness tests.
// -----------------------------------------------------------------------------

/// Wrap the honest-step setup into a closure so each tampering test can
/// produce a fresh `(acc_instance, proof, base_challenger)` to mutate.
fn produce_proof() -> (
    MyMmcs,
    MyChallenger,
    BooleanPesat<F, EF>,
    ReedSolomonCode<F, MyDft>,
    WarpParams,
    p3_warp::AccumulatorInstance<EF, MyComm>,
    WarpProof<F, EF, MyComm, MyMtProof>,
) {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(100 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (acc, proof) = prover.prove(&mut p_ch, &witnesses, &[]);
    (
        mmcs,
        base_challenger,
        pesat,
        code,
        params,
        acc.instance,
        proof,
    )
}

#[test]
fn tampered_twin_constraint_round_poly_rejected() {
    let (mmcs, base_challenger, pesat, code, params, acc_instance, mut proof) = produce_proof();
    proof.twin_constraint_sumcheck.round_polys[0][1] += EF::ONE;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("tampered twin-constraint round polynomial must be rejected");
    assert!(
        matches!(
            err,
            VerifierError::SumcheckConsistency {
                phase: "twin-constraint",
                ..
            }
        ),
        "expected twin-constraint SumcheckConsistency, got {err:?}"
    );
}

#[test]
fn tampered_batching_round_poly_rejected() {
    let (mmcs, base_challenger, pesat, code, params, acc_instance, mut proof) = produce_proof();
    proof.batching_sumcheck.round_polys[0][0] += EF::TWO;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("tampered batching round polynomial must be rejected");
    assert!(
        matches!(
            err,
            VerifierError::SumcheckConsistency {
                phase: "multilinear-batching",
                ..
            }
        ),
        "expected multilinear-batching SumcheckConsistency, got {err:?}"
    );
}

#[test]
fn tampered_mu_final_rejected() {
    let (mmcs, base_challenger, pesat, code, params, mut acc_instance, mut proof) = produce_proof();
    let bad = proof.mu_final + EF::ONE;
    proof.mu_final = bad;
    acc_instance.mu = bad;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("tampered µ must be rejected by §8.2 final-claim check");
    assert!(
        matches!(err, VerifierError::MultilinearBatchingFinalClaim),
        "expected MultilinearBatchingFinalClaim, got {err:?}"
    );
}

#[test]
fn tampered_eta_rejected_by_twin_oracle_check() {
    let (mmcs, base_challenger, pesat, code, params, mut acc_instance, mut proof) = produce_proof();
    let bad = proof.eta + EF::ONE;
    proof.eta = bad;
    acc_instance.eta = bad;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("tampered η must be rejected by the §6.3 oracle check");
    assert!(
        matches!(err, VerifierError::TwinConstraintFinalClaim),
        "expected TwinConstraintFinalClaim, got {err:?}"
    );
}

#[test]
fn tampered_acc_alpha_rejected() {
    let (mmcs, base_challenger, pesat, code, params, mut acc_instance, proof) = produce_proof();
    acc_instance.alpha[0] += EF::ONE;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("acc.α mismatch must be rejected");
    assert!(
        matches!(err, VerifierError::AccumulatorMismatch { field } if field.contains("alpha")),
        "expected AccumulatorMismatch(alpha), got {err:?}"
    );
}

#[test]
fn tampered_acc_beta_rejected() {
    let (mmcs, base_challenger, pesat, code, params, mut acc_instance, proof) = produce_proof();
    acc_instance.beta[0] += EF::ONE;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("acc.β mismatch must be rejected");
    assert!(
        matches!(err, VerifierError::AccumulatorMismatch { field } if field.contains("beta")),
        "expected AccumulatorMismatch(beta), got {err:?}"
    );
}

#[test]
fn tampered_shift_query_answer_rejected() {
    let (mmcs, base_challenger, pesat, code, params, acc_instance, mut proof) = produce_proof();
    // Mutate one base-field value in a shift-query answer; the Merkle proof
    // for that index will fail to verify against rt_0.
    proof.fresh_shift_answers[0][0] += F::ONE;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify(&mut v_ch, NUM_FRESH, &[], &acc_instance, &proof)
        .expect_err("tampered shift answer must be rejected");
    assert!(
        matches!(err, VerifierError::MerkleProof { .. }),
        "expected MerkleProof error, got {err:?}"
    );
}

#[test]
fn malformed_prior_accumulator_openings_are_rejected_without_panic() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);

    let witnesses1: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(800 + i as u64))
        .collect();
    let mut p_ch1 = base_challenger.clone();
    let (acc1, _proof1) = prover.prove(&mut p_ch1, &witnesses1, &[]);

    let witnesses2: Vec<Vec<F>> = (0..3)
        .map(|i| make_satisfying_witness(900 + i as u64))
        .collect();
    let prior_inst1 = acc1.instance.clone();
    let priors = std::vec![acc1];
    let mut p_ch2 = base_challenger.clone();
    let (acc2, proof2) = prover.prove(&mut p_ch2, &witnesses2, &priors);

    let mut proof_with_short_answers = proof2.clone();
    proof_with_short_answers.acc_shift_answers[0].pop();
    let mut v_ch_answers = base_challenger.clone();
    let err = verifier
        .verify(
            &mut v_ch_answers,
            3,
            std::slice::from_ref(&prior_inst1),
            &acc2.instance,
            &proof_with_short_answers,
        )
        .expect_err("short prior-accumulator answer list must be rejected");
    assert!(
        matches!(
            err,
            VerifierError::ShiftQueryCount {
                expected: 2,
                got: 1
            }
        ),
        "expected ShiftQueryCount for short answers, got {err:?}"
    );

    let mut proof_with_short_paths = proof2;
    proof_with_short_paths.acc_merkle_proofs[0].pop();
    let mut v_ch_paths = base_challenger.clone();
    let err = verifier
        .verify(
            &mut v_ch_paths,
            3,
            std::slice::from_ref(&prior_inst1),
            &acc2.instance,
            &proof_with_short_paths,
        )
        .expect_err("short prior-accumulator Merkle path list must be rejected");
    assert!(
        matches!(
            err,
            VerifierError::ShiftQueryCount {
                expected: 2,
                got: 1
            }
        ),
        "expected ShiftQueryCount for short paths, got {err:?}"
    );
}

#[test]
fn tampered_witness_rejected_by_decider() {
    let (mmcs, _base_challenger, pesat, code, _params, acc_instance, _proof) = produce_proof();
    // Get the matching witness side via a fresh prover run.
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(100 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, _params);
    let (mmcs2, base_challenger2, pesat2, code2) = make_components();
    let mut p_ch = base_challenger2.clone();
    let (mut acc, _proof) = prover.prove(&mut p_ch, &witnesses, &[]);
    // Flip one EF coordinate of the merged witness vector.
    acc.witness.w[0] += EF::ONE;

    let decider = WarpDecider::new(&mmcs2, &code2, &pesat2);
    let err = decider
        .decide(&acc.instance, &acc.witness)
        .expect_err("tampered w must be rejected");
    assert!(
        matches!(
            err,
            DeciderError::BundledPesat | DeciderError::EncodingMismatch
        ),
        "expected BundledPesat or EncodingMismatch, got {err:?}"
    );
    let _ = acc_instance;
}

// -----------------------------------------------------------------------------
// 3. Finalizer trait — modular finalisation strategies.
// -----------------------------------------------------------------------------

/// Honest path: `LocalDeciderFinalizer` finalises locally, returns `Ok(())`,
/// and refuses remote verification (no transmissible proof).
#[test]
fn local_finalizer_finalizes_and_refuses_remote_verify() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(400 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (acc, _proof) = prover.prove(&mut p_ch, &witnesses, &[]);

    let finalizer = LocalDeciderFinalizer::new(&mmcs, &code, &pesat);

    // `finalize` accepts the honest accumulator.
    finalizer
        .finalize(&acc.instance, &acc.witness)
        .expect("local finalize on honest acc");

    // `verify` always fails: there's no transmissible proof.
    let err = finalizer
        .verify(&acc.instance, &())
        .expect_err("local finalizer cannot remote-verify");
    assert!(matches!(err, FinalizerError::NoTransmissibleProof));
}

/// Honest path: `WitnessFinalizer` produces a transmissible proof; a
/// fresh verifier (no access to the prover's `acc.w`) re-runs the four
/// decider checks and accepts.
#[test]
fn witness_finalizer_round_trips_honest_proof() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(500 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (acc, _proof) = prover.prove(&mut p_ch, &witnesses, &[]);

    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);
    let final_proof = finalizer
        .finalize(&acc.instance, &acc.witness)
        .expect("witness finalize on honest acc");

    finalizer
        .verify(&acc.instance, &final_proof)
        .expect("witness verify on honest proof");
}

/// Soundness: tampering with `f` in the transmissible proof flips one of
/// the decider's algebraic checks, and the verifier rejects.
#[test]
fn witness_finalizer_rejects_tampered_codeword() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(600 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (acc, _proof) = prover.prove(&mut p_ch, &witnesses, &[]);

    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);
    let mut final_proof = finalizer.finalize(&acc.instance, &acc.witness).unwrap();

    // Flip a single coordinate of f. The Merkle re-commit check fires
    // first (rt mismatch).
    final_proof.f[0] += EF::ONE;

    let err = finalizer
        .verify(&acc.instance, &final_proof)
        .expect_err("tampered f in transmissible proof must be rejected");
    assert!(
        matches!(
            err,
            FinalizerError::Decider(
                DeciderError::MerkleRoot | DeciderError::MlEval | DeciderError::EncodingMismatch,
            ),
        ),
        "expected Merkle/MlEval/Encoding error, got {err:?}"
    );
}

/// Soundness: tampering with `w` (but recomputing `f` to match) flips
/// the bundled-PESAT check `Pb(β, w) = η`.
#[test]
fn witness_finalizer_rejects_tampered_witness() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(700 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (acc, _proof) = prover.prove(&mut p_ch, &witnesses, &[]);

    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);
    let mut final_proof = finalizer.finalize(&acc.instance, &acc.witness).unwrap();

    // Flip a coordinate of w only; `f` no longer matches `C(w)`.
    final_proof.w[0] += EF::ONE;

    let err = finalizer
        .verify(&acc.instance, &final_proof)
        .expect_err("tampered w must be rejected");
    assert!(
        matches!(
            err,
            FinalizerError::Decider(DeciderError::EncodingMismatch | DeciderError::BundledPesat,),
        ),
        "expected EncodingMismatch / BundledPesat, got {err:?}"
    );
}

// -----------------------------------------------------------------------------
// 4. Root proof composition — all VACC steps + final DACC.
// -----------------------------------------------------------------------------

#[test]
fn witness_root_proof_verifies_chain_and_final_decider() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(1700 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(1800 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(1900 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (claimed_final, root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");

    let verified_final = root_verifier
        .verify_linear_chain(&base_challenger, &root_proof, &finalizer)
        .expect("root verify");
    let receipt = root_verifier
        .verify_linear_chain_with_receipt(&base_challenger, &root_proof, &finalizer)
        .expect("root verify with receipt");

    assert_eq!(verified_final.alpha, claimed_final.alpha);
    assert_eq!(verified_final.mu, claimed_final.mu);
    assert_eq!(verified_final.beta, claimed_final.beta);
    assert_eq!(verified_final.eta, claimed_final.eta);
    assert_eq!(receipt.final_instance, verified_final);

    let claim = root_proof.claim();
    assert_eq!(claim.step_num_fresh, vec![NUM_FRESH, 3, 3]);
    assert_eq!(claim.step_instances.len(), 3);
    assert_eq!(claim.final_instance().unwrap().mu, claimed_final.mu);
    assert_eq!(receipt.claim, claim);
    assert_eq!(
        receipt.claim_digest,
        claim.digest::<F, MyChallenger>(base_challenger.clone())
    );
}

#[test]
fn witness_root_proof_rejects_tampered_step_proof() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(2000 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2100 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (_claimed_final, mut root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");

    root_proof.steps[1].proof.eta += EF::ONE;

    let err = root_verifier
        .verify_linear_chain(&base_challenger, &root_proof, &finalizer)
        .expect_err("tampered root step must be rejected");
    assert!(
        matches!(err, WarpError::Verifier(_)),
        "expected verifier error for tampered step, got {err:?}"
    );
}

#[test]
fn witness_root_proof_rejects_removed_intermediate_step() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(2050 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2150 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2250 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (_claimed_final, mut root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");

    root_proof.steps.remove(1);

    root_verifier
        .verify_linear_chain(&base_challenger, &root_proof, &finalizer)
        .expect_err("dropping a WARP root step must be rejected");
}

#[test]
fn witness_root_proof_rejects_reordered_steps() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(2350 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2450 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2550 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (_claimed_final, mut root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");

    root_proof.steps.swap(0, 1);

    root_verifier
        .verify_linear_chain(&base_challenger, &root_proof, &finalizer)
        .expect_err("reordering WARP root steps must be rejected");
}

#[test]
fn witness_root_proof_rejects_wrong_fiat_shamir_state() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(2650 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2750 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (_claimed_final, root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");
    let mut wrong_challenger = base_challenger.clone();
    wrong_challenger.observe(F::ONE);

    root_verifier
        .verify_linear_chain(&wrong_challenger, &root_proof, &finalizer)
        .expect_err("root proof must be bound to the initial Fiat-Shamir state");
}

#[test]
fn witness_root_proof_rejects_wrong_warp_params() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let wrong_params_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, WarpParams::new(2, 1));
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(2850 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2950 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (_claimed_final, root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");

    wrong_params_verifier
        .verify_linear_chain(&base_challenger, &root_proof, &finalizer)
        .expect_err("root proof must be bound to the WARP soundness parameters");
}

#[test]
fn witness_root_proof_rejects_valid_intermediate_accumulator_substitution() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let make_chain = |offset: u64| {
        vec![
            (0..NUM_FRESH)
                .map(|i| make_satisfying_witness(offset + i as u64))
                .collect::<Vec<_>>(),
            (0..3)
                .map(|i| make_satisfying_witness(offset + 100 + i as u64))
                .collect::<Vec<_>>(),
            (0..3)
                .map(|i| make_satisfying_witness(offset + 200 + i as u64))
                .collect::<Vec<_>>(),
        ]
    };
    let (_final_a, mut proof_a) = root_prover
        .prove_linear_chain(&base_challenger, &make_chain(3050), &finalizer)
        .expect("root prove A");
    let (_final_b, proof_b) = root_prover
        .prove_linear_chain(&base_challenger, &make_chain(4050), &finalizer)
        .expect("root prove B");

    proof_a.steps[1].instance = proof_b.steps[1].instance.clone();

    root_verifier
        .verify_linear_chain(&base_challenger, &proof_a, &finalizer)
        .expect_err("substituting another valid intermediate accumulator must be rejected");
}

#[test]
fn witness_root_proof_rejects_valid_finalizer_replay_from_other_chain() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let make_chain = |offset: u64| {
        vec![
            (0..NUM_FRESH)
                .map(|i| make_satisfying_witness(offset + i as u64))
                .collect::<Vec<_>>(),
            (0..3)
                .map(|i| make_satisfying_witness(offset + 100 + i as u64))
                .collect::<Vec<_>>(),
        ]
    };
    let (_final_a, mut proof_a) = root_prover
        .prove_linear_chain(&base_challenger, &make_chain(5050), &finalizer)
        .expect("root prove A");
    let (_final_b, proof_b) = root_prover
        .prove_linear_chain(&base_challenger, &make_chain(6050), &finalizer)
        .expect("root prove B");

    proof_a.final_proof = proof_b.final_proof;

    root_verifier
        .verify_linear_chain(&base_challenger, &proof_a, &finalizer)
        .expect_err("replaying another chain's valid finalizer proof must be rejected");
}

#[test]
fn witness_root_proof_rejects_tampered_finalizer_proof() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_witnesses = vec![
        (0..NUM_FRESH)
            .map(|i| make_satisfying_witness(2200 + i as u64))
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| make_satisfying_witness(2300 + i as u64))
            .collect::<Vec<_>>(),
    ];

    let (_claimed_final, mut root_proof) = root_prover
        .prove_linear_chain(&base_challenger, &step_witnesses, &finalizer)
        .expect("root prove");

    root_proof.final_proof.w[0] += EF::ONE;

    let err = root_verifier
        .verify_linear_chain(&base_challenger, &root_proof, &finalizer)
        .expect_err("tampered finalizer proof must be rejected");
    assert!(
        matches!(err, WarpError::Finalizer(_)),
        "expected finalizer error for tampered final proof, got {err:?}"
    );
}

#[test]
fn external_root_proof_verifies_chain_and_final_decider() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let committer = WarpProver::new(&mmcs, &code, &pesat, params);
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let fresh_verifier = MmcsExternalOpeningVerifier::new(&mmcs);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_fresh_committed = vec![
        (0..NUM_FRESH)
            .map(|i| committer.commit_witness(make_satisfying_witness(2400 + i as u64)))
            .collect::<Vec<CommittedCodeword<F, MyMmcs>>>(),
        (0..3)
            .map(|i| committer.commit_witness(make_satisfying_witness(2500 + i as u64)))
            .collect::<Vec<CommittedCodeword<F, MyMmcs>>>(),
    ];

    let (claimed_final, root_proof) = root_prover
        .prove_external_linear_chain(&base_challenger, &mmcs, step_fresh_committed, &finalizer)
        .expect("external root prove");

    let verified_final = root_verifier
        .verify_external_linear_chain(&base_challenger, &fresh_verifier, &root_proof, &finalizer)
        .expect("external root verify");
    let receipt = root_verifier
        .verify_external_linear_chain_with_receipt(
            &base_challenger,
            &fresh_verifier,
            &root_proof,
            &finalizer,
        )
        .expect("external root verify with receipt");

    assert_eq!(verified_final.mu, claimed_final.mu);
    assert_eq!(verified_final.eta, claimed_final.eta);
    assert_eq!(receipt.final_instance, verified_final);

    let claim = root_proof.claim();
    assert_eq!(claim.step_fresh_commitments.len(), 2);
    assert_eq!(claim.step_fresh_commitments[0].len(), NUM_FRESH);
    assert_eq!(claim.step_fresh_commitments[1].len(), 3);
    assert_eq!(claim.step_instances.len(), 2);
    assert_eq!(claim.final_instance().unwrap().mu, claimed_final.mu);
    assert_eq!(receipt.claim, claim);
    assert_eq!(
        receipt.claim_digest,
        claim.digest::<F, MyChallenger, _>(base_challenger.clone(), &fresh_verifier)
    );
}

#[test]
fn external_root_proof_rejects_tampered_step_proof() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let committer = WarpProver::new(&mmcs, &code, &pesat, params);
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let fresh_verifier = MmcsExternalOpeningVerifier::new(&mmcs);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let step_fresh_committed = vec![
        (0..NUM_FRESH)
            .map(|i| committer.commit_witness(make_satisfying_witness(2600 + i as u64)))
            .collect::<Vec<CommittedCodeword<F, MyMmcs>>>(),
        (0..3)
            .map(|i| committer.commit_witness(make_satisfying_witness(2700 + i as u64)))
            .collect::<Vec<CommittedCodeword<F, MyMmcs>>>(),
    ];

    let (_claimed_final, mut root_proof) = root_prover
        .prove_external_linear_chain(&base_challenger, &mmcs, step_fresh_committed, &finalizer)
        .expect("external root prove");

    root_proof.steps[1].proof.eta += EF::ONE;

    let err = root_verifier
        .verify_external_linear_chain(&base_challenger, &fresh_verifier, &root_proof, &finalizer)
        .expect_err("tampered external root step must be rejected");
    assert!(
        matches!(err, WarpError::Verifier(_)),
        "expected verifier error for tampered external step, got {err:?}"
    );
}

#[test]
fn external_root_proof_rejects_valid_fresh_commitment_substitution() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let committer = WarpProver::new(&mmcs, &code, &pesat, params);
    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let root_verifier = WarpRootVerifier::new(&mmcs, &code, &pesat, params);
    let fresh_verifier = MmcsExternalOpeningVerifier::new(&mmcs);
    let finalizer = WitnessFinalizer::new(&mmcs, &code, &pesat);

    let alternate = committer.commit_witness(make_satisfying_witness(2800));
    let step_fresh_committed = vec![
        (0..NUM_FRESH)
            .map(|i| committer.commit_witness(make_satisfying_witness(2900 + i as u64)))
            .collect::<Vec<CommittedCodeword<F, MyMmcs>>>(),
        (0..3)
            .map(|i| committer.commit_witness(make_satisfying_witness(3000 + i as u64)))
            .collect::<Vec<CommittedCodeword<F, MyMmcs>>>(),
    ];

    let (_claimed_final, mut root_proof) = root_prover
        .prove_external_linear_chain(&base_challenger, &mmcs, step_fresh_committed, &finalizer)
        .expect("external root prove");

    root_proof.steps[0].fresh_commitments[0] = alternate.commitment;

    root_verifier
        .verify_external_linear_chain(&base_challenger, &fresh_verifier, &root_proof, &finalizer)
        .expect_err("substituting a different valid fresh commitment must be rejected");
}

#[test]
fn tampered_codeword_rejected_by_decider() {
    let (mmcs, base_challenger, pesat, code, params, _acc_instance_unused, _proof_unused) =
        produce_proof();
    // Re-prove to get the witness side.
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(100 + i as u64))
        .collect();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let mut p_ch = base_challenger.clone();
    let (mut acc, _proof) = prover.prove(&mut p_ch, &witnesses, &[]);
    // Flip a single symbol of the merged codeword.
    acc.witness.f[0] += EF::ONE;

    let decider = WarpDecider::new(&mmcs, &code, &pesat);
    let err = decider
        .decide(&acc.instance, &acc.witness)
        .expect_err("tampered f must be rejected");
    assert!(
        matches!(
            err,
            DeciderError::MerkleRoot | DeciderError::MlEval | DeciderError::EncodingMismatch
        ),
        "expected MerkleRoot/MlEval/EncodingMismatch, got {err:?}"
    );
}

// -----------------------------------------------------------------------------
// 5. prove_with_committed (alphabet-F variant of Construction 10.4) tests.
// -----------------------------------------------------------------------------

/// Honest path: encode + commit each fresh witness externally via
/// `commit_witness`, then run `prove_with_committed` + `verify_with_committed`.
/// One step.
#[test]
fn prove_with_committed_round_trips_one_step() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(1300 + i as u64))
        .collect();

    let prover = WarpProver::new(&mmcs, &code, &pesat, params);

    // Commit each fresh witness externally, simulating an upstream PCS
    // pipeline where segments arrive pre-committed.
    let fresh_committed: Vec<CommittedCodeword<F, MyMmcs>> = witnesses
        .into_iter()
        .map(|w| prover.commit_witness(w))
        .collect();
    let fresh_commitments: Vec<MyComm> = fresh_committed
        .iter()
        .map(|c| c.commitment.clone())
        .collect();

    let mut p_ch = base_challenger.clone();
    let (acc, proof) = prover.prove_with_committed(&mut p_ch, fresh_committed, &[]);

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    verifier
        .verify_with_committed(&mut v_ch, &fresh_commitments, &[], &acc.instance, &proof)
        .expect("verify_with_committed should accept honest proof");

    let decider = WarpDecider::new(&mmcs, &code, &pesat);
    decider
        .decide(&acc.instance, &acc.witness)
        .expect("decider should accept the resulting accumulator");
}

/// Tampering with one of the fresh shift answers — the corresponding Merkle
/// path verification against the external commitment must fail.
#[test]
fn prove_with_committed_rejects_tampered_fresh_shift_answer() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let witnesses: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(1400 + i as u64))
        .collect();

    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let fresh_committed: Vec<CommittedCodeword<F, MyMmcs>> = witnesses
        .into_iter()
        .map(|w| prover.commit_witness(w))
        .collect();
    let fresh_commitments: Vec<MyComm> = fresh_committed
        .iter()
        .map(|c| c.commitment.clone())
        .collect();

    let mut p_ch = base_challenger.clone();
    let (acc, mut proof) = prover.prove_with_committed(&mut p_ch, fresh_committed, &[]);

    // Flip a single fresh shift answer — the Merkle path against the
    // external commitment will then fail.
    proof.fresh_shift_answers[0][0] += F::ONE;

    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let mut v_ch = base_challenger.clone();
    let err = verifier
        .verify_with_committed(&mut v_ch, &fresh_commitments, &[], &acc.instance, &proof)
        .expect_err("tampered fresh shift answer must be rejected");
    assert!(
        matches!(err, VerifierError::MerkleProof { .. }),
        "expected MerkleProof error, got {err:?}"
    );
}

/// Two-step honest path with `prove_with_committed`. Step 2 takes one
/// fresh segment + one prior accumulator (`ℓ_1=3, ℓ_2=1, ℓ=4`).
#[test]
fn prove_with_committed_two_step_accumulation_accepts() {
    let (mmcs, base_challenger, pesat, code) = make_components();
    let params = make_params();
    let prover = WarpProver::new(&mmcs, &code, &pesat, params);
    let verifier = WarpVerifier::new(&mmcs, &code, &pesat, params);
    let decider = WarpDecider::new(&mmcs, &code, &pesat);

    // Step 1: ℓ_1 = 4 fresh.
    let witnesses1: Vec<Vec<F>> = (0..NUM_FRESH)
        .map(|i| make_satisfying_witness(1500 + i as u64))
        .collect();
    let committed1: Vec<CommittedCodeword<F, MyMmcs>> = witnesses1
        .into_iter()
        .map(|w| prover.commit_witness(w))
        .collect();
    let commits1: Vec<MyComm> = committed1.iter().map(|c| c.commitment.clone()).collect();
    let mut p_ch1 = base_challenger.clone();
    let (acc1, proof1) = prover.prove_with_committed(&mut p_ch1, committed1, &[]);
    let mut v_ch1 = base_challenger.clone();
    verifier
        .verify_with_committed(&mut v_ch1, &commits1, &[], &acc1.instance, &proof1)
        .expect("step-1 verify_with_committed");

    // Step 2: ℓ_1 = 3 fresh + 1 prior = ℓ = 4.
    let witnesses2: Vec<Vec<F>> = (0..3)
        .map(|i| make_satisfying_witness(1600 + i as u64))
        .collect();
    let committed2: Vec<CommittedCodeword<F, MyMmcs>> = witnesses2
        .into_iter()
        .map(|w| prover.commit_witness(w))
        .collect();
    let commits2: Vec<MyComm> = committed2.iter().map(|c| c.commitment.clone()).collect();
    let prior_inst1 = acc1.instance.clone();
    let priors = std::vec![acc1];
    let mut p_ch2 = base_challenger.clone();
    let (acc2, proof2) = prover.prove_with_committed(&mut p_ch2, committed2, &priors);
    let mut v_ch2 = base_challenger.clone();
    verifier
        .verify_with_committed(
            &mut v_ch2,
            &commits2,
            std::slice::from_ref(&prior_inst1),
            &acc2.instance,
            &proof2,
        )
        .expect("step-2 verify_with_committed");

    decider
        .decide(&acc2.instance, &acc2.witness)
        .expect("step-2 decide");
}
