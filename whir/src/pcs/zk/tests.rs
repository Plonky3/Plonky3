//! End-to-end and zero-knowledge tests for the HVZK-WHIR pipeline.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::adapter::HidingWhirPcs;
use super::base_case::BaseCaseZkError;
use super::config::{ZkParameters, ZkWhirConfig};
use super::proof::ZkWhirProof;
use super::verifier::ZkVerifierError;
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type TestZkPcs = HidingWhirPcs<EF, F, MyDft, MyMmcs, MyChallenger, SmallRng>;

/// Commitment type of the test PCS.
type TestCommitment = <TestZkPcs as MultilinearPcs<EF, MyChallenger>>::Commitment;

/// Builder for one HVZK-WHIR test instance, in the standard-library
/// `OpenOptions` style.
///
/// Starts from the suite's canonical shape and overrides only what a test
/// varies, then finishes with a terminal operation:
///
/// ```text
///     assert_round_trip()  ->  honest commit / open / verify lifecycle
///     prove()              ->  honest run, returned for tampering
///     prove_with(w, pts)   ->  honest run on a caller-chosen statement
/// ```
struct Setup {
    /// Arity of the committed polynomial.
    num_variables: usize,
    /// Number of opened evaluation claims.
    num_points: usize,
    /// Per-round fold widths of the pipeline.
    folding_factor: FoldingFactor,
    /// Soundness regime driving query counts and OOD samples.
    soundness_type: SecurityAssumption,
    /// Grinding bits per round.
    pow_bits: usize,
    /// Drives the PCS hiding randomness and the random statement.
    seed: u64,
}

impl Setup {
    /// Canonical shape: 12 variables, one point, constant folding 4,
    /// capacity-bound soundness, no grinding.
    ///
    /// The seed drives both the PCS hiding randomness and the witness.
    const fn new(seed: u64) -> Self {
        Self {
            num_variables: 12,
            num_points: 1,
            folding_factor: FoldingFactor::Constant(4),
            soundness_type: SecurityAssumption::CapacityBound,
            pow_bits: 0,
            seed,
        }
    }

    const fn num_variables(mut self, num_variables: usize) -> Self {
        self.num_variables = num_variables;
        self
    }

    const fn num_points(mut self, num_points: usize) -> Self {
        self.num_points = num_points;
        self
    }

    fn folding_factor(mut self, folding_factor: FoldingFactor) -> Self {
        self.folding_factor = folding_factor;
        self
    }

    const fn soundness(mut self, soundness_type: SecurityAssumption) -> Self {
        self.soundness_type = soundness_type;
        self
    }

    const fn pow_bits(mut self, pow_bits: usize) -> Self {
        self.pow_bits = pow_bits;
        self
    }

    /// Builds the hiding PCS for this shape.
    fn pcs(&self) -> TestZkPcs {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
        let config = ZkWhirConfig::new(
            self.num_variables,
            ProtocolParameters {
                security_level: 32,
                pow_bits: self.pow_bits,
                round_log_inv_rates: vec![],
                folding_factor: self.folding_factor.clone(),
                soundness_type: self.soundness_type,
                starting_log_inv_rate: 2,
            },
            ZkParameters {
                ell_zk: 4,
                mask_log_inv_rate: 1,
            },
        )
        .unwrap();
        HidingWhirPcs::new(
            config,
            MyDft::default(),
            mmcs,
            SmallRng::seed_from_u64(self.seed),
        )
    }

    /// Runs the honest commit / open phases on a random statement.
    fn prove(self) -> Proven {
        // Witness and points come from the seed's companion stream, so a
        // fixed seed replays the exact same transcript.
        let mut rng = SmallRng::seed_from_u64(self.seed.wrapping_add(1));
        let witness = Poly::<F>::rand(&mut rng, self.num_variables);
        let points: Vec<Point<EF>> = (0..self.num_points)
            .map(|_| Point::rand(&mut rng, self.num_variables))
            .collect();
        self.prove_with(witness, points)
    }

    /// Runs the honest commit / open phases on a caller-chosen statement.
    fn prove_with(self, witness: Poly<F>, points: Vec<Point<EF>>) -> Proven {
        let pcs = self.pcs();
        let mut prover_challenger = separated_challenger(&pcs);
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
        let proof = pcs.open(prover_data, points.clone(), &mut prover_challenger);
        Proven {
            pcs,
            commitment,
            proof,
            points,
        }
    }

    /// Full honest lifecycle: commit, open, check the evaluations, verify.
    fn assert_round_trip(self) {
        let mut rng = SmallRng::seed_from_u64(self.seed.wrapping_add(1));
        let witness = Poly::<F>::rand(&mut rng, self.num_variables);
        let points: Vec<Point<EF>> = (0..self.num_points)
            .map(|_| Point::rand(&mut rng, self.num_variables))
            .collect();

        let proven = self.prove_with(witness.clone(), points);
        // The claimed evaluations are the honest ones.
        for (point, &eval) in proven.points.iter().zip(&proven.proof.evals) {
            assert_eq!(witness.eval_base(point), eval);
        }
        proven
            .verify()
            .expect("honest HVZK-WHIR proof should verify");
    }
}

/// One honest run, held open for tampering and verification.
struct Proven {
    /// The hiding PCS the proof was produced with.
    pcs: TestZkPcs,
    /// Commitment to the (possibly simulated) witness.
    commitment: TestCommitment,
    /// The opening proof; tamper tests mutate it in place.
    proof: ZkWhirProof<F, EF, MyMmcs>,
    /// The opened points.
    points: Vec<Point<EF>>,
}

impl Proven {
    /// Replays verification against the stored statement.
    fn verify(&self) -> Result<(), ZkVerifierError> {
        let mut challenger = separated_challenger(&self.pcs);
        self.pcs.verify(
            &self.commitment,
            &self.proof,
            &mut challenger,
            self.points.clone(),
        )
    }
}

/// Fresh challenger seeded with the protocol's domain separator.
fn separated_challenger(pcs: &TestZkPcs) -> MyChallenger {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let mut challenger = MyChallenger::new(perm);
    let mut separator = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut separator);
    separator.observe_domain_separator(&mut challenger);
    challenger
}

#[test]
fn zk_whir_end_to_end_single_round() {
    // One code-switching round: 12 vars, constant folding 4
    // (initial fold + 1 switch + final batch).
    Setup::new(1).num_points(2).assert_round_trip();
}

#[test]
fn zk_whir_end_to_end_no_rounds() {
    // Degenerate pipeline: a single fold batch straight into the base case.
    Setup::new(2)
        .num_variables(6)
        .folding_factor(FoldingFactor::Constant(3))
        .assert_round_trip();
}

#[test]
fn zk_whir_end_to_end_multi_round() {
    // Two code-switching rounds with mixed folding factors and grinding,
    // so the round-to-round oracle carry path is exercised.
    let setup = Setup::new(3)
        .num_variables(17)
        .num_points(3)
        .folding_factor(FoldingFactor::ConstantFromSecondRound(5, 3))
        .pow_bits(5);
    assert!(
        setup.pcs().config.n_rounds() >= 2,
        "fixture must drive at least two code-switching rounds",
    );
    setup.assert_round_trip();
}

#[test]
fn zk_whir_end_to_end_partial_final_fold() {
    // Clamped schedule: Constant(8) on 15 variables derives [8, 7].
    // The final-round zk oracle then carries a single message row,
    // which is the tightest point of the config's rate-slack bound.
    let setup = Setup::new(4)
        .num_variables(15)
        .folding_factor(FoldingFactor::Constant(8));
    assert_eq!(
        setup.pcs().config.folding_schedule,
        vec![8, 7],
        "fixture must derive a clamped final fold",
    );
    setup.assert_round_trip();
}

#[test]
fn zk_whir_code_switch_overhead_accounting() {
    // Construction 9.7 per-round overhead (eprint 2026/391, #1587):
    //
    //     mask message  =  Fold(r_prev, gamma) || pad      (the m_zk part)
    //     pad length    =  t_ood                           (one slot per OOD)
    //     wire          =  t_ood private OOD answers
    //
    // The mask codeword is Merkle-committed, so only its commitment and the
    // t_ood answers reach the proof; assert that observable part.
    let setup = Setup::new(3)
        .num_variables(17)
        .folding_factor(FoldingFactor::ConstantFromSecondRound(5, 3))
        .pow_bits(5);
    let pcs = setup.pcs();
    assert!(
        pcs.config.n_rounds() >= 2,
        "need code-switching rounds to audit"
    );

    let proven = setup.prove();
    assert_eq!(proven.proof.rounds.len(), pcs.config.n_rounds());

    for (round, round_proof) in proven.proof.rounds.iter().enumerate() {
        let t_ood = pcs.config.round_parameters[round].ood_samples;
        // Wire: exactly t_ood private OOD answers this round.
        assert_eq!(round_proof.ood_answers.len(), t_ood);
        // Mask message: folded source randomness followed by a t_ood pad.
        let folded_randomness = pcs.config.oracle_randomness[round];
        assert_eq!(
            pcs.config.switch_masks[round].message_len,
            folded_randomness + t_ood,
        );
    }
}

#[test]
fn zk_whir_end_to_end_unique_decoding() {
    // Unique decoding has zero OOD samples: empty pads and OOD layers.
    Setup::new(4)
        .num_points(2)
        .soundness(SecurityAssumption::UniqueDecoding)
        .assert_round_trip();
}

#[test]
fn zk_whir_end_to_end_johnson_bound() {
    Setup::new(5)
        .folding_factor(FoldingFactor::Constant(3))
        .soundness(SecurityAssumption::JohnsonBound)
        .pow_bits(5)
        .assert_round_trip();
}

#[test]
fn zk_whir_rejects_wrong_eval() {
    // A tampered claimed evaluation must be rejected.
    let mut proven = Setup::new(6).prove();
    proven.proof.evals[0] += EF::ONE;
    let err = proven.verify().unwrap_err();
    // Diverged transcript: the verifier samples different STIR positions,
    // so a round-0 opening fails to authenticate.
    //
    // The exact position is a transcript artifact, not a protocol
    // invariant, so only the variant and round are pinned.
    assert!(matches!(
        err,
        ZkVerifierError::MerkleVerificationFailed { round: 0, .. },
    ));
}

#[test]
fn zk_whir_rejects_wrong_arity_opening_point() {
    // Invariant: a malformed public statement errors, never panics.
    //
    //     committed arity      = 12
    //     verification point   = 3 variables
    //     -> arity mismatch error before any folding arithmetic
    // Honest commit and open against the full-arity point.
    let mut proven = Setup::new(6).prove();
    // Verify against a short point instead.
    proven.points = vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(7), 3)];
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::ClaimArityMismatch {
            claim: 0,
            expected: 12,
            actual: 3,
        },
    );
}

#[test]
fn zk_whir_rejects_tampered_ood_answer() {
    // A shifted private OOD answer desynchronizes the carried claim.
    let mut proven = Setup::new(8).prove();
    assert!(
        !proven.proof.rounds.is_empty() && !proven.proof.rounds[0].ood_answers.is_empty(),
        "fixture should produce at least one private OOD answer",
    );
    proven.proof.rounds[0].ood_answers[0] += EF::ONE;
    let err = proven.verify().unwrap_err();
    // Diverged transcript: the verifier samples different STIR positions,
    // so a round-0 opening fails to authenticate.
    //
    // The exact position is a transcript artifact, not a protocol
    // invariant, so only the variant and round are pinned.
    assert!(matches!(
        err,
        ZkVerifierError::MerkleVerificationFailed { round: 0, .. },
    ));
}

#[test]
fn zk_whir_rejects_tampered_base_case_reveal() {
    // A shifted blinded message breaks the joint target identity.
    let mut proven = Setup::new(10).prove();
    proven.proof.base_case.blinded_message[0] += EF::ONE;
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::BaseCase(BaseCaseZkError::TargetCheckFailed),
    );
}

#[test]
fn zk_whir_rejects_tampered_masked_claim() {
    // The fresh-side claim mu_g is bound before gamma is sampled.
    //
    // Mutation: shift mu_g -> the joint target identity no longer closes.
    let mut proven = Setup::new(22).prove();
    proven.proof.base_case.masked_claim += EF::ONE;
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::BaseCase(BaseCaseZkError::TargetCheckFailed),
    );
}

#[test]
fn zk_whir_end_to_end_no_claims() {
    // Edge case: an opening with zero claims still runs the full pipeline.
    //
    //     no claims  ->  empty initial relation, target = 0
    //                ->  the masks alone carry the base-case identity
    Setup::new(12).num_points(0).assert_round_trip();
}

#[test]
fn zk_whir_rejects_truncated_rounds() {
    // Fixture state: 12 vars, folding 4 -> 1 code-switching round.
    //
    // Mutation: drop it -> 0 != 1 -> reject before any transcript work.
    let mut proven = Setup::new(13).prove();
    let expected = proven.proof.rounds.len();
    let _ = proven.proof.rounds.pop().expect("fixture has one round");
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::RoundCountMismatch {
            expected,
            actual: 0,
        },
    );
}

#[test]
fn zk_whir_rejects_truncated_sumchecks() {
    // Fixture state: 1 round -> 2 fold batches -> 2 sumcheck transcripts.
    //
    // Mutation: drop one -> 1 != 2 -> reject before any transcript work.
    let mut proven = Setup::new(14).prove();
    let _ = proven
        .proof
        .sumchecks
        .pop()
        .expect("fixture has two batches");
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::SumcheckBatchCountMismatch {
            expected: 2,
            actual: 1,
        },
    );
}

#[test]
fn zk_whir_rejects_missing_ood_answer() {
    // Mutation: drop one private OOD answer from the round.
    //
    //     answers:      [y_1, ..., y_{t-1}]
    //     ood_samples:  t                    -> count mismatch -> reject
    let mut proven = Setup::new(15).prove();
    let _ = proven.proof.rounds[0]
        .ood_answers
        .pop()
        .expect("fixture has OOD answers");
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::OodAnswerCountMismatch {
            round: 0,
            expected: 1,
            actual: 0,
        },
    );
}

#[test]
fn zk_whir_rejects_missing_query_opening() {
    // Mutation: drop one STIR query opening from the round.
    //
    // Openings are not absorbed, so the transcript matches up to the
    // count check itself.
    let mut proven = Setup::new(16).prove();
    let _ = proven.proof.rounds[0]
        .queries
        .pop()
        .expect("fixture has query openings");
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::QueryCountMismatch {
            round: 0,
            expected: 17,
            actual: 16,
        },
    );
}

#[test]
fn zk_whir_rejects_extra_eval() {
    // Mutation: one more claimed evaluation than opened points.
    //
    //     evals:  [v_1, EXTRA]
    //     points: [z_1]          -> 2 != 1 -> reject
    let mut proven = Setup::new(17).prove();
    proven.proof.evals.push(EF::ONE);
    let err = proven.verify().unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::EvalCountMismatch {
            expected: 1,
            actual: 2,
        },
    );
}

#[test]
fn zk_whir_rejects_tampered_pow_witness() {
    // Fixture state: 5 grinding bits on the code-switching round.
    //
    // Mutation: shift the round's PoW witness -> the grind check fails.
    let mut proven = Setup::new(18).pow_bits(5).prove();
    proven.proof.rounds[0].pow_witness += F::ONE;
    let err = proven.verify().unwrap_err();
    // The bad witness usually fails the grind; if it coincidentally still
    // satisfies it, the diverged transcript fails a later round-0 opening.
    assert!(matches!(
        err,
        ZkVerifierError::InvalidPowWitness { round: 0 }
            | ZkVerifierError::MerkleVerificationFailed { round: 0, .. },
    ));
}

#[test]
fn zk_whir_rejects_tampered_sumcheck_wire() {
    // Mutation: shift one coefficient of the first sumcheck wire.
    //
    // The wire is absorbed, so the remaining transcript diverges and the
    // replayed batch cannot close.
    let mut proven = Setup::new(19).prove();
    proven.proof.sumchecks[0].round_coefficients[0][0] += EF::ONE;
    let err = proven.verify().unwrap_err();
    // Diverged transcript: the verifier samples different STIR positions,
    // so a round-0 opening fails to authenticate.
    //
    // The exact position is a transcript artifact, not a protocol
    // invariant, so only the variant and round are pinned.
    assert!(matches!(
        err,
        ZkVerifierError::MerkleVerificationFailed { round: 0, .. },
    ));
}

#[test]
fn zk_whir_rejects_wrong_commitment() {
    // Mutation: verify an honest proof against another witness's commitment.
    //
    // The commitment is the first absorb, so the whole transcript diverges.
    let mut proven = Setup::new(20).prove();
    let other = Setup::new(21).prove();
    proven.commitment = other.commitment;
    let err = proven.verify().unwrap_err();
    // Diverged transcript: the verifier samples different STIR positions,
    // so a round-0 opening fails to authenticate.
    //
    // The exact position is a transcript artifact, not a protocol
    // invariant, so only the variant and round are pinned.
    assert!(matches!(
        err,
        ZkVerifierError::MerkleVerificationFailed { round: 0, .. },
    ));
}

/// Conditions a uniformly random witness on the public claims.
///
/// - Claims at base-field points have base-field values.
/// - Corrections on a few hypercube coefficients therefore suffice.
/// - With pivot slots `b_j`, the corrections solve the linear system
///
/// ```text
///     sum_j delta_j * eq(z_i, b_j) = v_i - g(z_i)
/// ```
///
/// - The system is solved jointly via Gaussian elimination over the
///   extension field.
/// - Base-field inputs keep the solution in the base field.
fn condition_witness(
    rng: &mut SmallRng,
    num_variables: usize,
    claims: &[(Point<EF>, EF)],
) -> Poly<F> {
    let mut garbage = Poly::<F>::rand(rng, num_variables);
    let n = claims.len();
    if n == 0 {
        return garbage;
    }

    // System matrix A[i][j] = eq(z_i, pivot_j) and right-hand side.
    let pivots: Vec<Point<EF>> = (0..n)
        .map(|j| Point::<EF>::hypercube(j, num_variables))
        .collect();
    let mut matrix: Vec<Vec<EF>> = claims
        .iter()
        .map(|(point, _)| {
            pivots
                .iter()
                .map(|pivot| Point::eval_eq(pivot.as_slice(), point.as_slice()))
                .collect()
        })
        .collect();
    let mut rhs: Vec<EF> = claims
        .iter()
        .map(|(point, value)| *value - garbage.eval_base(point))
        .collect();

    // Gaussian elimination without pivot search: random base points make the
    // leading minors nonzero with overwhelming probability.
    for col in 0..n {
        let inv = matrix[col][col].inverse();
        for entry in matrix[col].iter_mut() {
            *entry *= inv;
        }
        rhs[col] *= inv;
        for row in 0..n {
            if row != col {
                let factor = matrix[row][col];
                let pivot_row = matrix[col].clone();
                for (entry, &pivot) in matrix[row].iter_mut().zip(&pivot_row) {
                    *entry -= factor * pivot;
                }
                let sub = factor * rhs[col];
                rhs[row] -= sub;
            }
        }
    }

    for (j, delta) in rhs.iter().enumerate() {
        // Base-field points with base-field values keep corrections in F.
        let coefficients = <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(delta);
        assert!(coefficients[1..].iter().all(|c| *c == F::ZERO));
        garbage.as_mut_slice()[j] += coefficients[0];
    }
    garbage
}

#[test]
fn zk_whir_simulator_witness_free_transcript_accepts() {
    // Completeness direction of the simulator.
    //
    //     witness-free transcript  ->  verifies
    //                              ->  exposes exactly the public claims
    //
    // The message is random, conditioned only on those claims.
    // This is not the composed-distribution argument (Theorem 4.5).
    // The per-component distribution proofs live where the masks are drawn:
    //
    //     masked sumcheck wires  ->  p3-sumcheck simulator tests
    //     private OOD answers    ->  code_switch programmability tests
    //     one-time-pad reveals   ->  base case OTP test
    let num_variables = 12;
    let mut rng = SmallRng::seed_from_u64(21);

    // Real witness and its public claims at base-field points.
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points: Vec<Point<EF>> = (0..2)
        .map(|_| {
            Point::new(
                (0..num_variables)
                    .map(|_| EF::from(rng.random::<F>()))
                    .collect(),
            )
        })
        .collect();
    let claims: Vec<(Point<EF>, EF)> = points
        .iter()
        .map(|point| (point.clone(), witness.eval_base(point)))
        .collect();

    // Simulator side: garbage witness agreeing exactly on the claims.
    let mut simulator_rng = SmallRng::seed_from_u64(22);
    let simulated_witness = condition_witness(&mut simulator_rng, num_variables, &claims);
    for (point, value) in &claims {
        assert_eq!(simulated_witness.eval_base(point), *value);
    }
    assert_ne!(
        simulated_witness.as_slice(),
        witness.as_slice(),
        "the simulated witness must differ from the real one off the claims",
    );

    // The simulated transcript verifies against the simulated commitment
    // with the real public claims.
    let proven = Setup::new(23).prove_with(simulated_witness, points);
    assert_eq!(
        proven.proof.evals,
        claims.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
        "the simulated transcript must expose exactly the public claims",
    );
    proven
        .verify()
        .expect("simulated HVZK transcript must verify");
}

#[test]
fn zk_whir_masks_are_witness_independent() {
    // Invariant: the hiding material is drawn from the prover RNG
    // independently of the witness.
    //
    //     matched RNG streams + different witnesses
    //         ->  RNG-only draws coincide
    //         ->  both runs verify
    //         ->  blinded reveals differ even though the claims pin
    //             the evaluations
    let num_variables = 12;
    let mut rng = SmallRng::seed_from_u64(31);
    let witness_a = Poly::<F>::rand(&mut rng, num_variables);
    let witness_b = Poly::<F>::rand(&mut rng, num_variables);
    let point = Point::new(
        (0..num_variables)
            .map(|_| EF::from(rng.random::<F>()))
            .collect::<Vec<_>>(),
    );

    let run = |witness: Poly<F>| {
        let proven = Setup::new(41).prove_with(witness, vec![point.clone()]);
        proven.verify().expect("honest proof verifies");
        proven.proof
    };

    let proof_a = run(witness_a);
    let proof_b = run(witness_b);
    assert_ne!(
        proof_a.base_case.blinded_message, proof_b.base_case.blinded_message,
        "different witnesses produce different (uniformly padded) reveals",
    );
}
