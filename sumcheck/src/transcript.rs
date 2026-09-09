//! Fiat-Shamir transcript of the quadratic sumcheck.
//!
//! # Overview
//!
//! This module is the single statement of what a batch of quadratic sumcheck rounds absorbs.
//! Both sides consume it through the same three calls.
//!
//! The description itself is a value, built from the numbers that shape a batch.
//!
//! # Shape
//!
//! ```text
//!     round 0:  polynomial     2 extension elements
//!               grinding       present only when the difficulty is positive
//!               challenge      1 extension element
//!     round 1:  ...
//! ```
//!
//! A quadratic round polynomial has three values and two of them cross the wire.
//! The third is what the round identity of the basis reconstructs.
//!
//! # Sub-protocol
//!
//! A sumcheck is almost always a phase of something larger.
//!
//! It therefore seeds a transcript of its own from a borrowed sponge, and hands the sponge back.
//! The surrounding protocol keeps its own description, and brackets this run inside it.
//!
//! That is what lets the two descriptions be written independently and still compose.
//!
//! # Empty runs
//!
//! A batch of zero rounds absorbs nothing and draws nothing.
//!
//! Such a batch is therefore not a transcript at all.
//! No seed is folded in, and no step is played.
//!
//! A caller whose configuration reduces to zero rounds needs no special case of its own.
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//!
//! The round count and the grinding difficulty both change the description.
//! The fingerprint therefore covers them.
//!
//! The basis does not change it.
//! It renames the two values a round sends without moving a single step.
//! It is bound as an instance label instead.
//!
//! No shipped prover or verifier drives the projective reading yet.
//! The label makes the two seeds differ before one exists.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::ExtensionField;

use crate::error::SumcheckError;
use crate::strategy::Basis;

/// Version byte bound into the transcript seed.
///
/// Bumping it separates two revisions of this protocol.
/// It does so even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-sumcheck-quadratic";

/// Step label of the two round-polynomial values a round sends.
const ROUND_POLY: &str = "round_poly";

/// Step label of a per-round grinding step.
const ROUND_POW: &str = "round_pow";

/// Step label of a per-round challenge.
const ROUND_CHALLENGE: &str = "round_challenge";

/// Values of the round polynomial that cross the wire.
const VALUES_PER_ROUND: usize = 2;

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Numbers that fix the transcript of one batch of quadratic sumcheck rounds.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    /// Number of variables the batch binds, and so the number of rounds.
    pub num_rounds: usize,
    /// Grinding difficulty guarding each round's challenge, or zero to omit grinding.
    pub pow_bits: usize,
    /// Basis the two transmitted values are read in.
    pub basis: Basis,
}

impl SumcheckShape {
    /// Collect the numbers that fix one batch of rounds.
    ///
    /// # Arguments
    ///
    /// - `num_rounds`: number of variables the batch binds.
    /// - `pow_bits`: grinding difficulty per round, or zero to omit grinding.
    /// - `basis`: how the two transmitted values are read.
    #[must_use]
    pub const fn new(num_rounds: usize, pow_bits: usize, basis: Basis) -> Self {
        Self {
            num_rounds,
            pow_bits,
            basis,
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// A flat sequence of leaf steps always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Both readings send two values per round, so both describe the same steps.
        //
        //     evaluation basis:  [h(0), h(inf)]
        //     projective basis:  [s(1), s(inf)]
        //
        // Matching every variant here is what turns a future reading of some other width
        // into a compile error, rather than a second protocol sharing this description.
        let values_per_round = match self.basis {
            Basis::Evaluation => VALUES_PER_ROUND,
            Basis::Projective => VALUES_PER_ROUND,
        };

        // Up to three steps per round.
        let mut steps = Vec::with_capacity(3 * self.num_rounds);

        for _ in 0..self.num_rounds {
            // The round polynomial is one step carrying both transmitted values.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                ROUND_POLY,
                Length::Fixed(values_per_round),
            ));

            // Grinding sits between the polynomial and the challenge it protects.
            //
            // The difficulty travels inside the step.
            // A verifier expecting a cheaper grind therefore fails the shape check.
            if self.pow_bits > 0 {
                steps.push(Interaction::algebra::<F, F>(
                    Hierarchy::Atomic,
                    Kind::Pow,
                    ROUND_POW,
                    Length::Fixed(self.pow_bits),
                ));
            }

            // The challenge binds the variable this round reduces away.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                ROUND_CHALLENGE,
                Length::Scalar,
            ));
        }

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity, this shape, and the basis.
    ///
    /// The round count and the difficulty both move the step sequence.
    /// The fingerprint therefore carries them.
    ///
    /// The basis does not move it.
    ///
    /// ```text
    ///     evaluation basis:  the pair is  [h(0), h(inf)]
    ///     projective basis:  the pair is  [s(1), s(inf)]
    /// ```
    ///
    /// Both are two extension elements followed by one challenge.
    /// Only the instance label separates two runs that disagree on which reading is meant.
    ///
    /// Every shipped driver picks the evaluation reading, so no live pair can disagree today.
    /// The label is what keeps the two apart once a projective driver exists.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        separator.instance(&[match self.basis {
            Basis::Evaluation => 0,
            Basis::Projective => 1,
        }]);

        separator
    }
}

/// Prover-side transcript of one batch of quadratic sumcheck rounds.
///
/// # Overview
///
/// Holds the only definition of what a prover writes per round.
///
/// Every prover of these rounds drives them through this type.
/// No two prover loops can then drift from each other, or from the verifier.
///
/// # Borrowing
///
/// The challenger is borrowed, not consumed.
///
/// A sumcheck runs inside a larger protocol.
/// That protocol's own transcript continues where this one stops.
pub struct ProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// Absent when the batch has no rounds, because then there is nothing to describe.
    state: Option<ProverState<&'a mut C, Alphabet<F>>>,
    /// The numbers this batch was described with.
    shape: SumcheckShape,
    /// Marker for the extension field the rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// A batch of zero rounds leaves the sponge exactly as it was found.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the batch.
    /// - `shape`: the numbers that fix this batch's transcript.
    pub fn new(challenger: &'a mut C, shape: SumcheckShape) -> Self {
        let state = (shape.num_rounds > 0).then(|| {
            // Seeding folds the shape fingerprint into the sponge before any step.
            let separator = shape.domain_separator::<F, EF>();
            ProverState::new(challenger, &separator)
        });

        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Play one round: bind the two transmitted values, grind, and draw the challenge.
    ///
    /// # Returns
    ///
    /// - The challenge that binds this round's variable.
    /// - The grinding witness, when the difficulty is positive.
    ///
    /// The caller stores both in its own proof.
    ///
    /// # Panics
    ///
    /// When the batch was described with no rounds at all.
    pub fn round(&mut self, c_a: EF, c_inf: EF) -> (EF, Option<F>) {
        let state = self
            .state
            .as_mut()
            .expect("a batch described with no rounds has no round to play");

        // Bind both values before the challenge that will be evaluated against them.
        state.observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, &[c_a, c_inf]);

        // Grinding raises the cost of searching for a favourable challenge.
        let witness =
            (self.shape.pow_bits > 0).then(|| state.observe_pow(ROUND_POW, self.shape.pow_bits));

        // Draw the challenge for the caller to fold with.
        let challenge = state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner();

        (challenge, witness)
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer rounds were played than the batch was described with.
    pub fn finish(self) {
        if let Some(state) = self.state {
            // Nothing was written to the driver's own buffer.
            // Closing is therefore purely the check that the description was consumed.
            assert!(
                state.finalize().is_empty(),
                "the quadratic sumcheck carries every value in its own proof",
            );
        }
    }
}

/// Verifier-side transcript of one batch of quadratic sumcheck rounds.
///
/// Mirrors the prover side call for call, over the same description.
///
/// The values come from the proof rather than from a wire.
/// The caller must therefore have counted them against the described shape first.
pub struct VerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    ///
    /// Absent when the batch has no rounds, because then there is nothing to describe.
    state: Option<VerifierState<'static, &'a mut C, Alphabet<F>>>,
    /// The numbers this batch was described with.
    shape: SumcheckShape,
    /// Index of the next round to play, used to place a round failure.
    round: usize,
    /// Marker for the extension field the rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// The argument matches the prover's, so both sides seed identically.
    pub fn new(challenger: &'a mut C, shape: SumcheckShape) -> Self {
        let state = (shape.num_rounds > 0).then(|| {
            // Seeding folds the shape fingerprint into the sponge before any step.
            let separator = shape.domain_separator::<F, EF>();
            VerifierState::new(challenger, &separator, &[])
        });

        Self {
            state,
            shape,
            round: 0,
            _ef: PhantomData,
        }
    }

    /// Replay one round: bind both values, re-check the grind, draw the challenge.
    ///
    /// Every value comes from the proof, so every disagreement is a rejection.
    ///
    /// # Errors
    ///
    /// - Grinding is enabled and the round carries no witness.
    /// - The witness misses the required difficulty.
    ///
    /// # Panics
    ///
    /// When the batch was described with no rounds at all.
    pub fn round(&mut self, c_a: EF, c_inf: EF, witness: Option<F>) -> Result<EF, SumcheckError> {
        let round = self.round;
        self.round += 1;

        let state = self
            .state
            .as_mut()
            .expect("a batch described with no rounds has no round to play");

        // Bind both values before the challenge that will be evaluated against them.
        //
        // A round always sends exactly two, so this step cannot disagree on its width.
        state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, &[c_a, c_inf])
            .expect("a round always carries exactly the two values the step describes");

        // Re-run the prover's grinding step on the witness it committed to.
        if self.shape.pow_bits > 0 {
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                state.abort();
                return Err(SumcheckError::MissingPowWitness {
                    round,
                    difficulty: self.shape.pow_bits,
                });
            };
            // A failing check releases the completeness check on its own way out.
            state
                .observe_pow(ROUND_POW, self.shape.pow_bits, witness)
                .map_err(|_| SumcheckError::InvalidPowWitness {
                    round,
                    difficulty: self.shape.pow_bits,
                })?;
        }

        // Draw the same challenge the prover saw.
        Ok(state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner())
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer rounds were played than the batch was described with.
    pub fn finish(self) {
        if let Some(state) = self.state {
            // The proof carries every value, so no unread wire bytes can remain.
            state
                .finalize()
                .expect("the quadratic sumcheck reads an empty wire");
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: SumcheckShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    #[test]
    fn every_number_that_shapes_a_batch_reaches_the_seed() {
        // Fixture state: 4 rounds, no grinding, values read in the evaluation basis.
        let base = SumcheckShape::new(4, 0, Basis::Evaluation);

        // One more round appends two more steps.
        assert_ne!(
            first_challenge(base),
            first_challenge(SumcheckShape::new(5, 0, Basis::Evaluation))
        );

        // Enabling grinding inserts a step per round.
        assert_ne!(
            first_challenge(base),
            first_challenge(SumcheckShape::new(4, 8, Basis::Evaluation))
        );

        // Two positive difficulties differ only inside the grinding steps.
        assert_ne!(
            first_challenge(SumcheckShape::new(4, 8, Basis::Evaluation)),
            first_challenge(SumcheckShape::new(4, 9, Basis::Evaluation))
        );

        // The basis leaves the step sequence identical, so only the label separates the two.
        assert_ne!(
            first_challenge(base),
            first_challenge(SumcheckShape::new(4, 0, Basis::Projective))
        );
    }

    #[test]
    fn a_batch_of_no_rounds_leaves_the_sponge_untouched() {
        // Invariant: a batch with nothing to absorb is not a transcript.
        //
        // A caller whose configuration reduces to zero rounds runs no steps and seeds nothing.
        // Its counterpart on the other side, which may not call in at all, therefore stays in step.
        let mut untouched = fresh_challenger();
        let expected: F = untouched.sample();

        let mut challenger = fresh_challenger();
        let transcript = ProverTranscript::<Ch, F, EF>::new(
            &mut challenger,
            SumcheckShape::new(0, 0, Basis::Evaluation),
        );
        transcript.finish();

        assert_eq!(CanSample::<F>::sample(&mut challenger), expected);
    }

    #[test]
    fn a_round_missing_its_grinding_witness_is_rejected() {
        // Described batch: 2 rounds guarded by 4 bits of grinding each.
        //
        // A described grinding step cannot be replayed with no witness to feed it.
        //
        // This is the path a downstream crate reaches with a proof whose witness list is short.
        // A panic here would compound with the driver's own drop check and abort the process.
        let mut challenger = fresh_challenger();
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(
            &mut challenger,
            SumcheckShape::new(2, 4, Basis::Evaluation),
        );

        let err = transcript
            .round(EF::ONE, EF::ONE, None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(
            err,
            SumcheckError::MissingPowWitness {
                round: 0,
                difficulty: 4
            }
        );
    }

    #[test]
    fn a_grinding_witness_below_the_difficulty_is_rejected() {
        // Described batch: 1 round guarded by 12 bits of grinding.
        //
        // Mutation: hand the round a witness that was never ground at all.
        let mut challenger = fresh_challenger();
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(
            &mut challenger,
            SumcheckShape::new(1, 12, Basis::Evaluation),
        );

        let err = transcript
            .round(EF::ONE, EF::ONE, Some(F::ZERO))
            .expect_err("a witness below the required difficulty must error");

        assert_eq!(
            err,
            SumcheckError::InvalidPowWitness {
                round: 0,
                difficulty: 12
            }
        );
    }

    /// The challenges a verifier redraws from one recorded pair of values, per round.
    fn replay(rounds: &[[EF; 2]]) -> Vec<EF> {
        let mut challenger = fresh_challenger();
        let shape = SumcheckShape::new(rounds.len(), 0, Basis::Evaluation);
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let replayed = rounds
            .iter()
            .map(|&[c_a, c_inf]| transcript.round(c_a, c_inf, None).unwrap())
            .collect();
        transcript.finish();
        replayed
    }

    #[test]
    fn a_perturbed_finite_point_value_moves_every_later_challenge() {
        // Invariant: the pair is bound before the challenge that is evaluated against it.
        //
        // Without that, a prover could pick the round polynomial to suit the challenge.
        //
        // Fixture state: two rounds, no grinding.
        //
        // Mutation: change the first round's finite-point value.
        //
        //     honest:   [[1, 2], [2, 1]]
        //     tampered: [[3, 2], [2, 1]]
        let honest = [[EF::ONE, EF::TWO], [EF::TWO, EF::ONE]];
        let mut tampered = honest;
        tampered[0][0] = EF::from_u8(3);

        assert_ne!(replay(&honest), replay(&tampered));
    }

    #[test]
    fn a_perturbed_leading_coefficient_moves_every_later_challenge() {
        // Same invariant for the other half of the pair.
        //
        // Fixture state: two rounds, no grinding.
        //
        // Mutation: change the first round's leading coefficient.
        //
        //     honest:   [[1, 2], [2, 1]]
        //     tampered: [[1, 3], [2, 1]]
        let honest = [[EF::ONE, EF::TWO], [EF::TWO, EF::ONE]];
        let mut tampered = honest;
        tampered[0][1] = EF::from_u8(3);

        assert_ne!(replay(&honest), replay(&tampered));
    }

    #[test]
    fn a_recorded_grinding_witness_replays_to_the_same_challenge() {
        // Completeness of the guarded path: the witness is absorbed, not merely checked.
        //
        // The verifier redraws the challenge only after feeding back the witness the prover found.
        //
        // Fixture state: one round guarded by 4 bits.
        let shape = SumcheckShape::new(1, 4, Basis::Evaluation);

        let mut challenger = fresh_challenger();
        let mut prover = ProverTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let (challenge, witness) = prover.round(EF::ONE, EF::TWO);
        prover.finish();
        let witness = witness.expect("a positive difficulty always yields a witness");

        // Replaying the recorded witness reaches the recorded challenge.
        let mut challenger = fresh_challenger();
        let mut verifier = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let replayed = verifier.round(EF::ONE, EF::TWO, Some(witness)).unwrap();
        verifier.finish();

        assert_eq!(challenge, replayed);
    }

    #[test]
    fn both_sides_of_a_round_draw_the_same_challenge() {
        // Completeness: the two drivers walk one description and land on one challenge.
        let shape = SumcheckShape::new(2, 0, Basis::Evaluation);

        let mut prover_challenger = fresh_challenger();
        let mut prover = ProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape);
        let (first, _) = prover.round(EF::ONE, EF::TWO);
        let (second, _) = prover.round(EF::TWO, EF::ONE);
        prover.finish();

        let mut verifier_challenger = fresh_challenger();
        let mut verifier = VerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
        let replayed_first = verifier.round(EF::ONE, EF::TWO, None).unwrap();
        let replayed_second = verifier.round(EF::TWO, EF::ONE, None).unwrap();
        verifier.finish();

        assert_eq!(first, replayed_first);
        assert_eq!(second, replayed_second);

        // The sponge is handed back in one state, so the surrounding protocol stays in step.
        assert_eq!(
            CanSample::<F>::sample(&mut prover_challenger),
            CanSample::<F>::sample(&mut verifier_challenger)
        );
    }
}
