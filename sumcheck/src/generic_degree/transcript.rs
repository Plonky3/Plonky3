//! Fiat-Shamir transcript of the generic-degree sumcheck.
//!
//! # Overview
//!
//! This module is the single statement of what this protocol's transcript is.
//! Both sides consume it through the same three calls.
//!
//! The description itself is a value, built from the numbers that shape a run.
//!
//! # Shape
//!
//! ```text
//!     claimed sum              1 extension element
//!     round 0:  polynomial     degree extension elements
//!               grinding       present only when the difficulty is positive
//!               challenge      1 extension element
//!     round 1:  ...
//! ```
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//!
//! The round count, the degree and the difficulty each change the description.
//! All three are therefore bound.
//! Two runs differing in any of them cannot share a transcript.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::ExtensionField;

use super::error::GenericDegreeError;

/// Version byte bound into the transcript seed.
///
/// Bumping it separates two revisions of this protocol.
/// It does so even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-sumcheck-generic-degree";

/// Step label of the claimed sum.
const CLAIMED_SUM: &str = "claimed_sum";

/// Step label of a round polynomial.
const ROUND_POLY: &str = "round_poly";

/// Step label of a per-round grinding step.
const ROUND_POW: &str = "round_pow";

/// Step label of a per-round challenge.
const ROUND_CHALLENGE: &str = "round_challenge";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Numbers that fix the transcript of one generic-degree sumcheck run.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenericDegreeShape {
    /// Number of variables the run binds, and so the number of rounds.
    pub num_rounds: usize,
    /// Per-variable degree, and so the evaluation count of every round polynomial.
    pub degree: usize,
    /// Grinding difficulty guarding each round's challenge, or zero to omit grinding.
    pub pow_bits: usize,
}

impl GenericDegreeShape {
    /// Collect the numbers that fix one run.
    ///
    /// # Arguments
    ///
    /// - `num_rounds`: number of variables the run binds.
    /// - `degree`: per-variable degree, and so the evaluation count per round.
    /// - `pow_bits`: grinding difficulty per round, or zero to omit grinding.
    #[must_use]
    pub const fn new(num_rounds: usize, degree: usize, pow_bits: usize) -> Self {
        Self {
            num_rounds,
            degree,
            pow_bits,
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
        // One step for the claimed sum, then up to three per round.
        let mut steps = Vec::with_capacity(1 + 3 * self.num_rounds);

        // The claimed sum comes first, so every challenge depends on the statement.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            CLAIMED_SUM,
            Length::Scalar,
        ));

        for _ in 0..self.num_rounds {
            // The round polynomial is one step carrying `degree` evaluations.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                ROUND_POLY,
                Length::Fixed(self.degree),
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

    /// Bind the protocol identity and the transcript shape into a seed.
    ///
    /// Every number that shapes a run also shapes the description.
    /// The fingerprint of the description therefore covers all of them.
    /// No separate instance label is needed.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>())
    }
}

/// Prover-side transcript of one sumcheck run.
///
/// # Overview
///
/// Holds the only definition of what a prover writes per round.
///
/// Every prover of one of these proofs drives it through this type.
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
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: GenericDegreeShape,
    /// Marker for the extension field the rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript and bind the claimed sum.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
    /// - `shape`: the numbers that fix this run's transcript.
    /// - `claimed_sum`: value claimed for the sum over the cube.
    pub fn new(challenger: &'a mut C, shape: GenericDegreeShape, claimed_sum: EF) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        let mut state = ProverState::new(challenger, &separator);

        // The claimed sum is prover-chosen and travels in the proof.
        // So it is absorbed here, not written into the driver's own buffer.
        state.observe_extension::<F, EF, FieldToFieldCodec<F>>(CLAIMED_SUM, &claimed_sum);

        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Play one round: bind the polynomial, grind, and draw the challenge.
    ///
    /// # Returns
    ///
    /// - The challenge that binds this round's variable.
    /// - The grinding witness, when the difficulty is positive.
    ///
    /// The caller stores both in its own proof.
    pub fn round(&mut self, evals: &[EF]) -> (EF, Option<F>) {
        // Bind the polynomial before the challenge that will be evaluated on it.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, evals);

        // Grinding raises the cost of searching for a favourable challenge.
        let witness = (self.shape.pow_bits > 0)
            .then(|| self.state.observe_pow(ROUND_POW, self.shape.pow_bits));

        // Draw the challenge for the caller to fold with.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner();

        (challenge, witness)
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer rounds were played than the run was described with.
    pub fn finish(self) {
        // Nothing was written to the driver's own buffer.
        // Closing is therefore purely the check that the description was consumed.
        assert!(
            self.state.finalize().is_empty(),
            "the generic-degree sumcheck carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one sumcheck run.
///
/// Mirrors the prover side call for call, over the same description.
///
/// The values come from the proof rather than from a wire, so the caller must
/// have checked their lengths against the described shape first.
pub struct VerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: GenericDegreeShape,
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
    /// Seed the transcript and bind the claimed sum.
    ///
    /// The arguments match the prover's, so both sides seed identically.
    pub fn new(challenger: &'a mut C, shape: GenericDegreeShape, claimed_sum: EF) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        let mut state = VerifierState::new(challenger, &separator, &[]);

        // Absorbed from the proof, exactly as the prover absorbed it.
        state.observe_extension::<F, EF, FieldToFieldCodec<F>>(CLAIMED_SUM, &claimed_sum);

        Self {
            state,
            shape,
            round: 0,
            _ef: PhantomData,
        }
    }

    /// Replay one round: bind the polynomial, re-check the grind, draw the challenge.
    ///
    /// Every value comes from the proof, so every disagreement is a rejection.
    ///
    /// # Errors
    ///
    /// - The evaluation count differs from the described one.
    /// - Grinding is enabled and the round carries no witness.
    /// - The witness misses the required difficulty.
    pub fn round(&mut self, evals: &[EF], witness: Option<F>) -> Result<EF, GenericDegreeError> {
        let round = self.round;
        self.round += 1;

        // Bind the polynomial before the challenge that will be evaluated on it.
        //
        // The count comes from the proof, so a mismatch is a rejection.
        // The driver poisons itself on the way out, which releases its own drop check.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, evals)
            .map_err(|_| GenericDegreeError::PolyEvalCountMismatch {
                round,
                expected: self.shape.degree,
                actual: evals.len(),
            })?;

        // Re-run the prover's grinding step on the witness it committed to.
        if self.shape.pow_bits > 0 {
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                self.state.abort();
                return Err(GenericDegreeError::MissingPowWitness { round });
            };
            self.state
                .observe_pow(ROUND_POW, self.shape.pow_bits, witness)
                .map_err(|_| GenericDegreeError::InvalidPowWitness { round })?;
        }

        // Draw the same challenge the prover saw.
        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner())
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer rounds were played than the run was described with.
    pub fn finish(self) {
        // The proof carries every value, so no unread wire bytes can remain.
        self.state
            .finalize()
            .expect("the generic-degree sumcheck reads an empty wire");
    }
}

#[cfg(test)]
mod tests {
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
        // Fixed seed so a transcript built here is reproducible.
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        let perm = Perm::new_from_rng_128(&mut rng);
        Ch::new(perm)
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: GenericDegreeShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    #[test]
    fn every_number_that_shapes_a_run_reaches_the_seed() {
        // Fixture state: 4 rounds, degree 3, no grinding.
        let base = first_challenge(GenericDegreeShape::new(4, 3, 0));

        // One more round appends three more steps.
        assert_ne!(base, first_challenge(GenericDegreeShape::new(5, 3, 0)));

        // A wider round polynomial changes each round step's declared width.
        assert_ne!(base, first_challenge(GenericDegreeShape::new(4, 4, 0)));

        // Enabling grinding inserts a step per round.
        assert_ne!(base, first_challenge(GenericDegreeShape::new(4, 3, 8)));

        // Two positive difficulties differ only inside the grinding steps.
        assert_ne!(
            first_challenge(GenericDegreeShape::new(4, 3, 8)),
            first_challenge(GenericDegreeShape::new(4, 3, 9)),
        );
    }

    // These two tests drive the verifier transcript the way a downstream crate
    // would: straight from proof fields, with no shape pre-check in front.
    //
    // Both malformed inputs must return an error.
    // A panic here would compound with the driver's own drop check and abort.

    #[test]
    fn a_round_polynomial_of_the_wrong_width_is_rejected() {
        // Described run: 2 rounds, 3 evaluations per round polynomial, no grinding.
        //
        //     described:    Fixed(3)
        //     proof holds:  2        -> rejected on round 0
        let mut challenger = fresh_challenger();
        let shape = GenericDegreeShape::new(2, 3, 0);
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape, EF::ZERO);

        let err = transcript
            .round(&[EF::ONE, EF::ONE], None)
            .expect_err("a round polynomial outside the described width must error");

        assert_eq!(
            err,
            GenericDegreeError::PolyEvalCountMismatch {
                round: 0,
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn a_round_missing_its_grinding_witness_is_rejected() {
        // Described run: 2 rounds, 3 evaluations, 4 bits of grinding per round.
        //
        // A described grinding step cannot be replayed with no witness to feed it.
        let mut challenger = fresh_challenger();
        let shape = GenericDegreeShape::new(2, 3, 4);
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape, EF::ZERO);

        let err = transcript
            .round(&[EF::ONE; 3], None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(err, GenericDegreeError::MissingPowWitness { round: 0 });
    }
}
