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
    Kind, Length, ProverState, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};

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

/// Describe the transcript of one sumcheck run.
///
/// # Arguments
///
/// - `num_rounds`: number of variables the run binds.
/// - `degree`: per-variable degree, and so the evaluation count per round.
/// - `pow_bits`: grinding difficulty per round, or zero to omit grinding.
///
/// # Panics
///
/// Never in practice.
/// A flat sequence of leaf steps always passes structural validation.
#[must_use]
pub fn pattern<F, EF>(num_rounds: usize, degree: usize, pow_bits: usize) -> InteractionPattern
where
    F: PrimeField64,
    EF: ExtensionField<F>,
{
    // One step for the claimed sum, then up to three per round.
    let mut steps = Vec::with_capacity(1 + 3 * num_rounds);

    // The claimed sum comes first, so every challenge depends on the statement.
    steps.push(Interaction::algebra::<F, EF>(
        Hierarchy::Atomic,
        Kind::Message,
        CLAIMED_SUM,
        Length::Scalar,
    ));

    for _ in 0..num_rounds {
        // The round polynomial is one step carrying `degree` evaluations.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            ROUND_POLY,
            Length::Fixed(degree),
        ));

        // Grinding sits between the polynomial and the challenge it protects.
        //
        // The difficulty travels inside the step.
        // A verifier expecting a cheaper grind therefore fails the shape check.
        if pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                ROUND_POW,
                Length::Fixed(pow_bits),
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
pub fn domain_separator<F, EF>(
    num_rounds: usize,
    degree: usize,
    pow_bits: usize,
) -> DomainSeparator<Alphabet<F>>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
{
    DomainSeparator::new(
        VERSION,
        NAME,
        pattern::<F, EF>(num_rounds, degree, pow_bits),
    )
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
pub struct ProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Grinding difficulty per round, or zero to omit grinding.
    pow_bits: usize,
    /// Marker for the extension field the rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript and bind the claimed sum.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
    /// - `num_rounds`: number of variables the run binds.
    /// - `degree`: per-variable degree.
    /// - `pow_bits`: grinding difficulty per round, or zero.
    /// - `claimed_sum`: value claimed for the sum over the cube.
    pub fn new(
        challenger: &'a mut C,
        num_rounds: usize,
        degree: usize,
        pow_bits: usize,
        claimed_sum: EF,
    ) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = domain_separator::<F, EF>(num_rounds, degree, pow_bits);
        let mut state = ProverState::new(challenger, &separator);

        // The claimed sum is prover-chosen and travels in the proof.
        // So it is absorbed here, not written into the driver's own buffer.
        state.observe_extension::<F, EF, FieldToFieldCodec<F>>(CLAIMED_SUM, &claimed_sum);

        Self {
            state,
            pow_bits,
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
        let witness = (self.pow_bits > 0).then(|| self.state.observe_pow(ROUND_POW, self.pow_bits));

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
pub struct VerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Grinding difficulty per round, or zero to omit grinding.
    pow_bits: usize,
    /// Index of the next round to play, used to place a grinding failure.
    round: usize,
    /// Marker for the extension field the rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript and bind the claimed sum.
    ///
    /// The arguments match the prover's, so both sides seed identically.
    pub fn new(
        challenger: &'a mut C,
        num_rounds: usize,
        degree: usize,
        pow_bits: usize,
        claimed_sum: EF,
    ) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = domain_separator::<F, EF>(num_rounds, degree, pow_bits);
        let mut state = VerifierState::new(challenger, &separator, &[]);

        // Absorbed from the proof, exactly as the prover absorbed it.
        state.observe_extension::<F, EF, FieldToFieldCodec<F>>(CLAIMED_SUM, &claimed_sum);

        Self {
            state,
            pow_bits,
            round: 0,
            _ef: PhantomData,
        }
    }

    /// Replay one round: bind the polynomial, re-check the grind, draw the challenge.
    ///
    /// # Errors
    ///
    /// When the supplied witness misses the required difficulty.
    ///
    /// # Panics
    ///
    /// When the evaluation count differs from the described one.
    /// The caller checks that first and rejects a mismatch with an error.
    pub fn round(&mut self, evals: &[EF], witness: Option<F>) -> Result<EF, GenericDegreeError> {
        let round = self.round;
        self.round += 1;

        // Bind the polynomial before the challenge that will be evaluated on it.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, evals);

        // Re-run the prover's grinding step on the witness it committed to.
        if self.pow_bits > 0 {
            let witness = witness.expect("a positive difficulty requires a witness per round");
            self.state
                .observe_pow(ROUND_POW, self.pow_bits, witness)
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
