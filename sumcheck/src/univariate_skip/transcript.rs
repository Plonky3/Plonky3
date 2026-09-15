//! Fiat-Shamir transcript of a univariate-skip round and the sumcheck it delegates to.
//!
//! # Shape
//!
//! ```text
//!     round message       one extension element per transmitted point
//!     grinding            present only when the difficulty is positive
//!     skip challenge      1 extension element
//!     residual sumcheck   a bracketed sub-protocol under its own seed
//! ```
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//!
//! The skipped width, the transmitted width and the difficulty each change the description.
//!
//! The residual round count and degree do not appear here.
//!
//! They shape the nested description instead, which seeds itself inside the bracket.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::ExtensionField;

use super::opening::OPENING_DEGREE;
use crate::generic_degree::GenericDegreeShape;

/// Version byte bound into the transcript seed.
///
/// Bumping it separates two revisions of this protocol even when their steps agree.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
pub(crate) const NAME: &[u8] = b"p3-sumcheck-univariate-skip";

/// Step label of the transmitted round polynomial.
const ROUND_MESSAGE: &str = "round_message";

/// Step label of the grinding step guarding the skip challenge.
const ROUND_POW: &str = "round_pow";

/// Step label of the challenge the skipped variables collapse to.
const SKIP_CHALLENGE: &str = "skip_challenge";

/// Step label of the delegated sumcheck over the variables the round did not bind.
const RESIDUAL_SUMCHECK: &str = "residual_sumcheck";

/// Marker recorded on the bracket around the delegated sumcheck.
///
/// It is a local diagnostic and does not reach the pattern fingerprint.
struct ResidualSumcheck;

/// Protocol name bound into the opening reduction's seed.
const OPENING_NAME: &[u8] = b"p3-sumcheck-univariate-skip-opening";

/// Step label of the challenge that batches the committed polynomials.
const OPENING_BATCHING: &str = "opening_batching";

/// Step label of the sumcheck the opening reduction delegates to.
const OPENING_SUMCHECK: &str = "opening_sumcheck";

/// Marker recorded on the bracket around the opening reduction's sumcheck.
struct OpeningSumcheck;

/// Numbers that fix the transcript of one opening reduction.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkipOpeningShape {
    /// Number of skipped variables the reduction binds, one round each.
    pub num_variables: usize,
    /// Number of committed polynomials batched into one run.
    pub num_polynomials: usize,
    /// Grinding difficulty guarding each sumcheck challenge, or zero to omit grinding.
    pub pow_bits: usize,
}

impl SkipOpeningShape {
    /// Collect the numbers that fix one reduction.
    #[must_use]
    pub const fn new(num_variables: usize, num_polynomials: usize, pow_bits: usize) -> Self {
        Self {
            num_variables,
            num_polynomials,
            pow_bits,
        }
    }

    /// The description the delegated sumcheck seeds itself from.
    ///
    /// Both sides derive it here, so neither can drift on the nested run.
    #[must_use]
    pub const fn sumcheck_shape(&self) -> GenericDegreeShape {
        GenericDegreeShape::new(self.num_variables, OPENING_DEGREE, self.pow_bits)
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice, since one matched bracket always validates.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // One challenge, then the bracket the rounds run inside.
        //
        // A single-polynomial run has no batching step at all.
        //
        // The description therefore differs, so one cannot be replayed as the other.
        let mut steps = Vec::with_capacity(3);

        if self.num_polynomials > 1 {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OPENING_BATCHING,
                Length::Scalar,
            ));
        }

        steps.push(Interaction::marker::<OpeningSumcheck>(
            Hierarchy::Begin,
            Kind::Protocol,
            OPENING_SUMCHECK,
        ));
        steps.push(Interaction::marker::<OpeningSumcheck>(
            Hierarchy::End,
            Kind::Protocol,
            OPENING_SUMCHECK,
        ));

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and the transcript shape into a seed.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, OPENING_NAME, self.pattern::<F, EF>())
    }
}

/// Prover-side transcript of one opening reduction.
pub struct SkipOpeningProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: SkipOpeningShape,
    /// Marker for the extension field the challenge carries.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> SkipOpeningProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape both sides agreed on.
    pub fn new(challenger: &'a mut C, shape: SkipOpeningShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let state = ProverState::new(challenger, &separator);
        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the challenge that separates the committed polynomials.
    ///
    /// A single-polynomial run has nothing to separate, so it draws nothing and batches by one.
    pub fn batching_challenge(&mut self) -> EF {
        if self.shape.num_polynomials <= 1 {
            return EF::ONE;
        }
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OPENING_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the reduction's sumcheck, bracketed as a sub-protocol.
    pub fn sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<OpeningSumcheck>(OPENING_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<OpeningSumcheck>(OPENING_SUMCHECK);
        output
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the run was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the opening reduction carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one opening reduction.
///
/// Mirrors the prover side call for call, over the same description.
pub struct SkipOpeningVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: SkipOpeningShape,
    /// Marker for the extension field the challenge carries.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> SkipOpeningVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape both sides agreed on.
    pub fn new(challenger: &'a mut C, shape: SkipOpeningShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let state = VerifierState::new(challenger, &separator, &[]);
        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the same batching challenge the prover saw.
    pub fn batching_challenge(&mut self) -> EF {
        if self.shape.num_polynomials <= 1 {
            return EF::ONE;
        }
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OPENING_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the reduction's sumcheck, bracketed as a sub-protocol.
    ///
    /// A delegated rejection releases the driver.
    ///
    /// Without that, dropping this one after a malformed proof panics.
    ///
    /// # Errors
    ///
    /// Returns whatever the delegated run rejected with.
    pub fn sumcheck<T, E>(&mut self, run: impl FnOnce(&mut C) -> Result<T, E>) -> Result<T, E> {
        self.state
            .begin_protocol::<OpeningSumcheck>(OPENING_SUMCHECK);
        let output = run(self.state.challenger_mut());

        if output.is_err() {
            self.state.abort();
            return output;
        }

        self.state.end_protocol::<OpeningSumcheck>(OPENING_SUMCHECK);
        output
    }

    /// Release the completeness check after a rejection outside this driver.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the run was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the opening reduction reads an empty wire");
    }
}

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Numbers that fix the transcript of one univariate-skip reduction.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnivariateSkipShape {
    /// Dimension of the subspace the round polynomial vanishes on.
    ///
    /// This is the number of variables the round binds in one go.
    pub log_size: usize,
    /// Dimension of the subspace the round polynomial is transmitted on.
    ///
    /// The difference from the smaller dimension fixes the message width.
    pub log_extended: usize,
    /// Number of variables the delegated sumcheck binds one at a time.
    pub num_residual_rounds: usize,
    /// Per-variable degree of the delegated sumcheck's summand.
    pub residual_degree: usize,
    /// Grinding difficulty guarding each challenge, or zero to omit grinding.
    pub pow_bits: usize,
}

impl UnivariateSkipShape {
    /// Collect the numbers that fix one reduction.
    #[must_use]
    pub const fn new(
        log_size: usize,
        log_extended: usize,
        num_residual_rounds: usize,
        residual_degree: usize,
        pow_bits: usize,
    ) -> Self {
        Self {
            log_size,
            log_extended,
            num_residual_rounds,
            residual_degree,
            pow_bits,
        }
    }

    /// Number of extension elements the round message carries.
    #[must_use]
    pub const fn message_len(&self) -> usize {
        (1 << self.log_extended) - (1 << self.log_size)
    }

    /// The description the delegated sumcheck seeds itself from.
    ///
    /// Both sides derive it here, so neither can drift from the other on the nested run.
    #[must_use]
    pub const fn residual_shape(&self) -> GenericDegreeShape {
        GenericDegreeShape::new(
            self.num_residual_rounds,
            self.residual_degree,
            self.pow_bits,
        )
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice, since one matched bracket always validates.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Up to three leaf steps, then one matched bracket.
        let mut steps = Vec::with_capacity(5);

        // The message comes first, so the challenge depends on every value it carries.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            ROUND_MESSAGE,
            Length::Fixed(self.message_len()),
        ));

        // Grinding sits between the message and the challenge it protects.
        //
        // The difficulty travels inside the step.
        //
        // A verifier expecting a cheaper grind therefore fails the shape check.
        if self.pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                ROUND_POW,
                Length::Fixed(self.pow_bits),
            ));
        }

        // One challenge collapses every skipped variable at once.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            SKIP_CHALLENGE,
            Length::Scalar,
        ));

        // The bracket records that a sub-protocol runs here.
        //
        // Its steps live in the callee's description, under the callee's own seed.
        steps.push(Interaction::marker::<ResidualSumcheck>(
            Hierarchy::Begin,
            Kind::Protocol,
            RESIDUAL_SUMCHECK,
        ));
        steps.push(Interaction::marker::<ResidualSumcheck>(
            Hierarchy::End,
            Kind::Protocol,
            RESIDUAL_SUMCHECK,
        ));

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and the transcript shape into a seed.
    ///
    /// Every number that shapes this run also shapes the description.
    ///
    /// The fingerprint covers all of them, so no separate instance label is needed.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>())
    }
}

/// Prover-side transcript of one univariate-skip reduction.
///
/// # Borrowing
///
/// The challenger is borrowed, not consumed.
///
/// A skip round runs inside a larger protocol, whose own transcript continues where this stops.
pub struct UnivariateSkipProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: UnivariateSkipShape,
    /// Marker for the extension field the message and challenge carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> UnivariateSkipProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape both sides agreed on.
    pub fn new(challenger: &'a mut C, shape: UnivariateSkipShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        let state = ProverState::new(challenger, &separator);
        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Bind the round message, grind, and draw the challenge it collapses to.
    ///
    /// # Returns
    ///
    /// - The challenge the skipped variables collapse to.
    /// - The grinding witness, when the difficulty is positive.
    pub fn round_message(&mut self, message: &[EF]) -> (EF, Option<F>) {
        // Bind the whole message before the challenge that will be evaluated on it.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_MESSAGE, message);

        // Grinding raises the cost of searching for a favourable challenge.
        let witness = (self.shape.pow_bits > 0)
            .then(|| self.state.observe_pow(ROUND_POW, self.shape.pow_bits));

        // Draw the challenge the caller reads the message at.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(SKIP_CHALLENGE)
            .into_inner();

        (challenge, witness)
    }

    /// Lend the sponge to the residual sumcheck, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    pub fn residual_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<ResidualSumcheck>(RESIDUAL_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<ResidualSumcheck>(RESIDUAL_SUMCHECK);
        output
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the run was described with.
    pub fn finish(self) {
        // Nothing was written to the driver's own buffer.
        //
        // Closing is purely the check that the description was consumed.
        assert!(
            self.state.finalize().is_empty(),
            "the univariate-skip reduction carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one univariate-skip reduction.
///
/// Mirrors the prover side call for call, over the same description.
pub struct UnivariateSkipVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: UnivariateSkipShape,
    /// Marker for the extension field the message and challenge carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> UnivariateSkipVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape both sides agreed on.
    pub fn new(challenger: &'a mut C, shape: UnivariateSkipShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        let state = VerifierState::new(challenger, &separator, &[]);
        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay the round: bind the message, re-check the grind, draw the challenge.
    ///
    /// Every value comes from the proof, so every disagreement is a rejection.
    ///
    /// # Errors
    ///
    /// - The message width differs from the described one.
    /// - Grinding is enabled and the proof carries no witness.
    /// - The witness misses the required difficulty.
    /// - Grinding is off and the proof carries a witness anyway.
    pub fn round_message(
        &mut self,
        message: &[EF],
        witness: Option<F>,
    ) -> Result<EF, UnivariateSkipTranscriptError> {
        // Bind the whole message before the challenge that will be evaluated on it.
        //
        // The width comes from the proof, so a mismatch is a rejection.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_MESSAGE, message)
            .map_err(|_| UnivariateSkipTranscriptError::MessageLenMismatch {
                expected: self.shape.message_len(),
                actual: message.len(),
            })?;

        // Re-run the prover's grinding step on the witness it committed to.
        if self.shape.pow_bits > 0 {
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                self.state.abort();
                return Err(UnivariateSkipTranscriptError::MissingPowWitness);
            };
            self.state
                .observe_pow(ROUND_POW, self.shape.pow_bits, witness)
                .map_err(|_| UnivariateSkipTranscriptError::InvalidPowWitness)?;
        } else if witness.is_some() {
            // At zero difficulty the description has no grinding step to play.
            //
            // Ignoring a witness would leave one proof with two accepting forms.
            //
            // Refusing it is what keeps the shape canonical.
            self.state.abort();
            return Err(UnivariateSkipTranscriptError::UnexpectedPowWitness);
        }

        // Draw the same challenge the prover saw.
        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(SKIP_CHALLENGE)
            .into_inner())
    }

    /// Lend the sponge to the residual sumcheck, bracketed as a sub-protocol.
    ///
    /// The delegated run reports its own rejection, and a rejection ends this reduction too.
    ///
    /// Releasing the completeness check there is what makes a malformed proof a rejection.
    ///
    /// Without it, dropping this driver panics instead.
    /// Taking the failure rather than a plain value is what stops a caller forgetting to.
    ///
    /// # Errors
    ///
    /// Returns whatever the delegated run rejected with.
    pub fn residual_sumcheck<T, E>(
        &mut self,
        run: impl FnOnce(&mut C) -> Result<T, E>,
    ) -> Result<T, E> {
        self.state
            .begin_protocol::<ResidualSumcheck>(RESIDUAL_SUMCHECK);
        let output = run(self.state.challenger_mut());

        // A delegated rejection leaves the bracket half-played, so the driver is released here.
        if output.is_err() {
            self.state.abort();
            return output;
        }

        self.state
            .end_protocol::<ResidualSumcheck>(RESIDUAL_SUMCHECK);
        output
    }

    /// Release the completeness check after a rejection outside this driver.
    ///
    /// A caller that stops early still has to leave the driver in a droppable state.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the run was described with.
    pub fn finish(self) {
        // The proof carries every value, so no unread wire bytes can remain.
        self.state
            .finalize()
            .expect("the univariate-skip reduction reads an empty wire");
    }
}

/// Reasons the transcript replay rejects a proof.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UnivariateSkipTranscriptError {
    /// The round message carries a different number of values than described.
    #[error("round message length mismatch: expected {expected}, got {actual}")]
    MessageLenMismatch {
        /// Number of values the description fixes.
        expected: usize,
        /// Number of values the proof carries.
        actual: usize,
    },
    /// Grinding is enabled but the proof carries no witness for it.
    #[error("the round carries no grinding witness")]
    MissingPowWitness,
    /// Grinding is off but the proof carries a witness anyway.
    ///
    /// Accepting it would give one statement two accepting proofs.
    #[error("the round carries a grinding witness at zero difficulty")]
    UnexpectedPowWitness,
    /// The grinding witness does not meet the required difficulty.
    #[error("the round's grinding witness is invalid")]
    InvalidPowWitness,
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField128};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;

    use super::*;

    type F = BinaryField128;
    type Ch = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

    fn fresh_challenger() -> Ch {
        Ch::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: UnivariateSkipShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, F>().seed(&mut challenger);
        challenger.sample()
    }

    #[test]
    fn the_numbers_shaping_this_run_reach_its_own_seed() {
        // Fixture state: skip 6 of 10 variables, degree 3 residual, no grinding.
        let base = first_challenge(UnivariateSkipShape::new(6, 7, 4, 3, 0));

        // A wider transmitted domain changes the message step's declared width.
        assert_ne!(
            base,
            first_challenge(UnivariateSkipShape::new(6, 8, 4, 3, 0))
        );

        // A narrower skip changes the same width from the other side.
        assert_ne!(
            base,
            first_challenge(UnivariateSkipShape::new(5, 7, 4, 3, 0))
        );

        // Enabling grinding inserts a step.
        assert_ne!(
            base,
            first_challenge(UnivariateSkipShape::new(6, 7, 4, 3, 8))
        );

        // Two positive difficulties differ only inside the grinding step.
        assert_ne!(
            first_challenge(UnivariateSkipShape::new(6, 7, 4, 3, 8)),
            first_challenge(UnivariateSkipShape::new(6, 7, 4, 3, 9)),
        );
    }

    #[test]
    fn the_residual_numbers_are_bound_by_the_nested_seed() {
        // The bracket records only that a sub-protocol runs.
        //
        // The residual numbers therefore do not move this run's own fingerprint.
        //
        //     outer seed  : skip widths and grinding
        //     nested seed : residual rounds, residual degree, grinding
        //
        // They are bound when the nested run seeds itself inside the bracket.
        //
        // That is why both sides must derive that description from the same place.
        let base = UnivariateSkipShape::new(6, 7, 4, 3, 0);
        assert_eq!(
            first_challenge(base),
            first_challenge(UnivariateSkipShape::new(6, 7, 5, 4, 0))
        );

        // The nested description does separate them, so nothing is left unbound overall.
        let nested = |shape: UnivariateSkipShape| {
            let mut challenger = fresh_challenger();
            shape
                .residual_shape()
                .domain_separator::<F, F>()
                .seed(&mut challenger);
            CanSample::<F>::sample(&mut challenger)
        };
        assert_ne!(
            nested(base),
            nested(UnivariateSkipShape::new(6, 7, 5, 3, 0))
        );
        assert_ne!(
            nested(base),
            nested(UnivariateSkipShape::new(6, 7, 4, 4, 0))
        );
    }

    #[test]
    fn the_skip_transcript_does_not_collide_with_the_plain_sumcheck() {
        // Two protocols sharing a sponge must not share a seed.
        //
        // Otherwise a message bound in one could be replayed as a message in the other.
        let mut skip = fresh_challenger();
        UnivariateSkipShape::new(6, 7, 4, 3, 0)
            .domain_separator::<F, F>()
            .seed(&mut skip);

        let mut plain = fresh_challenger();
        crate::generic_degree::GenericDegreeShape::new(4, 3, 0)
            .domain_separator::<F, F>()
            .seed(&mut plain);

        assert_ne!(
            CanSample::<F>::sample(&mut skip),
            CanSample::<F>::sample(&mut plain)
        );
    }

    #[test]
    fn a_message_of_the_wrong_width_is_rejected() {
        // Described run: 64 transmitted values, no grinding.
        //
        //     described:    Fixed(64)
        //     proof holds:  2        -> rejected
        let mut challenger = fresh_challenger();
        let shape = UnivariateSkipShape::new(6, 7, 4, 3, 0);
        let mut transcript =
            UnivariateSkipVerifierTranscript::<Ch, F, F>::new(&mut challenger, shape);

        let err = transcript
            .round_message(&[F::ONE, F::ONE], None)
            .expect_err("a message outside the described width must error");

        assert_eq!(
            err,
            UnivariateSkipTranscriptError::MessageLenMismatch {
                expected: 64,
                actual: 2,
            }
        );

        // Absorbing the width error releases the driver's completeness check.
        drop(transcript);
    }

    #[test]
    fn a_round_missing_its_grinding_witness_is_rejected() {
        // A described grinding step cannot be replayed with no witness to feed it.
        let mut challenger = fresh_challenger();
        let shape = UnivariateSkipShape::new(3, 4, 2, 3, 4);
        let mut transcript =
            UnivariateSkipVerifierTranscript::<Ch, F, F>::new(&mut challenger, shape);

        let err = transcript
            .round_message(&[F::ONE; 8], None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(err, UnivariateSkipTranscriptError::MissingPowWitness);
    }

    #[test]
    fn the_message_width_follows_the_two_dimensions() {
        // The transmitted count is the extension size minus the subspace size.
        //
        //     2^7 - 2^6 = 64        one coset, the product-form shape
        //     2^8 - 2^6 = 192       three cosets, what a degree-three composition needs
        assert_eq!(UnivariateSkipShape::new(6, 7, 4, 3, 0).message_len(), 64);
        assert_eq!(UnivariateSkipShape::new(6, 8, 4, 3, 0).message_len(), 192);
    }
}
