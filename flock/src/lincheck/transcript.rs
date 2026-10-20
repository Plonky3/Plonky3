//! Fiat-Shamir transcript of the lincheck, shaped by the batch.

use alloc::vec::Vec;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptError, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};

use crate::point::QuirkyPoint;
use crate::skip::{K_SKIP, SKIP_SIZE};
use crate::{Batch, Challenge, Val};

/// Protocol version bound into the seed.
const VERSION: u8 = 1;

/// Protocol name bound into the seed.
const NAME: &[u8] = b"p3-flock-lincheck";

/// Zerocheck point label.
const POINT: &str = "zerocheck_point";

/// Zerocheck values label.
const EVALS: &str = "zerocheck_evals";

/// Batching challenge label.
const ALPHA: &str = "alpha";

/// Round message label.
const ROUND: &str = "round";

/// Round challenge label.
const CHALLENGE: &str = "round_challenge";

/// Witness slices label.
const SLICES: &str = "slices";

/// Slice combination label.
const SKIP: &str = "skip";

/// Sponge alphabet.
type Alphabet = FieldUnit<Val>;

/// Codec from challenges to sponge elements.
type Codec = FieldToFieldCodec<Val>;

/// Everything that fixes the transcript of one lincheck.
#[derive(Clone, Debug)]
pub(crate) struct LincheckShape {
    /// Bits per instance, as a power of two.
    k_log: usize,
    /// Instances, as a power of two.
    log_instances: usize,
    /// The circuit's digest.
    statement: [u8; 32],
}

impl LincheckShape {
    /// The shape of a batch's lincheck.
    pub(crate) const fn new(batch: &Batch<'_>) -> Self {
        Self {
            k_log: batch.circuit().k_log(),
            log_instances: batch.log_instances(),
            statement: *batch.circuit().digest(),
        }
    }

    /// Product-sumcheck rounds: the inner variables past the skip.
    pub(crate) const fn rounds(&self) -> usize {
        self.k_log - K_SKIP
    }

    /// Interaction pattern of this shape.
    fn pattern(&self) -> InteractionPattern {
        let algebra = |kind, label, length| {
            Interaction::algebra::<Val, Challenge>(Hierarchy::Atomic, kind, label, length)
        };
        let mut steps = Vec::with_capacity(2 * self.rounds() + 5);
        // The input claims are bound before any draw.
        steps.push(algebra(
            Kind::Message,
            POINT,
            Length::Fixed(1 + self.k_log + self.log_instances - K_SKIP),
        ));
        steps.push(algebra(Kind::Message, EVALS, Length::Fixed(3)));
        steps.push(algebra(Kind::Challenge, ALPHA, Length::Scalar));
        // Each round sends q(0) and q(inf), then draws its challenge.
        for _ in 0..self.rounds() {
            steps.push(algebra(Kind::Message, ROUND, Length::Fixed(2)));
            steps.push(algebra(Kind::Challenge, CHALLENGE, Length::Scalar));
        }
        // The slices close the sumcheck; one draw folds them into one claim.
        steps.push(algebra(Kind::Message, SLICES, Length::Fixed(SKIP_SIZE)));
        steps.push(algebra(Kind::Challenge, SKIP, Length::Scalar));
        InteractionPattern::new(steps).expect("a flat pattern is always well formed")
    }

    /// Bind the protocol identity, this shape and the circuit.
    fn domain_separator(&self) -> DomainSeparator<Alphabet> {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern());
        for value in [self.k_log, self.log_instances] {
            separator.instance(&(value as u64).to_be_bytes());
        }
        separator.instance(&self.statement);
        separator
    }
}

/// Prover side of the lincheck transcript, over a borrowed challenger.
pub(crate) struct ProverTranscript<'a, C> {
    /// Transcript driver.
    state: ProverState<&'a mut C, Alphabet>,
}

impl<'a, C: CanObserve<Val> + CanSample<Val>> ProverTranscript<'a, C> {
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: &LincheckShape) -> Self {
        Self {
            state: ProverState::new(challenger, &shape.domain_separator()),
        }
    }

    /// Bind the input claims, then draw the batching challenge.
    pub(crate) fn claims(&mut self, point: &QuirkyPoint, evals: [Challenge; 3]) -> Challenge {
        let coordinates: Vec<Challenge> = point.coordinates().collect();
        let _ = self
            .state
            .observe_extensions::<Val, Challenge, Codec>(POINT, &coordinates);
        let _ = self
            .state
            .observe_extensions::<Val, Challenge, Codec>(EVALS, &evals);
        self.state
            .challenge_extension::<Val, Challenge, Codec>(ALPHA)
            .into_inner()
    }

    /// Bind one round's `q(0)` and `q(inf)`, then draw its challenge.
    pub(crate) fn round(&mut self, message: [Challenge; 2]) -> Challenge {
        let _ = self
            .state
            .observe_extensions::<Val, Challenge, Codec>(ROUND, &message);
        self.state
            .challenge_extension::<Val, Challenge, Codec>(CHALLENGE)
            .into_inner()
    }

    /// Bind the slices, draw the combining coordinate, and close the transcript.
    ///
    /// # Panics
    ///
    /// - An earlier step was skipped.
    pub(crate) fn finish(mut self, slices: &[Challenge]) -> Challenge {
        let _ = self
            .state
            .observe_extensions::<Val, Challenge, Codec>(SLICES, slices);
        let skip = self
            .state
            .challenge_extension::<Val, Challenge, Codec>(SKIP)
            .into_inner();
        assert!(
            self.state.finalize().is_empty(),
            "the lincheck carries every value in its own proof"
        );
        skip
    }
}

/// Verifier side of the lincheck transcript, mirroring the prover.
pub(crate) struct VerifierTranscript<'a, C> {
    /// Transcript driver over an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet>,
}

impl<'a, C: CanObserve<Val> + CanSample<Val>> VerifierTranscript<'a, C> {
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: &LincheckShape) -> Self {
        Self {
            state: VerifierState::new(challenger, &shape.domain_separator(), &[]),
        }
    }

    /// Bind the input claims, then draw the batching challenge.
    pub(crate) fn claims(
        &mut self,
        point: &QuirkyPoint,
        evals: [Challenge; 3],
    ) -> Result<Challenge, TranscriptError> {
        let coordinates: Vec<Challenge> = point.coordinates().collect();
        self.state
            .observe_extensions::<Val, Challenge, Codec>(POINT, &coordinates)?;
        self.state
            .observe_extensions::<Val, Challenge, Codec>(EVALS, &evals)?;
        Ok(self
            .state
            .challenge_extension::<Val, Challenge, Codec>(ALPHA)
            .into_inner())
    }

    /// Bind one round's `q(0)` and `q(inf)`, then draw its challenge.
    ///
    /// The message is a fixed pair, so its length always matches the pattern.
    pub(crate) fn round(&mut self, message: [Challenge; 2]) -> Challenge {
        let _ = self
            .state
            .observe_extensions::<Val, Challenge, Codec>(ROUND, &message);
        self.state
            .challenge_extension::<Val, Challenge, Codec>(CHALLENGE)
            .into_inner()
    }

    /// Bind the slices and draw the combining coordinate, the transcript staying open.
    pub(crate) fn slices(&mut self, slices: &[Challenge]) -> Result<Challenge, TranscriptError> {
        self.state
            .observe_extensions::<Val, Challenge, Codec>(SLICES, slices)?;
        Ok(self
            .state
            .challenge_extension::<Val, Challenge, Codec>(SKIP)
            .into_inner())
    }

    /// Close the transcript.
    ///
    /// # Panics
    ///
    /// - An earlier step was skipped; an empty wire never leaves bytes over.
    pub(crate) fn finish(self) {
        self.state
            .finalize()
            .expect("the lincheck reads an empty wire");
    }

    /// Release the completeness check before rejecting.
    pub(crate) fn abort(&mut self) {
        self.state.abort();
    }
}
