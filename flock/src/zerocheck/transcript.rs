//! Fiat-Shamir transcript of one circuit batch's zerocheck, shaped by the statement alone.

use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, SymmetricSteps, TranscriptBound, TranscriptError, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};

use super::fixed::N_FIXED;
use crate::Batch;
use crate::skip::{K_SKIP, SKIP_SIZE};

/// Protocol version bound into the seed.
const VERSION: u8 = 1;

/// Protocol name bound into the seed.
const NAME: &[u8] = b"p3-flock-zerocheck";

/// Drawn eq coordinates label.
const OUTER: &str = "outer";

/// Round-1 message label.
const ROUND1: &str = "round1";

/// Round-1 grinding label.
const ROUND1_POW: &str = "round1_pow";

/// Skip challenge label.
const SKIP: &str = "skip";

/// Multilinear round message label.
const ROUND: &str = "round";

/// Multilinear round grinding label.
const ROUND_POW: &str = "round_pow";

/// Multilinear round challenge label.
const CHALLENGE: &str = "challenge";

/// Final evaluations label.
const EVALS: &str = "evals";

/// Base-field sponge alphabet.
type Alphabet = FieldUnit<Poly64>;

/// Everything that fixes the transcript of one zerocheck.
#[derive(Clone, Debug)]
pub(crate) struct ZerocheckShape {
    /// Total variables: instance bits and slot bits.
    pub(crate) log_n: usize,
    /// Instances, as a power of two.
    pub(crate) log_instances: usize,
    /// Grinding bits before every challenge that follows a message.
    pub(crate) pow_bits: usize,
    /// The circuit's digest.
    pub(crate) statement: [u8; 32],
}

impl ZerocheckShape {
    /// The shape of a batch's zerocheck.
    pub(crate) const fn new(batch: &Batch<'_>, pow_bits: usize) -> Self {
        Self {
            log_n: batch.log_n(),
            log_instances: batch.log_instances(),
            pow_bits,
            statement: *batch.circuit().digest(),
        }
    }

    /// Multilinear rounds after the skip.
    pub(crate) const fn n_rounds(&self) -> usize {
        self.log_n - K_SKIP
    }

    /// Drawn eq coordinates past the fixed ones.
    pub(crate) const fn n_outer(&self) -> usize {
        self.n_rounds() - N_FIXED
    }

    /// Interaction pattern of this shape.
    fn pattern(&self) -> InteractionPattern {
        let algebra = |kind, label, length| {
            Interaction::algebra::<Poly64, Poly192>(Hierarchy::Atomic, kind, label, length)
        };
        let pow = || {
            Interaction::algebra::<Poly64, Poly64>(
                Hierarchy::Atomic,
                Kind::Pow,
                ROUND_POW,
                Length::Fixed(self.pow_bits),
            )
        };
        let mut steps = Vec::with_capacity(3 * self.n_rounds() + 6);

        // The drawn coordinates come first, so every message depends on them.
        if self.n_outer() > 0 {
            steps.push(algebra(
                Kind::Challenge,
                OUTER,
                Length::Fixed(self.n_outer()),
            ));
        }

        // Round 1, then the skip challenge, grinding between them.
        steps.push(algebra(Kind::Message, ROUND1, Length::Fixed(SKIP_SIZE)));
        if self.pow_bits > 0 {
            steps.push(Interaction::algebra::<Poly64, Poly64>(
                Hierarchy::Atomic,
                Kind::Pow,
                ROUND1_POW,
                Length::Fixed(self.pow_bits),
            ));
        }
        steps.push(algebra(Kind::Challenge, SKIP, Length::Scalar));

        // One message and one challenge per multilinear round.
        for _ in 0..self.n_rounds() {
            steps.push(algebra(Kind::Message, ROUND, Length::Fixed(2)));
            if self.pow_bits > 0 {
                steps.push(pow());
            }
            steps.push(algebra(Kind::Challenge, CHALLENGE, Length::Scalar));
        }

        // The three evaluations, bound before whatever consumes them.
        steps.push(algebra(Kind::Message, EVALS, Length::Fixed(3)));
        InteractionPattern::new(steps).expect("a flat pattern is always well formed")
    }

    /// Bind the protocol identity, this shape, and the circuit.
    fn domain_separator(&self) -> DomainSeparator<Alphabet> {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern());
        for value in [self.log_n, self.log_instances, K_SKIP, self.pow_bits] {
            separator.instance(&(value as u64).to_be_bytes());
        }
        separator.instance(&self.statement);
        separator
    }
}

/// Prover side of the zerocheck transcript, over a borrowed challenger.
pub(crate) struct ProverTranscript<'a, C> {
    /// Transcript driver.
    state: ProverState<&'a mut C, Alphabet>,
    /// The shape.
    shape: ZerocheckShape,
}

impl<'a, C> ProverTranscript<'a, C>
where
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: ZerocheckShape) -> Self {
        let separator = shape.domain_separator();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
        }
    }

    /// Draw the eq coordinates past the fixed ones.
    pub(crate) fn outer(&mut self) -> Vec<Poly192> {
        play_outer(&mut self.state, &self.shape)
    }

    /// Bind round 1, grind, and draw the skip challenge.
    pub(crate) fn round1(&mut self, message: &[Poly192]) -> (Poly192, Option<Poly64>) {
        let _ = self
            .state
            .observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(ROUND1, message);
        let pow = (self.shape.pow_bits > 0)
            .then(|| self.state.observe_pow(ROUND1_POW, self.shape.pow_bits));
        let z = self
            .state
            .challenge_extension::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(SKIP)
            .into_inner();
        (z, pow)
    }

    /// Bind one round's coefficients, grind, and draw its challenge.
    pub(crate) fn round(&mut self, coeffs: &[Poly192; 2]) -> (Poly192, Option<Poly64>) {
        let _ = self
            .state
            .observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(ROUND, coeffs);
        let pow = (self.shape.pow_bits > 0)
            .then(|| self.state.observe_pow(ROUND_POW, self.shape.pow_bits));
        let chi = self
            .state
            .challenge_extension::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(CHALLENGE)
            .into_inner();
        (chi, pow)
    }

    /// Release the completeness check, for a transcript left unfinished.
    #[cfg(test)]
    pub(crate) fn abort(&mut self) {
        self.state.abort();
    }

    /// Bind the evaluations and close the transcript.
    ///
    /// # Panics
    ///
    /// - An earlier step was skipped.
    pub(crate) fn finish(mut self, evals: &[Poly192; 3]) {
        let _ = self
            .state
            .observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(EVALS, evals);
        assert!(
            self.state.finalize().is_empty(),
            "the zerocheck carries every value in its own proof"
        );
    }
}

/// Verifier side of the zerocheck transcript, mirroring the prover.
pub(crate) struct VerifierTranscript<'a, C> {
    /// Transcript driver over an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet>,
    /// The shape.
    shape: ZerocheckShape,
}

impl<'a, C> VerifierTranscript<'a, C>
where
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: ZerocheckShape) -> Self {
        let separator = shape.domain_separator();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
        }
    }

    /// Draw the eq coordinates past the fixed ones.
    pub(crate) fn outer(&mut self) -> Vec<Poly192> {
        play_outer(&mut self.state, &self.shape)
    }

    /// Bind round 1, check its grinding, and draw the skip challenge.
    pub(crate) fn round1(
        &mut self,
        message: &[Poly192],
        pow: Option<Poly64>,
    ) -> Result<Poly192, TranscriptError> {
        self.state
            .observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(ROUND1, message)?;
        self.pow(ROUND1_POW, pow)?;
        Ok(self
            .state
            .challenge_extension::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(SKIP)
            .into_inner())
    }

    /// Bind one round's coefficients, check its grinding, and draw its challenge.
    pub(crate) fn round(
        &mut self,
        coeffs: &[Poly192; 2],
        pow: Option<Poly64>,
    ) -> Result<Poly192, TranscriptError> {
        self.state
            .observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(ROUND, coeffs)?;
        self.pow(ROUND_POW, pow)?;
        Ok(self
            .state
            .challenge_extension::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(CHALLENGE)
            .into_inner())
    }

    /// Check a grinding witness, present exactly when grinding is on.
    fn pow(&mut self, label: &'static str, pow: Option<Poly64>) -> Result<(), TranscriptError> {
        match (self.shape.pow_bits, pow) {
            (0, None) => Ok(()),
            (bits, Some(witness)) if bits > 0 => self.state.observe_pow(label, bits, witness),
            _ => {
                self.state.abort();
                Err(TranscriptError::BadProofShape {
                    reason: "a grinding witness is present exactly when grinding is on",
                })
            }
        }
    }

    /// Bind the evaluations and close the transcript.
    ///
    /// # Panics
    ///
    /// - An earlier step was skipped; an empty wire never leaves bytes over.
    pub(crate) fn finish(mut self, evals: &[Poly192; 3]) {
        let _ = self
            .state
            .observe_extensions::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(EVALS, evals);
        self.state
            .finalize()
            .expect("the zerocheck reads an empty wire");
    }

    /// Release the completeness check before rejecting.
    pub(crate) fn abort(&mut self) {
        self.state.abort();
    }
}

/// Draw the eq coordinates past the fixed ones on either side; none past the fixed ones draws nothing.
fn play_outer<S>(state: &mut S, shape: &ZerocheckShape) -> Vec<Poly192>
where
    S: SymmetricSteps,
    S::Challenger: CanObserve<Poly64> + CanSample<Poly64>,
{
    if shape.n_outer() == 0 {
        return Vec::new();
    }
    // Accepting every candidate draws exactly the plain list.
    state
        .challenge_extensions_rejecting::<Poly64, Poly192, FieldToFieldCodec<Poly64>>(
            OUTER,
            shape.n_outer(),
            |_, _| true,
        )
        .into_iter()
        .map(TranscriptBound::into_inner)
        .collect()
}
