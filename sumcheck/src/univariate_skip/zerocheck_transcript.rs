//! Fiat-Shamir transcript of a whole binary zerocheck.
//!
//! # Shape
//!
//! ```text
//!     zerocheck point     one extension element per residual variable
//!     round message       one extension element per transmitted point
//!     grinding            present only when the difficulty is positive
//!     skip challenge      1 extension element
//!     residual sumcheck   a bracketed sub-protocol under its own seed
//!     operand blends      one extension element per operand
//!     opening batching    1 extension element, absent for a single operand
//!     opening sumcheck    a bracketed sub-protocol under its own seed
//! ```
//!
//! # Why the point is drawn here
//!
//! The equality point comes after the commitment, and from the transcript.
//! Drawing it inside this description stops a caller drawing it early.
//! It stops a prover choosing it at all.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-sumcheck-binary-zerocheck";

/// Step label of the equality point the constraint is weighted by.
const ZEROCHECK_POINT: &str = "zerocheck_point";

/// Step label of the skip round's transmitted polynomial.
const ROUND_MESSAGE: &str = "round_message";

/// Step label of the grinding step guarding the skip challenge.
const ROUND_POW: &str = "round_pow";

/// Step label of the challenge the skipped variables collapse to.
const SKIP_CHALLENGE: &str = "skip_challenge";

/// Step label of the residual sumcheck.
const RESIDUAL_SUMCHECK: &str = "residual_sumcheck";

/// Step label of the operand values the residual rounds end on.
const OPERAND_BLENDS: &str = "operand_blends";

/// Step label of the challenge that batches the operands.
const OPENING_BATCHING: &str = "opening_batching";

/// Step label of the opening reduction's sumcheck.
const OPENING_SUMCHECK: &str = "opening_sumcheck";

/// Marker recorded on the bracket around the residual sumcheck.
struct Residual;

/// Marker recorded on the bracket around the opening reduction's sumcheck.
struct Opening;

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Numbers that fix the transcript of one binary zerocheck.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZerocheckShape {
    /// Total number of variables, rows and skipped together.
    pub log_height: usize,
    /// Number of variables the skip round binds in one go.
    pub log_skip: usize,
    /// Dimension of the subspace the round polynomial is transmitted on.
    pub log_extended: usize,
    /// Per-variable degree of the residual summand, equality weight included.
    pub residual_degree: usize,
    /// Number of operands the constraint reads.
    pub arity: usize,
    /// Grinding difficulty, or zero to omit it.
    /// It guards the skip challenge and every sumcheck round.
    ///
    /// The zerocheck point and the batching challenge carry no grinding step.
    pub pow_bits: usize,
}

impl ZerocheckShape {
    /// Collect the numbers that fix one run.
    #[must_use]
    pub const fn new(
        log_height: usize,
        log_skip: usize,
        log_extended: usize,
        residual_degree: usize,
        arity: usize,
        pow_bits: usize,
    ) -> Self {
        Self {
            log_height,
            log_skip,
            log_extended,
            residual_degree,
            arity,
            pow_bits,
        }
    }

    /// Number of variables the residual rounds bind one at a time.
    #[must_use]
    pub const fn log_rows(&self) -> usize {
        self.log_height.saturating_sub(self.log_skip)
    }

    /// Number of extension elements the round message carries.
    #[must_use]
    pub const fn message_len(&self) -> usize {
        (1 << self.log_extended) - (1 << self.log_skip)
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice, since two matched brackets always validate.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut steps = Vec::with_capacity(10);

        // The point comes first, so every later draw depends on the claim.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ZEROCHECK_POINT,
            Length::Fixed(self.log_rows()),
        ));

        // The whole message is bound before the challenge it will be read at.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            ROUND_MESSAGE,
            Length::Fixed(self.message_len()),
        ));

        // Grinding sits between the message and the challenge it protects.
        if self.pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                ROUND_POW,
                Length::Fixed(self.pow_bits),
            ));
        }

        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            SKIP_CHALLENGE,
            Length::Scalar,
        ));

        steps.push(Interaction::marker::<Residual>(
            Hierarchy::Begin,
            Kind::Protocol,
            RESIDUAL_SUMCHECK,
        ));
        steps.push(Interaction::marker::<Residual>(
            Hierarchy::End,
            Kind::Protocol,
            RESIDUAL_SUMCHECK,
        ));

        // The residual rounds end on the constraint of the operand blends.
        // That is one equation in as many unknowns as there are operands.
        // The blends therefore cross the wire.
        //
        // They are bound before the challenge that combines them.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            OPERAND_BLENDS,
            Length::Fixed(self.arity),
        ));

        // A single operand has nothing to combine.
        if self.arity > 1 {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OPENING_BATCHING,
                Length::Scalar,
            ));
        }

        steps.push(Interaction::marker::<Opening>(
            Hierarchy::Begin,
            Kind::Protocol,
            OPENING_SUMCHECK,
        ));
        steps.push(Interaction::marker::<Opening>(
            Hierarchy::End,
            Kind::Protocol,
            OPENING_SUMCHECK,
        ));

        InteractionPattern::new(steps).expect("two matched brackets are always well formed")
    }

    /// Bind the protocol identity and the transcript shape into a seed.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>())
    }
}

/// Reasons the transcript replay rejects a proof.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ZerocheckTranscriptError {
    /// A message carries a different number of values than described.
    #[error("{label} length mismatch: expected {expected}, got {actual}")]
    LenMismatch {
        /// Which message disagreed.
        label: &'static str,
        /// Number of values the description fixes.
        expected: usize,
        /// Number of values the proof carries.
        actual: usize,
    },
    /// Grinding is enabled but the proof carries no witness for it.
    #[error("the skip round carries no grinding witness")]
    MissingPowWitness,
    /// The grinding witness does not meet the required difficulty.
    #[error("the skip round's grinding witness is invalid")]
    InvalidPowWitness,
    /// Grinding is off but the proof carries a witness anyway.
    ///
    /// Accepting it would give one statement two accepting proofs.
    #[error("the skip round carries a grinding witness at zero difficulty")]
    UnexpectedPowWitness,
}

/// Prover-side transcript of one binary zerocheck.
pub struct ZerocheckProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: ZerocheckShape,
    /// Marker for the extension field the messages and challenges carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZerocheckProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape both sides agreed on.
    pub fn new(challenger: &'a mut C, shape: ZerocheckShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let state = ProverState::new(challenger, &separator);
        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the equality point the constraint is weighted by.
    ///
    /// # Panics
    ///
    /// Panics if the requested width is not the one described.
    pub fn zerocheck_point(&mut self, log_rows: usize) -> Point<EF> {
        assert_eq!(log_rows, self.shape.log_rows(), "described residual width");
        Point::new(
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(ZEROCHECK_POINT, log_rows)
                .into_iter()
                .map(p3_challenger::fs::TranscriptBound::into_inner)
                .collect(),
        )
    }

    /// Bind the round message, grind, and draw the challenge it collapses to.
    pub fn skip_round(&mut self, message: &[EF]) -> (EF, Option<F>) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_MESSAGE, message);
        let witness = (self.shape.pow_bits > 0)
            .then(|| self.state.observe_pow(ROUND_POW, self.shape.pow_bits));
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(SKIP_CHALLENGE)
            .into_inner();
        (challenge, witness)
    }

    /// Lend the sponge to the residual sumcheck, bracketed as a sub-protocol.
    pub fn residual_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Residual>(RESIDUAL_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<Residual>(RESIDUAL_SUMCHECK);
        output
    }

    /// Bind the operand values the residual rounds ended on.
    pub fn operand_blends(&mut self, blends: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OPERAND_BLENDS, blends);
    }

    /// Draw the challenge that combines the operands.
    ///
    /// A single operand has nothing to combine, so it combines by one.
    pub fn opening_batching(&mut self) -> EF {
        if self.shape.arity <= 1 {
            return EF::ONE;
        }
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OPENING_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the opening sumcheck, bracketed as a sub-protocol.
    pub fn opening_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Opening>(OPENING_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<Opening>(OPENING_SUMCHECK);
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
            "the binary zerocheck carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one binary zerocheck.
///
/// Mirrors the prover side call for call, over the same description.
pub struct ZerocheckVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: ZerocheckShape,
    /// Marker for the extension field the messages and challenges carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZerocheckVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape both sides agreed on.
    pub fn new(challenger: &'a mut C, shape: ZerocheckShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let state = VerifierState::new(challenger, &separator, &[]);
        Self {
            state,
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the equality point, which is never read from a proof.
    ///
    /// # Panics
    ///
    /// Panics if the requested width is not the one described.
    pub fn zerocheck_point(&mut self, log_rows: usize) -> Point<EF> {
        assert_eq!(log_rows, self.shape.log_rows(), "described residual width");
        Point::new(
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(ZEROCHECK_POINT, log_rows)
                .into_iter()
                .map(p3_challenger::fs::TranscriptBound::into_inner)
                .collect(),
        )
    }

    /// Replay the skip round: bind the message, re-grind, draw the challenge.
    ///
    /// # Errors
    ///
    /// - The message width differs from the described one.
    /// - Grinding is enabled and the witness is missing or invalid.
    /// - Grinding is off and the proof carries a witness anyway.
    pub fn skip_round(
        &mut self,
        message: &[EF],
        witness: Option<F>,
    ) -> Result<EF, ZerocheckTranscriptError> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_MESSAGE, message)
            .map_err(|_| ZerocheckTranscriptError::LenMismatch {
                label: "round message",
                expected: self.shape.message_len(),
                actual: message.len(),
            })?;

        if self.shape.pow_bits > 0 {
            let Some(witness) = witness else {
                self.state.abort();
                return Err(ZerocheckTranscriptError::MissingPowWitness);
            };
            self.state
                .observe_pow(ROUND_POW, self.shape.pow_bits, witness)
                .map_err(|_| ZerocheckTranscriptError::InvalidPowWitness)?;
        } else if witness.is_some() {
            // At zero difficulty the description has no grinding step.
            // Ignoring one would give a statement two accepting proofs.
            self.state.abort();
            return Err(ZerocheckTranscriptError::UnexpectedPowWitness);
        }

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(SKIP_CHALLENGE)
            .into_inner())
    }

    /// Lend the sponge to the residual sumcheck, bracketed as a sub-protocol.
    ///
    /// A delegated rejection releases the driver.
    /// Without that, dropping this one after a malformed proof panics.
    ///
    /// # Errors
    ///
    /// Returns whatever the delegated run rejected with.
    pub fn residual_sumcheck<T, E>(
        &mut self,
        run: impl FnOnce(&mut C) -> Result<T, E>,
    ) -> Result<T, E> {
        self.state.begin_protocol::<Residual>(RESIDUAL_SUMCHECK);
        let output = run(self.state.challenger_mut());
        if output.is_err() {
            self.state.abort();
            return output;
        }
        self.state.end_protocol::<Residual>(RESIDUAL_SUMCHECK);
        output
    }

    /// Bind the operand values the proof carries.
    ///
    /// # Errors
    ///
    /// Returns an error when the count differs from the constraint's arity.
    pub fn operand_blends(&mut self, blends: &[EF]) -> Result<(), ZerocheckTranscriptError> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OPERAND_BLENDS, blends)
            .map(|_| ())
            .map_err(|_| ZerocheckTranscriptError::LenMismatch {
                label: "operand blends",
                expected: self.shape.arity,
                actual: blends.len(),
            })
    }

    /// Draw the same batching challenge the prover saw.
    pub fn opening_batching(&mut self) -> EF {
        if self.shape.arity <= 1 {
            return EF::ONE;
        }
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OPENING_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the opening sumcheck, bracketed as a sub-protocol.
    ///
    /// # Errors
    ///
    /// Returns whatever the delegated run rejected with.
    pub fn opening_sumcheck<T, E>(
        &mut self,
        run: impl FnOnce(&mut C) -> Result<T, E>,
    ) -> Result<T, E> {
        self.state.begin_protocol::<Opening>(OPENING_SUMCHECK);
        let output = run(self.state.challenger_mut());
        if output.is_err() {
            self.state.abort();
            return output;
        }
        self.state.end_protocol::<Opening>(OPENING_SUMCHECK);
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
            .expect("the binary zerocheck reads an empty wire");
    }
}
