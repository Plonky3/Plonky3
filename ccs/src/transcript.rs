//! Fiat-Shamir transcript of the CCS reduction, shaped by the system alone.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, SymmetricSteps, TranscriptBound, TranscriptError, TranscriptField,
    VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use crate::operator::{LinearOperator, encode_usize};
use crate::relation::Ccs;

/// Protocol version bound into the seed.
const VERSION: u8 = 1;

/// Protocol name bound into the seed.
const NAME: &[u8] = b"p3-ccs-superspartan";

/// Constraint weights label.
const ALPHA: &str = "alpha";

/// Zerocheck point label.
const TAU: &str = "tau";

/// Zerocheck sumcheck bracket label.
const ZEROCHECK: &str = "zerocheck";

/// Matrix outputs label.
const OPERAND_EVALS: &str = "operand_evals";

/// Batching weights label.
const BATCHING: &str = "batching";

/// Linear sumcheck bracket label.
const LINEAR: &str = "linear";

/// Witness value label.
const WITNESS_EVAL: &str = "witness_eval";

/// Base-field sponge alphabet.
type Alphabet<F> = FieldUnit<F>;

/// Zerocheck sub-protocol marker.
struct ZerocheckSumcheck;

/// Linear sub-protocol marker.
struct LinearSumcheck;

/// Everything that fixes the transcript of one reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReductionShape {
    /// Row variables.
    log_rows: usize,
    /// Witness variables.
    log_cols: usize,
    /// Number of constraints.
    num_constraints: usize,
    /// Number of matrices.
    num_operators: usize,
    /// Grinding bits per sumcheck round.
    pow_bits: usize,
    /// Injective system encoding.
    statement: Vec<u8>,
}

impl ReductionShape {
    /// The shape of a reduction over a system, binding its full encoding.
    pub(crate) fn new<F, O>(ccs: &Ccs<F, O>, pow_bits: usize) -> Self
    where
        F: TranscriptField,
        O: LinearOperator<F>,
    {
        let mut statement = Vec::new();
        ccs.encode(&mut statement);
        Self {
            log_rows: ccs.log_rows(),
            log_cols: ccs.log_cols(),
            num_constraints: ccs.constraints().len(),
            num_operators: ccs.operators().len(),
            pow_bits,
            statement,
        }
    }

    /// Interaction pattern of this shape.
    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let algebra = |kind, label, length| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, kind, label, length)
        };
        let mut steps = Vec::with_capacity(9);

        // A lone constraint draws no weight.
        if self.num_constraints > 1 {
            steps.push(algebra(
                Kind::Challenge,
                ALPHA,
                Length::Fixed(self.num_constraints),
            ));
        }

        // A single row draws no point.
        if self.log_rows > 0 {
            steps.push(algebra(Kind::Challenge, TAU, Length::Fixed(self.log_rows)));
        }

        // Zerocheck runs under its own seed.
        steps.push(Interaction::marker::<ZerocheckSumcheck>(
            Hierarchy::Begin,
            Kind::Protocol,
            ZEROCHECK,
        ));
        steps.push(Interaction::marker::<ZerocheckSumcheck>(
            Hierarchy::End,
            Kind::Protocol,
            ZEROCHECK,
        ));

        // Outputs are bound before their batching weights.
        steps.push(algebra(
            Kind::Message,
            OPERAND_EVALS,
            Length::Fixed(self.num_operators),
        ));
        steps.push(algebra(
            Kind::Challenge,
            BATCHING,
            Length::Fixed(self.num_operators),
        ));

        // Linear sumcheck runs under its own seed.
        steps.push(Interaction::marker::<LinearSumcheck>(
            Hierarchy::Begin,
            Kind::Protocol,
            LINEAR,
        ));
        steps.push(Interaction::marker::<LinearSumcheck>(
            Hierarchy::End,
            Kind::Protocol,
            LINEAR,
        ));

        // Final witness value.
        steps.push(algebra(Kind::Message, WITNESS_EVAL, Length::Scalar));

        InteractionPattern::new(steps).expect("two matched brackets are always well formed")
    }

    /// Bind the protocol identity, this shape, and the statement.
    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // Bind what the step sequence does not.
        for value in [self.log_cols, self.pow_bits] {
            let mut bytes = Vec::with_capacity(8);
            encode_usize(&mut bytes, value);
            separator.instance(&bytes);
        }
        separator.instance(&self.statement);
        separator
    }
}

/// Prover side of the reduction transcript, over a borrowed challenger.
pub(crate) struct ProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Transcript driver.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Reduction shape.
    shape: ReductionShape,
    /// Challenge field marker.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: ReductionShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// One weight per constraint; a lone constraint gets one.
    pub(crate) fn alphas(&mut self) -> Vec<EF> {
        play_alphas::<F, EF, _>(&mut self.state, &self.shape)
    }

    /// Draw the zerocheck point.
    pub(crate) fn tau(&mut self) -> Vec<EF> {
        play_tau::<F, EF, _>(&mut self.state, &self.shape)
    }

    /// Run the zerocheck sumcheck as a bracketed sub-protocol.
    pub(crate) fn zerocheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<ZerocheckSumcheck>(ZEROCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ZerocheckSumcheck>(ZEROCHECK);
        output
    }

    /// Bind the matrix outputs at the zerocheck point.
    pub(crate) fn operand_evals(&mut self, values: &[EF]) {
        let _ = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OPERAND_EVALS, values);
    }

    /// One batching weight per matrix.
    pub(crate) fn batching(&mut self) -> Vec<EF> {
        draw::<F, EF, _>(&mut self.state, BATCHING, self.shape.num_operators)
    }

    /// Run the linear sumcheck as a bracketed sub-protocol.
    pub(crate) fn linear<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<LinearSumcheck>(LINEAR);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<LinearSumcheck>(LINEAR);
        output
    }

    /// Bind the final witness value and close the transcript.
    ///
    /// # Panics
    ///
    /// - An earlier step was skipped.
    pub(crate) fn finish(mut self, witness_eval: EF) {
        let _ = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(WITNESS_EVAL, &witness_eval);

        // The proof carries every value.
        assert!(
            self.state.finalize().is_empty(),
            "the reduction carries every value in its own proof"
        );
    }
}

/// Verifier side of the reduction transcript, mirroring the prover.
pub(crate) struct VerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Transcript driver over an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Reduction shape.
    shape: ReductionShape,
    /// Challenge field marker.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: ReductionShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// One weight per constraint; a lone constraint gets one.
    pub(crate) fn alphas(&mut self) -> Vec<EF> {
        play_alphas::<F, EF, _>(&mut self.state, &self.shape)
    }

    /// Draw the zerocheck point.
    pub(crate) fn tau(&mut self) -> Vec<EF> {
        play_tau::<F, EF, _>(&mut self.state, &self.shape)
    }

    /// Run the zerocheck sumcheck as a bracketed sub-protocol.
    pub(crate) fn zerocheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<ZerocheckSumcheck>(ZEROCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ZerocheckSumcheck>(ZEROCHECK);
        output
    }

    /// Bind the matrix outputs, rejecting a wrong count before absorbing.
    pub(crate) fn operand_evals(&mut self, values: &[EF]) -> Result<(), TranscriptError> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OPERAND_EVALS, values)
            .map(drop)
    }

    /// One batching weight per matrix.
    pub(crate) fn batching(&mut self) -> Vec<EF> {
        draw::<F, EF, _>(&mut self.state, BATCHING, self.shape.num_operators)
    }

    /// Run the linear sumcheck as a bracketed sub-protocol.
    pub(crate) fn linear<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<LinearSumcheck>(LINEAR);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<LinearSumcheck>(LINEAR);
        output
    }

    /// Bind the final witness value and close the transcript.
    ///
    /// # Panics
    ///
    /// - An earlier step was skipped; an empty wire never leaves bytes over.
    pub(crate) fn finish(mut self, witness_eval: EF) {
        let _ = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(WITNESS_EVAL, &witness_eval);

        // The wire is empty.
        self.state
            .finalize()
            .expect("the reduction reads an empty wire");
    }

    /// Release the completeness check before rejecting.
    pub(crate) fn abort(&mut self) {
        self.state.abort();
    }
}

/// Draw one weight per constraint, or the lone weight one, on either side.
fn play_alphas<F, EF, S>(state: &mut S, shape: &ReductionShape) -> Vec<EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    S: SymmetricSteps,
    S::Challenger: CanObserve<F> + CanSample<F>,
{
    if shape.num_constraints == 1 {
        return alloc::vec![EF::ONE];
    }
    draw::<F, EF, S>(state, ALPHA, shape.num_constraints)
}

/// Draw the zerocheck point on either side; a single row draws nothing.
fn play_tau<F, EF, S>(state: &mut S, shape: &ReductionShape) -> Vec<EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    S: SymmetricSteps,
    S::Challenger: CanObserve<F> + CanSample<F>,
{
    if shape.log_rows == 0 {
        return Vec::new();
    }
    draw::<F, EF, S>(state, TAU, shape.log_rows)
}

/// Draw a list of challenges as one step on either side.
fn draw<F, EF, S>(state: &mut S, label: &'static str, count: usize) -> Vec<EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    S: SymmetricSteps,
    S::Challenger: CanObserve<F> + CanSample<F>,
{
    // Accepting every candidate draws exactly the plain list.
    state
        .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(label, count, |_, _| true)
        .into_iter()
        .map(TranscriptBound::into_inner)
        .collect()
}
