//! Typed transcript for the bus composition sumcheck.

use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_field::ExtensionField;

const VERSION: u8 = 1;
const NAME: &[u8] = b"p3-multi-stark-bus-composition";
const DIRECTION: &str = "direction_batching";
const SUMCHECK: &str = "composition_sumcheck";

type Alphabet<F> = FieldUnit<F>;

struct CompositionSumcheck;

/// Prover transcript that batches directions before delegating the composition sumcheck.
pub(crate) struct BusCompositionProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Typed pattern player borrowing the statement challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Challenge-field marker used by the typed codec.
    _ef: PhantomData<EF>,
}

/// Verifier transcript mirroring the prover's direction batching and delegation.
pub(crate) struct BusCompositionVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Typed pattern player borrowing the statement challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Challenge-field marker used by the typed codec.
    _ef: PhantomData<EF>,
}

fn separator<F, EF>(
    num_variables: usize,
    degree: usize,
    pow_bits: usize,
) -> DomainSeparator<Alphabet<F>>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    // The outer driver records one fresh combiner and one delegated fixed-shape sumcheck.
    let pattern = InteractionPattern::new(alloc::vec![
        Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            DIRECTION,
            Length::Scalar,
        ),
        Interaction::marker::<CompositionSumcheck>(Hierarchy::Begin, Kind::Protocol, SUMCHECK),
        Interaction::marker::<CompositionSumcheck>(Hierarchy::End, Kind::Protocol, SUMCHECK),
    ])
    .expect("one matched composition-sumcheck bracket is well formed");
    let mut separator = DomainSeparator::new(VERSION, NAME, pattern);
    separator
        .instance(&(num_variables as u64).to_be_bytes())
        .instance(&(degree as u64).to_be_bytes())
        .instance(&(pow_bits as u64).to_be_bytes());
    separator
}

impl<'a, C, F, EF> BusCompositionProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    /// Seed the typed composition transcript from verifier-derived dimensions.
    pub(crate) fn new(
        challenger: &'a mut C,
        num_variables: usize,
        degree: usize,
        pow_bits: usize,
    ) -> Self {
        Self {
            state: ProverState::new(
                challenger,
                &separator::<F, EF>(num_variables, degree, pow_bits),
            ),
            _ef: PhantomData,
        }
    }

    /// Sample the sole combiner between push and pull terminal identities.
    pub(crate) fn direction_challenge(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(DIRECTION)
            .into_inner()
    }

    /// Run the fixed-shape generic-degree sumcheck under an explicit delegation.
    pub(crate) fn sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<CompositionSumcheck>(SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<CompositionSumcheck>(SUMCHECK);
        output
    }

    /// Finish after the direction challenge and delegated proof are complete.
    pub(crate) fn finish(self) {
        assert!(self.state.finalize().is_empty());
    }
}

impl<'a, C, F, EF> BusCompositionVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    /// Seed the verifier transcript from the same statement-derived dimensions.
    pub(crate) fn new(
        challenger: &'a mut C,
        num_variables: usize,
        degree: usize,
        pow_bits: usize,
    ) -> Self {
        Self {
            state: VerifierState::new(
                challenger,
                &separator::<F, EF>(num_variables, degree, pow_bits),
                &[],
            ),
            _ef: PhantomData,
        }
    }

    /// Replay the direction combiner.
    pub(crate) fn direction_challenge(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(DIRECTION)
            .into_inner()
    }

    /// Verify the fixed-shape sumcheck under the matching delegation.
    pub(crate) fn sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<CompositionSumcheck>(SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<CompositionSumcheck>(SUMCHECK);
        output
    }

    /// Finish after a complete replay of the empty outer wire.
    pub(crate) fn finish(self) {
        self.state
            .finalize()
            .expect("the bus composition transcript reads an empty wire");
    }

    /// Release the typed completeness check after delegated verification rejects.
    pub(crate) fn abort(&mut self) {
        self.state.abort();
    }
}
