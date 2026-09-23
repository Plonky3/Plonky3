//! Typed transcript for batching word-operation claims.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_word::Word;

use super::ShiftClaim;

/// Number of variables selecting one of three relation families.
pub(super) const OPERATION_VARIABLES: usize = 2;

/// Number of variables selecting one of at most four operands.
pub(super) const OPERAND_VARIABLES: usize = 2;

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-word-shift-reduction";

/// Label of the shared constraint point.
const CONSTRAINT_POINT: &str = "constraint_point";

/// Label of the shared within-word point.
const BIT_POINT: &str = "bit_point";

/// Label of the operand evaluation claims.
const CLAIMS: &str = "claims";

/// Label of the relation-family batching point.
const OPERATION_BATCH: &str = "operation_batch";

/// Label of the operand-position batching point.
const OPERAND_BATCH: &str = "operand_batch";

/// Label around the within-word sumcheck.
const BIT_SUMCHECK: &str = "bit_sumcheck";

/// Label around the committed-word sumcheck.
const WORD_SUMCHECK: &str = "word_sumcheck";

/// Label of the committed-trace evaluation left by the reduction.
const TRACE_EVALUATION: &str = "trace_evaluation";

/// Type-level identity of the within-word sub-protocol.
struct BitSumcheck;

/// Type-level identity of the committed-word sub-protocol.
struct WordSumcheck;

/// Sponge alphabet of a challenger native to the challenge field.
type Alphabet<F> = FieldUnit<F>;

/// Statement dimensions that fix every transcript interaction.
#[derive(Clone, Copy)]
pub(super) struct TranscriptShape {
    /// Number of padded constraint-index variables.
    constraint_variables: usize,
    /// Number of within-word variables.
    bit_variables: usize,
    /// Number of verifier-known words.
    public_words: usize,
    /// Number of committed words.
    witness_words: usize,
}

impl TranscriptShape {
    /// Creates a transcript shape from verifier-derived dimensions.
    pub(super) const fn new(
        constraint_variables: usize,
        bit_variables: usize,
        public_words: usize,
        witness_words: usize,
    ) -> Self {
        // All values originate in a checked constraint-system shape.
        Self {
            constraint_variables,
            bit_variables,
            public_words,
            witness_words,
        }
    }

    /// Describes the shared statement and its two batching draws.
    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Every length comes from the verifier's constraint system.
        let steps = vec![
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                CONSTRAINT_POINT,
                Length::Fixed(self.constraint_variables * EF::DIMENSION),
            ),
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                BIT_POINT,
                Length::Fixed(self.bit_variables * EF::DIMENSION),
            ),
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                CLAIMS,
                Length::Fixed(8 * EF::DIMENSION),
            ),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OPERATION_BATCH,
                Length::Fixed(OPERATION_VARIABLES),
            ),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OPERAND_BATCH,
                Length::Fixed(OPERAND_VARIABLES),
            ),
            Interaction::marker::<BitSumcheck>(Hierarchy::Begin, Kind::Protocol, BIT_SUMCHECK),
            Interaction::marker::<BitSumcheck>(Hierarchy::End, Kind::Protocol, BIT_SUMCHECK),
            Interaction::marker::<WordSumcheck>(Hierarchy::Begin, Kind::Protocol, WORD_SUMCHECK),
            Interaction::marker::<WordSumcheck>(Hierarchy::End, Kind::Protocol, WORD_SUMCHECK),
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                TRACE_EVALUATION,
                Length::Fixed(EF::DIMENSION),
            ),
        ];
        InteractionPattern::new(steps).expect("a flat sequence of atomic steps is well formed")
    }

    /// Binds the protocol identity and all statement dimensions.
    fn domain_separator<F, EF, W>(&self, public_words: &[W]) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        W: Word,
    {
        // Zero-length public runs still need their dimensions bound explicitly.
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        for dimension in [
            self.constraint_variables,
            self.bit_variables,
            self.public_words,
            self.witness_words,
        ] {
            separator.instance(&(dimension as u64).to_le_bytes());
        }
        for word in public_words {
            // Public words must precede every batching draw, not arrive as adaptable inputs.
            separator.instance(&word.to_u64().to_le_bytes());
        }
        separator
    }
}

/// Equality-weight tables sampled for the two batching axes.
pub(super) struct BatchWeights<F> {
    /// Weights selecting the relation family.
    pub(super) operation: Vec<F>,
    /// Weights selecting the operand position.
    pub(super) operand: Vec<F>,
}

/// Prover-side driver for claim batching.
pub(super) struct ShiftProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player and borrowed challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> ShiftProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Binds the statement before either batching point is sampled.
    pub(super) fn new<W: Word>(
        challenger: &'a mut C,
        shape: TranscriptShape,
        claim: &ShiftClaim<EF>,
        public_words: &[W],
    ) -> Self {
        // Shared inputs are absorbed without adding duplicate proof bytes.
        debug_assert_eq!(public_words.len(), shape.public_words);
        let separator = shape.domain_separator::<F, EF, W>(public_words);
        let mut state = ProverState::new(challenger, &separator);
        state.add_public_scalars::<F, FieldToFieldCodec<F>>(
            CONSTRAINT_POINT,
            &flatten_extensions(claim.constraint_point()),
        );
        state.add_public_scalars::<F, FieldToFieldCodec<F>>(
            BIT_POINT,
            &flatten_extensions(claim.bit_point()),
        );
        state.add_public_scalars::<F, FieldToFieldCodec<F>>(
            CLAIMS,
            &flatten_extensions(&claim.flattened()),
        );
        Self {
            state,
            _field: PhantomData,
        }
    }

    /// Samples the relation axis before the operand axis.
    pub(super) fn batching(&mut self) -> BatchWeights<EF> {
        // The order is protocol-visible because each draw depends on the earlier sponge state.
        let operation_point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                OPERATION_BATCH,
                OPERATION_VARIABLES,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect::<Vec<_>>();
        let operand_point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(OPERAND_BATCH, OPERAND_VARIABLES)
            .into_iter()
            .map(|value| value.into_inner())
            .collect::<Vec<_>>();

        BatchWeights {
            operation: Point::new(operation_point.as_slice()).equality_weights_msb(),
            operand: Point::new(operand_point.as_slice()).equality_weights_msb(),
        }
    }

    /// Lends the challenger to the within-word sumcheck.
    pub(super) fn bit_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // A named bracket prevents either quadratic phase from being replayed in the other slot.
        self.state.begin_protocol::<BitSumcheck>(BIT_SUMCHECK);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<BitSumcheck>(BIT_SUMCHECK);
        result
    }

    /// Lends the challenger to the committed-word sumcheck.
    pub(super) fn word_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The word phase follows the bit phase in the one canonical transcript order.
        self.state.begin_protocol::<WordSumcheck>(WORD_SUMCHECK);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<WordSumcheck>(WORD_SUMCHECK);
        result
    }

    /// Absorbs the evaluation the reduction hands back as an opening claim.
    pub(super) fn trace_evaluation(&mut self, value: EF) {
        // Later challenges drawn by the caller must depend on this value.
        self.state.add_public_scalars::<F, FieldToFieldCodec<F>>(
            TRACE_EVALUATION,
            &flatten_extensions(&[value]),
        );
    }

    /// Closes the complete reduction transcript.
    pub(super) fn finish(self) {
        // Every value is public or carried by a delegated proof.
        assert!(self.state.finalize().is_empty());
    }
}

/// Verifier-side replay for claim batching.
pub(super) struct ShiftVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player over an empty wire and a borrowed challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> ShiftVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Reabsorbs the shared statement before replaying either draw.
    pub(super) fn new<W: Word>(
        challenger: &'a mut C,
        shape: TranscriptShape,
        claim: &ShiftClaim<EF>,
        public_words: &[W],
    ) -> Self {
        // A fixed-width public step cannot fail after shape validation.
        debug_assert_eq!(public_words.len(), shape.public_words);
        let separator = shape.domain_separator::<F, EF, W>(public_words);
        let mut state = VerifierState::new(challenger, &separator, &[]);
        state.observe_public_scalars::<F, FieldToFieldCodec<F>>(
            CONSTRAINT_POINT,
            &flatten_extensions(claim.constraint_point()),
        );
        state.observe_public_scalars::<F, FieldToFieldCodec<F>>(
            BIT_POINT,
            &flatten_extensions(claim.bit_point()),
        );
        state.observe_public_scalars::<F, FieldToFieldCodec<F>>(
            CLAIMS,
            &flatten_extensions(&claim.flattened()),
        );
        Self {
            state,
            _field: PhantomData,
        }
    }

    /// Replays the relation-axis draw before the operand-axis draw.
    pub(super) fn batching(&mut self) -> BatchWeights<EF> {
        // Both draws use the same extension-field codec as the prover.
        let operation_point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                OPERATION_BATCH,
                OPERATION_VARIABLES,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect::<Vec<_>>();
        let operand_point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(OPERAND_BATCH, OPERAND_VARIABLES)
            .into_iter()
            .map(|value| value.into_inner())
            .collect::<Vec<_>>();

        BatchWeights {
            operation: Point::new(operation_point.as_slice()).equality_weights_msb(),
            operand: Point::new(operand_point.as_slice()).equality_weights_msb(),
        }
    }

    /// Lends the challenger to the within-word sumcheck replay.
    pub(super) fn bit_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The verifier follows the prover's named sub-protocol boundary exactly.
        self.state.begin_protocol::<BitSumcheck>(BIT_SUMCHECK);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<BitSumcheck>(BIT_SUMCHECK);
        result
    }

    /// Lends the challenger to the committed-word sumcheck replay.
    pub(super) fn word_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The word phase cannot be substituted for the preceding bit phase.
        self.state.begin_protocol::<WordSumcheck>(WORD_SUMCHECK);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<WordSumcheck>(WORD_SUMCHECK);
        result
    }

    /// Reabsorbs the evaluation carried by the proof.
    pub(super) fn trace_evaluation(&mut self, value: EF) {
        // The replay must reach the same sponge state as the prover.
        self.state
            .observe_public_scalars::<F, FieldToFieldCodec<F>>(
                TRACE_EVALUATION,
                &flatten_extensions(&[value]),
            );
    }

    /// Closes the complete reduction transcript.
    pub(super) fn finish(self) {
        // The reduction itself carries no top-level wire values.
        self.state
            .finalize()
            .expect("the shift reduction reads an empty top-level wire");
    }

    /// Releases completeness checks after a rejected delegated proof.
    pub(super) fn abort(&mut self) {
        // A malformed sumcheck may stop after consuming only part of its own transcript.
        self.state.abort();
    }
}

/// Flattens extension elements into their canonical base-field coordinates.
fn flatten_extensions<F, EF>(values: &[EF]) -> Vec<F>
where
    F: Copy + PrimeCharacteristicRing,
    EF: BasedVectorSpace<F>,
{
    // Public extension values enter the native-field sponge coefficient by coefficient.
    values
        .iter()
        .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_word::Word64;

    use super::*;

    type F = BinaryField128;
    type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Challenger {
        // An empty byte transcript gives both roles the same initial state.
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn claim() -> ShiftClaim<F> {
        // Distinct public values make any transcript reordering observable.
        ShiftClaim::new(
            vec![F::from_repr(1 << 1), F::from_repr(1 << 2)],
            (3..9).map(|bit| F::from_repr(1 << bit)).collect(),
            [F::from_repr(1 << 9)],
            [
                F::from_repr(1 << 10),
                F::from_repr(1 << 11),
                F::from_repr(1 << 12),
            ],
            [
                F::from_repr(1 << 13),
                F::from_repr(1 << 14),
                F::from_repr(1 << 15),
                F::from_repr(1 << 16),
            ],
        )
    }

    #[test]
    fn prover_and_verifier_sample_identical_batching_weights() {
        // Fixture state: two constraint coordinates and six bit coordinates.
        let shape = TranscriptShape::new(2, 6, 3, 8);
        let claim = claim();
        let public = [Word64::new(13), Word64::new(17), Word64::new(19)];
        let mut prover = challenger();
        let mut verifier = challenger();

        // Both roles absorb the same public statement and draw the same two points.
        let mut prover_transcript =
            ShiftProverTranscript::<_, F, F>::new(&mut prover, shape, &claim, &public);
        let mut verifier_transcript =
            ShiftVerifierTranscript::<_, F, F>::new(&mut verifier, shape, &claim, &public);
        let prover_weights = prover_transcript.batching();
        let verifier_weights = verifier_transcript.batching();
        assert_eq!(prover_weights.operation, verifier_weights.operation);
        assert_eq!(prover_weights.operand, verifier_weights.operand);

        // Empty closures still consume both named sub-protocol boundaries.
        prover_transcript.bit_sumcheck(|_| {});
        prover_transcript.word_sumcheck(|_| {});
        verifier_transcript.bit_sumcheck(|_| {});
        verifier_transcript.word_sumcheck(|_| {});
        prover_transcript.trace_evaluation(F::from_repr(1 << 17));
        verifier_transcript.trace_evaluation(F::from_repr(1 << 17));
        prover_transcript.finish();
        verifier_transcript.finish();

        // Matching next draws prove that neither side consumed an extra interaction.
        assert_eq!(
            CanSample::<F>::sample(&mut prover),
            CanSample::<F>::sample(&mut verifier)
        );
    }

    #[test]
    fn every_absorbed_statement_input_moves_the_batching_weights() {
        // Sampling only, so nothing here depends on the arithmetic built from these inputs.
        let sample = |shape, claim: &ShiftClaim<F>, public: &[Word64]| {
            let mut challenger = challenger();
            let mut transcript =
                ShiftProverTranscript::<_, F, F>::new(&mut challenger, shape, claim, public);
            let weights = transcript.batching();
            transcript.bit_sumcheck(|_| {});
            transcript.word_sumcheck(|_| {});
            transcript.trace_evaluation(F::ZERO);
            transcript.finish();
            (weights.operation, weights.operand)
        };
        let shape = TranscriptShape::new(2, 6, 3, 8);
        let public = [Word64::new(13), Word64::new(17), Word64::new(19)];
        let base = sample(shape, &claim(), &public);

        // Perturbing one operand claim must move both batching points.
        let mut zero = *claim().zero();
        zero[0] += F::ONE;
        let perturbed = ShiftClaim::new(
            claim().constraint_point().to_vec(),
            claim().bit_point().to_vec(),
            zero,
            *claim().bitwise_and(),
            *claim().integer_mul(),
        );
        assert_ne!(sample(shape, &perturbed, &public), base);

        // A committed-segment length that no claim value mentions must also move them.
        assert_ne!(
            sample(TranscriptShape::new(2, 6, 3, 9), &claim(), &public),
            base
        );

        // So must a public word, which never enters the sampled points arithmetically.
        let changed = [Word64::new(13), Word64::new(17), Word64::new(23)];
        assert_ne!(sample(shape, &claim(), &changed), base);
    }
}
