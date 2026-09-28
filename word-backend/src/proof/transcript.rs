//! Typed transcript binding the statement before the relation challenges.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field};
use p3_word::Word;

/// Coefficients separating the relation terms, whose powers weight every term.
pub(super) const RELATION_BATCH_COEFFICIENTS: usize = 1;

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-word-relation-proof";

/// Label of the sampled vanishing point.
const VANISHING_POINT: &str = "vanishing_point";

/// Label of the relation-family batching coefficient.
const RELATION_BATCH: &str = "relation_batch";

/// Label around the relation vanishing check.
const ZEROCHECK: &str = "relation_zerocheck";

/// Label around the multiplication reduction.
const INTEGER_MUL: &str = "integer_mul";

/// Type-level identity of the relation vanishing sub-protocol.
struct Zerocheck;

/// Type-level identity of the multiplication sub-protocol.
struct IntegerMul;

/// Sponge alphabet of a challenger native to the challenge field.
type Alphabet<F> = FieldUnit<F>;

/// Statement dimensions that fix every transcript interaction.
#[derive(Clone, Copy)]
pub(super) struct TranscriptShape {
    /// Number of padded constraint-index variables.
    constraint_variables: usize,
    /// Number of within-word variables.
    bit_variables: usize,
    /// Number of variables the commitment spans, which the padded trace sits inside.
    commitment_variables: usize,
    /// Number of verifier-known words.
    public_words: usize,
    /// Number of committed words.
    witness_words: usize,
    /// Relation counts in linear, bitwise, then unsigned-product order.
    relation_counts: [usize; 3],
}

impl TranscriptShape {
    /// Creates a transcript shape from verifier-derived dimensions.
    pub(super) const fn new(
        constraint_variables: usize,
        bit_variables: usize,
        commitment_variables: usize,
        public_words: usize,
        witness_words: usize,
        relation_counts: [usize; 3],
    ) -> Self {
        // Every value originates in a checked shape or in the commitment itself.
        Self {
            constraint_variables,
            bit_variables,
            commitment_variables,
            public_words,
            witness_words,
            relation_counts,
        }
    }

    /// Number of variables the batched vanishing check binds.
    pub(super) const fn zerocheck_variables(&self) -> usize {
        // Constraint rows are the high coordinates and within-word bits the low ones.
        self.constraint_variables + self.bit_variables
    }

    /// Describes the multiplication bracket, then the two draws around the vanishing check.
    ///
    /// The bracket is played even when the statement declares no product.
    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Both lengths come from the verifier's own constraint system.
        let steps = vec![
            Interaction::marker::<IntegerMul>(Hierarchy::Begin, Kind::Protocol, INTEGER_MUL),
            Interaction::marker::<IntegerMul>(Hierarchy::End, Kind::Protocol, INTEGER_MUL),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                VANISHING_POINT,
                Length::Fixed(self.zerocheck_variables()),
            ),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                RELATION_BATCH,
                Length::Fixed(RELATION_BATCH_COEFFICIENTS),
            ),
            Interaction::marker::<Zerocheck>(Hierarchy::Begin, Kind::Protocol, ZEROCHECK),
            Interaction::marker::<Zerocheck>(Hierarchy::End, Kind::Protocol, ZEROCHECK),
        ];
        InteractionPattern::new(steps).expect("a flat sequence of atomic steps is well formed")
    }

    /// Binds the protocol identity, every dimension, and every public word.
    fn domain_separator<F, EF, W>(&self, public_words: &[W]) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        W: Word,
    {
        // A statement with no public words still binds its declared dimensions.
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        let dimensions = [
            self.constraint_variables,
            self.bit_variables,
            self.commitment_variables,
            self.public_words,
            self.witness_words,
        ];
        for dimension in dimensions.into_iter().chain(self.relation_counts) {
            separator.instance(&(dimension as u64).to_le_bytes());
        }
        for word in public_words {
            // Public words precede every challenge, so no draw can adapt to them.
            separator.instance(&word.to_u64().to_le_bytes());
        }
        separator
    }
}

/// Prover-side driver for the relation challenges.
pub(super) struct ProofProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player and borrowed challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> ProofProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Binds the statement before any relation challenge is drawn.
    pub(super) fn new<W: Word>(
        challenger: &'a mut C,
        shape: TranscriptShape,
        public_words: &[W],
    ) -> Self {
        // Shared inputs are absorbed without adding duplicate proof bytes.
        debug_assert_eq!(public_words.len(), shape.public_words);
        let separator = shape.domain_separator::<F, EF, W>(public_words);
        Self {
            state: ProverState::new(challenger, &separator),
            _field: PhantomData,
        }
    }

    /// Lends the challenger to the multiplication reduction, before either relation draw.
    pub(super) fn integer_mul<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // Every product claim is bound before the vanishing point that weights it.
        self.state.begin_protocol::<IntegerMul>(INTEGER_MUL);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<IntegerMul>(INTEGER_MUL);
        result
    }

    /// Samples the vanishing point before the batching coefficient.
    ///
    /// Returns nothing when the sampled coefficient vanishes, releasing the transcript first.
    pub(super) fn challenges(&mut self, variables: usize) -> Option<(Vec<EF>, EF)>
    where
        EF: Field,
    {
        let drawn = challenges(variables, |label, count| {
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(label, count)
                .into_iter()
                .map(|value| value.into_inner())
                .collect()
        });
        if drawn.1.is_zero() {
            self.abort();
            return None;
        }
        Some(drawn)
    }

    /// Lends the challenger to the relation vanishing check.
    pub(super) fn zerocheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // A named bracket stops the sub-protocol being replayed in another slot.
        self.state.begin_protocol::<Zerocheck>(ZEROCHECK);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Zerocheck>(ZEROCHECK);
        result
    }

    /// Closes the relation transcript.
    pub(super) fn finish(self) {
        // Every value is public or carried by a delegated proof.
        assert!(self.state.finalize().is_empty());
    }

    /// Releases the completeness check when the proof is abandoned part-way.
    fn abort(&mut self) {
        // Dropping a transcript that still owes pattern steps would panic instead.
        self.state.abort();
    }
}

/// Verifier-side replay for the relation challenges.
pub(super) struct ProofVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player over an empty wire and a borrowed challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> ProofVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Reabsorbs the statement before replaying either draw.
    pub(super) fn new<W: Word>(
        challenger: &'a mut C,
        shape: TranscriptShape,
        public_words: &[W],
    ) -> Self {
        // The replay reaches the prover's sponge state from the same inputs.
        debug_assert_eq!(public_words.len(), shape.public_words);
        let separator = shape.domain_separator::<F, EF, W>(public_words);
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            _field: PhantomData,
        }
    }

    /// Lends the challenger to the multiplication replay, before either relation draw.
    pub(super) fn integer_mul<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The verifier follows the prover's named sub-protocol boundary exactly.
        self.state.begin_protocol::<IntegerMul>(INTEGER_MUL);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<IntegerMul>(INTEGER_MUL);
        result
    }

    /// Replays the vanishing draw before the batching draw.
    ///
    /// Returns nothing when the sampled coefficient vanishes, releasing the transcript first.
    pub(super) fn challenges(&mut self, variables: usize) -> Option<(Vec<EF>, EF)>
    where
        EF: Field,
    {
        let drawn = challenges(variables, |label, count| {
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(label, count)
                .into_iter()
                .map(|value| value.into_inner())
                .collect()
        });
        if drawn.1.is_zero() {
            self.abort();
            return None;
        }
        Some(drawn)
    }

    /// Lends the challenger to the vanishing check replay.
    pub(super) fn zerocheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The verifier follows the prover's named sub-protocol boundary exactly.
        self.state.begin_protocol::<Zerocheck>(ZEROCHECK);
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Zerocheck>(ZEROCHECK);
        result
    }

    /// Closes the relation transcript.
    pub(super) fn finish(self) {
        // The reduction itself carries no top-level wire values.
        self.state
            .finalize()
            .expect("the relation proof reads an empty top-level wire");
    }

    /// Releases completeness checks after a rejected delegated proof.
    pub(super) fn abort(&mut self) {
        // A malformed sumcheck may stop after consuming part of its own transcript.
        self.state.abort();
    }
}

/// Draws the vanishing point and then the batching coefficient.
fn challenges<EF: Copy>(
    variables: usize,
    mut draw: impl FnMut(&'static str, usize) -> Vec<EF>,
) -> (Vec<EF>, EF) {
    // The order is protocol-visible because each draw depends on the earlier sponge state.
    let point = draw(VANISHING_POINT, variables);
    let batch = draw(RELATION_BATCH, RELATION_BATCH_COEFFICIENTS);
    (point, batch[0])
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
    use p3_challenger::{CanObserve, CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_word::Word64;

    use super::*;

    type F = BinaryField128;
    type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

    // A sponge whose every draw is zero, which is the degenerate batching coefficient.
    struct ZeroChallenger;

    impl CanObserve<F> for ZeroChallenger {
        fn observe(&mut self, _value: F) {}
    }

    impl CanSample<F> for ZeroChallenger {
        fn sample(&mut self) -> F {
            F::ZERO
        }
    }

    fn challenger() -> Challenger {
        // An empty byte transcript gives both roles the same initial state.
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn shape() -> TranscriptShape {
        // Fixture state: three constraint variables, 64-bit words, nine commitment variables.
        TranscriptShape::new(3, 6, 9, 2, 8, [1, 2, 0])
    }

    fn words() -> [Word64; 2] {
        [Word64::new(0x0102_0304_0506_0708), Word64::new(0x1111_2222)]
    }

    // Sampling only, so nothing here depends on the arithmetic built from these inputs.
    fn sample(prefix: Option<F>, shape: TranscriptShape, public: &[Word64]) -> (Vec<F>, F) {
        let mut challenger = challenger();
        if let Some(value) = prefix {
            challenger.observe(value);
        }
        let mut transcript = ProofProverTranscript::<_, F, F>::new(&mut challenger, shape, public);
        transcript.integer_mul(|_| {});
        let drawn = transcript
            .challenges(shape.zerocheck_variables())
            .expect("a hashed sponge does not draw zero");
        transcript.zerocheck(|_| {});
        transcript.finish();
        drawn
    }

    #[test]
    fn prover_and_verifier_draw_identical_relation_challenges() {
        let mut prover = challenger();
        let mut verifier = challenger();
        let public = words();

        // Both roles absorb the same statement and draw the same two challenges.
        let mut prover_transcript =
            ProofProverTranscript::<_, F, F>::new(&mut prover, shape(), &public);
        let mut verifier_transcript =
            ProofVerifierTranscript::<_, F, F>::new(&mut verifier, shape(), &public);
        let variables = shape().zerocheck_variables();
        prover_transcript.integer_mul(|_| {});
        verifier_transcript.integer_mul(|_| {});
        assert_eq!(
            prover_transcript.challenges(variables).unwrap(),
            verifier_transcript.challenges(variables).unwrap()
        );

        // Empty closures still consume the named sub-protocol boundary.
        prover_transcript.zerocheck(|_| {});
        verifier_transcript.zerocheck(|_| {});
        prover_transcript.finish();
        verifier_transcript.finish();

        // Matching next draws prove that neither side consumed an extra interaction.
        assert_eq!(
            CanSample::<F>::sample(&mut prover),
            CanSample::<F>::sample(&mut verifier)
        );
    }

    #[test]
    fn every_absorbed_statement_input_moves_the_relation_challenges() {
        let public = words();
        let base = sample(None, shape(), &public);

        // Anything absorbed before the transcript, such as a commitment, must move both draws.
        assert_ne!(sample(Some(F::from_repr(1)), shape(), &public), base);
        assert_ne!(
            sample(Some(F::from_repr(1)), shape(), &public),
            sample(Some(F::from_repr(2)), shape(), &public)
        );

        // Changing a public word must move them, though no draw reads one arithmetically.
        let changed = [words()[0], Word64::new(0x1111_2223)];
        assert_ne!(sample(None, shape(), &changed), base);

        // A relation count no other dimension reveals must move them too.
        assert_ne!(
            sample(
                None,
                TranscriptShape::new(3, 6, 9, 2, 8, [2, 1, 0]),
                &public
            ),
            base
        );
        assert_ne!(
            sample(
                None,
                TranscriptShape::new(3, 6, 9, 2, 8, [1, 2, 1]),
                &public
            ),
            base
        );

        // So must the committed segment length and the width of the committed trace.
        assert_ne!(
            sample(
                None,
                TranscriptShape::new(3, 6, 9, 2, 7, [1, 2, 0]),
                &public
            ),
            base
        );
        assert_ne!(
            sample(
                None,
                TranscriptShape::new(3, 6, 10, 2, 8, [1, 2, 0]),
                &public
            ),
            base
        );
    }

    #[test]
    #[should_panic(expected = "Dropped unfinalized VerifierState")]
    fn abandoning_a_drawn_transcript_is_a_hard_failure() {
        // Returning after the two draws leaves the named bracket owed on both sides.
        let mut challenger = challenger();
        let mut transcript =
            ProofVerifierTranscript::<_, F, F>::new(&mut challenger, shape(), &words());
        transcript.integer_mul(|_| {});
        let _ = transcript.challenges(shape().zerocheck_variables());
    }

    #[test]
    fn a_vanishing_coefficient_releases_both_transcripts() {
        // Mutation: force the degenerate draw the refusal is written for.
        let variables = shape().zerocheck_variables();
        let mut prover = ZeroChallenger;
        let mut verifier = ZeroChallenger;

        // Both roles refuse, and dropping either state afterwards must not panic.
        let mut prover_transcript =
            ProofProverTranscript::<_, F, F>::new(&mut prover, shape(), &words());
        prover_transcript.integer_mul(|_| {});
        assert!(prover_transcript.challenges(variables).is_none());
        drop(prover_transcript);

        let mut verifier_transcript =
            ProofVerifierTranscript::<_, F, F>::new(&mut verifier, shape(), &words());
        verifier_transcript.integer_mul(|_| {});
        assert!(verifier_transcript.challenges(variables).is_none());
    }

    #[test]
    fn the_vanishing_point_is_drawn_before_the_batching_coefficient() {
        // A transcript that reversed the two draws would swap these values.
        let recorded = core::cell::RefCell::new(Vec::new());
        let (point, batch) = challenges(3, |label, count| {
            recorded.borrow_mut().push(label);
            (0..count)
                .map(|index| F::from_repr(index as u128 + 1))
                .collect()
        });

        assert_eq!(*recorded.borrow(), [VANISHING_POINT, RELATION_BATCH]);
        assert_eq!(point.len(), 3);
        assert_eq!(batch, F::from_repr(1));
    }
}
