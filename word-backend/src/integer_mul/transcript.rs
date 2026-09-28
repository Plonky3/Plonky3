//! Typed transcript fixing the order of every multiplication-reduction interaction.
//!
//! ```text
//!     row point -> shared root
//!               -> factor layers (sumcheck, halves, line) x 2k -> factor leaf, factor values
//!               -> result layers (sumcheck, halves, line) x (k + 1) -> result leaf, limb values
//! ```
//!
//! Every value the prover sends is absorbed before the challenge that consumes it.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use super::Tree;

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-word-integer-mul";

/// Label of the row point both lifts are compared at.
const ROW_POINT: &str = "row_point";

/// Label of the shared value of both lifts at the row point.
const ROOT: &str = "root";

/// Label of the two halves one layer sumcheck ends on.
const HALVES: &str = "halves";

/// Label of the challenge joining two halves into one claim.
const LINE: &str = "line";

/// Label of the two operand evaluations one leaf sumcheck ends on.
const VALUES: &str = "values";

/// Type-level identity of one layer sumcheck.
struct Layer;

/// Type-level identity of one leaf sumcheck.
struct Leaf;

/// Sponge alphabet of a challenger native to the challenge field.
type Alphabet<F> = FieldUnit<F>;

/// Dimensions that fix every multiplication-reduction interaction.
#[derive(Clone, Copy)]
pub(super) struct TranscriptShape {
    /// Number of padded multiplication-row variables.
    pub(super) row_variables: usize,
    /// Number of within-word variables.
    pub(super) bit_variables: usize,
}

impl TranscriptShape {
    /// Describes the row draw, the root, and both trees in order.
    fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let message = |label, count| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Message, label, count)
        };
        let challenge = |label, count| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, label, count)
        };

        let mut steps = vec![
            challenge(ROW_POINT, Length::Fixed(self.row_variables)),
            message(ROOT, Length::Scalar),
        ];
        for tree in [Tree::Factor, Tree::Result] {
            for _ in 0..tree.depth(self.bit_variables) {
                steps.push(Interaction::marker::<Layer>(
                    Hierarchy::Begin,
                    Kind::Protocol,
                    tree.layer_label(),
                ));
                steps.push(Interaction::marker::<Layer>(
                    Hierarchy::End,
                    Kind::Protocol,
                    tree.layer_label(),
                ));
                steps.push(message(HALVES, Length::Fixed(2)));
                steps.push(challenge(LINE, Length::Scalar));
            }
            steps.push(Interaction::marker::<Leaf>(
                Hierarchy::Begin,
                Kind::Protocol,
                tree.leaf_label(),
            ));
            steps.push(Interaction::marker::<Leaf>(
                Hierarchy::End,
                Kind::Protocol,
                tree.leaf_label(),
            ));
            steps.push(message(VALUES, Length::Fixed(2)));
        }
        InteractionPattern::new(steps)
            .expect("balanced brackets around atomic steps are well formed")
    }

    /// Binds the protocol identity and both dimensions.
    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        for dimension in [self.row_variables, self.bit_variables] {
            separator.instance(&(dimension as u64).to_le_bytes());
        }
        separator
    }
}

/// Prover-side driver for the multiplication reduction.
pub(super) struct ProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player and borrowed challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seeds the reduction from its dimensions.
    pub(super) fn new(challenger: &'a mut C, shape: TranscriptShape) -> Self {
        Self {
            state: ProverState::new(challenger, &shape.domain_separator::<F, EF>()),
            _field: PhantomData,
        }
    }

    /// Draws the row point, then binds the root the prover computes at it.
    pub(super) fn row_point(
        &mut self,
        count: usize,
        root: impl FnOnce(&[EF]) -> EF,
    ) -> (Vec<EF>, EF) {
        let point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(ROW_POINT, count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect::<Vec<_>>();
        let value = root(&point);
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(ROOT, &value);
        (point, value)
    }

    /// Lends the challenger to one layer sumcheck, then binds its halves and draws the line.
    pub(super) fn layer<R>(
        &mut self,
        tree: Tree,
        run: impl FnOnce(&mut &'a mut C) -> (R, [EF; 2]),
    ) -> (R, [EF; 2], EF) {
        self.state.begin_protocol::<Layer>(tree.layer_label());
        let (result, halves) = run(self.state.challenger_mut());
        self.state.end_protocol::<Layer>(tree.layer_label());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(HALVES, &halves);
        let line = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LINE)
            .into_inner();
        (result, halves, line)
    }

    /// Lends the challenger to one leaf sumcheck, then binds its operand values.
    pub(super) fn leaf<R>(
        &mut self,
        tree: Tree,
        run: impl FnOnce(&mut &'a mut C) -> (R, [EF; 2]),
    ) -> (R, [EF; 2]) {
        self.state.begin_protocol::<Leaf>(tree.leaf_label());
        let (result, values) = run(self.state.challenger_mut());
        self.state.end_protocol::<Leaf>(tree.leaf_label());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(VALUES, &values);
        (result, values)
    }

    /// Closes the reduction transcript.
    pub(super) fn finish(self) {
        // Every value is carried by the record, so nothing was written.
        assert!(self.state.finalize().is_empty());
    }
}

/// Verifier-side replay of the multiplication reduction.
pub(super) struct VerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player over an empty wire and a borrowed challenger.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the sampled field.
    _field: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Reseeds the reduction from the same dimensions.
    pub(super) fn new(challenger: &'a mut C, shape: TranscriptShape) -> Self {
        Self {
            state: VerifierState::new(challenger, &shape.domain_separator::<F, EF>(), &[]),
            _field: PhantomData,
        }
    }

    /// Replays the row draw, then binds the root the record carries.
    pub(super) fn row_point(&mut self, count: usize, root: EF) -> Vec<EF> {
        let point = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(ROW_POINT, count)
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(ROOT, &root);
        point
    }

    /// Lends the challenger to one layer replay, then binds its halves and draws the line.
    pub(super) fn layer<R>(
        &mut self,
        tree: Tree,
        halves: &[EF; 2],
        run: impl FnOnce(&mut &'a mut C) -> R,
    ) -> (R, EF) {
        self.state.begin_protocol::<Layer>(tree.layer_label());
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Layer>(tree.layer_label());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(HALVES, halves)
            .expect("a fixed-width array matches its fixed-width step");
        let line = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LINE)
            .into_inner();
        (result, line)
    }

    /// Lends the challenger to one leaf replay, then binds its operand values.
    pub(super) fn leaf<R>(
        &mut self,
        tree: Tree,
        values: &[EF; 2],
        run: impl FnOnce(&mut &'a mut C) -> R,
    ) -> R {
        self.state.begin_protocol::<Leaf>(tree.leaf_label());
        let result = run(self.state.challenger_mut());
        self.state.end_protocol::<Leaf>(tree.leaf_label());
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(VALUES, values)
            .expect("a fixed-width array matches its fixed-width step");
        result
    }

    /// Closes the reduction transcript.
    pub(super) fn finish(self) {
        self.state
            .finalize()
            .expect("the multiplication reduction reads an empty wire");
    }

    /// Releases completeness checks after a rejected record.
    pub(super) fn abort(&mut self) {
        // A malformed record may stop the replay part-way through the pattern.
        self.state.abort();
    }
}
