//! Fiat-Shamir transcript of the stacked layout's claim-recording phase.
//!
//! # Overview
//!
//! Three components carry what the layout binds before its residual fold runs.
//!
//! - One recorded batch of concrete openings on one source table.
//! - One out-of-domain evaluation of the whole stacked polynomial.
//! - The batching challenge that collapses every recorded claim into one.
//!
//! Each of the three is a component in its own right.
//!
//! Each seeds a sub-transcript from the sponge it borrows.
//!
//! Each plays its own steps.
//!
//! Each hands the sponge back advanced.
//!
//! # Shape
//!
//! ```text
//!     opening batch, point drawn here:
//!         opening point         1 extension element, drawn
//!         current evaluations   one step of the direct column count
//!         next evaluations      one step of the successor column count
//!
//!     opening batch, point fixed by the caller:
//!         current evaluations   one step of the direct column count
//!         next evaluations      one step of the successor column count
//!
//!     out-of-domain claim:
//!         opening point         1 extension element, drawn
//!         evaluation            1 extension element
//!
//!     batching:
//!         challenge             1 extension element, drawn
//! ```
//!
//! A column group of size zero contributes no step.
//!
//! A batch that opens only successor views describes one step, not two.
//!
//! # Why the three components stay apart
//!
//! An out-of-domain claim draws one point and absorbs one value.
//!
//! So does a batch opening a single column of a full-width table.
//!
//! The two mean different things.
//!
//! One pins the stacked polynomial.
//!
//! The other pins one column.
//!
//! Separate protocol names keep them on separate seeds.
//!
//! Neither can then be replayed in the other's place.
//!
//! # Why one description covers both variable orders
//!
//! Prefix-first and suffix-first binding are two variable orders of one protocol.
//!
//! - They record the same claims.
//! - They absorb the same values.
//! - They draw the same challenges.
//!
//! Their step sequences are identical, step for step.
//!
//! Two descriptions would state that equality twice.
//!
//! One copy would then rot.
//!
//! One description carrying the order as a value states it once.
//!
//! The order must still separate the two runs.
//!
//! It moves no step.
//!
//! The fingerprint therefore cannot carry it.
//!
//! It is bound as an instance label instead.
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//!
//! - Whether the opening point is drawn here.
//! - The direct column count.
//! - The successor column count.
//!
//! Each of the three moves the step sequence.
//!
//! # What the instance label binds
//!
//! The rest of the configuration moves no step.
//!
//! Each entry below is its own delimited chunk.
//!
//! - Arity of the stacked polynomial.
//! - Whether selector bits sit after the local bits.
//! - Order the residual rounds bind the variables in.
//! - Index of the source table a batch opens.
//! - Arity of that source table.
//! - Number of concrete claims.
//! - Number of out-of-domain claims.
//!
//! Both sides read every one of these from their own configuration.
//!
//! None is ever read back out of a proof.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;

use crate::error::SumcheckError;
use crate::layout::LayoutStrategy;
use crate::strategy::VariableOrder;
use crate::table::OpeningEvals;

/// Version byte bound into every seed this phase produces.
///
/// Bumping it separates two revisions of the phase.
///
/// It separates them even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name of one recorded batch of concrete openings.
const OPENING_NAME: &[u8] = b"p3-sumcheck-layout-opening";

/// Protocol name of one out-of-domain evaluation of the stacked polynomial.
const VIRTUAL_NAME: &[u8] = b"p3-sumcheck-layout-ood";

/// Protocol name of the claim-batching challenge.
const BATCHING_NAME: &[u8] = b"p3-sumcheck-layout-batching";

/// Step label of a local-frame opening point drawn from the transcript.
const OPENING_POINT: &str = "opening_point";

/// Step label of the evaluations of the directly opened columns.
const CURRENT_EVALS: &str = "current_evals";

/// Step label of the evaluations of the successor views.
const NEXT_EVALS: &str = "next_evals";

/// Step label of an out-of-domain point drawn from the transcript.
const VIRTUAL_POINT: &str = "virtual_point";

/// Step label of an out-of-domain evaluation of the stacked polynomial.
const VIRTUAL_EVAL: &str = "virtual_eval";

/// Step label of the claim-batching challenge.
const BATCHING: &str = "batching";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Geometry of one stacked layout.
///
/// Both sides derive it from their own configuration.
///
/// Every field moves the transcript seed of every component that carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayoutBinding {
    /// Arity of the stacked polynomial claims are lifted into.
    pub(crate) num_variables: usize,
    /// Selector placement and residual binding order of this layout.
    pub(crate) strategy: LayoutStrategy,
}

impl LayoutBinding {
    /// Collect the geometry shared by every component of one layout.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: arity of the stacked polynomial.
    /// - `strategy`: selector placement and residual binding order.
    pub(crate) const fn new(num_variables: usize, strategy: LayoutStrategy) -> Self {
        Self {
            num_variables,
            strategy,
        }
    }

    /// Append this geometry to a separator's instance label.
    ///
    /// One chunk is written per field.
    ///
    /// None of these numbers moves a step.
    ///
    /// The fingerprint therefore cannot carry them.
    ///
    /// Chunks are delimited.
    ///
    /// Two fields never collapse into one blob.
    fn bind<F>(&self, separator: &mut DomainSeparator<Alphabet<F>>)
    where
        F: TranscriptField,
    {
        // Stacked arity decides how wide a lifted claim point is.
        separator.instance(&(self.num_variables as u64).to_be_bytes());

        // Selector placement decides which end of a claim point the slot bits occupy.
        separator.instance(&[u8::from(self.strategy.reverse_selectors)]);

        // Residual binding order decides which variables the fold consumes first.
        //
        // It moves no step of this phase.
        //
        // This byte is the only thing separating the two orders.
        separator.instance(&[match self.strategy.variable_order {
            VariableOrder::Prefix => 0,
            VariableOrder::Suffix => 1,
        }]);
    }
}

/// How the local-frame opening point of one recorded batch is fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PointSource {
    /// Drawn here, as one challenge expanded into a whole point.
    Drawn,
    /// Fixed by the caller.
    ///
    /// The caller then owes the transcript its own binding of that point.
    Given,
}

/// Numbers that fix the transcript of one recorded batch of concrete openings.
///
/// Both sides build this from their own opening schedule.
///
/// No part of it is ever read out of a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OpeningShape {
    /// Geometry of the layout this batch is recorded against.
    pub(crate) binding: LayoutBinding,
    /// Index of the source table whose columns this batch opens.
    pub(crate) table_index: usize,
    /// Arity of that source table.
    ///
    /// This is the width of the local-frame opening point.
    pub(crate) table_variables: usize,
    /// Number of columns opened directly.
    pub(crate) num_current: usize,
    /// Number of columns opened through the successor view.
    pub(crate) num_next: usize,
    /// How the local-frame opening point is fixed.
    pub(crate) point: PointSource,
}

impl OpeningShape {
    /// Collect the numbers that fix one recorded batch.
    ///
    /// # Arguments
    ///
    /// - `binding`: geometry of the layout the batch is recorded against.
    /// - `table_index`: index of the source table the batch opens.
    /// - `table_variables`: arity of that source table.
    /// - `num_current`: number of columns opened directly.
    /// - `num_next`: number of columns opened through the successor view.
    /// - `point`: how the local-frame opening point is fixed.
    pub(crate) const fn new(
        binding: LayoutBinding,
        table_index: usize,
        table_variables: usize,
        num_current: usize,
        num_next: usize,
        point: PointSource,
    ) -> Self {
        Self {
            binding,
            table_index,
            table_variables,
            num_current,
            num_next,
            point,
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A flat sequence of leaf steps always passes structural validation.
    pub(crate) fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // At most a drawn point followed by the two column groups.
        let mut steps = Vec::with_capacity(3);

        // A point drawn here is one challenge.
        //
        // It is expanded into a whole point afterwards.
        //
        // A point the caller fixes contributes no step.
        //
        // The two cases therefore cannot be confused.
        if self.point == PointSource::Drawn {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OPENING_POINT,
                Length::Scalar,
            ));
        }

        // The directly opened columns are one step.
        //
        // It carries one value per column.
        //
        // A group of size zero would declare a step carrying nothing.
        //
        // Such a step is omitted.
        if self.num_current > 0 {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                CURRENT_EVALS,
                Length::Fixed(self.num_current),
            ));
        }

        // The successor views follow, in the same one-step-per-group form.
        if self.num_next > 0 {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                NEXT_EVALS,
                Length::Fixed(self.num_next),
            ));
        }

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity, this shape, and the layout it is recorded against.
    pub(crate) fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, OPENING_NAME, self.pattern::<F, EF>());

        // Layout geometry comes first.
        //
        // Every component of one layout then shares that prefix.
        self.binding.bind::<F>(&mut separator);

        // Two batches of equal widths on two tables must not share a seed.
        separator.instance(&(self.table_index as u64).to_be_bytes());

        // Table arity fixes how wide the point is.
        //
        // The draw itself is one challenge at any arity.
        separator.instance(&(self.table_variables as u64).to_be_bytes());

        separator
    }
}

/// Prover-side transcript of one recorded batch of concrete openings.
///
/// # Overview
///
/// Holds the only definition of what a prover binds when it records one batch.
///
/// # Borrowing
///
/// The challenger is borrowed, not consumed.
///
/// A layout is opened inside a larger protocol.
///
/// That protocol's own transcript continues where this one stops.
pub(crate) struct OpeningProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this batch was described with.
    shape: OpeningShape,
    /// Marker for the extension field the claims carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> OpeningProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this batch.
    /// - `shape`: the numbers that fix this batch's transcript.
    pub(crate) fn new(challenger: &'a mut C, shape: OpeningShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the local-frame opening point.
    ///
    /// One challenge is drawn.
    ///
    /// It is expanded into successive powers of itself.
    ///
    /// ```text
    ///     draw z  ->  point = (z, z^2, ..., z^n)
    /// ```
    ///
    /// The expansion width is the source table's arity.
    ///
    /// A wider point costs no extra draw.
    ///
    /// The arity is therefore bound through the instance label.
    ///
    /// # Panics
    ///
    /// When the batch was described with a caller-fixed point.
    pub(crate) fn point(&mut self) -> Point<EF> {
        // A described step is the only thing that may be played.
        assert_eq!(
            self.shape.point,
            PointSource::Drawn,
            "a batch whose point the caller fixes draws no point of its own",
        );

        // One challenge, expanded to one coordinate per source-table variable.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OPENING_POINT)
            .into_inner();
        Point::expand_from_univariate(challenge, self.shape.table_variables)
    }

    /// Bind the claimed evaluations.
    ///
    /// Direct columns come first.
    ///
    /// Successor views come second.
    ///
    /// # Panics
    ///
    /// When either group's length differs from the described one.
    pub(crate) fn evaluations(&mut self, evals: &OpeningEvals<EF>) {
        // Direct columns come first.
        //
        // That is the order the batched sum walks them in.
        if self.shape.num_current > 0 {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(CURRENT_EVALS, evals.current());
        }

        // Successor views continue the same walk.
        if self.shape.num_next > 0 {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(NEXT_EVALS, evals.next());
        }
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the batch was described with.
    pub(crate) fn finish(self) {
        // Nothing was written to the driver's own buffer.
        //
        // Closing is therefore purely the check that the description was consumed.
        assert!(
            self.state.finalize().is_empty(),
            "the layout carries every claimed evaluation in its own proof",
        );
    }
}

/// Verifier-side transcript of one recorded batch of concrete openings.
///
/// Mirrors the prover side call for call, over the same description.
///
/// The claimed evaluations arrive from a proof rather than from a wire.
///
/// Their counts are checked against the description before anything is absorbed.
pub(crate) struct OpeningVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value.
    ///
    /// The driver therefore reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this batch was described with.
    shape: OpeningShape,
    /// Marker for the extension field the claims carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> OpeningVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    ///
    /// The arguments match the prover's.
    ///
    /// Both sides therefore seed identically.
    pub(crate) fn new(challenger: &'a mut C, shape: OpeningShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the same local-frame opening point the prover drew.
    ///
    /// # Panics
    ///
    /// When the batch was described with a caller-fixed point.
    pub(crate) fn point(&mut self) -> Point<EF> {
        // A described step is the only thing that may be replayed.
        assert_eq!(
            self.shape.point,
            PointSource::Drawn,
            "a batch whose point the caller fixes draws no point of its own",
        );

        // One challenge, expanded exactly as the prover expanded it.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OPENING_POINT)
            .into_inner();
        Point::expand_from_univariate(challenge, self.shape.table_variables)
    }

    /// Bind the claimed evaluations.
    ///
    /// Direct columns come first.
    ///
    /// Successor views come second.
    ///
    /// The counts come from a proof.
    ///
    /// A disagreement with the description is therefore a rejection.
    ///
    /// # Errors
    ///
    /// When either group carries a count the schedule did not ask for.
    pub(crate) fn evaluations(&mut self, evals: &OpeningEvals<EF>) -> Result<(), SumcheckError> {
        // The schedule fixes both counts.
        //
        // The proof has to match them exactly.
        //
        //     described:   3 direct, 1 successor
        //     proof holds: 2 direct, 1 successor
        //     -> rejected before a single value is absorbed
        //
        // A group of size zero has no step to fail on.
        //
        // This check is the only guard on such a group.
        if evals.current().len() != self.shape.num_current
            || evals.next().len() != self.shape.num_next
        {
            // Releasing the completeness check keeps this rejection the only failure.
            self.state.abort();
            return Err(SumcheckError::OpeningShapeMismatch {
                table_idx: self.shape.table_index,
                expected_current: self.shape.num_current,
                expected_next: self.shape.num_next,
                actual_current: evals.current().len(),
                actual_next: evals.next().len(),
            });
        }

        // Both counts now match the description.
        //
        // Neither absorb can then disagree with its step.
        if self.shape.num_current > 0 {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(CURRENT_EVALS, evals.current())
                .expect("the direct group was counted against its step first");
        }
        if self.shape.num_next > 0 {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(NEXT_EVALS, evals.next())
                .expect("the successor group was counted against its step first");
        }

        Ok(())
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When fewer steps were replayed than the batch was described with.
    pub(crate) fn finish(self) {
        // The proof carries every value.
        //
        // No unread wire bytes can therefore remain.
        self.state
            .finalize()
            .expect("the layout reads an empty wire");
    }
}

/// Numbers that fix the transcript of one out-of-domain claim.
///
/// The claim is an evaluation of the whole stacked polynomial.
///
/// Both sides build this from their own configuration.
///
/// No part of it is ever read out of a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VirtualShape {
    /// Geometry of the layout the claim is recorded against.
    pub(crate) binding: LayoutBinding,
}

impl VirtualShape {
    /// Collect the numbers that fix one out-of-domain claim.
    ///
    /// # Arguments
    ///
    /// - `binding`: geometry of the layout the claim is recorded against.
    pub(crate) const fn new(binding: LayoutBinding) -> Self {
        Self { binding }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A flat sequence of leaf steps always passes structural validation.
    pub(crate) fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        InteractionPattern::new(vec![
            // The point is drawn before the value.
            //
            // The value can then not be chosen to suit it.
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                VIRTUAL_POINT,
                Length::Scalar,
            ),
            // The claimed evaluation of the whole stacked polynomial at that point.
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                VIRTUAL_EVAL,
                Length::Scalar,
            ),
        ])
        .expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity, this shape, and the layout it is recorded against.
    pub(crate) fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, VIRTUAL_NAME, self.pattern::<F, EF>());

        // Layout geometry, including the arity the drawn point is expanded to.
        self.binding.bind::<F>(&mut separator);

        separator
    }
}

/// Prover-side transcript of one out-of-domain claim on the stacked polynomial.
///
/// The challenger is borrowed, not consumed.
pub(crate) struct VirtualProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this claim was described with.
    shape: VirtualShape,
    /// Marker for the extension field the claim carries.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VirtualProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub(crate) fn new(challenger: &'a mut C, shape: VirtualShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the out-of-domain point covering every stacked variable.
    pub(crate) fn point(&mut self) -> Point<EF> {
        // One challenge, expanded to one coordinate per stacked variable.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(VIRTUAL_POINT)
            .into_inner();
        Point::expand_from_univariate(challenge, self.shape.binding.num_variables)
    }

    /// Bind the claimed evaluation at the drawn point.
    pub(crate) fn evaluation(&mut self, eval: EF) {
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(VIRTUAL_EVAL, &eval);
    }

    /// Close the transcript once both described steps have been played.
    ///
    /// # Panics
    ///
    /// When either described step was skipped.
    pub(crate) fn finish(self) {
        // Nothing was written to the driver's own buffer.
        assert!(
            self.state.finalize().is_empty(),
            "the layout carries every claimed evaluation in its own proof",
        );
    }
}

/// Verifier-side transcript of one out-of-domain claim.
///
/// Mirrors the prover side call for call, over the same description.
pub(crate) struct VirtualVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries the claimed value.
    ///
    /// The driver therefore reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this claim was described with.
    shape: VirtualShape,
    /// Marker for the extension field the claim carries.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VirtualVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    ///
    /// The argument matches the prover's.
    ///
    /// Both sides therefore seed identically.
    pub(crate) fn new(challenger: &'a mut C, shape: VirtualShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the same out-of-domain point the prover drew.
    pub(crate) fn point(&mut self) -> Point<EF> {
        // One challenge, expanded exactly as the prover expanded it.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(VIRTUAL_POINT)
            .into_inner();
        Point::expand_from_univariate(challenge, self.shape.binding.num_variables)
    }

    /// Bind the claimed evaluation the proof carries.
    ///
    /// The step describes exactly one value.
    ///
    /// One value cannot disagree with that, so this never fails.
    pub(crate) fn evaluation(&mut self, eval: EF) {
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(VIRTUAL_EVAL, &eval);
    }

    /// Close the transcript once both described steps have been replayed.
    ///
    /// # Panics
    ///
    /// When either described step was skipped.
    pub(crate) fn finish(self) {
        // The proof carries the claimed value.
        //
        // No unread wire bytes can therefore remain.
        self.state
            .finalize()
            .expect("the layout reads an empty wire");
    }
}

/// Numbers that fix the transcript of the claim-batching challenge.
///
/// Both sides build this from the claims they recorded.
///
/// No part of it is ever read out of a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchingShape {
    /// Geometry of the layout whose claims are collapsed.
    pub(crate) binding: LayoutBinding,
    /// Number of concrete openings recorded across every source table.
    pub(crate) num_claims: usize,
    /// Number of out-of-domain claims recorded on the stacked polynomial.
    pub(crate) num_virtual_claims: usize,
}

impl BatchingShape {
    /// Collect the numbers that fix the batching challenge.
    ///
    /// # Arguments
    ///
    /// - `binding`: geometry of the layout whose claims are collapsed.
    /// - `num_claims`: number of concrete openings recorded.
    /// - `num_virtual_claims`: number of out-of-domain claims recorded.
    pub(crate) const fn new(
        binding: LayoutBinding,
        num_claims: usize,
        num_virtual_claims: usize,
    ) -> Self {
        Self {
            binding,
            num_claims,
            num_virtual_claims,
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A single leaf step always passes structural validation.
    pub(crate) fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        InteractionPattern::new(vec![Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BATCHING,
            Length::Scalar,
        )])
        .expect("a single leaf step is always well formed")
    }

    /// Bind the protocol identity, the layout, and how many claims are collapsed.
    ///
    /// The challenge weights each claim by a successive power of itself.
    ///
    /// ```text
    ///     sum = sum_i  alpha^i * eval_i
    /// ```
    ///
    /// The claim counts decide which power lands on which claim.
    ///
    /// They move no step.
    ///
    /// They are bound here instead.
    pub(crate) fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, BATCHING_NAME, self.pattern::<F, EF>());

        // Layout geometry comes first.
        //
        // Every component of one layout then shares that prefix.
        self.binding.bind::<F>(&mut separator);

        // Concrete openings take the low powers.
        //
        // Out-of-domain claims take the ones after them.
        separator.instance(&(self.num_claims as u64).to_be_bytes());
        separator.instance(&(self.num_virtual_claims as u64).to_be_bytes());

        separator
    }
}

/// Draw the claim-batching challenge on the prover side.
///
/// One driver spans the whole component.
///
/// It seeds, draws, then closes.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the draw.
/// - `shape`: the numbers that fix this draw's transcript.
///
/// # Panics
///
/// Never in practice.
///
/// The single described step is played before the driver closes.
pub(crate) fn prover_batching_challenge<C, F, EF>(challenger: &mut C, shape: BatchingShape) -> EF
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    // Seeding folds the shape fingerprint into the sponge before the draw.
    let separator = shape.domain_separator::<F, EF>();
    let mut state = ProverState::new(challenger, &separator);

    // The one described step.
    let alpha = state
        .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCHING)
        .into_inner();

    // Nothing was written to the driver's own buffer.
    assert!(
        state.finalize().is_empty(),
        "drawing a challenge writes nothing to the proof",
    );

    alpha
}

/// Draw the claim-batching challenge on the verifier side.
///
/// The argument matches the prover's.
///
/// Both sides therefore seed identically.
///
/// Both then land on one challenge.
///
/// # Arguments
///
/// - `challenger`: sponge of the surrounding protocol, borrowed for the draw.
/// - `shape`: the numbers that fix this draw's transcript.
///
/// # Panics
///
/// Never in practice.
///
/// The single described step is replayed before the driver closes.
pub(crate) fn verifier_batching_challenge<C, F, EF>(challenger: &mut C, shape: BatchingShape) -> EF
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    // Seeding folds the shape fingerprint into the sponge before the draw.
    let separator = shape.domain_separator::<F, EF>();
    let mut state = VerifierState::new(challenger, &separator, &[]);

    // The one described step.
    let alpha = state
        .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCHING)
        .into_inner();

    // A drawn challenge reads nothing.
    //
    // No unread wire bytes can therefore remain.
    state
        .finalize()
        .expect("drawing a challenge reads an empty wire");

    alpha
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_challenger::testing::{SeedDigest, assert_seeds_pairwise_distinct, seed_digest};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::table::OpeningBatch;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// The layout geometry every shape in these tests is built on.
    fn base_binding() -> LayoutBinding {
        // Stacked arity 11.
        //
        // Selectors after the local bits.
        //
        // Prefix-first residual binding.
        LayoutBinding::new(11, LayoutStrategy::new(true, VariableOrder::Prefix))
    }

    /// A batch of three direct and one successor opening on table 1 of arity 9.
    fn base_opening() -> OpeningShape {
        OpeningShape::new(base_binding(), 1, 9, 3, 1, PointSource::Drawn)
    }

    /// Evaluations matching the described widths of the reference batch.
    fn base_evals() -> OpeningEvals<EF> {
        // Three direct values and one successor value, all distinct.
        OpeningBatch::new(vec![EF::ONE, EF::TWO, EF::from_u8(3)], vec![EF::from_u8(4)])
    }

    /// Label a shape's seed digest so a failing pairwise check names the knob.
    fn labelled_opening(name: &str, shape: OpeningShape) -> (String, SeedDigest) {
        (
            String::from(name),
            seed_digest(&shape.domain_separator::<F, EF>()),
        )
    }

    #[test]
    fn every_knob_of_a_recorded_batch_reaches_the_seed() {
        // Invariant:
        //     One field changed moves the seed.
        //     No two of the changed shapes share a seed either.
        //
        // Fixture state:
        //     stacked arity     11
        //     selectors         reversed
        //     binding order     prefix
        //     source table      index 1, arity 9
        //     direct openings   3
        //     successor         1
        //     opening point     drawn here
        let base = base_opening();

        // Each entry moves exactly one field away from the baseline.
        let mut seeds = vec![labelled_opening("baseline", base)];

        // Stacked arity: the width every claim point is lifted into.
        seeds.push(labelled_opening(
            "stacked arity",
            OpeningShape {
                binding: LayoutBinding::new(12, base.binding.strategy),
                ..base
            },
        ));

        // Selector placement: which end of a claim point the slot bits occupy.
        seeds.push(labelled_opening(
            "selector placement",
            OpeningShape {
                binding: LayoutBinding::new(11, LayoutStrategy::new(false, VariableOrder::Prefix)),
                ..base
            },
        ));

        // Residual binding order: the knob that moves no step at all.
        seeds.push(labelled_opening(
            "variable order",
            OpeningShape {
                binding: LayoutBinding::new(11, LayoutStrategy::new(true, VariableOrder::Suffix)),
                ..base
            },
        ));

        // Source table index: two equal-width batches on two tables.
        seeds.push(labelled_opening(
            "table index",
            OpeningShape {
                table_index: 2,
                ..base
            },
        ));

        // Source table arity: the expansion width of the drawn point.
        seeds.push(labelled_opening(
            "table arity",
            OpeningShape {
                table_variables: 8,
                ..base
            },
        ));

        // Direct column count: a declared step width.
        seeds.push(labelled_opening(
            "direct count",
            OpeningShape {
                num_current: 2,
                ..base
            },
        ));

        // Successor column count: the other declared step width.
        seeds.push(labelled_opening(
            "successor count",
            OpeningShape {
                num_next: 2,
                ..base
            },
        ));

        // Point source: a step present in one case and absent in the other.
        seeds.push(labelled_opening(
            "point source",
            OpeningShape {
                point: PointSource::Given,
                ..base
            },
        ));

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn every_knob_of_an_out_of_domain_claim_reaches_the_seed() {
        // Invariant:
        //     The layout geometry is the whole configuration here.
        //     Each of its three fields moves the seed.
        //
        // Fixture state:
        //     stacked arity  11
        //     selectors      reversed
        //     binding order  prefix
        let digest = |binding: LayoutBinding| {
            seed_digest(&VirtualShape::new(binding).domain_separator::<F, EF>())
        };

        let seeds = [
            ("baseline", digest(base_binding())),
            // Stacked arity: the width the drawn point is expanded to.
            (
                "stacked arity",
                digest(LayoutBinding::new(
                    12,
                    LayoutStrategy::new(true, VariableOrder::Prefix),
                )),
            ),
            // Selector placement.
            (
                "selector placement",
                digest(LayoutBinding::new(
                    11,
                    LayoutStrategy::new(false, VariableOrder::Prefix),
                )),
            ),
            // Residual binding order.
            (
                "variable order",
                digest(LayoutBinding::new(
                    11,
                    LayoutStrategy::new(true, VariableOrder::Suffix),
                )),
            ),
        ];

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn every_knob_of_the_batching_challenge_reaches_the_seed() {
        // Invariant:
        //     The claim counts decide which power lands on which claim.
        //     Both counts therefore move the seed, alongside the geometry.
        //
        // Fixture state:
        //     stacked arity  11
        //     selectors      reversed
        //     binding order  prefix
        //     concrete       4
        //     out-of-domain  2
        let digest = |binding: LayoutBinding, claims: usize, virtuals: usize| {
            seed_digest(&BatchingShape::new(binding, claims, virtuals).domain_separator::<F, EF>())
        };

        let seeds = [
            ("baseline", digest(base_binding(), 4, 2)),
            // Stacked arity.
            (
                "stacked arity",
                digest(
                    LayoutBinding::new(12, LayoutStrategy::new(true, VariableOrder::Prefix)),
                    4,
                    2,
                ),
            ),
            // Selector placement.
            (
                "selector placement",
                digest(
                    LayoutBinding::new(11, LayoutStrategy::new(false, VariableOrder::Prefix)),
                    4,
                    2,
                ),
            ),
            // Residual binding order.
            (
                "variable order",
                digest(
                    LayoutBinding::new(11, LayoutStrategy::new(true, VariableOrder::Suffix)),
                    4,
                    2,
                ),
            ),
            // One more concrete opening shifts every out-of-domain power by one.
            ("concrete count", digest(base_binding(), 5, 2)),
            // One more out-of-domain claim extends the power sequence.
            ("out-of-domain count", digest(base_binding(), 4, 3)),
        ];

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn the_three_components_of_one_layout_stay_on_distinct_seeds() {
        // Invariant:
        //     One column of a full-width table draws one point and absorbs one value.
        //     An out-of-domain claim does exactly the same.
        //     Distinct protocol names are what keep the two apart.
        //
        // Fixture state:
        //     shared geometry  stacked arity 11, selectors reversed, prefix binding
        //     the batch        table 0, arity 11, 1 direct opening, no successor
        let binding = base_binding();
        let single_column = OpeningShape::new(binding, 0, 11, 1, 0, PointSource::Drawn);

        let seeds = [
            (
                "one-column batch",
                seed_digest(&single_column.domain_separator::<F, EF>()),
            ),
            (
                "out-of-domain claim",
                seed_digest(&VirtualShape::new(binding).domain_separator::<F, EF>()),
            ),
            (
                "batching challenge",
                seed_digest(&BatchingShape::new(binding, 1, 1).domain_separator::<F, EF>()),
            ),
        ];

        assert_seeds_pairwise_distinct(&seeds);
    }

    /// Replay one batch on the verifier driver.
    ///
    /// Returns the drawn point and the sponge state the caller would continue from.
    fn replay_opening(shape: OpeningShape, evals: &OpeningEvals<EF>) -> (Point<EF>, F) {
        let mut challenger = fresh_challenger();
        let mut transcript = OpeningVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let point = transcript.point();
        transcript.evaluations(evals).unwrap();
        transcript.finish();
        // This is the sponge state the surrounding protocol continues from.
        (point, CanSample::<F>::sample(&mut challenger))
    }

    #[test]
    fn both_sides_of_a_recorded_batch_land_on_one_point_and_one_state() {
        // Invariant:
        //     Prover and verifier walk one description.
        //     Both hand the sponge back in one state.
        //     The surrounding protocol then stays in lockstep.
        //
        // Fixture state:
        //     table 1, arity 9, 3 direct openings, 1 successor opening.
        let shape = base_opening();
        let evals = base_evals();

        let mut prover_challenger = fresh_challenger();
        let mut prover = OpeningProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape);
        let prover_point = prover.point();
        prover.evaluations(&evals);
        prover.finish();
        let prover_state = CanSample::<F>::sample(&mut prover_challenger);

        let (verifier_point, verifier_state) = replay_opening(shape, &evals);

        assert_eq!(prover_point, verifier_point);
        assert_eq!(prover_state, verifier_state);
    }

    #[test]
    fn a_perturbed_direct_evaluation_moves_the_state_the_caller_continues_from() {
        // Invariant:
        //     Every direct evaluation is absorbed.
        //     Changing one moves every later challenge.
        //
        // Fixture state:
        //     3 direct openings [1, 2, 3], 1 successor opening [4].
        //
        // Mutation:
        //     direct group: [1, 2, 3]  ->  [1, 5, 3]
        //                       ^                ^
        //     -> the state handed back must differ
        let shape = base_opening();
        let honest = base_evals();

        let mut tampered_current = honest.current().to_vec();
        tampered_current[1] = EF::from_u8(5);
        let tampered = OpeningBatch::new(tampered_current, honest.next().to_vec());

        assert_ne!(
            replay_opening(shape, &honest).1,
            replay_opening(shape, &tampered).1
        );
    }

    #[test]
    fn a_perturbed_successor_evaluation_moves_the_state_the_caller_continues_from() {
        // Invariant:
        //     The successor group is absorbed under its own step.
        //     It is bound as tightly as the direct group.
        //
        // Fixture state:
        //     3 direct openings [1, 2, 3], 1 successor opening [4].
        //
        // Mutation:
        //     successor group: [4]  ->  [5]
        //                       ^        ^
        //     -> the state handed back must differ
        let shape = base_opening();
        let honest = base_evals();

        let tampered = OpeningBatch::new(honest.current().to_vec(), vec![EF::from_u8(5)]);

        assert_ne!(
            replay_opening(shape, &honest).1,
            replay_opening(shape, &tampered).1
        );
    }

    #[test]
    fn a_recorded_batch_of_the_wrong_width_is_rejected() {
        // Invariant:
        //     The evaluations arrive from a proof.
        //     A count that disagrees with the schedule is a rejection.
        //     It is never a panic.
        //
        // Fixture state:
        //     described:   3 direct, 1 successor, on table 1
        //     proof holds: 2 direct, 1 successor
        //
        // Mutation:
        //     direct group: [1, 2, 3]  ->  [1, 2]
        //     -> rejected before a single value is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript =
            OpeningVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base_opening());
        let _point = transcript.point();

        let short = OpeningBatch::new(vec![EF::ONE, EF::TWO], vec![EF::from_u8(4)]);
        let err = transcript
            .evaluations(&short)
            .expect_err("a batch outside the described widths must error");

        assert_eq!(
            err,
            SumcheckError::OpeningShapeMismatch {
                table_idx: 1,
                expected_current: 3,
                expected_next: 1,
                actual_current: 2,
                actual_next: 1,
            }
        );

        // The rejection leaves the successor step unplayed.
        //
        // Absorbing the width error is what releases the completeness check.
        //
        // Without that release this drop panics on top of the error already in flight.
        drop(transcript);
    }

    #[test]
    fn a_batch_with_no_successor_group_describes_one_step_fewer() {
        // Invariant:
        //     An empty column group declares no step.
        //     A direct-only batch and a mixed batch therefore differ in shape.
        //     Neither can be replayed in the other's place.
        //
        // Fixture state:
        //     direct-only: 3 direct, 0 successor
        //     mixed:       3 direct, 1 successor
        let binding = base_binding();
        let direct_only = OpeningShape::new(binding, 1, 9, 3, 0, PointSource::Drawn);

        let seeds = [
            (
                "direct only",
                seed_digest(&direct_only.domain_separator::<F, EF>()),
            ),
            (
                "direct and successor",
                seed_digest(&base_opening().domain_separator::<F, EF>()),
            ),
        ];
        assert_seeds_pairwise_distinct(&seeds);

        // The direct-only description accepts an empty successor group.
        let evals = OpeningBatch::new(vec![EF::ONE, EF::TWO, EF::from_u8(3)], Vec::new());
        let mut challenger = fresh_challenger();
        let mut transcript =
            OpeningVerifierTranscript::<Ch, F, EF>::new(&mut challenger, direct_only);
        let _point = transcript.point();
        transcript.evaluations(&evals).unwrap();
        transcript.finish();
    }

    #[test]
    fn a_caller_fixed_point_draws_nothing_of_its_own() {
        // Invariant:
        //     A caller-supplied point plays only the evaluation steps.
        //     Such a description holds no challenge at all.
        //
        // Fixture state:
        //     table 1 of arity 9, 3 direct openings, 1 successor opening.
        let shape = OpeningShape::new(base_binding(), 1, 9, 3, 1, PointSource::Given);
        let evals = base_evals();

        let mut prover_challenger = fresh_challenger();
        let mut prover = OpeningProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape);
        prover.evaluations(&evals);
        prover.finish();

        let mut verifier_challenger = fresh_challenger();
        let mut verifier =
            OpeningVerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
        verifier.evaluations(&evals).unwrap();
        verifier.finish();

        // Both sides hand the sponge back in one state.
        assert_eq!(
            CanSample::<F>::sample(&mut prover_challenger),
            CanSample::<F>::sample(&mut verifier_challenger),
        );
    }

    #[test]
    #[should_panic(expected = "draws no point of its own")]
    fn drawing_a_point_a_batch_does_not_describe_is_a_caller_bug() {
        // Invariant:
        //     Asking for a point the description does not hold is a caller bug.
        //     It is reported loudly rather than desynchronising the sponge.
        let shape = OpeningShape::new(base_binding(), 1, 9, 3, 1, PointSource::Given);
        let mut challenger = fresh_challenger();
        let mut transcript = OpeningProverTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _ = transcript.point();
    }

    #[test]
    fn both_sides_of_an_out_of_domain_claim_land_on_one_point_and_one_state() {
        // Invariant:
        //     The point is drawn before the value is absorbed.
        //     A prover cannot then pick the value to suit the point.
        //
        // Fixture state:
        //     stacked arity 11, claimed value 7.
        let shape = VirtualShape::new(base_binding());

        let mut prover_challenger = fresh_challenger();
        let mut prover = VirtualProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape);
        let prover_point = prover.point();
        prover.evaluation(EF::from_u8(7));
        prover.finish();

        let mut verifier_challenger = fresh_challenger();
        let mut verifier =
            VirtualVerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
        let verifier_point = verifier.point();
        verifier.evaluation(EF::from_u8(7));
        verifier.finish();

        // One point, and one sponge state to continue from.
        assert_eq!(prover_point.num_variables(), 11);
        assert_eq!(prover_point, verifier_point);
        assert_eq!(
            CanSample::<F>::sample(&mut prover_challenger),
            CanSample::<F>::sample(&mut verifier_challenger),
        );
    }

    #[test]
    fn a_perturbed_out_of_domain_evaluation_moves_the_state_the_caller_continues_from() {
        // Invariant:
        //     The claimed value is absorbed.
        //     Changing it moves every later challenge.
        //
        // Fixture state:
        //     stacked arity 11, one claimed value.
        //
        // Mutation:
        //     claimed value: 7  ->  8
        //     -> the state handed back must differ
        let shape = VirtualShape::new(base_binding());

        let replay = |eval: EF| {
            let mut challenger = fresh_challenger();
            let mut transcript =
                VirtualVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
            let _point = transcript.point();
            transcript.evaluation(eval);
            transcript.finish();
            CanSample::<F>::sample(&mut challenger)
        };

        assert_ne!(replay(EF::from_u8(7)), replay(EF::from_u8(8)));
    }

    #[test]
    fn both_sides_of_the_batching_challenge_draw_the_same_value() {
        // Invariant:
        //     The two drivers walk one description and land on one challenge.
        //
        // Fixture state:
        //     4 concrete openings, 2 out-of-domain claims.
        let shape = BatchingShape::new(base_binding(), 4, 2);

        let mut prover_challenger = fresh_challenger();
        let prover: EF = prover_batching_challenge::<Ch, F, EF>(&mut prover_challenger, shape);

        let mut verifier_challenger = fresh_challenger();
        let verifier: EF =
            verifier_batching_challenge::<Ch, F, EF>(&mut verifier_challenger, shape);

        assert_eq!(prover, verifier);

        // The sponge is handed back in one state, so the residual fold stays in step.
        assert_eq!(
            CanSample::<F>::sample(&mut prover_challenger),
            CanSample::<F>::sample(&mut verifier_challenger),
        );
    }
}
