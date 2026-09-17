//! Fiat-Shamir transcript of one bit-alphabet ring-switching reduction.
//!
//! ```text
//!     evaluation point   ->  one element per coordinate
//!     tensor rows        ->  one element per row of the bit matrix
//!     batching point     ->  one element per absorbed coordinate, drawn
//!     sumcheck           ->  a bracket around a delegated run
//!     surviving claim    ->  one element
//! ```
//!
//! # What the order settles
//!
//! The batching draw comes after the tensor rows, and nothing else does.
//! That is the ordering the reduction's soundness rests on, so it lives here.
//!
//! A bit matrix solving two `F_2`-linear systems moves the claim with the sum held.
//! Drawing the challenge first would let a prover pick such a matrix.
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//! The coordinate count moves the first width, the level fixes the other two.
//!
//! No grinding step is described, so every challenge here is resampleable.

use alloc::vec;
use core::marker::PhantomData;

use p3_binary_field::TowerLevel;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_multilinear_util::point::Point;

use super::basis::Coefficients;
use super::reduction::BitRingSwitch;

/// Version byte bound into the seed, separating two revisions of this protocol.
const VERSION: u8 = 1;

/// Protocol name bound into the seed, distinct from the general reduction's.
const NAME: &[u8] = b"p3-sumcheck-bit-ring-switch";

/// Step label of the point the incoming evaluation claim is stated at.
const EVALUATION_POINT: &str = "evaluation_point";

/// Step label of the rows of the shared tensor element.
const TENSOR_ROWS: &str = "tensor_rows";

/// Step label of the challenges that collapse the row claims into one.
const BATCHING_POINT: &str = "batching_point";

/// Step label of the bracket around the delegated sumcheck.
const BATCHED_SUMCHECK: &str = "batched_sumcheck";

/// Step label of the value the surviving claim carries.
const SURVIVING_CLAIM: &str = "surviving_claim";

/// Sponge alphabet of a challenger that speaks the level natively.
type Alphabet<EF> = FieldUnit<EF>;

/// Type-level name of the sub-protocol the reduction delegates its rounds to.
struct BatchedSumcheck;

/// Numbers that fix the transcript of one bit-alphabet reduction.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitRingSwitchShape {
    /// Coordinates the incoming evaluation point names.
    pub num_variables: usize,
}

impl BitRingSwitchShape {
    /// Describe a reduction of a claim at a point of this width.
    #[must_use]
    pub const fn new(num_variables: usize) -> Self {
        Self { num_variables }
    }

    /// Rounds the delegated sumcheck runs, panicking on a point narrower than one element.
    #[must_use]
    pub fn sumcheck_rounds<EF: TowerLevel>(&self) -> usize {
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        assert!(
            self.num_variables >= absorbed,
            "the evaluation point must name at least the {absorbed} absorbed coordinates"
        );
        self.num_variables - absorbed
    }

    /// Describe every step of one reduction, which one matched bracket always validates.
    #[must_use]
    pub fn pattern<EF: TranscriptField + TowerLevel>(&self) -> InteractionPattern {
        let steps = vec![
            // The point comes first, so no later draw is chosen before the claim is placed.
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                EVALUATION_POINT,
                Length::Fixed(self.num_variables),
            ),
            // The rows are the whole element, one bit per entry.
            // Binding them therefore binds both readings of it.
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                TENSOR_ROWS,
                Length::Fixed(Coefficients::<EF>::DIMENSION),
            ),
            // One challenge per absorbed coordinate collapses the rows into a single claim.
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BATCHING_POINT,
                Length::Fixed(BitRingSwitch::<EF>::ABSORBED),
            ),
            // The bracket records that a sub-protocol runs here, under its own seed.
            Interaction::marker::<BatchedSumcheck>(
                Hierarchy::Begin,
                Kind::Protocol,
                BATCHED_SUMCHECK,
            ),
            Interaction::marker::<BatchedSumcheck>(
                Hierarchy::End,
                Kind::Protocol,
                BATCHED_SUMCHECK,
            ),
            // The value left behind is bound before the sponge goes back.
            // That is what ties whatever discharges it to this run.
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                SURVIVING_CLAIM,
                Length::Scalar,
            ),
        ];

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and this shape into a seed.
    #[must_use]
    pub fn domain_separator<EF: TranscriptField + TowerLevel>(
        &self,
    ) -> DomainSeparator<Alphabet<EF>> {
        DomainSeparator::new(VERSION, NAME, self.pattern::<EF>())
    }
}

/// Prover-side transcript of one bit-alphabet reduction.
///
/// The challenger is borrowed, because a reduction always runs inside something larger.
pub struct BitRingSwitchProverTranscript<'a, C, EF: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<EF>>,
}

impl<'a, C, EF> BitRingSwitchProverTranscript<'a, C, EF>
where
    EF: TranscriptField + TowerLevel,
    C: CanObserve<EF> + CanSample<EF>,
{
    /// Seed the transcript from the shape, folding its fingerprint into the sponge.
    pub fn new(challenger: &'a mut C, shape: BitRingSwitchShape) -> Self {
        Self {
            state: ProverState::new(challenger, &shape.domain_separator::<EF>()),
        }
    }

    /// Bind the claim's point and the element's rows, then draw the batching challenges.
    ///
    /// # Returns
    ///
    /// The batching point, one challenge per absorbed coordinate.
    ///
    /// # Panics
    ///
    /// When either list is not the width the run was described with.
    pub fn statement(&mut self, point: &Point<EF>, rows: &[EF]) -> Point<EF> {
        // The point is bound first, so the element cannot be chosen to suit it.
        self.state
            .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                EVALUATION_POINT,
                point.as_slice(),
            );
        self.state
            .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_ROWS, rows);

        // Only now is the batching point drawn, so no message above it knew it.
        Point::new(
            self.state
                .challenge_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    BATCHING_POINT,
                    BitRingSwitch::<EF>::ABSORBED,
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        )
    }

    /// Hand the sponge to the delegated sumcheck rounds.
    pub fn batched_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        output
    }

    /// Bind the value the surviving claim carries.
    pub fn surviving_claim(&mut self, value: EF) {
        self.state
            .observe_extension::<EF, EF, FieldToFieldCodec<EF>>(SURVIVING_CLAIM, &value);
    }

    /// Close the transcript, panicking unless every described step was played.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "a bit-alphabet reduction carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one bit-alphabet reduction.
///
/// Mirrors the prover side call for call, over the same description.
pub struct BitRingSwitchVerifierTranscript<'a, C, EF: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    state: VerifierState<'static, &'a mut C, Alphabet<EF>>,
    /// The numbers this run was described with.
    shape: BitRingSwitchShape,
    /// Marker keeping the level out of the driver's own signature.
    _level: PhantomData<EF>,
}

impl<'a, C, EF> BitRingSwitchVerifierTranscript<'a, C, EF>
where
    EF: TranscriptField + TowerLevel,
    C: CanObserve<EF> + CanSample<EF>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: BitRingSwitchShape) -> Self {
        Self {
            state: VerifierState::new(challenger, &shape.domain_separator::<EF>(), &[]),
            shape,
            _level: PhantomData,
        }
    }

    /// Replay the statement, then redraw the batching challenges.
    ///
    /// # Errors
    ///
    /// Returns an error unless both lists are the described width, releasing the driver.
    pub fn statement(
        &mut self,
        point: &Point<EF>,
        rows: &[EF],
    ) -> Result<Point<EF>, TranscriptWidth> {
        if point.num_variables() != self.shape.num_variables {
            self.state.abort();
            return Err(TranscriptWidth::Point {
                expected: self.shape.num_variables,
                actual: point.num_variables(),
            });
        }
        if rows.len() != Coefficients::<EF>::DIMENSION {
            self.state.abort();
            return Err(TranscriptWidth::TensorRows {
                expected: Coefficients::<EF>::DIMENSION,
                actual: rows.len(),
            });
        }

        let _ = self
            .state
            .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                EVALUATION_POINT,
                point.as_slice(),
            );
        let _ = self
            .state
            .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_ROWS, rows);

        Ok(Point::new(
            self.state
                .challenge_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    BATCHING_POINT,
                    BitRingSwitch::<EF>::ABSORBED,
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        ))
    }

    /// Hand the sponge to the delegated sumcheck replay.
    pub fn batched_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        output
    }

    /// Replay the value the surviving claim carries.
    pub fn surviving_claim(&mut self, value: EF) {
        self.state
            .observe_extension::<EF, EF, FieldToFieldCodec<EF>>(SURVIVING_CLAIM, &value);
    }

    /// Release the completeness check because the proof is being rejected.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript, panicking unless every described step was replayed.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("a bit-alphabet reduction reads an empty wire");
    }
}

/// A list whose width does not match the one the run was described with.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranscriptWidth {
    /// The evaluation point names the wrong number of coordinates.
    #[error("the evaluation point names {actual} coordinates, expected {expected}")]
    Point {
        /// Coordinates the description fixes.
        expected: usize,
        /// Coordinates supplied.
        actual: usize,
    },
    /// The tensor element carries the wrong number of rows.
    #[error("the tensor element carries {actual} rows, expected {expected}")]
    TensorRows {
        /// Rows the level's dimension fixes.
        expected: usize,
        /// Rows supplied.
        actual: usize,
    },
}
