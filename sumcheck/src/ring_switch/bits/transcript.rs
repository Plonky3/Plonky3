//! Fiat-Shamir transcript of one bit-alphabet ring-switching reduction.
//!
//! ```text
//!     evaluation point   ->  one element per coordinate
//!     tensor rows        ->  one element per row of the bit matrix
//!     carry rows         ->  the same, only when the shape has successor rows
//!     last rows          ->  the same, only when the shape has successor rows
//!     batching point     ->  one element per absorbed coordinate, drawn
//!     tensor batching    ->  one element, drawn, only when the shape has successor rows
//!     sumcheck           ->  a bracket around a delegated run
//!     surviving claim    ->  one element
//! ```
//!
//! # What the order settles
//!
//! The batching draws come after every element's rows, and nothing else does.
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
//! The successor row count is bound as an instance label.
//! It moves no width, yet it decides which successor the column check reads.
//!
//! No grinding step is described, so every challenge here is resampleable.

use alloc::vec;
use alloc::vec::Vec;
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
pub(crate) const NAME: &[u8] = b"p3-sumcheck-bit-ring-switch";

/// Step label of the point the incoming evaluation claim is stated at.
const EVALUATION_POINT: &str = "evaluation_point";

/// Step label of the rows of the shared tensor element.
const TENSOR_ROWS: &str = "tensor_rows";

/// Step label of the rows of the element carrying the successor into the next element.
const CARRY_ROWS: &str = "carry_rows";

/// Step label of the rows of the element holding the repeating last row.
const LAST_ROWS: &str = "last_rows";

/// Step label of the challenges that collapse the row claims into one.
const BATCHING_POINT: &str = "batching_point";

/// Step label of the challenge that batches the successor elements with the tensor.
const TENSOR_BATCHING: &str = "tensor_batching";

/// Elements a successor view adds to the statement when its rows outrun one element.
const SUCCESSOR_ELEMENTS: usize = 2;

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
    /// Row coordinates of a successor view whose rows outrun one element.
    ///
    /// Set only when the reduction sends the two successor elements.
    /// A successor view inside one element is a column reading, and plays the plain run.
    pub successor_rows: Option<usize>,
}

impl BitRingSwitchShape {
    /// Describe a reduction of a claim at a point of this width.
    #[must_use]
    pub const fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            successor_rows: None,
        }
    }

    /// Describe a reduction that also sends the two successor elements.
    ///
    /// For a successor view stepping within the trailing `row_variables` coordinates.
    /// Only a view with more row coordinates than one element absorbs sends them.
    #[must_use]
    pub const fn with_successor_rows(num_variables: usize, row_variables: usize) -> Self {
        Self {
            num_variables,
            successor_rows: Some(row_variables),
        }
    }

    /// Maximum delegated rounds for this point width.
    ///
    /// A Boolean prefix of length `p` leaves `l' - p` rounds from this maximum `l'`.
    ///
    /// # Panics
    ///
    /// Panics when the point is narrower than one element.
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
        let rows = |label| {
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                label,
                Length::Fixed(Coefficients::<EF>::DIMENSION),
            )
        };
        let successor = self.successor_rows.is_some();

        let mut steps = vec![
            // The point comes first, so no later draw is chosen before the claim is placed.
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                EVALUATION_POINT,
                Length::Fixed(self.num_variables),
            ),
            // The rows are the whole element, one bit per entry.
            // Binding them therefore binds both readings of it.
            rows(TENSOR_ROWS),
        ];
        // The successor elements are bound beside the tensor, before any draw.
        if successor {
            steps.extend([rows(CARRY_ROWS), rows(LAST_ROWS)]);
        }
        // One challenge per absorbed coordinate collapses the rows into a single claim.
        steps.push(Interaction::algebra::<EF, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BATCHING_POINT,
            Length::Fixed(BitRingSwitch::<EF>::ABSORBED),
        ));
        // One more challenge collapses the three elements' batched rows into one sum.
        if successor {
            steps.push(Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                TENSOR_BATCHING,
                Length::Scalar,
            ));
        }
        steps.extend([
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
        ]);

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and this shape into a seed.
    ///
    /// The successor row count, when there is one, is an instance label.
    /// A plain shape adds none, so its seed is the protocol's and the pattern's alone.
    #[must_use]
    pub fn domain_separator<EF: TranscriptField + TowerLevel>(
        &self,
    ) -> DomainSeparator<Alphabet<EF>> {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<EF>());
        if let Some(rows) = self.successor_rows {
            separator.instance(&(rows as u64).to_be_bytes());
        }
        separator
    }
}

/// Prover-side transcript of one bit-alphabet reduction.
///
/// The challenger is borrowed, because a reduction always runs inside something larger.
pub struct BitRingSwitchProverTranscript<'a, C, EF: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<EF>>,
    /// The numbers this run was described with.
    shape: BitRingSwitchShape,
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
            shape,
        }
    }

    /// Bind the claim's point and every element's rows, then draw the batching challenges.
    ///
    /// # Arguments
    ///
    /// - The evaluation point of the claim.
    /// - The rows of the tensor element.
    /// - The rows of the carry and last elements, when the shape has successor rows.
    ///
    /// # Returns
    ///
    /// - The batching point, one challenge per absorbed coordinate.
    /// - The challenge batching the three elements, when the shape has successor rows.
    ///
    /// # Panics
    ///
    /// - When the presence of the successor rows disagrees with the shape.
    /// - When a list is not the width the run was described with.
    pub fn statement(
        &mut self,
        point: &Point<EF>,
        rows: &[EF],
        successor: Option<(&[EF], &[EF])>,
    ) -> (Point<EF>, Option<EF>) {
        assert_eq!(
            successor.is_some(),
            self.shape.successor_rows.is_some(),
            "the successor elements are sent exactly when the shape describes them",
        );

        // The point is bound first, so the element cannot be chosen to suit it.
        self.state
            .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                EVALUATION_POINT,
                point.as_slice(),
            );
        self.state
            .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_ROWS, rows);
        if let Some((carry, last)) = successor {
            self.state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(CARRY_ROWS, carry);
            self.state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(LAST_ROWS, last);
        }

        // Only now are the batching challenges drawn, so no message above them knew them.
        let batching_point = Point::new(
            self.state
                .challenge_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    BATCHING_POINT,
                    BitRingSwitch::<EF>::ABSORBED,
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        );
        let alpha = successor.map(|_| {
            self.state
                .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_BATCHING)
                .into_inner()
        });
        (batching_point, alpha)
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
    /// Takes the same arguments as the prover side and returns the same draws.
    ///
    /// # Errors
    ///
    /// Returns an error, releasing the driver, before anything is absorbed:
    ///
    /// - when the point or a row list is not the described width
    /// - when the presence of the successor rows disagrees with the shape
    pub fn statement(
        &mut self,
        point: &Point<EF>,
        rows: &[EF],
        successor: Option<(&[EF], &[EF])>,
    ) -> Result<(Point<EF>, Option<EF>), TranscriptWidth> {
        if let Err(error) = self.check_widths(point, rows, successor) {
            self.state.abort();
            return Err(error);
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
        if let Some((carry, last)) = successor {
            let _ = self
                .state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(CARRY_ROWS, carry);
            let _ = self
                .state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(LAST_ROWS, last);
        }

        let batching_point = Point::new(
            self.state
                .challenge_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    BATCHING_POINT,
                    BitRingSwitch::<EF>::ABSORBED,
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        );
        let alpha = successor.map(|_| {
            self.state
                .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_BATCHING)
                .into_inner()
        });
        Ok((batching_point, alpha))
    }

    /// Check every list of the statement against the description, absorbing nothing.
    fn check_widths(
        &self,
        point: &Point<EF>,
        rows: &[EF],
        successor: Option<(&[EF], &[EF])>,
    ) -> Result<(), TranscriptWidth> {
        check_statement_widths(&self.shape, point, rows, successor)
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

/// Check one claim's statement against the shape it was described with, absorbing nothing.
fn check_statement_widths<EF: TowerLevel>(
    shape: &BitRingSwitchShape,
    point: &Point<EF>,
    rows: &[EF],
    successor: Option<(&[EF], &[EF])>,
) -> Result<(), TranscriptWidth> {
    if point.num_variables() != shape.num_variables {
        return Err(TranscriptWidth::Point {
            expected: shape.num_variables,
            actual: point.num_variables(),
        });
    }
    // Every element is a square bit matrix, so every row list has the level's dimension.
    let successor_rows = successor
        .into_iter()
        .flat_map(|(carry, last)| [carry, last]);
    if let Some(malformed) = core::iter::once(rows)
        .chain(successor_rows)
        .find(|rows| rows.len() != Coefficients::<EF>::DIMENSION)
    {
        return Err(TranscriptWidth::TensorRows {
            expected: Coefficients::<EF>::DIMENSION,
            actual: malformed.len(),
        });
    }

    let expected = shape.successor_rows.is_some();
    if successor.is_some() == expected {
        Ok(())
    } else {
        Err(TranscriptWidth::successor_elements(
            expected,
            successor.is_some(),
        ))
    }
}

/// Protocol name of a batch of claims, distinct from the one-claim run's.
pub(crate) const CLAIMS_NAME: &[u8] = b"p3-sumcheck-bit-ring-switch-claims";

/// Step label of the challenge that folds the claims' batched rows into one sum.
const CLAIM_BATCHING: &str = "claim_batching";

/// What one claim of a batch binds before any batching draw.
#[derive(Clone, Copy, Debug)]
pub struct ClaimStatement<'a, EF> {
    /// The evaluation point of the claim.
    pub point: &'a Point<EF>,
    /// The rows of its tensor element.
    pub rows: &'a [EF],
    /// The rows of its carry and last elements, when its shape has successor rows.
    pub successor: Option<(&'a [EF], &'a [EF])>,
}

/// The challenges a batch of claims draws once every claim is bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimsDraws<EF> {
    /// One challenge per absorbed coordinate, shared by every claim.
    pub batching_point: Point<EF>,
    /// The challenge batching successor elements, when some claim sends them.
    pub alpha: Option<EF>,
    /// The challenge whose powers weigh the claims, claim `i` by `lambda^i`.
    pub lambda: EF,
}

/// Numbers that fix the transcript of a batch of at least two bit-alphabet claims.
///
/// ```text
///     per claim        point, tensor rows [, carry rows, last rows]
///     then, once       batching point, [tensor batching,] claim batching
///     then, once       sumcheck bracket, surviving claim
/// ```
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitRingSwitchClaimsShape {
    /// The shape of each claim, in the order the claims are bound.
    pub claims: Vec<BitRingSwitchShape>,
}

impl BitRingSwitchClaimsShape {
    /// Whether some claim sends successor elements, so the shared `alpha` is drawn.
    #[must_use]
    pub fn sends_successor(&self) -> bool {
        self.claims
            .iter()
            .any(|claim| claim.successor_rows.is_some())
    }

    /// Describe every step of one batched run, which one matched bracket always validates.
    #[must_use]
    pub fn pattern<EF: TranscriptField + TowerLevel>(&self) -> InteractionPattern {
        let rows = |label| {
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                label,
                Length::Fixed(Coefficients::<EF>::DIMENSION),
            )
        };
        let scalar_challenge = |label| {
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                label,
                Length::Scalar,
            )
        };

        let mut steps = Vec::new();
        // Every claim is bound whole before any draw, which is what the batching bound needs.
        for claim in &self.claims {
            steps.push(Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                EVALUATION_POINT,
                Length::Fixed(claim.num_variables),
            ));
            steps.push(rows(TENSOR_ROWS));
            if claim.successor_rows.is_some() {
                steps.extend([rows(CARRY_ROWS), rows(LAST_ROWS)]);
            }
        }
        // One batching point shared by every claim collapses each claim's rows.
        steps.push(Interaction::algebra::<EF, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BATCHING_POINT,
            Length::Fixed(BitRingSwitch::<EF>::ABSORBED),
        ));
        if self.sends_successor() {
            steps.push(scalar_challenge(TENSOR_BATCHING));
        }
        // Powers of one more challenge fold the claims' sums into the one the rounds prove.
        steps.push(scalar_challenge(CLAIM_BATCHING));
        steps.extend([
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
            Interaction::algebra::<EF, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                SURVIVING_CLAIM,
                Length::Scalar,
            ),
        ]);

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and this shape into a seed.
    ///
    /// Each claim's successor row count is an instance label, as in the one-claim run.
    /// An empty label marks a claim without successor rows, so no two shapes share a seed.
    #[must_use]
    pub fn domain_separator<EF: TranscriptField + TowerLevel>(
        &self,
    ) -> DomainSeparator<Alphabet<EF>> {
        let mut separator = DomainSeparator::new(VERSION, CLAIMS_NAME, self.pattern::<EF>());
        separator.instance(&(self.claims.len() as u64).to_be_bytes());
        for claim in &self.claims {
            match claim.successor_rows {
                Some(rows) => separator.instance(&(rows as u64).to_be_bytes()),
                None => separator.instance(&[]),
            };
        }
        separator
    }
}

/// Prover-side transcript of a batch of bit-alphabet claims.
pub struct BitRingSwitchClaimsProverTranscript<'a, C, EF: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<EF>>,
    /// The numbers this run was described with.
    shape: BitRingSwitchClaimsShape,
}

impl<'a, C, EF> BitRingSwitchClaimsProverTranscript<'a, C, EF>
where
    EF: TranscriptField + TowerLevel,
    C: CanObserve<EF> + CanSample<EF>,
{
    /// Seed the transcript from the shape, folding its fingerprint into the sponge.
    pub fn new(challenger: &'a mut C, shape: BitRingSwitchClaimsShape) -> Self {
        Self {
            state: ProverState::new(challenger, &shape.domain_separator::<EF>()),
            shape,
        }
    }

    /// Bind every claim's point and elements, then draw the batching challenges.
    ///
    /// # Panics
    ///
    /// - When the claim count or a claim's successor rows disagree with the shape.
    /// - When a list is not the width the run was described with.
    pub fn statement(&mut self, claims: &[ClaimStatement<'_, EF>]) -> ClaimsDraws<EF> {
        assert_eq!(
            claims.len(),
            self.shape.claims.len(),
            "one statement per claim the shape describes"
        );
        for (claim, shape) in claims.iter().zip(&self.shape.claims) {
            assert_eq!(
                claim.successor.is_some(),
                shape.successor_rows.is_some(),
                "the successor elements are sent exactly when the shape describes them",
            );
            self.state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    EVALUATION_POINT,
                    claim.point.as_slice(),
                );
            self.state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_ROWS, claim.rows);
            if let Some((carry, last)) = claim.successor {
                self.state
                    .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(CARRY_ROWS, carry);
                self.state
                    .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(LAST_ROWS, last);
            }
        }
        // Only now are the batching challenges drawn, so no message above them knew them.
        let batching_point = Point::new(
            self.state
                .challenge_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    BATCHING_POINT,
                    BitRingSwitch::<EF>::ABSORBED,
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        );
        let alpha = self.shape.sends_successor().then(|| {
            self.state
                .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_BATCHING)
                .into_inner()
        });
        let lambda = self
            .state
            .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(CLAIM_BATCHING)
            .into_inner();
        ClaimsDraws {
            batching_point,
            alpha,
            lambda,
        }
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
            "a batch of bit-alphabet claims carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of a batch of bit-alphabet claims.
///
/// Mirrors the prover side call for call, over the same description.
pub struct BitRingSwitchClaimsVerifierTranscript<'a, C, EF: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    state: VerifierState<'static, &'a mut C, Alphabet<EF>>,
    /// The numbers this run was described with.
    shape: BitRingSwitchClaimsShape,
}

impl<'a, C, EF> BitRingSwitchClaimsVerifierTranscript<'a, C, EF>
where
    EF: TranscriptField + TowerLevel,
    C: CanObserve<EF> + CanSample<EF>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: BitRingSwitchClaimsShape) -> Self {
        Self {
            state: VerifierState::new(challenger, &shape.domain_separator::<EF>(), &[]),
            shape,
        }
    }

    /// Replay every claim's statement, then redraw the batching challenges.
    ///
    /// # Errors
    ///
    /// Returns an error, releasing the driver, before anything is absorbed:
    ///
    /// - when the claim count disagrees with the shape
    /// - when a point or a row list is not the described width
    /// - when a claim's successor rows disagree with its shape
    pub fn statement(
        &mut self,
        claims: &[ClaimStatement<'_, EF>],
    ) -> Result<ClaimsDraws<EF>, TranscriptWidth> {
        let checked = if claims.len() == self.shape.claims.len() {
            claims
                .iter()
                .zip(&self.shape.claims)
                .try_for_each(|(claim, shape)| {
                    check_statement_widths(shape, claim.point, claim.rows, claim.successor)
                })
        } else {
            Err(TranscriptWidth::Claims {
                expected: self.shape.claims.len(),
                actual: claims.len(),
            })
        };
        if let Err(error) = checked {
            self.state.abort();
            return Err(error);
        }

        for claim in claims {
            let _ = self
                .state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    EVALUATION_POINT,
                    claim.point.as_slice(),
                );
            let _ = self
                .state
                .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_ROWS, claim.rows);
            if let Some((carry, last)) = claim.successor {
                let _ = self
                    .state
                    .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(CARRY_ROWS, carry);
                let _ = self
                    .state
                    .observe_extensions::<EF, EF, FieldToFieldCodec<EF>>(LAST_ROWS, last);
            }
        }
        // Only now are the batching challenges drawn, so no message above them knew them.
        let batching_point = Point::new(
            self.state
                .challenge_extensions::<EF, EF, FieldToFieldCodec<EF>>(
                    BATCHING_POINT,
                    BitRingSwitch::<EF>::ABSORBED,
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        );
        let alpha = self.shape.sends_successor().then(|| {
            self.state
                .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(TENSOR_BATCHING)
                .into_inner()
        });
        let lambda = self
            .state
            .challenge_extension::<EF, EF, FieldToFieldCodec<EF>>(CLAIM_BATCHING)
            .into_inner();
        Ok(ClaimsDraws {
            batching_point,
            alpha,
            lambda,
        })
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
            .expect("a batch of bit-alphabet claims reads an empty wire");
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
    /// A tensor element carries the wrong number of rows.
    #[error("the tensor element carries {actual} rows, expected {expected}")]
    TensorRows {
        /// Rows the level's dimension fixes.
        expected: usize,
        /// Rows supplied.
        actual: usize,
    },
    /// A batch binds a different number of claims than its description.
    #[error("the batch binds {actual} claims, expected {expected}")]
    Claims {
        /// Claims the description fixes.
        expected: usize,
        /// Claims supplied.
        actual: usize,
    },
    /// The presence of the successor elements disagrees with the description.
    ///
    /// The count is zero or two, one carry element and one last element.
    #[error("the statement carries {actual} successor elements, expected {expected}")]
    SuccessorElements {
        /// Elements the description fixes.
        expected: usize,
        /// Elements supplied.
        actual: usize,
    },
}

impl TranscriptWidth {
    /// The successor-element mismatch, from whether each side has the elements.
    #[must_use]
    pub(crate) const fn successor_elements(expected: bool, actual: bool) -> Self {
        const fn count(present: bool) -> usize {
            if present { SUCCESSOR_ELEMENTS } else { 0 }
        }
        Self::SuccessorElements {
            expected: count(expected),
            actual: count(actual),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField16};
    use p3_challenger::HashChallenger;
    use p3_challenger::testing::{
        SeedDigest, assert_seeds_pairwise_distinct, pow_difficulties, seed_digest,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;

    use super::*;

    type EF = BinaryField16;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

    /// Coordinate count every run below is described with.
    ///
    /// A 16-bit level absorbs four of them, leaving four sumcheck rounds.
    const NUM_VARIABLES: usize = 8;

    /// Rows a 16-bit level's tensor element carries, one per basis coordinate.
    const NUM_ROWS: usize = 16;

    fn fresh_challenger() -> Chal {
        // A fresh sponge, so two runs differ only where the transcript makes them differ.
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// Row coordinates of the successor shape, two more than one element absorbs.
    const SUCCESSOR_ROWS: usize = 6;

    /// Baseline shape every mutation below is measured against.
    const fn base_shape() -> BitRingSwitchShape {
        BitRingSwitchShape::new(NUM_VARIABLES)
    }

    /// A shape whose reduction sends the two successor elements.
    const fn successor_shape() -> BitRingSwitchShape {
        BitRingSwitchShape::with_successor_rows(NUM_VARIABLES, SUCCESSOR_ROWS)
    }

    /// The digest of the byte stream a shape seeds its sponge with.
    ///
    /// Comparing seed streams keeps the sponge out of the separation claim.
    fn seed_of(shape: BitRingSwitchShape) -> SeedDigest {
        seed_digest(&shape.domain_separator::<EF>())
    }

    /// Every field of the shape, each moved one step away from the baseline.
    ///
    /// A field added to the shape stops the destructuring below from compiling.
    fn one_step_from_base() -> Vec<(&'static str, BitRingSwitchShape)> {
        // Exhaustiveness check: every field named, none elided by a rest pattern.
        let BitRingSwitchShape {
            num_variables: _,
            successor_rows: _,
        } = base_shape();

        // One more coordinate widens the step that binds the point.
        let mut wider = base_shape();
        wider.num_variables += 1;

        // Two successor shapes differing only in the row count.
        // The pattern is the same for both, so only the instance binding can part them.
        alloc::vec![
            ("num_variables", wider),
            ("successor_rows", successor_shape()),
            (
                "successor_rows_count",
                BitRingSwitchShape::with_successor_rows(NUM_VARIABLES, SUCCESSOR_ROWS + 1),
            ),
        ]
    }

    /// A point whose coordinates are distinct and seed-dependent.
    fn point_of(num_variables: usize, seed: u32) -> Point<EF> {
        Point::new(
            (0..num_variables)
                .map(|i| EF::from_u32(seed * 31 + i as u32 + 1))
                .collect(),
        )
    }

    /// A full-width row list whose entries are distinct and seed-dependent.
    fn rows_of(seed: u32) -> Vec<EF> {
        (0..NUM_ROWS)
            .map(|i| EF::from_u32(seed * 17 + i as u32 + 1))
            .collect()
    }

    /// Everything one run produces, in the order the description fixes.
    ///
    /// The batching point comes first, so a test can read it on its own.
    /// The second draw batches the successor elements, when the shape sends them.
    type Draws = (Point<EF>, Option<EF>, EF);

    /// The carry and last rows a statement binds, when its shape has successor rows.
    type SuccessorRows<'a> = Option<(&'a [EF], &'a [EF])>;

    /// Drive a full prover-side run and hand back the draws it produced.
    ///
    /// The delegated sumcheck is stood in for by a single draw.
    /// That keeps the bracket on the path a real delegation takes.
    fn drive_prover(
        challenger: &mut Chal,
        shape: BitRingSwitchShape,
        point: &Point<EF>,
        rows: &[EF],
        successor: SuccessorRows<'_>,
        claim: EF,
    ) -> Draws {
        let mut transcript = BitRingSwitchProverTranscript::<Chal, EF>::new(challenger, shape);
        let (batching_point, alpha) = transcript.statement(point, rows, successor);
        let delegated = transcript.batched_sumcheck(<Chal as CanSample<EF>>::sample);
        transcript.surviving_claim(claim);
        transcript.finish();
        (batching_point, alpha, delegated)
    }

    /// Replay a full verifier-side run against recorded values.
    fn drive_verifier(
        challenger: &mut Chal,
        shape: BitRingSwitchShape,
        point: &Point<EF>,
        rows: &[EF],
        successor: SuccessorRows<'_>,
        claim: EF,
    ) -> Result<Draws, TranscriptWidth> {
        let mut transcript = BitRingSwitchVerifierTranscript::<Chal, EF>::new(challenger, shape);
        let (batching_point, alpha) = transcript.statement(point, rows, successor)?;
        let delegated = transcript.batched_sumcheck(<Chal as CanSample<EF>>::sample);
        transcript.surviving_claim(claim);
        transcript.finish();
        Ok((batching_point, alpha, delegated))
    }

    /// Everything a verifier redraws from one recorded run, plus the state it hands back.
    ///
    /// The trailing draw is taken after the run closes.
    /// It exposes a value bound with nothing left inside the description to move.
    fn replay(point: &Point<EF>, rows: &[EF], claim: EF) -> (Draws, EF) {
        replay_in(
            BitRingSwitchShape::new(point.num_variables()),
            point,
            rows,
            None,
            claim,
        )
    }

    /// The same replay, over a shape and successor elements of the caller's choosing.
    fn replay_in(
        shape: BitRingSwitchShape,
        point: &Point<EF>,
        rows: &[EF],
        successor: SuccessorRows<'_>,
        claim: EF,
    ) -> (Draws, EF) {
        let mut challenger = fresh_challenger();
        let draws = drive_verifier(&mut challenger, shape, point, rows, successor, claim)
            .expect("a well-shaped run always replays");
        (draws, CanSample::<EF>::sample(&mut challenger))
    }

    #[test]
    fn the_same_shape_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the shape.
        assert_eq!(seed_of(base_shape()), seed_of(base_shape()));
    }

    #[test]
    fn no_two_configurations_of_the_shape_share_a_seed() {
        // Invariant: the knobs are separated from each other, not merely from a baseline.
        //
        //     baseline in the set    ->  every knob has to reach the seed
        //     pairwise over the set  ->  no two knobs may land on one seed
        let mut seeds = alloc::vec![("base", seed_of(base_shape()))];
        seeds.extend(
            one_step_from_base()
                .into_iter()
                .map(|(field, shape)| (field, seed_of(shape))),
        );

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn the_description_asks_for_no_grinding() {
        // Invariant: this reduction grinds nowhere.
        //
        // A step appearing here would be a cost neither side accounts for.
        assert!(pow_difficulties(&base_shape().pattern::<EF>()).is_empty());
    }

    #[test]
    fn the_two_derived_widths_come_from_the_level() {
        // Fixture state: a 16-bit level over F_2.
        //
        //     - absorbed coordinates  ->  log2(16) = 4
        //     - tensor rows           ->  16
        //     - sumcheck rounds       ->  8 - 4   = 4
        assert_eq!(BitRingSwitch::<EF>::ABSORBED, 4);
        assert_eq!(Coefficients::<EF>::DIMENSION, NUM_ROWS);
        assert_eq!(base_shape().sumcheck_rounds::<EF>(), 4);
    }

    #[test]
    fn both_sides_of_a_run_draw_the_same_stream() {
        // Completeness: the two drivers walk one description and land on one stream.
        //
        // Fixture state: 8 coordinates, 16 rows, one delegated draw.
        let point = point_of(NUM_VARIABLES, 3);
        let rows = rows_of(5);

        let mut prover_challenger = fresh_challenger();
        let proved = drive_prover(
            &mut prover_challenger,
            base_shape(),
            &point,
            &rows,
            None,
            EF::ONE,
        );

        let mut verifier_challenger = fresh_challenger();
        let replayed = drive_verifier(
            &mut verifier_challenger,
            base_shape(),
            &point,
            &rows,
            None,
            EF::ONE,
        )
        .expect("the honest run must replay");

        assert_eq!(proved, replayed);

        // The sponge is handed back in one state, so the surrounding protocol stays in step.
        assert_eq!(
            CanSample::<EF>::sample(&mut prover_challenger),
            CanSample::<EF>::sample(&mut verifier_challenger),
        );
    }

    #[test]
    fn a_perturbed_point_coordinate_moves_every_later_draw() {
        // Invariant: the point is bound before anything is drawn against it.
        //
        // Without it one proof would replay at every point sharing its absorbed tail.
        //
        // Mutation: bump one coordinate by one, once per position.
        //
        //     honest:    [c_0, ..., c_i,     ..., c_7]
        //     tampered:  [c_0, ..., c_i + 1, ..., c_7]
        let honest = point_of(NUM_VARIABLES, 3);
        let rows = rows_of(5);
        let baseline = replay(&honest, &rows, EF::ONE);

        for position in 0..NUM_VARIABLES {
            let mut moved = honest.as_slice().to_vec();
            moved[position] += EF::ONE;
            assert_ne!(
                baseline,
                replay(&Point::new(moved), &rows, EF::ONE),
                "tampering with coordinate {position} left the stream where it was",
            );
        }
    }

    #[test]
    fn a_perturbed_tensor_row_moves_every_later_draw() {
        // Invariant: the rows that cross the wire are what enters the sponge.
        //
        // The rows are the whole element, so binding them binds both of its readings.
        // Binding one derived reading instead would leave the other unbound.
        //
        // Mutation: bump one row by one, at both ends and in the middle.
        //
        //     16 rows  ->  positions 0, 8 and 15
        let point = point_of(NUM_VARIABLES, 3);
        let honest = rows_of(5);
        let baseline = replay(&point, &honest, EF::ONE);

        for position in [0, NUM_ROWS / 2, NUM_ROWS - 1] {
            let mut tampered = honest.clone();
            tampered[position] += EF::ONE;
            assert_ne!(
                baseline,
                replay(&point, &tampered, EF::ONE),
                "tampering with row {position} left the stream where it was",
            );
        }
    }

    #[test]
    fn the_batching_draw_is_the_first_challenge_and_answers_to_the_rows() {
        // Invariant: the batching challenge is drawn after the element is bound.
        //
        // This is the ordering the reduction's soundness rests on.
        //
        // A bit matrix solving two F_2-linear systems moves the claim with the sum held.
        // A prover who knew the challenge first could search for such a matrix.
        //
        // Two things are pinned below.
        // Either one alone can be satisfied while the order is still wrong.
        //
        // First, where the steps sit in the description:
        //
        //     ... evaluation_point | tensor_rows | batching_point ...
        //                                ^             ^
        //                                |             the first challenge of the run
        //                                bound before it
        let pattern = base_shape().pattern::<EF>();
        let labels: Vec<&str> = pattern
            .interactions()
            .iter()
            .map(Interaction::label)
            .collect();

        let point_at = labels
            .iter()
            .position(|&label| label == EVALUATION_POINT)
            .expect("the description binds the evaluation point");
        let rows_at = labels
            .iter()
            .position(|&label| label == TENSOR_ROWS)
            .expect("the description binds the tensor rows");
        let first_challenge = pattern
            .interactions()
            .iter()
            .position(|interaction| interaction.kind() == Kind::Challenge)
            .expect("the description draws a challenge");

        assert_eq!(labels[first_challenge], BATCHING_POINT);
        assert!(
            point_at < first_challenge,
            "the evaluation point is bound at step {point_at}, the first challenge is step \
             {first_challenge}",
        );
        assert!(
            rows_at < first_challenge,
            "the rows are bound at step {rows_at}, the first challenge is step \
             {first_challenge}",
        );

        // Second, that the draw answers to the rows rather than merely following them.
        //
        // A run binding the rows after the draw would still move every later draw.
        // Only the batching point itself discriminates here.
        //
        // Mutation: bump one row by one, once per position.
        let point = point_of(NUM_VARIABLES, 3);
        let honest = rows_of(5);
        let (baseline, _) = replay(&point, &honest, EF::ONE);

        for position in 0..NUM_ROWS {
            let mut tampered = honest.clone();
            tampered[position] += EF::ONE;
            let (moved, _) = replay(&point, &tampered, EF::ONE);
            assert_ne!(
                baseline.0, moved.0,
                "row {position} left the batching point where it was",
            );
        }
    }

    #[test]
    fn a_perturbed_surviving_claim_moves_the_state_the_sponge_is_handed_back_in() {
        // Invariant: the surviving claim is bound before the sponge goes back.
        //
        // Nothing inside this description is drawn after it.
        // The effect therefore shows up outside.
        //
        // Mutation: bump the claim by one and compare the draw taken past the run's end.
        let point = point_of(NUM_VARIABLES, 3);
        let rows = rows_of(5);

        assert_ne!(
            replay(&point, &rows, EF::ONE),
            replay(&point, &rows, EF::TWO),
        );
    }

    // The two tests below drive the verifier transcript the way a downstream crate would.
    //
    // Values go in straight from proof fields, with no shape pre-check in front.
    // A panic here would compound with the driver's own drop check and abort.

    #[test]
    fn a_tensor_element_of_the_wrong_width_is_rejected() {
        // Described run: 16 rows, fixed by the level alone.
        //
        //     described:    16
        //     proof holds:  1   -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript =
            BitRingSwitchVerifierTranscript::<Chal, EF>::new(&mut challenger, base_shape());

        let err = transcript
            .statement(&point_of(NUM_VARIABLES, 3), &[EF::ONE], None)
            .expect_err("a tensor element outside the described width must error");

        assert_eq!(
            err,
            TranscriptWidth::TensorRows {
                expected: NUM_ROWS,
                actual: 1,
            },
        );

        // The rejection leaves the run part-played, with later steps still described.
        // Absorbing the width error is what releases the completeness check.
        drop(transcript);
    }

    #[test]
    fn an_evaluation_point_of_the_wrong_width_is_rejected() {
        // Described run: 8 coordinates, taken from the replaying side's configuration.
        //
        //     described:    8
        //     handed over:  3  -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript =
            BitRingSwitchVerifierTranscript::<Chal, EF>::new(&mut challenger, base_shape());

        let err = transcript
            .statement(&point_of(3, 3), &rows_of(5), None)
            .expect_err("a point outside the described width must error");

        assert_eq!(
            err,
            TranscriptWidth::Point {
                expected: NUM_VARIABLES,
                actual: 3,
            },
        );

        drop(transcript);
    }

    #[test]
    fn a_successor_element_list_outside_the_description_is_rejected() {
        // Described run: the shape alone fixes whether two successor elements follow the rows.
        //
        //     shape       proof holds        -> rejected before anything is absorbed
        //     plain       two elements       -> 0 expected, 2 supplied
        //     successor   none               -> 2 expected, 0 supplied
        //     successor   a one-row carry    -> a malformed element, like a short tensor
        let point = point_of(NUM_VARIABLES, 3);
        let rows = rows_of(5);
        let (carry, last) = (rows_of(7), rows_of(11));
        let cases: [(BitRingSwitchShape, SuccessorRows<'_>, TranscriptWidth); 3] = [
            (
                base_shape(),
                Some((&carry, &last)),
                TranscriptWidth::SuccessorElements {
                    expected: 0,
                    actual: 2,
                },
            ),
            (
                successor_shape(),
                None,
                TranscriptWidth::SuccessorElements {
                    expected: 2,
                    actual: 0,
                },
            ),
            (
                successor_shape(),
                Some((&[EF::ONE], &last)),
                TranscriptWidth::TensorRows {
                    expected: NUM_ROWS,
                    actual: 1,
                },
            ),
        ];

        for (shape, successor, expected) in cases {
            let mut challenger = fresh_challenger();
            let mut transcript =
                BitRingSwitchVerifierTranscript::<Chal, EF>::new(&mut challenger, shape);

            let err = transcript
                .statement(&point, &rows, successor)
                .expect_err("a successor list outside the description must error");

            assert_eq!(err, expected);
            drop(transcript);
        }
    }

    #[test]
    fn both_sides_of_a_successor_run_draw_the_same_stream() {
        // Completeness: the successor steps are walked the same way on both sides.
        //
        // Fixture state: the base run plus two successor elements and one scalar draw.
        let point = point_of(NUM_VARIABLES, 3);
        let rows = rows_of(5);
        let (carry, last) = (rows_of(7), rows_of(11));
        let successor = Some((carry.as_slice(), last.as_slice()));

        let mut prover_challenger = fresh_challenger();
        let proved = drive_prover(
            &mut prover_challenger,
            successor_shape(),
            &point,
            &rows,
            successor,
            EF::ONE,
        );

        let mut verifier_challenger = fresh_challenger();
        let replayed = drive_verifier(
            &mut verifier_challenger,
            successor_shape(),
            &point,
            &rows,
            successor,
            EF::ONE,
        )
        .expect("the honest run must replay");

        assert_eq!(proved, replayed);
        assert!(
            proved.1.is_some(),
            "a successor shape draws the element batching"
        );
        assert_eq!(
            CanSample::<EF>::sample(&mut prover_challenger),
            CanSample::<EF>::sample(&mut verifier_challenger),
        );
    }

    #[test]
    fn the_batching_draws_answer_to_the_successor_rows() {
        // Invariant: both batching draws come after every element is bound.
        //
        // A successor element formed after r'' or alpha is the tensor's forgery again.
        //
        // First, where the steps sit in the description:
        //
        //     ... tensor_rows | carry_rows | last_rows | batching_point | tensor_batching ...
        //                                       ^             ^
        //                                       |             the first challenge of the run
        //                                       bound before it
        let pattern = successor_shape().pattern::<EF>();
        let labels: Vec<&str> = pattern
            .interactions()
            .iter()
            .map(Interaction::label)
            .collect();

        let position = |wanted: &str| {
            labels
                .iter()
                .position(|&label| label == wanted)
                .expect("the successor description holds every step")
        };
        let first_challenge = pattern
            .interactions()
            .iter()
            .position(|interaction| interaction.kind() == Kind::Challenge)
            .expect("the description draws a challenge");

        assert_eq!(labels[first_challenge], BATCHING_POINT);
        assert!(position(TENSOR_ROWS) < position(CARRY_ROWS));
        assert!(position(CARRY_ROWS) < position(LAST_ROWS));
        assert!(
            position(LAST_ROWS) < first_challenge,
            "the last rows are bound at step {}, the first challenge is step {first_challenge}",
            position(LAST_ROWS),
        );
        assert_eq!(position(TENSOR_BATCHING), first_challenge + 1);

        // Second, that both draws answer to the successor rows rather than merely following them.
        //
        // Mutation: bump one carry row by one, then one last row.
        let point = point_of(NUM_VARIABLES, 3);
        let rows = rows_of(5);
        let (carry, last) = (rows_of(7), rows_of(11));
        let replay_with = |carry: &[EF], last: &[EF]| {
            replay_in(
                successor_shape(),
                &point,
                &rows,
                Some((carry, last)),
                EF::ONE,
            )
            .0
        };
        let baseline = replay_with(&carry, &last);

        let mut bumped_carry = carry.clone();
        bumped_carry[NUM_ROWS / 2] += EF::ONE;
        let mut bumped_last = last.clone();
        bumped_last[NUM_ROWS - 1] += EF::ONE;

        for (element, moved) in [
            ("carry", replay_with(&bumped_carry, &last)),
            ("last", replay_with(&carry, &bumped_last)),
        ] {
            assert_ne!(
                baseline.0, moved.0,
                "a {element} row left the batching point where it was",
            );
            assert_ne!(
                baseline.1, moved.1,
                "a {element} row left the element batching where it was",
            );
        }
    }

    #[test]
    #[should_panic(expected = "the evaluation point must name at least the 4 absorbed")]
    fn a_point_shorter_than_the_absorbed_tail_has_no_round_count() {
        // Boundary: three coordinates cannot host the four an element absorbs.
        //
        // The split that separates the absorbed tail from the rest is undefined below four.
        let _ = BitRingSwitchShape::new(3).sumcheck_rounds::<EF>();
    }

    #[test]
    fn every_batch_of_claims_seeds_its_own_stream() {
        // Invariant: the claim count and each claim's successor rows reach the seed.
        //
        //     count        two plain claims  vs  three
        //     order        plain then successor  vs  successor then plain
        //     row count    the same claims, one successor row count apart
        //     one claim    a batch of two never shares the one-claim run's seed
        let plain = base_shape();
        let successor = successor_shape();
        let wider = BitRingSwitchShape::with_successor_rows(NUM_VARIABLES, SUCCESSOR_ROWS + 1);
        let batch = |claims: &[BitRingSwitchShape]| {
            seed_digest(
                &BitRingSwitchClaimsShape {
                    claims: claims.to_vec(),
                }
                .domain_separator::<EF>(),
            )
        };
        let seeds = [
            ("two plain", batch(&[plain, plain])),
            ("three plain", batch(&[plain, plain, plain])),
            ("plain, successor", batch(&[plain, successor])),
            ("successor, plain", batch(&[successor, plain])),
            ("plain, wider successor", batch(&[plain, wider])),
            ("one-claim run", seed_of(plain)),
        ];
        assert_seeds_pairwise_distinct(&seeds);

        // Nothing in a batch grinds either.
        assert!(
            pow_difficulties(
                &BitRingSwitchClaimsShape {
                    claims: vec![plain, successor]
                }
                .pattern::<EF>()
            )
            .is_empty()
        );
    }

    #[test]
    fn a_batch_statement_of_the_wrong_count_is_refused_before_absorbing() {
        // The verifier checks every claim's widths before the sponge moves.
        let shape = BitRingSwitchClaimsShape {
            claims: vec![base_shape(), base_shape()],
        };
        let point = point_of(NUM_VARIABLES, 1);
        let rows = rows_of(1);
        let claim = ClaimStatement {
            point: &point,
            rows: &rows,
            successor: None,
        };
        let mut challenger = fresh_challenger();
        let mut transcript =
            BitRingSwitchClaimsVerifierTranscript::<Chal, EF>::new(&mut challenger, shape);
        assert_eq!(
            transcript.statement(&[claim]).unwrap_err(),
            TranscriptWidth::Claims {
                expected: 2,
                actual: 1
            }
        );
    }
}
