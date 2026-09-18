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
pub(crate) const NAME: &[u8] = b"p3-sumcheck-bit-ring-switch";

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

    /// Baseline shape every mutation below is measured against.
    const fn base_shape() -> BitRingSwitchShape {
        BitRingSwitchShape::new(NUM_VARIABLES)
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
        let BitRingSwitchShape { num_variables: _ } = base_shape();

        // One more coordinate widens the step that binds the point.
        let mut wider = base_shape();
        wider.num_variables += 1;

        alloc::vec![("num_variables", wider)]
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
    type Draws = (Point<EF>, EF);

    /// Drive a full prover-side run and hand back the draws it produced.
    ///
    /// The delegated sumcheck is stood in for by a single draw.
    /// That keeps the bracket on the path a real delegation takes.
    fn drive_prover(
        challenger: &mut Chal,
        shape: BitRingSwitchShape,
        point: &Point<EF>,
        rows: &[EF],
        claim: EF,
    ) -> Draws {
        let mut transcript = BitRingSwitchProverTranscript::<Chal, EF>::new(challenger, shape);
        let batching_point = transcript.statement(point, rows);
        let delegated = transcript.batched_sumcheck(<Chal as CanSample<EF>>::sample);
        transcript.surviving_claim(claim);
        transcript.finish();
        (batching_point, delegated)
    }

    /// Replay a full verifier-side run against recorded values.
    fn drive_verifier(
        challenger: &mut Chal,
        shape: BitRingSwitchShape,
        point: &Point<EF>,
        rows: &[EF],
        claim: EF,
    ) -> Result<Draws, TranscriptWidth> {
        let mut transcript = BitRingSwitchVerifierTranscript::<Chal, EF>::new(challenger, shape);
        let batching_point = transcript.statement(point, rows)?;
        let delegated = transcript.batched_sumcheck(<Chal as CanSample<EF>>::sample);
        transcript.surviving_claim(claim);
        transcript.finish();
        Ok((batching_point, delegated))
    }

    /// Everything a verifier redraws from one recorded run, plus the state it hands back.
    ///
    /// The trailing draw is taken after the run closes.
    /// It exposes a value bound with nothing left inside the description to move.
    fn replay(point: &Point<EF>, rows: &[EF], claim: EF) -> (Draws, EF) {
        let mut challenger = fresh_challenger();
        let draws = drive_verifier(
            &mut challenger,
            BitRingSwitchShape::new(point.num_variables()),
            point,
            rows,
            claim,
        )
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
        let proved = drive_prover(&mut prover_challenger, base_shape(), &point, &rows, EF::ONE);

        let mut verifier_challenger = fresh_challenger();
        let replayed = drive_verifier(
            &mut verifier_challenger,
            base_shape(),
            &point,
            &rows,
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
            .statement(&point_of(NUM_VARIABLES, 3), &[EF::ONE])
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
            .statement(&point_of(3, 3), &rows_of(5))
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
    #[should_panic(expected = "the evaluation point must name at least the 4 absorbed")]
    fn a_point_shorter_than_the_absorbed_tail_has_no_round_count() {
        // Boundary: three coordinates cannot host the four an element absorbs.
        //
        // The split that separates the absorbed tail from the rest is undefined below four.
        let _ = BitRingSwitchShape::new(3).sumcheck_rounds::<EF>();
    }
}
