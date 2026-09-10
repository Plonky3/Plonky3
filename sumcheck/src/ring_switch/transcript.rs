//! Fiat-Shamir transcript of the ring-switching reduction.
//!
//! # Overview
//!
//! One description of everything a ring-switching reduction absorbs and draws.
//!
//! Both sides consume it through the same four calls.
//!
//! The description itself is a value, built from the numbers that shape a run.
//!
//! # Shape
//!
//! ```text
//!     evaluation point   ->  one extension element per coordinate
//!     tensor element     ->  one base element per coefficient
//!     batching point     ->  one extension element per packed variable, drawn
//!     sumcheck           ->  a bracket around a delegated run
//!     surviving claim    ->  1 extension element
//! ```
//!
//! # Two element widths, two steps
//!
//! The evaluation point and the tensor element are both prover-bound lists.
//!
//! They live in different fields.
//!
//! They are therefore described as different steps.
//!
//! ```text
//!     evaluation point  ->  coordinates of the extension
//!     tensor element    ->  coefficients of the base field
//! ```
//!
//! Each step records the width of the values it carries, alongside their count.
//!
//! Absorb base coefficients where a point belongs and the shape check fails.
//!
//! # Sub-protocol
//!
//! A ring-switching reduction is always a phase of something larger.
//!
//! It seeds a transcript of its own from a borrowed sponge, and hands the sponge back.
//!
//! The batched sumcheck it runs does the same thing, one level further down.
//!
//! ```text
//!     caller     ->  its own description, bracketing this run inside it
//!     reduction  ->  this description, bracketing the sumcheck inside it
//!     sumcheck   ->  its own description, under its own seed
//! ```
//!
//! The three descriptions compose through the order their seeds reach the sponge.
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//!
//! ```text
//!     coordinate count   ->  width of the evaluation-point step
//!     base field         ->  identity carried on every step
//!     extension degree   ->  carried on every step, fixing both derived widths
//! ```
//!
//! The extension degree fixes both remaining widths on its own.
//!
//! ```text
//!     tensor element  ->  degree squared coefficients
//!     batching point  ->  log2 of the degree coordinates
//! ```
//!
//! Two runs differing in any of the three cannot share a transcript.
//!
//! # Grinding
//!
//! The description holds no proof-of-work step.
//!
//! Every challenge here is a function of messages the prover chooses.
//!
//! A prover may therefore resample.
//!
//! The soundness bound is a per-attempt probability.
//!
//! A protocol needing the total bound supplies the grinding outside this run.

use alloc::vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;

use super::RingSwitchError;
use super::packing::packed_vars;
use super::tensor::TensorAlgebra;

/// Version byte bound into the transcript seed.
///
/// Bumping it separates two revisions of this protocol.
///
/// It separates them even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
///
/// Distinct from every plain sumcheck name.
///
/// No reduction then shares a seed with a bare batch.
const NAME: &[u8] = b"p3-sumcheck-ring-switch";

/// Step label of the point the incoming evaluation claim is stated at.
const EVALUATION_POINT: &str = "evaluation_point";

/// Step label of the base coefficients of the shared tensor element.
const TENSOR_ELEMENT: &str = "tensor_element";

/// Step label of the challenges that collapse the row claims into one.
const BATCHING_POINT: &str = "batching_point";

/// Step label of the bracket around the delegated sumcheck.
const BATCHED_SUMCHECK: &str = "batched_sumcheck";

/// Step label of the value the surviving claim carries.
const SURVIVING_CLAIM: &str = "surviving_claim";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the sub-protocol the reduction delegates its rounds to.
///
/// Recorded on the bracket markers as a local diagnostic.
///
/// It does not reach the pattern fingerprint.
struct BatchedSumcheck;

/// Numbers that fix the transcript of one ring-switching reduction.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingSwitchShape {
    /// Number of coordinates the incoming evaluation point names.
    ///
    /// Fixes the width of the step that binds that point.
    ///
    /// The packed variables sit at its tail.
    ///
    /// This count therefore fixes the round count too.
    pub num_variables: usize,
}

impl RingSwitchShape {
    /// Collect the numbers that fix one reduction.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: number of coordinates the incoming evaluation point names.
    #[must_use]
    pub const fn new(num_variables: usize) -> Self {
        Self { num_variables }
    }

    /// Number of rounds the delegated sumcheck runs.
    ///
    /// The packed variables are the ones a single extension element already absorbs.
    ///
    /// They are removed from the point.
    ///
    /// What is left is what the rounds reduce away.
    ///
    /// ```text
    ///     rounds = coordinate count - packed variables
    /// ```
    ///
    /// # Panics
    ///
    /// - When the extension degree is not a power of two.
    /// - When the point names fewer coordinates than one packed element absorbs.
    #[must_use]
    pub fn sumcheck_rounds<F, EF>(&self) -> usize
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let kappa = packed_vars::<F, EF>();
        // Below this the split that separates the packed tail from the rest is undefined.
        assert!(
            self.num_variables >= kappa,
            "the evaluation point must name at least the {kappa} packed variables, got {}",
            self.num_variables
        );
        self.num_variables - kappa
    }

    /// Number of base coefficients the shared tensor element carries.
    ///
    /// The element is a square matrix over the base field, one side per basis vector.
    ///
    /// ```text
    ///     coefficients = extension degree squared
    /// ```
    #[must_use]
    pub const fn tensor_coefficients<F, EF>() -> usize
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let dimension = TensorAlgebra::<F, EF>::DIMENSION;
        dimension * dimension
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// - When the extension degree is not a power of two.
    /// - Never for structural reasons: one matched bracket always validates.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Three leaf steps, one matched bracket, then the closing leaf.
        let steps = vec![
            // The point comes first.
            //
            // Every later draw then depends on where the claim is stated.
            //
            // Its coordinates are extension elements.
            //
            // The step's own width records that.
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                EVALUATION_POINT,
                Length::Fixed(self.num_variables),
            ),
            // The tensor element is bound through the coefficients that cross the wire.
            //
            // Binding a derived reading would leave the other reading unbound.
            //
            // A base coefficient is one element wide.
            //
            // That width parts this step from the one above, whatever the counts.
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Message,
                TENSOR_ELEMENT,
                Length::Fixed(Self::tensor_coefficients::<F, EF>()),
            ),
            // One challenge per packed variable collapses the rows into a single claim.
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BATCHING_POINT,
                Length::Fixed(packed_vars::<F, EF>()),
            ),
            // The bracket records that a sub-protocol runs here.
            //
            // Its steps live in the callee's description, under the callee's seed.
            //
            // This description states only that the delegation happens, and where.
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
            // The value the reduction leaves behind is bound before the sponge goes back.
            //
            // The surrounding protocol discharges that value against a commitment.
            //
            // Binding it here is what ties the discharge to this run.
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                SURVIVING_CLAIM,
                Length::Scalar,
            ),
        ];

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity and the transcript shape into a seed.
    ///
    /// - The coordinate count moves the width of the first step.
    /// - The field pair moves the identity every step carries.
    ///
    /// The fingerprint of the description covers both.
    ///
    /// No separate instance label is needed.
    ///
    /// # Panics
    ///
    /// When the extension degree is not a power of two.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>())
    }
}

/// Prover-side transcript of one ring-switching reduction.
///
/// # Overview
///
/// Holds the only definition of what a prover binds and draws in this reduction.
///
/// No prover loop can then drift from the verifier.
///
/// # Borrowing
///
/// The challenger is borrowed, not consumed.
///
/// A reduction runs inside a larger protocol.
///
/// That protocol's own transcript continues where this one stops.
pub struct RingSwitchProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the extension field the point and the claims live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> RingSwitchProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
    /// - `shape`: the numbers that fix this run's transcript.
    ///
    /// # Panics
    ///
    /// When the extension degree is not a power of two.
    pub fn new(challenger: &'a mut C, shape: RingSwitchShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: ProverState::new(challenger, &separator),
            _ef: PhantomData,
        }
    }

    /// Bind the statement, then draw the challenges that batch it down to one claim.
    ///
    /// # Arguments
    ///
    /// - `point`: where the incoming evaluation claim is stated.
    /// - `tensor_coefficients`: base coefficients of the shared tensor element.
    ///
    /// # Returns
    ///
    /// The batching point, one challenge per packed variable.
    ///
    /// # Panics
    ///
    /// When either list is not the width the run was described with.
    pub fn statement(&mut self, point: &Point<EF>, tensor_coefficients: &[F]) -> Point<EF> {
        // The point is bound first.
        //
        // The tensor element cannot then be chosen to suit it.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(EVALUATION_POINT, point.as_slice());

        // The coefficients are what crosses the wire.
        //
        // They are therefore what enters the sponge.
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(TENSOR_ELEMENT, tensor_coefficients);

        // Only now is the batching point drawn.
        //
        // No message above it was chosen knowing it.
        Point::new(
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                    BATCHING_POINT,
                    packed_vars::<F, EF>(),
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        )
    }

    /// Lend the sponge to the batched sumcheck, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn batched_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        output
    }

    /// Bind the value the surviving claim carries.
    ///
    /// # Arguments
    ///
    /// - `value`: the packed polynomial's evaluation at the point the rounds produced.
    pub fn surviving_claim(&mut self, value: EF) {
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(SURVIVING_CLAIM, &value);
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the run was described with.
    pub fn finish(self) {
        // Nothing was written to the driver's own buffer.
        //
        // Closing is purely the check that the description was consumed.
        assert!(
            self.state.finalize().is_empty(),
            "the ring-switching reduction carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one ring-switching reduction.
///
/// Mirrors the prover side call for call, over the same description.
///
/// Every value comes from the proof or from the verifier's own inputs, never from a wire.
///
/// The described widths are what reject a value the proof got wrong.
pub struct RingSwitchVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value.
    ///
    /// The driver therefore reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: RingSwitchShape,
    /// Marker for the extension field the point and the claims live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> RingSwitchVerifierTranscript<'a, C, F, EF>
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
    ///
    /// # Panics
    ///
    /// When the extension degree is not a power of two.
    pub fn new(challenger: &'a mut C, shape: RingSwitchShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Bind the statement, then draw the challenges that batch it down to one claim.
    ///
    /// # Arguments
    ///
    /// - `point`: where the incoming evaluation claim is stated.
    /// - `tensor_coefficients`: base coefficients of the tensor element in the proof.
    ///
    /// # Returns
    ///
    /// The batching point the prover saw.
    ///
    /// # Errors
    ///
    /// - The point does not name the described number of coordinates.
    /// - The coefficient list is not the described width.
    ///
    /// Either rejection releases the driver's completeness check on its way out.
    pub fn statement(
        &mut self,
        point: &Point<EF>,
        tensor_coefficients: &[F],
    ) -> Result<Point<EF>, RingSwitchError> {
        // The point is the verifier's own input.
        //
        // A mismatch is therefore a configuration failure.
        //
        // It is still reported rather than asserted.
        //
        // A downstream verifier must never abort over an input of the wrong shape.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(EVALUATION_POINT, point.as_slice())
            .map_err(|_| RingSwitchError::PointWidthMismatch {
                expected: self.shape.num_variables,
                actual: point.num_variables(),
            })?;

        // The coefficient count arrives inside the proof.
        //
        // It is therefore attacker-controlled.
        //
        //     described:    the extension degree squared
        //     proof holds:  anything at all      -> rejected, nothing absorbed
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(TENSOR_ELEMENT, tensor_coefficients)
            .map_err(|_| RingSwitchError::MalformedTensor {
                expected: RingSwitchShape::tensor_coefficients::<F, EF>(),
                actual: tensor_coefficients.len(),
            })?;

        // Both messages are bound.
        //
        // The batching point is now safe to draw.
        Ok(Point::new(
            self.state
                .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                    BATCHING_POINT,
                    packed_vars::<F, EF>(),
                )
                .into_iter()
                .map(TranscriptBound::into_inner)
                .collect(),
        ))
    }

    /// Lend the sponge to the batched sumcheck, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    ///
    /// A rejection inside it leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn batched_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<BatchedSumcheck>(BATCHED_SUMCHECK);
        output
    }

    /// Bind the value the surviving claim carries.
    ///
    /// The prover bound its own copy of the same value.
    ///
    /// A proof carrying a different one moves every draw that follows.
    ///
    /// # Arguments
    ///
    /// - `value`: the surviving claim's value, as the proof states it.
    pub fn surviving_claim(&mut self, value: EF) {
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(SURVIVING_CLAIM, &value);
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// A rejection that leaves steps unplayed would raise a drop-time panic.
    ///
    /// That panic would land on top of the error already on its way out.
    ///
    /// Idempotent.
    ///
    /// A step that already released the check may be aborted again.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When fewer steps were replayed than the run was described with.
    pub fn finish(self) {
        // The proof carries every value.
        //
        // No unread wire bytes can remain.
        self.state
            .finalize()
            .expect("the ring-switching reduction reads an empty wire");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_challenger::testing::{
        SeedDigest, assert_seeds_pairwise_distinct, pow_difficulties, seed_digest,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    /// Coordinate count every run below is described with.
    ///
    /// Two packed variables sit at the tail.
    ///
    /// That leaves four sumcheck rounds.
    const NUM_VARIABLES: usize = 6;

    /// Base coefficients a degree-four extension's tensor element carries.
    const TENSOR_WIDTH: usize = 16;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x1E56);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// Baseline shape every mutation below is measured against.
    const fn base_shape() -> RingSwitchShape {
        RingSwitchShape::new(NUM_VARIABLES)
    }

    /// The digest of the byte stream a shape seeds its sponge with.
    ///
    /// The comparison is over seed streams, not over a sampled challenge.
    ///
    /// That keeps the sponge out of it.
    fn seed_of(shape: RingSwitchShape) -> SeedDigest {
        seed_digest(&shape.domain_separator::<F, EF>())
    }

    /// Every field of the shape, each moved one step away from the baseline.
    ///
    /// One entry per configuration knob.
    ///
    /// A field added to the shape stops the destructuring below from compiling.
    fn one_step_from_base() -> Vec<(&'static str, RingSwitchShape)> {
        // Exhaustiveness check: every field named, none elided by a rest pattern.
        //
        // The binding goes unused: naming the field is all this has to do.
        let RingSwitchShape { num_variables: _ } = base_shape();

        // One more coordinate widens the step that binds the point.
        let mut wider = base_shape();
        wider.num_variables += 1;

        vec![("num_variables", wider)]
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
        let mut seeds = vec![("base", seed_of(base_shape()))];
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
        // No step may declare a difficulty.
        //
        // A protocol needing grinding supplies it outside this run.
        //
        // A step appearing here would be a cost neither side accounts for.
        assert!(pow_difficulties(&base_shape().pattern::<F, EF>()).is_empty());
    }

    #[test]
    fn the_two_derived_widths_come_from_the_extension_degree() {
        // Fixture state: a degree-four extension over the base field.
        //
        //     packed variables  ->  log2(4) = 2
        //     tensor width      ->  4 * 4   = 16
        //     sumcheck rounds   ->  6 - 2   = 4
        assert_eq!(packed_vars::<F, EF>(), 2);
        assert_eq!(
            RingSwitchShape::tensor_coefficients::<F, EF>(),
            TENSOR_WIDTH
        );
        assert_eq!(base_shape().sumcheck_rounds::<F, EF>(), 4);
    }

    /// An evaluation point whose coordinates are distinct and seed-dependent.
    fn point_of(num_variables: usize, seed: u32) -> Point<EF> {
        Point::new(
            (0..num_variables)
                .map(|i| EF::from_u32(seed * 31 + i as u32 + 1))
                .collect(),
        )
    }

    /// A full-width coefficient list whose entries are distinct and seed-dependent.
    fn coefficients_of(seed: u32) -> Vec<F> {
        (0..TENSOR_WIDTH)
            .map(|i| F::from_u32(seed * 17 + i as u32 + 1))
            .collect()
    }

    /// Everything one prover-side run produces, in the order the description fixes.
    type Draws = (Point<EF>, F);

    /// Drive a full prover-side run and hand back the draws it produced.
    ///
    /// The delegated sumcheck is stood in for by a single base-field draw.
    ///
    /// That keeps the bracket on the path a real delegation takes.
    ///
    /// The sponge is touched just as it would be.
    fn drive_prover(
        challenger: &mut Ch,
        shape: RingSwitchShape,
        point: &Point<EF>,
        coefficients: &[F],
        claim: EF,
    ) -> Draws {
        let mut transcript = RingSwitchProverTranscript::<Ch, F, EF>::new(challenger, shape);
        let batching_point = transcript.statement(point, coefficients);
        let delegated = transcript.batched_sumcheck(<Ch as CanSample<F>>::sample);
        transcript.surviving_claim(claim);
        transcript.finish();
        (batching_point, delegated)
    }

    /// Replay a full verifier-side run against recorded values.
    fn drive_verifier(
        challenger: &mut Ch,
        shape: RingSwitchShape,
        point: &Point<EF>,
        coefficients: &[F],
        claim: EF,
    ) -> Result<Draws, RingSwitchError> {
        let mut transcript = RingSwitchVerifierTranscript::<Ch, F, EF>::new(challenger, shape);
        let batching_point = transcript.statement(point, coefficients)?;
        let delegated = transcript.batched_sumcheck(<Ch as CanSample<F>>::sample);
        transcript.surviving_claim(claim);
        transcript.finish();
        Ok((batching_point, delegated))
    }

    #[test]
    fn both_sides_of_a_run_draw_the_same_stream() {
        // Completeness: the two drivers walk one description and land on one stream.
        //
        // Fixture state: 6 coordinates, 16 coefficients, one delegated draw.
        let point = point_of(NUM_VARIABLES, 3);
        let coefficients = coefficients_of(5);

        let mut prover_challenger = fresh_challenger();
        let proved = drive_prover(
            &mut prover_challenger,
            base_shape(),
            &point,
            &coefficients,
            EF::ONE,
        );

        let mut verifier_challenger = fresh_challenger();
        let replayed = drive_verifier(
            &mut verifier_challenger,
            base_shape(),
            &point,
            &coefficients,
            EF::ONE,
        )
        .expect("the honest run must replay");

        assert_eq!(proved, replayed);

        // The sponge is handed back in one state.
        //
        // The surrounding protocol stays in step.
        assert_eq!(
            CanSample::<F>::sample(&mut prover_challenger),
            CanSample::<F>::sample(&mut verifier_challenger),
        );
    }

    /// Everything a verifier redraws from one recorded run, plus the state it hands back.
    ///
    /// The trailing draw is taken after the run closes.
    ///
    /// It exposes a value bound with nothing left inside the description to move.
    fn replay(point: &Point<EF>, coefficients: &[F], claim: EF) -> (Draws, F) {
        let mut challenger = fresh_challenger();
        let draws = drive_verifier(
            &mut challenger,
            RingSwitchShape::new(point.num_variables()),
            point,
            coefficients,
            claim,
        )
        .expect("a well-shaped run always replays");
        (draws, CanSample::<F>::sample(&mut challenger))
    }

    #[test]
    fn a_perturbed_point_coordinate_moves_every_later_draw() {
        // Invariant: the point is bound before anything is drawn against it.
        //
        // Without it one proof would replay at every point sharing its packed tail.
        //
        // Mutation: bump one coordinate by one, once per position.
        //
        //     honest:    [c_0, ..., c_i,     ..., c_5]
        //     tampered:  [c_0, ..., c_i + 1, ..., c_5]
        let honest = point_of(NUM_VARIABLES, 3);
        let coefficients = coefficients_of(5);
        let baseline = replay(&honest, &coefficients, EF::ONE);

        for position in 0..NUM_VARIABLES {
            let mut moved = honest.as_slice().to_vec();
            moved[position] += EF::ONE;
            assert_ne!(
                baseline,
                replay(&Point::new(moved), &coefficients, EF::ONE),
                "tampering with coordinate {position} left the stream where it was",
            );
        }
    }

    #[test]
    fn a_perturbed_tensor_coefficient_moves_every_later_draw() {
        // Invariant: the coefficients that cross the wire are what enters the sponge.
        //
        // Binding one of the two derived readings instead would leave the other unbound.
        //
        // Mutation: bump one coefficient by one, at both ends and in the middle.
        //
        //     16 coefficients  ->  positions 0, 8 and 15
        let point = point_of(NUM_VARIABLES, 3);
        let honest = coefficients_of(5);
        let baseline = replay(&point, &honest, EF::ONE);

        for position in [0, TENSOR_WIDTH / 2, TENSOR_WIDTH - 1] {
            let mut tampered = honest.clone();
            tampered[position] += F::ONE;
            assert_ne!(
                baseline,
                replay(&point, &tampered, EF::ONE),
                "tampering with coefficient {position} left the stream where it was",
            );
        }
    }

    #[test]
    fn a_perturbed_surviving_claim_moves_the_state_the_sponge_is_handed_back_in() {
        // Invariant: the surviving claim is bound before the sponge goes back.
        //
        // Nothing inside this description is drawn after it.
        //
        // The effect therefore shows up outside.
        //
        // Mutation: bump the claim by one and compare the draw taken past the run's end.
        let point = point_of(NUM_VARIABLES, 3);
        let coefficients = coefficients_of(5);

        assert_ne!(
            replay(&point, &coefficients, EF::ONE),
            replay(&point, &coefficients, EF::TWO),
        );
    }

    // The two tests below drive the verifier transcript the way a downstream crate would.
    //
    // Values go in straight from proof fields, with no shape pre-check in front.
    //
    // Both malformed inputs must return an error.
    //
    // A panic here would compound with the driver's own drop check and abort.

    #[test]
    fn a_tensor_element_of_the_wrong_width_is_rejected() {
        // Described run: 16 base coefficients, fixed by the extension degree alone.
        //
        //     described:    16
        //     proof holds:  1   -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript =
            RingSwitchVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base_shape());

        let err = transcript
            .statement(&point_of(NUM_VARIABLES, 3), &[F::ONE])
            .expect_err("a tensor element outside the described width must error");

        assert_eq!(
            err,
            RingSwitchError::MalformedTensor {
                expected: TENSOR_WIDTH,
                actual: 1,
            },
        );

        // The rejection leaves the run part-played, with three steps still described.
        //
        // Absorbing the width error is what releases the completeness check.
        //
        // Without that release this drop panics on top of the error on its way out.
        drop(transcript);
    }

    #[test]
    fn an_evaluation_point_of_the_wrong_width_is_rejected() {
        // Described run: 6 coordinates, taken from the replaying side's configuration.
        //
        //     described:    6
        //     handed over:  3  -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript =
            RingSwitchVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base_shape());

        let err = transcript
            .statement(&point_of(3, 3), &coefficients_of(5))
            .expect_err("a point outside the described width must error");

        assert_eq!(
            err,
            RingSwitchError::PointWidthMismatch {
                expected: NUM_VARIABLES,
                actual: 3,
            },
        );

        drop(transcript);
    }

    #[test]
    #[should_panic(expected = "the evaluation point must name at least the 2 packed variables")]
    fn a_point_shorter_than_the_packed_tail_has_no_round_count() {
        // Boundary: one coordinate cannot host the two variables a packed element holds.
        //
        // The split that separates the packed tail from the rest is undefined below two.
        let _ = RingSwitchShape::new(1).sumcheck_rounds::<F, EF>();
    }
}
