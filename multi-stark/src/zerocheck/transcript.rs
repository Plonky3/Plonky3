//! Fiat-Shamir transcript of the AIR zerocheck.
//!
//! # Overview
//!
//! One statement of what the zerocheck draws, consumed by both sides.
//!
//! It is built from the AIRs, the trace heights and the grinding difficulty.
//! Both sides hold all three before a proof exists, so neither reads a count out of one.
//!
//! # Shape
//!
//! ```text
//!     alpha                        one extension element
//!     beta                         one extension element
//!     eta                          only when a lookup reduction supplies a point
//!     tau                          one element per free coordinate, all nonzero
//!     Begin  constraint sumcheck   bracket around the delegated run
//!     End    constraint sumcheck
//! ```
//!
//! The reduction point of the lookup argument becomes the tail of the zerocheck point.
//! Only the coordinates ahead of that tail are drawn here.
//!
//! # What is bound
//!
//! - Shape: how many coordinates are drawn, and whether a lookup contributes at all.
//! - Instance label: the cube width, the tail length, the grinding difficulty.
//! - Instance label: the two constraint degrees of every AIR in the batch.
//! - Nothing here: the sumcheck's round width, which its own seed binds inside the bracket.
//!
//! # Soundness
//!
//! Every coordinate of the zerocheck point is drawn nonzero.
//!
//! The prover rebuilds a transmitted round message by dividing by that coordinate.
//! A zero would leave the message undetermined, so the draw rejects one and takes the next.
//!
//! The number of candidates a draw consumes is not part of the shape.
//! It depends on the sponge, and both sides consume the same one.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use crate::rounds::AirDegrees;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-multi-stark-zerocheck";

/// Step label of the challenge batching one AIR's own constraints.
const ALPHA: &str = "alpha";

/// Step label of the challenge batching the AIRs against each other.
const BETA: &str = "beta";

/// Step label of the challenge separating lookup links from ordinary constraints.
const ETA: &str = "eta";

/// Step label of the freely drawn coordinates of the zerocheck point.
const TAU: &str = "tau";

/// Step label of the bracket around the delegated constraint sumcheck.
const CONSTRAINT_SUMCHECK: &str = "constraint_sumcheck";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the sub-protocol the zerocheck delegates its reduction to.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct ConstraintSumcheck;

/// The four challenges one zerocheck run draws before its sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckChallenges<EF> {
    /// Batches one AIR's own constraints against each other.
    pub alpha: EF,
    /// Batches the AIRs of the batch against each other.
    pub beta: EF,
    /// Separates lookup links from ordinary constraints, or zero when there are no links.
    pub eta: EF,
    /// The zerocheck point, freely drawn coordinates first and the lookup tail last.
    pub tau: Vec<EF>,
}

/// Numbers that fix the transcript of one AIR zerocheck.
///
/// Both sides build this from the AIRs and the trace heights, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckShape {
    /// Number of variables the global sumcheck cube spans.
    pub log_height: usize,
    /// Coordinates the lookup reduction already fixed at the tail of the zerocheck point.
    pub lookup_point_len: usize,
    /// Grinding difficulty inside each sumcheck round, or zero to omit grinding.
    pub pow_bits: usize,
    /// Per-variable degrees of every AIR in the batch, in caller order.
    pub air_degrees: Vec<AirDegrees>,
}

impl ZerocheckShape {
    /// Assemble the shape from the numbers both sides already hold.
    ///
    /// # Arguments
    ///
    /// - `air_degrees`: the two constraint degrees of every AIR, in caller order.
    /// - `log_height`: number of variables the global sumcheck cube spans.
    /// - `lookup_point_len`: coordinates the lookup reduction fixed at the tail of the point.
    /// - `pow_bits`: grinding difficulty inside each sumcheck round.
    #[must_use]
    pub fn new(
        air_degrees: &[AirDegrees],
        log_height: usize,
        lookup_point_len: usize,
        pow_bits: usize,
    ) -> Self {
        Self {
            log_height,
            lookup_point_len,
            pow_bits,
            air_degrees: air_degrees.to_vec(),
        }
    }

    /// Coordinates of the zerocheck point this run draws for itself.
    ///
    /// The rest are the tail the lookup reduction already fixed.
    #[must_use]
    pub const fn free_coordinates(&self) -> usize {
        self.log_height.saturating_sub(self.lookup_point_len)
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// One matched bracket always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let challenge = |label, length| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, label, length)
        };

        // Two batching challenges, then at most two more steps and the bracket.
        let mut steps = Vec::with_capacity(6);

        // Alpha batches one AIR's constraints, beta batches the AIRs against each other.
        steps.push(challenge(ALPHA, Length::Scalar));
        steps.push(challenge(BETA, Length::Scalar));

        // Eta exists only to weigh lookup links against ordinary constraints.
        // With no reduction point there are no links, so the step is absent.
        if self.lookup_point_len > 0 {
            steps.push(challenge(ETA, Length::Scalar));
        }

        // The freely drawn coordinates form one step, whatever the rejection costs.
        if self.free_coordinates() > 0 {
            steps.push(challenge(TAU, Length::Fixed(self.free_coordinates())));
        }

        // The bracket records that a sub-protocol runs here.
        //
        // Its steps live in the callee's own pattern, under the callee's own seed.
        // What this pattern states is that the delegation happens, and where.
        steps.push(Interaction::marker::<ConstraintSumcheck>(
            Hierarchy::Begin,
            Kind::Protocol,
            CONSTRAINT_SUMCHECK,
        ));
        steps.push(Interaction::marker::<ConstraintSumcheck>(
            Hierarchy::End,
            Kind::Protocol,
            CONSTRAINT_SUMCHECK,
        ));

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity, this shape, and the AIRs it runs over.
    ///
    /// The step sequence sees only how many coordinates are drawn.
    /// Everything else that tells two batches apart goes in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // A wider cube with a longer tail draws the same number of coordinates.
        //
        //     log_height 10, tail 2  ->  8 drawn
        //     log_height  8, tail 0  ->  8 drawn
        //
        // Both share a step sequence, so the label must carry the two numbers.
        separator
            .instance(&(self.log_height as u64).to_be_bytes())
            .instance(&(self.lookup_point_len as u64).to_be_bytes())
            .instance(&(self.pow_bits as u64).to_be_bytes())
            .instance(&(self.air_degrees.len() as u64).to_be_bytes());

        // The degrees fix how wide a round message is and which family each AIR contributes.
        // They reach the step sequence only through the sumcheck's own seed, inside the bracket.
        for degrees in &self.air_degrees {
            separator
                .instance(&(degrees.constraints as u64).to_be_bytes())
                .instance(&(degrees.interactions as u64).to_be_bytes());
        }

        separator
    }
}

/// Assemble the zerocheck point from the freshly drawn coordinates and the fixed tail.
///
/// ```text
///     tau = [ drawn here | reduction output point ]
/// ```
fn zerocheck_point<EF: Copy>(free: Vec<TranscriptBound<EF>>, tail: &[EF]) -> Vec<EF> {
    let mut tau = Vec::with_capacity(free.len() + tail.len());
    tau.extend(free.into_iter().map(TranscriptBound::into_inner));
    tau.extend_from_slice(tail);
    tau
}

/// Prover-side transcript of one AIR zerocheck.
///
/// Holds the only definition of what a prover draws before its sumcheck.
///
/// The challenger is borrowed, not consumed.
/// The zerocheck runs inside a STARK whose transcript continues afterwards.
pub struct ZerocheckProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: ZerocheckShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZerocheckProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: ZerocheckShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw every challenge the zerocheck owns.
    ///
    /// # Arguments
    ///
    /// - `lookup_tail`: the reduction point the lookup argument already fixed.
    ///
    /// # Panics
    ///
    /// When the tail is not the length the run was described with.
    pub fn challenges(&mut self, lookup_tail: &[EF]) -> ZerocheckChallenges<EF> {
        assert_eq!(
            lookup_tail.len(),
            self.shape.lookup_point_len,
            "the lookup tail must be the length the shape describes"
        );

        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner();
        let beta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BETA)
            .into_inner();

        // With no links to weigh, eta contributes nothing and no step describes it.
        let eta = if self.shape.lookup_point_len > 0 {
            self.state
                .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ETA)
                .into_inner()
        } else {
            EF::ZERO
        };

        let free = if self.shape.free_coordinates() > 0 {
            self.state
                .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                    TAU,
                    self.shape.free_coordinates(),
                    |candidate, _| !candidate.is_zero(),
                )
        } else {
            Vec::new()
        };

        ZerocheckChallenges {
            alpha,
            beta,
            eta,
            tau: zerocheck_point(free, lookup_tail),
        }
    }

    /// Lend the sponge to the constraint sumcheck, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn constraint_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<ConstraintSumcheck>(CONSTRAINT_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<ConstraintSumcheck>(CONSTRAINT_SUMCHECK);
        output
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the zerocheck carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one AIR zerocheck.
///
/// Mirrors the prover side call for call, over the same description.
pub struct ZerocheckVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: ZerocheckShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZerocheckVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: ZerocheckShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Redraw every challenge the zerocheck owns.
    ///
    /// The tail comes from the lookup reduction, which the verifier checked first.
    ///
    /// # Arguments
    ///
    /// - `lookup_tail`: the reduction point the lookup argument already fixed.
    ///
    /// # Panics
    ///
    /// When the tail is not the length the run was described with.
    pub fn challenges(&mut self, lookup_tail: &[EF]) -> ZerocheckChallenges<EF> {
        assert_eq!(
            lookup_tail.len(),
            self.shape.lookup_point_len,
            "the lookup tail must be the length the shape describes"
        );

        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner();
        let beta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BETA)
            .into_inner();

        let eta = if self.shape.lookup_point_len > 0 {
            self.state
                .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ETA)
                .into_inner()
        } else {
            EF::ZERO
        };

        let free = if self.shape.free_coordinates() > 0 {
            self.state
                .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                    TAU,
                    self.shape.free_coordinates(),
                    |candidate, _| !candidate.is_zero(),
                )
        } else {
            Vec::new()
        };

        ZerocheckChallenges {
            alpha,
            beta,
            eta,
            tau: zerocheck_point(free, lookup_tail),
        }
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// A rejection that leaves steps unplayed would otherwise raise a drop-time panic.
    /// That panic would land on top of the error already travelling to the caller.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Lend the sponge to the constraint sumcheck, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn constraint_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<ConstraintSumcheck>(CONSTRAINT_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<ConstraintSumcheck>(CONSTRAINT_SUMCHECK);
        output
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the zerocheck reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x2E80);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// Baseline shape every walk below perturbs exactly one field of.
    fn base_shape() -> ZerocheckShape {
        ZerocheckShape {
            log_height: 10,
            lookup_point_len: 2,
            pow_bits: 4,
            air_degrees: vec![
                AirDegrees {
                    constraints: 3,
                    interactions: 2,
                },
                AirDegrees {
                    constraints: 2,
                    interactions: 0,
                },
            ],
        }
    }

    /// First challenge a shape's seed produces on a fresh sponge.
    fn first_challenge(shape: &ZerocheckShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    /// Assert that perturbing one field of the shape moves the seed.
    fn field_moves_the_seed(name: &str, perturb: impl FnOnce(&mut ZerocheckShape)) {
        let mut shape = base_shape();
        perturb(&mut shape);
        assert_ne!(
            first_challenge(&base_shape()),
            first_challenge(&shape),
            "changing {name} left the seed where it was",
        );
    }

    #[test]
    fn the_same_shape_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the shape.
        let shape = base_shape();
        assert_eq!(first_challenge(&shape), first_challenge(&shape));
    }

    #[test]
    fn every_field_of_the_shape_reaches_the_seed() {
        // Each row below perturbs the shape and asserts the seed moved with it.
        //
        // Some of these move the step sequence and some only the label.
        // The walk does not care which, only that the seed moves either way.
        field_moves_the_seed("log_height", |s| s.log_height += 1);
        field_moves_the_seed("lookup_point_len", |s| s.lookup_point_len += 1);
        field_moves_the_seed("pow_bits", |s| s.pow_bits += 1);
        field_moves_the_seed("air_degrees.len", |s| {
            s.air_degrees.pop();
        });
        field_moves_the_seed("air.constraints", |s| s.air_degrees[0].constraints += 1);
        field_moves_the_seed("air.interactions", |s| s.air_degrees[1].interactions += 1);
    }

    #[test]
    fn a_wider_cube_with_a_longer_tail_still_splits_the_seed() {
        // Both shapes draw eight coordinates, so both describe one step sequence.
        //
        //     log_height 10, tail 2  ->  8 drawn
        //     log_height  8, tail 0  ->  8 drawn
        //
        // Only the instance label separates them.
        let wide = base_shape();
        let narrow = ZerocheckShape {
            log_height: 8,
            lookup_point_len: 0,
            ..base_shape()
        };

        assert_eq!(wide.free_coordinates(), narrow.free_coordinates());
        assert_ne!(first_challenge(&wide), first_challenge(&narrow));
    }

    #[test]
    fn a_batch_without_a_lookup_never_describes_eta() {
        // With no reduction point there are no links, so nothing needs separating.
        //
        //     with a tail : alpha, beta, eta, tau, Begin, End  -> 6 steps
        //     without one : alpha, beta,      tau, Begin, End  -> 5 steps
        let with_lookup = base_shape();
        let without_lookup = ZerocheckShape {
            lookup_point_len: 0,
            ..base_shape()
        };

        assert_eq!(with_lookup.pattern::<F, EF>().len(), 6);
        assert_eq!(without_lookup.pattern::<F, EF>().len(), 5);
    }

    #[test]
    fn a_point_fixed_entirely_by_the_lookup_draws_nothing() {
        // Boundary: a reduction point as wide as the cube leaves no coordinate free.
        //
        //     alpha, beta, eta, Begin, End  -> 5 steps, no tau
        let shape = ZerocheckShape {
            log_height: 4,
            lookup_point_len: 4,
            ..base_shape()
        };

        assert_eq!(shape.free_coordinates(), 0);
        assert_eq!(shape.pattern::<F, EF>().len(), 5);
    }

    #[test]
    fn every_drawn_coordinate_is_nonzero() {
        // Completeness of the rejection: the prover divides by each coordinate.
        let mut challenger = fresh_challenger();
        let mut transcript =
            ZerocheckProverTranscript::<Ch, F, EF>::new(&mut challenger, base_shape());

        let challenges = transcript.challenges(&[EF::ONE, EF::TWO]);
        transcript.constraint_sumcheck(|_| ());
        transcript.finish();

        assert_eq!(challenges.tau.len(), 10);
        assert!(challenges.tau.iter().all(|coord| coord != &EF::ZERO));
        assert_eq!(&challenges.tau[8..], &[EF::ONE, EF::TWO]);
    }

    #[test]
    fn both_sides_draw_the_same_challenges() {
        // Completeness: prover and verifier walk one description over one seed.
        let tail = [EF::ONE, EF::TWO];

        let mut prover_challenger = fresh_challenger();
        let mut prover =
            ZerocheckProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, base_shape());
        let written = prover.challenges(&tail);
        prover.constraint_sumcheck(|_| ());
        prover.finish();

        let mut verifier_challenger = fresh_challenger();
        let mut verifier =
            ZerocheckVerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, base_shape());
        let replayed = verifier.challenges(&tail);
        verifier.constraint_sumcheck(|_| ());
        verifier.finish();

        assert_eq!(written, replayed);
        let prover_next: F = prover_challenger.sample();
        let verifier_next: F = verifier_challenger.sample();
        assert_eq!(prover_next, verifier_next);
    }

    #[test]
    fn an_aborted_verifier_transcript_does_not_panic_on_drop() {
        // A verifier that rejects mid-pattern must let the error reach its caller.
        //
        // Fixture state: alpha and beta drawn, every later step still unplayed.
        let mut challenger = fresh_challenger();
        let mut transcript =
            ZerocheckVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base_shape());
        let _challenges = transcript.challenges(&[EF::ONE, EF::TWO]);

        // Mutation: the run is abandoned before the bracket, as a rejection would abandon it.
        transcript.abort();
        drop(transcript);
    }
}
