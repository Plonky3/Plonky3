//! Fiat-Shamir transcript of the fractional-GKR reduction.
//!
//! # Overview
//!
//! One statement of what the reduction absorbs and draws, consumed by both sides.
//!
//! A single number fixes it: the variable count of the padded fraction tables.
//! Both sides read that number off the lookup plan they each derive from the AIRs.
//! Neither side ever takes a count out of a proof.
//!
//! # Shape
//!
//! ```text
//!     root denominator                 one extension element
//!     Begin  layer 0
//!             batching                 one extension element
//!             claims                   four extension elements
//!             branch                   one extension element
//!     End    layer 0
//!     ...
//!     Begin  layer i
//!             batching                 one extension element
//!             i times:  round poly     three extension elements
//!                       round challenge  one extension element
//!             claims                   four extension elements
//!             branch                   one extension element
//!     End    layer i
//! ```
//!
//! There are as many layers as the tables have variables.
//!
//! Layer `i` runs exactly `i` sumcheck rounds.
//!
//! The tree doubles going down.
//! Each layer therefore proves over a cube one variable wider than the layer above it.
//! The root layer has no rounds at all.
//!
//! # What is bound
//!
//! - Shape: the layer count, and with it the round count of every layer.
//! - Nothing else: no other number reaches this protocol.
//!
//! # Soundness
//!
//! Each layer sends its two child fractions and then draws the coordinate that picks between them.
//! A prover who saw that coordinate first could pick children that agree at it and nowhere else.
//! The claims step therefore always precedes the branch step.
//!
//! The same ordering holds inside a layer's sumcheck.
//! Every round polynomial is absorbed before the challenge evaluated on it is drawn.
//! Round and branch coordinates reject zero because the AIR zerocheck inherits this point
//! and divides by its coordinates. Both sides use the same rejection sampling.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use super::SplitFraction;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 2;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-multi-stark-fraction-gkr";

/// Step label of the fully reduced root denominator.
const ROOT_DENOMINATOR: &str = "root_denominator";

/// Step label of the bracket around one reduction layer.
const LAYER: &str = "layer";

/// Step label of the challenge batching a layer's numerator and denominator claims.
const BATCHING: &str = "batching";

/// Step label of one sumcheck round polynomial.
const ROUND_POLY: &str = "round_poly";

/// Step label of one sumcheck round challenge.
const ROUND_CHALLENGE: &str = "round_challenge";

/// Step label of the two child fractions a layer reduces to.
const CLAIMS: &str = "claims";

/// Step label of the coordinate selecting between those two children.
const BRANCH: &str = "branch";

/// Evaluations one round polynomial carries.
///
/// The gate is cubic, and its value at one is determined by the running sum.
pub const ROUND_POLY_LEN: usize = 3;

/// Values one layer's closing claim carries.
///
/// Two children, each a numerator and a denominator.
const CLAIMS_LEN: usize = 4;

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of one reduction layer's container.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct ReductionLayer;

/// Numbers that fix the transcript of one fractional-GKR reduction.
///
/// Both sides build this from the lookup plan, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FractionGkrShape {
    /// Variable count of the padded fraction tables the reduction consumes.
    ///
    /// It is also the number of layers, since the tree halves once per layer.
    pub num_variables: usize,
}

impl FractionGkrShape {
    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// One matched bracket per layer always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The root denominator, then four fixed steps and two per round in each layer.
        let n = self.num_variables;
        let mut steps = Vec::with_capacity(1 + n * n + 3 * n);

        // The root denominator is the statement, so it precedes every challenge.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            ROOT_DENOMINATOR,
            Length::Scalar,
        ));

        for layer in 0..n {
            // The bracket makes each layer a container of its own.
            //
            // Two layers of the same round count then still occupy distinct positions.
            steps.push(Interaction::marker::<ReductionLayer>(
                Hierarchy::Begin,
                Kind::Protocol,
                LAYER,
            ));

            // One challenge folds the numerator claim and the denominator claim into one sum.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BATCHING,
                Length::Scalar,
            ));

            // Layer `layer` proves a sum over a cube of `layer` variables.
            for _ in 0..layer {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    ROUND_POLY,
                    Length::Fixed(ROUND_POLY_LEN),
                ));
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    ROUND_CHALLENGE,
                    Length::Fixed(1),
                ));
            }

            // The two children cross the wire together, as one step.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                CLAIMS,
                Length::Fixed(CLAIMS_LEN),
            ));

            // The branch coordinate is drawn only once the children are bound.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                BRANCH,
                Length::Fixed(1),
            ));

            steps.push(Interaction::marker::<ReductionLayer>(
                Hierarchy::End,
                Kind::Protocol,
                LAYER,
            ));
        }

        InteractionPattern::new(steps).expect("one matched bracket per layer is always well formed")
    }

    /// Bind the protocol identity and this shape.
    ///
    /// The variable count moves the step sequence, so the fingerprint covers it.
    /// No other number reaches this protocol, so there is no instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>())
    }
}

/// Prover-side transcript of one fractional-GKR reduction.
///
/// Holds the only definition of what a prover writes per layer.
///
/// The challenger is borrowed, not consumed.
/// The reduction runs inside a lookup argument whose transcript continues afterwards.
pub struct FractionGkrProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the extension field the reduction runs over.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> FractionGkrProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape and bind the root denominator.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
    /// - `shape`: the numbers this run is described with.
    /// - `root_denominator`: denominator of the fully reduced root fraction.
    pub fn new(challenger: &'a mut C, shape: FractionGkrShape, root_denominator: EF) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let mut state = ProverState::new(challenger, &separator);

        // The root denominator is prover-chosen and travels in the proof.
        // It is absorbed here, not written into the driver's own buffer.
        state.observe_extension::<F, EF, FieldToFieldCodec<F>>(ROOT_DENOMINATOR, &root_denominator);

        Self {
            state,
            _ef: PhantomData,
        }
    }

    /// Open one reduction layer and draw the challenge batching its two claims.
    pub fn begin_layer(&mut self) -> EF {
        self.state.begin_protocol::<ReductionLayer>(LAYER);
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCHING)
            .into_inner()
    }

    /// Play one sumcheck round of the open layer.
    ///
    /// # Returns
    ///
    /// The challenge binding the variable this round reduces away.
    pub fn round(&mut self, round_poly: &[EF; ROUND_POLY_LEN]) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, round_poly);
        self.state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                ROUND_CHALLENGE,
                1,
                |candidate, _| !candidate.is_zero(),
            )
            .pop()
            .expect("one nonzero coordinate")
            .into_inner()
    }

    /// Bind the open layer's two child fractions and close it.
    ///
    /// # Returns
    ///
    /// The coordinate selecting between the two children.
    pub fn end_layer(&mut self, claims: &SplitFraction<EF>) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(
                CLAIMS,
                &[claims.n0, claims.d0, claims.n1, claims.d1],
            );
        let branch = self
            .state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                BRANCH,
                1,
                |candidate, _| !candidate.is_zero(),
            )
            .pop()
            .expect("one nonzero coordinate")
            .into_inner();
        self.state.end_protocol::<ReductionLayer>(LAYER);
        branch
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the fractional reduction carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one fractional-GKR reduction.
///
/// Mirrors the prover side call for call, over the same description.
///
/// Every value comes from the proof rather than from a wire.
/// The caller checks their counts against the described shape before building this.
pub struct FractionGkrVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the extension field the reduction runs over.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> FractionGkrVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape and bind the root denominator.
    ///
    /// The arguments match the prover's, so both sides seed identically.
    pub fn new(challenger: &'a mut C, shape: FractionGkrShape, root_denominator: EF) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        let mut state = VerifierState::new(challenger, &separator, &[]);

        // Absorbed from the proof, exactly as the prover absorbed it.
        state.observe_extension::<F, EF, FieldToFieldCodec<F>>(ROOT_DENOMINATOR, &root_denominator);

        Self {
            state,
            _ef: PhantomData,
        }
    }

    /// Open one reduction layer and redraw the challenge batching its two claims.
    pub fn begin_layer(&mut self) -> EF {
        self.state.begin_protocol::<ReductionLayer>(LAYER);
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCHING)
            .into_inner()
    }

    /// Replay one sumcheck round of the open layer.
    ///
    /// The round polynomial is a fixed-size array.
    /// Its width can never disagree with the description.
    ///
    /// # Returns
    ///
    /// The same challenge the prover saw.
    pub fn round(&mut self, round_poly: &[EF; ROUND_POLY_LEN]) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, round_poly)
            .expect("a round polynomial of fixed width always matches the description");
        self.state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                ROUND_CHALLENGE,
                1,
                |candidate, _| !candidate.is_zero(),
            )
            .pop()
            .expect("one nonzero coordinate")
            .into_inner()
    }

    /// Replay the open layer's two child fractions and close it.
    ///
    /// # Returns
    ///
    /// The same branch coordinate the prover saw.
    pub fn end_layer(&mut self, claims: &SplitFraction<EF>) -> EF {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(
                CLAIMS,
                &[claims.n0, claims.d0, claims.n1, claims.d1],
            )
            .expect("a claim of fixed width always matches the description");
        let branch = self
            .state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                BRANCH,
                1,
                |candidate, _| !candidate.is_zero(),
            )
            .pop()
            .expect("one nonzero coordinate")
            .into_inner();
        self.state.end_protocol::<ReductionLayer>(LAYER);
        branch
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the fractional reduction reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn inherited_coordinates_resample_zero_on_both_sides() {
        use p3_field::PrimeCharacteristicRing;
        #[derive(Default)]
        struct ZeroThenOne(usize);
        impl CanObserve<F> for ZeroThenOne {
            fn observe(&mut self, _: F) {}
        }
        impl CanSample<F> for ZeroThenOne {
            fn sample(&mut self) -> F {
                self.0 += 1;
                F::from_bool(self.0.is_multiple_of(2))
            }
        }
        let shape = FractionGkrShape { num_variables: 3 };
        let claims = SplitFraction {
            n0: F::ZERO,
            d0: F::ONE,
            n1: F::ZERO,
            d1: F::ONE,
        };
        let mut pc = ZeroThenOne::default();
        let mut vc = ZeroThenOne::default();
        let mut prover = FractionGkrProverTranscript::<_, F, F>::new(&mut pc, shape, F::ONE);
        let mut verifier = FractionGkrVerifierTranscript::<_, F, F>::new(&mut vc, shape, F::ONE);
        let mut coordinates = Vec::new();
        for layer in 0..3 {
            assert_eq!(prover.begin_layer(), verifier.begin_layer());
            for _ in 0..layer {
                let p = prover.round(&[F::ZERO; ROUND_POLY_LEN]);
                let v = verifier.round(&[F::ZERO; ROUND_POLY_LEN]);
                assert_eq!(p, v);
                coordinates.push(p);
            }
            let p = prover.end_layer(&claims);
            assert_eq!(p, verifier.end_layer(&claims));
            coordinates.push(p);
        }
        prover.finish();
        verifier.finish();
        assert!(coordinates.iter().all(|coordinate| *coordinate != F::ZERO));
        assert_eq!(pc.0, vc.0);
    }

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x6C87);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// First challenge a shape's seed produces on a fresh sponge.
    fn first_challenge(shape: FractionGkrShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    #[test]
    fn the_pattern_grows_by_two_steps_per_extra_round() {
        // A layer contributes two bracket markers, a batching draw, a claim and a branch.
        //
        //     layer 0 : 5 steps
        //     layer 1 : 5 + 2 steps
        //     layer i : 5 + 2i steps
        //
        // With the root denominator in front, three layers make 1 + 15 + 6 = 22 steps.
        let shape = FractionGkrShape { num_variables: 3 };
        assert_eq!(shape.pattern::<F, EF>().len(), 22);
    }

    #[test]
    fn the_variable_count_reaches_the_seed() {
        // The layer count moves the step sequence, so the fingerprint must separate the two.
        assert_ne!(
            first_challenge(FractionGkrShape { num_variables: 4 }),
            first_challenge(FractionGkrShape { num_variables: 5 }),
        );
    }

    #[test]
    fn the_same_shape_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the shape.
        let shape = FractionGkrShape { num_variables: 4 };
        assert_eq!(first_challenge(shape), first_challenge(shape));
    }
}
