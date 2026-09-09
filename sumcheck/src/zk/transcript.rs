//! Fiat-Shamir transcript of the hiding sumcheck.
//!
//! One description of everything a masked batch absorbs.
//!
//! Prover, verifier and simulator all drive that same description.
//!
//! # Shape
//!
//! ```text
//!     prelude      ->  1 extension element, drawn or bound
//!     mask oracle  ->  1 commitment
//!     mu_tilde     ->  1 extension element
//!     eps          ->  1 extension element, drawn
//!
//!     round i:  wire       ->  max(ell_zk, 3) - 1 extension elements
//!               grinding   ->  only when the difficulty is positive
//!               challenge  ->  1 extension element
//! ```
//!
//! A round polynomial has `max(ell_zk, 3)` coefficients, and the linear one stays off the wire.
//!
//! The affine round identity reconstructs it.
//!
//! # Two preludes, one description
//!
//! ```text
//!     recorded claims  ->  one challenge weights the claims a layout already recorded
//!     inherited claim  ->  one scalar arrives from the caller, bound before anything is drawn
//! ```
//!
//! Everything after the prelude is identical, so one description covers both.
//!
//! They are distinct steps, so a side that binds where the other draws fails loudly.
//!
//! # Sub-protocol
//!
//! A masked sumcheck is always a phase of something larger.
//!
//! It seeds a transcript of its own from a borrowed sponge, and hands the sponge back.
//!
//! The surrounding protocol brackets this run inside its own description.
//!
//! That is what lets the two descriptions be written independently and still compose.
//!
//! # What the shape binds
//!
//! A fingerprint of the description enters the sponge before any step runs.
//!
//! ```text
//!     num_rounds  ->  round count, and so the mask oracle's column count
//!     pow_bits    ->  presence and difficulty of the per-round grinding step
//!     ell_zk      ->  wire width, and the instance label on top of it
//!     prelude     ->  which of the two opening steps is played
//! ```
//!
//! The mask oracle is one opaque step, binding its content and not its width.
//!
//! One column is committed per round, so the round count bounds that width.
//!
//! ```text
//!     rate, randomness budget, domain size  ->  committed content only, never seen here
//! ```
//!
//! The protocol one layer up carries all three in its own instance label.
//!
//! # Simulator
//!
//! The witness-free simulator of Lemma 6.4 plays this same description.
//!
//! Hiding holds only while its transcript is indistinguishable from the prover's.
//!
//! A divergence between the two is a loud pattern failure here, not a silent loss of hiding.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::{ExtensionField, Field};

use crate::error::SumcheckError;

/// Version byte bound into the transcript seed.
///
/// Bumping it separates two revisions of this protocol.
/// It does so even when their step sequences agree.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
///
/// Distinct from the plain sumcheck names, so no masked batch shares a seed with a plain one.
const NAME: &[u8] = b"p3-sumcheck-hvzk";

/// Step label of the challenge that weights the claims a layout recorded.
const CLAIM_BATCHING: &str = "claim_batching";

/// Step label of the scalar claim a batch inherits from its caller.
const JOINT_CLAIM: &str = "joint_claim";

/// Step label of the interleaved mask oracle of a batch.
const MASK_COMMITMENT: &str = "mask_commitment";

/// Step label of `mu_tilde`, the sum of the batch's mask evaluations over the cube.
const MU_TILDE: &str = "mu_tilde";

/// Step label of `eps`, the challenge combining the mask and plain pieces.
const MASK_COMBINATION: &str = "mask_combination";

/// Step label of the wire coefficients one round sends.
const ROUND_POLY: &str = "round_poly";

/// Step label of a per-round grinding step.
const ROUND_POW: &str = "round_pow";

/// Step label of a per-round challenge.
const ROUND_CHALLENGE: &str = "round_challenge";

/// Coefficient count of the plain round polynomial.
///
/// The plain piece is quadratic, so it occupies three coefficient slots.
/// A round polynomial is therefore never shorter than this, whatever the mask length is.
const PLAIN_COEFFICIENTS: usize = 3;

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// How a batch reaches the claim it runs against.
///
/// The two variants are two different opening steps, never two readings of one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZkPrelude {
    /// One drawn challenge weights the claims a layout already recorded.
    ///
    /// The claim is then derived on both sides from those recorded claims.
    RecordedClaims,
    /// One scalar claim arrives from the caller and is bound before anything is drawn.
    ///
    /// The two sides supply it independently, so the step is what keeps them honest.
    InheritedClaim,
}

/// Numbers that fix the transcript of one masked batch of sumcheck rounds.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkSumcheckShape {
    /// Which of the two opening steps this batch plays.
    pub prelude: ZkPrelude,
    /// Number of rounds the batch runs, and so the number of masks it commits.
    ///
    /// The mask oracle interleaves one column per round.
    /// This count is therefore what binds that oracle's width.
    pub num_rounds: usize,
    /// Message length of the zero-knowledge mask code.
    ///
    /// Sets the wire width through `max(ell_zk, 3) - 1`.
    pub ell_zk: usize,
    /// Grinding difficulty guarding each round's challenge, or zero to omit grinding.
    pub pow_bits: usize,
}

impl ZkSumcheckShape {
    /// Collect the numbers that fix a batch opening on the claims a layout recorded.
    ///
    /// # Arguments
    ///
    /// - `num_rounds`: number of rounds the batch runs.
    /// - `ell_zk`: message length of the mask code.
    /// - `pow_bits`: grinding difficulty per round, or zero to omit grinding.
    #[must_use]
    pub const fn new_batching(num_rounds: usize, ell_zk: usize, pow_bits: usize) -> Self {
        Self {
            prelude: ZkPrelude::RecordedClaims,
            num_rounds,
            ell_zk,
            pow_bits,
        }
    }

    /// Collect the numbers that fix a batch opening on a claim it inherits.
    ///
    /// # Arguments
    ///
    /// - `num_rounds`: number of rounds the batch runs.
    /// - `ell_zk`: message length of the mask code.
    /// - `pow_bits`: grinding difficulty per round, or zero to omit grinding.
    #[must_use]
    pub const fn new_inherited(num_rounds: usize, ell_zk: usize, pow_bits: usize) -> Self {
        Self {
            prelude: ZkPrelude::InheritedClaim,
            num_rounds,
            ell_zk,
            pow_bits,
        }
    }

    /// Coefficient count of one round polynomial.
    ///
    /// ```text
    ///     h_size = max(ell_zk, 3)
    /// ```
    #[must_use]
    pub const fn round_poly_len(&self) -> usize {
        if self.ell_zk > PLAIN_COEFFICIENTS {
            self.ell_zk
        } else {
            PLAIN_COEFFICIENTS
        }
    }

    /// Number of coefficients one round puts on the wire.
    ///
    /// The linear coefficient is dropped, and the verifier rebuilds it from the round identity.
    #[must_use]
    pub const fn wire_len(&self) -> usize {
        self.round_poly_len() - 1
    }

    /// Reject a configuration that no masked batch can run under.
    ///
    /// Both sides derive this shape from their own configuration.
    ///
    /// A rejection here is therefore a setup failure, not a malformed proof.
    ///
    /// A verifier still reports it instead of panicking.
    ///
    /// # Errors
    ///
    /// - The base field has characteristic two, which Lemma 6.4 rules out.
    /// - The mask is too short to cover the degree-2 plain piece.
    /// - The batch runs no rounds at all.
    pub fn validate<F: Field>(&self) -> Result<(), SumcheckError> {
        // Lemma 6.4 divides by two when it inverts the endpoint identity.
        if F::TWO == F::ZERO {
            return Err(SumcheckError::EvenCharacteristic);
        }
        // A mask of degree ell_zk - 1 must reach the degree-2 plain piece it hides.
        if self.ell_zk < PLAIN_COEFFICIENTS {
            return Err(SumcheckError::MaskTooShort {
                ell_zk: self.ell_zk,
            });
        }
        // A batch with no rounds has no mask to commit and no claim to reduce.
        if self.num_rounds == 0 {
            return Err(SumcheckError::NoRounds);
        }
        Ok(())
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// A flat sequence of leaf steps always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Four prelude steps, then up to three per round.
        let mut steps = Vec::with_capacity(4 + 3 * self.num_rounds);

        // The prelude fixes the claim the batch runs against.
        //
        // Matching every variant keeps a future third opening from sharing this description.
        steps.push(match self.prelude {
            ZkPrelude::RecordedClaims => Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                CLAIM_BATCHING,
                Length::Scalar,
            ),
            ZkPrelude::InheritedClaim => Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                JOINT_CLAIM,
                Length::Scalar,
            ),
        });

        // One interleaved oracle carries every mask of the batch.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            MASK_COMMITMENT,
            Length::Scalar,
        ));

        // The endpoint sum is fixed before the challenge that mixes it with the plain piece.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            MU_TILDE,
            Length::Scalar,
        ));
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            MASK_COMBINATION,
            Length::Scalar,
        ));

        let wire_len = self.wire_len();
        for _ in 0..self.num_rounds {
            // The round polynomial is one step carrying every transmitted coefficient.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                ROUND_POLY,
                Length::Fixed(wire_len),
            ));

            // Grinding sits between the polynomial and the challenge it protects.
            //
            // The difficulty travels inside the step.
            // A verifier expecting a cheaper grind therefore fails the shape check.
            if self.pow_bits > 0 {
                steps.push(Interaction::algebra::<F, F>(
                    Hierarchy::Atomic,
                    Kind::Pow,
                    ROUND_POW,
                    Length::Fixed(self.pow_bits),
                ));
            }

            // The challenge binds the variable this round reduces away.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                ROUND_CHALLENGE,
                Length::Scalar,
            ));
        }

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity, this shape, and the mask length.
    ///
    /// The round count, the difficulty and the prelude all move the step sequence.
    /// The fingerprint therefore carries them.
    ///
    /// # Soundness
    ///
    /// The mask length reaches the description only through the wire width.
    ///
    /// ```text
    ///     wire_len = max(ell_zk, 3) - 1
    ///
    ///     ell_zk = 2  ->  wire_len = 2
    ///     ell_zk = 3  ->  wire_len = 2
    /// ```
    ///
    /// The clamp is not injective, so two mask lengths can share a width.
    /// The label carries `ell_zk` itself, so they cannot share a seed.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        separator.instance(&(self.ell_zk as u64).to_be_bytes());

        separator
    }
}

/// Prover-side transcript of one masked batch of sumcheck rounds.
///
/// # Overview
///
/// Holds the only definition of what a prover writes per masked round.
///
/// Both hiding provers and the witness-free simulator drive them through this type.
/// No two of the three can then drift from each other, or from the verifier.
///
/// # Borrowing
///
/// The challenger is borrowed, not consumed.
///
/// A masked sumcheck runs inside a larger protocol.
/// That protocol's own transcript continues where this one stops.
pub struct ZkProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this batch was described with.
    shape: ZkSumcheckShape,
    /// Marker for the extension field the masks and rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZkProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the batch.
    /// - `shape`: the numbers that fix this batch's transcript.
    pub fn new(challenger: &'a mut C, shape: ZkSumcheckShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Draw the challenge that weights the claims a layout recorded.
    ///
    /// # Panics
    ///
    /// When the batch was described as inheriting its claim instead.
    pub fn batching_challenge(&mut self) -> EF {
        assert_eq!(
            self.shape.prelude,
            ZkPrelude::RecordedClaims,
            "a batch described as inheriting its claim draws no batching challenge",
        );

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(CLAIM_BATCHING)
            .into_inner()
    }

    /// Bind the scalar claim this batch inherits from its caller.
    ///
    /// # Panics
    ///
    /// When the batch was described as batching recorded claims instead.
    pub fn bind_claim(&mut self, claim: EF) {
        assert_eq!(
            self.shape.prelude,
            ZkPrelude::InheritedClaim,
            "a batch described as batching recorded claims binds no inherited claim",
        );

        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(JOINT_CLAIM, &claim);
    }

    /// Bind the mask oracle and its endpoint sum, then draw the combining challenge.
    ///
    /// # Arguments
    ///
    /// - `commitment`: the batch's interleaved mask oracle.
    /// - `mu_tilde`: sum of the mask evaluations over the boolean cube.
    ///
    /// # Returns
    ///
    /// The challenge `eps` that scales the plain piece against the masks.
    pub fn masks<Com>(&mut self, commitment: Com, mu_tilde: EF) -> EF
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        // The oracle is fixed before the value summed out of the masks behind it.
        self.state.observe_opaque(MASK_COMMITMENT, commitment);
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(MU_TILDE, &mu_tilde);

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(MASK_COMBINATION)
            .into_inner()
    }

    /// Play one round: bind the transmitted coefficients, grind, and draw the challenge.
    ///
    /// # Returns
    ///
    /// - The challenge that binds this round's variable.
    /// - The grinding witness, when the difficulty is positive.
    ///
    /// The caller stores both in its own proof.
    ///
    /// # Panics
    ///
    /// When the wire is not the width the batch was described with.
    pub fn round(&mut self, wire: &[EF]) -> (EF, Option<F>) {
        // Bind every transmitted coefficient before the challenge evaluated against them.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, wire);

        // Grinding raises the cost of searching for a favourable challenge.
        let witness = (self.shape.pow_bits > 0)
            .then(|| self.state.observe_pow(ROUND_POW, self.shape.pow_bits));

        // Draw the challenge for the caller to fold with.
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner();

        (challenge, witness)
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the batch was described with.
    pub fn finish(self) {
        // Nothing was written to the driver's own buffer.
        // Closing is therefore purely the check that the description was consumed.
        assert!(
            self.state.finalize().is_empty(),
            "the hiding sumcheck carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one masked batch of sumcheck rounds.
///
/// Mirrors the prover side call for call, over the same description.
///
/// Every value comes from the proof rather than from a wire.
///
/// The described widths are what reject one the proof got wrong.
pub struct ZkVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this batch was described with.
    shape: ZkSumcheckShape,
    /// Index of the next round to play, used to place a round failure.
    round: usize,
    /// Marker for the extension field the masks and rounds carry.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZkVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// The argument matches the prover's, so both sides seed identically.
    pub fn new(challenger: &'a mut C, shape: ZkSumcheckShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();

        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            round: 0,
            _ef: PhantomData,
        }
    }

    /// Draw the challenge that weights the claims this verifier recorded.
    ///
    /// # Panics
    ///
    /// When the batch was described as inheriting its claim instead.
    pub fn batching_challenge(&mut self) -> EF {
        assert_eq!(
            self.shape.prelude,
            ZkPrelude::RecordedClaims,
            "a batch described as inheriting its claim draws no batching challenge",
        );

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(CLAIM_BATCHING)
            .into_inner()
    }

    /// Bind the scalar claim this batch inherits from its caller.
    ///
    /// The prover bound its own view of the same scalar.
    /// A caller that hands over a different one moves every challenge that follows.
    ///
    /// # Panics
    ///
    /// When the batch was described as batching recorded claims instead.
    pub fn bind_claim(&mut self, claim: EF) {
        assert_eq!(
            self.shape.prelude,
            ZkPrelude::InheritedClaim,
            "a batch described as batching recorded claims binds no inherited claim",
        );

        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(JOINT_CLAIM, &claim);
    }

    /// Bind the mask oracle and its endpoint sum, then draw the combining challenge.
    ///
    /// # Arguments
    ///
    /// - `commitment`: the mask oracle the proof carries.
    /// - `mu_tilde`: the endpoint sum the proof carries.
    ///
    /// # Returns
    ///
    /// The challenge `eps` the prover saw.
    pub fn masks<Com>(&mut self, commitment: Com, mu_tilde: EF) -> EF
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        // The oracle is fixed before the value summed out of the masks behind it.
        self.state.observe_opaque(MASK_COMMITMENT, commitment);
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(MU_TILDE, &mu_tilde);

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(MASK_COMBINATION)
            .into_inner()
    }

    /// Replay one round: bind the wire, re-check the grind, draw the challenge.
    ///
    /// Every value comes from the proof, so every disagreement is a rejection.
    ///
    /// A round that returns without error absorbed exactly `wire_len` coefficients.
    /// The caller may index the wire on that guarantee alone.
    ///
    /// # Errors
    ///
    /// - The wire is not the width the batch was described with.
    /// - Grinding is enabled and the round carries no witness.
    /// - The witness misses the required difficulty.
    pub fn round(&mut self, wire: &[EF], witness: Option<F>) -> Result<EF, SumcheckError> {
        let round = self.round;
        self.round += 1;

        // Bind every transmitted coefficient before the challenge evaluated against them.
        //
        // The count comes from the proof, so a mismatch is a rejection.
        // The driver poisons itself on the way out, which releases its own drop check.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(ROUND_POLY, wire)
            .map_err(|_| SumcheckError::WireSizeMismatch {
                round,
                expected: self.shape.wire_len(),
                actual: wire.len(),
            })?;

        // Re-run the prover's grinding step on the witness it committed to.
        if self.shape.pow_bits > 0 {
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                self.state.abort();
                return Err(SumcheckError::MissingPowWitness {
                    round,
                    difficulty: self.shape.pow_bits,
                });
            };
            // A failing check releases the completeness check on its own way out.
            self.state
                .observe_pow(ROUND_POW, self.shape.pow_bits, witness)
                .map_err(|_| SumcheckError::InvalidPowWitness {
                    round,
                    difficulty: self.shape.pow_bits,
                })?;
        }

        // Draw the same challenge the prover saw.
        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_CHALLENGE)
            .into_inner())
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When fewer steps were played than the batch was described with.
    pub fn finish(self) {
        // The proof carries every value, so no unread wire bytes can remain.
        self.state
            .finalize()
            .expect("the hiding sumcheck reads an empty wire");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

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

    /// Commitment stand-in: the sponge absorbs it as one opaque value.
    type Com = [F; 4];

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A commitment stand-in built from one number.
    fn commitment(tag: u32) -> Com {
        [F::from_u32(tag), F::ZERO, F::ONE, F::TWO]
    }

    /// Baseline shape every walk below perturbs exactly one field of.
    const fn base_shape() -> ZkSumcheckShape {
        ZkSumcheckShape::new_batching(3, 4, 0)
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: ZkSumcheckShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    /// Assert that perturbing one field of the shape moves the seed.
    fn field_moves_the_seed(name: &str, perturb: impl FnOnce(&mut ZkSumcheckShape)) {
        let mut tweaked = base_shape();
        perturb(&mut tweaked);
        assert_ne!(
            first_challenge(base_shape()),
            first_challenge(tweaked),
            "changing {name} left the seed where it was",
        );
    }

    #[test]
    fn the_same_shape_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the shape.
        assert_eq!(first_challenge(base_shape()), first_challenge(base_shape()));
    }

    #[test]
    fn every_number_that_shapes_a_batch_reaches_the_seed() {
        // Walk `ZkSumcheckShape` field by field.
        field_moves_the_seed("prelude", |s| s.prelude = ZkPrelude::InheritedClaim);
        field_moves_the_seed("num_rounds", |s| s.num_rounds += 1);
        field_moves_the_seed("ell_zk", |s| s.ell_zk += 1);
        field_moves_the_seed("pow_bits", |s| s.pow_bits += 8);

        // Two positive difficulties differ only inside the grinding steps.
        assert_ne!(
            first_challenge(ZkSumcheckShape::new_batching(3, 4, 8)),
            first_challenge(ZkSumcheckShape::new_batching(3, 4, 9)),
        );

        // The clamp hides one mask length behind another, and the label does not.
        //
        //     ell_zk = 3  ->  wire_len = 2
        //     ell_zk = 2  ->  wire_len = 2
        assert_eq!(
            ZkSumcheckShape::new_batching(3, 2, 0).wire_len(),
            ZkSumcheckShape::new_batching(3, 3, 0).wire_len(),
        );
        assert_ne!(
            first_challenge(ZkSumcheckShape::new_batching(3, 2, 0)),
            first_challenge(ZkSumcheckShape::new_batching(3, 3, 0)),
        );
    }

    #[test]
    fn a_configuration_no_batch_can_run_under_is_rejected() {
        // A mask shorter than the plain quadratic cannot hide it.
        assert_eq!(
            ZkSumcheckShape::new_batching(3, 2, 0).validate::<F>(),
            Err(SumcheckError::MaskTooShort { ell_zk: 2 }),
        );

        // A batch of no rounds has no mask to commit and no claim to reduce.
        assert_eq!(
            ZkSumcheckShape::new_batching(0, 4, 0).validate::<F>(),
            Err(SumcheckError::NoRounds),
        );

        // The baseline shape is accepted.
        assert_eq!(base_shape().validate::<F>(), Ok(()));
    }

    /// Drive a full prover-side batch and hand back everything it produced.
    ///
    /// Returns the batching challenge, the combining challenge and the per-round challenges.
    fn drive_prover(
        challenger: &mut Ch,
        shape: ZkSumcheckShape,
        commitment: Com,
        mu_tilde: EF,
        wires: &[Vec<EF>],
    ) -> (EF, EF, Vec<EF>, Vec<F>) {
        let mut transcript = ZkProverTranscript::<Ch, F, EF>::new(challenger, shape);
        let alpha = transcript.batching_challenge();
        let eps = transcript.masks(commitment, mu_tilde);
        let mut gammas = Vec::new();
        let mut witnesses = Vec::new();
        for wire in wires {
            let (gamma, witness) = transcript.round(wire);
            gammas.push(gamma);
            witnesses.extend(witness);
        }
        transcript.finish();
        (alpha, eps, gammas, witnesses)
    }

    /// Replay a full verifier-side batch against recorded values.
    fn drive_verifier(
        challenger: &mut Ch,
        shape: ZkSumcheckShape,
        commitment: Com,
        mu_tilde: EF,
        wires: &[Vec<EF>],
        witnesses: &[F],
    ) -> Result<(EF, EF, Vec<EF>), SumcheckError> {
        let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(challenger, shape);
        let alpha = transcript.batching_challenge();
        let eps = transcript.masks(commitment, mu_tilde);
        let mut gammas = Vec::new();
        for (round, wire) in wires.iter().enumerate() {
            let witness = (shape.pow_bits > 0).then(|| witnesses[round]);
            gammas.push(transcript.round(wire, witness)?);
        }
        transcript.finish();
        Ok((alpha, eps, gammas))
    }

    /// Two rounds of wire coefficients for the baseline shape.
    fn honest_wires() -> Vec<Vec<EF>> {
        vec![
            vec![EF::ONE, EF::TWO, EF::from_u8(3)],
            vec![EF::from_u8(4), EF::from_u8(5), EF::from_u8(6)],
        ]
    }

    #[test]
    fn both_sides_of_a_batch_draw_the_same_challenges() {
        // Completeness: the two drivers walk one description and land on one stream.
        //
        // Fixture state: 2 rounds, mask length 4, no grinding.
        let shape = ZkSumcheckShape::new_batching(2, 4, 0);
        let wires = honest_wires();

        let mut prover_challenger = fresh_challenger();
        let (alpha, eps, gammas, witnesses) = drive_prover(
            &mut prover_challenger,
            shape,
            commitment(7),
            EF::from_u8(9),
            &wires,
        );
        assert!(witnesses.is_empty(), "no grinding means no witness");

        let mut verifier_challenger = fresh_challenger();
        let (replayed_alpha, replayed_eps, replayed_gammas) = drive_verifier(
            &mut verifier_challenger,
            shape,
            commitment(7),
            EF::from_u8(9),
            &wires,
            &[],
        )
        .expect("the honest batch must replay");

        assert_eq!(alpha, replayed_alpha);
        assert_eq!(eps, replayed_eps);
        assert_eq!(gammas, replayed_gammas);

        // The sponge is handed back in one state, so the surrounding protocol stays in step.
        assert_eq!(
            CanSample::<F>::sample(&mut prover_challenger),
            CanSample::<F>::sample(&mut verifier_challenger),
        );
    }

    #[test]
    fn a_recorded_grinding_witness_replays_to_the_same_challenges() {
        // Completeness of the guarded path: the witness is absorbed, not merely checked.
        //
        // Fixture state: 2 rounds guarded by 4 bits each.
        let shape = ZkSumcheckShape::new_batching(2, 4, 4);
        let wires = honest_wires();

        let mut prover_challenger = fresh_challenger();
        let (_, eps, gammas, witnesses) = drive_prover(
            &mut prover_challenger,
            shape,
            commitment(7),
            EF::from_u8(9),
            &wires,
        );
        assert_eq!(witnesses.len(), 2, "one witness per guarded round");

        let mut verifier_challenger = fresh_challenger();
        let (_, replayed_eps, replayed_gammas) = drive_verifier(
            &mut verifier_challenger,
            shape,
            commitment(7),
            EF::from_u8(9),
            &wires,
            &witnesses,
        )
        .expect("the recorded witnesses must replay");

        assert_eq!(eps, replayed_eps);
        assert_eq!(gammas, replayed_gammas);
    }

    /// The challenges a verifier redraws from one recorded batch.
    fn replay(commitment: Com, mu_tilde: EF, wires: &[Vec<EF>]) -> (EF, Vec<EF>) {
        let mut challenger = fresh_challenger();
        let shape = ZkSumcheckShape::new_batching(wires.len(), 4, 0);
        let (_, eps, gammas) =
            drive_verifier(&mut challenger, shape, commitment, mu_tilde, wires, &[])
                .expect("a well-shaped batch always replays");
        (eps, gammas)
    }

    #[test]
    fn a_perturbed_mask_commitment_moves_every_later_challenge() {
        // Invariant: the masks are fixed before the challenge that combines them.
        //
        // Mutation: swap the committed oracle for another.
        let wires = honest_wires();
        assert_ne!(
            replay(commitment(7), EF::from_u8(9), &wires),
            replay(commitment(8), EF::from_u8(9), &wires),
        );
    }

    #[test]
    fn a_perturbed_mu_tilde_moves_every_later_challenge() {
        // Invariant: the endpoint sum is bound before `eps` weighs it against the plain piece.
        //
        // Mutation: bump the endpoint sum by one.
        let wires = honest_wires();
        assert_ne!(
            replay(commitment(7), EF::from_u8(9), &wires),
            replay(commitment(7), EF::from_u8(10), &wires),
        );
    }

    #[test]
    fn a_perturbed_wire_coefficient_moves_every_later_challenge() {
        // Invariant: the wire is bound before the challenge evaluated against it.
        //
        // Without that, a prover could pick the round polynomial to suit the challenge.
        //
        // Mutation: bump one coefficient of the first round, once per position.
        let honest = honest_wires();
        for position in 0..honest[0].len() {
            let mut tampered = honest.clone();
            tampered[0][position] += EF::ONE;
            assert_ne!(
                replay(commitment(7), EF::from_u8(9), &honest),
                replay(commitment(7), EF::from_u8(9), &tampered),
                "tampering with wire coefficient {position} left the stream where it was",
            );
        }
    }

    #[test]
    fn a_perturbed_inherited_claim_moves_every_later_challenge() {
        // Invariant: the inherited claim is a step, so the two sides cannot differ on it silently.
        //
        // Mutation: hand the verifier a claim one away from the prover's.
        let shape = ZkSumcheckShape::new_inherited(1, 4, 0);
        let wire = [EF::ONE, EF::TWO, EF::from_u8(3)];

        let draw = |claim: EF| {
            let mut challenger = fresh_challenger();
            let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
            transcript.bind_claim(claim);
            let eps = transcript.masks(commitment(7), EF::from_u8(9));
            let gamma = transcript.round(&wire, None).unwrap();
            transcript.finish();
            (eps, gamma)
        };

        assert_ne!(draw(EF::from_u8(11)), draw(EF::from_u8(12)));
    }

    #[test]
    fn a_wire_of_the_wrong_width_is_rejected() {
        // Described batch: 1 round of mask length 4, so a 3-coefficient wire.
        //
        //     described:    Fixed(3)
        //     proof holds:  2        -> rejected on round 0
        //
        // This is the path a downstream crate reaches with a proof built for another mask.
        // A panic here would compound with the driver's own drop check and abort the process.
        let mut challenger = fresh_challenger();
        let shape = ZkSumcheckShape::new_batching(1, 4, 0);
        let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _ = transcript.batching_challenge();
        let _ = transcript.masks(commitment(7), EF::from_u8(9));

        let err = transcript
            .round(&[EF::ONE, EF::TWO], None)
            .expect_err("a wire outside the described width must error");

        assert_eq!(
            err,
            SumcheckError::WireSizeMismatch {
                round: 0,
                expected: 3,
                actual: 2,
            },
        );

        // The rejection leaves the round half-played, with one round still described.
        //
        // Absorbing the width error is what releases the completeness check.
        drop(transcript);
    }

    #[test]
    fn a_round_missing_its_grinding_witness_is_rejected() {
        // Described batch: 1 round guarded by 4 bits.
        //
        // A described grinding step cannot be replayed with no witness to feed it.
        let mut challenger = fresh_challenger();
        let shape = ZkSumcheckShape::new_batching(1, 4, 4);
        let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _ = transcript.batching_challenge();
        let _ = transcript.masks(commitment(7), EF::from_u8(9));

        let err = transcript
            .round(&[EF::ONE, EF::TWO, EF::from_u8(3)], None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(
            err,
            SumcheckError::MissingPowWitness {
                round: 0,
                difficulty: 4,
            },
        );
    }

    #[test]
    fn a_grinding_witness_below_the_difficulty_is_rejected() {
        // Described batch: 1 round guarded by 12 bits.
        //
        // Mutation: hand the round a witness that was never ground at all.
        let mut challenger = fresh_challenger();
        let shape = ZkSumcheckShape::new_batching(1, 4, 12);
        let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _ = transcript.batching_challenge();
        let _ = transcript.masks(commitment(7), EF::from_u8(9));

        let err = transcript
            .round(&[EF::ONE, EF::TWO, EF::from_u8(3)], Some(F::ZERO))
            .expect_err("a witness below the required difficulty must error");

        assert_eq!(
            err,
            SumcheckError::InvalidPowWitness {
                round: 0,
                difficulty: 12,
            },
        );
    }
}
