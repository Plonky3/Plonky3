//! Fiat-Shamir transcript of the univariate STARK.
//!
//! # Overview
//!
//! One statement of what a uni-STARK run's transcript is, consumed by both sides.
//!
//! It is built from the configuration and from the AIR's shape.
//! Both exist before any proof does, so neither side reads the shape from one.
//!
//! # Shape
//!
//! ```text
//!     trace commitment           one opaque value
//!     preprocessed commitment    only when the AIR has preprocessed columns
//!     public values              num_public_values base elements
//!     alpha                      one extension element
//!     quotient commitment        one opaque value
//!     randomization commitment   only when the PCS is zero-knowledge
//!     out-of-domain grinding     only when the difficulty is positive
//!     zeta                       one extension element
//!     Begin  opening argument    bracket around the delegated run
//!     End    opening argument
//! ```
//!
//! # What is bound
//!
//! - Shape: the public-value count, the two optional commitments, the grinding difficulty.
//! - Instance label: the two trace heights and the rest of the AIR's shape.
//! - Nothing: the constraint polynomials the AIR evaluates.
//!
//! Two AIRs of identical shape whose constraints differ therefore share a seed.
//!
//! ```text
//!     bound:   widths, counts, heights   ->  the frame the constraints live in
//!     unbound: the constraints           ->  checked directly, at zeta
//! ```
//!
//! The verifier evaluates the AIR it was handed at `zeta`.
//! An AIR other than the one it holds is not a transcript question at all.
//!
//! # Soundness
//!
//! `alpha` collapses every constraint into one polynomial identity.
//!
//! ```text
//!     sum_j alpha^j * C_j(x)  =  Z_H(x) * Q(x)
//! ```
//!
//! A prover who learns `alpha` first picks a trace that satisfies the collapsed identity alone.
//! The trace commitment and the public values are therefore absorbed before `alpha` is drawn.
//!
//! `zeta` is the single point that identity is tested at.
//!
//! ```text
//!     absorb Q's commitment   ->   grind   ->   draw zeta
//! ```
//!
//! A prover who learns `zeta` first picks a `Q` that matches at that one point.
//! The grinding step in between prices each retry at `2^ood_pow_bits`.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::BaseAir;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Label, Length, ProverState, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};
use thiserror::Error;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-uni-stark";

/// Step label of the main trace commitment.
const TRACE_COMMITMENT: &str = "trace_commitment";

/// Step label of the preprocessed trace commitment.
const PREPROCESSED_COMMITMENT: &str = "preprocessed_commitment";

/// Step label of the public values.
const PUBLIC_VALUES: &str = "public_values";

/// Step label of the constraint-batching challenge.
const ALPHA: &str = "alpha";

/// Step label of the quotient-chunk commitment.
const QUOTIENT_COMMITMENT: &str = "quotient_commitment";

/// Step label of the zero-knowledge randomization commitment.
const RANDOM_COMMITMENT: &str = "random_commitment";

/// Step label of the grinding step guarding the out-of-domain point.
const OOD_POW: &str = "ood_pow";

/// Step label of the out-of-domain point.
const ZETA: &str = "zeta";

/// Step label of the bracket around the delegated opening argument.
const OPENING_ARGUMENT: &str = "opening_argument";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the sub-protocol the STARK delegates its openings to.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct OpeningArgument;

/// Numbers that fix the transcript of one uni-STARK run.
///
/// Both sides build this from their own configuration and their own AIR.
///
/// The prover's degree comes from the trace it holds.
/// The verifier's comes from a proof field it has already validated against the PCS bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarkShape {
    /// Log-height of the committed trace, including any zero-knowledge extension.
    pub log_ext_degree: usize,
    /// Log-height of the trace before any zero-knowledge extension.
    pub log_degree: usize,
    /// Column count of the main trace.
    pub main_width: usize,
    /// Column count of the preprocessed trace, zero when the AIR declares none.
    pub preprocessed_width: usize,
    /// Number of public values the AIR reads.
    pub num_public_values: usize,
    /// Number of periodic columns the AIR declares.
    pub num_periodic_columns: usize,
    /// Number of chunks the quotient polynomial is split into.
    pub num_quotient_chunks: usize,
    /// Whether the AIR's constraints read the next row of the main trace.
    pub opens_main_next_row: bool,
    /// Whether the AIR's constraints read the next row of the preprocessed trace.
    pub opens_preprocessed_next_row: bool,
    /// Whether the PCS commits to a randomization polynomial.
    pub has_randomization: bool,
    /// Grinding difficulty guarding the out-of-domain point.
    pub ood_pow_bits: usize,
}

impl StarkShape {
    /// Derive the shape of one run from its AIR and its configuration.
    ///
    /// Five of the eleven numbers are read straight off the AIR.
    ///
    /// ```text
    ///     main_width                   air.width()
    ///     num_public_values            air.num_public_values()
    ///     num_periodic_columns         air.num_periodic_columns()
    ///     opens_main_next_row          air.main_next_row_columns()
    ///     opens_preprocessed_next_row  air.preprocessed_next_row_columns()
    /// ```
    ///
    /// # Arguments
    ///
    /// - `air`: the AIR being proven.
    /// - `preprocessed_width`: the preprocessed column count in force, which a verifier key may pin.
    /// - `log_ext_degree`: log-height of the committed trace.
    /// - `log_degree`: log-height of the trace before any zero-knowledge extension.
    /// - `num_quotient_chunks`: number of chunks the quotient polynomial is split into.
    /// - `has_randomization`: whether the PCS commits to a randomization polynomial.
    /// - `ood_pow_bits`: grinding difficulty guarding the out-of-domain point.
    #[must_use]
    pub fn new<F, A>(
        air: &A,
        preprocessed_width: usize,
        log_ext_degree: usize,
        log_degree: usize,
        num_quotient_chunks: usize,
        has_randomization: bool,
        ood_pow_bits: usize,
    ) -> Self
    where
        A: BaseAir<F> + ?Sized,
    {
        Self {
            log_ext_degree,
            log_degree,
            main_width: air.width(),
            preprocessed_width,
            num_public_values: air.num_public_values(),
            num_periodic_columns: air.num_periodic_columns(),
            num_quotient_chunks,
            opens_main_next_row: !air.main_next_row_columns().is_empty(),
            opens_preprocessed_next_row: !air.preprocessed_next_row_columns().is_empty(),
            has_randomization,
            ood_pow_bits,
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// The only nesting is one matched bracket, which always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        // Six mandatory steps, three conditional ones, and the bracket.
        let mut steps = Vec::with_capacity(9);

        // The commitment's encoding belongs to the commitment scheme.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            TRACE_COMMITMENT,
            Length::Scalar,
        ));

        // An AIR with no preprocessed columns commits to none, so the step is absent.
        if self.preprocessed_width > 0 {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                PREPROCESSED_COMMITMENT,
                Length::Scalar,
            ));
        }

        // Public values are the verifier's own input, so they never travel on the wire.
        //
        // The count is the AIR's, which makes it part of the shape rather than of the data.
        steps.push(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Public,
            PUBLIC_VALUES,
            Length::Fixed(self.num_public_values),
        ));

        // One challenge collapses every constraint into a single identity.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ALPHA,
            Length::Scalar,
        ));

        // The quotient answers the identity `alpha` just fixed, so it is bound after it.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            QUOTIENT_COMMITMENT,
            Length::Scalar,
        ));

        // Only a zero-knowledge PCS commits to a randomization polynomial.
        if self.has_randomization {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                RANDOM_COMMITMENT,
                Length::Scalar,
            ));
        }

        // Grinding sits between the last commitment and the point it is tested at.
        if self.ood_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                OOD_POW,
                Length::Fixed(self.ood_pow_bits),
            ));
        }

        // The out-of-domain point tests the identity away from the trace domain.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ZETA,
            Length::Scalar,
        ));

        // The bracket records that a sub-protocol runs here.
        //
        // Its steps live in the callee's own pattern, under the callee's own seed.
        // What this pattern states is that the delegation happens, and where.
        steps.push(Interaction::marker::<OpeningArgument>(
            Hierarchy::Begin,
            Kind::Protocol,
            OPENING_ARGUMENT,
        ));
        steps.push(Interaction::marker::<OpeningArgument>(
            Hierarchy::End,
            Kind::Protocol,
            OPENING_ARGUMENT,
        ));

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity, this shape, and the rest of the AIR's shape.
    ///
    /// A number that changes the step sequence is covered by the fingerprint.
    /// The rest go in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The step sequence sees a preprocessed commitment as present or absent.
        //
        //     width 0   ->  no step
        //     width 3   ->  one step
        //     width 7   ->  one step
        //
        // Two nonzero widths therefore share a sequence, so the value is bound here.
        //
        // The remaining numbers never touch the sequence at all.
        // They are what makes one instance of this AIR's shape distinct from another.
        for value in [
            self.log_ext_degree,
            self.log_degree,
            self.main_width,
            self.preprocessed_width,
            self.num_periodic_columns,
            self.num_quotient_chunks,
        ] {
            separator.instance(&(value as u64).to_be_bytes());
        }

        // Which rows the AIR reads decides how many points each trace is opened at.
        for flag in [self.opens_main_next_row, self.opens_preprocessed_next_row] {
            separator.instance(&u64::from(flag).to_be_bytes());
        }

        separator
    }
}

/// Prover-side transcript of one uni-STARK run.
///
/// Holds the only definition of what a prover writes outside the opening argument.
///
/// The challenger is borrowed, not consumed.
/// The caller keeps whatever it was doing with the sponge before and after.
pub struct StarkProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: StarkShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> StarkProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: StarkShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Bind the committed traces and the public values, then draw `alpha`.
    ///
    /// # Arguments
    ///
    /// - `trace_commitment`: commitment to the main trace.
    /// - `preprocessed_commitment`: commitment to the preprocessed trace, when the AIR has one.
    /// - `public_values`: the public values, exactly as many as the AIR reads.
    ///
    /// # Returns
    ///
    /// The challenge that batches every constraint into one identity.
    ///
    /// # Panics
    ///
    /// When the arguments do not match the shape this run was described with.
    pub fn constraint_phase<Com>(
        &mut self,
        trace_commitment: Com,
        preprocessed_commitment: Option<Com>,
        public_values: &[F],
    ) -> EF
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(TRACE_COMMITMENT, trace_commitment);

        if let Some(commitment) = preprocessed_commitment {
            self.state
                .observe_opaque(PREPROCESSED_COMMITMENT, commitment);
        }

        self.state
            .add_public_scalars::<F, FieldToFieldCodec<F>>(PUBLIC_VALUES, public_values);

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner()
    }

    /// Bind the quotient commitment, grind, and draw `zeta`.
    ///
    /// # Arguments
    ///
    /// - `quotient_commitment`: commitment to the quotient chunks.
    /// - `random_commitment`: commitment to the randomization polynomial, under a hiding PCS.
    ///
    /// # Returns
    ///
    /// - The out-of-domain point.
    /// - The grinding witness, when the difficulty is positive.
    ///
    /// # Panics
    ///
    /// When the arguments do not match the shape this run was described with.
    pub fn ood_phase<Com>(
        &mut self,
        quotient_commitment: Com,
        random_commitment: Option<Com>,
    ) -> (EF, Option<F>)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(QUOTIENT_COMMITMENT, quotient_commitment);

        if let Some(commitment) = random_commitment {
            self.state.observe_opaque(RANDOM_COMMITMENT, commitment);
        }

        let witness = (self.shape.ood_pow_bits > 0)
            .then(|| self.state.observe_pow(OOD_POW, self.shape.ood_pow_bits));

        let zeta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ZETA)
            .into_inner();

        (zeta, witness)
    }

    /// Lend the sponge to the opening argument, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<OpeningArgument>(OPENING_ARGUMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<OpeningArgument>(OPENING_ARGUMENT);
        output
    }

    /// Abandon proof generation after a recoverable error, releasing the drop check.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "a uni-STARK carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one uni-STARK run.
///
/// Mirrors the prover side call for call, over the same description.
pub struct StarkVerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: StarkShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> StarkVerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: StarkShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay a commitment step the shape may or may not have described.
    ///
    /// # Errors
    ///
    /// - `missing`: the step is described and no commitment arrived to replay it.
    /// - `unexpected`: a commitment arrived for a step the run never described.
    fn optional_commitment<Com>(
        &mut self,
        described: bool,
        label: Label,
        commitment: Option<Com>,
        missing: StarkTranscriptFailure,
        unexpected: StarkTranscriptFailure,
    ) -> Result<(), StarkTranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        match (described, commitment) {
            (true, Some(commitment)) => {
                self.state.observe_opaque(label, commitment);
                Ok(())
            }
            (false, None) => Ok(()),
            // Either way the run cannot continue, so the completeness check is released first.
            (true, None) => {
                self.state.abort();
                Err(missing)
            }
            (false, Some(_)) => {
                self.state.abort();
                Err(unexpected)
            }
        }
    }

    /// Replay the committed traces and the public values, then redraw `alpha`.
    ///
    /// # Arguments
    ///
    /// - `trace_commitment`: commitment to the main trace, from the proof.
    /// - `preprocessed_commitment`: commitment to the preprocessed trace, from the verifier key.
    /// - `public_values`: the public values, which are this verifier's own input.
    ///
    /// # Errors
    ///
    /// - The run describes a preprocessed commitment and none arrived.
    /// - A preprocessed commitment arrived for a run described without one.
    ///
    /// # Panics
    ///
    /// When the public values are not the count the AIR declares.
    /// Both sides read that count off the same AIR, so a mismatch is a caller bug.
    pub fn constraint_phase<Com>(
        &mut self,
        trace_commitment: Com,
        preprocessed_commitment: Option<Com>,
        public_values: &[F],
    ) -> Result<EF, StarkTranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(TRACE_COMMITMENT, trace_commitment);

        self.optional_commitment(
            self.shape.preprocessed_width > 0,
            PREPROCESSED_COMMITMENT,
            preprocessed_commitment,
            StarkTranscriptFailure::MissingPreprocessedCommitment {
                width: self.shape.preprocessed_width,
            },
            StarkTranscriptFailure::UnexpectedPreprocessedCommitment,
        )?;

        self.state
            .observe_public_scalars::<F, FieldToFieldCodec<F>>(PUBLIC_VALUES, public_values);

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner())
    }

    /// Replay the quotient commitment and the grind, then redraw `zeta`.
    ///
    /// # Arguments
    ///
    /// - `quotient_commitment`: commitment to the quotient chunks, from the proof.
    /// - `random_commitment`: commitment to the randomization polynomial, from the proof.
    /// - `witness`: the grinding witness the proof carries.
    ///
    /// # Errors
    ///
    /// - The run describes a randomization commitment and none arrived.
    /// - A randomization commitment arrived for a run described without one.
    /// - The witness misses the difficulty its step requires.
    pub fn ood_phase<Com>(
        &mut self,
        quotient_commitment: Com,
        random_commitment: Option<Com>,
        witness: F,
    ) -> Result<EF, StarkTranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(QUOTIENT_COMMITMENT, quotient_commitment);

        self.optional_commitment(
            self.shape.has_randomization,
            RANDOM_COMMITMENT,
            random_commitment,
            StarkTranscriptFailure::MissingRandomCommitment,
            StarkTranscriptFailure::UnexpectedRandomCommitment,
        )?;

        if self.shape.ood_pow_bits > 0 {
            // A failed check poisons the driver on its way out, so this rejection travels alone.
            self.state
                .observe_pow(OOD_POW, self.shape.ood_pow_bits, witness)
                .map_err(|_| StarkTranscriptFailure::OodPowWitness {
                    bits: self.shape.ood_pow_bits,
                })?;
        }

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ZETA)
            .into_inner())
    }

    /// Lend the sponge to the opening argument, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<OpeningArgument>(OPENING_ARGUMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<OpeningArgument>(OPENING_ARGUMENT);
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
            .expect("a uni-STARK reads an empty wire, so no bytes can remain");
    }
}

/// A transcript step the proof failed to satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum StarkTranscriptFailure {
    /// The described preprocessed-commitment step arrived with no commitment.
    #[error(
        "preprocessed commitment step described for a width of {width} arrived with no commitment"
    )]
    MissingPreprocessedCommitment {
        /// Preprocessed column count the run was described with.
        width: usize,
    },
    /// A preprocessed commitment arrived for a run described without one.
    #[error("preprocessed commitment arrived for a run described with a width of 0")]
    UnexpectedPreprocessedCommitment,
    /// The described randomization-commitment step arrived with no commitment.
    #[error("randomization commitment step described by a zero-knowledge PCS arrived with none")]
    MissingRandomCommitment,
    /// A randomization commitment arrived for a run described without one.
    #[error("randomization commitment arrived for a run described without one")]
    UnexpectedRandomCommitment,
    /// The grinding witness did not meet the difficulty its step requires.
    #[error("out-of-domain PoW witness does not meet the required {bits} bits")]
    OodPowWitness {
        /// Grinding difficulty the step requires.
        bits: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::str::from_utf8;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::testing::{
        SeedDigest, assert_seeds_pairwise_distinct, pow_difficulties, seed_digest,
    };
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_security::grinding::{
        GRINDING_VOCABULARY, GrindingBudget, GrindingSite, GrindingSites, RecordedGrind,
        ZeroBitConvention, grinding_step,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    /// An AIR that declares a shape and nothing else.
    ///
    /// Every field is one of the numbers the shape constructor reads off an AIR.
    struct ShapeAir {
        /// Column count of the main trace.
        width: usize,
        /// Column count of the preprocessed trace.
        preprocessed_width: usize,
        /// Number of public values the AIR reads.
        num_public_values: usize,
        /// Number of periodic columns the AIR declares.
        num_periodic_columns: usize,
    }

    impl BaseAir<F> for ShapeAir {
        fn width(&self) -> usize {
            self.width
        }

        fn preprocessed_width(&self) -> usize {
            self.preprocessed_width
        }

        fn num_public_values(&self) -> usize {
            self.num_public_values
        }

        fn num_periodic_columns(&self) -> usize {
            self.num_periodic_columns
        }
    }

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xF21);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A shape with no preprocessed trace, no randomization and no grinding.
    fn plain_shape() -> StarkShape {
        StarkShape {
            log_ext_degree: 5,
            log_degree: 5,
            main_width: 2,
            preprocessed_width: 0,
            num_public_values: 1,
            num_periodic_columns: 0,
            num_quotient_chunks: 2,
            opens_main_next_row: true,
            opens_preprocessed_next_row: false,
            has_randomization: false,
            ood_pow_bits: 0,
        }
    }

    /// The digest of the byte stream a shape seeds its sponge with.
    ///
    /// Comparing seed streams, rather than a sampled challenge, keeps the sponge out of it.
    fn seed_of(shape: &StarkShape) -> SeedDigest {
        seed_digest(&shape.domain_separator::<F, EF>())
    }

    /// Every field of the shape, each bumped by one step away from `plain_shape`.
    ///
    /// One entry per configuration knob, so a knob added without a binding shows up here.
    fn one_step_from_plain() -> Vec<(&'static str, StarkShape)> {
        let mut mutations = Vec::new();

        let mut shape = plain_shape();
        shape.log_ext_degree += 1;
        mutations.push(("log_ext_degree", shape));

        let mut shape = plain_shape();
        shape.log_degree += 1;
        mutations.push(("log_degree", shape));

        let mut shape = plain_shape();
        shape.main_width += 1;
        mutations.push(("main_width", shape));

        let mut shape = plain_shape();
        shape.preprocessed_width += 1;
        mutations.push(("preprocessed_width", shape));

        let mut shape = plain_shape();
        shape.num_public_values += 1;
        mutations.push(("num_public_values", shape));

        let mut shape = plain_shape();
        shape.num_periodic_columns += 1;
        mutations.push(("num_periodic_columns", shape));

        let mut shape = plain_shape();
        shape.num_quotient_chunks += 1;
        mutations.push(("num_quotient_chunks", shape));

        let mut shape = plain_shape();
        shape.opens_main_next_row = false;
        mutations.push(("opens_main_next_row", shape));

        let mut shape = plain_shape();
        shape.opens_preprocessed_next_row = true;
        mutations.push(("opens_preprocessed_next_row", shape));

        let mut shape = plain_shape();
        shape.has_randomization = true;
        mutations.push(("has_randomization", shape));

        let mut shape = plain_shape();
        shape.ood_pow_bits += 1;
        mutations.push(("ood_pow_bits", shape));

        mutations
    }

    #[test]
    fn no_two_configurations_of_the_shape_share_a_seed() {
        // Invariant: the knobs are separated from each other, not merely from a baseline.
        //
        //     plain shape in the set  ->  every knob has to reach the seed
        //     pairwise over the set   ->  no two knobs may land on one seed
        //
        // Two knobs bound as one number differ from the baseline and still agree with each other.
        let mut seeds = vec![("plain", seed_of(&plain_shape()))];
        seeds.extend(
            one_step_from_plain()
                .iter()
                .map(|(field, shape)| (*field, seed_of(shape))),
        );

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn two_airs_differing_only_in_shape_seed_differently() {
        // The AIR's own shape reaches the seed, so two AIRs cannot share one.
        //
        // Two AIRs proven under one configuration, at one trace height, at one chunk count.
        // Everything the run is configured with therefore agrees between them.
        // Only the AIR's own shape differs, one field at a time.
        let base = ShapeAir {
            width: 4,
            preprocessed_width: 0,
            num_public_values: 2,
            num_periodic_columns: 0,
        };

        let shape_of = |air: &ShapeAir| StarkShape::new::<F, _>(air, 0, 6, 6, 2, false, 0);
        let baseline = seed_of(&shape_of(&base));

        // A wider trace.
        let wider = ShapeAir { width: 5, ..base };
        assert_ne!(baseline, seed_of(&shape_of(&wider)));

        // One more public value.
        let more_public = ShapeAir {
            num_public_values: 3,
            ..base
        };
        assert_ne!(baseline, seed_of(&shape_of(&more_public)));

        // One periodic column instead of none.
        let periodic = ShapeAir {
            num_periodic_columns: 1,
            ..base
        };
        assert_ne!(baseline, seed_of(&shape_of(&periodic)));
    }

    #[test]
    fn the_shape_constructor_reads_the_air_it_is_given() {
        // Fixture state: an AIR whose five self-reported numbers are all distinct.
        let air = ShapeAir {
            width: 7,
            preprocessed_width: 3,
            num_public_values: 5,
            num_periodic_columns: 2,
        };

        // The other six numbers come from the caller's configuration.
        let shape = StarkShape::new::<F, _>(&air, 3, 9, 8, 4, true, 6);

        assert_eq!(
            shape,
            StarkShape {
                log_ext_degree: 9,
                log_degree: 8,
                main_width: 7,
                preprocessed_width: 3,
                num_public_values: 5,
                num_periodic_columns: 2,
                num_quotient_chunks: 4,
                // Both default to every column of their trace, so both are non-empty here.
                opens_main_next_row: true,
                opens_preprocessed_next_row: true,
                has_randomization: true,
                ood_pow_bits: 6,
            }
        );
    }

    #[test]
    fn a_run_with_no_grinding_differs_from_one_with_a_single_bit() {
        // At zero bits the step is absent, which is a shorter sequence, not a cheaper one.
        let ungrounded = plain_shape();
        let mut ground = ungrounded.clone();
        ground.ood_pow_bits = 1;

        assert_ne!(seed_of(&ungrounded), seed_of(&ground));
    }

    #[test]
    fn the_two_sides_agree_on_every_challenge() {
        // Fixture state: a run with a preprocessed trace, randomization, and grinding.
        let mut shape = plain_shape();
        shape.preprocessed_width = 3;
        shape.has_randomization = true;
        shape.ood_pow_bits = 1;

        // The public values are the verifier's own input, so both sides absorb the same slice.
        let public_values = [F::from_u8(9)];

        // Prover: play every step, keeping the witness the grind produced.
        let mut prover_challenger = fresh_challenger();
        let mut prover =
            StarkProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape.clone());
        let prover_alpha = prover.constraint_phase([F::ONE; 8], Some([F::TWO; 8]), &public_values);
        let (prover_zeta, witness) = prover.ood_phase([F::from_u8(3); 8], Some([F::from_u8(4); 8]));
        prover.delegate(|_| ());
        prover.finish();

        // Verifier: replay the same steps against the same values.
        let mut verifier_challenger = fresh_challenger();
        let mut verifier =
            StarkVerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
        let verifier_alpha = verifier
            .constraint_phase([F::ONE; 8], Some([F::TWO; 8]), &public_values)
            .expect("the described preprocessed commitment is supplied");
        let verifier_zeta = verifier
            .ood_phase(
                [F::from_u8(3); 8],
                Some([F::from_u8(4); 8]),
                witness.expect("a positive difficulty grinds a witness"),
            )
            .expect("the witness is the one the prover ground");
        verifier.delegate(|_| ());
        verifier.finish();

        assert_eq!(prover_alpha, verifier_alpha);
        assert_eq!(prover_zeta, verifier_zeta);

        // Both sponges advanced identically, so they still agree on what comes next.
        let prover_next: F = prover_challenger.sample();
        let verifier_next: F = verifier_challenger.sample();
        assert_eq!(prover_next, verifier_next);
    }

    #[test]
    fn a_described_preprocessed_commitment_that_never_arrives_is_rejected() {
        // Described run: an AIR with three preprocessed columns, so the step exists.
        let mut shape = plain_shape();
        shape.preprocessed_width = 3;

        let mut challenger = fresh_challenger();
        let mut transcript = StarkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);

        // A described commitment step cannot be replayed with nothing to feed it.
        let err = transcript
            .constraint_phase([F::ONE; 8], None, &[F::ONE])
            .expect_err("a described commitment step with no commitment must error");

        assert_eq!(
            err,
            StarkTranscriptFailure::MissingPreprocessedCommitment { width: 3 }
        );
    }

    #[test]
    fn a_preprocessed_commitment_nobody_described_is_rejected() {
        // Described run: an AIR with no preprocessed columns, so there is no step to replay.
        let mut challenger = fresh_challenger();
        let mut transcript =
            StarkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, plain_shape());

        let err = transcript
            .constraint_phase([F::ONE; 8], Some([F::TWO; 8]), &[F::ONE])
            .expect_err("a commitment outside the described sequence must error");

        assert_eq!(
            err,
            StarkTranscriptFailure::UnexpectedPreprocessedCommitment
        );
    }

    #[test]
    fn a_missing_randomization_commitment_is_rejected() {
        // Described run: a zero-knowledge PCS, so the randomization step exists.
        let mut shape = plain_shape();
        shape.has_randomization = true;

        let mut challenger = fresh_challenger();
        let mut transcript = StarkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _alpha = transcript
            .constraint_phase([F::ONE; 8], None, &[F::ONE])
            .expect("the constraint phase is unaffected by the randomization step");

        let err = transcript
            .ood_phase([F::TWO; 8], None, F::ZERO)
            .expect_err("a described commitment step with no commitment must error");

        assert_eq!(err, StarkTranscriptFailure::MissingRandomCommitment);
    }

    #[test]
    fn a_forged_grinding_witness_is_rejected() {
        // Described run: sixteen bits of grinding, which no fixed witness meets by luck.
        let mut shape = plain_shape();
        shape.ood_pow_bits = 16;

        let mut challenger = fresh_challenger();
        let mut transcript = StarkVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _alpha = transcript
            .constraint_phase([F::ONE; 8], None, &[F::ONE])
            .expect("the constraint phase runs before any grinding step");

        let err = transcript
            .ood_phase([F::TWO; 8], None, F::ZERO)
            .expect_err("a witness below the difficulty must error");

        assert_eq!(err, StarkTranscriptFailure::OodPowWitness { bits: 16 });
    }

    #[test]
    fn the_delegation_bracket_leaves_the_sponge_untouched() {
        // Invariant: the markers are structural.
        // They record the delegation, they absorb nothing.
        //
        // Fixture state: two runs over the same shape, one bracketing an empty delegation.
        let shape = plain_shape();
        let public_values = [F::from_u8(11)];

        let mut bracketed_challenger = fresh_challenger();
        let mut bracketed =
            StarkVerifierTranscript::<Ch, F, EF>::new(&mut bracketed_challenger, shape.clone());
        let _alpha = bracketed
            .constraint_phase([F::ONE; 8], None, &public_values)
            .unwrap();
        let bracketed_zeta = bracketed.ood_phase([F::TWO; 8], None, F::ZERO).unwrap();
        bracketed.delegate(|_| ());
        bracketed.finish();

        // The prover side plays the same steps and must land on the same point.
        let mut plain_challenger = fresh_challenger();
        let mut plain = StarkProverTranscript::<Ch, F, EF>::new(&mut plain_challenger, shape);
        let _alpha = plain.constraint_phase([F::ONE; 8], None, &public_values);
        let (plain_zeta, witness) = plain.ood_phase([F::TWO; 8], None);
        plain.delegate(|_| ());
        plain.finish();

        assert_eq!(bracketed_zeta, plain_zeta);
        assert_eq!(witness, None);

        let bracketed_next: F = bracketed_challenger.sample();
        let plain_next: F = plain_challenger.sample();
        assert_eq!(bracketed_next, plain_next);
    }

    #[test]
    fn a_public_value_the_verifier_holds_reaches_the_challenges() {
        // Public values never travel on the wire, so only the absorb binds them.
        //
        //     verifier holds [7]  ->  one alpha
        //     verifier holds [8]  ->  another
        let mut with_seven = fresh_challenger();
        let mut seven = StarkVerifierTranscript::<Ch, F, EF>::new(&mut with_seven, plain_shape());
        let seven_alpha = seven
            .constraint_phase([F::ONE; 8], None, &[F::from_u8(7)])
            .unwrap();
        let _zeta = seven.ood_phase([F::TWO; 8], None, F::ZERO).unwrap();
        seven.delegate(|_| ());
        seven.finish();

        let mut with_eight = fresh_challenger();
        let mut eight = StarkVerifierTranscript::<Ch, F, EF>::new(&mut with_eight, plain_shape());
        let eight_alpha = eight
            .constraint_phase([F::ONE; 8], None, &[F::from_u8(8)])
            .unwrap();
        let _zeta = eight.ood_phase([F::TWO; 8], None, F::ZERO).unwrap();
        eight.delegate(|_| ());
        eight.finish();

        assert_ne!(seven_alpha, eight_alpha);
    }
    /// This protocol's name, as the vocabulary table keys it.
    fn protocol() -> &'static str {
        from_utf8(NAME).expect("the protocol name is ASCII")
    }

    #[test]
    fn the_grinding_vocabulary_maps_the_one_grind_this_protocol_describes() {
        // The security model keys its table on the name and the label bound here.
        let ood = grinding_step(protocol(), OOD_POW).expect("the out-of-domain grind is mapped");
        assert_eq!(ood.site, GrindingSite::OutOfDomain);
        assert_eq!(ood.zero_bits, ZeroBitConvention::Elided);

        // A uni-STARK has no lookups, so it describes no lookup grind and owns no second row.
        assert_eq!(
            GRINDING_VOCABULARY
                .iter()
                .filter(|step| step.protocol == protocol())
                .count(),
            1,
        );
    }

    #[test]
    fn the_out_of_domain_grind_carries_the_difficulty_the_model_credits() {
        // Invariant: one number reaches the pattern and the report by two routes.
        //
        //     ood_pow_bits  --pattern-->        Kind::Pow Length::Fixed
        //                   --GrindingSites-->  out_of_domain
        //
        // Elided at zero, so the sweep covers the step being absent and present.
        for ood_pow_bits in [0, 1, 8] {
            let shape = StarkShape {
                log_ext_degree: 5,
                log_degree: 5,
                main_width: 2,
                preprocessed_width: 0,
                num_public_values: 1,
                num_periodic_columns: 0,
                num_quotient_chunks: 2,
                opens_main_next_row: true,
                opens_preprocessed_next_row: false,
                has_randomization: false,
                ood_pow_bits,
            };

            let recorded: Vec<_> = pow_difficulties(&shape.pattern::<F, EF>())
                .into_iter()
                .map(|(label, bits)| RecordedGrind::new(protocol(), label, bits))
                .collect();

            GrindingBudget::from_sites(&GrindingSites {
                out_of_domain: ood_pow_bits,
                ..GrindingSites::NONE
            })
            .check(&[protocol()], &recorded)
            .unwrap_or_else(|mismatch| panic!("{mismatch}"));
        }
    }
}
