//! Fiat-Shamir transcript of the FRI low-degree test.
//!
//! # Overview
//!
//! One statement of what FRI's transcript is, consumed by both sides.
//!
//! It is built from the parameters and the folding schedule.
//! Both are known up front, so neither side reads the shape from a proof.
//!
//! # Shape
//!
//! ```text
//!     per round:  commitment   one opaque value
//!                 grinding     only when the difficulty is positive
//!                 folding      one extension element
//!     final polynomial         final_poly_len extension elements
//!     query grinding           only when the difficulty is positive
//!     query indices            num_queries draws of index_bits bits
//! ```
//!
//! # What is bound
//!
//! - Shape: round count, index width, both grinding difficulties.
//! - Instance label: blowup and arity cap, which leave the shape unchanged.
//! - Nothing: a commitment's width, which this layer cannot see.
//!
//! A wrong commitment width does not desynchronise the transcript.
//! The opening check that recomputes it is what rejects it.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};

use crate::{FriParameters, fold_schedule};

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-fri";

/// Step label of a commit-phase commitment.
const COMMITMENT: &str = "commit_phase_commitment";

/// Step label of the grinding step guarding a folding challenge.
const COMMIT_POW: &str = "commit_pow";

/// Step label of a folding challenge.
const FOLD_CHALLENGE: &str = "fold_challenge";

/// Step label of the final polynomial's coefficients.
const FINAL_POLY: &str = "final_poly";

/// Step label of the grinding step guarding the query indices.
const QUERY_POW: &str = "query_pow";

/// Step label of the query indices.
const QUERY_INDICES: &str = "query_indices";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Numbers that fix the transcript of one FRI run.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriShape {
    /// One log-arity per commit round, in round order.
    pub log_arities: Vec<usize>,
    /// Coefficient count of the final polynomial.
    pub final_poly_len: usize,
    /// Grinding difficulty guarding each folding challenge.
    pub commit_pow_bits: usize,
    /// Grinding difficulty guarding the query indices.
    pub query_pow_bits: usize,
    /// Number of query indices drawn.
    pub num_queries: usize,
    /// Bit width of each query index.
    pub index_bits: usize,
    /// Log of the evaluation-domain blowup.
    pub log_blowup: usize,
    /// Largest arity any single round may use.
    pub max_log_arity: usize,
}

/// Derive the shape of one FRI run from its configuration.
///
/// # Arguments
///
/// - `params`: the protocol parameters.
/// - `input_log_heights`: log-heights of the folding inputs, strictly decreasing.
/// - `index_bits`: bit width of each query index.
#[must_use]
pub fn fri_shape<M>(
    params: &FriParameters<M>,
    input_log_heights: &[usize],
    index_bits: usize,
) -> FriShape {
    FriShape {
        // The schedule is a function of the heights and the parameters.
        log_arities: fold_schedule(
            input_log_heights,
            params.log_blowup + params.log_final_poly_len,
            params.max_log_arity,
        ),
        final_poly_len: params.final_poly_len(),
        commit_pow_bits: params.commit_proof_of_work_bits,
        query_pow_bits: params.query_proof_of_work_bits,
        num_queries: params.num_queries,
        index_bits,
        log_blowup: params.log_blowup,
        max_log_arity: params.max_log_arity,
    }
}

/// Describe the transcript of one FRI run.
///
/// # Panics
///
/// Never in practice.
/// A flat sequence of leaf steps always passes structural validation.
#[must_use]
pub fn pattern<F, EF>(shape: &FriShape) -> InteractionPattern
where
    F: PrimeField64,
    EF: ExtensionField<F>,
{
    // Up to three steps per commit round, then three closing steps.
    let mut steps = Vec::with_capacity(3 * shape.log_arities.len() + 3);

    for _ in &shape.log_arities {
        // The commitment's encoding belongs to the commitment scheme.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            COMMITMENT,
            Length::Scalar,
        ));

        // Grinding sits between the commitment and the challenge it protects.
        if shape.commit_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                COMMIT_POW,
                Length::Fixed(shape.commit_pow_bits),
            ));
        }

        // The folding challenge collapses this round's arity.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            FOLD_CHALLENGE,
            Length::Scalar,
        ));
    }

    // The final polynomial is sent in full, so it is one fixed-length step.
    steps.push(Interaction::algebra::<F, EF>(
        Hierarchy::Atomic,
        Kind::Message,
        FINAL_POLY,
        Length::Fixed(shape.final_poly_len),
    ));

    // Grinding here raises the cost of searching for favourable query indices.
    if shape.query_pow_bits > 0 {
        steps.push(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            QUERY_POW,
            Length::Fixed(shape.query_pow_bits),
        ));
    }

    // Every index is drawn at the same width, so they form one step.
    steps.push(Interaction::bits(
        Hierarchy::Atomic,
        Kind::Challenge,
        QUERY_INDICES,
        shape.index_bits,
        Length::Fixed(shape.num_queries),
    ));

    InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
}

/// Bind the protocol identity, the shape, and the remaining parameters.
///
/// A parameter that changes the step sequence is covered by its fingerprint.
/// The rest go in the instance label.
#[must_use]
pub fn domain_separator<F, EF>(shape: &FriShape) -> DomainSeparator<Alphabet<F>>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
{
    let mut separator = DomainSeparator::new(VERSION, NAME, pattern::<F, EF>(shape));

    // Blowup and arity cap leave the step sequence untouched.
    // They still change what the protocol is, so they are bound here.
    for value in [shape.log_blowup, shape.max_log_arity] {
        separator.instance(&(value as u64).to_be_bytes());
    }

    separator
}

/// Prover-side transcript of one FRI run.
///
/// Holds the only definition of what a prover writes per round.
///
/// The challenger is borrowed, not consumed.
/// FRI runs inside a larger protocol whose transcript continues afterwards.
pub struct ProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: FriShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: FriShape) -> Self {
        let separator = domain_separator::<F, EF>(&shape);
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Play one commit round: bind the commitment, grind, draw the challenge.
    ///
    /// # Returns
    ///
    /// - The folding challenge for this round.
    /// - The grinding witness, when the difficulty is positive.
    pub fn commit_round<Com>(&mut self, commitment: Com) -> (EF, Option<F>)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        // Bind the commitment before the challenge that folds against it.
        self.state.observe_opaque(COMMITMENT, commitment);

        let witness = (self.shape.commit_pow_bits > 0).then(|| {
            self.state
                .observe_pow(COMMIT_POW, self.shape.commit_pow_bits)
        });

        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(FOLD_CHALLENGE)
            .into_inner();

        (challenge, witness)
    }

    /// Bind the final polynomial, grind, and draw every query index.
    ///
    /// # Returns
    ///
    /// - The query indices.
    /// - The grinding witness, when the difficulty is positive.
    pub fn query_phase(&mut self, final_poly: &[EF]) -> (Vec<usize>, Option<F>) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, final_poly);

        let witness = (self.shape.query_pow_bits > 0)
            .then(|| self.state.observe_pow(QUERY_POW, self.shape.query_pow_bits));

        let indices = self
            .state
            .challenge_bits(QUERY_INDICES, self.shape.index_bits, self.shape.num_queries)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();

        (indices, witness)
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "FRI carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one FRI run.
///
/// Mirrors the prover side call for call, over the same description.
pub struct VerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: FriShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: FriShape) -> Self {
        let separator = domain_separator::<F, EF>(&shape);
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay one commit round against the witness the proof carries.
    ///
    /// # Errors
    ///
    /// When the witness misses the required difficulty.
    pub fn commit_round<Com>(
        &mut self,
        commitment: Com,
        witness: Option<F>,
    ) -> Result<EF, TranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(COMMITMENT, commitment);

        if self.shape.commit_pow_bits > 0 {
            let witness = witness.expect("a positive difficulty requires a witness per round");
            self.state
                .observe_pow(COMMIT_POW, self.shape.commit_pow_bits, witness)
                .map_err(|_| TranscriptFailure::PowWitness)?;
        }

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(FOLD_CHALLENGE)
            .into_inner())
    }

    /// Replay the closing steps and redraw every query index.
    ///
    /// # Errors
    ///
    /// When the witness misses the required difficulty.
    pub fn query_phase(
        &mut self,
        final_poly: &[EF],
        witness: Option<F>,
    ) -> Result<Vec<usize>, TranscriptFailure> {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, final_poly);

        if self.shape.query_pow_bits > 0 {
            let witness = witness.expect("a positive difficulty requires a witness");
            self.state
                .observe_pow(QUERY_POW, self.shape.query_pow_bits, witness)
                .map_err(|_| TranscriptFailure::PowWitness)?;
        }

        Ok(self
            .state
            .challenge_bits(QUERY_INDICES, self.shape.index_bits, self.shape.num_queries)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect())
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("FRI reads an empty wire, so no bytes can remain");
    }
}

/// A transcript step the proof failed to satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptFailure {
    /// A grinding witness did not meet the required difficulty.
    PowWitness,
}
