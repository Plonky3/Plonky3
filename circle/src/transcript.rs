//! Fiat-Shamir transcript of the Circle polynomial commitment scheme.
//!
//! # Overview
//!
//! One statement of what the Circle PCS transcript is, consumed by both sides.
//!
//! It is built from the FRI parameters and the shape of the claimed openings.
//! Both are known up front, so neither side reads the shape from a proof.
//!
//! Circle FRI folds one bit per round and nothing else.
//! The round count is therefore an arithmetic consequence of the committed heights.
//!
//! # Shape
//!
//! ```text
//!     per claimed opening:  opening point   two coordinates on the circle
//!                           opened values   one extension element per column
//!     batching grinding                     only when the difficulty is positive
//!     batching challenge                    one extension element
//!     first-layer commitment                one opaque value
//!     bivariate challenge                   one extension element
//!     per commit round:     commitment      one opaque value
//!                           grinding        only when the difficulty is positive
//!                           folding         one extension element
//!     final polynomial                      one extension element
//!     query grinding                        only when the difficulty is positive
//!     query indices                         num_queries draws of index_bits bits
//! ```
//!
//! # What is bound
//!
//! - Shape: opened widths, commit-round count, index width, all grinding difficulties.
//! - Instance label: the blowup, and the batch / matrix / point nesting of the claims.
//! - Nothing: a commitment's width, which this layer cannot see.
//! - Nothing: the lambda corrections, which the first-layer opening check pins instead.
//!
//! A wrong commitment width does not desynchronise the transcript.
//! The opening check that recomputes it is what rejects it.
//!
//! A lambda is the same case one step further in.
//!
//! ```text
//!     committed leaf  =  reduced opening - lambda * v_n(P)
//! ```
//!
//! The verifier recomputes that leaf and authenticates it.
//! What it authenticates against is the first-layer commitment, which the seed does bind.
//! A forged lambda therefore cannot survive.
//!
//! # Soundness
//!
//! The opening point enters the transcript as the circle point it is.
//!
//! ```text
//!     zeta on the projective line  ->  P = ((1 - zeta^2)/(1 + zeta^2), 2 zeta/(1 + zeta^2))
//!     P satisfies                      x^2 + y^2 = 1
//! ```
//!
//! Binding `P` alongside the claimed values makes the PCS bind its own statement.
//! The claim is "this matrix takes these values at this point", not "these values exist".

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};
use p3_fri::FriParameters;
use p3_fri::verifier::PowPhase;
use thiserror::Error;

use crate::point::Point;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-circle-pcs";

/// Step label of an opening point.
const OPENING_POINT: &str = "opening_point";

/// Step label of the values claimed at one opening point.
const OPENED_VALUES: &str = "opened_values";

/// Step label of the grinding step guarding the opening-batching challenge.
const BATCH_POW: &str = "batch_pow";

/// Step label of the challenge batching every claim into one FRI instance.
const BATCH_CHALLENGE: &str = "batch_challenge";

/// Step label of the commitment to the reduced openings.
const FIRST_LAYER_COMMITMENT: &str = "first_layer_commitment";

/// Step label of the challenge folding the first, bivariate layer.
const BIVARIATE_CHALLENGE: &str = "bivariate_challenge";

/// Step label of a commit-phase commitment.
const COMMITMENT: &str = "commit_phase_commitment";

/// Step label of the grinding step guarding a folding challenge.
const COMMIT_POW: &str = "commit_pow";

/// Step label of a folding challenge.
const FOLD_CHALLENGE: &str = "fold_challenge";

/// Step label of the constant the folding chain lands on.
const FINAL_POLY: &str = "final_poly";

/// Step label of the grinding step guarding the query indices.
const QUERY_POW: &str = "query_pow";

/// Step label of the query indices.
const QUERY_INDICES: &str = "query_indices";

/// Coordinates a circle point carries: the pair `(x, y)`.
const POINT_COORDINATES: usize = 2;

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// One claimed opening: a point on the circle and the values taken there.
///
/// The two travel together because the transcript binds them together.
pub type OpeningClaim<'a, EF> = (Point<EF>, &'a [EF]);

/// A described transcript step the proof failed to satisfy.
///
/// A Circle PCS proof carries one grinding witness per commit round.
/// It carries one more for the query indices.
///
/// Those counts are checked before the transcript is seeded.
/// A described grinding step therefore always has a witness to replay it.
/// Only the strength of that witness can still fail here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CircleTranscriptFailure {
    /// A grinding witness did not produce the zero bits its step requires.
    #[error("invalid proof-of-work witness for the {phase} phase: {bits} bits required")]
    PowWitness {
        /// Grinding phase whose step rejected the witness.
        phase: PowPhase,
        /// Difficulty that step requires, in bits.
        bits: usize,
    },
}

impl CircleTranscriptFailure {
    /// Grinding phase whose step rejected the witness.
    #[must_use]
    pub const fn phase(&self) -> PowPhase {
        match self {
            Self::PowWitness { phase, .. } => *phase,
        }
    }
}

/// Numbers that fix the transcript of one Circle PCS run.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CirclePcsShape {
    /// Column count of every claim, nested batch -> matrix -> point.
    ///
    /// A matrix opened at no points contributes an empty innermost list.
    pub opened_widths: Vec<Vec<Vec<usize>>>,
    /// Number of commit-phase rounds the folding chain walks.
    pub num_commit_rounds: usize,
    /// Grinding difficulty guarding the opening-batching challenge.
    pub batch_pow_bits: usize,
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
}

impl CirclePcsShape {
    /// Derive the shape of one Circle PCS run from its configuration.
    ///
    /// # Arguments
    ///
    /// - `params`: the FRI parameters the run is driven with.
    /// - `opened_widths`: column count of every claim, nested batch -> matrix -> point.
    /// - `log_max_height`: log-height of the tallest committed low-degree extension.
    /// - `extra_query_index_bits`: index bits the folding strategy asks for beyond the domain.
    ///
    /// # Shape
    ///
    /// Circle folding halves the domain, and the first, bivariate layer takes one bit.
    ///
    /// ```text
    ///     rounds     = log_max_height - 1 - log_blowup
    ///     index_bits = log_max_height - 1 + extra_query_index_bits
    /// ```
    ///
    /// FRI starts one bit below the committed height, because the first layer folded there.
    /// The extra bit the folding strategy asks for is what re-indexes that layer's sibling.
    ///
    /// # Panics
    ///
    /// When `log_max_height` does not exceed the blowup, leaving no bit for the first layer.
    #[must_use]
    pub fn new<M>(
        params: &FriParameters<M>,
        opened_widths: Vec<Vec<Vec<usize>>>,
        log_max_height: usize,
        extra_query_index_bits: usize,
    ) -> Self {
        // The first layer consumes one bit before FRI folds anything.
        assert!(
            log_max_height > params.log_blowup,
            "log_max_height {log_max_height} leaves no bit for the first-layer fold above \
             the blowup 2^{}",
            params.log_blowup,
        );

        Self {
            opened_widths,
            num_commit_rounds: log_max_height - params.log_blowup - 1,
            batch_pow_bits: params.batch_proof_of_work_bits,
            commit_pow_bits: params.commit_proof_of_work_bits,
            query_pow_bits: params.query_proof_of_work_bits,
            num_queries: params.num_queries,
            index_bits: log_max_height - 1 + extra_query_index_bits,
            log_blowup: params.log_blowup,
        }
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
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        // Two steps per claim, three per commit round, and up to seven surrounding steps.
        let num_claims: usize = self.opened_widths.iter().flatten().map(Vec::len).sum();
        let mut steps = Vec::with_capacity(2 * num_claims + 3 * self.num_commit_rounds + 7);

        for &width in self.opened_widths.iter().flatten().flatten() {
            // The point is bound before the values claimed there.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                OPENING_POINT,
                Length::Fixed(POINT_COORDINATES),
            ));

            // One value per column of the matrix this claim opens.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                OPENED_VALUES,
                Length::Fixed(width),
            ));
        }

        // Every claim is fixed before grinding protects the batching challenge.
        if self.batch_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                BATCH_POW,
                Length::Fixed(self.batch_pow_bits),
            ));
        }

        // One challenge collapses every claim into a single DEEP quotient.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BATCH_CHALLENGE,
            Length::Scalar,
        ));

        // The reduced openings are committed before the fold that consumes them.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            FIRST_LAYER_COMMITMENT,
            Length::Scalar,
        ));

        // The first layer folds in `y`, which FRI's own rounds never do.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BIVARIATE_CHALLENGE,
            Length::Scalar,
        ));

        for _ in 0..self.num_commit_rounds {
            // The commitment's encoding belongs to the commitment scheme.
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                COMMITMENT,
                Length::Scalar,
            ));

            // Grinding sits between the commitment and the challenge it protects.
            if self.commit_pow_bits > 0 {
                steps.push(Interaction::algebra::<F, F>(
                    Hierarchy::Atomic,
                    Kind::Pow,
                    COMMIT_POW,
                    Length::Fixed(self.commit_pow_bits),
                ));
            }

            // The folding challenge halves this round's domain.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FOLD_CHALLENGE,
                Length::Scalar,
            ));
        }

        // Circle FRI folds to a constant, so the final polynomial is one element.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            FINAL_POLY,
            Length::Scalar,
        ));

        // Grinding here raises the cost of searching for favourable query indices.
        if self.query_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                QUERY_POW,
                Length::Fixed(self.query_pow_bits),
            ));
        }

        // Every index is drawn at the same width, so they form one step.
        steps.push(Interaction::bits(
            Hierarchy::Atomic,
            Kind::Challenge,
            QUERY_INDICES,
            self.index_bits,
            Length::Fixed(self.num_queries),
        ));

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the protocol identity, this shape, and the remaining parameters.
    ///
    /// A parameter that changes the step sequence is covered by the fingerprint.
    /// The rest go in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The blowup reaches the round count, but nothing in the step sequence names it.
        separator.instance(&(self.log_blowup as u64).to_be_bytes());

        // The step sequence flattens the claims, so their grouping needs binding here.
        //
        //     one batch of two matrices, one point each
        //     one batch of one matrix, two points
        //
        // Both flatten to two claims of the same width, so both describe one shape.
        // Writing the counts as a prefix code gives the two runs distinct seeds.
        separator.instance(&self.opening_layout());

        separator
    }

    /// Serialise the batch / matrix / point nesting of the claims.
    ///
    /// # Returns
    ///
    /// A prefix code, every entry a big-endian `u64`.
    ///
    /// ```text
    ///     [batch count][per batch: matrix count][per matrix: point count]
    /// ```
    ///
    /// Reading it back is unambiguous, so two distinct nestings never collide.
    fn opening_layout(&self) -> Vec<u8> {
        let num_matrices: usize = self.opened_widths.iter().map(Vec::len).sum();
        let mut layout = Vec::with_capacity(8 * (1 + self.opened_widths.len() + num_matrices));

        layout.extend_from_slice(&(self.opened_widths.len() as u64).to_be_bytes());
        for batch in &self.opened_widths {
            layout.extend_from_slice(&(batch.len() as u64).to_be_bytes());
            for matrix in batch {
                layout.extend_from_slice(&(matrix.len() as u64).to_be_bytes());
            }
        }

        layout
    }
}

/// Prover-side transcript of one Circle PCS run.
///
/// Holds the only definition of what a prover writes per phase.
///
/// The challenger is borrowed, not consumed.
/// The PCS runs inside a larger protocol whose transcript continues afterwards.
pub struct CircleProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: CirclePcsShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> CircleProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: CirclePcsShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Bind every claimed opening, grind, then draw the challenge that batches them.
    ///
    /// # Arguments
    ///
    /// - `claims`: every claim in batch, then matrix, then point order.
    ///
    /// # Returns
    ///
    /// The batching challenge `alpha` and the witness when grinding is enabled.
    pub fn batch_phase<'c, I>(&mut self, claims: I) -> (EF, Option<F>)
    where
        I: IntoIterator<Item = OpeningClaim<'c, EF>>,
        EF: 'c,
    {
        for (point, values) in claims {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(
                    OPENING_POINT,
                    &[point.x, point.y],
                );
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OPENED_VALUES, values);
        }

        let witness = (self.shape.batch_pow_bits > 0)
            .then(|| self.state.observe_pow(BATCH_POW, self.shape.batch_pow_bits));
        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCH_CHALLENGE)
            .into_inner();
        (alpha, witness)
    }

    /// Bind the commitment to the reduced openings and draw the bivariate challenge.
    pub fn first_layer<Com>(&mut self, commitment: Com) -> EF
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(FIRST_LAYER_COMMITMENT, commitment);

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BIVARIATE_CHALLENGE)
            .into_inner()
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

    /// Bind the final constant, grind, and draw every query index.
    ///
    /// # Returns
    ///
    /// - The query indices.
    /// - The grinding witness, when the difficulty is positive.
    pub fn query_phase(&mut self, final_poly: EF) -> (Vec<usize>, Option<F>) {
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, &final_poly);

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
            "the Circle PCS carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one Circle PCS run.
///
/// Mirrors the prover side call for call, over the same description.
pub struct CircleVerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: CirclePcsShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> CircleVerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: CirclePcsShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay every claimed opening, check its grind, then redraw the batching challenge.
    ///
    /// # Arguments
    ///
    /// - `claims`: every claim in batch, then matrix, then point order.
    /// - `witness`: the proof's batching witness, ignored when the difficulty is zero.
    ///
    /// # Errors
    ///
    /// When the batching witness misses the configured difficulty.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// The shape and the claims come from the same caller-supplied statement.
    /// A described width and a supplied width therefore cannot differ.
    pub fn batch_phase<'c, I>(
        &mut self,
        claims: I,
        witness: F,
    ) -> Result<EF, CircleTranscriptFailure>
    where
        I: IntoIterator<Item = OpeningClaim<'c, EF>>,
        EF: 'c,
    {
        for (point, values) in claims {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(
                    OPENING_POINT,
                    &[point.x, point.y],
                )
                .expect("a circle point always carries two coordinates");
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OPENED_VALUES, values)
                .expect("the described widths come from these same claims");
        }

        if self.shape.batch_pow_bits > 0 {
            self.state
                .observe_pow(BATCH_POW, self.shape.batch_pow_bits, witness)
                .map_err(|_| CircleTranscriptFailure::PowWitness {
                    phase: PowPhase::Batch,
                    bits: self.shape.batch_pow_bits,
                })?;
        }

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BATCH_CHALLENGE)
            .into_inner())
    }

    /// Replay the first-layer commitment and redraw the bivariate challenge.
    pub fn first_layer<Com>(&mut self, commitment: Com) -> EF
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(FIRST_LAYER_COMMITMENT, commitment);

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BIVARIATE_CHALLENGE)
            .into_inner()
    }

    /// Replay one commit round against the witness the proof carries.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the commit phase requires.
    pub fn commit_round<Com>(
        &mut self,
        commitment: Com,
        witness: F,
    ) -> Result<EF, CircleTranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(COMMITMENT, commitment);

        if self.shape.commit_pow_bits > 0 {
            self.state
                .observe_pow(COMMIT_POW, self.shape.commit_pow_bits, witness)
                .map_err(|_| CircleTranscriptFailure::PowWitness {
                    phase: PowPhase::CommitPhase,
                    bits: self.shape.commit_pow_bits,
                })?;
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
    /// When the witness misses the difficulty the query phase requires.
    pub fn query_phase(
        &mut self,
        final_poly: EF,
        witness: F,
    ) -> Result<Vec<usize>, CircleTranscriptFailure> {
        self.state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, &final_poly);

        if self.shape.query_pow_bits > 0 {
            self.state
                .observe_pow(QUERY_POW, self.shape.query_pow_bits, witness)
                .map_err(|_| CircleTranscriptFailure::PowWitness {
                    phase: PowPhase::Query,
                    bits: self.shape.query_pow_bits,
                })?;
        }

        Ok(self
            .state
            .challenge_bits(QUERY_INDICES, self.shape.index_bits, self.shape.num_queries)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect())
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// A caller that bails part-way through calls this before returning its error.
    ///
    /// Dropping an unfinished driver otherwise panics.
    /// That panic would land on top of an error already travelling to the caller.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the Circle PCS reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;
    type Perm = Poseidon2Mersenne31<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x0C17_C1E5);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A shape with the given claim nesting, one commit round, and no grinding.
    fn shape_with(opened_widths: Vec<Vec<Vec<usize>>>) -> CirclePcsShape {
        CirclePcsShape {
            opened_widths,
            num_commit_rounds: 1,
            batch_pow_bits: 0,
            commit_pow_bits: 0,
            query_pow_bits: 0,
            num_queries: 2,
            index_bits: 8,
            log_blowup: 1,
        }
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: &CirclePcsShape) -> F {
        let mut challenger = fresh_challenger();
        let separator = shape.domain_separator::<F, EF>();
        separator.seed(&mut challenger);
        challenger.sample()
    }

    /// Play a whole prover run over `shape`, returning the batching challenge.
    ///
    /// Every described step is played, so the driver finalises cleanly.
    fn run_prover(shape: &CirclePcsShape, claims: &[(Point<EF>, Vec<EF>)]) -> EF {
        let mut challenger = fresh_challenger();
        let mut transcript =
            CircleProverTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());

        let (alpha, _) = transcript.batch_phase(claims.iter().map(|(p, v)| (*p, v.as_slice())));
        let _bivariate = transcript.first_layer([F::ONE; 8]);
        for _ in 0..shape.num_commit_rounds {
            let _beta = transcript.commit_round([F::TWO; 8]);
        }
        let _indices = transcript.query_phase(EF::ONE);
        transcript.finish();

        alpha
    }

    #[test]
    fn the_derived_round_count_drops_the_first_layer_bit() {
        // Circle folds one bit per round, and the bivariate layer takes one before FRI.
        //
        //     log_max_height = 10, log_blowup = 2  ->  10 - 2 - 1 = 7 rounds
        let params = FriParameters::new_testing((), 0);
        let shape = CirclePcsShape::new(&params, vec![vec![vec![1]]], 10, 1);

        assert_eq!(shape.num_commit_rounds, 10 - params.log_blowup - 1);
        // FRI starts one bit down, and the extra bit re-indexes the first-layer sibling.
        assert_eq!(shape.index_bits, 10);
    }

    #[test]
    fn the_grouping_of_every_claim_reaches_the_seed() {
        // Two nestings flatten to the same claim list, so they share a step sequence.
        //
        //     one batch, two matrices, one point each
        //     one batch, one matrix, two points
        //
        // Only the instance label separates them, so it must carry the counts.
        let split = first_challenge(&shape_with(vec![vec![vec![4], vec![4]]]));
        let merged = first_challenge(&shape_with(vec![vec![vec![4, 4]]]));

        assert_ne!(split, merged);
    }

    #[test]
    fn every_knob_of_the_shape_reaches_the_seed() {
        // Walk each field of the shape and check the seed notices it moving.
        //
        // A knob that changes the step sequence lands in the fingerprint.
        // One that does not lands in the instance label.
        let base = shape_with(vec![vec![vec![4]]]);
        let baseline = first_challenge(&base);

        // Claim width lands in the length of an absorbed step.
        let mut width = base.clone();
        width.opened_widths = vec![vec![vec![5]]];
        assert_ne!(first_challenge(&width), baseline, "claim width");

        let mut rounds = base.clone();
        rounds.num_commit_rounds += 1;
        assert_ne!(first_challenge(&rounds), baseline, "commit round count");

        let mut batch_pow = base.clone();
        batch_pow.batch_pow_bits = 4;
        assert_ne!(first_challenge(&batch_pow), baseline, "batch grinding");

        let mut commit_pow = base.clone();
        commit_pow.commit_pow_bits = 4;
        assert_ne!(first_challenge(&commit_pow), baseline, "commit grinding");

        let mut query_pow = base.clone();
        query_pow.query_pow_bits = 4;
        assert_ne!(first_challenge(&query_pow), baseline, "query grinding");

        let mut queries = base.clone();
        queries.num_queries += 1;
        assert_ne!(first_challenge(&queries), baseline, "query count");

        let mut index_bits = base.clone();
        index_bits.index_bits += 1;
        assert_ne!(first_challenge(&index_bits), baseline, "index width");

        // The blowup leaves the step sequence untouched, so the label carries it.
        let mut blowup = base;
        blowup.log_blowup += 1;
        assert_ne!(first_challenge(&blowup), baseline, "blowup");
    }

    #[test]
    fn the_opening_point_is_bound_alongside_its_values() {
        // Fixture state: one claim of one column, replayed at two different points.
        //
        // Both runs bind the same claimed value, so only the point can separate them.
        let shape = shape_with(vec![vec![vec![1]]]);

        let at = |t: u32| {
            let point = Point::from_projective_line(EF::from_u32(t));
            run_prover(&shape, &[(point, vec![EF::ONE])])
        };

        assert_ne!(at(3), at(5));
    }

    #[test]
    fn a_claimed_value_reaches_the_batching_challenge() {
        // Same point, different claimed value: the challenge must move.
        let shape = shape_with(vec![vec![vec![1]]]);
        let point = Point::from_projective_line(EF::from_u32(3));

        let one = run_prover(&shape, &[(point, vec![EF::ONE])]);
        let two = run_prover(&shape, &[(point, vec![EF::TWO])]);

        assert_ne!(one, two);
    }

    #[test]
    fn both_sides_derive_the_same_challenges() {
        for batch_pow_bits in [0, 8] {
            // Completeness: a verifier replaying the prover's values redraws them exactly.
            let mut shape = shape_with(vec![vec![vec![2]]]);
            shape.batch_pow_bits = batch_pow_bits;
            let point = Point::from_projective_line(EF::from_u32(7));
            let values = vec![EF::ONE, EF::TWO];

            let mut prover_challenger = fresh_challenger();
            let mut prover =
                CircleProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape.clone());
            let (prover_alpha, batch_witness) = prover.batch_phase([(point, values.as_slice())]);
            let prover_bivariate = prover.first_layer([F::ONE; 8]);
            let (prover_beta, _) = prover.commit_round([F::TWO; 8]);
            let (prover_indices, _) = prover.query_phase(EF::ONE);
            prover.finish();

            let mut verifier_challenger = fresh_challenger();
            let mut verifier =
                CircleVerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
            let verifier_alpha = verifier
                .batch_phase(
                    [(point, values.as_slice())],
                    batch_witness.unwrap_or_default(),
                )
                .unwrap();
            let verifier_bivariate = verifier.first_layer([F::ONE; 8]);
            let verifier_beta = verifier
                .commit_round([F::TWO; 8], F::ZERO)
                .expect("a round without grinding replays from the commitment alone");
            let verifier_indices = verifier
                .query_phase(EF::ONE, F::ZERO)
                .expect("a query phase without grinding replays from the constant alone");
            verifier.finish();

            assert_eq!(prover_alpha, verifier_alpha);
            assert_eq!(prover_bivariate, verifier_bivariate);
            assert_eq!(prover_beta, verifier_beta);
            assert_eq!(prover_indices, verifier_indices);
        }
    }

    #[test]
    fn a_commit_round_with_a_weak_grinding_witness_is_rejected() {
        // Described run: one round guarded by 8 bits of grinding.
        //
        // A witness that does not clear the difficulty cannot replay that step.
        let mut shape = shape_with(vec![]);
        shape.commit_pow_bits = 8;

        let mut challenger = fresh_challenger();
        let mut transcript = CircleVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);

        let _alpha = transcript.batch_phase([], F::ZERO).unwrap();
        let _bivariate = transcript.first_layer([F::ONE; 8]);

        let err = transcript
            .commit_round([F::TWO; 8], F::ZERO)
            .expect_err("a witness that misses the difficulty must error");

        // The rejection released the completeness check, so the driver drops quietly.
        assert_eq!(
            err,
            CircleTranscriptFailure::PowWitness {
                phase: PowPhase::CommitPhase,
                bits: 8,
            }
        );
    }

    #[test]
    fn a_query_phase_with_a_weak_grinding_witness_is_rejected() {
        // Described run: no commit rounds, and 8 bits guarding the query indices.
        let mut shape = shape_with(vec![]);
        shape.num_commit_rounds = 0;
        shape.query_pow_bits = 8;

        let mut challenger = fresh_challenger();
        let mut transcript = CircleVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);

        let _alpha = transcript.batch_phase([], F::ZERO).unwrap();
        let _bivariate = transcript.first_layer([F::ONE; 8]);

        let err = transcript
            .query_phase(EF::ONE, F::ZERO)
            .expect_err("a witness that misses the difficulty must error");

        assert_eq!(
            err,
            CircleTranscriptFailure::PowWitness {
                phase: PowPhase::Query,
                bits: 8,
            }
        );
    }

    #[test]
    fn an_aborted_verifier_drops_without_panicking() {
        // A caller that rejects a proof part-way through releases the check first.
        let mut challenger = fresh_challenger();
        let mut transcript =
            CircleVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape_with(vec![]));

        let _alpha = transcript.batch_phase([], F::ZERO).unwrap();
        transcript.abort();
    }
}
