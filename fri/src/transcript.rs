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
//! - Instance label: blowup, arity cap, and the arity of every round.
//! - Nothing: a commitment's width, which this layer cannot see.
//!
//! A wrong commitment width does not desynchronise the transcript.
//! The opening check that recomputes it is what rejects it.
//!
//! # Why the arity of every round is bound separately
//!
//! The step sequence carries the round count.
//!
//! It emits one group of steps per round.
//!
//! It does not carry the arity each round folds by.
//!
//! ```text
//!     [3, 3, 2]   three rounds, eight levels folded
//!     [2, 3, 3]   three rounds, eight levels folded
//! ```
//!
//! Both give the same step sequence.
//!
//! The query index width follows the total, and the total is equal.
//!
//! Real configurations reach both.
//!
//! So the values go in the instance label.
//!
//! That is what gives the two runs distinct seeds.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};
use thiserror::Error;

use crate::verifier::PowPhase;
use crate::{FriParameters, fold_schedule};

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
///
/// Visible to the crate so the opening argument that brackets this protocol can
/// name it when checking their two halves of one `FriParameters` together.
pub(crate) const NAME: &[u8] = b"p3-fri";

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

impl FriShape {
    /// Derive the shape of one FRI run from its configuration.
    ///
    /// # Arguments
    ///
    /// - `params`: the protocol parameters.
    /// - `input_log_heights`: log-heights of the folding inputs, strictly decreasing.
    /// - `index_bits`: bit width of each query index.
    ///
    /// # Panics
    ///
    /// When the input heights are not strictly decreasing.
    #[must_use]
    pub fn new<M>(
        params: &FriParameters<M>,
        input_log_heights: &[usize],
        index_bits: usize,
    ) -> Self {
        // The schedule is a function of the heights and the parameters.
        let log_arities = fold_schedule(
            input_log_heights,
            params.log_blowup + params.log_final_poly_len,
            params.max_log_arity,
        );

        Self::with_schedule(params, log_arities, index_bits)
    }

    /// Build the shape around a schedule the caller already holds.
    ///
    /// Every other number still comes from the parameters.
    ///
    /// A caller holding its own schedule skips the derivation.
    /// A deliberately forged schedule is the other case: no derivation produces one.
    ///
    /// # Arguments
    ///
    /// - `params`: the protocol parameters.
    /// - `log_arities`: one log-arity per commit round, in round order.
    /// - `index_bits`: bit width of each query index.
    #[must_use]
    pub const fn with_schedule<M>(
        params: &FriParameters<M>,
        log_arities: Vec<usize>,
        index_bits: usize,
    ) -> Self {
        Self {
            log_arities,
            final_poly_len: params.final_poly_len(),
            commit_pow_bits: params.commit_proof_of_work_bits,
            query_pow_bits: params.query_proof_of_work_bits,
            num_queries: params.num_queries,
            index_bits,
            log_blowup: params.log_blowup,
            max_log_arity: params.max_log_arity,
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
        // Up to three steps per commit round, then three closing steps.
        let mut steps = Vec::with_capacity(3 * self.log_arities.len() + 3);

        for _ in &self.log_arities {
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
            Length::Fixed(self.final_poly_len),
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

        // Blowup and arity cap leave the step sequence untouched.
        // They still change what the protocol is, so they are bound here.
        for value in [self.log_blowup, self.max_log_arity] {
            separator.instance(&(value as u64).to_be_bytes());
        }

        // The round count alone does not pin the arity each round folds by.
        //
        //     [3, 3, 2]   three rounds, eight levels folded
        //     [2, 3, 3]   three rounds, eight levels folded
        //
        // The step sequence emits one group per round, so it agrees on the two.
        //
        // The query index width follows the total, and the total is equal too.
        //
        // Binding the values here is what gives the two runs distinct seeds.
        for &log_arity in &self.log_arities {
            separator.instance(&(log_arity as u64).to_be_bytes());
        }

        separator
    }
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
        let separator = shape.domain_separator::<F, EF>();
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
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay every commit round the run was described with.
    ///
    /// # Overview
    ///
    /// The round count is a length of the described transcript.
    ///
    /// The folding schedule fixes it.
    ///
    /// Both sides derive that schedule from their own configuration.
    ///
    /// A different round count is a different shape.
    ///
    /// It is rejected here, not replayed.
    ///
    /// ```text
    ///     described:  one commitment and one witness per round
    ///     supplied:   whatever the proof carries
    ///
    ///     equal      ->  replay, one folding challenge per round
    ///     different  ->  reject, nothing absorbed
    /// ```
    ///
    /// # Why the count is checked before anything is absorbed
    ///
    /// A step played past the end of a description is a programming error.
    ///
    /// The driver reports it by panicking.
    ///
    /// Untrusted input must never reach that path.
    ///
    /// Checking the count first keeps a malformed run on the rejection path.
    ///
    /// # Arguments
    ///
    /// - `commitments`: one commitment per round, in round order.
    /// - `witnesses`: one grinding witness per round, in the same order.
    ///
    /// # Returns
    ///
    /// One folding challenge per round, in round order.
    ///
    /// # Errors
    ///
    /// - The commitments do not number one per described round.
    /// - The witnesses do not number one per described round.
    /// - A witness misses the difficulty its grinding step requires.
    pub fn commit_rounds<Com>(
        &mut self,
        commitments: &[Com],
        witnesses: &[F],
    ) -> Result<Vec<EF>, TranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        // The described round count.
        //
        // It comes from the folding schedule alone.
        let expected = self.shape.log_arities.len();

        // A shorter run would leave described steps unplayed.
        //
        // A longer one would walk off the end of the description.
        //
        // Both are rejected before the first commitment is absorbed.
        if commitments.len() != expected {
            self.state.abort();
            return Err(TranscriptFailure::CommitRoundCount {
                expected,
                got: commitments.len(),
            });
        }

        // Each round replays its own grinding step.
        //
        // So the witnesses form a per-round list of the same length.
        if witnesses.len() != expected {
            self.state.abort();
            return Err(TranscriptFailure::CommitPowWitnessCount {
                expected,
                got: witnesses.len(),
            });
        }

        // Both counts agree with the description.
        //
        // Every round can now be replayed.
        commitments
            .iter()
            .zip(witnesses)
            .map(|(commitment, &witness)| self.commit_round(commitment.clone(), Some(witness)))
            .collect()
    }

    /// Replay one commit round against the witness the proof carries.
    ///
    /// Private because the described round count bounds how often this may run.
    ///
    /// Only the entry point that checks a supplied count against it may call it.
    ///
    /// # Errors
    ///
    /// - The round carries no witness for a described grinding step.
    /// - The witness misses the required difficulty.
    fn commit_round<Com>(
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
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                self.state.abort();
                return Err(TranscriptFailure::MissingPowWitness(PowPhase::CommitPhase));
            };
            self.state
                .observe_pow(COMMIT_POW, self.shape.commit_pow_bits, witness)
                .map_err(|_| TranscriptFailure::PowWitness(PowPhase::CommitPhase))?;
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
    /// - The final polynomial does not carry the described number of coefficients.
    /// - The run carries no witness for a described grinding step.
    /// - The witness misses the required difficulty.
    pub fn query_phase(
        &mut self,
        final_poly: &[EF],
        witness: Option<F>,
    ) -> Result<Vec<usize>, TranscriptFailure> {
        // The coefficient count comes from the proof, so a mismatch is a rejection.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, final_poly)
            .map_err(|_| TranscriptFailure::FinalPolyLen {
                expected: self.shape.final_poly_len,
                got: final_poly.len(),
            })?;

        if self.shape.query_pow_bits > 0 {
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                self.state.abort();
                return Err(TranscriptFailure::MissingPowWitness(PowPhase::Query));
            };
            self.state
                .observe_pow(QUERY_POW, self.shape.query_pow_bits, witness)
                .map_err(|_| TranscriptFailure::PowWitness(PowPhase::Query))?;
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
///
/// Only the two grinding steps a FRI run replays are reachable here.
/// The one guarding the batching challenge is drawn before the run starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TranscriptFailure {
    /// A grinding witness did not meet the difficulty its step requires.
    #[error("{0} phase PoW witness does not meet the required difficulty")]
    PowWitness(PowPhase),
    /// A described grinding step arrived with no witness to replay it.
    #[error("{0} phase PoW step arrived with no witness")]
    MissingPowWitness(PowPhase),
    /// The final polynomial carries a coefficient count the run never described.
    #[error("final polynomial length mismatch: expected {expected}, got {got}")]
    FinalPolyLen {
        /// Coefficient count the run was described with.
        expected: usize,
        /// Coefficient count the proof carries.
        got: usize,
    },
    /// The run carries a round count the folding schedule does not fix.
    #[error("commit round count mismatch: expected {expected}, got {got}")]
    CommitRoundCount {
        /// Round count the folding schedule fixes.
        expected: usize,
        /// Round count the proof carries.
        got: usize,
    },
    /// The per-round grinding witnesses do not number one per commit round.
    #[error("commit phase PoW witness count mismatch: expected {expected}, got {got}")]
    CommitPowWitnessCount {
        /// Witness count the folding schedule fixes.
        expected: usize,
        /// Witness count the proof carries.
        got: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::str::from_utf8;
    #[cfg(panic = "unwind")]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_challenger::fs::TypeTag;
    use p3_challenger::testing::{
        SeedDigest, assert_seeds_pairwise_distinct, pow_difficulties, seed_digest,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_security::grinding::{
        GRINDING_VOCABULARY, GrindingBudget, GrindingSite, RecordedGrind, ZeroBitConvention,
        grinding_step,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xF21);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A shape with the given schedule and no grinding.
    fn shape_with(log_arities: Vec<usize>) -> FriShape {
        FriShape {
            log_arities,
            final_poly_len: 1,
            commit_pow_bits: 0,
            query_pow_bits: 0,
            num_queries: 2,
            index_bits: 8,
            log_blowup: 1,
            max_log_arity: 3,
        }
    }

    /// The shape every mutation below is measured against.
    ///
    /// Three rounds, so a per-round step is described more than once.
    fn plain_shape() -> FriShape {
        shape_with(vec![3, 3, 2])
    }

    /// The digest of the byte stream a shape seeds its sponge with.
    ///
    /// Comparing seed streams, rather than a sampled challenge, keeps the sponge out of it.
    fn seed_of(shape: &FriShape) -> SeedDigest {
        seed_digest(&shape.domain_separator::<F, EF>())
    }

    /// Every field of the shape, each moved one step away from `plain_shape`.
    ///
    /// One entry per configuration knob.
    /// A field added to the shape stops the destructuring below from compiling.
    ///
    /// The schedule is a vector, so it contributes one entry per way of moving it.
    fn one_step_from_plain() -> Vec<(&'static str, FriShape)> {
        // Exhaustiveness check: every field named, none elided by a rest pattern.
        // The bindings go unused, since naming the fields is all this has to do.
        let FriShape {
            log_arities: _,
            final_poly_len: _,
            commit_pow_bits: _,
            query_pow_bits: _,
            num_queries: _,
            index_bits: _,
            log_blowup: _,
            max_log_arity: _,
        } = plain_shape();

        let mut mutations = Vec::new();

        // The closing round folds by one more, at an unchanged round count.
        let mut shape = plain_shape();
        shape.log_arities[2] += 1;
        mutations.push(("log_arities value", shape));

        // One more round, which the pattern loop turns into one more group of steps.
        let mut shape = plain_shape();
        shape.log_arities.push(2);
        mutations.push(("log_arities length", shape));

        // A reordering keeps the round count, the total, and the step sequence.
        //
        //     [3, 3, 2]  folds 8 -> 5 -> 2 -> 0
        //     [2, 3, 3]  folds 8 -> 6 -> 3 -> 0
        //
        // Reachable as inputs [10] against [10, 8] at the same cap.
        // Only the instance label separates them, so it must carry the values.
        let mut shape = plain_shape();
        shape.log_arities.reverse();
        mutations.push(("log_arities order", shape));

        // One more coefficient the final polynomial's fixed-length step declares.
        let mut shape = plain_shape();
        shape.final_poly_len += 1;
        mutations.push(("final_poly_len", shape));

        // Elided at zero, so this is the transition where the step appears at all.
        // It appears once per round, which is why the plain shape runs three.
        let mut shape = plain_shape();
        shape.commit_pow_bits += 1;
        mutations.push(("commit_pow_bits", shape));

        // Elided at zero too, so again the transition is the step's presence.
        let mut shape = plain_shape();
        shape.query_pow_bits += 1;
        mutations.push(("query_pow_bits", shape));

        // One more index drawn, which is the fixed length of the query step.
        let mut shape = plain_shape();
        shape.num_queries += 1;
        mutations.push(("num_queries", shape));

        // One more bit per index, which the query step carries in its type tag.
        let mut shape = plain_shape();
        shape.index_bits += 1;
        mutations.push(("index_bits", shape));

        // Blowup and arity cap leave every step exactly where it was.
        // They still change what the protocol is, so the instance label carries them.
        let mut shape = plain_shape();
        shape.log_blowup += 1;
        mutations.push(("log_blowup", shape));

        let mut shape = plain_shape();
        shape.max_log_arity += 1;
        mutations.push(("max_log_arity", shape));

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
    fn a_grinding_step_that_is_already_present_still_binds_its_difficulty() {
        // Both difficulties are elided at zero, so a bump off zero only proves presence.
        //
        //     0 -> 1   the step joins the sequence
        //     1 -> 2   the step that is already there declares one more bit
        //
        // The second transition is the one the step's `Length::Fixed` carries.
        let mut commit_one = plain_shape();
        commit_one.commit_pow_bits = 1;
        let mut commit_two = commit_one.clone();
        commit_two.commit_pow_bits = 2;

        assert_ne!(seed_of(&commit_one), seed_of(&commit_two));

        let mut query_one = plain_shape();
        query_one.query_pow_bits = 1;
        let mut query_two = query_one.clone();
        query_two.query_pow_bits = 2;

        assert_ne!(seed_of(&query_one), seed_of(&query_two));
    }

    #[test]
    fn the_arity_of_every_round_reaches_the_seed_through_the_instance_label() {
        // Invariant: the step sequence does not determine the folding schedule.
        //
        // The schedule therefore has to be bound outside it.
        //
        // Fixture state: two runs of three rounds, folding the same total distance.
        //
        //     [3, 3, 2]  folds 8 -> 5 -> 2 -> 0
        //     [2, 3, 3]  folds 8 -> 6 -> 3 -> 0
        //
        // Both are reachable at one folding cap:
        //
        //     inputs [10]      ->  [3, 3, 2]
        //     inputs [10, 8]   ->  [2, 3, 3]
        //
        // So these are two real protocols, not a tampered field.
        //
        // Every quantity the step sequence carries agrees on the two:
        //
        //     round count   3 == 3   one group of steps per round
        //     index width   8 == 8   derived from the total, which is equal
        //     final poly    1 == 1
        //     grinding      elided on both
        //
        // Only the instance label is left to separate them.
        let forward = plain_shape();
        let mut reversed = plain_shape();
        reversed.log_arities.reverse();

        // The two runs share one sequence of steps.
        assert_eq!(
            forward.pattern::<F, EF>().interactions(),
            reversed.pattern::<F, EF>().interactions(),
            "the step sequence cannot tell the two schedules apart",
        );

        // The seed does separate them.
        //
        // That separation is the instance label doing its work.
        assert_ne!(
            seed_of(&forward),
            seed_of(&reversed),
            "the schedule must reach the seed",
        );
    }

    #[test]
    fn the_query_index_width_is_described_where_the_indices_are_drawn() {
        // A run draws `log_global_max_height + extra_query_index_bits` bits per query.
        // A narrower draw shrinks the query space, and the proximity-test soundness error with it.
        //
        // That width is not a length: the length of the step is how many indices it draws.
        // It reaches the fingerprint through the type tag instead, which is what this pins.
        let shape = plain_shape();
        let pattern = shape.pattern::<F, EF>();

        let drawn: Vec<_> = pattern
            .interactions()
            .iter()
            .filter(|step| step.label() == QUERY_INDICES)
            .collect();

        assert_eq!(drawn.len(), 1, "every index is drawn at one step");
        assert_eq!(
            drawn[0].type_tag(),
            TypeTag::Bits {
                width: shape.index_bits
            },
        );
        assert_eq!(drawn[0].length(), Length::Fixed(shape.num_queries));
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn a_panic_inside_a_live_prover_transcript_unwinds() {
        // Fixture state: a run described with one round, so the transcript is mid-pattern.
        let mut challenger = fresh_challenger();

        // Mutation: the caller panics between the first round and `finish`.
        //
        // `RowMajorMatrix::new`, the `.pop().unwrap()` on the committed matrices,
        // and any caller-supplied `Mmcs` or `FriFoldingStrategy` callback all sit here.
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut transcript =
                ProverTranscript::<Ch, F, EF>::new(&mut challenger, shape_with(vec![1]));
            let _beta = transcript.commit_round([F::ONE; 8]);
            panic!("folding strategy panicked");
        }));

        // The completeness check yields, so the caller's panic is what escapes.
        let payload = caught.expect_err("the caller's panic must unwind out of the scope");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("folding strategy panicked")
        );
    }

    #[test]
    fn a_commit_round_missing_its_grinding_witness_is_rejected() {
        // Described run: one round guarded by 4 bits of grinding.
        //
        // A described grinding step cannot be replayed with no witness to feed it.
        let mut shape = shape_with(vec![1]);
        shape.commit_pow_bits = 4;

        let mut challenger = fresh_challenger();
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);

        let err = transcript
            .commit_round([F::ONE; 8], None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(
            err,
            TranscriptFailure::MissingPowWitness(PowPhase::CommitPhase)
        );
    }

    #[test]
    fn a_run_carrying_the_wrong_number_of_commit_rounds_is_rejected() {
        // Invariant: the round count is a length of the description.
        //
        // It is never a length of the proof.
        //
        // Fixture state: a schedule of [3, 3, 2], so exactly 3 commit rounds.
        //
        // Mutation: hand the replay 2 commitments, then 4.
        //
        //     described:  [round_0, round_1, round_2]
        //     short:      [round_0, round_1]          -> 2 != 3
        //     long:       [round_0, ..., round_3]     -> 4 != 3
        //
        // A short run would leave a described step unplayed.
        //
        // A long run would drive the player off the end of the description.
        for got in [2, 4] {
            let mut challenger = fresh_challenger();
            let mut transcript =
                VerifierTranscript::<Ch, F, EF>::new(&mut challenger, plain_shape());

            // One commitment and one witness per supplied round.
            //
            // Only the count differs from the description.
            let commitments = vec![[F::ONE; 8]; got];
            let witnesses = vec![F::ZERO; got];

            let err = transcript
                .commit_rounds(&commitments, &witnesses)
                .expect_err("a round count outside the description must error");

            assert_eq!(
                err,
                TranscriptFailure::CommitRoundCount { expected: 3, got }
            );
        }
    }

    #[test]
    fn a_run_carrying_the_wrong_number_of_grinding_witnesses_is_rejected() {
        // Invariant: every commit round replays its own grinding step.
        //
        // Fixture state: a schedule of [3, 3, 2], so 3 rounds and 3 witnesses.
        //
        // Mutation: keep the 3 commitments, supply only 2 witnesses.
        //
        //     commitments: [c_0, c_1, c_2]   ->  3 == 3, accepted
        //     witnesses:   [w_0, w_1]        ->  2 != 3, rejected
        //
        // Zipping the two lists instead would silently drop the last round.
        let mut challenger = fresh_challenger();
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, plain_shape());

        let err = transcript
            .commit_rounds(&[[F::ONE; 8]; 3], &[F::ZERO; 2])
            .expect_err("a witness count outside the description must error");

        assert_eq!(
            err,
            TranscriptFailure::CommitPowWitnessCount {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn a_wrong_round_count_is_an_error_and_not_a_panic() {
        // Invariant: a verifier rejects malformed input.
        //
        // It never panics on it.
        //
        // The driver panics on a step played past the end of a description.
        //
        // A count taken from a proof must never reach that path.
        //
        // Fixture state: 3 described rounds, 50 supplied.
        //
        //     described steps:  3 rounds * 3 steps, plus 3 closing steps
        //     4th commitment:   no step left to match it
        //
        // Reaching the assertion at all proves no panic was raised.
        let mut challenger = fresh_challenger();
        let mut transcript = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, plain_shape());

        let err = transcript
            .commit_rounds(&[[F::ONE; 8]; 50], &[F::ZERO; 50])
            .expect_err("a wildly wrong round count must still be a structured error");

        assert!(matches!(
            err,
            TranscriptFailure::CommitRoundCount {
                expected: 3,
                got: 50
            }
        ));
    }

    #[test]
    fn a_final_polynomial_of_the_wrong_length_is_rejected() {
        // Described run: one round, then a final polynomial of exactly 1 coefficient.
        //
        //     described:   1
        //     proof holds: 3   -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript =
            VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape_with(vec![1]));

        let _beta = transcript
            .commit_round([F::ONE; 8], None)
            .expect("a round without grinding replays from the commitment alone");

        let err = transcript
            .query_phase(&[EF::ONE; 3], None)
            .expect_err("a final polynomial outside the described length must error");

        assert_eq!(
            err,
            TranscriptFailure::FinalPolyLen {
                expected: 1,
                got: 3
            }
        );
    }
    /// This protocol's name, as the vocabulary table keys it.
    fn protocol() -> &'static str {
        from_utf8(NAME).expect("the protocol name is ASCII")
    }

    #[test]
    fn the_grinding_vocabulary_maps_both_grinds_this_protocol_describes() {
        // The security model keys its table on the name and the labels bound here.
        let commit = grinding_step(protocol(), COMMIT_POW).expect("the folding grind is mapped");
        assert_eq!(commit.site, GrindingSite::LdtCommitPhase);
        assert_eq!(commit.zero_bits, ZeroBitConvention::Elided);

        let query = grinding_step(protocol(), QUERY_POW).expect("the query grind is mapped");
        assert_eq!(query.site, GrindingSite::LdtQueryPhase);
        assert_eq!(query.zero_bits, ZeroBitConvention::Elided);

        // Two grinds described, so two rows: a third would credit bits nothing pays.
        assert_eq!(
            GRINDING_VOCABULARY
                .iter()
                .filter(|step| step.protocol == protocol())
                .count(),
            2,
        );
    }

    #[test]
    fn every_described_grind_carries_the_difficulty_the_regime_credits() {
        // Invariant: `security_regime` and the pattern read one number twice.
        //
        //     FriParameters  --security_regime-->  FriRegime      --> credited bits
        //                    --FriShape::pattern-->  Kind::Pow    --> recorded bits
        //
        // Three rounds, so the commit-phase grind is described three times.
        for commit_proof_of_work_bits in [0, 1, 12] {
            for query_proof_of_work_bits in [0, 1, 16] {
                let params = FriParameters {
                    log_blowup: 1,
                    log_final_poly_len: 0,
                    max_log_arity: 3,
                    num_queries: 64,
                    batch_proof_of_work_bits: 0,
                    commit_proof_of_work_bits,
                    query_proof_of_work_bits,
                    mmcs: (),
                };
                let shape = FriShape::with_schedule(&params, vec![3, 3, 2], 8);

                let recorded: Vec<_> = pow_difficulties(&shape.pattern::<F, EF>())
                    .into_iter()
                    .map(|(label, bits)| RecordedGrind::new(protocol(), label, bits))
                    .collect();

                GrindingBudget::NONE
                    .with_fri(&params.security_regime())
                    .check(&[protocol()], &recorded)
                    .unwrap_or_else(|mismatch| panic!("{mismatch}"));
            }
        }
    }
}
