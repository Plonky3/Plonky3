//! Fiat-Shamir transcript of the HVZK-WHIR pipeline.
//!
//! # Overview
//!
//! One statement of what a hiding WHIR run absorbs and draws.
//!
//! Both sides play it.
//!
//! It is built from the derived hiding configuration alone.
//!
//! Neither side ever reads a count out of a proof.
//!
//! # Shape
//!
//! ```text
//!     initial batching       one extension element, drawn
//!     Begin initial fold     the delegated masked sumcheck
//!     End   initial fold
//!     per round:  commitment     new oracle, then its code-switch mask
//!                 out-of-domain  one point drawn, one answer sent, per sample
//!                 grinding       only when the difficulty is positive
//!                 queries        num_queries draws of index_bits bits
//!                 batching       one extension element, drawn
//!                 Begin fold     the round's delegated masked sumcheck
//!                 End   fold
//!     Begin base case        the delegated masked base case
//!     End   base case
//! ```
//!
//! # Delegation
//!
//! Two phases of a hiding run are protocols of their own.
//!
//! ```text
//!     masked sumcheck  ->  its own seed, its own driver, one bracket here
//!     masked base case ->  its own seed, its own driver, one bracket here
//! ```
//!
//! A bracket states that a delegation happens.
//!
//! It also states where in the run it happens.
//!
//! The counts inside it reach this seed through the instance label.
//!
//! # What is bound
//!
//! - Shape: every round, every grinding difficulty, every query width, every draw count.
//! - Shape: where each delegated phase runs, through its bracket.
//! - Instance label: the plain WHIR parameters, unchanged.
//! - Instance label: the mask rate, the mask geometry, the randomness budgets.
//! - Instance label: the rounds and difficulty of every masked sumcheck batch.
//! - Instance label: every number the delegated base case runs against.
//!
//! # What is not bound
//!
//! The width of a commitment digest.
//!
//! This layer cannot see it.
//!
//! A commitment is absorbed opaquely, through the challenger's own encoding.
//!
//! A wrong digest width therefore does not part the two sponges.
//!
//! The Merkle opening checks compare it against explicit dimensions instead.
//!
//! # Soundness
//!
//! The seed is absorbed where the hiding run starts.
//!
//! It lands before the run's first challenge.
//!
//! Everything the caller bound earlier stays in the sponge.
//!
//! That earlier binding keeps its effect.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, Hierarchy, Interaction, InteractionPattern, Kind, Length,
    ProverState, TranscriptBound, TranscriptField, Unit, VerifierState,
};
use p3_challenger::{
    CanObserve, CanSample, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_field::{ExtensionField, TwoAdicField};
use p3_util::log2_strict_usize;

use super::{
    Alphabet, INITIAL_BATCHING, INITIAL_FOLD, OOD_ANSWER, OOD_POINT, QUERY_INDICES, QUERY_POW,
    ROUND_BATCHING, ROUND_FOLD, Sumcheck, TranscriptFailure, bind_folding_factor, push_delegation,
    push_pow, push_query_indices, push_u64, query_draws,
};
use crate::parameters::{FoldingFactor, SecurityAssumption};
use crate::pcs::zk::base_case::BaseCaseZkConfig;
use crate::pcs::zk::{BaseCaseZkError, MaskCodeShape, MaskGroupShape, ZkWhirConfig};

/// Version byte bound into the hiding run's transcript seed.
///
/// The byte parts the seeds of two incompatible descriptions of one run.
///
/// Version 2 records each delegated phase as one bracket.
const VERSION: u8 = 2;

/// Protocol name bound into the hiding run's transcript seed.
///
/// It is distinct from the plain pipeline's name.
///
/// The two pipelines can therefore never share a seed.
const NAME: &[u8] = b"p3-whir-hvzk";

/// Version byte bound into the masked base case's own transcript seed.
const BASE_VERSION: u8 = 1;

/// Protocol name bound into the masked base case's own transcript seed.
///
/// The base case runs under a seed of its own.
///
/// It therefore carries a name of its own.
const BASE_NAME: &[u8] = b"p3-whir-hvzk-base";

/// Step label of the oracle committed by a code-switching round.
const ORACLE_COMMITMENT: &str = "oracle_commitment";

/// Step label of the mask committed alongside it.
const SWITCH_MASK_COMMITMENT: &str = "switch_mask_commitment";

/// Container label of the masked base case that closes a hiding run.
const BASE_CASE: &str = "base_case";

/// Step label of the fresh source mask of the base case.
const BASE_FRESH_COMMITMENT: &str = "base_fresh_commitment";

/// Step label of one group of fresh blinds of the base case.
const BASE_BLIND_COMMITMENT: &str = "base_blind_commitment";

/// Step label of the fresh-side claim the base case sends.
const BASE_CLAIM: &str = "base_claim";

/// Step label of the blinding challenge the base case draws.
const BASE_GAMMA: &str = "base_gamma";

/// Step label of a one-time-pad reveal of a message word.
const BASE_REVEAL_MESSAGE: &str = "base_reveal_message";

/// Step label of a one-time-pad reveal of an encoding-randomness word.
const BASE_REVEAL_RANDOMNESS: &str = "base_reveal_randomness";

/// Step label of the grinding step guarding the base-case spot checks.
const BASE_POW: &str = "base_pow";

/// Step label of the source spot-check positions.
const BASE_SOURCE_QUERIES: &str = "base_source_queries";

/// Step label of one group's mask spot-check positions.
const BASE_MASK_QUERIES: &str = "base_mask_queries";

/// Type naming the delegated masked base case at the type level.
///
/// The name is compared locally where a closer meets its opener.
///
/// It never reaches the pattern fingerprint.
struct BaseCase;

/// Bind one mask code's three lengths, each as its own chunk.
///
/// Separate chunks keep two codes with the same total apart.
fn bind_mask_code<U: Unit>(separator: &mut DomainSeparator<U>, code: &MaskCodeShape) {
    // How many secret coefficients the code carries.
    push_u64(separator, code.message_len);
    // How many uniform coefficients pad them.
    push_u64(separator, code.randomness_len);
    // How long the codeword those two encode into is.
    push_u64(separator, code.domain_size);
}

/// Numbers that fix one masked sumcheck batch.
///
/// The batch runs under its own seed, played by its own driver.
///
/// Its numbers therefore reach the surrounding seed through the instance label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkSumcheckShape {
    /// Number of rounds this batch runs.
    pub rounds: usize,
    /// Grinding difficulty inside each round.
    pub pow_bits: usize,
}

impl ZkSumcheckShape {
    /// Append this batch's numbers to an instance label under construction.
    fn bind<U: Unit>(&self, separator: &mut DomainSeparator<U>) {
        // How many variables the batch reduces away.
        push_u64(separator, self.rounds);
        // How much work each of those rounds charges.
        push_u64(separator, self.pow_bits);
    }
}

/// Numbers that fix one HVZK code-switching round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkWhirRoundShape {
    /// Out-of-domain samples drawn against the new oracle.
    pub ood_samples: usize,
    /// Grinding difficulty guarding this round's query indices.
    pub query_pow_bits: usize,
    /// Number of query indices actually drawn.
    pub query_draws: usize,
    /// Bit width of each query index.
    pub index_bits: usize,
    /// Masked sumcheck batch folding the new oracle.
    pub sumcheck: ZkSumcheckShape,
    /// Log-inverse rate of the codeword committed by this round.
    pub log_inv_rate: usize,
    /// Code of the mask committed alongside the new oracle.
    pub switch_mask: MaskCodeShape,
}

impl ZkWhirRoundShape {
    /// Append this round's steps to a step sequence under construction.
    fn extend<F, EF>(&self, steps: &mut Vec<Interaction>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The folded message and the mask hiding it are committed together.
        for label in [ORACLE_COMMITMENT, SWITCH_MASK_COMMITMENT] {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                label,
                Length::Scalar,
            ));
        }

        // Point and answer alternate: each answer is bound before the next draw.
        for _ in 0..self.ood_samples {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OOD_POINT,
                Length::Scalar,
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                OOD_ANSWER,
                Length::Scalar,
            ));
        }

        // Grinding raises the cost of searching for favourable query indices.
        push_pow::<F>(steps, QUERY_POW, self.query_pow_bits);
        push_query_indices(steps, QUERY_INDICES, self.index_bits, self.query_draws);

        // One challenge weights this round's fresh constraints against the carried claim.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ROUND_BATCHING,
            Length::Scalar,
        ));

        push_delegation::<Sumcheck>(steps, ROUND_FOLD);
    }

    /// Number of steps this round contributes.
    const fn step_count(&self) -> usize {
        // Two commitments, one batching challenge, two bracket markers.
        5 + 2 * self.ood_samples
            + if self.query_pow_bits > 0 { 1 } else { 0 }
            + if self.query_draws > 0 { 1 } else { 0 }
    }
}

/// Numbers that fix the transcript of one masked base case.
///
/// Both sides build it from the base-case configuration they already share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkBaseCaseShape {
    /// Message length of the randomized terminal source code.
    pub source_message_len: usize,
    /// Encoding-randomness length of that code.
    pub source_randomness_len: usize,
    /// Codeword length of that code.
    pub source_domain_size: usize,
    /// Spot checks the configuration asks for against the source.
    pub source_queries: usize,
    /// Spot checks the configuration asks for against each mask group.
    pub mask_queries: usize,
    /// Grinding difficulty guarding every spot check.
    pub pow_bits: usize,
    /// Every committed mask group, in commitment order.
    pub groups: Vec<MaskGroupShape>,
}

impl ZkBaseCaseShape {
    /// Derive the shape of one masked base case from its configuration.
    ///
    /// # Arguments
    ///
    /// - `config`: the base-case shape the two sides agreed on.
    #[must_use]
    pub fn new<F: TwoAdicField>(config: &BaseCaseZkConfig<F>) -> Self {
        // Every number below is read straight off the shared configuration.
        //
        // Re-deriving any of them here would leave two copies free to drift apart.
        Self {
            source_message_len: config.code.message_len,
            source_randomness_len: config.code.randomness_len,
            source_domain_size: config.code.domain_size,
            source_queries: config.num_queries,
            mask_queries: config.mask_queries,
            pow_bits: config.pow_bits,
            groups: config.mask_groups.clone(),
        }
    }

    /// Bit width of one source spot-check position.
    #[must_use]
    pub const fn source_index_bits(&self) -> usize {
        log2_strict_usize(self.source_domain_size)
    }

    /// Number of source spot-check positions actually drawn.
    ///
    /// A count of zero means every position opens instead.
    ///
    /// Nothing is then drawn at all.
    #[must_use]
    pub const fn source_query_draws(&self) -> usize {
        query_draws(self.source_domain_size, self.source_queries)
    }

    /// Bit width and draw count of one group's spot-check positions.
    ///
    /// # Panics
    ///
    /// When no group sits at that index.
    #[must_use]
    pub fn mask_query_site(&self, group: usize) -> (usize, usize) {
        // Every member of a group shares the group's code.
        let shape = self.groups[group].shape;
        (
            // A position addresses one row of that code's codeword.
            log2_strict_usize(shape.domain_size),
            // Asking for at least as many positions as rows opens them all.
            query_draws(shape.domain_size, self.mask_queries),
        )
    }

    /// Total number of masks the groups tile.
    #[must_use]
    pub fn num_masks(&self) -> usize {
        self.groups.iter().map(|group| group.width).sum()
    }

    /// Message and randomness lengths of one reveal, in reveal order.
    ///
    /// ```text
    ///     position 0        the source word
    ///     position 1 ..     every group member, group by group
    /// ```
    ///
    /// # Returns
    ///
    /// `None` once the position runs past the last described reveal.
    #[must_use]
    pub fn reveal_lengths(&self, position: usize) -> Option<(usize, usize)> {
        // The source word opens the reveal sequence.
        if position == 0 {
            return Some((self.source_message_len, self.source_randomness_len));
        }
        // The rest walk the groups, each contributing one reveal per member.
        let mut remaining = position - 1;
        for group in &self.groups {
            if remaining < group.width {
                return Some((group.shape.message_len, group.shape.randomness_len));
            }
            remaining -= group.width;
        }
        None
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A flat sequence of leaf steps always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut steps = Vec::new();

        // One fresh source mask, then one fresh blind group per carried group.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            BASE_FRESH_COMMITMENT,
            Length::Scalar,
        ));
        for _ in &self.groups {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                BASE_BLIND_COMMITMENT,
                Length::Scalar,
            ));
        }

        // The fresh-side claim is fixed before the blinding challenge exists.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            BASE_CLAIM,
            Length::Scalar,
        ));
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            BASE_GAMMA,
            Length::Scalar,
        ));

        // Every committed word is revealed as a message and a randomness half.
        let mut reveal = |message_len: usize, randomness_len: usize| {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                BASE_REVEAL_MESSAGE,
                Length::Fixed(message_len),
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                BASE_REVEAL_RANDOMNESS,
                Length::Fixed(randomness_len),
            ));
        };
        reveal(self.source_message_len, self.source_randomness_len);
        for group in &self.groups {
            for _ in 0..group.width {
                reveal(group.shape.message_len, group.shape.randomness_len);
            }
        }

        // Grinding raises the cost of searching for favourable spot positions.
        push_pow::<F>(&mut steps, BASE_POW, self.pow_bits);
        push_query_indices(
            &mut steps,
            BASE_SOURCE_QUERIES,
            self.source_index_bits(),
            self.source_query_draws(),
        );

        // Positions are shared inside a group.
        //
        // One step therefore covers the whole group.
        for group in 0..self.groups.len() {
            let (bits, draws) = self.mask_query_site(group);
            push_query_indices(&mut steps, BASE_MASK_QUERIES, bits, draws);
        }

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
    }

    /// Bind the base case's identity, this shape, and the remaining parameters.
    ///
    /// # Soundness
    ///
    /// A requested spot-check count reaches the shape only after a clamp.
    ///
    /// ```text
    ///     domain 16, 16 asked  ->  every position opens, nothing is drawn
    ///     domain 16, 99 asked  ->  the same step sequence
    /// ```
    ///
    /// The clamp is not injective.
    ///
    /// The label therefore carries the raw counts too.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The fingerprint of the step sequence lands inside the identifier.
        let mut separator = DomainSeparator::new(BASE_VERSION, BASE_NAME, self.pattern::<F, EF>());
        // Everything the step sequence does not already pin follows it.
        self.bind(&mut separator);
        separator
    }

    /// Append every number of this base case to an instance label under construction.
    ///
    /// The surrounding run appends the same numbers to its own label.
    ///
    /// One change to this list therefore moves both seeds.
    fn bind<U: Unit>(&self, separator: &mut DomainSeparator<U>) {
        // The randomized terminal source code, by its three lengths.
        push_u64(separator, self.source_message_len);
        push_u64(separator, self.source_randomness_len);
        push_u64(separator, self.source_domain_size);

        // How many spot checks each side was asked for, before any clamp.
        push_u64(separator, self.source_queries);
        push_u64(separator, self.mask_queries);

        // How much work the spot checks are guarded by.
        push_u64(separator, self.pow_bits);

        // Every committed mask group, by its code and its width.
        push_u64(separator, self.groups.len());
        for group in &self.groups {
            bind_mask_code(separator, &group.shape);
            push_u64(separator, group.width);
        }
    }
}

/// Numbers that fix the transcript of one HVZK-WHIR run.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug)]
pub struct ZkWhirShape {
    /// Variable count of the committed multilinear polynomial.
    pub num_variables: usize,
    /// Plain-pipeline numbers the hiding run never replays.
    ///
    /// ```text
    ///     commitment out-of-domain samples   the hiding commit phase draws none
    ///     final sumcheck rounds              the masked base case replaces them
    ///     final folding grinding             same
    ///     plain terminal queries and pow     the base case re-derives its own
    /// ```
    ///
    /// The label carries them so no configuration difference is invisible.
    pub unreplayed_plain: [usize; 5],
    /// Masked sumcheck batch folding the polynomial before the first round.
    pub initial_sumcheck: ZkSumcheckShape,
    /// One entry per code-switching round, in round order.
    pub rounds: Vec<ZkWhirRoundShape>,
    /// The masked base case closing the run.
    pub base_case: ZkBaseCaseShape,
    /// Security level the parameters were derived against, in bits.
    pub security_level: usize,
    /// Grinding budget the derived difficulties were capped by.
    pub pow_budget: usize,
    /// Log-inverse rate of the first committed codeword.
    pub starting_log_inv_rate: usize,
    /// Proximity-gap assumption the query counts were derived under.
    pub soundness_type: SecurityAssumption,
    /// Folding strategy the schedule was derived from.
    pub folding_factor: FoldingFactor,
    /// Mask-code message length of the HVZK sumcheck.
    pub ell_zk: usize,
    /// Log-inverse rate of every mask codeword.
    pub mask_log_inv_rate: usize,
    /// Encoding-randomness budget of every committed oracle.
    pub oracle_randomness: Vec<usize>,
    /// Code shared by every HVZK sumcheck mask.
    pub sumcheck_mask: MaskCodeShape,
}

impl ZkWhirShape {
    /// Derive the shape of one HVZK-WHIR run from its configuration.
    ///
    /// # Arguments
    ///
    /// - `config`: the derived HVZK protocol configuration.
    #[must_use]
    pub fn new<EF, F, Challenger>(config: &ZkWhirConfig<EF, F, Challenger>) -> Self
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Each round queries its own domain, folded by that round's arity.
        let rounds = config
            .round_parameters
            .iter()
            .enumerate()
            .map(|(index, params)| {
                let folded = params.domain_size >> config.round_folding_factor(index);
                ZkWhirRoundShape {
                    ood_samples: params.ood_samples,
                    query_pow_bits: params.pow_bits,
                    query_draws: query_draws(folded, params.num_queries),
                    index_bits: log2_strict_usize(folded),
                    sumcheck: ZkSumcheckShape {
                        rounds: config.round_folding_factor(index + 1),
                        pow_bits: params.folding_pow_bits,
                    },
                    log_inv_rate: params.log_inv_rate,
                    switch_mask: config.switch_masks[index],
                }
            })
            .collect();

        // The base case is described from the very object the two sides run it against.
        //
        // Deriving it a second time here would leave two numbers free to drift apart.
        let base_case = ZkBaseCaseShape::new(&config.base_case_config());

        Self {
            num_variables: config.num_variables,
            unreplayed_plain: [
                config.commitment_ood_samples,
                config.inner.final_sumcheck_rounds,
                config.inner.final_folding_pow_bits,
                config.inner.final_queries,
                config.inner.final_pow_bits,
            ],
            initial_sumcheck: ZkSumcheckShape {
                rounds: config.round_folding_factor(0),
                pow_bits: config.starting_folding_pow_bits,
            },
            rounds,
            base_case,
            security_level: config.security_level,
            pow_budget: config.pow_bits,
            starting_log_inv_rate: config.starting_log_inv_rate,
            soundness_type: config.soundness_type,
            folding_factor: config.folding_factor.clone(),
            ell_zk: config.zk.ell_zk,
            mask_log_inv_rate: config.zk.mask_log_inv_rate,
            oracle_randomness: config.oracle_randomness.clone(),
            sumcheck_mask: config.sumcheck_mask,
        }
    }

    /// Number of code-switching rounds this shape runs.
    #[must_use]
    pub const fn n_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// Every bracket opened below is closed one step later, in the same call.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Opening challenge, initial bracket, every round, base-case bracket.
        let capacity = 3
            + self
                .rounds
                .iter()
                .map(ZkWhirRoundShape::step_count)
                .sum::<usize>()
            + 2;
        let mut steps = Vec::with_capacity(capacity);

        // One challenge weights the incoming evaluation claims into a single sum.
        //
        // The hiding pipeline draws it itself.
        //
        // It lands ahead of the batch handed to the delegate.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            INITIAL_BATCHING,
            Length::Scalar,
        ));
        push_delegation::<Sumcheck>(&mut steps, INITIAL_FOLD);

        for round in &self.rounds {
            round.extend::<F, EF>(&mut steps);
        }

        push_delegation::<BaseCase>(&mut steps, BASE_CASE);

        InteractionPattern::new(steps).expect("every bracket opened here is closed here")
    }

    /// Bind the protocol identity, this shape, and the remaining parameters.
    ///
    /// A parameter that changes the step sequence is covered by the fingerprint.
    ///
    /// The rest go in the instance label.
    ///
    /// # Soundness
    ///
    /// The mask rate sets the mask code's distance.
    ///
    /// Distance is what makes a spot check bind.
    ///
    /// The rate reaches the shape only through the mask domain sizes.
    ///
    /// The label therefore carries it directly.
    ///
    /// A delegated phase contributes one bracket whatever its length.
    ///
    /// Its own numbers therefore travel in the label too.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The fingerprint of the step sequence lands inside the identifier.
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The plain WHIR statement, bound exactly as the plain pipeline binds it.
        push_u64(&mut separator, self.num_variables);
        for value in self.unreplayed_plain {
            push_u64(&mut separator, value);
        }
        push_u64(&mut separator, self.security_level);
        push_u64(&mut separator, self.pow_budget);
        push_u64(&mut separator, self.starting_log_inv_rate);
        push_u64(&mut separator, self.soundness_type as usize);
        bind_folding_factor(&mut separator, &self.folding_factor);

        // Every delegated masked sumcheck batch, in the order the run plays them.
        self.initial_sumcheck.bind(&mut separator);
        push_u64(&mut separator, self.rounds.len());
        for round in &self.rounds {
            push_u64(&mut separator, round.log_inv_rate);
            round.sumcheck.bind(&mut separator);
        }

        // The hiding overlay: mask geometry and every randomness budget.
        push_u64(&mut separator, self.ell_zk);
        push_u64(&mut separator, self.mask_log_inv_rate);
        bind_mask_code(&mut separator, &self.sumcheck_mask);
        for round in &self.rounds {
            bind_mask_code(&mut separator, &round.switch_mask);
        }
        push_u64(&mut separator, self.oracle_randomness.len());
        for &budget in &self.oracle_randomness {
            push_u64(&mut separator, budget);
        }

        // The delegated base case runs under its own seed.
        //
        // Its numbers therefore land here.
        self.base_case.bind(&mut separator);

        separator
    }

    /// Grinding difficulty of the query site of one round.
    ///
    /// # Panics
    ///
    /// When no round sits at that index.
    fn query_pow_bits(&self, round: usize) -> usize {
        self.rounds[round].query_pow_bits
    }

    /// Index width and draw count of the query site of one round.
    ///
    /// A draw count of zero means the round opens every position instead.
    ///
    /// Nothing is then drawn at all.
    ///
    /// # Panics
    ///
    /// When no round sits at that index.
    fn query_index_site(&self, round: usize) -> (usize, usize) {
        let shape = &self.rounds[round];
        (shape.index_bits, shape.query_draws)
    }
}

/// Prover-side transcript of one HVZK-WHIR run.
///
/// Holds the only definition of what a hiding prover plays at each phase.
///
/// The challenger is borrowed, not consumed.
///
/// A hiding run sits inside a larger protocol.
///
/// That protocol's own transcript continues where this one stops.
pub struct ZkWhirProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: ZkWhirShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZkWhirProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this run.
    /// - `shape`: the numbers that fix this run's transcript.
    pub fn new(challenger: &'a mut C, shape: ZkWhirShape) -> Self {
        // Seeding folds the shape fingerprint into the sponge before any step.
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this run was described with.
    pub const fn shape(&self) -> &ZkWhirShape {
        &self.shape
    }

    /// Draw the challenge weighting the incoming evaluation claims.
    pub fn initial_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(INITIAL_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the masked sumcheck that opens the run.
    pub fn delegate_initial_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate::<Sumcheck, R>(INITIAL_FOLD, run)
    }

    /// Lend the sponge to the masked sumcheck that closes one code-switching round.
    pub fn delegate_round_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate::<Sumcheck, R>(ROUND_FOLD, run)
    }

    /// Lend the sponge to the masked base case that closes the run.
    pub fn delegate_base_case<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate::<BaseCase, R>(BASE_CASE, run)
    }

    /// Bind the oracle one code-switching round commits.
    pub fn oracle_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(ORACLE_COMMITMENT, commitment);
    }

    /// Bind the code-switch mask committed alongside that oracle.
    pub fn switch_mask_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(SWITCH_MASK_COMMITMENT, commitment);
    }

    /// Draw one out-of-domain evaluation point.
    pub fn ood_point(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OOD_POINT)
            .into_inner()
    }

    /// Bind the private answer at one out-of-domain point.
    ///
    /// Every answer is bound before the next point is drawn.
    pub fn ood_answer(&mut self, answer: EF) {
        let _bound = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(OOD_ANSWER, &answer);
    }

    /// Grind the query site of one round.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn query_pow(&mut self, round: usize) -> F {
        let bits = self.shape.query_pow_bits(round);
        if bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(QUERY_POW, bits)
    }

    /// Draw the query indices of one round.
    ///
    /// # Returns
    ///
    /// Every index in draw order, repeats included.
    ///
    /// A saturated round opens every position instead.
    pub fn query_indices(&mut self, round: usize) -> Vec<usize> {
        let (width, draws) = self.shape.query_index_site(round);
        // A saturated round has nothing left to decide.
        //
        // No draw is described there.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(QUERY_INDICES, width, draws)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Draw the challenge weighting one round's fresh constraints.
    pub fn round_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_BATCHING)
            .into_inner()
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "a hiding WHIR run carries every value in its own proof",
        );
    }

    /// Bracket one delegated phase and hand it the borrowed sponge.
    fn delegate<T: ?Sized, R>(&mut self, label: &'static str, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<T>(label);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<T>(label);
        output
    }
}

/// Verifier-side transcript of one HVZK-WHIR run.
///
/// Mirrors the prover side call for call, over the same description.
///
/// Every value comes from the proof rather than from a wire.
pub struct ZkWhirVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value.
    ///
    /// The driver therefore reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: ZkWhirShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZkWhirVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this run.
    /// - `shape`: the numbers that fix this run's transcript.
    pub fn new(challenger: &'a mut C, shape: ZkWhirShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this run was described with.
    pub const fn shape(&self) -> &ZkWhirShape {
        &self.shape
    }

    /// Redraw the challenge weighting the incoming evaluation claims.
    pub fn initial_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(INITIAL_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the masked sumcheck that opens the run.
    pub fn delegate_initial_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate::<Sumcheck, R>(INITIAL_FOLD, run)
    }

    /// Lend the sponge to the masked sumcheck that closes one code-switching round.
    pub fn delegate_round_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate::<Sumcheck, R>(ROUND_FOLD, run)
    }

    /// Lend the sponge to the masked base case that closes the run.
    pub fn delegate_base_case<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate::<BaseCase, R>(BASE_CASE, run)
    }

    /// Bind the oracle one code-switching round commits.
    pub fn oracle_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(ORACLE_COMMITMENT, commitment);
    }

    /// Bind the code-switch mask committed alongside that oracle.
    pub fn switch_mask_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state
            .observe_opaque(SWITCH_MASK_COMMITMENT, commitment);
    }

    /// Redraw one out-of-domain evaluation point.
    pub fn ood_point(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OOD_POINT)
            .into_inner()
    }

    /// Bind the private answer the proof carries for one out-of-domain point.
    pub fn ood_answer(&mut self, answer: EF) {
        let _bound = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(OOD_ANSWER, &answer);
    }

    /// Replay the grind of the query site of one round.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn query_pow(&mut self, round: usize, witness: F) -> Result<(), TranscriptFailure> {
        let bits = self.shape.query_pow_bits(round);
        if bits == 0 {
            return Ok(());
        }
        // A failed check poisons the driver.
        //
        // The rejection therefore travels alone.
        self.state
            .observe_pow(QUERY_POW, bits, witness)
            .map_err(|_| TranscriptFailure::PowWitness { round, bits })
    }

    /// Redraw the query indices of one round.
    pub fn query_indices(&mut self, round: usize) -> Vec<usize> {
        let (width, draws) = self.shape.query_index_site(round);
        // A saturated round has nothing left to decide.
        //
        // No draw is described there.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(QUERY_INDICES, width, draws)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Redraw the challenge weighting one round's fresh constraints.
    pub fn round_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ROUND_BATCHING)
            .into_inner()
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// Every path that leaves the transcript early goes through this.
    ///
    /// Dropping an unfinished driver otherwise panics.
    ///
    /// That panic would land on top of an error already on its way out.
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
            .expect("a hiding WHIR run reads an empty wire, so no bytes can remain");
    }

    /// Bracket one delegated phase and hand it the borrowed sponge.
    fn delegate<T: ?Sized, R>(&mut self, label: &'static str, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<T>(label);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<T>(label);
        output
    }
}

/// Prover-side transcript of one masked base case.
///
/// Holds the only definition of what a base-case prover plays at each move.
///
/// The challenger is borrowed, not consumed.
///
/// The surrounding run's transcript continues where this one stops.
pub struct ZkBaseCaseProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this base case was described with.
    shape: ZkBaseCaseShape,
    /// Marker for the extension field the reveals and challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZkBaseCaseProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this base case.
    /// - `shape`: the numbers that fix this base case's transcript.
    pub fn new(challenger: &'a mut C, shape: ZkBaseCaseShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this base case was described with.
    pub const fn shape(&self) -> &ZkBaseCaseShape {
        &self.shape
    }

    /// Bind the fresh mask committed against the source code.
    pub fn fresh_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(BASE_FRESH_COMMITMENT, commitment);
    }

    /// Bind the fresh blinds committed against one carried mask group.
    pub fn blind_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(BASE_BLIND_COMMITMENT, commitment);
    }

    /// Bind the fresh-side claim.
    ///
    /// It is fixed before the challenge that is tested against it.
    pub fn claim(&mut self, claim: EF) {
        let _bound = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(BASE_CLAIM, &claim);
    }

    /// Draw the blinding challenge.
    pub fn gamma(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BASE_GAMMA)
            .into_inner()
    }

    /// Bind one one-time-pad reveal of a committed word.
    ///
    /// # Panics
    ///
    /// When either half is not the length the base case was described with.
    pub fn reveal(&mut self, message: &[EF], randomness: &[EF]) {
        let _message = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(BASE_REVEAL_MESSAGE, message);
        let _randomness = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(BASE_REVEAL_RANDOMNESS, randomness);
    }

    /// Grind the site guarding every spot check.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn spot_check_pow(&mut self) -> F {
        if self.shape.pow_bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(BASE_POW, self.shape.pow_bits)
    }

    /// Draw the source spot-check positions.
    ///
    /// # Returns
    ///
    /// Every position in draw order, repeats included.
    ///
    /// A saturated source opens every position instead.
    pub fn source_queries(&mut self) -> Vec<usize> {
        let draws = self.shape.source_query_draws();
        let width = self.shape.source_index_bits();
        // A saturated source has nothing left to decide.
        //
        // No draw is described there.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(BASE_SOURCE_QUERIES, width, draws)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Draw one group's spot-check positions.
    ///
    /// Every member of the group shares them.
    ///
    /// # Panics
    ///
    /// When no group sits at that index.
    pub fn mask_queries(&mut self, group: usize) -> Vec<usize> {
        let (width, draws) = self.shape.mask_query_site(group);
        // A saturated group has nothing left to decide.
        //
        // No draw is described there.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(BASE_MASK_QUERIES, width, draws)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the base case played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "a masked base case carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one masked base case.
///
/// Mirrors the prover side call for call, over the same description.
///
/// The described lengths are what reject a reveal the proof got wrong.
pub struct ZkBaseCaseVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value.
    ///
    /// The driver therefore reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this base case was described with.
    shape: ZkBaseCaseShape,
    /// Position of the next reveal, used to look up the lengths it must carry.
    reveal: usize,
    /// Marker for the extension field the reveals and challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ZkBaseCaseVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    ///
    /// # Arguments
    ///
    /// - `challenger`: sponge of the surrounding protocol, borrowed for this base case.
    /// - `shape`: the numbers that fix this base case's transcript.
    pub fn new(challenger: &'a mut C, shape: ZkBaseCaseShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            reveal: 0,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this base case was described with.
    pub const fn shape(&self) -> &ZkBaseCaseShape {
        &self.shape
    }

    /// Bind the fresh mask committed against the source code.
    pub fn fresh_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(BASE_FRESH_COMMITMENT, commitment);
    }

    /// Bind the fresh blinds committed against one carried mask group.
    pub fn blind_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(BASE_BLIND_COMMITMENT, commitment);
    }

    /// Bind the fresh-side claim the proof carries.
    pub fn claim(&mut self, claim: EF) {
        let _bound = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(BASE_CLAIM, &claim);
    }

    /// Redraw the blinding challenge.
    pub fn gamma(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(BASE_GAMMA)
            .into_inner()
    }

    /// Bind one one-time-pad reveal the proof carries.
    ///
    /// Reveals arrive in the described order.
    ///
    /// The source word comes first.
    ///
    /// # Errors
    ///
    /// - Either half is not the length the base case was described with.
    /// - The proof carries more reveals than the base case was described with.
    pub fn reveal(&mut self, message: &[EF], randomness: &[EF]) -> Result<(), BaseCaseZkError> {
        // Reveal order is fixed by the description.
        //
        // A counter is therefore enough to name this one.
        let position = self.reveal;
        self.reveal += 1;

        // Past the last described reveal there is no length to compare against.
        let Some((expected_message, expected_randomness)) = self.shape.reveal_lengths(position)
        else {
            return Err(BaseCaseZkError::MaskCountMismatch {
                expected: self.shape.num_masks(),
                actual: position,
            });
        };

        // Position zero is the source word, every later one is a group member.
        let (message_kind, randomness_kind) = if position == 0 {
            ("message", "randomness")
        } else {
            ("mask message", "mask randomness")
        };

        // Both halves are pinned before either one reaches the sponge.
        //
        //     described  (message_len, randomness_len)
        //     supplied   anything else                 -> rejected, nothing absorbed
        if message.len() != expected_message {
            return Err(BaseCaseZkError::BlindedLengthMismatch {
                kind: message_kind,
                expected: expected_message,
                actual: message.len(),
            });
        }
        if randomness.len() != expected_randomness {
            return Err(BaseCaseZkError::BlindedLengthMismatch {
                kind: randomness_kind,
                expected: expected_randomness,
                actual: randomness.len(),
            });
        }

        // Both lengths now match the description.
        //
        // Neither step below can reject.
        let _message = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(BASE_REVEAL_MESSAGE, message);
        let _randomness = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(BASE_REVEAL_RANDOMNESS, randomness);
        Ok(())
    }

    /// Replay the grind of the site guarding every spot check.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn spot_check_pow(&mut self, witness: F) -> Result<(), BaseCaseZkError> {
        if self.shape.pow_bits == 0 {
            return Ok(());
        }
        // A failed check poisons the driver.
        //
        // The rejection therefore travels alone.
        self.state
            .observe_pow(BASE_POW, self.shape.pow_bits, witness)
            .map_err(|_| BaseCaseZkError::InvalidPowWitness)
    }

    /// Redraw the source spot-check positions.
    pub fn source_queries(&mut self) -> Vec<usize> {
        let draws = self.shape.source_query_draws();
        let width = self.shape.source_index_bits();
        // A saturated source has nothing left to decide.
        //
        // No draw is described there.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(BASE_SOURCE_QUERIES, width, draws)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Redraw one group's spot-check positions.
    ///
    /// # Panics
    ///
    /// When no group sits at that index.
    pub fn mask_queries(&mut self, group: usize) -> Vec<usize> {
        let (width, draws) = self.shape.mask_query_site(group);
        // A saturated group has nothing left to decide.
        //
        // No draw is described there.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(BASE_MASK_QUERIES, width, draws)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Release the completeness check because the proof is being rejected.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the base case replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("a masked base case reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    #[cfg(panic = "unwind")]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::fs::TypeTag;
    use p3_challenger::testing::{assert_seeds_pairwise_distinct, pow_difficulties, seed_digest};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::parameters::{ProtocolParameters, WhirConfig};
    use crate::pcs::zk::ZkParameters;
    use crate::transcript::WhirShape;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;
    type Config = ZkWhirConfig<EF, F, Ch>;

    /// Variable count every configuration in this module is derived at.
    const NUM_VARIABLES: usize = 17;

    /// Log-inverse rate of the first committed codeword in every fixture.
    const STARTING_LOG_INV_RATE: usize = 2;

    /// A commitment shaped like the ones a Merkle scheme hands this layer.
    const DIGEST: [F; 8] = [F::ONE; 8];

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// One code rate per code-switching round, growing with the folding schedule.
    fn round_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
        let schedule = folding_factor
            .compute_folding_schedule(num_variables)
            .expect("the fixture schedules are all valid");
        let mut rate = STARTING_LOG_INV_RATE;
        schedule[..schedule.len() - 1]
            .iter()
            .map(|folding| {
                rate += folding - 1;
                rate
            })
            .collect()
    }

    /// Baseline plain parameters every walk below perturbs exactly one field of.
    fn base_params() -> ProtocolParameters {
        params_with(NUM_VARIABLES, FoldingFactor::ConstantFromSecondRound(5, 3))
    }

    /// Baseline plain parameters carrying a caller-chosen folding strategy.
    fn params_with(num_variables: usize, folding_factor: FoldingFactor) -> ProtocolParameters {
        ProtocolParameters {
            security_level: 32,
            pow_bits: 12,
            round_log_inv_rates: round_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: STARTING_LOG_INV_RATE,
        }
    }

    /// Baseline hiding parameters.
    const fn base_zk() -> ZkParameters {
        ZkParameters {
            ell_zk: 4,
            mask_log_inv_rate: 1,
        }
    }

    fn config_from(params: ProtocolParameters, zk: ZkParameters) -> Config {
        ZkWhirConfig::new(NUM_VARIABLES, params, zk).expect("the fixture parameters are all valid")
    }

    fn base_config() -> Config {
        config_from(base_params(), base_zk())
    }

    /// First challenge the seed of a configuration produces on a fresh sponge.
    fn first_challenge(config: &Config) -> F {
        let mut challenger = fresh_challenger();
        ZkWhirShape::new(config)
            .domain_separator::<F, EF>()
            .seed(&mut challenger);
        challenger.sample()
    }

    /// Assert that re-deriving from perturbed user parameters moves the seed.
    fn user_knob_moves_the_seed(
        name: &str,
        perturb: impl FnOnce(&mut ProtocolParameters, &mut ZkParameters),
    ) {
        let mut params = base_params();
        let mut zk = base_zk();
        perturb(&mut params, &mut zk);
        assert_ne!(
            first_challenge(&base_config()),
            first_challenge(&config_from(params, zk)),
            "changing {name} left the seed where it was",
        );
    }

    /// Assert that perturbing one derived field moves the seed.
    fn derived_field_moves_the_seed(name: &str, perturb: impl FnOnce(&mut Config)) {
        let base = base_config();
        let mut tweaked = base.clone();
        perturb(&mut tweaked);
        assert_ne!(
            first_challenge(&base),
            first_challenge(&tweaked),
            "changing {name} left the seed where it was",
        );
    }

    /// Stand-in for a delegated phase: absorbs one value, then draws one.
    ///
    /// The real phase seeds its own driver from the state this one has reached.
    ///
    /// A bracket records only that the delegation happened.
    ///
    /// It also records where in the run it happened.
    fn delegate_stub(challenger: &mut Ch) -> EF {
        challenger.observe_algebra_element(EF::ONE);
        challenger.sample_algebra_element()
    }

    /// Everything one side of a played run produces.
    #[derive(Debug, PartialEq, Eq)]
    struct Played {
        /// Challenges drawn, in draw order.
        challenges: Vec<EF>,
        /// Grinding witnesses, one per grinding site.
        witnesses: Vec<F>,
        /// Query and spot-check positions, one list per site.
        indices: Vec<Vec<usize>>,
    }

    /// Values a described run carries inside its own proof.
    #[derive(Clone, Debug)]
    struct Carried {
        /// One private out-of-domain answer per sample, per round.
        ood_answers: Vec<Vec<EF>>,
        /// The fresh-side claim the base case sends.
        masked_claim: EF,
        /// One reveal pair per committed word, in reveal order.
        reveals: Vec<(Vec<EF>, Vec<EF>)>,
    }

    impl Carried {
        /// Values that fit `shape`, drawn from a single seed.
        fn new(shape: &ZkWhirShape, seed: u64) -> Self {
            let mut rng = SmallRng::seed_from_u64(seed);

            // One answer per out-of-domain sample of each round.
            let ood_answers: Vec<Vec<EF>> = shape
                .rounds
                .iter()
                .map(|round| (0..round.ood_samples).map(|_| rng.random()).collect())
                .collect();

            // Reveal order walks the source word first.
            //
            // Every group member follows, group by group.
            let mut reveals = Vec::new();
            let mut position = 0;
            while let Some((message_len, randomness_len)) = shape.base_case.reveal_lengths(position)
            {
                let message = (0..message_len).map(|_| rng.random()).collect();
                let randomness = (0..randomness_len).map(|_| rng.random()).collect();
                reveals.push((message, randomness));
                position += 1;
            }

            Self {
                ood_answers,
                masked_claim: rng.random(),
                reveals,
            }
        }
    }

    /// Play every described step of the base case, prover side.
    fn play_base_case_prover(
        challenger: &mut Ch,
        shape: &ZkBaseCaseShape,
        carried: &Carried,
        played: &mut Played,
    ) {
        let mut transcript =
            ZkBaseCaseProverTranscript::<Ch, F, EF>::new(challenger, shape.clone());

        // Move 1: the fresh source mask, then one fresh blind group per carried group.
        transcript.fresh_commitment(DIGEST);
        for _ in &shape.groups {
            transcript.blind_commitment(DIGEST);
        }

        // Moves 2 and 3: the fresh-side claim, then the challenge tested against it.
        transcript.claim(carried.masked_claim);
        played.challenges.push(transcript.gamma());

        // Move 4: every one-time-pad reveal, in the described order.
        for (message, randomness) in &carried.reveals {
            transcript.reveal(message, randomness);
        }

        // Move 5: grinding, then the spot-check positions it guards.
        played.witnesses.push(transcript.spot_check_pow());
        played.indices.push(transcript.source_queries());
        for group in 0..shape.groups.len() {
            played.indices.push(transcript.mask_queries(group));
        }

        transcript.finish();
    }

    /// Replay every described step of the base case, verifier side.
    fn play_base_case_verifier(
        challenger: &mut Ch,
        shape: &ZkBaseCaseShape,
        carried: &Carried,
        witness: F,
        played: &mut Played,
    ) {
        let mut transcript =
            ZkBaseCaseVerifierTranscript::<Ch, F, EF>::new(challenger, shape.clone());

        // The same five moves, in the same order, over the prover's own values.
        transcript.fresh_commitment(DIGEST);
        for _ in &shape.groups {
            transcript.blind_commitment(DIGEST);
        }
        transcript.claim(carried.masked_claim);
        played.challenges.push(transcript.gamma());
        for (message, randomness) in &carried.reveals {
            transcript
                .reveal(message, randomness)
                .expect("the described reveal lengths");
        }
        transcript
            .spot_check_pow(witness)
            .expect("the prover's own witness satisfies the site");
        played.indices.push(transcript.source_queries());
        for group in 0..shape.groups.len() {
            played.indices.push(transcript.mask_queries(group));
        }

        transcript.finish();
    }

    /// Play every described step, prover side, in the order the pipeline plays them.
    fn play_prover(challenger: &mut Ch, shape: &ZkWhirShape, carried: &Carried) -> Played {
        let mut played = Played {
            challenges: Vec::new(),
            witnesses: Vec::new(),
            indices: Vec::new(),
        };
        let mut transcript = ZkWhirProverTranscript::<Ch, F, EF>::new(challenger, shape.clone());

        // The run draws its own claim-batching challenge, then hands over the batch.
        played.challenges.push(transcript.initial_batching());
        played
            .challenges
            .push(transcript.delegate_initial_fold(delegate_stub));

        for round in 0..shape.n_rounds() {
            // Two commitments open the round: the new oracle and its code-switch mask.
            transcript.oracle_commitment(DIGEST);
            transcript.switch_mask_commitment(DIGEST);

            // Each private answer is bound before the next point is drawn.
            for &answer in &carried.ood_answers[round] {
                played.challenges.push(transcript.ood_point());
                transcript.ood_answer(answer);
            }

            played.witnesses.push(transcript.query_pow(round));
            played.indices.push(transcript.query_indices(round));
            played.challenges.push(transcript.round_batching());
            played
                .challenges
                .push(transcript.delegate_round_fold(delegate_stub));
        }

        // The base case runs under a seed of its own, inside the run's last bracket.
        transcript.delegate_base_case(|challenger| {
            play_base_case_prover(challenger, &shape.base_case, carried, &mut played);
        });

        transcript.finish();
        played
    }

    /// Replay every described step, verifier side, over the prover's own values.
    fn play_verifier(
        challenger: &mut Ch,
        shape: &ZkWhirShape,
        carried: &Carried,
        witnesses: &[F],
    ) -> Played {
        let mut played = Played {
            challenges: Vec::new(),
            witnesses: witnesses.to_vec(),
            indices: Vec::new(),
        };
        let mut transcript = ZkWhirVerifierTranscript::<Ch, F, EF>::new(challenger, shape.clone());

        played.challenges.push(transcript.initial_batching());
        played
            .challenges
            .push(transcript.delegate_initial_fold(delegate_stub));

        for (round, &witness) in witnesses.iter().take(shape.n_rounds()).enumerate() {
            transcript.oracle_commitment(DIGEST);
            transcript.switch_mask_commitment(DIGEST);
            for &answer in &carried.ood_answers[round] {
                played.challenges.push(transcript.ood_point());
                transcript.ood_answer(answer);
            }
            transcript
                .query_pow(round, witness)
                .expect("the prover's own witness satisfies the site");
            played.indices.push(transcript.query_indices(round));
            played.challenges.push(transcript.round_batching());
            played
                .challenges
                .push(transcript.delegate_round_fold(delegate_stub));
        }

        // The base case grinds once, at the site after every round's own.
        let base_witness = witnesses[shape.n_rounds()];
        transcript.delegate_base_case(|challenger| {
            play_base_case_verifier(
                challenger,
                &shape.base_case,
                carried,
                base_witness,
                &mut played,
            );
        });

        transcript.finish();
        played
    }

    #[test]
    fn every_position_step_describes_uniform_sampling() {
        // Invariant: a position is drawn by rejection sampling.
        //
        // Its distribution is therefore exactly uniform.
        //
        // A plain bit draw is a different distribution.
        // The type tag is what tells the two apart.
        //
        //     described  UniformBits(w)   ->  replayed with a uniform draw
        //     described  Bits(w)          ->  replayed with a plain draw
        //
        // A step described as one and played as the other is a pattern mismatch.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);

        // Fixture state: the run draws query indices per round.
        let round_steps: Vec<_> = shape
            .pattern::<F, EF>()
            .interactions()
            .iter()
            .filter(|step| step.label() == QUERY_INDICES)
            .map(Interaction::type_tag)
            .collect();
        assert!(
            !round_steps.is_empty(),
            "the fixture configuration must draw at least one query"
        );

        // Fixture state: the base case draws source positions and per-group positions.
        let base_steps: Vec<_> = shape
            .base_case
            .pattern::<F, EF>()
            .interactions()
            .iter()
            .filter(|step| matches!(step.label(), BASE_SOURCE_QUERIES | BASE_MASK_QUERIES))
            .map(Interaction::type_tag)
            .collect();
        assert!(
            !base_steps.is_empty(),
            "the fixture configuration must draw at least one spot check"
        );

        for tag in round_steps.into_iter().chain(base_steps) {
            assert!(
                matches!(tag, TypeTag::UniformBits { .. }),
                "a position step is described as {tag:?}, which is not how it is drawn",
            );
        }
    }

    #[test]
    fn every_delegated_phase_is_recorded_as_one_bracket() {
        // A delegated phase runs under its own seed.
        //
        // Its rounds are therefore not steps of the description built here.
        //
        //     initial fold        one bracket
        //     per round           one bracket
        //     base case           one bracket
        //
        // Every opener is matched.
        //
        // The description therefore passes structural validation.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);
        let pattern = shape.pattern::<F, EF>();

        let openers: Vec<_> = pattern
            .interactions()
            .iter()
            .filter(|step| step.hierarchy() == Hierarchy::Begin)
            .map(Interaction::label)
            .collect();
        let closers = pattern
            .interactions()
            .iter()
            .filter(|step| step.hierarchy() == Hierarchy::End)
            .count();

        let expected = 2 + shape.n_rounds();
        assert_eq!(openers.len(), expected);
        assert_eq!(openers.len(), closers);
        assert_eq!(openers[0], INITIAL_FOLD);
        assert_eq!(openers[openers.len() - 1], BASE_CASE);
    }

    #[test]
    fn no_step_of_a_delegated_phase_reaches_this_description() {
        // A masked batch names its own steps inside its own description.
        //
        // None of those labels belongs to the run described here.
        //
        //     child plays   mask_commitment  mu_tilde  round_poly  round_challenge
        //     run records   one bracket
        //
        // A label leaking through would mean the run tries to play the child's steps.
        let config = base_config();
        let pattern = ZkWhirShape::new(&config).pattern::<F, EF>();

        for label in [
            "mask_commitment",
            "mu_tilde",
            "mask_combination",
            "round_poly",
            "round_pow",
            "round_challenge",
            "joint_claim",
        ] {
            assert!(
                pattern.interactions().iter().all(|s| s.label() != label),
                "the run describes a step the delegate plays for itself: {label}",
            );
        }
    }

    #[test]
    fn the_run_describes_one_grind_per_round_and_the_delegates_describe_the_rest() {
        // Invariant: the run grinds once per code-switching round.
        //
        // It grinds nowhere else.
        //
        // Every other grinding site belongs to a delegate.
        //
        //     run       ->  one query grind per round
        //     base case ->  one grind before its spot checks
        //     sumcheck  ->  one grind per masked round, under its own seed
        //
        // Fixture state: grinding is capped at 12 bits and derived per round.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);

        // The recorded difficulties must be exactly the configured ones, in order.
        let described = pow_difficulties(&shape.pattern::<F, EF>());
        let configured: Vec<_> = shape
            .rounds
            .iter()
            .map(|round| round.query_pow_bits)
            .filter(|&bits| bits > 0)
            .collect();
        assert_eq!(
            described.iter().map(|&(_, bits)| bits).collect::<Vec<_>>(),
            configured,
        );
        assert!(described.iter().all(|&(label, _)| label == QUERY_POW));

        // The base case describes its own single grind, under its own seed.
        let base = pow_difficulties(&shape.base_case.pattern::<F, EF>());
        assert_eq!(base.len(), usize::from(shape.base_case.pow_bits > 0));
    }

    #[test]
    fn the_same_configuration_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the configuration.
        let config = base_config();
        assert_eq!(first_challenge(&config), first_challenge(&config));
    }

    #[test]
    fn the_hiding_pipeline_never_shares_a_seed_with_the_plain_one() {
        // Two protocols, one parameter set: the seeds must still differ.
        //
        // The protocol name inside the identifier is what separates them.
        let zk = base_config();
        let plain = WhirConfig::<EF, F, Ch>::new(NUM_VARIABLES, base_params()).unwrap();

        let mut hiding_challenger = fresh_challenger();
        ZkWhirShape::new(&zk)
            .domain_separator::<F, EF>()
            .seed(&mut hiding_challenger);

        let mut plain_challenger = fresh_challenger();
        WhirShape::new(&plain, 1)
            .domain_separator::<F, EF>()
            .seed(&mut plain_challenger);

        let hiding: F = hiding_challenger.sample();
        let plain: F = plain_challenger.sample();
        assert_ne!(hiding, plain);
    }

    #[test]
    fn the_base_case_never_shares_a_seed_with_the_run_around_it() {
        // The base case is a protocol of its own.
        //
        // It therefore carries a name of its own.
        //
        //     run       ->  [2 | p3-whir-hvzk      | .. ]
        //     base case ->  [1 | p3-whir-hvzk-base | .. ]
        //
        // A shared seed would let one description stand in for the other.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);

        let seeds = [
            ("run", seed_digest(&shape.domain_separator::<F, EF>())),
            (
                "base case",
                seed_digest(&shape.base_case.domain_separator::<F, EF>()),
            ),
        ];
        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn every_user_facing_parameter_reaches_the_seed() {
        // Walk the plain and hiding parameter sets, field by field.
        user_knob_moves_the_seed("security_level", |p, _| p.security_level += 1);
        user_knob_moves_the_seed("pow_bits", |p, _| p.pow_bits -= 1);
        user_knob_moves_the_seed("starting_log_inv_rate", |p, _| {
            p.starting_log_inv_rate += 1;
            p.round_log_inv_rates[0] += 1;
        });
        user_knob_moves_the_seed("round_log_inv_rates", |p, _| p.round_log_inv_rates[0] += 1);
        user_knob_moves_the_seed("soundness_type", |p, _| {
            p.soundness_type = SecurityAssumption::JohnsonBound;
        });
        user_knob_moves_the_seed("folding_factor", |p, _| {
            p.folding_factor = FoldingFactor::PerRound(vec![5, 3, 3]);
            p.round_log_inv_rates = round_rates(NUM_VARIABLES, &p.folding_factor);
        });
        user_knob_moves_the_seed("ell_zk", |_, z| z.ell_zk += 1);
        user_knob_moves_the_seed("mask_log_inv_rate", |_, z| z.mask_log_inv_rate += 1);

        // The variable count lives beside the parameters, not inside them.
        let wide_params = params_with(
            NUM_VARIABLES + 1,
            FoldingFactor::ConstantFromSecondRound(5, 3),
        );
        let wide = ZkWhirConfig::<EF, F, Ch>::new(NUM_VARIABLES + 1, wide_params, base_zk())
            .expect("one more variable is still a valid instance");
        assert_ne!(
            first_challenge(&base_config()),
            first_challenge(&wide),
            "changing num_variables left the seed where it was",
        );
    }

    #[test]
    fn every_derived_field_that_shapes_the_transcript_reaches_the_seed() {
        // Walk the plain half of the hiding configuration, field by field.
        derived_field_moves_the_seed("inner.commitment_ood_samples", |c| {
            c.inner.commitment_ood_samples += 1;
        });
        derived_field_moves_the_seed("inner.starting_folding_pow_bits", |c| {
            c.inner.starting_folding_pow_bits += 1;
        });
        derived_field_moves_the_seed("inner.folding_schedule", |c| {
            c.inner.folding_schedule[0] -= 1;
        });
        derived_field_moves_the_seed("inner.final_sumcheck_rounds", |c| {
            c.inner.final_sumcheck_rounds -= 1;
        });
        derived_field_moves_the_seed("inner.final_folding_pow_bits", |c| {
            c.inner.final_folding_pow_bits += 1;
        });
        derived_field_moves_the_seed("round.ood_samples", |c| {
            c.inner.round_parameters[0].ood_samples += 1;
        });
        derived_field_moves_the_seed("round.pow_bits", |c| {
            c.inner.round_parameters[0].pow_bits += 1;
        });
        derived_field_moves_the_seed("round.folding_pow_bits", |c| {
            c.inner.round_parameters[0].folding_pow_bits += 1;
        });
        derived_field_moves_the_seed("round.num_queries", |c| {
            c.inner.round_parameters[0].num_queries += 1;
        });
        derived_field_moves_the_seed("round.domain_size", |c| {
            c.inner.round_parameters[0].domain_size >>= 1;
        });
        derived_field_moves_the_seed("round.log_inv_rate", |c| {
            c.inner.round_parameters[0].log_inv_rate += 1;
        });
        derived_field_moves_the_seed("round_parameters.len", |c| {
            c.inner.round_parameters.pop();
            c.switch_masks.pop();
        });

        // Walk the hiding half.
        derived_field_moves_the_seed("final_queries", |c| c.final_queries += 1);
        derived_field_moves_the_seed("final_pow_bits", |c| c.final_pow_bits += 1);
        derived_field_moves_the_seed("mask_queries", |c| c.mask_queries += 1);
        derived_field_moves_the_seed("oracle_randomness[0]", |c| c.oracle_randomness[0] += 1);
        derived_field_moves_the_seed("oracle_randomness[last]", |c| {
            let last = c.oracle_randomness.len() - 1;
            c.oracle_randomness[last] += 1;
        });
        derived_field_moves_the_seed("sumcheck_mask.message_len", |c| {
            c.sumcheck_mask.message_len += 1;
        });
        derived_field_moves_the_seed("sumcheck_mask.randomness_len", |c| {
            c.sumcheck_mask.randomness_len += 1;
        });
        derived_field_moves_the_seed("sumcheck_mask.domain_size", |c| {
            c.sumcheck_mask.domain_size <<= 1;
        });
        derived_field_moves_the_seed("switch_masks[0].message_len", |c| {
            c.switch_masks[0].message_len += 1;
        });
        derived_field_moves_the_seed("switch_masks[0].randomness_len", |c| {
            c.switch_masks[0].randomness_len += 1;
        });
        derived_field_moves_the_seed("switch_masks[0].domain_size", |c| {
            c.switch_masks[0].domain_size <<= 1;
        });
    }

    #[test]
    fn both_sides_draw_the_same_challenges_from_the_same_values() {
        // Described run: the fixture configuration, at its own grinding difficulties.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);
        let carried = Carried::new(&shape, 0xC1A1);

        // Prover side: play every phase in order.
        let mut prover_challenger = fresh_challenger();
        let prover = play_prover(&mut prover_challenger, &shape, &carried);

        // Verifier side: the same calls, in the same order, over the same values.
        let mut verifier_challenger = fresh_challenger();
        let verifier = play_verifier(
            &mut verifier_challenger,
            &shape,
            &carried,
            &prover.witnesses,
        );

        assert_eq!(prover, verifier);

        // Both sponges land on the same state.
        //
        // Whatever runs next therefore agrees too.
        let prover_next: F = prover_challenger.sample();
        let verifier_next: F = verifier_challenger.sample();
        assert_eq!(prover_next, verifier_next);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_both_sides_agree_over_random_carried_values(seed in any::<u64>()) {
            // Completeness over the values the proof carries.
            //
            // The two sides agree whatever the answers, the claim and the reveals are.
            let config = base_config();
            let shape = ZkWhirShape::new(&config);
            let carried = Carried::new(&shape, seed);

            let mut prover_challenger = fresh_challenger();
            let prover = play_prover(&mut prover_challenger, &shape, &carried);

            let mut verifier_challenger = fresh_challenger();
            let verifier =
                play_verifier(&mut verifier_challenger, &shape, &carried, &prover.witnesses);

            prop_assert_eq!(prover, verifier);
        }
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn a_prover_that_plays_a_step_out_of_order_fails_loudly() {
        // Described order: the round's oracle commitment lands before its mask.
        //
        //     described  oracle_commitment       switch_mask_commitment
        //     played     switch_mask_commitment  oracle_commitment
        //
        // Once in the sponge, nothing about the two digests tells them apart.
        //
        // The step they are played at is what parts them.
        //
        // The player is what checks that step.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);
        let mut challenger = fresh_challenger();

        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut transcript =
                ZkWhirProverTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());
            let _alpha = transcript.initial_batching();
            let _folded = transcript.delegate_initial_fold(delegate_stub);
            // Mutation: the mask is bound where the new oracle belongs.
            transcript.switch_mask_commitment(DIGEST);
        }));

        let payload = caught.expect_err("a step played out of order must panic");
        let message = payload
            .downcast_ref::<String>()
            .expect("the player reports the mismatch as a formatted message");
        assert!(
            message.contains("but expected"),
            "the panic must diff the played step against the described one, got {message}",
        );
        assert!(
            message.contains(ORACLE_COMMITMENT),
            "the panic must name the step that was due, got {message}",
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn a_prover_that_skips_the_base_case_bracket_fails_loudly() {
        // Described run: the base case closes it, inside its own bracket.
        //
        //     described  .. round brackets ..   Begin base_case   End base_case
        //     played     .. round brackets ..   -- nothing --
        //
        // Skipping the bracket leaves two steps unplayed.
        //
        // A verifier would play them and land on a different sponge state.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);
        let carried = Carried::new(&shape, 0x5C1B);
        let mut challenger = fresh_challenger();

        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut transcript =
                ZkWhirProverTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());
            let _alpha = transcript.initial_batching();
            let _folded = transcript.delegate_initial_fold(delegate_stub);
            for round in 0..shape.n_rounds() {
                transcript.oracle_commitment(DIGEST);
                transcript.switch_mask_commitment(DIGEST);
                for &answer in &carried.ood_answers[round] {
                    let _point = transcript.ood_point();
                    transcript.ood_answer(answer);
                }
                let _witness = transcript.query_pow(round);
                let _indices = transcript.query_indices(round);
                let _batching = transcript.round_batching();
                let _folded = transcript.delegate_round_fold(delegate_stub);
            }
            // Mutation: the base case is never bracketed.
            transcript.finish();
        }));

        let payload = caught.expect_err("a skipped bracket must panic");
        let message = payload
            .downcast_ref::<String>()
            .expect("the player reports the gap as a formatted message");
        assert!(
            message.contains("not fully replayed"),
            "the panic must name the step that was skipped, got {message}",
        );
    }

    #[test]
    fn a_query_grinding_witness_that_misses_its_difficulty_is_rejected() {
        // Described run: 20 bits of grinding before the first round's query indices.
        //
        // Zero is a witness like any other.
        //
        // It clears 20 bits with probability 2^-20.
        let config = base_config();
        let mut shape = ZkWhirShape::new(&config);
        shape.rounds[0].query_pow_bits = 20;
        let ood_samples = shape.rounds[0].ood_samples;

        let mut challenger = fresh_challenger();
        let mut transcript = ZkWhirVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _alpha = transcript.initial_batching();
        let _folded = transcript.delegate_initial_fold(delegate_stub);
        transcript.oracle_commitment(DIGEST);
        transcript.switch_mask_commitment(DIGEST);
        for _ in 0..ood_samples {
            let _point = transcript.ood_point();
            transcript.ood_answer(EF::ONE);
        }

        let err = transcript
            .query_pow(0, F::ZERO)
            .expect_err("a witness that clears no bits must be rejected");

        assert_eq!(err, TranscriptFailure::PowWitness { round: 0, bits: 20 });
        // The failed read poisoned the driver.
        //
        // Dropping it here therefore raises nothing.
    }

    #[test]
    fn a_base_case_grinding_witness_that_misses_its_difficulty_is_rejected() {
        // Described base case: 20 bits of grinding before the spot-check positions.
        //
        // Fixture state: every reveal carries its described length.
        //
        // The grind is therefore reached at all.
        let config = base_config();
        let shape = ZkWhirShape::new(&config);
        let carried = Carried::new(&shape, 0xBADF);
        let mut base = shape.base_case;
        base.pow_bits = 20;

        let mut challenger = fresh_challenger();
        let mut transcript =
            ZkBaseCaseVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base.clone());
        transcript.fresh_commitment(DIGEST);
        for _ in &base.groups {
            transcript.blind_commitment(DIGEST);
        }
        transcript.claim(carried.masked_claim);
        let _gamma = transcript.gamma();
        for (message, randomness) in &carried.reveals {
            transcript
                .reveal(message, randomness)
                .expect("the described reveal lengths");
        }

        let err = transcript
            .spot_check_pow(F::ZERO)
            .expect_err("a witness that clears no bits must be rejected");

        assert_eq!(err, BaseCaseZkError::InvalidPowWitness);
    }

    #[test]
    fn a_base_case_reveal_of_the_wrong_length_is_rejected() {
        // Described source reveal: source_message_len values, then source_randomness_len.
        //
        //     described  (message_len, randomness_len)
        //     supplied   (message_len + 1, randomness_len)  -> rejected, nothing absorbed
        let config = base_config();
        let base = ZkWhirShape::new(&config).base_case;
        let expected = base.source_message_len;

        let mut challenger = fresh_challenger();
        let mut transcript =
            ZkBaseCaseVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base.clone());
        transcript.fresh_commitment(DIGEST);
        for _ in &base.groups {
            transcript.blind_commitment(DIGEST);
        }
        transcript.claim(EF::ONE);
        let _gamma = transcript.gamma();

        // Mutation: one extra value rides along in the message half.
        let message = vec![EF::ONE; expected + 1];
        let randomness = vec![EF::ONE; base.source_randomness_len];
        let err = transcript
            .reveal(&message, &randomness)
            .expect_err("a reveal outside the described length must error");

        assert_eq!(
            err,
            BaseCaseZkError::BlindedLengthMismatch {
                kind: "message",
                expected,
                actual: expected + 1,
            },
        );
        transcript.abort();
    }

    #[test]
    fn a_base_case_reveal_past_the_last_described_one_is_rejected() {
        // Described reveals: one for the source word, then one per group member.
        //
        //     described  1 + num_masks
        //     supplied   1 + num_masks + 1  -> rejected at the extra one
        let config = base_config();
        let base = ZkWhirShape::new(&config).base_case;
        let carried = Carried::new(&ZkWhirShape::new(&config), 0x0FF5);
        let num_masks = base.num_masks();

        let mut challenger = fresh_challenger();
        let mut transcript =
            ZkBaseCaseVerifierTranscript::<Ch, F, EF>::new(&mut challenger, base.clone());
        transcript.fresh_commitment(DIGEST);
        for _ in &base.groups {
            transcript.blind_commitment(DIGEST);
        }
        transcript.claim(carried.masked_claim);
        let _gamma = transcript.gamma();
        for (message, randomness) in &carried.reveals {
            transcript
                .reveal(message, randomness)
                .expect("the described reveal lengths");
        }

        // Mutation: one reveal more than the description carries.
        let err = transcript
            .reveal(&[EF::ONE], &[EF::ONE])
            .expect_err("a reveal past the last described one must error");

        assert_eq!(
            err,
            BaseCaseZkError::MaskCountMismatch {
                expected: num_masks,
                actual: num_masks + 1,
            },
        );
        transcript.abort();
    }

    #[test]
    fn a_saturated_position_phase_draws_nothing() {
        // Boundary: asking for at least as many positions as the domain opens them all.
        //
        //     16 positions, 16 asked  -> every position, no draw
        //     16 positions, 15 asked  -> 15 draws
        let config = base_config();
        let mut base = ZkWhirShape::new(&config).base_case;

        base.source_queries = base.source_domain_size;
        assert_eq!(base.source_query_draws(), 0);

        base.source_queries = base.source_domain_size - 1;
        assert_eq!(base.source_query_draws(), base.source_domain_size - 1);
    }

    #[test]
    fn the_reveal_order_walks_the_source_word_then_every_group_member() {
        // Invariant: the reveal sequence is fixed by the group list, not by the proof.
        //
        // Fixture state: the base case carries one source word and one reveal per mask.
        //
        //     position 0            source
        //     position 1 .. n       group 0 members, then group 1 members, ..
        //     position n + 1        past the end
        let config = base_config();
        let base = ZkWhirShape::new(&config).base_case;

        // The source word opens the sequence at its own code's lengths.
        assert_eq!(
            base.reveal_lengths(0),
            Some((base.source_message_len, base.source_randomness_len)),
        );

        // The first group member follows, at its group's code lengths.
        let first = base.groups[0].shape;
        assert_eq!(
            base.reveal_lengths(1),
            Some((first.message_len, first.randomness_len)),
        );

        // The sequence ends exactly one past the last mask.
        assert!(base.reveal_lengths(base.num_masks()).is_some());
        assert_eq!(base.reveal_lengths(base.num_masks() + 1), None);
    }
}
