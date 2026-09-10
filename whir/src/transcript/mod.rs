//! Fiat-Shamir transcript of the WHIR proximity test.
//!
//! # Overview
//!
//! One statement of what a WHIR run absorbs and draws, played by both sides.
//!
//! It is built from the derived configuration alone.
//! Neither side ever reads a count out of a proof.
//!
//! # Shape
//!
//! ```text
//!     Begin initial fold     batching challenge, then the delegated sumcheck
//!     End   initial fold
//!     per round:  commitment     one opaque value
//!                 out-of-domain  one point drawn, one answer sent, per sample
//!                 grinding       only when the difficulty is positive
//!                 queries        num_queries draws of index_bits bits
//!                 batching       one extension element
//!                 Begin fold     the round's delegated sumcheck
//!                 End   fold
//!     final polynomial       2^final_sumcheck_rounds extension elements
//!     final grinding         only when the difficulty is positive
//!     final queries          final_queries draws of index_bits bits
//!     Begin final fold       the closing delegated sumcheck, when it runs at all
//!     End   final fold
//! ```
//!
//! # Delegation
//!
//! A sumcheck phase is a protocol of its own, with its own seed and its own driver.
//!
//! WHIR therefore records the phase as a bracket, not as the rounds inside it.
//!
//! ```text
//!     Begin round_fold  ->  p3_sumcheck plays its own pattern  ->  End round_fold
//! ```
//!
//! The bracket states that the delegation happens, and where.
//! The counts inside it reach this seed through the instance label.
//!
//! The initial bracket also covers the challenge that batches the incoming claims.
//! The delegate draws that challenge itself, so the bracket is where it lands.
//!
//! # What is bound
//!
//! - Shape: every round, every grinding difficulty, every query width, every draw count.
//! - Shape: where each sumcheck phase runs, through its bracket.
//! - Instance label: the rounds and difficulty of every sumcheck phase.
//! - Instance label: the code rates, the security level, the soundness assumption.
//! - Instance label: the folding strategy and the variable count.
//! - Instance label: how many opening claims the run batches, and the committed row width.
//!
//! # What is not described
//!
//! Hints.
//! A hint never enters the sponge, so it cannot move a challenge.
//!
//! WHIR carries every hint inside its own serde proof.
//! The verifier length-checks each one against the configuration before use.
//!
//! # What is not bound
//!
//! The width of a commitment digest, which this layer cannot see.
//!
//! A commitment is absorbed opaquely, through the challenger's own encoding.
//! A wrong digest width therefore does not part the two sponges.
//!
//! ```text
//!     digest width  ->  Merkle opening check, against explicit Dimensions
//!     row width     ->  instance label, as commitment_row_width
//! ```
//!
//! The row width is the width of the matrix the initial commitment covers.
//! It is what one opened leaf carries, and the label binds it directly.
//!
//! # Soundness
//!
//! The seed is absorbed where the WHIR run starts, before its first challenge.
//! Everything the caller bound earlier stays in the sponge and keeps its effect.

pub mod zk;

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, TranscriptField, Unit, VerifierState,
};
use p3_challenger::{
    CanObserve, CanSample, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_field::{ExtensionField, TwoAdicField};
use p3_util::log2_strict_usize;
use thiserror::Error;

use crate::parameters::{FoldingFactor, SecurityAssumption, WhirConfig};

/// Version byte bound into the transcript seed.
///
/// The byte parts the seeds of two incompatible descriptions of one run.
///
/// Version 2 reserves the constant batching coefficient for the carried claim.
///
/// Version 3 keeps that reservation and records each sumcheck phase as one bracket.
const VERSION: u8 = 3;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-whir";

/// Step label of the challenge batching the incoming evaluation claims.
const INITIAL_BATCHING: &str = "initial_batching";

/// Container label of the fold that opens the run.
///
/// Covers the batching challenge and the sumcheck the delegate runs after it.
const INITIAL_FOLD: &str = "initial_fold";

/// Step label of a round commitment.
const COMMITMENT: &str = "commitment";

/// Step label of an out-of-domain evaluation point.
const OOD_POINT: &str = "ood_point";

/// Step label of the answer at an out-of-domain point.
const OOD_ANSWER: &str = "ood_answer";

/// Step label of the grinding step guarding a round's query indices.
const QUERY_POW: &str = "query_pow";

/// Step label of a round's query indices.
const QUERY_INDICES: &str = "query_indices";

/// Step label of the challenge batching one round's fresh constraints.
const ROUND_BATCHING: &str = "round_batching";

/// Container label of the fold that closes one intermediate round.
const ROUND_FOLD: &str = "round_fold";

/// Step label of the final polynomial, sent in the clear.
const FINAL_POLY: &str = "final_poly";

/// Step label of the grinding step guarding the final query indices.
const FINAL_QUERY_POW: &str = "final_query_pow";

/// Step label of the final query indices.
const FINAL_QUERY_INDICES: &str = "final_query_indices";

/// Container label of the fold that closes the run.
const FINAL_FOLD: &str = "final_fold";

/// Sponge alphabet of a challenger that speaks the base field natively.
pub type Alphabet<F> = FieldUnit<F>;

/// Type naming a delegated sumcheck phase at the type level.
///
/// The name is compared locally when a closer meets its opener.
/// It never reaches the pattern fingerprint.
struct Sumcheck;

/// Number of index draws a query phase actually makes.
///
/// ```text
///     num_queries <  folded  ->  num_queries draws
///     num_queries >= folded  ->  every position opens, nothing is drawn
/// ```
///
/// # Arguments
///
/// - `folded_domain_size`: number of addressable positions.
/// - `num_queries`: how many the configuration asks for.
#[must_use]
pub const fn query_draws(folded_domain_size: usize, num_queries: usize) -> usize {
    if num_queries >= folded_domain_size {
        0
    } else {
        num_queries
    }
}

/// Append `value` as eight big-endian bytes to the instance label.
fn push_u64<U: Unit>(separator: &mut DomainSeparator<U>, value: usize) {
    separator.instance(&(value as u64).to_be_bytes());
}

/// Append a grinding step, unless the site asks for no work at all.
///
/// A zero-difficulty grind absorbs nothing on either side, so it is not a step.
///
/// The witness lives in the base field, which is what both drivers record.
fn push_pow<F: TranscriptField>(steps: &mut Vec<Interaction>, label: &'static str, bits: usize) {
    if bits > 0 {
        steps.push(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            label,
            Length::Fixed(bits),
        ));
    }
}

/// Append a query-index step, unless the phase opens every position.
///
/// Every index is drawn at the same width, so they form one step.
///
/// The draw is rejection-sampled, which the uniform tag is what records.
fn push_query_indices(
    steps: &mut Vec<Interaction>,
    label: &'static str,
    index_bits: usize,
    draws: usize,
) {
    if draws > 0 {
        steps.push(Interaction::uniform_bits(
            Hierarchy::Atomic,
            Kind::Challenge,
            label,
            index_bits,
            Length::Fixed(draws),
        ));
    }
}

/// Append the opener and closer of one delegated phase.
///
/// The phase's own steps live in its own pattern, under its own seed.
///
/// The type parameter names the delegate.
///
/// It is compared where a closer meets its opener.
///
/// It never reaches the pattern fingerprint.
fn push_delegation<T: ?Sized>(steps: &mut Vec<Interaction>, label: &'static str) {
    steps.push(Interaction::marker::<T>(
        Hierarchy::Begin,
        Kind::Protocol,
        label,
    ));
    steps.push(Interaction::marker::<T>(
        Hierarchy::End,
        Kind::Protocol,
        label,
    ));
}

/// Bind the folding strategy: its variant, then the numbers it carries.
///
/// The derived schedule alone does not identify the strategy.
///
/// ```text
///     Constant(4)                  and  ConstantFromSecondRound(4, 4)
/// ```
///
/// Both derive the same schedule, so both describe one step sequence.
/// Binding the variant keeps the two configurations on distinct seeds.
fn bind_folding_factor<U: Unit>(separator: &mut DomainSeparator<U>, factor: &FoldingFactor) {
    // Discriminant first, so two variants carrying the same numbers stay distinct.
    match factor {
        FoldingFactor::Constant(f) => {
            separator
                .instance(&0u64.to_be_bytes())
                .instance(&(*f as u64).to_be_bytes());
        }
        FoldingFactor::ConstantFromSecondRound(first, rest) => {
            separator
                .instance(&1u64.to_be_bytes())
                .instance(&(*first as u64).to_be_bytes())
                .instance(&(*rest as u64).to_be_bytes());
        }
        FoldingFactor::PerRound(factors) => {
            separator
                .instance(&2u64.to_be_bytes())
                .instance(&(factors.len() as u64).to_be_bytes());
            for &f in factors {
                separator.instance(&(f as u64).to_be_bytes());
            }
        }
    }
}

/// Numbers that fix one sumcheck phase.
///
/// The phase runs under its own seed, played by its own driver.
/// Its numbers therefore reach this seed through the instance label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    /// Number of rounds this phase runs.
    pub rounds: usize,
    /// Grinding difficulty inside each round.
    pub pow_bits: usize,
}

impl SumcheckShape {
    /// Append this phase's numbers to the instance label under construction.
    fn bind<U: Unit>(&self, separator: &mut DomainSeparator<U>) {
        push_u64(separator, self.rounds);
        push_u64(separator, self.pow_bits);
    }
}

/// Numbers that fix one intermediate WHIR round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WhirRoundShape {
    /// Out-of-domain samples drawn against the new commitment.
    pub ood_samples: usize,
    /// Grinding difficulty guarding this round's query indices.
    pub query_pow_bits: usize,
    /// Number of query indices actually drawn.
    pub query_draws: usize,
    /// Bit width of each query index.
    pub index_bits: usize,
    /// Sumcheck phase folding the polynomial for the next round.
    pub sumcheck: SumcheckShape,
    /// Log-inverse rate of the codeword committed by this round.
    pub log_inv_rate: usize,
}

impl WhirRoundShape {
    /// Append this round's steps to a step sequence under construction.
    fn extend<F, EF>(&self, steps: &mut Vec<Interaction>)
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The commitment's encoding belongs to the commitment scheme.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            COMMITMENT,
            Length::Scalar,
        ));

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
        // Commitment, batching challenge, and the two bracket markers.
        4 + 2 * self.ood_samples
            + if self.query_pow_bits > 0 { 1 } else { 0 }
            + if self.query_draws > 0 { 1 } else { 0 }
    }
}

/// Numbers that fix the transcript of one WHIR run.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug)]
pub struct WhirShape {
    /// Variable count of the committed multilinear polynomial.
    pub num_variables: usize,
    /// Out-of-domain samples the commitment phase draws.
    pub commitment_ood_samples: usize,
    /// Opening claims the run batches into its first sum.
    ///
    /// Each claim is absorbed by the caller before the seed lands.
    /// The count is what fixes how many draws that pre-seed phase makes.
    pub num_opening_claims: usize,
    /// Base-field width of one row of the matrix the initial commitment covers.
    ///
    /// One opened leaf carries exactly this many elements.
    pub commitment_row_width: usize,
    /// Sumcheck phase folding the polynomial before the first round.
    pub initial_sumcheck: SumcheckShape,
    /// One entry per intermediate round, in round order.
    pub rounds: Vec<WhirRoundShape>,
    /// Coefficient count of the final polynomial, sent in the clear.
    pub final_poly_len: usize,
    /// Grinding difficulty guarding the final query indices.
    pub final_pow_bits: usize,
    /// Number of final query indices actually drawn.
    pub final_query_draws: usize,
    /// Bit width of each final query index.
    pub final_index_bits: usize,
    /// Sumcheck phase closing the run.
    pub final_sumcheck: SumcheckShape,
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
}

impl WhirShape {
    /// Derive the shape of one WHIR run from its configuration.
    ///
    /// # Arguments
    ///
    /// - `config`: the derived protocol configuration.
    /// - `num_opening_claims`: opening claims the caller bound before the seed.
    #[must_use]
    pub fn new<EF, F, Challenger>(
        config: &WhirConfig<EF, F, Challenger>,
        num_opening_claims: usize,
    ) -> Self
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
                WhirRoundShape {
                    ood_samples: params.ood_samples,
                    query_pow_bits: params.pow_bits,
                    query_draws: query_draws(folded, params.num_queries),
                    index_bits: log2_strict_usize(folded),
                    sumcheck: SumcheckShape {
                        rounds: config.round_folding_factor(index + 1),
                        pow_bits: params.folding_pow_bits,
                    },
                    log_inv_rate: params.log_inv_rate,
                }
            })
            .collect();

        // The final phase queries the last committed codeword.
        let final_config = config.final_round_config();
        let final_folded = final_config.domain_size >> final_config.folding_factor;

        Self {
            num_variables: config.num_variables,
            commitment_ood_samples: config.commitment_ood_samples,
            num_opening_claims,
            // The initial commitment lays the first fold's cosets out along one row.
            commitment_row_width: 1 << config.round_folding_factor(0),
            initial_sumcheck: SumcheckShape {
                rounds: config.round_folding_factor(0),
                pow_bits: config.starting_folding_pow_bits,
            },
            rounds,
            final_poly_len: 1 << final_config.num_variables,
            final_pow_bits: config.final_pow_bits,
            final_query_draws: query_draws(final_folded, config.final_queries),
            final_index_bits: log2_strict_usize(final_folded),
            final_sumcheck: SumcheckShape {
                rounds: config.final_sumcheck_rounds,
                pow_bits: config.final_folding_pow_bits,
            },
            security_level: config.security_level,
            pow_budget: config.pow_bits,
            starting_log_inv_rate: config.starting_log_inv_rate,
            soundness_type: config.soundness_type,
            folding_factor: config.folding_factor.clone(),
        }
    }

    /// Number of intermediate rounds this shape runs.
    #[must_use]
    pub const fn n_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Grinding label and difficulty of the query site of `round`.
    ///
    /// ```text
    ///     round <  n_rounds  ->  that round's own site
    ///     round >= n_rounds  ->  the final site
    /// ```
    fn query_pow_site(&self, round: usize) -> (&'static str, usize) {
        if round < self.rounds.len() {
            (QUERY_POW, self.rounds[round].query_pow_bits)
        } else {
            (FINAL_QUERY_POW, self.final_pow_bits)
        }
    }

    /// Index label, index width, and draw count of the query site of `round`.
    ///
    /// A draw count of zero means the phase opens every position instead.
    fn query_index_site(&self, round: usize) -> (&'static str, usize, usize) {
        if round < self.rounds.len() {
            let shape = &self.rounds[round];
            (QUERY_INDICES, shape.index_bits, shape.query_draws)
        } else {
            (
                FINAL_QUERY_INDICES,
                self.final_index_bits,
                self.final_query_draws,
            )
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// Every container opened below is closed one step later, in the same call.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // Initial bracket, every round, three closing steps, and the final bracket.
        let capacity = 2
            + self
                .rounds
                .iter()
                .map(WhirRoundShape::step_count)
                .sum::<usize>()
            + 5;
        let mut steps = Vec::with_capacity(capacity);

        // The delegate draws the claim-batching challenge, so the bracket covers it.
        push_delegation::<Sumcheck>(&mut steps, INITIAL_FOLD);

        for round in &self.rounds {
            round.extend::<F, EF>(&mut steps);
        }

        // The final polynomial is sent in full, so it is one fixed-length step.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            FINAL_POLY,
            Length::Fixed(self.final_poly_len),
        ));

        push_pow::<F>(&mut steps, FINAL_QUERY_POW, self.final_pow_bits);
        push_query_indices(
            &mut steps,
            FINAL_QUERY_INDICES,
            self.final_index_bits,
            self.final_query_draws,
        );

        // A run described with no closing rounds delegates nothing at all.
        if self.final_sumcheck.rounds > 0 {
            push_delegation::<Sumcheck>(&mut steps, FINAL_FOLD);
        }

        InteractionPattern::new(steps).expect("every container opened here is closed here")
    }

    /// Bind the protocol identity, this shape, and the remaining parameters.
    ///
    /// A parameter that changes the step sequence is covered by the fingerprint.
    /// The rest go in the instance label.
    ///
    /// # Soundness
    ///
    /// The code rates and the assumption reach the shape only indirectly.
    ///
    /// ```text
    ///     rate, assumption  ->  query count  ->  step length
    /// ```
    ///
    /// That arrow is not injective.
    ///
    /// Two parameter sets landing on one query count would share a shape.
    /// The label therefore carries the rates and the assumption directly.
    ///
    /// A delegated sumcheck contributes one bracket whatever its length.
    /// Its rounds and difficulty therefore travel in the label too.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The whole soundness statement is phrased in these numbers.
        push_u64(&mut separator, self.num_variables);
        // The commitment phase draws its samples before the described run starts.
        push_u64(&mut separator, self.commitment_ood_samples);
        // So does the claim phase, once per claim the caller batches.
        push_u64(&mut separator, self.num_opening_claims);
        // Width of one committed row, which fixes what a single opened leaf carries.
        push_u64(&mut separator, self.commitment_row_width);
        push_u64(&mut separator, self.security_level);
        push_u64(&mut separator, self.pow_budget);
        push_u64(&mut separator, self.starting_log_inv_rate);
        push_u64(&mut separator, self.soundness_type as usize);

        bind_folding_factor(&mut separator, &self.folding_factor);

        // Every delegated phase, in the order the run plays them.
        self.initial_sumcheck.bind(&mut separator);
        self.final_sumcheck.bind(&mut separator);

        // Each round commits at its own rate, and the rate sets that round's distance.
        push_u64(&mut separator, self.rounds.len());
        for round in &self.rounds {
            push_u64(&mut separator, round.log_inv_rate);
            round.sumcheck.bind(&mut separator);
        }

        separator
    }
}

/// A transcript step the proof failed to satisfy.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranscriptFailure {
    /// A grinding witness did not meet the difficulty its step requires.
    #[error("round {round}: query grinding witness clears fewer than {bits} bits")]
    PowWitness {
        /// Round whose query site rejected the witness, `n_rounds` for the final one.
        round: usize,
        /// Difficulty the site requires, in bits.
        bits: usize,
    },
    /// The final polynomial carries a coefficient count the run never described.
    #[error("expected {expected} final evaluations, got {got}")]
    FinalPolyLength {
        /// Count the run was described with.
        expected: usize,
        /// Count the proof carries.
        got: usize,
    },
}

/// Prover-side transcript of one plain WHIR run.
///
/// Holds the only definition of what a prover plays at each phase.
///
/// The challenger is borrowed, not consumed.
/// WHIR runs inside a larger protocol whose transcript continues afterwards.
pub struct WhirProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: WhirShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> WhirProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: WhirShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this run was described with.
    pub const fn shape(&self) -> &WhirShape {
        &self.shape
    }

    /// Lend the sponge to the sumcheck that opens the run.
    ///
    /// The delegate draws the claim-batching challenge itself.
    /// The bracket therefore covers that challenge as well as the rounds.
    pub fn delegate_initial_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate(INITIAL_FOLD, run)
    }

    /// Lend the sponge to the sumcheck that closes one intermediate round.
    pub fn delegate_round_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate(ROUND_FOLD, run)
    }

    /// Lend the sponge to the sumcheck that closes the run.
    ///
    /// # Returns
    ///
    /// `None` when the run is described with no closing rounds, which runs nothing.
    pub fn delegate_final_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> Option<R> {
        let delegates = self.shape.final_sumcheck.rounds > 0;
        delegates.then(|| self.delegate(FINAL_FOLD, run))
    }

    /// Bind one round's commitment.
    pub fn commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(COMMITMENT, commitment);
    }

    /// Draw one out-of-domain evaluation point.
    pub fn ood_point(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OOD_POINT)
            .into_inner()
    }

    /// Bind the answer at one out-of-domain point.
    ///
    /// Each answer is bound before the next point is drawn.
    pub fn ood_answer(&mut self, answer: EF) {
        let _bound = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(OOD_ANSWER, &answer);
    }

    /// Grind the query site of `round`.
    ///
    /// A `round` at or past the last one names the final site.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn query_pow(&mut self, round: usize) -> F {
        let (label, bits) = self.shape.query_pow_site(round);
        if bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(label, bits)
    }

    /// Draw the query indices of `round`.
    ///
    /// A `round` at or past the last one names the final site.
    ///
    /// # Returns
    ///
    /// Every index in draw order, repeats included.
    /// A saturated phase opens every position instead and draws nothing.
    pub fn query_indices(&mut self, round: usize) -> Vec<usize> {
        let (label, width, draws) = self.shape.query_index_site(round);
        // A saturated phase has nothing left to decide, so no draw is described.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(label, width, draws)
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

    /// Bind the final polynomial, sent in the clear.
    ///
    /// # Panics
    ///
    /// When the polynomial is not the length the run was described with.
    pub fn final_poly(&mut self, evaluations: &[EF]) {
        let _bound = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, evaluations);
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "WHIR carries every value in its own proof",
        );
    }

    /// Bracket one delegated sumcheck and hand it the borrowed sponge.
    fn delegate<R>(&mut self, label: &'static str, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Sumcheck>(label);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<Sumcheck>(label);
        output
    }
}

/// Verifier-side transcript of one plain WHIR run.
///
/// Mirrors the prover side call for call, over the same description.
pub struct WhirVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: WhirShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> WhirVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: WhirShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this run was described with.
    pub const fn shape(&self) -> &WhirShape {
        &self.shape
    }

    /// Lend the sponge to the sumcheck that opens the run.
    pub fn delegate_initial_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate(INITIAL_FOLD, run)
    }

    /// Lend the sponge to the sumcheck that closes one intermediate round.
    pub fn delegate_round_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.delegate(ROUND_FOLD, run)
    }

    /// Lend the sponge to the sumcheck that closes the run.
    ///
    /// # Returns
    ///
    /// `None` when the run is described with no closing rounds, which runs nothing.
    pub fn delegate_final_fold<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> Option<R> {
        let delegates = self.shape.final_sumcheck.rounds > 0;
        delegates.then(|| self.delegate(FINAL_FOLD, run))
    }

    /// Bind one round's commitment.
    pub fn commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(COMMITMENT, commitment);
    }

    /// Redraw one out-of-domain evaluation point.
    pub fn ood_point(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OOD_POINT)
            .into_inner()
    }

    /// Bind the answer at one out-of-domain point.
    pub fn ood_answer(&mut self, answer: EF) {
        let _bound = self
            .state
            .observe_extension::<F, EF, FieldToFieldCodec<F>>(OOD_ANSWER, &answer);
    }

    /// Replay the grind of the query site of `round`.
    ///
    /// A `round` at or past the last one names the final site.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn query_pow(&mut self, round: usize, witness: F) -> Result<(), TranscriptFailure> {
        let (label, bits) = self.shape.query_pow_site(round);
        if bits == 0 {
            return Ok(());
        }
        // A failed check poisons the driver, so the rejection travels alone.
        self.state
            .observe_pow(label, bits, witness)
            .map_err(|_| TranscriptFailure::PowWitness { round, bits })
    }

    /// Redraw the query indices of `round`.
    ///
    /// A `round` at or past the last one names the final site.
    pub fn query_indices(&mut self, round: usize) -> Vec<usize> {
        let (label, width, draws) = self.shape.query_index_site(round);
        // A saturated phase has nothing left to decide, so no draw is described.
        if draws == 0 {
            return (0..1usize << width).collect();
        }
        self.state
            .challenge_uniform_bits::<F>(label, width, draws)
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

    /// Bind the final polynomial the proof carries.
    ///
    /// # Errors
    ///
    /// When the evaluation count differs from the described one.
    pub fn final_poly(&mut self, evaluations: &[EF]) -> Result<(), TranscriptFailure> {
        let expected = self.shape.final_poly_len;
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLY, evaluations)
            .map(|_| ())
            .map_err(|_| TranscriptFailure::FinalPolyLength {
                expected,
                got: evaluations.len(),
            })
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// Every path that leaves the transcript early goes through this.
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
            .expect("WHIR reads an empty wire, so no bytes can remain");
    }

    /// Bracket one delegated sumcheck and hand it the borrowed sponge.
    fn delegate<R>(&mut self, label: &'static str, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Sumcheck>(label);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<Sumcheck>(label);
        output
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    #[cfg(panic = "unwind")]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::fs::TypeTag;
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::parameters::ProtocolParameters;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;
    type Config = WhirConfig<EF, F, Ch>;

    /// Variable count every configuration in this module is derived at.
    const NUM_VARIABLES: usize = 16;

    /// Opening claims every shape in this module is derived with.
    const CLAIMS: usize = 3;

    /// Step label a delegated sumcheck round carries inside its own description.
    ///
    /// No step of a delegated phase belongs to the description built here.
    /// The label is what one check below looks for and must not find.
    const FOLD_CHALLENGE: &str = "fold_challenge";

    /// A commitment shaped like the ones a Merkle scheme hands this layer.
    const DIGEST: [F; 8] = [F::ONE; 8];

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// One code rate per intermediate round, growing with the folding schedule.
    fn round_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
        let schedule = folding_factor
            .compute_folding_schedule(num_variables)
            .expect("the fixture schedules are all valid");
        let mut rate = 1;
        schedule[..schedule.len() - 1]
            .iter()
            .map(|folding| {
                rate += folding - 1;
                rate
            })
            .collect()
    }

    /// Baseline parameters every walk below perturbs exactly one field of.
    fn base_params() -> ProtocolParameters {
        params_with(NUM_VARIABLES, FoldingFactor::Constant(4))
    }

    /// Baseline parameters carrying a caller-chosen folding strategy.
    fn params_with(num_variables: usize, folding_factor: FoldingFactor) -> ProtocolParameters {
        ProtocolParameters {
            security_level: 32,
            pow_bits: 12,
            round_log_inv_rates: round_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        }
    }

    fn config_from(params: ProtocolParameters) -> Config {
        WhirConfig::new(NUM_VARIABLES, params).expect("the fixture parameters are all valid")
    }

    /// Shape of the fixture configuration, at the fixture claim count.
    fn shape_of(config: &Config) -> WhirShape {
        WhirShape::new(config, CLAIMS)
    }

    /// First challenge the seed of a configuration produces on a fresh sponge.
    fn first_challenge(config: &Config) -> F {
        first_challenge_of(&shape_of(config))
    }

    /// First challenge the seed of a shape produces on a fresh sponge.
    fn first_challenge_of(shape: &WhirShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    /// Assert that re-deriving from perturbed user parameters moves the seed.
    fn user_knob_moves_the_seed(name: &str, perturb: impl FnOnce(&mut ProtocolParameters)) {
        let mut params = base_params();
        perturb(&mut params);
        assert_ne!(
            first_challenge(&config_from(base_params())),
            first_challenge(&config_from(params)),
            "changing {name} left the seed where it was",
        );
    }

    /// Assert that perturbing one derived field moves the seed.
    fn derived_field_moves_the_seed(name: &str, perturb: impl FnOnce(&mut Config)) {
        let base = config_from(base_params());
        let mut tweaked = base.clone();
        perturb(&mut tweaked);
        assert_ne!(
            first_challenge(&base),
            first_challenge(&tweaked),
            "changing {name} left the seed where it was",
        );
    }

    /// Stand-in for a delegated sumcheck: absorbs one value, then draws one.
    ///
    /// The real phase seeds its own driver from the state this one has reached.
    /// All the bracket has to record is that the delegation happened, and where.
    fn delegate_stub(challenger: &mut Ch) -> EF {
        challenger.observe_algebra_element(EF::ONE);
        challenger.sample_algebra_element()
    }

    /// Everything one side of a played run produces.
    #[derive(Debug, PartialEq, Eq)]
    struct Played {
        /// Challenges drawn, in draw order.
        challenges: Vec<EF>,
        /// Grinding witnesses, one per query site.
        witnesses: Vec<F>,
        /// Query indices, one list per query site.
        indices: Vec<Vec<usize>>,
    }

    /// Values a described run carries inside its own proof.
    #[derive(Clone, Debug)]
    struct Carried {
        /// One out-of-domain answer per sample, per round.
        ood_answers: Vec<Vec<EF>>,
        /// The final polynomial, sent in the clear.
        final_poly: Vec<EF>,
    }

    impl Carried {
        /// Values that fit `shape`, drawn from a single seed.
        fn new(shape: &WhirShape, seed: u64) -> Self {
            let mut rng = SmallRng::seed_from_u64(seed);
            let ood_answers: Vec<Vec<EF>> = shape
                .rounds
                .iter()
                .map(|round| (0..round.ood_samples).map(|_| rng.random()).collect())
                .collect();
            let final_poly = (0..shape.final_poly_len).map(|_| rng.random()).collect();
            Self {
                ood_answers,
                final_poly,
            }
        }
    }

    /// Play every described step, prover side, in the order the pipeline plays them.
    fn play_prover(challenger: &mut Ch, shape: &WhirShape, carried: &Carried) -> Played {
        let mut played = Played {
            challenges: Vec::new(),
            witnesses: Vec::new(),
            indices: Vec::new(),
        };
        let mut transcript = WhirProverTranscript::<Ch, F, EF>::new(challenger, shape.clone());

        played
            .challenges
            .push(transcript.delegate_initial_fold(delegate_stub));

        for round in 0..shape.n_rounds() {
            transcript.commitment(DIGEST);
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

        transcript.final_poly(&carried.final_poly);
        let final_site = shape.n_rounds();
        played.witnesses.push(transcript.query_pow(final_site));
        played.indices.push(transcript.query_indices(final_site));
        played
            .challenges
            .extend(transcript.delegate_final_fold(delegate_stub));

        transcript.finish();
        played
    }

    /// Replay every described step, verifier side, over the prover's own values.
    fn play_verifier(
        challenger: &mut Ch,
        shape: &WhirShape,
        carried: &Carried,
        witnesses: &[F],
    ) -> Played {
        let mut played = Played {
            challenges: Vec::new(),
            witnesses: witnesses.to_vec(),
            indices: Vec::new(),
        };
        let mut transcript = WhirVerifierTranscript::<Ch, F, EF>::new(challenger, shape.clone());

        played
            .challenges
            .push(transcript.delegate_initial_fold(delegate_stub));

        for (round, &witness) in witnesses.iter().take(shape.n_rounds()).enumerate() {
            transcript.commitment(DIGEST);
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

        transcript
            .final_poly(&carried.final_poly)
            .expect("the described evaluation count");
        let final_site = shape.n_rounds();
        transcript
            .query_pow(final_site, witnesses[final_site])
            .expect("the prover's own witness satisfies the site");
        played.indices.push(transcript.query_indices(final_site));
        played
            .challenges
            .extend(transcript.delegate_final_fold(delegate_stub));

        transcript.finish();
        played
    }

    #[test]
    fn every_query_step_describes_uniform_sampling() {
        // The query phase draws through rejection sampling, so its positions are exactly uniform.
        // A plain bit draw is a different distribution, and the tag is what tells the two apart.
        //
        //     described  UniformBits(w)   ->  replayed with a uniform draw
        //     described  Bits(w)          ->  replayed with a plain draw
        //
        // A step described as one and played as the other is a pattern mismatch.
        let config = config_from(base_params());
        let pattern = shape_of(&config).pattern::<F, EF>();

        // Fixture state: the plain pipeline draws at two labels, per round and once at the end.
        let query_steps: Vec<_> = pattern
            .interactions()
            .iter()
            .filter(|step| matches!(step.label(), QUERY_INDICES | FINAL_QUERY_INDICES))
            .collect();

        assert!(
            !query_steps.is_empty(),
            "the fixture configuration must draw at least one query"
        );

        for step in query_steps {
            assert!(
                matches!(step.type_tag(), TypeTag::UniformBits { .. }),
                "{} is described as {:?}, which is not how it is drawn",
                step.label(),
                step.type_tag(),
            );
        }
    }

    #[test]
    fn every_sumcheck_phase_is_recorded_as_one_bracket() {
        // A sumcheck phase runs under its own seed, so its rounds are not steps here.
        //
        //     initial fold        one bracket
        //     per round           one bracket
        //     final fold          one bracket, when the run has closing rounds
        //
        // Every opener is matched, so the description passes structural validation.
        let config = config_from(base_params());
        let shape = shape_of(&config);
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

        let expected = 1 + shape.n_rounds() + usize::from(shape.final_sumcheck.rounds > 0);
        assert_eq!(openers.len(), expected);
        assert_eq!(openers.len(), closers);
        assert_eq!(openers[0], INITIAL_FOLD);

        // No round of any sumcheck phase reaches this pattern as a step of its own.
        assert!(
            pattern
                .interactions()
                .iter()
                .all(|step| step.label() != FOLD_CHALLENGE),
        );
    }

    #[test]
    fn the_same_configuration_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the configuration.
        let config = config_from(base_params());
        assert_eq!(first_challenge(&config), first_challenge(&config));
    }

    #[test]
    fn every_user_facing_parameter_reaches_the_seed() {
        // Walk `ProtocolParameters` field by field, plus the variable count.
        //
        // Each case re-derives the whole configuration, so a knob reaches the
        // seed whether it moves the step sequence or only the instance label.
        user_knob_moves_the_seed("security_level", |p| p.security_level += 1);
        user_knob_moves_the_seed("pow_bits", |p| p.pow_bits -= 1);
        user_knob_moves_the_seed("starting_log_inv_rate", |p| p.starting_log_inv_rate += 1);
        user_knob_moves_the_seed("round_log_inv_rates", |p| p.round_log_inv_rates[0] += 1);
        user_knob_moves_the_seed("soundness_type", |p| {
            p.soundness_type = SecurityAssumption::JohnsonBound;
        });
        user_knob_moves_the_seed("folding_factor", |p| {
            p.folding_factor = FoldingFactor::PerRound(vec![4, 4, 4]);
            p.round_log_inv_rates = round_rates(NUM_VARIABLES, &p.folding_factor);
        });

        // The variable count lives beside the parameters, not inside them.
        let wide_params = params_with(NUM_VARIABLES + 1, FoldingFactor::Constant(4));
        let wide = WhirConfig::<EF, F, Ch>::new(NUM_VARIABLES + 1, wide_params).unwrap();
        assert_ne!(
            first_challenge(&config_from(base_params())),
            first_challenge(&wide),
            "changing num_variables left the seed where it was",
        );
    }

    #[test]
    fn two_folding_strategies_deriving_one_schedule_still_split_the_seed() {
        // At 16 variables both strategies derive the schedule [4, 4, 4].
        //
        //     Constant(4)                  -> 4, 4, 4
        //     ConstantFromSecondRound(4, 4) -> 4, 4, 4
        //
        // Every derived number agrees, so the two share a step sequence.
        // Only the instance label separates them, so it must carry the variant.
        let constant = config_from(params_with(NUM_VARIABLES, FoldingFactor::Constant(4)));
        let from_second = config_from(params_with(
            NUM_VARIABLES,
            FoldingFactor::ConstantFromSecondRound(4, 4),
        ));

        assert_eq!(constant.folding_schedule, from_second.folding_schedule);
        assert_ne!(first_challenge(&constant), first_challenge(&from_second));
    }

    #[test]
    fn every_derived_field_that_shapes_the_transcript_reaches_the_seed() {
        // Walk `WhirConfig` field by field, perturbing the derived value itself.
        //
        // A derived field reaches the seed either through the step sequence or
        // through the instance label, so this is the check that no
        // transcript-bearing number was left out of both.
        derived_field_moves_the_seed("commitment_ood_samples", |c| c.commitment_ood_samples += 1);
        derived_field_moves_the_seed("starting_folding_pow_bits", |c| {
            c.starting_folding_pow_bits += 1;
        });
        derived_field_moves_the_seed("folding_schedule", |c| c.folding_schedule[0] -= 1);
        derived_field_moves_the_seed("final_queries", |c| c.final_queries += 1);
        derived_field_moves_the_seed("final_pow_bits", |c| c.final_pow_bits += 1);
        derived_field_moves_the_seed("final_sumcheck_rounds", |c| c.final_sumcheck_rounds -= 1);
        derived_field_moves_the_seed("final_folding_pow_bits", |c| c.final_folding_pow_bits += 1);

        // Every per-round number the transcript reads, one at a time.
        derived_field_moves_the_seed("round.ood_samples", |c| {
            c.round_parameters[0].ood_samples += 1;
        });
        derived_field_moves_the_seed("round.pow_bits", |c| c.round_parameters[0].pow_bits += 1);
        derived_field_moves_the_seed("round.folding_pow_bits", |c| {
            c.round_parameters[0].folding_pow_bits += 1;
        });
        derived_field_moves_the_seed("round.num_queries", |c| {
            c.round_parameters[0].num_queries += 1;
        });
        derived_field_moves_the_seed("round.domain_size", |c| {
            c.round_parameters[0].domain_size >>= 1;
        });
        derived_field_moves_the_seed("round.log_inv_rate", |c| {
            c.round_parameters[0].log_inv_rate += 1;
        });
        derived_field_moves_the_seed("round_parameters.len", |c| {
            c.round_parameters.pop();
        });
    }

    #[test]
    fn the_pre_seed_claim_phase_reaches_the_seed() {
        // Two numbers fix what the caller absorbed before this seed landed.
        //
        //     opening claims       one point drawn and one batch absorbed each
        //     committed row width  what a single opened leaf carries
        //
        // Neither moves the step sequence, so only the instance label can carry them.
        let config = config_from(base_params());
        let base = shape_of(&config);

        let mut more_claims = base.clone();
        more_claims.num_opening_claims += 1;
        assert_eq!(more_claims.pattern::<F, EF>(), base.pattern::<F, EF>());
        assert_ne!(
            first_challenge_of(&more_claims),
            first_challenge_of(&base),
            "the opening-claim count left the seed where it was",
        );

        let mut wider = base.clone();
        wider.commitment_row_width *= 2;
        assert_eq!(wider.pattern::<F, EF>(), base.pattern::<F, EF>());
        assert_ne!(
            first_challenge_of(&wider),
            first_challenge_of(&base),
            "the committed row width left the seed where it was",
        );
    }

    #[test]
    fn both_sides_draw_the_same_challenges_from_the_same_values() {
        // Described run: the fixture configuration, at its own grinding difficulties.
        let config = config_from(base_params());
        let shape = shape_of(&config);
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

        // Both sponges land on the same state, so whatever runs next agrees too.
        let prover_next: F = prover_challenger.sample();
        let verifier_next: F = verifier_challenger.sample();
        assert_eq!(prover_next, verifier_next);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_both_sides_agree_over_random_carried_values(seed in any::<u64>()) {
            // Completeness over the values the proof carries: the two sides
            // agree whatever the out-of-domain answers and final polynomial are.
            let config = config_from(base_params());
            let shape = shape_of(&config);
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
        // Described order: the round's commitment lands before its batching challenge.
        //
        //     described  commitment       round_batching
        //     played     round_batching   commitment
        //
        // Nothing about the two values says which is which once they are in the sponge.
        // The step they are played at is what parts them, and the player checks it.
        let config = config_from(base_params());
        let shape = shape_of(&config);
        let mut challenger = fresh_challenger();

        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut transcript =
                WhirProverTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());
            let _folded = transcript.delegate_initial_fold(delegate_stub);
            // Mutation: the batching challenge is drawn where the commitment belongs.
            let _swapped = transcript.round_batching();
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
            message.contains(COMMITMENT),
            "the panic must name the step that was due, got {message}",
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn a_prover_that_skips_a_step_fails_loudly() {
        // Described run: the final query indices are drawn after the final grind.
        //
        //     described  final_query_pow   final_query_indices
        //     played     final_query_pow   -- nothing --
        //
        // Skipping the draw leaves one step unplayed.
        //
        // A verifier would draw at that step and land on a different sponge state.
        let config = config_from(base_params());
        let shape = shape_of(&config);
        let carried = Carried::new(&shape, 0x5C1B);
        let mut challenger = fresh_challenger();

        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut transcript =
                WhirProverTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());
            let _folded = transcript.delegate_initial_fold(delegate_stub);
            for round in 0..shape.n_rounds() {
                transcript.commitment(DIGEST);
                for &answer in &carried.ood_answers[round] {
                    let _point = transcript.ood_point();
                    transcript.ood_answer(answer);
                }
                let _witness = transcript.query_pow(round);
                let _indices = transcript.query_indices(round);
                let _batching = transcript.round_batching();
                let _folded = transcript.delegate_round_fold(delegate_stub);
            }
            transcript.final_poly(&carried.final_poly);
            let _witness = transcript.query_pow(shape.n_rounds());
            // Mutation: the final query indices are never drawn.
            let _folded = transcript.delegate_final_fold(delegate_stub);
            transcript.finish();
        }));

        let payload = caught.expect_err("a skipped step must panic");
        let message = payload
            .downcast_ref::<String>()
            .expect("the player reports the gap as a formatted message");
        assert!(
            message.contains("but expected") || message.contains("not fully replayed"),
            "the panic must name the step that was skipped, got {message}",
        );
    }

    #[test]
    fn a_grinding_witness_that_misses_its_difficulty_is_rejected() {
        // Described run: 20 bits of grinding before the first round's query indices.
        //
        // Zero is a witness like any other, and it clears 20 bits with probability 2^-20.
        let config = config_from(base_params());
        let mut shape = shape_of(&config);
        shape.rounds[0].query_pow_bits = 20;
        let ood_samples = shape.rounds[0].ood_samples;

        let mut challenger = fresh_challenger();
        let mut transcript = WhirVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        let _folded = transcript.delegate_initial_fold(delegate_stub);
        transcript.commitment(DIGEST);
        for _ in 0..ood_samples {
            let _point = transcript.ood_point();
            transcript.ood_answer(EF::ONE);
        }

        let err = transcript
            .query_pow(0, F::ZERO)
            .expect_err("a witness that clears no bits must be rejected");

        assert_eq!(err, TranscriptFailure::PowWitness { round: 0, bits: 20 });
        // The failed read poisoned the driver, so dropping it here raises nothing.
    }

    #[test]
    fn a_final_polynomial_of_the_wrong_length_is_rejected() {
        // Described length: 2^final_round_config().num_variables evaluations.
        //
        //     described  final_poly_len
        //     supplied   final_poly_len + 1   -> rejected, nothing absorbed
        let config = config_from(base_params());
        let mut shape = shape_of(&config);
        // Fixture state: no grinding anywhere, so a zero witness clears every site.
        for round in &mut shape.rounds {
            round.query_pow_bits = 0;
        }
        shape.final_pow_bits = 0;
        let expected = shape.final_poly_len;
        let carried = Carried::new(&shape, 0xBAD1);

        let mut challenger = fresh_challenger();
        let mut transcript =
            WhirVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape.clone());
        let _folded = transcript.delegate_initial_fold(delegate_stub);
        for round in 0..shape.n_rounds() {
            transcript.commitment(DIGEST);
            for &answer in &carried.ood_answers[round] {
                let _point = transcript.ood_point();
                transcript.ood_answer(answer);
            }
            transcript
                .query_pow(round, F::ZERO)
                .expect("the fixture asks for no work at this site");
            let _indices = transcript.query_indices(round);
            let _batching = transcript.round_batching();
            let _folded = transcript.delegate_round_fold(delegate_stub);
        }

        let mut too_long = carried.final_poly;
        too_long.push(EF::ONE);
        let err = transcript
            .final_poly(&too_long)
            .expect_err("an evaluation count outside the described one must error");

        assert_eq!(
            err,
            TranscriptFailure::FinalPolyLength {
                expected,
                got: expected + 1,
            }
        );
    }

    #[test]
    fn a_saturated_query_phase_draws_nothing() {
        // Boundary: asking for at least as many queries as positions opens them all.
        //
        //     8 positions, 8 queries  -> every position, no draw
        //     8 positions, 7 queries  -> 7 draws
        assert_eq!(query_draws(8, 8), 0);
        assert_eq!(query_draws(8, 9), 0);
        assert_eq!(query_draws(8, 7), 7);
    }
}
