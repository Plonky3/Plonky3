//! Fiat-Shamir transcript of the STIR proximity test.
//!
//! # Overview
//!
//! One description of what a STIR run absorbs and squeezes, consumed by both sides.
//!
//! It is built from the round schedule alone.
//!
//! Both sides derive that schedule from their own configuration.
//!
//! Neither of them reads the shape of a run from a proof.
//!
//! A batch of instances is described the same way as a single one.
//!
//! A single run is the batch of size one, so the two paths share every step.
//!
//! # Shape
//!
//! ```text
//!     initial commitments      one opaque value per instance, when STIR commits
//!
//!     per global round:
//!       folding grinding       only when the shared difficulty is positive
//!       per active instance:   fold challenge
//!       per active instance:   folded-oracle commitment
//!       per active instance:   out-of-domain points, their answers
//!       query grinding         only when the shared difficulty is positive
//!       per active instance:   combination challenge, query indices
//!       per active instance:   answer polynomial, its consistency challenge
//!
//!     final folding grinding   only when the shared difficulty is positive
//!     per instance:            final fold challenge
//!     per instance:            final polynomial
//!     final query grinding     only when the shared difficulty is positive
//!     per instance:            final query indices
//! ```
//!
//! # Batching
//!
//! Instances are right-aligned, so every one reaches its final round together.
//!
//! ```text
//!     global round  0     1     2
//!     instance A    -     A0    A1
//!     instance B    B0    B1    B2
//! ```
//!
//! A grind is shared by the instances active at its site.
//!
//! Its difficulty is the largest any of them asks for.
//!
//! All challenges covered by a shared grind precede every prover response at that site.
//! Interleaving responses would let a prover resample later challenges without new work.
//!
//! Each instance's block inside a phase is bracketed by its own container.
//!
//! Without the brackets, two blocks of one step flatten into one block of two.
//!
//! The fingerprint could then not tell two batchings of the same instances apart.
//!
//! # What is bound
//!
//! - Shape: round counts, grinding difficulties, query counts and widths, OOD counts.
//! - Shape: which instances are active at each global round, through the containers.
//! - Instance label: the rate, the proximity-gap regime, and the security target.
//! - Instance label: every folding factor, every domain, and every round's degree.
//! - Nothing: a commitment's width, which this layer cannot see.
//!
//! The rate and the regime reach the step sequence only through their query counts.
//!
//! Two parameter sets that happen to agree on those counts share a fingerprint.
//!
//! So the instance label carries both of them directly.
//!
//! # What is not bound
//!
//! The compact-answers option is absent from the shape, and deliberately so.
//!
//! It chooses how the answer polynomial travels, not what enters the sponge.
//!
//! The verifier rebuilds exactly the coefficients the prover absorbed.
//!
//! ```text
//!     sent      -> absorb the coefficients the proof carries
//!     compact   -> absorb the coefficients rebuilt from the same points and values
//! ```
//!
//! Two sides configured differently already fail.
//!
//! The answer step binds its own length.
//!
//! An empty vector against a full one therefore parts the two sponges on the spot.
//!
//! Leaving it out is what keeps the two encodings of one run comparable.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptBound, VerifierState,
};
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger, GrindingChallenger,
};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_security::whir::SecurityAssumption;
use thiserror::Error;

use crate::config::StirConfig;
use crate::error::{GrindStage, ProofShapeError, RoundLabel, StirError};
use crate::utils::OodFilter;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-stir";

/// Step label of an initial-oracle commitment.
const INITIAL_COMMITMENT: &str = "initial_commitment";

/// Step label of the grinding step guarding a round's folding challenge.
const FOLDING_POW: &str = "folding_pow";

/// Container label of one instance's folding-challenge block.
const FOLD_BLOCK: &str = "instance_fold";

/// Container label of one instance's folded-oracle commitment block.
const FOLD_COMMITMENT_BLOCK: &str = "instance_fold_commitment";

/// Step label of a round's folding challenge.
const FOLD_CHALLENGE: &str = "fold_challenge";

/// Step label of a folded-oracle commitment.
const ROUND_COMMITMENT: &str = "round_commitment";

/// Container label of one instance's out-of-domain block.
const OOD_BLOCK: &str = "instance_ood";

/// Step label of a round's out-of-domain points.
const OOD_POINTS: &str = "ood_points";

/// Step label of a round's out-of-domain answers.
const OOD_ANSWERS: &str = "ood_answers";

/// Step label of the grinding step guarding a round's query phase.
const QUERY_POW: &str = "query_pow";

/// Container label of one instance's query block.
const QUERY_BLOCK: &str = "instance_query";

/// Step label of a round's random-combination challenge.
const COMBINATION_CHALLENGE: &str = "combination_challenge";

/// Step label of a round's query indices.
const QUERY_INDICES: &str = "query_indices";

/// Container label of one instance's answer block.
const ANSWER_BLOCK: &str = "instance_answer";

/// Step label of a round's answer polynomial.
const ANS_POLYNOMIAL: &str = "ans_polynomial";

/// Step label of a round's answer-consistency challenge.
const ANS_CHALLENGE: &str = "ans_challenge";

/// Step label of the grinding step guarding the final folding challenge.
const FINAL_FOLDING_POW: &str = "final_folding_pow";

/// Container label of one instance's final folding-challenge block.
const FINAL_FOLD_BLOCK: &str = "instance_final_fold";

/// Container label of one instance's final-polynomial block.
const FINAL_POLYNOMIAL_BLOCK: &str = "instance_final_polynomial";

/// Step label of the final folding challenge.
const FINAL_FOLD_CHALLENGE: &str = "final_fold_challenge";

/// Step label of the final polynomial.
const FINAL_POLYNOMIAL: &str = "final_polynomial";

/// Step label of the grinding step guarding the final query indices.
const FINAL_POW: &str = "final_pow";

/// Container label of one instance's final query block.
const FINAL_QUERY_BLOCK: &str = "instance_final_query";

/// Step label of the final query indices.
const FINAL_QUERY_INDICES: &str = "final_query_indices";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type naming every per-instance container at the type level.
///
/// The name is compared locally when a closer meets its opener.
/// It never reaches the pattern fingerprint.
type Block = ();

/// Append `value` as eight big-endian bytes.
fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Stable byte tag of a proximity-gap regime.
const fn soundness_tag(assumption: SecurityAssumption) -> u64 {
    match assumption {
        SecurityAssumption::UniqueDecoding => 0,
        SecurityAssumption::JohnsonBound => 1,
        SecurityAssumption::CapacityBound => 2,
    }
}

/// Numbers that fix the transcript of one intermediate STIR round.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirRoundShape {
    /// Grinding difficulty this round asks for before its folding challenge.
    pub folding_pow_bits: usize,
    /// Number of out-of-domain points drawn, and of answers sent back.
    pub num_ood_samples: usize,
    /// Grinding difficulty this round asks for before its query phase.
    pub pow_bits: usize,
    /// Number of query indices drawn.
    pub num_queries: usize,
    /// Bit width of each query index, which is the log of the fold-query domain.
    pub log_fold_domain_size: usize,
    /// Log of the degree bound the round's witness respects.
    ///
    /// Reaches the seed through the instance label.
    pub log_degree: usize,
    /// Log of the domain the round's oracle lives on.
    ///
    /// Reaches the seed through the instance label.
    pub log_domain_size: usize,
    /// Log of the folding arity this round collapses by.
    ///
    /// Reaches the seed through the instance label.
    pub log_folding_factor: usize,
    /// Canonical representative of the coset shift of the round's domain.
    ///
    /// Reaches the seed through the instance label.
    pub domain_shift: u64,
    /// Bit pattern of the round's proximity gap `eta`.
    ///
    /// Reaches the seed through the instance label.
    pub eta_bits: u64,
}

impl StirRoundShape {
    /// Largest number of points the round's answer polynomial can interpolate.
    ///
    /// The point set is the OOD points together with the distinct query points.
    ///
    /// ```text
    ///     |P| = num_ood_samples + |distinct query indices| <= num_ood_samples + num_queries
    /// ```
    ///
    /// Deduplication is settled by the draw, so only the cap is known up front.
    #[must_use]
    pub const fn max_ans_len(&self) -> usize {
        self.num_ood_samples + self.num_queries
    }
}

/// Numbers that fix the transcript of one STIR instance.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirInstanceShape {
    /// One entry per intermediate round, in round order.
    pub rounds: Vec<StirRoundShape>,
    /// Grinding difficulty asked for before the final folding challenge.
    pub final_folding_pow_bits: usize,
    /// Coefficient count of the final polynomial.
    pub final_poly_len: usize,
    /// Grinding difficulty asked for before the final query indices.
    pub final_pow_bits: usize,
    /// Number of final-round query indices drawn.
    pub final_queries: usize,
    /// Bit width of each final-round query index.
    pub final_log_domain_size: usize,
    /// Log of the degree bound of the starting witness.
    ///
    /// Reaches the seed through the instance label.
    pub log_starting_degree: usize,
    /// Log of the inverse rate of every Reed-Solomon code in the run.
    ///
    /// Reaches the seed through the instance label.
    pub log_blowup: usize,
    /// Log of the steady-state folding arity.
    ///
    /// Reaches the seed through the instance label.
    pub log_folding_factor: usize,
    /// Log of the folding arity of the first fold.
    ///
    /// Reaches the seed through the instance label.
    pub log_starting_folding_factor: usize,
    /// Log of the degree bound of the final polynomial.
    ///
    /// Reaches the seed through the instance label.
    pub log_final_degree: usize,
    /// Security target the schedule was solved for, in bits.
    ///
    /// Reaches the seed through the instance label.
    pub security_level: usize,
    /// Largest grinding difficulty the schedule may spend at any one site.
    ///
    /// Reaches the seed through the instance label.
    pub max_pow_bits: usize,
    /// Proximity-gap regime the soundness argument is stated in.
    ///
    /// Reaches the seed through the instance label.
    pub soundness_type: SecurityAssumption,
    /// Bit pattern of the final round's proximity gap `eta`.
    ///
    /// Reaches the seed through the instance label.
    pub final_eta_bits: u64,
    /// Cap that stopped the folding schedule, when one was set.
    ///
    /// Reaches the seed through the instance label.
    pub max_log_final_poly_len: Option<usize>,
}

impl StirInstanceShape {
    /// Derive one instance's shape from its configuration.
    #[must_use]
    pub fn new<F, EF, M, C>(config: &StirConfig<F, EF, M, C>) -> Self
    where
        F: TwoAdicField + PrimeField64,
        EF: ExtensionField<F>,
        M: Mmcs<EF>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Every round shrinks its domain by exactly one, so the geometry is a
        // function of the starting size and the round index.
        //
        //     round r:  log_domain = L0 - r,  log_fold_domain = L0 - r - arity_r
        let log_starting_domain = config.log_starting_domain_size();
        let rounds = config
            .round_configs
            .iter()
            .enumerate()
            .map(|(round, rc)| StirRoundShape {
                folding_pow_bits: rc.folding_pow_bits,
                num_ood_samples: rc.num_ood_samples,
                pow_bits: rc.pow_bits,
                num_queries: rc.num_queries,
                log_fold_domain_size: log_starting_domain - round - rc.log_folding_factor,
                log_degree: rc.log_degree,
                log_domain_size: rc.log_domain_size,
                log_folding_factor: rc.log_folding_factor,
                domain_shift: rc.domain_shift.as_canonical_u64(),
                eta_bits: rc.eta.to_bits(),
            })
            .collect();

        let options = config.options();
        Self {
            rounds,
            final_folding_pow_bits: config.final_folding_pow_bits,
            final_poly_len: config.final_poly_len(),
            final_pow_bits: config.final_pow_bits,
            final_queries: config.final_queries,
            final_log_domain_size: log_starting_domain
                - config.num_rounds()
                - config.final_log_folding_factor(),
            log_starting_degree: config.log_starting_degree,
            log_blowup: config.log_blowup,
            log_folding_factor: config.log_folding_factor,
            log_starting_folding_factor: config.log_starting_folding_factor,
            log_final_degree: config.log_final_degree,
            security_level: config.security_level,
            max_pow_bits: config.max_pow_bits,
            soundness_type: config.soundness_type,
            final_eta_bits: config.final_eta.to_bits(),
            max_log_final_poly_len: options.max_log_final_poly_len,
        }
    }

    /// Number of intermediate rounds this instance runs.
    #[must_use]
    pub const fn num_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Every number of this instance, packed for the instance label.
    ///
    /// Each value is eight big-endian bytes, so the packing is self-delimiting.
    fn label_bytes(&self) -> Vec<u8> {
        // Seventeen instance-wide values, then ten per round.
        let mut out = Vec::with_capacity(8 * (17 + 10 * self.rounds.len()));
        for value in [
            self.rounds.len(),
            self.final_folding_pow_bits,
            self.final_poly_len,
            self.final_pow_bits,
            self.final_queries,
            self.final_log_domain_size,
            self.log_starting_degree,
            self.log_blowup,
            self.log_folding_factor,
            self.log_starting_folding_factor,
            self.log_final_degree,
            self.security_level,
            self.max_pow_bits,
        ] {
            push_u64(&mut out, value as u64);
        }
        push_u64(&mut out, soundness_tag(self.soundness_type));
        push_u64(&mut out, self.final_eta_bits);
        // An absent cap and a cap of zero are different schedules, so the flag
        // is carried alongside the value rather than folded into it.
        push_u64(&mut out, u64::from(self.max_log_final_poly_len.is_some()));
        push_u64(&mut out, self.max_log_final_poly_len.unwrap_or(0) as u64);

        for round in &self.rounds {
            for value in [
                round.folding_pow_bits,
                round.num_ood_samples,
                round.pow_bits,
                round.num_queries,
                round.log_fold_domain_size,
                round.log_degree,
                round.log_domain_size,
                round.log_folding_factor,
            ] {
                push_u64(&mut out, value as u64);
            }
            push_u64(&mut out, round.domain_shift);
            push_u64(&mut out, round.eta_bits);
        }
        out
    }
}

/// Numbers that fix the transcript of one batched STIR run.
///
/// A run of a single instance is the batch of size one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StirShape {
    /// Whether STIR commits and absorbs each instance's initial oracle.
    ///
    /// False when the caller has already bound those codewords itself.
    pub commits_initial: bool,
    /// One entry per instance, in batch order.
    pub instances: Vec<StirInstanceShape>,
}

impl StirShape {
    /// Derive the shape of one batched run from its configurations.
    ///
    /// # Arguments
    ///
    /// - `configs`: one configuration per instance, in batch order.
    /// - `commits_initial`: whether STIR commits the initial oracles itself.
    #[must_use]
    pub fn new<F, EF, M, C>(configs: &[&StirConfig<F, EF, M, C>], commits_initial: bool) -> Self
    where
        F: TwoAdicField + PrimeField64,
        EF: ExtensionField<F>,
        M: Mmcs<EF>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        Self {
            commits_initial,
            instances: configs
                .iter()
                .map(|config| StirInstanceShape::new(config))
                .collect(),
        }
    }

    /// Derive the shape of a single-instance run from its configuration.
    #[must_use]
    pub fn single<F, EF, M, C>(config: &StirConfig<F, EF, M, C>, commits_initial: bool) -> Self
    where
        F: TwoAdicField + PrimeField64,
        EF: ExtensionField<F>,
        M: Mmcs<EF>,
        C: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        Self::new(&[config], commits_initial)
    }

    /// Number of instances in the batch.
    #[must_use]
    pub const fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Number of global rounds, which is the deepest instance's round count.
    #[must_use]
    pub fn max_rounds(&self) -> usize {
        self.instances
            .iter()
            .map(StirInstanceShape::num_rounds)
            .max()
            .unwrap_or(0)
    }

    /// Global round at which instance `instance` starts.
    ///
    /// Right-alignment puts every instance's final round on the same global step.
    #[must_use]
    pub fn offset(&self, instance: usize) -> usize {
        self.max_rounds() - self.instances[instance].num_rounds()
    }

    /// Local round index of instance `instance` at global round `round`.
    ///
    /// # Panics
    ///
    /// When the instance is not active at that global round.
    #[must_use]
    pub fn local_round(&self, round: usize, instance: usize) -> usize {
        round
            .checked_sub(self.offset(instance))
            .expect("the instance is active at this global round")
    }

    /// Instances active at global round `round`, in batch order.
    #[must_use]
    pub fn active(&self, round: usize) -> Vec<usize> {
        (0..self.num_instances())
            .filter(|&instance| self.offset(instance) <= round)
            .collect()
    }

    /// The round shape instance `instance` plays at global round `round`.
    #[must_use]
    pub fn round(&self, round: usize, instance: usize) -> &StirRoundShape {
        &self.instances[instance].rounds[self.local_round(round, instance)]
    }

    /// Grinding difficulty of the folding site of global round `round`.
    ///
    /// One grind covers every active instance, at the largest difficulty any asks for.
    #[must_use]
    pub fn folding_pow_bits(&self, round: usize) -> usize {
        self.active(round)
            .into_iter()
            .map(|instance| self.round(round, instance).folding_pow_bits)
            .max()
            .unwrap_or(0)
    }

    /// Grinding difficulty of the query site of global round `round`.
    #[must_use]
    pub fn query_pow_bits(&self, round: usize) -> usize {
        self.active(round)
            .into_iter()
            .map(|instance| self.round(round, instance).pow_bits)
            .max()
            .unwrap_or(0)
    }

    /// Grinding difficulty of the final folding site.
    #[must_use]
    pub fn final_folding_pow_bits(&self) -> usize {
        self.instances
            .iter()
            .map(|instance| instance.final_folding_pow_bits)
            .max()
            .unwrap_or(0)
    }

    /// Grinding difficulty of the final query site.
    #[must_use]
    pub fn final_pow_bits(&self) -> usize {
        self.instances
            .iter()
            .map(|instance| instance.final_pow_bits)
            .max()
            .unwrap_or(0)
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// Every container opened below is closed two steps later, in the same call.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut steps = Vec::new();

        // The initial oracles are bound before any challenge depends on them.
        if self.commits_initial {
            for _ in &self.instances {
                steps.push(Interaction::opaque(
                    Hierarchy::Atomic,
                    Kind::Message,
                    INITIAL_COMMITMENT,
                    Length::Scalar,
                ));
            }
        }

        for round in 0..self.max_rounds() {
            let active = self.active(round);

            // No prover message may separate this grind from any challenge it protects.
            push_pow::<F>(&mut steps, FOLDING_POW, self.folding_pow_bits(round));

            for _ in &active {
                open(&mut steps, FOLD_BLOCK);
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    FOLD_CHALLENGE,
                    Length::Scalar,
                ));
                close(&mut steps, FOLD_BLOCK);
            }
            for _ in &active {
                open(&mut steps, FOLD_COMMITMENT_BLOCK);
                steps.push(Interaction::opaque(
                    Hierarchy::Atomic,
                    Kind::Message,
                    ROUND_COMMITMENT,
                    Length::Scalar,
                ));
                close(&mut steps, FOLD_COMMITMENT_BLOCK);
            }

            for &instance in &active {
                let shape = self.round(round, instance);
                open(&mut steps, OOD_BLOCK);
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    OOD_POINTS,
                    Length::Fixed(shape.num_ood_samples),
                ));
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    OOD_ANSWERS,
                    Length::Fixed(shape.num_ood_samples),
                ));
                close(&mut steps, OOD_BLOCK);
            }

            push_pow::<F>(&mut steps, QUERY_POW, self.query_pow_bits(round));

            for &instance in &active {
                let shape = self.round(round, instance);
                open(&mut steps, QUERY_BLOCK);
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    COMBINATION_CHALLENGE,
                    Length::Scalar,
                ));
                // Query indices are drawn without modular bias, which the tag records.
                steps.push(Interaction::uniform_bits(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    QUERY_INDICES,
                    shape.log_fold_domain_size,
                    Length::Fixed(shape.num_queries),
                ));
                close(&mut steps, QUERY_BLOCK);
            }

            for &instance in &active {
                let shape = self.round(round, instance);
                open(&mut steps, ANSWER_BLOCK);
                // The point set is deduplicated by the draw, so only its cap is known here.
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    ANS_POLYNOMIAL,
                    Length::Bounded(shape.max_ans_len()),
                ));
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    ANS_CHALLENGE,
                    Length::Scalar,
                ));
                close(&mut steps, ANSWER_BLOCK);
            }
        }

        push_pow::<F>(&mut steps, FINAL_FOLDING_POW, self.final_folding_pow_bits());

        for _ in &self.instances {
            open(&mut steps, FINAL_FOLD_BLOCK);
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FINAL_FOLD_CHALLENGE,
                Length::Scalar,
            ));
            close(&mut steps, FINAL_FOLD_BLOCK);
        }
        for instance in &self.instances {
            open(&mut steps, FINAL_POLYNOMIAL_BLOCK);
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                FINAL_POLYNOMIAL,
                Length::Fixed(instance.final_poly_len),
            ));
            close(&mut steps, FINAL_POLYNOMIAL_BLOCK);
        }

        push_pow::<F>(&mut steps, FINAL_POW, self.final_pow_bits());

        for instance in &self.instances {
            open(&mut steps, FINAL_QUERY_BLOCK);
            steps.push(Interaction::uniform_bits(
                Hierarchy::Atomic,
                Kind::Challenge,
                FINAL_QUERY_INDICES,
                instance.final_log_domain_size,
                Length::Fixed(instance.final_queries),
            ));
            close(&mut steps, FINAL_QUERY_BLOCK);
        }

        InteractionPattern::new(steps).expect("every container opened here is closed here")
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

        // Batch header: how many instances, and who commits the initial oracles.
        let mut header = Vec::with_capacity(16);
        push_u64(&mut header, self.instances.len() as u64);
        push_u64(&mut header, u64::from(self.commits_initial));
        separator.instance(&header);

        // One delimited chunk per instance, so two batchings never collapse into one.
        for instance in &self.instances {
            separator.instance(&instance.label_bytes());
        }

        separator
    }
}

/// Append a grinding step, unless the site asks for no work at all.
///
/// A zero-difficulty grind absorbs nothing on either side, so it is not a step.
///
/// The witness lives in the base field, which is what both drivers record.
fn push_pow<F: PrimeField64>(steps: &mut Vec<Interaction>, label: &'static str, bits: usize) {
    if bits > 0 {
        steps.push(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            label,
            Length::Fixed(bits),
        ));
    }
}

/// Append the opener of one instance's block.
fn open(steps: &mut Vec<Interaction>, label: &'static str) {
    steps.push(Interaction::marker::<Block>(
        Hierarchy::Begin,
        Kind::Protocol,
        label,
    ));
}

/// Append the closer of one instance's block.
fn close(steps: &mut Vec<Interaction>, label: &'static str) {
    steps.push(Interaction::marker::<Block>(
        Hierarchy::End,
        Kind::Protocol,
        label,
    ));
}

/// A transcript step the proof failed to satisfy.
///
/// Every variant names the round it came from.
///
/// A count carries both the number the run was described with and the number supplied.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum TranscriptFailure {
    /// A grinding witness did not meet the difficulty its step requires.
    #[error("{round}: {stage} proof-of-work witness clears fewer than {bits} bits")]
    PowWitness {
        /// Round whose grinding site rejected the witness.
        round: RoundLabel,
        /// Site inside that round the grind guards.
        stage: GrindStage,
        /// Difficulty the site requires, in bits.
        bits: usize,
    },
    /// An answer polynomial carries more coefficients than its step accepts.
    #[error("{round}: ans polynomial holds at most {maximum} coefficients, got {got}")]
    AnsPolynomialTooLong {
        /// Round whose answer polynomial was rejected.
        round: RoundLabel,
        /// Largest coefficient count the step accepts.
        maximum: usize,
        /// Coefficient count the run supplied.
        got: usize,
    },
    /// A round's out-of-domain answers do not match the described count.
    #[error("{round}: expected {expected} out-of-domain answers, got {got}")]
    OodAnswerCount {
        /// Round whose answers were rejected.
        round: RoundLabel,
        /// Count the run was described with.
        expected: usize,
        /// Count the proof carries.
        got: usize,
    },
    /// The final polynomial carries a coefficient count the run never described.
    #[error("{round}: expected {expected} final coefficients, got {got}")]
    FinalPolynomialLength {
        /// Round whose polynomial was rejected, always the final one.
        round: RoundLabel,
        /// Count the run was described with.
        expected: usize,
        /// Count the proof carries.
        got: usize,
    },
}

impl<MmcsError, InputError> From<TranscriptFailure> for StirError<MmcsError, InputError> {
    fn from(failure: TranscriptFailure) -> Self {
        match failure {
            TranscriptFailure::PowWitness { round, stage, bits } => {
                Self::InvalidPowWitness { round, stage, bits }
            }
            TranscriptFailure::AnsPolynomialTooLong {
                round,
                maximum,
                got,
            } => ProofShapeError::AnsPolynomialTooLong {
                round,
                maximum,
                got,
            }
            .into(),
            TranscriptFailure::OodAnswerCount {
                round,
                expected,
                got,
            } => ProofShapeError::OodAnswerCount {
                round,
                expected,
                got,
            }
            .into(),
            TranscriptFailure::FinalPolynomialLength {
                round: _,
                expected,
                got,
            } => ProofShapeError::FinalPolynomialLength { expected, got }.into(),
        }
    }
}

/// Prover-side transcript of one batched STIR run.
///
/// Holds the only definition of what a prover writes at each phase.
///
/// The challenger is borrowed, not consumed.
/// STIR runs inside a larger protocol whose transcript continues afterwards.
pub struct ProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: StirShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> ProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F>
        + CanSample<F>
        + CanSampleBits<usize>
        + CanSampleUniformBits<F>
        + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: StirShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this run was described with.
    pub const fn shape(&self) -> &StirShape {
        &self.shape
    }

    /// Bind one instance's initial-oracle commitment.
    ///
    /// Called once per instance, in batch order, and only when STIR commits them.
    pub fn initial_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(INITIAL_COMMITMENT, commitment);
    }

    /// Grind the folding site of global round `round`.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn folding_pow(&mut self, round: usize) -> F {
        let bits = self.shape.folding_pow_bits(round);
        if bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(FOLDING_POW, bits)
    }

    /// Draw one instance's folding challenge before any folded-oracle commitment.
    pub fn fold_challenge(&mut self) -> EF {
        self.state.begin_protocol::<Block>(FOLD_BLOCK);
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(FOLD_CHALLENGE)
            .into_inner();
        self.state.end_protocol::<Block>(FOLD_BLOCK);
        challenge
    }

    /// Bind one folded-oracle commitment after every active folding challenge.
    pub fn fold_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.begin_protocol::<Block>(FOLD_COMMITMENT_BLOCK);
        self.state.observe_opaque(ROUND_COMMITMENT, commitment);
        self.state.end_protocol::<Block>(FOLD_COMMITMENT_BLOCK);
    }

    /// Open one instance's out-of-domain block and draw its points.
    ///
    /// The answer step that follows closes the block.
    ///
    /// The points avoid the round's three excluded domains and each other.
    pub fn ood_points(&mut self, round: usize, instance: usize, filter: &OodFilter<F>) -> Vec<EF> {
        let count = self.shape.round(round, instance).num_ood_samples;
        self.state.begin_protocol::<Block>(OOD_BLOCK);
        self.state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                OOD_POINTS,
                count,
                |candidate, kept| filter.accepts(candidate, kept),
            )
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Bind the out-of-domain answers and close the block.
    pub fn ood_answers(&mut self, answers: &[EF]) {
        let _bound = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OOD_ANSWERS, answers);
        self.state.end_protocol::<Block>(OOD_BLOCK);
    }

    /// Grind the query site of global round `round`.
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

    /// Play one instance's query block: the combination challenge, then the indices.
    ///
    /// Nothing is absorbed between the two, so they form one block.
    ///
    /// # Returns
    ///
    /// - The random-combination challenge.
    /// - Every query index, in draw order, repeats included.
    pub fn query_phase(&mut self, round: usize, instance: usize) -> (EF, Vec<usize>) {
        let shape = self.shape.round(round, instance).clone();
        self.state.begin_protocol::<Block>(QUERY_BLOCK);
        let r_comb = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COMBINATION_CHALLENGE)
            .into_inner();
        let indices = self
            .state
            .challenge_uniform_bits::<F>(
                QUERY_INDICES,
                shape.log_fold_domain_size,
                shape.num_queries,
            )
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        self.state.end_protocol::<Block>(QUERY_BLOCK);
        (r_comb, indices)
    }

    /// Play one instance's answer block: bind the answer polynomial, draw its challenge.
    ///
    /// The challenge is drawn and dropped here.
    ///
    /// Only the verifier's interpolation check reads its value.
    ///
    /// The recorded step is what keeps the two sides playing it in lockstep.
    ///
    /// # Panics
    ///
    /// When the answer polynomial is longer than the round's point set can be.
    pub fn answer_phase(&mut self, round: usize, instance: usize, ans: &[EF]) {
        let max_ans_len = self.shape.round(round, instance).max_ans_len();
        self.state.begin_protocol::<Block>(ANSWER_BLOCK);
        let _bound = self
            .state
            .observe_extensions_bounded::<F, EF, FieldToFieldCodec<F>>(
                ANS_POLYNOMIAL,
                ans,
                max_ans_len,
            );
        let _rho = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ANS_CHALLENGE);
        self.state.end_protocol::<Block>(ANSWER_BLOCK);
    }

    /// Grind the final folding site.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn final_folding_pow(&mut self) -> F {
        let bits = self.shape.final_folding_pow_bits();
        if bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(FINAL_FOLDING_POW, bits)
    }

    /// Draw one instance's final folding challenge before any final polynomial.
    pub fn final_fold_challenge(&mut self) -> EF {
        self.state.begin_protocol::<Block>(FINAL_FOLD_BLOCK);
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(FINAL_FOLD_CHALLENGE)
            .into_inner();
        self.state.end_protocol::<Block>(FINAL_FOLD_BLOCK);
        challenge
    }

    /// Bind one final polynomial after every final folding challenge.
    pub fn final_polynomial(&mut self, final_poly: &[EF]) {
        self.state.begin_protocol::<Block>(FINAL_POLYNOMIAL_BLOCK);
        let _bound = self
            .state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLYNOMIAL, final_poly);
        self.state.end_protocol::<Block>(FINAL_POLYNOMIAL_BLOCK);
    }

    /// Grind the final query site.
    ///
    /// # Returns
    ///
    /// The witness the search found, or zero when the site asks for no work.
    pub fn final_pow(&mut self) -> F {
        let bits = self.shape.final_pow_bits();
        if bits == 0 {
            return F::ZERO;
        }
        self.state.observe_pow(FINAL_POW, bits)
    }

    /// Draw one instance's final query indices.
    pub fn final_query_indices(&mut self, instance: usize) -> Vec<usize> {
        let shape = &self.shape.instances[instance];
        let (width, count) = (shape.final_log_domain_size, shape.final_queries);
        self.state.begin_protocol::<Block>(FINAL_QUERY_BLOCK);
        let indices = self
            .state
            .challenge_uniform_bits::<F>(FINAL_QUERY_INDICES, width, count)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        self.state.end_protocol::<Block>(FINAL_QUERY_BLOCK);
        indices
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "STIR carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one batched STIR run.
///
/// Mirrors the prover side call for call, over the same description.
pub struct VerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: StirShape,
    /// Marker for the extension field the challenges live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> VerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F>
        + CanSample<F>
        + CanSampleBits<usize>
        + CanSampleUniformBits<F>
        + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: StirShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Read-only access to the numbers this run was described with.
    pub const fn shape(&self) -> &StirShape {
        &self.shape
    }

    /// Bind one instance's initial-oracle commitment.
    pub fn initial_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.observe_opaque(INITIAL_COMMITMENT, commitment);
    }

    /// Replay the folding grind of global round `round`.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn folding_pow(&mut self, round: usize, witness: F) -> Result<(), TranscriptFailure> {
        let bits = self.shape.folding_pow_bits(round);
        let site = (RoundLabel::Round(round), GrindStage::Folding);
        self.replay_pow(FOLDING_POW, bits, witness, site)
    }

    /// Draw one instance's folding challenge before any folded-oracle commitment.
    pub fn fold_challenge(&mut self) -> EF {
        self.state.begin_protocol::<Block>(FOLD_BLOCK);
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(FOLD_CHALLENGE)
            .into_inner();
        self.state.end_protocol::<Block>(FOLD_BLOCK);
        challenge
    }

    /// Bind one folded-oracle commitment after every active folding challenge.
    pub fn fold_commitment<Com>(&mut self, commitment: Com)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        self.state.begin_protocol::<Block>(FOLD_COMMITMENT_BLOCK);
        self.state.observe_opaque(ROUND_COMMITMENT, commitment);
        self.state.end_protocol::<Block>(FOLD_COMMITMENT_BLOCK);
    }

    /// Open one instance's out-of-domain block and redraw its points.
    ///
    /// The answer step that follows closes the block.
    pub fn ood_points(&mut self, round: usize, instance: usize, filter: &OodFilter<F>) -> Vec<EF> {
        let count = self.shape.round(round, instance).num_ood_samples;
        self.state.begin_protocol::<Block>(OOD_BLOCK);
        self.state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                OOD_POINTS,
                count,
                |candidate, kept| filter.accepts(candidate, kept),
            )
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect()
    }

    /// Bind the out-of-domain answers and close the block.
    ///
    /// # Errors
    ///
    /// When the answer count differs from the one the round was described with.
    pub fn ood_answers(
        &mut self,
        round: usize,
        instance: usize,
        answers: &[EF],
    ) -> Result<(), TranscriptFailure> {
        let local = self.shape.local_round(round, instance);
        let expected = self.shape.round(round, instance).num_ood_samples;
        // The count is also checked against the proof before the transcript is built.
        // Rejecting here as well keeps this method safe to call on its own.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(OOD_ANSWERS, answers)
            .map_err(|_| TranscriptFailure::OodAnswerCount {
                round: RoundLabel::Round(local),
                expected,
                got: answers.len(),
            })?;
        self.state.end_protocol::<Block>(OOD_BLOCK);
        Ok(())
    }

    /// Replay the query grind of global round `round`.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn query_pow(&mut self, round: usize, witness: F) -> Result<(), TranscriptFailure> {
        let bits = self.shape.query_pow_bits(round);
        let site = (RoundLabel::Round(round), GrindStage::Query);
        self.replay_pow(QUERY_POW, bits, witness, site)
    }

    /// Play one instance's query block: the combination challenge, then the indices.
    ///
    /// # Returns
    ///
    /// - The random-combination challenge.
    /// - Every query index, in draw order, repeats included.
    pub fn query_phase(&mut self, round: usize, instance: usize) -> (EF, Vec<usize>) {
        let shape = self.shape.round(round, instance).clone();
        self.state.begin_protocol::<Block>(QUERY_BLOCK);
        let r_comb = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COMBINATION_CHALLENGE)
            .into_inner();
        let indices = self
            .state
            .challenge_uniform_bits::<F>(
                QUERY_INDICES,
                shape.log_fold_domain_size,
                shape.num_queries,
            )
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        self.state.end_protocol::<Block>(QUERY_BLOCK);
        (r_comb, indices)
    }

    /// Play one instance's answer block: bind the answer polynomial, draw its challenge.
    ///
    /// # Returns
    ///
    /// The consistency challenge the interpolation identity is checked at.
    ///
    /// # Errors
    ///
    /// When the answer polynomial is longer than the round's point set can be.
    pub fn answer_phase(
        &mut self,
        round: usize,
        instance: usize,
        ans: &[EF],
    ) -> Result<EF, TranscriptFailure> {
        let local = self.shape.local_round(round, instance);
        let max_ans_len = self.shape.round(round, instance).max_ans_len();
        self.state.begin_protocol::<Block>(ANSWER_BLOCK);
        self.state
            .observe_extensions_bounded::<F, EF, FieldToFieldCodec<F>>(
                ANS_POLYNOMIAL,
                ans,
                max_ans_len,
            )
            .map_err(|_| TranscriptFailure::AnsPolynomialTooLong {
                round: RoundLabel::Round(local),
                maximum: max_ans_len,
                got: ans.len(),
            })?;
        let rho = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ANS_CHALLENGE)
            .into_inner();
        self.state.end_protocol::<Block>(ANSWER_BLOCK);
        Ok(rho)
    }

    /// Replay the final folding grind.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn final_folding_pow(&mut self, witness: F) -> Result<(), TranscriptFailure> {
        let bits = self.shape.final_folding_pow_bits();
        let site = (RoundLabel::Final, GrindStage::Folding);
        self.replay_pow(FINAL_FOLDING_POW, bits, witness, site)
    }

    /// Draw one instance's final folding challenge before any final polynomial.
    pub fn final_fold_challenge(&mut self) -> EF {
        self.state.begin_protocol::<Block>(FINAL_FOLD_BLOCK);
        let challenge = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(FINAL_FOLD_CHALLENGE)
            .into_inner();
        self.state.end_protocol::<Block>(FINAL_FOLD_BLOCK);
        challenge
    }

    /// Bind one final polynomial after every final folding challenge.
    ///
    /// # Errors
    ///
    /// When the coefficient count differs from the described one.
    pub fn final_polynomial(
        &mut self,
        instance: usize,
        final_poly: &[EF],
    ) -> Result<(), TranscriptFailure> {
        let expected = self.shape.instances[instance].final_poly_len;
        self.state.begin_protocol::<Block>(FINAL_POLYNOMIAL_BLOCK);
        // The length is also checked against the proof before the transcript is built.
        // Rejecting here as well keeps this method safe to call on its own.
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(FINAL_POLYNOMIAL, final_poly)
            .map_err(|_| TranscriptFailure::FinalPolynomialLength {
                round: RoundLabel::Final,
                expected,
                got: final_poly.len(),
            })?;
        self.state.end_protocol::<Block>(FINAL_POLYNOMIAL_BLOCK);
        Ok(())
    }

    /// Replay the final query grind.
    ///
    /// # Errors
    ///
    /// When the witness misses the difficulty the site requires.
    pub fn final_pow(&mut self, witness: F) -> Result<(), TranscriptFailure> {
        let bits = self.shape.final_pow_bits();
        let site = (RoundLabel::Final, GrindStage::Query);
        self.replay_pow(FINAL_POW, bits, witness, site)
    }

    /// Redraw one instance's final query indices.
    pub fn final_query_indices(&mut self, instance: usize) -> Vec<usize> {
        let shape = &self.shape.instances[instance];
        let (width, count) = (shape.final_log_domain_size, shape.final_queries);
        self.state.begin_protocol::<Block>(FINAL_QUERY_BLOCK);
        let indices = self
            .state
            .challenge_uniform_bits::<F>(FINAL_QUERY_INDICES, width, count)
            .into_iter()
            .map(TranscriptBound::into_inner)
            .collect();
        self.state.end_protocol::<Block>(FINAL_QUERY_BLOCK);
        indices
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// Every path that leaves the transcript early goes through this.
    ///
    /// Dropping an unfinished driver otherwise panics.
    ///
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
            .expect("STIR reads an empty wire, so no bytes can remain");
    }

    /// Replay one grinding step, or skip a site that asks for no work.
    ///
    /// # Errors
    ///
    /// When the witness misses the required difficulty.
    fn replay_pow(
        &mut self,
        label: &'static str,
        bits: usize,
        witness: F,
        site: (RoundLabel, GrindStage),
    ) -> Result<(), TranscriptFailure> {
        if bits == 0 {
            return Ok(());
        }
        let (round, stage) = site;
        // A failed check poisons the driver, so the rejection travels alone.
        self.state
            .observe_pow(label, bits, witness)
            .map_err(|_| TranscriptFailure::PowWitness { round, stage, bits })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    #[cfg(panic = "unwind")]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    /// One named mutation of an instance-wide number.
    type InstanceKnob = (&'static str, fn(&mut StirInstanceShape));

    /// One named mutation of a per-round number.
    type RoundKnob = (&'static str, fn(&mut StirRoundShape));

    /// A commitment shaped like the ones a Merkle scheme hands this layer.
    const COMMITMENT: [F; 8] = [F::ONE; 8];

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x5717);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// The three domains a round's out-of-domain points must miss.
    fn excluded_domains() -> OodFilter<F> {
        OodFilter::new([(F::GENERATOR, 3), (F::GENERATOR, 2), (F::GENERATOR, 1)])
    }

    /// One round with no grinding and small counts.
    fn round_shape() -> StirRoundShape {
        StirRoundShape {
            folding_pow_bits: 0,
            num_ood_samples: 1,
            pow_bits: 0,
            num_queries: 3,
            log_fold_domain_size: 6,
            log_degree: 8,
            log_domain_size: 9,
            log_folding_factor: 3,
            domain_shift: 31,
            eta_bits: 0.25_f64.to_bits(),
        }
    }

    /// One instance running the round above and then the final send.
    fn instance_shape() -> StirInstanceShape {
        StirInstanceShape {
            rounds: vec![round_shape()],
            final_folding_pow_bits: 0,
            final_poly_len: 2,
            final_pow_bits: 0,
            final_queries: 2,
            final_log_domain_size: 5,
            log_starting_degree: 8,
            log_blowup: 1,
            log_folding_factor: 3,
            log_starting_folding_factor: 3,
            log_final_degree: 1,
            security_level: 100,
            max_pow_bits: 20,
            soundness_type: SecurityAssumption::JohnsonBound,
            final_eta_bits: 0.125_f64.to_bits(),
            max_log_final_poly_len: None,
        }
    }

    /// A batch of one, which is what a single STIR run is.
    fn shape() -> StirShape {
        StirShape {
            commits_initial: true,
            instances: vec![instance_shape()],
        }
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: &StirShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F, EF>().seed(&mut challenger);
        challenger.sample()
    }

    #[test]
    fn every_configuration_knob_reaches_the_seed() {
        // Baseline: the seed of the fixture, untouched.
        let base = first_challenge(&shape());

        // Every instance-wide number, bumped by one.
        //
        // Some of these move the step sequence and some do not.
        // The test does not care which: the seed has to move either way.
        let instance_knobs: [InstanceKnob; 13] = [
            ("final_folding_pow_bits", |i| i.final_folding_pow_bits += 1),
            ("final_poly_len", |i| i.final_poly_len += 1),
            ("final_pow_bits", |i| i.final_pow_bits += 1),
            ("final_queries", |i| i.final_queries += 1),
            ("final_log_domain_size", |i| i.final_log_domain_size += 1),
            ("log_starting_degree", |i| i.log_starting_degree += 1),
            ("log_blowup", |i| i.log_blowup += 1),
            ("log_folding_factor", |i| i.log_folding_factor += 1),
            ("log_starting_folding_factor", |i| {
                i.log_starting_folding_factor += 1;
            }),
            ("log_final_degree", |i| i.log_final_degree += 1),
            ("security_level", |i| i.security_level += 1),
            ("max_pow_bits", |i| i.max_pow_bits += 1),
            ("final_eta_bits", |i| i.final_eta_bits += 1),
        ];
        for (name, bump) in instance_knobs {
            let mut moved = shape();
            bump(&mut moved.instances[0]);
            assert_ne!(first_challenge(&moved), base, "{name} misses the seed");
        }

        // Every per-round number, bumped by one in the round the fixture runs.
        let round_knobs: [RoundKnob; 10] = [
            ("folding_pow_bits", |r| r.folding_pow_bits += 1),
            ("num_ood_samples", |r| r.num_ood_samples += 1),
            ("pow_bits", |r| r.pow_bits += 1),
            ("num_queries", |r| r.num_queries += 1),
            ("log_fold_domain_size", |r| r.log_fold_domain_size += 1),
            ("log_degree", |r| r.log_degree += 1),
            ("log_domain_size", |r| r.log_domain_size += 1),
            ("log_folding_factor", |r| r.log_folding_factor += 1),
            ("domain_shift", |r| r.domain_shift += 1),
            ("eta_bits", |r| r.eta_bits += 1),
        ];
        for (name, bump) in round_knobs {
            let mut moved = shape();
            bump(&mut moved.instances[0].rounds[0]);
            assert_ne!(first_challenge(&moved), base, "{name} misses the seed");
        }

        // The proximity-gap regime is an enum, so it is walked by value.
        let mut regime = shape();
        regime.instances[0].soundness_type = SecurityAssumption::CapacityBound;
        assert_ne!(
            first_challenge(&regime),
            base,
            "soundness_type misses the seed"
        );

        // An absent folding cap and a cap of zero are different schedules.
        let mut capped = shape();
        capped.instances[0].max_log_final_poly_len = Some(0);
        assert_ne!(
            first_challenge(&capped),
            base,
            "max_log_final_poly_len misses the seed"
        );

        // Whether STIR commits the initial oracle at all.
        let mut external = shape();
        external.commits_initial = false;
        assert_ne!(
            first_challenge(&external),
            base,
            "commits_initial misses the seed"
        );

        // How many instances share the run.
        let mut batched = shape();
        batched.instances.push(instance_shape());
        assert_ne!(
            first_challenge(&batched),
            base,
            "the instance count misses the seed"
        );

        // A round added to the schedule.
        let mut deeper = shape();
        deeper.instances[0].rounds.push(round_shape());
        assert_ne!(
            first_challenge(&deeper),
            base,
            "the round count misses the seed"
        );
    }

    #[test]
    fn the_rate_and_the_proximity_regime_reach_the_seed() {
        // The soundness argument is stated in terms of these two.
        //
        // Neither appears in the step sequence: they reach it only through the query
        // counts they imply, and two parameter sets can land on the same counts.
        //
        //     same rounds, same queries, same OOD samples, same grinding
        //     different rate, or different regime
        //
        // So the instance label has to carry them directly, and it does.
        let base = first_challenge(&shape());

        let mut cheaper_rate = shape();
        cheaper_rate.instances[0].log_blowup += 1;
        assert_eq!(
            cheaper_rate.pattern::<F, EF>(),
            shape().pattern::<F, EF>(),
            "the rate leaves the step sequence untouched, which is the point"
        );
        assert_ne!(first_challenge(&cheaper_rate), base);

        let mut conjectured = shape();
        conjectured.instances[0].soundness_type = SecurityAssumption::CapacityBound;
        assert_eq!(conjectured.pattern::<F, EF>(), shape().pattern::<F, EF>());
        assert_ne!(first_challenge(&conjectured), base);
    }

    #[test]
    fn a_batch_brackets_every_instance_it_runs() {
        // Two instances share every global round, so their steps interleave phase by phase.
        //
        //     Begin instance_fold             gamma       End
        //     Begin instance_fold             gamma       End
        //     Begin instance_fold_commitment  commitment  End
        //     Begin instance_fold_commitment  commitment  End
        //
        // Flattened, that is one run of four steps with nothing saying where the first
        // instance stops and the second begins.
        let mut batch = shape();
        batch.instances.push(instance_shape());
        let pattern = batch.pattern::<F, EF>();

        let openers = pattern
            .interactions()
            .iter()
            .filter(|step| step.hierarchy() == Hierarchy::Begin)
            .count();
        let closers = pattern
            .interactions()
            .iter()
            .filter(|step| step.hierarchy() == Hierarchy::End)
            .count();
        // Eight blocks per instance: five in the round, three in the final send.
        assert_eq!(openers, 16);
        assert_eq!(openers, closers);

        // The same leaf steps with the brackets removed fingerprint differently.
        let flattened: Vec<Interaction> = pattern
            .interactions()
            .iter()
            .copied()
            .filter(|step| step.hierarchy() == Hierarchy::Atomic)
            .collect();
        let flat = InteractionPattern::new(flattened).expect("leaves alone are well formed");
        assert_ne!(flat.pattern_hash(), pattern.pattern_hash());
    }

    #[test]
    fn a_right_aligned_batch_starts_its_shallow_instance_late() {
        // Instance 0 runs two rounds, instance 1 runs one.
        //
        //     global round  0     1
        //     instance 0    0.0   0.1
        //     instance 1    -     1.0
        //
        // Both reach the final send together, and the shallow one is idle at round 0.
        let mut batch = shape();
        batch.instances[0].rounds.push(round_shape());
        batch.instances.push(instance_shape());

        assert_eq!(batch.max_rounds(), 2);
        assert_eq!(batch.offset(0), 0);
        assert_eq!(batch.offset(1), 1);
        assert_eq!(batch.active(0), vec![0]);
        assert_eq!(batch.active(1), vec![0, 1]);
        assert_eq!(batch.local_round(1, 1), 0);
    }

    #[test]
    fn a_shared_grind_takes_the_largest_difficulty_at_its_site() {
        // Two instances active at the same site, asking for different work.
        //
        //     instance 0 asks for 3 bits
        //     instance 1 asks for 7 bits
        //
        // One grind covers both, so it has to satisfy the stricter of the two.
        let mut batch = shape();
        batch.instances.push(instance_shape());
        batch.instances[0].rounds[0].folding_pow_bits = 3;
        batch.instances[1].rounds[0].folding_pow_bits = 7;
        batch.instances[0].final_pow_bits = 4;
        batch.instances[1].final_pow_bits = 2;

        assert_eq!(batch.folding_pow_bits(0), 7);
        assert_eq!(batch.final_pow_bits(), 4);
    }

    /// A shared grind protects only the challenges before the next prover message.
    fn assert_shared_folding_challenges_are_contiguous(batch: &StirShape, final_round: bool) {
        let pattern = batch.pattern::<F, EF>();
        let steps: Vec<_> = pattern
            .interactions()
            .iter()
            .filter(|step| step.hierarchy() == Hierarchy::Atomic)
            .collect();
        let (pow_label, challenge_label, expected_counts) = if final_round {
            (
                FINAL_FOLDING_POW,
                FINAL_FOLD_CHALLENGE,
                vec![batch.instances.len()],
            )
        } else {
            (
                FOLDING_POW,
                FOLD_CHALLENGE,
                (0..batch.max_rounds())
                    .map(|round| batch.active(round).len())
                    .collect(),
            )
        };
        let covered: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, step)| step.label() == pow_label)
            .map(|(index, _)| {
                steps[index + 1..]
                    .iter()
                    .take_while(|step| step.kind() == Kind::Challenge)
                    .filter(|step| step.label() == challenge_label)
                    .count()
            })
            .collect();
        assert_eq!(
            covered, expected_counts,
            "every folding challenge must precede the first prover response after the shared grind"
        );
    }

    #[test]
    fn shared_round_folding_grinds_cover_every_active_instance() {
        let mut batch = shape();
        batch.instances[0].rounds[0].folding_pow_bits = 3;
        batch.instances.push(batch.instances[0].clone());
        let extra_round = batch.instances[0].rounds[0].clone();
        batch.instances[0].rounds.push(extra_round);
        assert_shared_folding_challenges_are_contiguous(&batch, false);
    }

    #[test]
    fn shared_final_folding_grind_covers_instances_without_intermediate_rounds() {
        let mut batch = shape();
        batch.instances[0].final_folding_pow_bits = 3;
        batch.instances.push(batch.instances[0].clone());
        batch.instances[1].rounds.clear();
        assert_shared_folding_challenges_are_contiguous(&batch, true);
    }

    #[test]
    fn both_sides_draw_the_same_challenges_from_the_same_values() {
        // Described run: one instance, one round, grinding at both of the round's sites.
        let mut shape = shape();
        shape.instances[0].rounds[0].folding_pow_bits = 3;
        shape.instances[0].rounds[0].pow_bits = 2;

        // Values the proof would carry.
        let ood_answers = [EF::TWO];
        let ans = [EF::ONE, EF::TWO];
        let final_poly = [EF::ONE, EF::ZERO];
        let filter = excluded_domains();

        // Prover side: play every phase in order.
        let mut prover_challenger = fresh_challenger();
        let mut prover = ProverTranscript::<Ch, F, EF>::new(&mut prover_challenger, shape.clone());
        prover.initial_commitment(COMMITMENT);
        let folding_witness = prover.folding_pow(0);
        let gamma = prover.fold_challenge();
        prover.fold_commitment(COMMITMENT);
        let ood_points = prover.ood_points(0, 0, &filter);
        prover.ood_answers(&ood_answers);
        let query_witness = prover.query_pow(0);
        let (r_comb, indices) = prover.query_phase(0, 0);
        prover.answer_phase(0, 0, &ans);
        let final_folding_witness = prover.final_folding_pow();
        let final_gamma = prover.final_fold_challenge();
        prover.final_polynomial(&final_poly);
        let final_witness = prover.final_pow();
        let final_indices = prover.final_query_indices(0);
        prover.finish();

        // Verifier side: the same calls, in the same order, over the same values.
        let mut verifier_challenger = fresh_challenger();
        let mut verifier = VerifierTranscript::<Ch, F, EF>::new(&mut verifier_challenger, shape);
        verifier.initial_commitment(COMMITMENT);
        verifier
            .folding_pow(0, folding_witness)
            .expect("the prover's own witness satisfies the site");
        assert_eq!(verifier.fold_challenge(), gamma);
        verifier.fold_commitment(COMMITMENT);
        assert_eq!(verifier.ood_points(0, 0, &filter), ood_points);
        verifier
            .ood_answers(0, 0, &ood_answers)
            .expect("the described answer count");
        verifier
            .query_pow(0, query_witness)
            .expect("the prover's own witness satisfies the site");
        assert_eq!(verifier.query_phase(0, 0), (r_comb, indices));
        let _rho = verifier
            .answer_phase(0, 0, &ans)
            .expect("an answer polynomial inside the cap");
        verifier
            .final_folding_pow(final_folding_witness)
            .expect("the prover's own witness satisfies the site");
        assert_eq!(verifier.final_fold_challenge(), final_gamma);
        verifier
            .final_polynomial(0, &final_poly)
            .expect("the described coefficient count");
        verifier
            .final_pow(final_witness)
            .expect("the prover's own witness satisfies the site");
        assert_eq!(verifier.final_query_indices(0), final_indices);
        verifier.finish();

        // Both sponges land on the same state, so whatever runs next agrees too.
        let prover_next: F = prover_challenger.sample();
        let verifier_next: F = verifier_challenger.sample();
        assert_eq!(prover_next, verifier_next);
    }

    #[test]
    fn a_grinding_witness_that_misses_its_difficulty_is_rejected() {
        // Described run: 20 bits of grinding before the round's folding challenge.
        //
        // Zero is a witness like any other, and it clears 20 bits with probability 2^-20.
        let mut shape = shape();
        shape.instances[0].rounds[0].folding_pow_bits = 20;

        let mut challenger = fresh_challenger();
        let mut verifier = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);
        verifier.initial_commitment(COMMITMENT);

        let err = verifier
            .folding_pow(0, F::ZERO)
            .expect_err("a witness that clears no bits must be rejected");

        assert_eq!(
            err,
            TranscriptFailure::PowWitness {
                round: RoundLabel::Round(0),
                stage: GrindStage::Folding,
                bits: 20,
            }
        );
        // The failed read poisoned the driver, so dropping it here raises nothing.
    }

    #[test]
    fn an_answer_polynomial_above_its_cap_is_rejected() {
        // Described cap: one point per OOD sample plus one per query draw.
        //
        //     cap = 1 + 3 = 4
        //     supplied = 5    -> rejected, nothing absorbed
        let mut challenger = fresh_challenger();
        let mut verifier = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape());
        let filter = excluded_domains();

        verifier.initial_commitment(COMMITMENT);
        verifier
            .folding_pow(0, F::ZERO)
            .expect("no work is asked for");
        let _gamma = verifier.fold_challenge();
        verifier.fold_commitment(COMMITMENT);
        let _ood = verifier.ood_points(0, 0, &filter);
        verifier
            .ood_answers(0, 0, &[EF::ONE])
            .expect("the described answer count");
        verifier
            .query_pow(0, F::ZERO)
            .expect("no work is asked for");
        let _query = verifier.query_phase(0, 0);

        let err = verifier
            .answer_phase(0, 0, &[EF::ONE; 5])
            .expect_err("an answer polynomial outside the cap must error");

        assert_eq!(
            err,
            TranscriptFailure::AnsPolynomialTooLong {
                round: RoundLabel::Round(0),
                maximum: 4,
                got: 5,
            }
        );
    }

    #[test]
    fn an_out_of_domain_answer_of_the_wrong_count_is_rejected() {
        // Described run: exactly one out-of-domain sample.
        //
        //     described:   1
        //     proof holds: 2   -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut verifier = VerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape());
        let filter = excluded_domains();

        verifier.initial_commitment(COMMITMENT);
        verifier
            .folding_pow(0, F::ZERO)
            .expect("no work is asked for");
        let _gamma = verifier.fold_challenge();
        verifier.fold_commitment(COMMITMENT);
        let _ood = verifier.ood_points(0, 0, &filter);

        let err = verifier
            .ood_answers(0, 0, &[EF::ONE, EF::TWO])
            .expect_err("an answer count outside the described one must error");

        assert_eq!(
            err,
            TranscriptFailure::OodAnswerCount {
                round: RoundLabel::Round(0),
                expected: 1,
                got: 2,
            }
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn a_panic_inside_a_live_prover_transcript_unwinds() {
        // Fixture state: a run described with one round, so the transcript is mid-pattern.
        let mut challenger = fresh_challenger();

        // Mutation: the caller panics between the first commitment and `finish`.
        //
        // The DFT calls, the `.expect` on the committed data, and any caller-supplied
        // `Mmcs` callback all sit inside this scope.
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut transcript = ProverTranscript::<Ch, F, EF>::new(&mut challenger, shape());
            transcript.initial_commitment(COMMITMENT);
            let _gamma = transcript.fold_challenge();
            panic!("the commitment scheme panicked");
        }));

        // The completeness check yields, so the caller's panic is what escapes.
        let payload = caught.expect_err("the caller's panic must unwind out of the scope");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("the commitment scheme panicked")
        );
    }
}
