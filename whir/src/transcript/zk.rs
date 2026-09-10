//! Fiat-Shamir transcript of the HVZK-WHIR pipeline.
//!
//! # Overview
//!
//! One statement of what a hiding WHIR run absorbs and draws, consumed by both sides.
//!
//! It is built from the derived HVZK configuration and the trusted opening-claim count.
//!
//! # Shape
//!
//! ```text
//!     initial batching       one extension element
//!     masked sumcheck        one batch over folding_factor(0) rounds
//!     per round:  commitment     new oracle, then its code-switch mask
//!                 out-of-domain  one point drawn, one answer sent, per sample
//!                 grinding       only when the difficulty is positive
//!                 queries        num_queries draws of index_bits bits
//!                 batching       one extension element
//!                 masked sumcheck one batch over folding_factor(round + 1) rounds
//!     base case              fresh commitments, one claim, one blinding challenge
//!                            one reveal pair per committed word
//!                            grinding, source spot checks, mask spot checks
//! ```
//!
//! A masked sumcheck batch opens with four steps.
//!
//! ```text
//!     joint claim  ->  interleaved mask oracle  ->  mu_tilde  ->  eps
//! ```
//!
//! Each of its rounds sends `max(ell_zk, 3) - 1` wire coefficients.
//!
//! # What is bound
//!
//! - Shape: every count above, every grinding difficulty, every query width.
//! - Instance label: the trusted number of opening claims.
//! - Instance label: the plain WHIR parameters, unchanged.
//! - Instance label: the mask rate, the mask geometry, the randomness budgets.

mod player;
use alloc::vec::Vec;

use p3_challenger::fs::{
    DomainSeparator, Hierarchy, Interaction, InteractionPattern, Kind, Length, Unit,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_util::log2_strict_usize;
pub(crate) use player::{ZkWhirProverTranscript, ZkWhirVerifierTranscript};

use super::{
    Alphabet, INITIAL_BATCHING, OOD_ANSWER, OOD_POINT, QUERY_INDICES, QUERY_POW, ROUND_BATCHING,
    bind_folding_factor, query_draws,
};
use crate::parameters::{FoldingFactor, SecurityAssumption};
use crate::pcs::zk::{MaskCodeShape, MaskGroupShape, ZkWhirConfig};

/// Version byte bound into the transcript seed.
const VERSION: u8 = 2;

/// Protocol name bound into the transcript seed.
///
/// Distinct from the plain name, so the two pipelines can never share a seed.
const NAME: &[u8] = b"p3-whir-hvzk";

/// Step label of the oracle committed by a code-switching round.
pub(crate) const ORACLE_COMMITMENT: &str = "oracle_commitment";

/// Step label of the mask committed alongside it.
pub(crate) const SWITCH_MASK_COMMITMENT: &str = "switch_mask_commitment";

/// Step label of the fresh source mask of the base case.
pub(crate) const BASE_FRESH_COMMITMENT: &str = "base_fresh_commitment";

/// Step label of one group of fresh blinds of the base case.
pub(crate) const BASE_BLIND_COMMITMENT: &str = "base_blind_commitment";

/// Step label of the fresh-side claim `mu_g`.
pub(crate) const BASE_CLAIM: &str = "base_claim";

/// Step label of the blinding challenge `gamma`.
pub(crate) const BASE_GAMMA: &str = "base_gamma";

/// Step label of a one-time-pad reveal of a message word.
pub(crate) const BASE_REVEAL_MESSAGE: &str = "base_reveal_message";

/// Step label of a one-time-pad reveal of an encoding-randomness word.
pub(crate) const BASE_REVEAL_RANDOMNESS: &str = "base_reveal_randomness";

/// Step label of the grinding step guarding the base-case spot checks.
pub(crate) const BASE_POW: &str = "base_pow";

/// Step label of the source spot-check positions.
pub(crate) const BASE_SOURCE_QUERIES: &str = "base_source_queries";

/// Step label of one group's mask spot-check positions.
pub(crate) const BASE_MASK_QUERIES: &str = "base_mask_queries";

/// Numbers that fix one masked sumcheck batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkSumcheckShape {
    /// Number of rounds this batch runs.
    pub rounds: usize,
    /// Grinding difficulty inside each round.
    pub pow_bits: usize,
    /// Wire coefficients one round sends, `max(ell_zk, 3) - 1`.
    pub wire_len: usize,
}

impl ZkSumcheckShape {
    /// Append this batch's steps to a step sequence under construction.
    fn extend(&self, steps: &mut Vec<Interaction>) {
        super::push_delegation(steps, "masked_sumcheck");
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
        F: PrimeField64,
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
                Length::Fixed(1),
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                OOD_ANSWER,
                Length::Scalar,
            ));
        }

        if self.query_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                QUERY_POW,
                Length::Fixed(self.query_pow_bits),
            ));
        }

        if self.query_draws > 0 {
            steps.push(Interaction::uniform_bits(
                Hierarchy::Atomic,
                Kind::Challenge,
                QUERY_INDICES,
                self.index_bits,
                Length::Fixed(self.query_draws),
            ));
        }

        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ROUND_BATCHING,
            Length::Scalar,
        ));

        self.sumcheck.extend(steps);
    }
}

/// Numbers that fix the masked base case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkBaseCaseShape {
    /// Message length of the randomized terminal source code.
    pub source_message_len: usize,
    /// Encoding-randomness length of that code.
    pub source_randomness_len: usize,
    /// Bit width of a source spot-check position.
    pub source_index_bits: usize,
    /// Number of source spot-check positions actually drawn.
    pub source_query_draws: usize,
    /// Grinding difficulty guarding every spot check.
    pub pow_bits: usize,
    /// Every committed mask group, in commitment order.
    pub groups: Vec<MaskGroupShape>,
    /// Spot-check positions drawn per group, in the same order.
    pub mask_query_draws: Vec<usize>,
}

impl ZkBaseCaseShape {
    /// Append the base case's steps to a step sequence under construction.
    pub(crate) fn extend<F, EF>(&self, steps: &mut Vec<Interaction>)
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
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

        if self.pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                BASE_POW,
                Length::Fixed(self.pow_bits),
            ));
        }

        if self.source_query_draws > 0 {
            steps.push(Interaction::uniform_bits(
                Hierarchy::Atomic,
                Kind::Challenge,
                BASE_SOURCE_QUERIES,
                self.source_index_bits,
                Length::Fixed(self.source_query_draws),
            ));
        }

        // Positions are shared inside a group, so one step covers the group.
        for (group, &draws) in self.groups.iter().zip(&self.mask_query_draws) {
            if draws > 0 {
                steps.push(Interaction::uniform_bits(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    BASE_MASK_QUERIES,
                    log2_strict_usize(group.shape.domain_size),
                    Length::Fixed(draws),
                ));
            }
        }
    }
}

/// Numbers that fix the transcript of one HVZK-WHIR run.
///
/// Both sides build this from their own configuration, never from a proof.
#[derive(Clone, Debug)]
pub struct ZkWhirShape {
    /// Number of trusted opening claims, supplied by the caller, never the proof.
    pub num_claims: usize,
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
    /// Spot checks made against each mask group.
    pub mask_queries: usize,
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
    /// - `num_claims`: the trusted statement's opening-claim count, never a proof length.
    #[must_use]
    pub fn new<EF, F, Challenger>(
        config: &ZkWhirConfig<EF, F, Challenger>,
        num_claims: usize,
    ) -> Self
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Every masked batch sends the same wire width.
        //
        // The sumcheck crate derives the same number from the same input, so this is a copy.
        // Both sides of this pipeline seed from this one, so the copy cannot desync them.
        // Drift would instead leave the fingerprint describing steps the run no longer takes.
        let wire_len = config.zk.ell_zk.max(3) - 1;

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
                        wire_len,
                    },
                    log_inv_rate: params.log_inv_rate,
                    switch_mask: config.switch_masks[index],
                }
            })
            .collect();

        // The base case runs against the randomized terminal source code.
        let final_config = config.final_round_config();
        let source_domain_size = final_config.domain_size >> final_config.folding_factor;
        let groups = config.mask_groups();
        let mask_query_draws = groups
            .iter()
            .map(|group| query_draws(group.shape.domain_size, config.mask_queries))
            .collect();

        Self {
            num_claims,
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
                wire_len,
            },
            rounds,
            base_case: ZkBaseCaseShape {
                source_message_len: 1 << final_config.num_variables,
                source_randomness_len: config.oracle_randomness[config.n_rounds()],
                source_index_bits: log2_strict_usize(source_domain_size),
                source_query_draws: query_draws(source_domain_size, config.final_queries),
                pow_bits: config.final_pow_bits,
                groups,
                mask_query_draws,
            },
            security_level: config.security_level,
            pow_budget: config.pow_bits,
            starting_log_inv_rate: config.starting_log_inv_rate,
            soundness_type: config.soundness_type,
            folding_factor: config.folding_factor.clone(),
            ell_zk: config.zk.ell_zk,
            mask_log_inv_rate: config.zk.mask_log_inv_rate,
            mask_queries: config.mask_queries,
            oracle_randomness: config.oracle_randomness.clone(),
            sumcheck_mask: config.sumcheck_mask,
        }
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Scope
    ///
    /// The outer player enforces every step. Masked sumchecks use typed delegation
    /// markers and enforce their own complete sub-transcripts.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// Leaf steps and their paired delegation markers are constructed together.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut steps = Vec::new();

        // One challenge weights the incoming evaluation claims into a single sum.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            INITIAL_BATCHING,
            Length::Scalar,
        ));

        self.initial_sumcheck.extend(&mut steps);

        for round in &self.rounds {
            round.extend::<F, EF>(&mut steps);
        }

        self.base_case.extend::<F, EF>(&mut steps);

        InteractionPattern::new(steps).expect("HVZK steps and delegation markers are well formed")
    }

    /// Bind the protocol identity, this shape, and the remaining parameters.
    ///
    /// # Soundness
    ///
    /// The mask rate sets the mask code's distance.
    ///
    /// Distance is what makes a spot check bind.
    ///
    /// The rate reaches the shape only through the mask domain sizes.
    /// The label therefore carries it directly.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        separator.instance(&(self.num_claims as u64).to_be_bytes());

        // Delegated sumchecks are also bound before entering their sub-transcripts.
        for batch in
            core::iter::once(&self.initial_sumcheck).chain(self.rounds.iter().map(|r| &r.sumcheck))
        {
            for value in [batch.rounds, batch.pow_bits, batch.wire_len] {
                separator.instance(&(value as u64).to_be_bytes());
            }
        }

        // The plain WHIR statement, bound exactly as the plain pipeline binds it.
        separator.instance(&(self.num_variables as u64).to_be_bytes());
        for value in self.unreplayed_plain {
            separator.instance(&(value as u64).to_be_bytes());
        }
        separator.instance(&(self.security_level as u64).to_be_bytes());
        separator.instance(&(self.pow_budget as u64).to_be_bytes());
        separator.instance(&(self.starting_log_inv_rate as u64).to_be_bytes());
        separator.instance(&(self.soundness_type as u64).to_be_bytes());
        bind_folding_factor(&mut separator, &self.folding_factor);
        separator.instance(&(self.rounds.len() as u64).to_be_bytes());
        for round in &self.rounds {
            separator.instance(&(round.log_inv_rate as u64).to_be_bytes());
        }

        // The hiding overlay: mask geometry and every randomness budget.
        separator.instance(&(self.ell_zk as u64).to_be_bytes());
        separator.instance(&(self.mask_log_inv_rate as u64).to_be_bytes());
        separator.instance(&(self.mask_queries as u64).to_be_bytes());
        bind_mask_code(&mut separator, &self.sumcheck_mask);
        for round in &self.rounds {
            bind_mask_code(&mut separator, &round.switch_mask);
        }
        separator.instance(&(self.oracle_randomness.len() as u64).to_be_bytes());
        for &budget in &self.oracle_randomness {
            separator.instance(&(budget as u64).to_be_bytes());
        }

        separator
    }
}

/// Bind one mask code's three lengths, each as its own chunk.
fn bind_mask_code<U: Unit>(separator: &mut DomainSeparator<U>, code: &MaskCodeShape) {
    separator.instance(&(code.message_len as u64).to_be_bytes());
    separator.instance(&(code.randomness_len as u64).to_be_bytes());
    separator.instance(&(code.domain_size as u64).to_be_bytes());
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

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

    #[test]
    fn trusted_claim_count_changes_the_first_challenge() {
        let first = |count| {
            let mut challenger = fresh_challenger();
            ZkWhirShape::new(&base_config(), count)
                .domain_separator::<F, EF>()
                .seed(&mut challenger);
            let challenge: EF = challenger.sample_algebra_element();
            challenge
        };
        assert_ne!(first(1), first(2));
    }

    #[test]
    fn masked_sumcheck_is_a_delegated_typed_protocol() {
        use p3_challenger::fs::{FieldToFieldCodec, VerifierState};
        let shape = ZkWhirShape::new(&base_config(), 1);
        let separator = shape.domain_separator::<F, EF>();
        let mut challenger = fresh_challenger();
        let mut state = VerifierState::new(&mut challenger, &separator, &[]);
        state.challenge_extension::<F, EF, FieldToFieldCodec<F>>(INITIAL_BATCHING);
        state.begin_protocol::<super::super::Sumcheck>("masked_sumcheck");
        state.end_protocol::<super::super::Sumcheck>("masked_sumcheck");
        state.abort();
    }

    #[test]
    fn ood_step_supports_rejection_without_consuming_another_step() {
        use p3_challenger::fs::{FieldToFieldCodec, VerifierState};
        let mut steps = Vec::new();
        ZkWhirShape::new(&base_config(), 1).rounds[0].extend::<F, EF>(&mut steps);
        let separator = DomainSeparator::<Alphabet<F>>::new(
            VERSION,
            NAME,
            InteractionPattern::new(steps).unwrap(),
        );
        let mut challenger = fresh_challenger();
        let mut state = VerifierState::new(&mut challenger, &separator, &[]);
        state.observe_opaque(ORACLE_COMMITMENT, [F::ONE; 8]);
        state.observe_opaque(SWITCH_MASK_COMMITMENT, [F::ONE; 8]);
        let points = state.challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
            OOD_POINT,
            1,
            |point, _| !point.is_zero(),
        );
        assert!(!points[0].as_inner().is_zero());
        state.abort();
    }

    /// First challenge the seed of a configuration produces on a fresh sponge.
    fn first_challenge(config: &Config) -> F {
        let mut challenger = fresh_challenger();
        ZkWhirShape::new(config, 1)
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
        ZkWhirShape::new(&zk, 1)
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
    fn every_user_facing_parameter_reaches_the_seed() {
        // Walk `ProtocolParameters` and `ZkParameters` field by field.
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
        // Walk the plain half of `ZkWhirConfig`, field by field.
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
}
