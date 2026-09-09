//! Fiat-Shamir transcript of the WHIR proximity test.
//!
//! # Overview
//!
//! One statement of what a WHIR run absorbs and draws, consumed by both sides.
//!
//! It is built from the derived configuration alone.
//! Neither side ever reads a count out of a proof.
//!
//! # Shape
//!
//! ```text
//!     initial batching       one extension element
//!     initial sumcheck       folding_factor(0) rounds
//!     per round:  commitment     one opaque value
//!                 out-of-domain  one point drawn, one answer sent, per sample
//!                 grinding       only when the difficulty is positive
//!                 queries        num_queries draws of index_bits bits
//!                 batching       one extension element
//!                 sumcheck       folding_factor(round + 1) rounds
//!     final polynomial       2^final_sumcheck_rounds extension elements
//!     final grinding         only when the difficulty is positive
//!     final queries          final_queries draws of index_bits bits
//!     final sumcheck         final_sumcheck_rounds rounds
//! ```
//!
//! A sumcheck round is three steps: two coefficients, optional grinding, one challenge.
//!
//! # What is bound
//!
//! - Shape: every count above, every grinding difficulty, every query width.
//! - Instance label: the code rates, the security level, the soundness assumption.
//! - Instance label: the folding strategy and the variable count.
//!
//! # What is not described
//!
//! Hints.
//! A hint never enters the sponge, so it cannot move a challenge.
//!
//! WHIR carries every hint inside its own serde proof.
//! The verifier length-checks each one against the configuration before use.
//!
//! # Soundness
//!
//! The seed is absorbed where the WHIR run starts, before its first challenge.
//! Everything the caller bound earlier stays in the sponge and keeps its effect.

pub mod zk;

use alloc::vec::Vec;

use p3_challenger::fs::{
    DomainSeparator, FieldUnit, Hierarchy, Interaction, InteractionPattern, Kind, Length, Unit,
};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_util::log2_strict_usize;

use crate::parameters::{FoldingFactor, SecurityAssumption, WhirConfig};

/// Version byte bound into the transcript seed.
// Version 2 reserves the constant batching coefficient for the carried claim.
const VERSION: u8 = 2;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-whir";

/// Step label of the challenge batching the incoming evaluation claims.
const INITIAL_BATCHING: &str = "initial_batching";

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

/// Step label of the two coefficients one sumcheck round sends.
const SUMCHECK_POLY: &str = "sumcheck_poly";

/// Step label of the grinding step inside a sumcheck round.
const SUMCHECK_POW: &str = "sumcheck_pow";

/// Step label of a sumcheck folding challenge.
const FOLD_CHALLENGE: &str = "fold_challenge";

/// Step label of the final polynomial, sent in the clear.
const FINAL_POLY: &str = "final_poly";

/// Step label of the grinding step guarding the final query indices.
const FINAL_QUERY_POW: &str = "final_query_pow";

/// Step label of the final query indices.
const FINAL_QUERY_INDICES: &str = "final_query_indices";

/// Sponge alphabet of a challenger that speaks the base field natively.
pub type Alphabet<F> = FieldUnit<F>;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    /// Number of rounds this phase runs.
    pub rounds: usize,
    /// Grinding difficulty inside each round.
    pub pow_bits: usize,
}

impl SumcheckShape {
    /// Append this phase's steps to a step sequence under construction.
    fn extend<F, EF>(&self, steps: &mut Vec<Interaction>)
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        for _ in 0..self.rounds {
            // Two of the quadratic's three values cross the wire.
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                SUMCHECK_POLY,
                Length::Fixed(2),
            ));

            // Grinding sits between the coefficients and the challenge they face.
            if self.pow_bits > 0 {
                steps.push(Interaction::algebra::<F, F>(
                    Hierarchy::Atomic,
                    Kind::Pow,
                    SUMCHECK_POW,
                    Length::Fixed(self.pow_bits),
                ));
            }

            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FOLD_CHALLENGE,
                Length::Scalar,
            ));
        }
    }

    /// Number of steps this phase contributes.
    const fn step_count(&self) -> usize {
        self.rounds * if self.pow_bits > 0 { 3 } else { 2 }
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
        F: PrimeField64,
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
        if self.query_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                QUERY_POW,
                Length::Fixed(self.query_pow_bits),
            ));
        }

        // Every index is drawn at the same width, so they form one step.
        if self.query_draws > 0 {
            steps.push(Interaction::uniform_bits(
                Hierarchy::Atomic,
                Kind::Challenge,
                QUERY_INDICES,
                self.index_bits,
                Length::Fixed(self.query_draws),
            ));
        }

        // One challenge weights this round's fresh constraints against the carried claim.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ROUND_BATCHING,
            Length::Scalar,
        ));

        self.sumcheck.extend::<F, EF>(steps);
    }

    /// Number of steps this round contributes.
    const fn step_count(&self) -> usize {
        2 + 2 * self.ood_samples
            + if self.query_pow_bits > 0 { 1 } else { 0 }
            + if self.query_draws > 0 { 1 } else { 0 }
            + self.sumcheck.step_count()
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
    #[must_use]
    pub fn new<EF, F, Challenger>(config: &WhirConfig<EF, F, Challenger>) -> Self
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
        let capacity = 1
            + self.initial_sumcheck.step_count()
            + self
                .rounds
                .iter()
                .map(WhirRoundShape::step_count)
                .sum::<usize>()
            + 3
            + self.final_sumcheck.step_count();
        let mut steps = Vec::with_capacity(capacity);

        // One challenge weights the incoming evaluation claims into a single sum.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            INITIAL_BATCHING,
            Length::Scalar,
        ));

        self.initial_sumcheck.extend::<F, EF>(&mut steps);

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

        if self.final_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                FINAL_QUERY_POW,
                Length::Fixed(self.final_pow_bits),
            ));
        }

        if self.final_query_draws > 0 {
            steps.push(Interaction::uniform_bits(
                Hierarchy::Atomic,
                Kind::Challenge,
                FINAL_QUERY_INDICES,
                self.final_index_bits,
                Length::Fixed(self.final_query_draws),
            ));
        }

        self.final_sumcheck.extend::<F, EF>(&mut steps);

        InteractionPattern::new(steps).expect("a flat sequence of leaf steps is always well formed")
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
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The whole soundness statement is phrased in these numbers.
        separator.instance(&(self.num_variables as u64).to_be_bytes());
        // The commitment phase draws its samples before the described run starts.
        separator.instance(&(self.commitment_ood_samples as u64).to_be_bytes());
        separator.instance(&(self.security_level as u64).to_be_bytes());
        separator.instance(&(self.pow_budget as u64).to_be_bytes());
        separator.instance(&(self.starting_log_inv_rate as u64).to_be_bytes());
        separator.instance(&(self.soundness_type as u64).to_be_bytes());

        bind_folding_factor(&mut separator, &self.folding_factor);

        // Each round commits at its own rate, and the rate sets that round's distance.
        separator.instance(&(self.rounds.len() as u64).to_be_bytes());
        for round in &self.rounds {
            separator.instance(&(round.log_inv_rate as u64).to_be_bytes());
        }

        separator
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::fs::TypeTag;
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::parameters::ProtocolParameters;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;
    type Config = WhirConfig<EF, F, Ch>;

    /// Variable count every configuration in this module is derived at.
    const NUM_VARIABLES: usize = 16;

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

    /// First challenge the seed of a configuration produces on a fresh sponge.
    fn first_challenge(config: &Config) -> F {
        let mut challenger = fresh_challenger();
        WhirShape::new(config)
            .domain_separator::<F, EF>()
            .seed(&mut challenger);
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
        let pattern = WhirShape::new(&config).pattern::<F, EF>();

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
        // A derived field can be reached only through the shape, so this is the
        // check that no transcript-bearing number was left out of the pattern.
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
