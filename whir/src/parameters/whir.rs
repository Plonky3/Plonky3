//! Derived WHIR protocol configuration computed from user-facing parameters.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use thiserror::Error;

use super::{FoldingFactor, FoldingFactorError, ProtocolParameters};
use crate::pcs::proof::WhirProof;

/// Reasons a set of user-facing parameters cannot form a valid WHIR configuration.
#[derive(Debug, Error)]
pub enum WhirConfigError {
    /// The folding factor is incompatible with the polynomial size.
    #[error(transparent)]
    FoldingFactor(#[from] FoldingFactorError),

    /// The domain after the first fold exceeds the base field two-adicity.
    ///
    /// - Twiddles and query equality polynomials must stay in the base field.
    /// - A larger first-round folding factor shrinks this domain.
    #[error(
        "folded domain 2^{log_folded_domain_size} exceeds base-field two-adicity 2^{two_adicity}; increase the first-round folding factor"
    )]
    FoldedDomainExceedsTwoAdicity {
        log_folded_domain_size: usize,
        two_adicity: usize,
    },

    /// The initial Reed-Solomon evaluation domain cannot be represented as a
    /// `usize` length.
    ///
    /// The initial domain has size `2^(num_variables + starting_log_inv_rate)`.
    /// That exponent must fit in `usize` and be strictly less than
    /// `usize::BITS` before computing `1 << exponent`.
    #[error(
        "initial domain exponent num_variables ({num_variables}) + starting_log_inv_rate ({starting_log_inv_rate}) must fit and be less than usize::BITS ({usize_bits})"
    )]
    InitialDomainExceedsUsize {
        num_variables: usize,
        starting_log_inv_rate: usize,
        usize_bits: usize,
    },

    /// Explicit per-round codeword rates have the wrong length.
    #[error("expected {expected} explicit codeword rates (one per round), got {actual}")]
    RoundRateCountMismatch { expected: usize, actual: usize },

    /// Explicit per-round folding factors have the wrong length.
    #[error("expected {expected} explicit folding factors (one per phase), got {actual}")]
    FoldingFactorCountMismatch { expected: usize, actual: usize },

    /// A requested codeword rate would require growing the Reed-Solomon domain.
    #[error("round {round}: requested codeword rate would require growing the RS domain")]
    RateGrowsDomain { round: usize },

    /// The starting code rate is not redundant.
    ///
    /// - A rate of `2^-0 = 1` adds no Reed-Solomon redundancy.
    /// - A proximity test needs redundancy, so there is nothing to check.
    /// - At rate `1` the proximity parameter `delta` is `<= 0`.
    /// - The query count `ceil(-lambda / log2(1 - delta))` is then non-positive.
    /// - A non-positive count rounds down to zero queries.
    /// - With zero queries the verifier accepts any committed function.
    ///
    /// Every code rate must be redundant: rate `<= 1/2`.
    #[error(
        "starting_log_inv_rate must be >= 1 (rate <= 1/2); got 2^-{log_inv_rate}, a non-redundant code whose proximity test makes zero queries"
    )]
    NonRedundantStartingRate { log_inv_rate: usize },

    /// A per-round code rate is not redundant.
    ///
    /// - The codeword committed after one intermediate round has rate `1`.
    /// - A rate of `1` drives that round's query count to zero.
    /// - The failure matches a non-redundant starting rate.
    #[error(
        "round {round}: log_inv_rate must be >= 1 (rate <= 1/2); got 2^-{log_inv_rate}, a non-redundant code whose proximity test makes zero queries"
    )]
    NonRedundantRoundRate { round: usize, log_inv_rate: usize },

    /// No out-of-domain sample count reaches the requested security level.
    ///
    /// - The field is too small for the requested security level.
    /// - Lower the security level or use a larger extension field.
    #[error(
        "no out-of-domain sample count reaches {security_level}-bit security with a {field_size_bits}-bit field"
    )]
    OodSamplesInfeasible {
        security_level: usize,
        field_size_bits: usize,
    },

    /// A derived proof-of-work difficulty exceeds the configured grinding budget.
    ///
    /// - Each phase grinds `security_level - algebraic_bits` bits to hit target.
    /// - When the field or rate is too weak, that gap exceeds the budget.
    /// - The instance then cannot reach `security_level` within allowed grinding.
    ///
    /// Raise the grinding budget, lower the security level, or use a larger
    /// extension field or smaller rate.
    #[error(
        "derived proof-of-work of {required} bits exceeds the {budget}-bit grinding budget; the field or rate is too weak for the requested security"
    )]
    PowBitsExceedBudget { required: usize, budget: usize },
}

/// Derived configuration for a single intermediate WHIR round.
///
/// All values are computed from the user-facing protocol parameters
/// and the accumulated state from prior rounds.
#[derive(Debug, Clone)]
pub struct RoundConfig<F> {
    /// Proof-of-work difficulty (in bits) for the STIR query phase.
    pub pow_bits: usize,
    /// Proof-of-work difficulty (in bits) for the folding sumcheck phase.
    pub folding_pow_bits: usize,
    /// Number of STIR proximity queries to make in this round.
    pub num_queries: usize,
    /// Number of out-of-domain evaluation samples.
    pub ood_samples: usize,
    /// Number of multilinear variables remaining after folding in this round.
    pub num_variables: usize,
    /// Number of variables folded in this round.
    pub folding_factor: usize,
    /// Log-inverse rate of the codeword committed after this round.
    pub log_inv_rate: usize,
    /// Size of the evaluation domain before folding in this round.
    pub domain_size: usize,
    /// Generator of the folded evaluation domain after this round's fold.
    pub folded_domain_gen: F,
}

/// Fully derived WHIR protocol configuration.
///
/// Built from user-facing protocol parameters plus the polynomial size.
///
/// Contains all precomputed values needed by the prover and verifier.
#[derive(Debug, Clone)]
pub struct WhirConfig<EF, F, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Number of variables in the original multilinear polynomial.
    pub num_variables: usize,
    /// Protocol parameters.
    pub params: ProtocolParameters,
    /// Per-round derived configuration for each intermediate STIR round.
    pub round_parameters: Vec<RoundConfig<F>>,
    /// Concrete folding factors used before the final direct-send phase.
    ///
    /// For constant schedules the last entry may be smaller than the nominal
    /// configured factor, e.g. `Constant(8)` on 15 variables derives `[8, 7]`.
    pub folding_schedule: Vec<usize>,
    /// Number of out-of-domain samples during the commitment phase.
    pub commitment_ood_samples: usize,
    /// PoW bits for the initial folding sumcheck (before any STIR rounds).
    pub starting_folding_pow_bits: usize,
    /// Number of STIR queries in the final proximity test.
    pub final_queries: usize,
    /// PoW bits for the final STIR query phase.
    pub final_pow_bits: usize,
    /// Number of sumcheck rounds in the final phase.
    pub final_sumcheck_rounds: usize,
    /// PoW bits for the final folding sumcheck.
    pub final_folding_pow_bits: usize,
    /// Phantom marker for the extension field type.
    pub _extension_field: PhantomData<EF>,
    /// Phantom marker for the challenger type.
    pub _challenger: PhantomData<Challenger>,
}

impl<EF, F, Challenger> Deref for WhirConfig<EF, F, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = ProtocolParameters;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

/// Round a real-valued proof-of-work gap up to whole grinding bits.
///
/// Grinding contributes whole bits of security: a witness with `b` leading
/// zero bits adds exactly `b` bits.
///
/// The gap to grind is `security_level - algebraic_bits`, a real number.
/// - Flooring drops the fractional part.
/// - Achieved security `algebraic_bits + floor(gap)` then dips below target.
/// - Rounding up keeps `algebraic_bits + ceil(gap) >= security_level`.
///
/// Mirrors the query-count derivation, which rounds up for the same reason.
fn ceil_pow_bits(gap: f64) -> usize {
    libm::ceil(gap) as usize
}

impl<EF, F, Challenger> WhirConfig<EF, F, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Construct an empty proof shaped by this configuration.
    pub(crate) fn empty_proof<MT: Mmcs<F>>(&self) -> WhirProof<F, EF, MT> {
        WhirProof::empty(self.n_rounds(), self.final_queries)
    }

    /// Derive a full protocol configuration from user-facing parameters.
    ///
    /// # Errors
    ///
    /// - The folding factor does not fit the polynomial size.
    /// - The first fold leaves a domain larger than the base-field two-adicity.
    /// - Explicit per-round rates or folding factors have the wrong length.
    /// - A requested rate would grow the Reed-Solomon domain.
    /// - The field is too small to reach the requested security level.
    /// - A derived proof-of-work difficulty exceeds the grinding budget.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        num_variables: usize,
        whir_parameters: ProtocolParameters,
    ) -> Result<Self, WhirConfigError> {
        // ---------------------------------------------------------------
        // Phase 1: Validate inputs and set up global constants.
        // ---------------------------------------------------------------

        // Preserve the original variable count; the mutable copy below
        // tracks the remaining variables as we consume them round by round.
        let initial_num_variables = num_variables;

        // Invariant: the Reed-Solomon code must be redundant, rate <= 1/2.
        //
        //     rate 1             -> proximity parameter delta <= 0
        //     delta <= 0         -> query count ceil(-lambda / log2(1 - delta)) <= 0
        //     non-positive count -> rounds down to zero STIR queries
        //     zero queries       -> verifier accepts any committed function
        if whir_parameters.starting_log_inv_rate == 0 {
            return Err(WhirConfigError::NonRedundantStartingRate {
                log_inv_rate: whir_parameters.starting_log_inv_rate,
            });
        }

        // Derive the concrete folding schedule once. Validation is a property
        // of this derivation, so all later code uses the same source of truth.
        let folding_schedule = whir_parameters
            .folding_factor
            .compute_folding_schedule(num_variables)?;

        // PoW contributes an independent additive term to security,
        // so the algebraic protocol only needs to cover the remainder.
        let protocol_security_level = whir_parameters
            .security_level
            .saturating_sub(whir_parameters.pow_bits);

        // Number of bits in the extension field; upper-bounds all per-element errors.
        let field_size_bits = EF::bits();

        // Mutable state that evolves as we derive per-round parameters.
        let mut log_inv_rate = whir_parameters.starting_log_inv_rate;
        let mut num_variables = num_variables;

        // Initial evaluation domain: 2^(num_variables + log_inv_rate) points.
        let log_domain_size = num_variables
            .checked_add(log_inv_rate)
            .filter(|&log_domain_size| log_domain_size < usize::BITS as usize)
            .ok_or(WhirConfigError::InitialDomainExceedsUsize {
                num_variables,
                starting_log_inv_rate: log_inv_rate,
                usize_bits: usize::BITS as usize,
            })?;
        let mut domain_size: usize = 1 << log_domain_size;

        // ---------------------------------------------------------------
        // Phase 2: Two-adicity guard.
        // ---------------------------------------------------------------
        //
        // After the first fold the domain has size 2^log_folded_domain_size.
        // We restrict this to F::TWO_ADICITY so that:
        //   - FFT twiddle factors stay in the base field (faster).
        //   - WHIR query equality polynomials stay in the base field.
        //
        // A larger folding_factor_0 pushes the folded domain below the limit.
        // This does NOT restrict how much data can be committed.
        let log_folded_domain_size = log_domain_size - folding_schedule[0];
        if log_folded_domain_size > F::TWO_ADICITY {
            return Err(WhirConfigError::FoldedDomainExceedsTwoAdicity {
                log_folded_domain_size,
                two_adicity: F::TWO_ADICITY,
            });
        }

        // ---------------------------------------------------------------
        // Phase 3: Determine round structure.
        // ---------------------------------------------------------------

        // How many intermediate STIR rounds, and how many variables remain
        // for the final direct-send sumcheck.
        // An invalid per-round schedule surfaces as a folding-factor config error.
        let folded_variables: usize = folding_schedule.iter().sum();
        let num_rounds = folding_schedule.len() - 1;
        let final_sumcheck_rounds = num_variables - folded_variables;

        let round_log_inv_rates = if whir_parameters.round_log_inv_rates.is_empty() {
            let mut rates = Vec::with_capacity(num_rounds);
            let mut rate = whir_parameters.starting_log_inv_rate;
            for &folding_factor in folding_schedule.iter().take(num_rounds) {
                rate += folding_factor - 1;
                rates.push(rate);
            }
            rates
        } else {
            if whir_parameters.round_log_inv_rates.len() != num_rounds {
                return Err(WhirConfigError::RoundRateCountMismatch {
                    expected: num_rounds,
                    actual: whir_parameters.round_log_inv_rates.len(),
                });
            }
            whir_parameters.round_log_inv_rates.clone()
        };

        // Same redundancy invariant, applied to each intermediate-round rate.
        //
        //     derived rates  : inherit the starting rate and only grow -> always >= 1
        //     explicit rates : caller-supplied, so a 0 entry must be rejected here
        if let Some(round) = round_log_inv_rates.iter().position(|&rate| rate == 0) {
            return Err(WhirConfigError::NonRedundantRoundRate {
                round,
                log_inv_rate: round_log_inv_rates[round],
            });
        }

        if let FoldingFactor::PerRound(factors) = &whir_parameters.folding_factor
            && factors.len() != num_rounds + 1
        {
            return Err(WhirConfigError::FoldingFactorCountMismatch {
                expected: num_rounds + 1,
                actual: factors.len(),
            });
        }

        // OOD samples for the commitment phase (before any folding).
        let commitment_ood_samples = whir_parameters
            .soundness_type
            .determine_ood_samples(
                whir_parameters.security_level,
                num_variables,
                log_inv_rate,
                field_size_bits,
            )
            .ok_or(WhirConfigError::OodSamplesInfeasible {
                security_level: whir_parameters.security_level,
                field_size_bits,
            })?;

        // PoW difficulty for the very first folding sumcheck.
        let starting_folding_pow_bits = whir_parameters.soundness_type.folding_pow_bits(
            whir_parameters.security_level,
            field_size_bits,
            num_variables,
            log_inv_rate,
        );

        // ---------------------------------------------------------------
        // Phase 4: Per-round parameter derivation.
        // ---------------------------------------------------------------
        //
        // After the initial fold, each round i:
        //   1. Reads the configured/derived code rate after folding.
        //   2. Determines query count from the old rate (queries test
        //      proximity to the code *before* this round's fold).
        //   3. Determines OOD sample count from the new rate.
        //   4. Derives PoW for both the query and folding sub-steps.
        //   5. Records the domain generator for the folded evaluation domain.

        let mut round_parameters = Vec::with_capacity(num_rounds);

        // Subtract the first-round folding factor; the loop below
        // handles subsequent rounds.
        num_variables -= folding_schedule[0];

        for (round, &next_rate) in round_log_inv_rates.iter().enumerate() {
            let folding_factor = folding_schedule[round];
            if next_rate > log_inv_rate + folding_factor {
                return Err(WhirConfigError::RateGrowsDomain { round });
            }
            let rs_reduction_factor = log_inv_rate + folding_factor - next_rate;

            // Queries use the *old* rate; OOD and folding use the *new* rate.
            // Number of STIR proximity queries at the current (old) rate.
            let num_queries = whir_parameters
                .soundness_type
                .queries(protocol_security_level, log_inv_rate);

            // OOD samples needed at the post-fold (new) rate.
            let ood_samples = whir_parameters
                .soundness_type
                .determine_ood_samples(
                    whir_parameters.security_level,
                    num_variables,
                    next_rate,
                    field_size_bits,
                )
                .ok_or(WhirConfigError::OodSamplesInfeasible {
                    security_level: whir_parameters.security_level,
                    field_size_bits,
                })?;

            // Two independent error sources bound the STIR round:
            //   - query_error: (1 - delta)^num_queries proximity test.
            //   - combination_error: random linear combination of OOD + queries.
            // The weaker bound determines how much PoW is needed.
            let query_error = whir_parameters
                .soundness_type
                .queries_error(log_inv_rate, num_queries);
            let combination_error = whir_parameters.soundness_type.queries_combination_error(
                field_size_bits,
                num_variables,
                next_rate,
                ood_samples,
                num_queries,
            );

            // PoW bridges the gap between the target and the weaker bound.
            let pow_bits = 0_f64
                .max(whir_parameters.security_level as f64 - (query_error.min(combination_error)));

            // PoW difficulty for the folding sumcheck at the new rate.
            let folding_pow_bits = whir_parameters.soundness_type.folding_pow_bits(
                whir_parameters.security_level,
                field_size_bits,
                num_variables,
                next_rate,
            );

            let next_folding_factor = folding_schedule[round + 1];

            // Generator of the two-adic subgroup for the folded domain.
            let folded_domain_gen =
                F::two_adic_generator(domain_size.ilog2() as usize - folding_factor);

            round_parameters.push(RoundConfig {
                pow_bits: ceil_pow_bits(pow_bits),
                folding_pow_bits: ceil_pow_bits(folding_pow_bits),
                num_queries,
                ood_samples,
                num_variables,
                folding_factor,
                log_inv_rate: next_rate,
                domain_size,
                folded_domain_gen,
            });

            // Advance mutable state for the next iteration.
            num_variables -= next_folding_factor;
            log_inv_rate = next_rate;
            domain_size >>= rs_reduction_factor;
        }

        // ---------------------------------------------------------------
        // Phase 5: Final round parameters.
        // ---------------------------------------------------------------

        // Final proximity queries at the last accumulated rate.
        let final_queries = whir_parameters
            .soundness_type
            .queries(protocol_security_level, log_inv_rate);

        // PoW for the final query phase: covers the gap between the
        // target and the query error at the final rate.
        let final_pow_bits = 0_f64.max(
            whir_parameters.security_level as f64
                - whir_parameters
                    .soundness_type
                    .queries_error(log_inv_rate, final_queries),
        );

        // The final folding sumcheck error is bounded by 1/|F|,
        // so PoW = max(0, security_level - (field_size - 1)).
        let final_folding_pow_bits =
            0_f64.max(whir_parameters.security_level as f64 - (field_size_bits - 1) as f64);

        // Validate construction
        assert_eq!(
            initial_num_variables,
            folding_schedule.iter().sum::<usize>() + final_sumcheck_rounds
        );

        let config = Self {
            params: whir_parameters,
            commitment_ood_samples,
            num_variables: initial_num_variables,
            starting_folding_pow_bits: ceil_pow_bits(starting_folding_pow_bits),
            round_parameters,
            folding_schedule,
            final_queries,
            final_pow_bits: ceil_pow_bits(final_pow_bits),
            final_sumcheck_rounds,
            final_folding_pow_bits: ceil_pow_bits(final_folding_pow_bits),
            _extension_field: PhantomData,
            _challenger: PhantomData,
        };

        // The final-round config must expose exactly the direct-send variable count.
        //
        //     prover     : sends 2^count final evaluations in the clear
        //     verifier   : length-checks the final polynomial against 2^count
        //     transcript : absorbs 2^count final coefficients
        assert_eq!(
            config.final_round_config().num_variables,
            config.final_sumcheck_rounds
        );

        // Enforce the grinding budget.
        // A derived PoW above the cap means the field or rate cannot reach
        // security_level within the allowed grinding, so the claimed security
        // would be aspirational rather than met.
        let required = config.max_pow_bits();
        if required > config.params.pow_bits {
            return Err(WhirConfigError::PowBitsExceedBudget {
                required,
                budget: config.params.pow_bits,
            });
        }

        Ok(config)
    }

    /// Returns the size of the initial evaluation domain.
    ///
    /// This is the size of the domain used to evaluate the original multilinear polynomial
    /// before any folding or reduction steps are applied in the WHIR protocol.
    ///
    /// It is computed as:
    ///
    /// \begin{equation}
    /// 2^{\text{num\_variables} + \text{starting\_log\_inv\_rate}}
    /// \end{equation}
    ///
    /// - `num_variables` is the number of variables in the original multivariate polynomial.
    /// - `starting_log_inv_rate` is the initial inverse rate of the Reed-Solomon code,
    ///   controlling redundancy relative to the degree.
    ///
    /// # Returns
    /// A power-of-two value representing the number of evaluation points in the starting domain.
    pub const fn starting_domain_size(&self) -> usize {
        1 << (self.num_variables + self.params.starting_log_inv_rate)
    }

    /// Returns the number of intermediate STIR rounds (excludes the final round).
    pub const fn n_rounds(&self) -> usize {
        self.round_parameters.len()
    }

    /// Returns how many bits the RS domain shrinks by at the given round.
    pub fn rs_reduction_factor(&self, round: usize) -> usize {
        let previous_log_inv_rate = if round == 0 {
            self.params.starting_log_inv_rate
        } else {
            self.round_parameters[round - 1].log_inv_rate
        };
        previous_log_inv_rate + self.round_folding_factor(round)
            - self.round_parameters[round].log_inv_rate
    }

    /// Returns the log2 size of the largest FFT
    /// (At commitment we perform 2^folding_factor FFT of size 2^max_fft_size)
    pub fn max_fft_size(&self) -> usize {
        self.num_variables + self.params.starting_log_inv_rate - self.round_folding_factor(0)
    }

    /// Returns whether all PoW difficulties are within the configured maximum.
    ///
    /// Checks the starting, final, and per-round PoW bits against the ceiling.
    /// Returns false if any value exceeds the limit.
    pub fn check_pow_bits(&self) -> bool {
        self.max_pow_bits() <= self.params.pow_bits
    }

    /// Largest proof-of-work difficulty (in bits) demanded by any phase.
    ///
    /// Scans every grinding step: the starting fold, each round's query and
    /// fold steps, and the final query and fold steps.
    ///
    /// Comparing this against the configured budget tells whether the field
    /// and rate can reach the requested security within allowed grinding.
    pub fn max_pow_bits(&self) -> usize {
        // Whole-protocol grinding steps outside the per-round loop.
        let outer = self
            .starting_folding_pow_bits
            .max(self.final_pow_bits)
            .max(self.final_folding_pow_bits);

        // Each round grinds once for queries and once for folding.
        self.round_parameters
            .iter()
            .map(|r| r.pow_bits.max(r.folding_pow_bits))
            .fold(outer, usize::max)
    }

    /// Retrieves the concrete derived folding factor for a given round.
    ///
    /// # Panics
    ///
    /// Panics if `round > self.n_rounds()`.
    /// The schedule has one entry per fold, including the final pre-direct-send fold.
    pub fn round_folding_factor(&self, round: usize) -> usize {
        self.folding_schedule[round]
    }

    /// Total variables folded through the given round index, inclusive.
    ///
    /// Indices past the last fold saturate to the full folded variable count.
    pub fn total_folded_through(&self, round: usize) -> usize {
        self.folding_schedule.iter().take(round + 1).sum()
    }

    /// Compute the synthetic or derived `RoundConfig` for the final phase.
    ///
    /// - If no folding rounds were configured, constructs a fallback config
    ///   based on the starting domain and folding factor.
    /// - If rounds were configured, derives the final config by adapting
    ///   the last round's values for the final folding phase.
    ///
    /// This is used by the verifier when verifying the final polynomial,
    /// ensuring consistent challenge selection and STIR constraint handling.
    pub fn final_round_config(&self) -> RoundConfig<F> {
        if self.round_parameters.is_empty() {
            // No intermediate rounds: the polynomial was small enough that
            // the initial fold leads directly to the final phase.
            // Use the starting domain and initial folding factor.
            RoundConfig {
                num_variables: self.num_variables - self.round_folding_factor(0),
                folding_factor: self.round_folding_factor(self.n_rounds()),
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                log_inv_rate: self.params.starting_log_inv_rate,
                domain_size: self.starting_domain_size(),
                folded_domain_gen: F::two_adic_generator(
                    self.starting_domain_size().ilog2() as usize - self.round_folding_factor(0),
                ),
                ood_samples: 0,
                folding_pow_bits: self.final_folding_pow_bits,
            }
        } else {
            // Apply the last round's domain reduction to get the domain
            // size entering the final phase.
            let rs_reduction_factor = self.rs_reduction_factor(self.n_rounds() - 1);
            let folding_factor = self.round_folding_factor(self.n_rounds());

            let last = self.round_parameters.last().unwrap();

            // The domain shrinks by the RS reduction factor from the last round.
            let domain_size = last.domain_size >> rs_reduction_factor;

            // Generator for the final folded domain.
            let folded_domain_gen = F::two_adic_generator(
                domain_size.ilog2() as usize - self.round_folding_factor(self.n_rounds()),
            );

            RoundConfig {
                // Variables remaining after this final fold.
                num_variables: last.num_variables - folding_factor,
                folding_factor,
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                log_inv_rate: last.log_inv_rate,
                domain_size,
                folded_domain_gen,
                // Inherit OOD count from the last intermediate round.
                ood_samples: last.ood_samples,
                folding_pow_bits: self.final_folding_pow_bits,
            }
        }
    }

    /// Returns the inverse rate of the codeword committed after an
    /// intermediate round.
    pub fn inv_rate(&self, round: usize) -> usize {
        1 << self.round_parameters[round].log_inv_rate
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::parameters::{FoldingFactor, SecurityAssumption};

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Generates default WHIR parameters
    fn default_whir_params() -> ProtocolParameters {
        ProtocolParameters {
            security_level: 100,
            pow_bits: 20,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        }
    }

    #[test]
    fn test_whir_config_creation() {
        let params = default_whir_params();

        // Quartic extension: a realistic challenge field that meets the budget.
        let config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        assert_eq!(config.security_level, 100);
        assert_eq!(config.params.pow_bits, 20);
        assert_eq!(config.soundness_type, SecurityAssumption::CapacityBound);
    }

    #[test]
    fn ceil_pow_bits_rounds_up() {
        // Grinding adds whole bits, so a fractional gap rounds up.
        // Flooring would undershoot the security target by up to one bit.
        //
        //     0.0  -> 0   (no grinding needed)
        //     20.0 -> 20  (already whole)
        //     19.001 -> 20 (round up)
        //     20.7 -> 21  (round up)
        assert_eq!(ceil_pow_bits(0.0), 0);
        assert_eq!(ceil_pow_bits(20.0), 20);
        assert_eq!(ceil_pow_bits(19.001), 20);
        assert_eq!(ceil_pow_bits(20.7), 21);
    }

    #[test]
    fn derived_pow_reaches_security_target_per_phase() {
        // Invariant: achieved security per phase = algebraic_bits + pow_bits.
        //
        //     pow_bits = ceil(security_level - algebraic_bits)
        //     => algebraic_bits + pow_bits >= security_level
        //
        // Flooring the gap would make the sum dip below the target whenever
        // algebraic_bits is fractional.
        //
        // Unique decoding needs no out-of-domain samples, so the 31-bit base
        // field is feasible, and its query and combination errors are
        // fractional -- exactly the case where floor and ceil differ.
        //
        // The 31-bit field forces a large grinding gap, so the budget is set
        // wide enough to admit it; this test probes the rounding, not the cap.
        let soundness = SecurityAssumption::UniqueDecoding;
        let params = ProtocolParameters {
            security_level: 128,
            pow_bits: 128,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(4),
            soundness_type: soundness,
            starting_log_inv_rate: 1,
        };
        let config = WhirConfig::<F, F, MyChallenger>::new(20, params).unwrap();

        // Target in bits, and the field size the combination error is taken over.
        let target = config.security_level as f64;
        let field_bits = F::bits();

        // Walk the intermediate rounds, tracking the pre-fold (old) rate.
        // Queries test proximity to the code before this round's fold.
        let mut old_rate = config.params.starting_log_inv_rate;
        for cfg in &config.round_parameters {
            // Query phase is bounded by the weaker of two error sources.
            let query_error = soundness.queries_error(old_rate, cfg.num_queries);
            let combination_error = soundness.queries_combination_error(
                field_bits,
                cfg.num_variables,
                cfg.log_inv_rate,
                cfg.ood_samples,
                cfg.num_queries,
            );
            let algebraic = query_error.min(combination_error);

            // Ceil keeps the sum at or above target; floor would drop below it.
            assert!(
                cfg.pow_bits as f64 + algebraic >= target,
                "round query phase undershoots: {} + {algebraic} < {target}",
                cfg.pow_bits
            );

            old_rate = cfg.log_inv_rate;
        }

        // Final query phase grinds at the last accumulated rate.
        let final_error = soundness.queries_error(old_rate, config.final_queries);
        assert!(
            config.final_pow_bits as f64 + final_error >= target,
            "final query phase undershoots: {} + {final_error} < {target}",
            config.final_pow_bits
        );
    }

    #[test]
    fn new_rejects_pow_above_budget() {
        // Invariant: a derived PoW above the budget fails construction.
        //
        //     BabyBear as its own field is 31 bits.
        //     The final folding sumcheck error is bounded by 1/|F| ~ 2^-30.
        //     Reaching 100-bit security needs ~70 PoW bits, far above budget 20.
        //
        // Unique decoding needs no out-of-domain samples, so the small field is
        // otherwise feasible and the failure is purely the budget.
        let params = ProtocolParameters {
            security_level: 100,
            pow_bits: 20,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(4),
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        let err = WhirConfig::<F, F, MyChallenger>::new(20, params)
            .expect_err("derived PoW exceeds the budget; construction must fail");
        match err {
            WhirConfigError::PowBitsExceedBudget { required, budget } => {
                // The cap is the configured budget; the demand exceeds it.
                assert_eq!(budget, 20);
                assert!(
                    required > budget,
                    "required {required} should exceed {budget}"
                );
            }
            other => panic!("expected PowBitsExceedBudget, got {other:?}"),
        }
    }

    #[test]
    fn max_pow_bits_reports_largest_phase() {
        // The reported maximum is the largest PoW demanded by any phase.
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        // Force one phase to dominate, then confirm it is the reported max.
        config.starting_folding_pow_bits = 7;
        config.final_pow_bits = 31;
        config.final_folding_pow_bits = 5;
        config.round_parameters = vec![RoundConfig {
            pow_bits: 11,
            folding_pow_bits: 13,
            num_queries: 5,
            ood_samples: 2,
            num_variables: 10,
            folding_factor: 2,
            log_inv_rate: 1,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert_eq!(config.max_pow_bits(), 31);
    }

    #[test]
    fn new_errors_when_field_too_small_for_security() {
        // Invariant: an infeasible field/security pair errors instead of panicking.
        //
        // BabyBear used as its own extension field is 31 bits.
        // At 31 variables the OOD term gains ~0 bits per sample.
        // No sample count then reaches 100-bit security.
        //
        // Folding factor 5 keeps the folded domain at 2^27 = BabyBear two-adicity.
        // So the two-adicity guard passes and the OOD feasibility check is reached.
        let params = ProtocolParameters {
            security_level: 100,
            pow_bits: 20,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(5),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        let err = WhirConfig::<F, F, MyChallenger>::new(31, params)
            .expect_err("31-bit field cannot reach 100-bit security");
        assert!(
            matches!(err, WhirConfigError::OodSamplesInfeasible { .. }),
            "expected OodSamplesInfeasible, got {err:?}"
        );
    }

    #[test]
    fn new_errors_when_initial_domain_exponent_exceeds_usize_bits() {
        // Invariant: `WhirConfig::new` must reject an initial domain that cannot
        // be represented as a `usize` length before computing `1 << exponent`.
        let num_variables = usize::BITS as usize;
        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(num_variables),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        let err = WhirConfig::<F, F, MyChallenger>::new(num_variables, params)
            .expect_err("oversized initial domain must be rejected");
        match err {
            WhirConfigError::InitialDomainExceedsUsize {
                num_variables: got_num_variables,
                starting_log_inv_rate,
                usize_bits,
            } => {
                assert_eq!(got_num_variables, num_variables);
                assert_eq!(starting_log_inv_rate, 1);
                assert_eq!(usize_bits, usize::BITS as usize);
            }
            other => panic!("expected InitialDomainExceedsUsize, got {other:?}"),
        }
    }

    #[test]
    fn new_errors_when_initial_domain_exponent_addition_overflows() {
        // Invariant: the exponent addition itself must be checked. In release
        // builds unchecked addition would wrap and could derive a nonsensical
        // small domain.
        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(usize::MAX),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        let err = WhirConfig::<F, F, MyChallenger>::new(usize::MAX, params)
            .expect_err("overflowing initial domain exponent must be rejected");
        assert!(
            matches!(err, WhirConfigError::InitialDomainExceedsUsize { .. }),
            "expected InitialDomainExceedsUsize, got {err:?}"
        );
    }

    #[test]
    fn config_rejects_per_round_factors_that_under_fold() {
        // Invariant: a per-round schedule that under-folds is rejected at config construction.
        //
        // Fixture state:
        //   num_variables = 20, direct-send threshold = 6
        //   PerRound([3, 2]) folds only 3 + 2 = 5 variables
        //
        //     remaining = 20 - 5 = 15 > 6  ->  under-folds
        let mut params = default_whir_params();
        params.folding_factor = FoldingFactor::PerRound(vec![3, 2]);

        let err = WhirConfig::<F, F, MyChallenger>::new(20, params)
            .expect_err("per-round factors under-fold; config must be rejected");

        // The folding-factor error is forwarded through the config error type,
        // carrying the variable accounting that explains the rejection.
        match err {
            WhirConfigError::FoldingFactor(FoldingFactorError::InsufficientFolding {
                num_variables,
                remaining,
                threshold,
            }) => {
                assert_eq!(num_variables, 20);
                // 20 variables minus the 3 + 2 folded by the schedule.
                assert_eq!(remaining, 15);
                // The direct-send threshold the schedule fails to reach.
                assert_eq!(threshold, 6);
            }
            other => panic!("expected InsufficientFolding, got {other:?}"),
        }
    }

    #[test]
    fn new_rejects_zero_starting_rate() {
        // Invariant: a code rate of 2^-0 = 1 has no Reed-Solomon redundancy.
        //
        //     rate 1 -> delta <= 0 -> zero queries -> proximity test checks nothing
        //
        // Mutation: drop the starting rate to 0.
        let mut params = default_whir_params();
        params.starting_log_inv_rate = 0;

        // Construction must reject a zero starting rate before deriving rounds.
        let err = WhirConfig::<F, F, MyChallenger>::new(10, params)
            .expect_err("rate 2^-0 = 1 must be rejected");
        assert!(
            matches!(
                err,
                WhirConfigError::NonRedundantStartingRate { log_inv_rate: 0 }
            ),
            "expected NonRedundantStartingRate, got {err:?}"
        );
    }

    #[test]
    fn new_rejects_zero_explicit_round_rate() {
        // Invariant: an explicit per-round rate of 2^-0 = 1 is rejected.
        //
        //     16 variables, fold 4 per round -> 2 intermediate rounds
        //     per-round rates [3, 0]         -> round 1 gets rate 1
        //
        // Mutation: set the second round's rate to 0.
        let mut params = default_whir_params();
        params.folding_factor = FoldingFactor::Constant(4);
        params.round_log_inv_rates = vec![3, 0];

        // Rejection names the offending round and its rate.
        let err = WhirConfig::<F, F, MyChallenger>::new(16, params)
            .expect_err("explicit round rate 2^-0 = 1 must be rejected");
        assert!(
            matches!(
                err,
                WhirConfigError::NonRedundantRoundRate {
                    round: 1,
                    log_inv_rate: 0
                }
            ),
            "expected NonRedundantRoundRate at round 1, got {err:?}"
        );
    }

    #[test]
    fn valid_rates_yield_positive_query_counts() {
        // Invariant: a redundant rate keeps delta > 0, so every phase queries.
        //
        //     rate <= 1/2 -> delta > 0 -> positive query count per phase
        //
        // A quartic extension keeps out-of-domain sampling feasible.
        // So all three soundness regimes reach the query derivation.
        for soundness_type in [
            SecurityAssumption::UniqueDecoding,
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ] {
            // Valid, redundant config at rate 2^-1 = 1/2.
            let params = ProtocolParameters {
                security_level: 100,
                pow_bits: 20,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type,
                starting_log_inv_rate: 1,
            };
            let config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

            // Final proximity phase must spot-check at least one position.
            assert!(
                config.final_queries > 0,
                "{soundness_type:?}: final_queries must be positive"
            );
            // Every intermediate round must spot-check at least one position.
            for (round, cfg) in config.round_parameters.iter().enumerate() {
                assert!(
                    cfg.num_queries > 0,
                    "{soundness_type:?} round {round}: num_queries must be positive"
                );
            }
        }
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        assert_eq!(config.n_rounds(), config.round_parameters.len());
    }

    #[test]
    fn final_round_config_num_variables_equals_final_sumcheck_rounds() {
        // Invariant: the final-round config exposes exactly `final_sumcheck_rounds` variables.
        //
        // Three places key off this single count and must agree:
        //
        //     prover     : sends 2^count final evaluations in the clear
        //     verifier   : length-checks the final polynomial against 2^count
        //     transcript : absorbs 2^count final coefficients
        //
        // Sweep every folding strategy across the direct-send threshold so both branches are hit.
        fn check(num_variables: usize, folding_factor: FoldingFactor) {
            // UniqueDecoding needs no out-of-domain samples, so construction always succeeds here.
            let params = ProtocolParameters {
                security_level: 100,
                pow_bits: 20,
                round_log_inv_rates: vec![],
                folding_factor,
                soundness_type: SecurityAssumption::UniqueDecoding,
                starting_log_inv_rate: 1,
            };
            // Construction runs the same assertion internally.
            // This explicit check documents and pins the invariant.
            let config = WhirConfig::<EF4, F, MyChallenger>::new(num_variables, params).unwrap();
            assert_eq!(
                config.final_round_config().num_variables,
                config.final_sumcheck_rounds,
                "num_variables = {num_variables}"
            );
        }

        // Constant: small sizes stay in the empty branch, larger ones reach the non-empty branch.
        for nv in 4..=24 {
            check(nv, FoldingFactor::Constant(4));
        }
        for nv in 2..=24 {
            check(nv, FoldingFactor::Constant(2));
        }

        // ConstantFromSecondRound: a larger first fold, then smaller folds.
        for nv in 5..=24 {
            check(nv, FoldingFactor::ConstantFromSecondRound(3, 2));
        }

        // PerRound: explicit schedules whose length is num_rounds + 1.
        //
        //     [3, 2]    @ 10  ->  1 intermediate round
        //     [4, 3, 2] @ 14  ->  2 intermediate rounds
        check(10, FoldingFactor::PerRound(vec![3, 2]));
        check(14, FoldingFactor::PerRound(vec![4, 3, 2]));
    }

    #[test]
    fn constant_schedule_uses_smaller_final_pre_direct_fold() {
        // Thomas's motivating case:
        //
        //     m = 15, Constant(8)  ->  concrete schedule [8, 7]
        //
        // Rejecting this would forbid a valid WHIR instance. The derived
        // schedule must be used everywhere downstream, not only in validation.
        let params = ProtocolParameters {
            security_level: 100,
            pow_bits: 20,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(8),
            soundness_type: SecurityAssumption::UniqueDecoding,
            starting_log_inv_rate: 1,
        };

        let config = WhirConfig::<EF4, F, MyChallenger>::new(15, params).unwrap();

        assert_eq!(config.folding_schedule, vec![8, 7]);
        assert_eq!(config.n_rounds(), 1);
        assert_eq!(config.round_folding_factor(0), 8);
        assert_eq!(config.round_folding_factor(1), 7);
        assert_eq!(config.total_folded_through(0), 8);
        assert_eq!(config.total_folded_through(1), 15);
        assert_eq!(config.round_parameters[0].folding_factor, 8);
        assert_eq!(config.final_sumcheck_rounds, 0);
        assert_eq!(config.final_round_config().folding_factor, 7);
        assert_eq!(config.final_round_config().num_variables, 0);
    }

    #[test]
    fn test_explicit_round_log_inv_rates() {
        let mut params = default_whir_params();
        params.folding_factor = FoldingFactor::Constant(4);
        params.round_log_inv_rates = vec![3, 2];

        let config = WhirConfig::<EF4, F, MyChallenger>::new(16, params).unwrap();

        assert_eq!(config.round_parameters[0].log_inv_rate, 3);
        assert_eq!(config.round_parameters[1].log_inv_rate, 2);
        assert_eq!(config.rs_reduction_factor(0), 2);
        assert_eq!(config.rs_reduction_factor(1), 5);
        assert_eq!(config.inv_rate(0), 1 << 3);
        assert_eq!(config.inv_rate(1), 1 << 2);
    }

    #[test]
    fn test_check_pow_bits_within_limits() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        // Set all values within limits
        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        // Ensure all rounds are within limits
        config.round_parameters = vec![
            RoundConfig {
                pow_bits: 17,
                folding_pow_bits: 19,
                num_queries: 5,
                ood_samples: 2,
                num_variables: 10,
                folding_factor: 2,
                log_inv_rate: 1,
                domain_size: 10,
                folded_domain_gen: F::from_u64(2),
            },
            RoundConfig {
                pow_bits: 18,
                folding_pow_bits: 19,
                num_queries: 6,
                ood_samples: 2,
                num_variables: 10,
                folding_factor: 2,
                log_inv_rate: 1,
                domain_size: 10,
                folded_domain_gen: F::from_u64(2),
            },
        ];

        assert!(
            config.check_pow_bits(),
            "All values are within limits, check_pow_bits should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_starting_folding_exceeds() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 21; // Exceeds max_pow_bits
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        assert!(
            !config.check_pow_bits(),
            "Starting folding pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_final_pow_exceeds() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 21; // Exceeds max_pow_bits
        config.final_folding_pow_bits = 19;

        assert!(
            !config.check_pow_bits(),
            "Final pow bits exceeds max_pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_pow_exceeds() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        // One round's pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 21, // Exceeds pow_bits
            folding_pow_bits: 19,
            num_queries: 5,
            ood_samples: 2,
            num_variables: 10,
            folding_factor: 2,
            log_inv_rate: 1,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has pow_bits exceeding pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_round_folding_pow_exceeds() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 15;
        config.final_pow_bits = 18;
        config.final_folding_pow_bits = 19;

        // One round's folding_pow_bits exceeds limit
        config.round_parameters = vec![RoundConfig {
            pow_bits: 19,
            folding_pow_bits: 21, // Exceeds pow_bits
            num_queries: 5,
            ood_samples: 2,
            num_variables: 10,
            folding_factor: 2,
            log_inv_rate: 1,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "A round has folding_pow_bits exceeding pow_bits, should return false."
        );
    }

    #[test]
    fn test_check_pow_bits_exactly_at_limit() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 20;
        config.final_pow_bits = 20;
        config.final_folding_pow_bits = 20;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 20,
            folding_pow_bits: 20,
            num_queries: 5,
            ood_samples: 2,
            num_variables: 10,
            folding_factor: 2,
            log_inv_rate: 1,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            config.check_pow_bits(),
            "All pow_bits are exactly at pow_bits, should return true."
        );
    }

    #[test]
    fn test_check_pow_bits_all_exceed() {
        let params = default_whir_params();
        let mut config = WhirConfig::<EF4, F, MyChallenger>::new(10, params).unwrap();

        config.params.pow_bits = 20;
        config.starting_folding_pow_bits = 22;
        config.final_pow_bits = 23;
        config.final_folding_pow_bits = 24;

        config.round_parameters = vec![RoundConfig {
            pow_bits: 25,
            folding_pow_bits: 26,
            num_queries: 5,
            ood_samples: 2,
            num_variables: 10,
            folding_factor: 2,
            log_inv_rate: 1,
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "All values exceed max_pow_bits, should return false."
        );
    }
}
