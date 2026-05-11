//! Derived WHIR protocol configuration computed from user-facing parameters.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};

use super::ProtocolParameters;
use crate::pcs::proof::WhirProof;

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

impl<EF, F, Challenger> WhirConfig<EF, F, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Construct an empty proof with this configuration.
    pub fn empty_proof<MT: Mmcs<F>>(&self) -> WhirProof<F, EF, MT> {
        WhirProof::from_protocol_parameters(&self.params, self.num_variables)
    }

    /// Derive a full protocol configuration from user-facing parameters.
    #[allow(clippy::too_many_lines)]
    pub fn new(num_variables: usize, whir_parameters: ProtocolParameters) -> Self {
        // ---------------------------------------------------------------
        // Phase 1: Validate inputs and set up global constants.
        // ---------------------------------------------------------------

        // Preserve the original variable count; the mutable copy below
        // tracks the remaining variables as we consume them round by round.
        let initial_num_variables = num_variables;

        // Reject folding factors that are incompatible with the polynomial size.
        whir_parameters
            .folding_factor
            .check_validity(num_variables)
            .unwrap();

        // The domain reduction at round 0 must not exceed the folding factor,
        // otherwise the code rate would *increase*, weakening soundness.
        assert!(
            whir_parameters.rs_domain_initial_reduction_factor
                <= whir_parameters.folding_factor.at_round(0),
            "Increasing the code rate is not a good idea"
        );

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
        let log_domain_size = num_variables + log_inv_rate;
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
        let log_folded_domain_size = log_domain_size - whir_parameters.folding_factor.at_round(0);
        assert!(
            log_folded_domain_size <= F::TWO_ADICITY,
            "Increase folding_factor_0"
        );

        // ---------------------------------------------------------------
        // Phase 3: Determine round structure.
        // ---------------------------------------------------------------

        // How many intermediate STIR rounds, and how many variables remain
        // for the final direct-send sumcheck.
        let (num_rounds, final_sumcheck_rounds) = whir_parameters
            .folding_factor
            .compute_number_of_rounds(num_variables);

        // OOD samples for the commitment phase (before any folding).
        let commitment_ood_samples = whir_parameters.soundness_type.determine_ood_samples(
            whir_parameters.security_level,
            num_variables,
            log_inv_rate,
            field_size_bits,
        );

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
        //   1. Computes the new code rate after folding.
        //   2. Determines query count from the old rate (queries test
        //      proximity to the code *before* this round's fold).
        //   3. Determines OOD sample count from the new rate.
        //   4. Derives PoW for both the query and folding sub-steps.
        //   5. Records the domain generator for the folded evaluation domain.

        let mut round_parameters = Vec::with_capacity(num_rounds);

        // Subtract the first-round folding factor; the loop below
        // handles subsequent rounds.
        num_variables -= whir_parameters.folding_factor.at_round(0);

        for round in 0..num_rounds {
            // Only round 0 applies the user-configured domain reduction;
            // all later rounds halve the domain (reduction factor = 1).
            let rs_reduction_factor = if round == 0 {
                whir_parameters.rs_domain_initial_reduction_factor
            } else {
                1
            };

            // The code rate increases by (folding_factor - rs_reduction_factor) bits.
            // Queries use the *old* rate; OOD and folding use the *new* rate.
            let next_rate = log_inv_rate
                + (whir_parameters.folding_factor.at_round(round) - rs_reduction_factor);

            // Number of STIR proximity queries at the current (old) rate.
            let num_queries = whir_parameters
                .soundness_type
                .queries(protocol_security_level, log_inv_rate);

            // OOD samples needed at the post-fold (new) rate.
            let ood_samples = whir_parameters.soundness_type.determine_ood_samples(
                whir_parameters.security_level,
                num_variables,
                next_rate,
                field_size_bits,
            );

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

            let folding_factor = whir_parameters.folding_factor.at_round(round);
            let next_folding_factor = whir_parameters.folding_factor.at_round(round + 1);

            // Generator of the two-adic subgroup for the folded domain.
            let folded_domain_gen =
                F::two_adic_generator(domain_size.ilog2() as usize - folding_factor);

            round_parameters.push(RoundConfig {
                pow_bits: pow_bits as usize,
                folding_pow_bits: folding_pow_bits as usize,
                num_queries,
                ood_samples,
                num_variables,
                folding_factor,
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
            whir_parameters
                .folding_factor
                .total_number(round_parameters.len())
                + final_sumcheck_rounds
        );

        Self {
            params: whir_parameters,
            commitment_ood_samples,
            num_variables: initial_num_variables,
            starting_folding_pow_bits: starting_folding_pow_bits as usize,
            round_parameters,
            final_queries,
            final_pow_bits: final_pow_bits as usize,
            final_sumcheck_rounds,
            final_folding_pow_bits: final_folding_pow_bits as usize,
            _extension_field: PhantomData,
            _challenger: PhantomData,
        }
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
    ///
    /// The first round uses the user-configured initial reduction factor.
    /// All subsequent rounds halve the domain (factor = 1).
    pub const fn rs_reduction_factor(&self, round: usize) -> usize {
        if round == 0 {
            self.params.rs_domain_initial_reduction_factor
        } else {
            1
        }
    }

    /// Returns the log2 size of the largest FFT
    /// (At commitment we perform 2^folding_factor FFT of size 2^max_fft_size)
    pub const fn max_fft_size(&self) -> usize {
        self.num_variables + self.params.starting_log_inv_rate - self.folding_factor(0)
    }

    /// Returns whether all PoW difficulties are within the configured maximum.
    ///
    /// Checks the starting, final, and per-round PoW bits against the ceiling.
    /// Returns false if any value exceeds the limit.
    pub fn check_pow_bits(&self) -> bool {
        let max_bits = self.params.pow_bits;

        // Check the main pow bits values
        if self.starting_folding_pow_bits > max_bits
            || self.final_pow_bits > max_bits
            || self.final_folding_pow_bits > max_bits
        {
            return false;
        }

        // Check all round parameters
        self.round_parameters
            .iter()
            .all(|r| r.pow_bits <= max_bits && r.folding_pow_bits <= max_bits)
    }

    /// Retrieves the folding factor for a given round.
    pub const fn folding_factor(&self, round: usize) -> usize {
        self.params.folding_factor.at_round(round)
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
                num_variables: self.num_variables - self.folding_factor(0),
                folding_factor: self.folding_factor(self.n_rounds()),
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                domain_size: self.starting_domain_size(),
                folded_domain_gen: F::two_adic_generator(
                    self.starting_domain_size().ilog2() as usize - self.folding_factor(0),
                ),
                ood_samples: 0,
                folding_pow_bits: self.final_folding_pow_bits,
            }
        } else {
            // Apply the last round's domain reduction to get the domain
            // size entering the final phase.
            let rs_reduction_factor = self.rs_reduction_factor(self.n_rounds() - 1);
            let folding_factor = self.folding_factor(self.n_rounds());

            let last = self.round_parameters.last().unwrap();

            // The domain shrinks by the RS reduction factor from the last round.
            let domain_size = last.domain_size >> rs_reduction_factor;

            // Generator for the final folded domain.
            let folded_domain_gen = F::two_adic_generator(
                domain_size.ilog2() as usize - self.folding_factor(self.n_rounds()),
            );

            RoundConfig {
                // Variables remaining after this final fold.
                num_variables: last.num_variables - folding_factor,
                folding_factor,
                num_queries: self.final_queries,
                pow_bits: self.final_pow_bits,
                domain_size,
                folded_domain_gen,
                // Inherit OOD count from the last intermediate round.
                ood_samples: last.ood_samples,
                folding_pow_bits: self.final_folding_pow_bits,
            }
        }
    }

    /// Returns the inverse rate of the RS code at the given round.
    ///
    /// The inverse rate is `domain_size / degree`, where:
    /// - `domain_size` is the evaluation domain after the round's reduction.
    /// - `degree` is 2^(remaining variables after all folds up to this round).
    ///
    /// ```text
    /// inv_rate = (round_domain_size >> rs_reduction) / 2^(num_variables - total_folded)
    /// ```
    pub fn inv_rate(&self, round: usize) -> usize {
        // Shrink the domain by this round's reduction factor.
        let domain_reduction = 1 << self.rs_reduction_factor(round);
        let new_domain_size = self.round_parameters[round].domain_size / domain_reduction;

        // Number of polynomial evaluations (= degree) after all folds so far.
        let num_evals = 1 << (self.num_variables - self.params.folding_factor.total_number(round));

        // Ratio gives the inverse rate.
        new_domain_size / num_evals
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::parameters::{FoldingFactor, SecurityAssumption};

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Generates default WHIR parameters
    fn default_whir_params() -> ProtocolParameters {
        ProtocolParameters {
            security_level: 100,
            pow_bits: 20,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        }
    }

    #[test]
    fn test_whir_config_creation() {
        let params = default_whir_params();

        let config = WhirConfig::<F, F, MyChallenger>::new(10, params);

        assert_eq!(config.security_level, 100);
        assert_eq!(config.params.pow_bits, 20);
        assert_eq!(config.soundness_type, SecurityAssumption::CapacityBound);
    }

    #[test]
    fn test_n_rounds() {
        let params = default_whir_params();
        let config = WhirConfig::<F, F, MyChallenger>::new(10, params);

        assert_eq!(config.n_rounds(), config.round_parameters.len());
    }

    #[test]
    fn test_check_pow_bits_within_limits() {
        let params = default_whir_params();
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
        let mut config = WhirConfig::<F, F, MyChallenger>::new(10, params);

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
            domain_size: 10,
            folded_domain_gen: F::from_u64(2),
        }];

        assert!(
            !config.check_pow_bits(),
            "All values exceed max_pow_bits, should return false."
        );
    }
}
