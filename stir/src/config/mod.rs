use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter, Result};

use itertools::Itertools;
use p3_challenger::FieldChallenger;
use p3_field::{Field, TwoAdicField};

use crate::utils::{compute_pow, observe_small_usize_slice};
use crate::SecurityAssumption;

#[cfg(test)]
mod tests;

/// STIR-related parameters defined by the user. These get expanded into a full `StirConfig`.
#[derive(Debug, Clone)]
pub struct StirParameters<M: Clone> {
    // There is/are
    // - num_rounds rounds: num_rounds - 1 of them happen inside the main loop,
    //   and the final one happens after it.
    // - one folded polynomial per round.
    // - num_rounds codewords:
    //   - one encoding the original polynomial's, which is not folded
    //   - num_rounds - 1 ones encoding folded polynomials (note that, in the
    //     last round, the folded polynomialis sent in plain).
    /// Security level desired in bits.
    pub(crate) security_level: usize,

    /// Security assumption under which to configure FRI.
    pub(crate) security_assumption: SecurityAssumption,

    /// log of the purported (degree + 1) of the initial polynomial.
    // Note: In variable or field names, "degree" refers to "degree bound (<=) +
    // 1". Comments are more accurate: "degree" means "degree bound (<=)"
    pub(crate) log_starting_degree: usize,

    /// log of the folding factor in each round (incl. final).
    pub(crate) log_folding_factors: Vec<usize>,

    /// log of the inverse of the starting rate used in the protocol.
    pub(crate) log_starting_inv_rate: usize,

    /// log of the inverses of the rates in non-first protocol codewords (incl. final). There are num_rounds - 1 of these.
    pub(crate) log_inv_rates: Vec<usize>,

    /// Number of PoW bits used to reduce query error.
    pub(crate) pow_bits: usize,

    /// Configuration of the Mixed Matrix Commitment Scheme (hasher and compressor).
    pub(crate) mmcs_config: M,
}

// Convenience methods to create STIR parameters with typical features
impl<M: Clone> StirParameters<M> {
    /// Create a STIR configuration with constant folding factor.
    pub fn fixed_rate(
        log_starting_degree: usize,
        log_starting_inv_rate: usize,
        log_inv_rates: Vec<usize>,
        log_folding_factor: usize,
        security_assumption: SecurityAssumption,
        security_level: usize,
        pow_bits: usize,
        mmcs_config: M,
    ) -> Self {
        StirParameters {
            log_starting_degree,
            log_starting_inv_rate,
            log_folding_factors: vec![log_folding_factor; log_inv_rates.len()],
            log_inv_rates,
            security_assumption,
            security_level,
            pow_bits,
            mmcs_config,
        }
    }

    /// Create a STIR configuration where, in each round,
    /// - the size of the (shifted) domain is reduced by a factor of 2.
    /// - the degree is reduced by a factor of 2^log_folding_factor.
    pub fn fixed_domain_shift(
        log_starting_degree: usize,
        log_starting_inv_rate: usize,
        log_folding_factor: usize,
        num_rounds: usize,
        security_assumption: SecurityAssumption,
        security_level: usize,
        pow_bits: usize,
        mmcs_config: M,
    ) -> Self {
        StirParameters {
            log_starting_degree,
            log_folding_factors: vec![log_folding_factor; num_rounds],
            log_starting_inv_rate,
            log_inv_rates: (0..num_rounds)
                .map(|i| log_starting_inv_rate + (i + 1) * (log_folding_factor - 1))
                .collect(),
            security_assumption,
            security_level,
            pow_bits,
            mmcs_config,
        }
    }
}

/// Round specific configuration
#[derive(Debug, Clone)]
pub struct RoundConfig {
    /// log of the folding factor for this round.
    pub(crate) log_folding_factor: usize,

    /// log of the folding factor for the next round.
    pub(crate) log_next_folding_factor: usize,

    /// log of the size of the evaluation domain of the oracle *sent this round*
    pub(crate) log_evaluation_domain_size: usize,

    /// Number of PoW bits used to reduce query error.
    pub(crate) pow_bits: usize,

    /// Number of queries in this round
    pub(crate) num_queries: usize,

    /// Number of out of domain samples in this round
    pub(crate) num_ood_samples: usize,

    /// log of the inverse of the rate of the current RS codeword
    pub(crate) log_inv_rate: usize,
}

#[derive(Debug, Clone)]
pub struct StirConfig<F: TwoAdicField, M: Clone> {
    // See the comment at the start of StirParameters for the convention on the
    // number of rounds, codewords, etc. In this structure there are
    // num_rounds - 1 round configs as the last round happening outside the
    // main loop doesn't have one.
    /// Input parameters for the STIR protocol.
    parameters: StirParameters<M>,

    /// log of the size of the initial domain.
    starting_domain_log_size: usize,

    /// Initial pow bits used in the first fold.
    starting_folding_pow_bits: usize,

    /// Round-specific parameters. There are `num_rounds - 1` of these (the last
    /// round works differently)
    round_parameters: Vec<RoundConfig>,

    /// log of the (degree + 1) of the final polynomial sent in plain.
    log_stopping_degree: usize,

    /// log of the inverse of therate of the final RS codeword.
    log_final_inv_rate: usize,

    /// Number of queries in the last round.
    final_num_queries: usize,

    /// Final of PoW bits (for the queries).
    final_pow_bits: usize,

    /// Generator of the (subgroup whose shift is the) initial domain, kept
    // throughout rounds for shifting purposes
    subgroup_generator: F,
}

impl<F: TwoAdicField, M: Clone> StirConfig<F, M> {
    /// Expand STIR parameters into a full STIR configuration.
    pub fn new(parameters: StirParameters<M>) -> Self {
        let StirParameters {
            security_level,
            security_assumption,
            log_starting_degree,
            log_folding_factors,
            log_starting_inv_rate,
            log_inv_rates,
            pow_bits,
            ..
        } = parameters.clone();

        assert!(
            log_folding_factors.iter().all(|&x| x != 0),
            "The logarithm of each folding factor should be positive"
        );
        assert_eq!(log_folding_factors.len(), log_inv_rates.len());

        // log(degree + 1) can not be reduced past 0
        // This also ensures the domain is large enough to be (actually) shrunk
        // by raising to the subsequent folding factors
        let total_reduction = log_folding_factors.iter().sum::<usize>();
        assert!(total_reduction <= log_starting_degree);

        let log_starting_folding_factor = log_folding_factors[0];

        // If the first round wants to reduce the degree more than possible, one
        // should send the polynomial directly instead
        assert!(log_starting_degree >= log_starting_folding_factor);

        // Compute the log of (final degree + 1) as well as the number of (non-final) rounds
        let log_stopping_degree = log_starting_degree - total_reduction;
        let num_rounds = log_folding_factors.len();

        // Compute the security level
        let protocol_security_level = 0.max(security_level - pow_bits);

        // Initial domain size
        let starting_domain_log_size =
            parameters.log_starting_degree + parameters.log_starting_inv_rate;

        // Degree of next polynomial to send
        let mut current_log_degree = log_starting_degree - log_starting_folding_factor;
        let mut log_inv_rate = log_starting_inv_rate;

        // We now start, the initial folding pow bits
        let field_bits = F::bits();

        let starting_folding_prox_gaps_error = security_assumption.prox_gaps_error(
            current_log_degree,
            log_inv_rate,
            field_bits,
            1 << log_starting_folding_factor,
        );

        let starting_folding_pow_bits =
            compute_pow(security_level, starting_folding_prox_gaps_error).ceil() as usize;

        let mut round_parameters = Vec::with_capacity(num_rounds);

        // If folding factors has length (i. e. num_rounds) 1, the only round is
        // by definition the last one, which is treated separately; In that
        // case, windows(2) returns no elements, as desired.
        for (log_folding_factor_pair, next_rate) in log_folding_factors
            .windows(2)
            .into_iter()
            .zip(log_inv_rates)
        {
            let (log_curr_folding_factor, log_next_folding_factor) =
                (log_folding_factor_pair[0], log_folding_factor_pair[1]);

            // This is the size of the new evaluation domain
            let new_evaluation_domain_size = current_log_degree + next_rate;

            // Compute the ood samples required
            let num_ood_samples = security_assumption.determine_ood_samples(
                security_level,
                current_log_degree,
                next_rate,
                field_bits,
            );

            // Compute the number of queries required
            let num_queries = security_assumption.queries(protocol_security_level, log_inv_rate);

            // We need to compute the errors, to compute the according PoW
            let query_error = security_assumption.queries_error(log_inv_rate, num_queries);

            let num_terms = num_queries + num_ood_samples;
            let prox_gaps_error_1 = parameters.security_assumption.prox_gaps_error(
                current_log_degree,
                next_rate,
                field_bits,
                num_terms,
            );

            let prox_gaps_error_2 = security_assumption.prox_gaps_error(
                current_log_degree - log_curr_folding_factor,
                next_rate,
                field_bits,
                1 << log_curr_folding_factor,
            );

            // Now compute the PoW
            let pow_bits = compute_pow(
                security_level,
                query_error.min(prox_gaps_error_1).min(prox_gaps_error_2),
            )
            .ceil() as usize;

            let round_config = RoundConfig {
                log_evaluation_domain_size: new_evaluation_domain_size,
                log_folding_factor: log_curr_folding_factor,
                log_next_folding_factor,
                num_queries,
                pow_bits,
                num_ood_samples,
                log_inv_rate,
            };
            round_parameters.push(round_config);
            log_inv_rate = next_rate;
            current_log_degree -= log_curr_folding_factor;
        }

        // Compute the number of queries required
        let final_num_queries = parameters
            .security_assumption
            .queries(protocol_security_level, log_inv_rate);

        // We need to compute the errors, to compute the according PoW
        let query_error = parameters
            .security_assumption
            .queries_error(log_inv_rate, final_num_queries);

        // Now compute the PoW
        let final_pow_bits = compute_pow(security_level, query_error).ceil() as usize;

        StirConfig {
            parameters,
            starting_domain_log_size,
            starting_folding_pow_bits,
            round_parameters,
            log_stopping_degree,
            log_final_inv_rate: log_inv_rate,
            final_num_queries,
            final_pow_bits,
            subgroup_generator: F::two_adic_generator(starting_domain_log_size),
        }
    }

    // Getters for all internal fields
    pub fn parameters(&self) -> &StirParameters<M> {
        &self.parameters
    }

    pub fn starting_domain_log_size(&self) -> usize {
        self.starting_domain_log_size
    }

    pub fn starting_folding_pow_bits(&self) -> usize {
        self.starting_folding_pow_bits
    }

    pub fn num_rounds(&self) -> usize {
        // See the comment at the start of StirParameters for the convention
        self.round_parameters.len() + 1
    }

    /// Configurations of non-final rounds (i. e. the ones which happen inside
    /// the main loop)
    pub fn round_configs(&self) -> &[RoundConfig] {
        &self.round_parameters
    }

    /// Returns the configuration of the i-th round (from 1 to `num_rounds - 1`:
    /// the last round has no `RoundConfig`)
    pub(crate) fn round_config(&self, i: usize) -> &RoundConfig {
        assert!(i > 0, "Rounds are numbered starting at i = 1");

        self.round_parameters
            .get(i - 1)
            // More optimal than .expect(format!...)
            .unwrap_or_else(|| {
                panic!(
                    "Index out of bounds: there are {} rounds, but only {} round \
                    configurations (the final round does not have one)",
                    self.num_rounds(),
                    self.round_parameters.len(),
                )
            })
    }

    pub fn log_stopping_degree(&self) -> usize {
        self.log_stopping_degree
    }

    pub fn final_log_inv_rate(&self) -> usize {
        self.log_final_inv_rate
    }

    pub fn final_num_queries(&self) -> usize {
        self.final_num_queries
    }

    pub fn final_pow_bits(&self) -> usize {
        self.final_pow_bits
    }

    // Getters for all fields of the internal StirParameters
    pub fn security_level(&self) -> usize {
        self.parameters.security_level
    }

    pub fn security_assumption(&self) -> SecurityAssumption {
        self.parameters.security_assumption
    }

    pub fn log_starting_degree(&self) -> usize {
        self.parameters.log_starting_degree
    }

    pub fn log_starting_folding_factor(&self) -> usize {
        self.parameters.log_folding_factors[0]
    }

    pub fn log_folding_factors(&self) -> &[usize] {
        &self.parameters.log_folding_factors
    }

    pub fn log_last_folding_factor(&self) -> usize {
        *self.parameters.log_folding_factors.last().unwrap()
    }

    pub fn log_starting_inv_rate(&self) -> usize {
        self.parameters.log_starting_inv_rate
    }

    pub fn log_inv_rates(&self) -> &[usize] {
        &self.parameters.log_inv_rates
    }

    pub fn pow_bits(&self) -> usize {
        self.parameters.pow_bits
    }

    pub fn pow_bits_all_rounds(&self) -> Vec<usize> {
        let mut pow_bits = vec![self.starting_folding_pow_bits];
        pow_bits.extend(self.round_parameters.iter().map(|x| x.pow_bits));
        pow_bits.push(self.final_pow_bits);
        pow_bits
    }

    pub fn subgroup_generator(&self) -> F {
        self.subgroup_generator
    }

    pub fn mmcs_config(&self) -> &M {
        &self.parameters.mmcs_config
    }
}

impl<M: Clone> Display for StirParameters<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "StirParameters\n\
            \t- security level: {} bits\n\
            \t- security assumption for the code: {}\n\
            \t- log of (starting degree bound + 1): {}\n\
            \t- log of the folding factors: {}\n\
            \t- log of the starting inverse rate: {}\n\
            \t- log of inverse rates for non-first codewords: {}\n\
            \t- proof-of-work bits: {}\n",
            self.security_level,
            self.security_assumption,
            self.log_starting_degree,
            self.log_folding_factors
                .iter()
                .map(|x| format!("{}", x))
                .collect_vec()
                .join(", "),
            self.log_starting_inv_rate,
            self.log_inv_rates
                .iter()
                .map(|x| format!("{}", x))
                .collect_vec()
                .join(", "),
            self.pow_bits
        )
    }
}

impl<F: TwoAdicField, M: Clone> Display for StirConfig<F, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "StirConfig with:\n* {}\n\
            * Other parameters:\n\
            \t- starting domain log size: {}\n\
            \t- starting folding pow bits: {}\n\
            \t- log of (stopping degree bound + 1): {}\n\n\
            {}\n\n\
            * Final round parameters:\n\n\
            \t- log of final inverse rate: {}\n\
            \t- final number of queries: {}\n\
            \t- final proof-of-work bits: {}\n\n",
            self.parameters,
            self.starting_domain_log_size,
            self.starting_folding_pow_bits,
            self.log_stopping_degree,
            self.round_parameters
                .iter()
                .enumerate()
                .map(|(i, x)| format!("* Round i = {} parameters: {}", i + 1, x))
                .collect_vec()
                .join("\n\n"),
            self.log_final_inv_rate,
            self.final_num_queries,
            self.final_pow_bits,
        )
    }
}

impl Display for RoundConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "RoundConfig\n\n\
            \t- log of folding factor: {}\n\
            \t- log of next folding factor: {}\n\
            \t- log of evaluation domain size: {}\n\
            \t- number of queries: {}\n\
            \t- number of OOD samples: {}\n\
            \t- log of inverse rate: {}\n\
            \t- proof of work bits: {}\n",
            self.log_folding_factor,
            self.log_next_folding_factor,
            self.log_evaluation_domain_size,
            self.num_queries,
            self.num_ood_samples,
            self.log_inv_rate,
            self.pow_bits
        )
    }
}

// Have the challenger observe the public parameters
pub(crate) fn observe_public_parameters<F, M>(
    parameters: &StirParameters<M>,
    challenger: &mut impl FieldChallenger<F>,
) where
    F: Field,
    M: Clone,
{
    observe_small_usize_slice(
        challenger,
        &[
            parameters.security_level,
            parameters.security_assumption as usize,
            parameters.log_starting_degree,
            parameters.log_starting_inv_rate,
            parameters.pow_bits,
        ],
        false,
    );
    observe_small_usize_slice(challenger, &parameters.log_folding_factors, false);
    observe_small_usize_slice(challenger, &parameters.log_inv_rates, false);

    // We do not absorb the MMCS configuration, as it would require stringent
    // trait bounds
}
