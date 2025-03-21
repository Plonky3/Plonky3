use alloc::vec::Vec;
use alloc::{format, vec};
use core::fmt::{Debug, Display, Formatter, Result};

use itertools::Itertools;
use p3_challenger::FieldChallenger;
use p3_field::Field;

use crate::utils::{compute_pow, observe_usize_slice};
use crate::SecurityAssumption;

#[cfg(test)]
mod tests;

/// STIR-related parameters chosen by the user. These get expanded into a full
/// [`StirConfig`] by the function [`StirConfig::new`].
///
/// In STIR, there are:
/// - `M + 1` rounds (in the notation of the article): `M` full rounds happen
///   inside the main loop, and the final, shorter one happens after it. Rounds
///   are numbered from 1 to `M + 1`.
/// - one folded polynomial `g_i` per round. The last polynomial `g_{M + 1}` is
///   denoted `p` in the article.
/// - `M` codewords:
///   - One encoding the original polynomial `f_0`, which is not folded.
///   - `M - 1` encoding folded polynomials (note that, in the last round, the
///     folded polynomial is sent in plain).
#[derive(Debug, Clone)]
pub struct StirParameters<M: Clone> {
    /// Desired number of bits of security.
    pub security_level: usize,

    /// Code-distance assumption to configure STIR with. The more assumptions,
    /// the fewer queries and proof-of-work bits. Cf. [`SecurityAssumption`] for
    /// more details.
    pub security_assumption: SecurityAssumption,

    /// log2 of the purported degree-plus-1 bound of the initial polynomial. For
    /// instance, setting this to 3 allows the prover to convince the verifier
    /// that the polynomial has degree at most `(2^3 - 1) = 7`.
    // N. B.: In variable or field names, "degree" refers to "degree bound + 1".
    // Comments and documentation are more accurate: "degree" means "degree
    // bound"
    pub log_starting_degree: usize,

    /// log2 of the folding factor `k_i` in each round (incl. final) `i = 1,
    /// ..., M + 1`.
    pub log_folding_factors: Vec<usize>,

    /// log2 of the inverse of the rate used to encode the initial polynomial
    /// `f_0`.
    pub log_starting_inv_rate: usize,

    /// log2 of the inverses of the rates in non-first protocol codewords (incl.
    /// the final one). There are M of these.
    ///
    /// In this implementation of STIR, between each round and the next, the
    /// domain size is reduced by a factor of 2 and the `degree + 1` bound is
    /// reduced by a factor of `k_i`. This means that the inverse of the rate
    /// increases by a factor of `k_i / 2`.
    pub log_inv_rates: Vec<usize>,

    /// Number of proof-of-work bits used to reduce the query error.
    pub pow_bits: usize,

    /// Configuration of the Mixed Matrix Commitment Scheme (hasher and
    /// compressor) used to commit to the initial polynomial `f_0` and round
    /// polynomials `g_1, ... g_M`.
    pub mmcs_config: M,
}

// Convenience methods to create STIR parameters with typical features
impl<M: Clone> StirParameters<M> {
    /// Create a STIR configuration where each round has a potentially different
    /// folding factor.
    ///
    /// # Parameters
    ///
    /// - `security_level`: Desired number of bits of security.
    /// - `security_assumption`: Code-distance assumption. Cf.
    ///   [`SecurityAssumption`].
    /// - `log_starting_degree`: log2 of the bound of the degree (plus one) of
    ///   the initial polynomial.
    /// - `log_starting_inv_rate`: log2 of the inverse of the rate of the
    ///   initial codeword (i. e. of the blowup factor).
    /// - `log_folding_factors`: log2 of the folding factors for each round.
    /// - `pow_bits`: Number of proof-of-work bits used to reduce the query
    ///   error.
    /// - `mmcs_config`: Configuration of the Mixed Matrix Commitment Scheme
    ///   used to commit to the initial and round polynomials.
    pub fn variable_folding_factor(
        // This is paired due to clippy disallowing functions with > 7 arguments
        // (causing ci.workflows to reject it)
        (security_level, security_assumption): (usize, SecurityAssumption),
        log_starting_degree: usize,
        log_starting_inv_rate: usize,
        log_folding_factors: Vec<usize>,
        pow_bits: usize,
        mmcs_config: M,
    ) -> Self {
        // With each subsequent round, the size of the evaluation domain is
        // decreased by a factor of 2 whereas the degree-plus-1 bound of the
        // polynomial is decreased by a factor of 2^log_folding_factor. Thus,
        // the logarithm of the inverse of the rate increases by log_k - 1.
        let mut i_th_log_rate = log_starting_inv_rate;

        let log_inv_rates = log_folding_factors
            .iter()
            .map(|log_k| {
                i_th_log_rate = i_th_log_rate + log_k - 1;
                i_th_log_rate
            })
            .collect();

        StirParameters {
            log_starting_degree,
            log_folding_factors,
            log_starting_inv_rate,
            log_inv_rates,
            security_assumption,
            security_level,
            pow_bits,
            mmcs_config,
        }
    }

    /// Create a STIR configuration where all rounds use the same folding factor
    /// `k_i = 2^log_folding_factor`.
    ///
    /// # Parameters
    ///
    /// - `security_level`: Desired number of bits of security.
    /// - `security_assumption`: Code-distance assumption. Cf.
    ///   [`SecurityAssumption`].
    /// - `log_starting_degree`: log2 of the bound of the degree (plus one) of
    ///   the initial polynomial.
    /// - `log_starting_inv_rate`: log2 of the inverse of the rate of the
    ///   initial codeword.
    /// - `log_folding_factor`: log2 of the folding factor for each round.
    /// - `num_rounds`: Number of rounds.
    /// - `pow_bits`: Number of proof-of-work bits used to reduce the query
    ///   error.
    /// - `mmcs_config`: Configuration of the Mixed Matrix Commitment Scheme
    ///   used to commit to the initial and round polynomials.
    pub fn constant_folding_factor(
        // This is paired due to clippy disallowing functions with > 7 arguments
        // (causing ci.workflows to reject it)
        (security_level, security_assumption): (usize, SecurityAssumption),
        log_starting_degree: usize,
        log_starting_inv_rate: usize,
        log_folding_factor: usize,
        num_rounds: usize,
        pow_bits: usize,
        mmcs_config: M,
    ) -> Self {
        Self::variable_folding_factor(
            (security_level, security_assumption),
            log_starting_degree,
            log_starting_inv_rate,
            vec![log_folding_factor; num_rounds],
            pow_bits,
            mmcs_config,
        )
    }
}

/// Configuration parameters specific to one round of STIR.
#[derive(Debug, Clone)]
pub struct RoundConfig {
    // log2 of the folding factor for this round.
    pub(crate) log_folding_factor: usize,

    // log2 of the folding factor for the next round.
    pub(crate) log_next_folding_factor: usize,

    // log2 of the size of the evaluation domain of the oracle *sent this
    // round*.
    pub(crate) log_evaluation_domain_size: usize,

    // Number of proof-of-work bits used to reduce query error in this round.
    pub(crate) pow_bits: usize,

    // Number of domain points queried in this round.
    pub(crate) num_queries: usize,

    // Number of out-of-domain points queried in this round.
    pub(crate) num_ood_samples: usize,

    // log of the inverse of the rate of the codeword sent this round.
    pub(crate) log_inv_rate: usize,
}

/// Full STIR configuration.
#[derive(Debug, Clone)]
pub struct StirConfig<M: Clone> {
    // See the comment at the start of StirParameters for the convention on the
    // number of rounds, codewords, etc.

    // User-defined parameters.
    parameters: StirParameters<M>,

    // log2 of the size of the initial domain L_0.
    starting_domain_log_size: usize,

    // Initial proof-of-work bits used in the first folding.
    starting_folding_pow_bits: usize,

    // Round-specific parameters. There are `num_rounds - 1` of these (the last
    // round works differently)
    round_parameters: Vec<RoundConfig>,

    // log2 of the degree-plus-1 bound of the final polynomial p = g_{M + 1}
    // sent in plain.
    log_stopping_degree: usize,

    // log2 of the inverse of therate of the final RS codeword.
    log_final_inv_rate: usize,

    // Number of domain points queried in the last round.
    final_num_queries: usize,

    // Number of proof-of-work bits for the last round.
    final_pow_bits: usize,
}

impl<M: Clone> StirConfig<M> {
    /// Expand STIR parameters into a full STIR configuration.
    pub fn new<F: Field>(parameters: StirParameters<M>) -> Self {
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

        // The polynomial has to be folded with arity at least 2 in each round
        assert!(
            log_folding_factors.iter().all(|&x| x != 0),
            "The logarithm of each folding factor should be positive"
        );
        assert_eq!(log_folding_factors.len(), log_inv_rates.len());

        // log2(degree + 1) can not be reduced past 0. This also ensures the
        // domain is large enough to be shrunk by raising it to all of the
        // subsequent folding factors iteratively.
        let total_reduction = log_folding_factors.iter().sum::<usize>();
        assert!(total_reduction <= log_starting_degree);

        let log_starting_folding_factor = log_folding_factors[0];

        // If the first round wants to reduce the degree more than possible, one
        // should send the polynomial directly instead
        assert!(log_starting_degree >= log_starting_folding_factor);

        // Compute the log of (final-degree-plus-1 bound) as well as the number
        // of (non-final) rounds
        let log_stopping_degree = log_starting_degree - total_reduction;
        let num_full_rounds = log_folding_factors.len();

        // Compute the security level afforded by the actual protocol without
        // grinding
        let protocol_security_level = 0.max(security_level - pow_bits);

        // Initial domain size
        let starting_domain_log_size =
            parameters.log_starting_degree + parameters.log_starting_inv_rate;

        // Degree of next polynomial to send
        let mut current_log_degree = log_starting_degree - log_starting_folding_factor;
        let mut log_inv_rate = log_starting_inv_rate;

        // Computing the proof-of-work bits for the initial folding
        let field_bits = F::bits();

        let starting_folding_prox_gaps_error = security_assumption.prox_gaps_error(
            current_log_degree,
            log_inv_rate,
            field_bits,
            1 << log_starting_folding_factor,
        );

        let starting_folding_pow_bits =
            compute_pow(security_level, starting_folding_prox_gaps_error).ceil() as usize;

        let mut round_parameters = Vec::with_capacity(num_full_rounds);

        // If folding factors has length (i. e. num_rounds) 1, the only round is
        // by definition the last one, which is treated separately; In that
        // case, windows(2) returns no elements, as desired.
        for (i, (log_folding_factor_pair, next_rate)) in log_folding_factors
            .windows(2)
            .zip(log_inv_rates.clone())
            .enumerate()
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

            // We need to compute the three errors from which the number of
            // proof-of-work bits is derived
            let query_error = security_assumption.queries_error(log_inv_rate, num_queries);

            let num_terms = num_queries + num_ood_samples;

            // Early termination check: if the  number of queries is large
            // enough that the quotient polynomial would be zero (i. e. greater
            // than the deg(g_i)), the protocol should terminate early. This is
            // identical to a protocol with stopping degree equal to deg(g_i)
            if num_terms > 1 << current_log_degree {
                let new_params = StirParameters {
                    log_folding_factors: log_folding_factors[0..i + 1].to_vec(),
                    log_inv_rates: log_inv_rates[0..i + 1].to_vec(),
                    ..parameters
                };

                tracing::info!(
                    "The requested configuration terminates early at round {}",
                    i + 1,
                );

                return StirConfig::new::<F>(new_params);
            }

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

            // Now compute the proof-of-work bits
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

        // We need to compute the three errors from which the final number of
        // proof-of-work bits is derived
        let query_error = parameters
            .security_assumption
            .queries_error(log_inv_rate, final_num_queries);

        // Now compute actual number of final proof-of-work bits
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
        }
    }

    /// User-defined parameters of the configuration.
    pub fn parameters(&self) -> &StirParameters<M> {
        &self.parameters
    }

    /// log2 of the size of the initial domain.
    pub fn starting_domain_log_size(&self) -> usize {
        self.starting_domain_log_size
    }

    /// Number of proof-of-work bits used in the initial folding.
    pub fn starting_folding_pow_bits(&self) -> usize {
        self.starting_folding_pow_bits
    }

    /// Number of rounds `M + 1` (`M` full and one final).
    pub fn num_rounds(&self) -> usize {
        // See the comment at the start of StirParameters for the convention
        self.round_parameters.len() + 1
    }

    /// Configurations of the `M` full rounds (the ones happening inside the
    /// the main prover/verifier loop)
    pub fn round_configs(&self) -> &[RoundConfig] {
        &self.round_parameters
    }

    /// Configuration of the i-th full round (from 1 to `M` = `num_rounds() -
    /// 1`)
    pub(crate) fn round_config(&self, i: usize) -> &RoundConfig {
        assert!(i > 0, "Rounds are numbered starting at i = 1");

        self.round_parameters
            .get(i - 1)
            // Using a closure avoids formatting the string when there is no
            // panic!, unlike expect()
            .unwrap_or_else(|| {
                panic!(
                    "Index out of bounds: there are {} rounds, but only {} round \
                    configurations (the final round does not have one)",
                    self.num_rounds(),
                    self.round_parameters.len(),
                )
            })
    }

    /// log2 of the degree-plus-1 bound of the final polynomial `p = g_{M + 1}`
    pub fn log_stopping_degree(&self) -> usize {
        self.log_stopping_degree
    }

    /// log2 of the inverse of the rate of the final codeword.
    pub fn final_log_inv_rate(&self) -> usize {
        self.log_final_inv_rate
    }

    /// Number of domain points queried in the final round.
    pub fn final_num_queries(&self) -> usize {
        self.final_num_queries
    }

    /// Number of proof-of-work bits for the final round.
    pub fn final_pow_bits(&self) -> usize {
        self.final_pow_bits
    }

    // ========= Getters for all fields of the internal StirParameters =========

    /// Security level of the protocol (including grinding).
    pub fn security_level(&self) -> usize {
        self.parameters.security_level
    }

    /// Code-distance assumption to configure STIR with. The more assumptions,
    /// the fewer queries and proof-of-work bits. Cf. [`SecurityAssumption`] for
    /// more details.
    pub fn security_assumption(&self) -> SecurityAssumption {
        self.parameters.security_assumption
    }

    /// log2 of the starting degree-plus-1 bound() of the initial polynomial.
    pub fn log_starting_degree(&self) -> usize {
        self.parameters.log_starting_degree
    }

    /// log2 of the folding factor `k_1` of the first round.
    pub fn log_starting_folding_factor(&self) -> usize {
        self.parameters.log_folding_factors[0]
    }

    /// log2 of the folding factors `k_1, ..., k_{M - 1}`.
    pub fn log_folding_factors(&self) -> &[usize] {
        &self.parameters.log_folding_factors
    }

    /// log2 of the folding factor `k_{M + 1}` used in the final round.
    pub fn log_last_folding_factor(&self) -> usize {
        *self.parameters.log_folding_factors.last().unwrap()
    }

    /// log2 of the inverse of the rate of the initial codeword.
    pub fn log_starting_inv_rate(&self) -> usize {
        self.parameters.log_starting_inv_rate
    }

    /// log2 of the inverses of the rates of the non-initial codewords.
    pub fn log_inv_rates(&self) -> &[usize] {
        &self.parameters.log_inv_rates
    }

    /// Number of proof-of-work bits used in the protocol.
    pub fn pow_bits(&self) -> usize {
        self.parameters.pow_bits
    }

    /// Number of proof-of-work bits used throughout all rounds.
    pub fn pow_bits_all_rounds(&self) -> Vec<usize> {
        let mut pow_bits = vec![self.starting_folding_pow_bits];
        pow_bits.extend(self.round_parameters.iter().map(|x| x.pow_bits));
        pow_bits.push(self.final_pow_bits);
        pow_bits
    }

    /// Configuration of the Mixed Matrix Commitment Scheme (hasher and
    /// compressor) used to commit to the initial polynomial `f_0` and
    /// full-round polynomials `g_1, ... g_M`.
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

impl<M: Clone> Display for StirConfig<M> {
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

// Have the challenger observe the public parameters at the start of the
// Fiat-Shamired interaction
pub(crate) fn observe_public_parameters<F, M>(
    parameters: &StirParameters<M>,
    challenger: &mut impl FieldChallenger<F>,
) where
    F: Field,
    M: Clone,
{
    observe_usize_slice(
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
    observe_usize_slice(challenger, &parameters.log_folding_factors, false);
    observe_usize_slice(challenger, &parameters.log_inv_rates, false);

    // We do not absorb the MMCS configuration, as it would require stringent
    // trait bounds
}
