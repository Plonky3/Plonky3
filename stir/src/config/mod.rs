use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::Field;
use p3_matrix::Matrix;

use crate::utils::compute_pow;
use crate::SecurityAssumption;

#[cfg(test)]
mod tests;

/// STIR-related parameters defined by the user. These get expanded into a full `StirConfig`.
#[derive(Debug, Clone)]
pub struct StirParameters<M: Clone> {
    /// Security level desired in bits.
    pub(crate) security_level: usize,

    /// Security assumption under which to configure FRI.
    pub(crate) security_assumption: SecurityAssumption,

    /// log of the purported (degree + 1) of the initial polynomial.
    // Note: In variable or field names, "degree" refers to "degree bound (<=) +
    // 1". Comments are more accurate: "degree" means "degree bound (<=)"
    pub(crate) log_starting_degree: usize,

    /// log of the folding factor in the first round.
    pub(crate) log_starting_folding_factor: usize,

    /// log of the folding factors in non-first rounds.
    pub(crate) log_folding_factors: Vec<usize>,

    /// log of the inverse of the starting rate used in the protocol.
    pub(crate) log_starting_inv_rate: usize,

    /// log of the inverses of the rates in non-first protocol rounds.
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
        log_starting_folding_factor: usize,
        security_assumption: SecurityAssumption,
        security_level: usize,
        pow_bits: usize,
        mmcs_config: M,
    ) -> Self {
        StirParameters {
            log_starting_degree,
            log_starting_inv_rate,
            log_starting_folding_factor,
            log_folding_factors: vec![log_starting_folding_factor; log_inv_rates.len()],
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
            log_starting_inv_rate,
            log_starting_folding_factor: log_folding_factor,
            log_folding_factors: vec![log_folding_factor; num_rounds],
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
pub(crate) struct RoundConfig {
    /// log of the folding factor for this round.
    pub(crate) log_folding_factor: usize,

    /// log of the size of the evaluation domain of the oracle *sent this round*
    pub(crate) log_evaluation_domain_size: usize,

    /// Number of PoW bits used to reduce query error.
    pub(crate) pow_bits: f64,

    /// Number of queries in this round
    pub(crate) num_queries: usize,

    /// Number of out of domain samples in this round
    pub(crate) ood_samples: usize,

    /// log of the inverse of the rate of the current RS codeword
    pub(crate) log_inv_rate: usize,
}

#[derive(Debug, Clone)]
pub struct StirConfig<M: Clone> {
    /// Input parameters for the STIR protocol.
    parameters: StirParameters<M>,

    /// log of the size of the initial domain.
    starting_domain_log_size: usize,

    /// Initial pow bits used in the first fold.
    // NP TODO can this be a usize?
    starting_folding_pow_bits: f64,

    /// Round-specific parameters.
    round_parameters: Vec<RoundConfig>,

    /// log of the (degree + 1) of the final polynomial sent in plain.
    log_stopping_degree: usize,

    /// log of the inverse of therate of the final RS codeword.
    log_final_inv_rate: usize,

    /// Number of queries in the last round.
    final_num_queries: usize,

    /// Final of PoW bits (for the queries).
    final_pow_bits: f64,
}

impl<M: Clone> StirConfig<M> {
    /// Expand STIR parameters into a full STIR configuration.
    pub fn new<F: Field>(parameters: StirParameters<M>) -> Self {
        let StirParameters {
            security_level,
            security_assumption,
            log_starting_degree,
            log_starting_folding_factor,
            log_folding_factors,
            log_starting_inv_rate,
            log_inv_rates,
            pow_bits,
            ..
        } = parameters.clone();

        assert!(
            log_starting_folding_factor > 0 && log_folding_factors.iter().all(|&x| x != 0),
            "Folding factors should be non zero"
        );
        assert_eq!(log_folding_factors.len(), log_inv_rates.len());

        // log(degree + 1) can not be reduced past 0
        let total_reduction =
            log_starting_folding_factor + log_folding_factors.iter().sum::<usize>();
        assert!(total_reduction <= log_starting_degree);

        // If the first round wants to reduce the degreemore than possible, one
        // should send the polynomial directly instead
        assert!(log_starting_degree >= log_folding_factors[0]);

        // Compute the log of (final degree + 1) as well as the number of (non-final) rounds
        let log_stopping_degree = log_starting_degree - total_reduction;
        let num_rounds = log_folding_factors.len();

        // Compute the security level
        let protocol_security_level = 0.max(security_level - pow_bits);

        // Initial domain size
        let starting_domain_log_size =
            parameters.log_starting_degree + parameters.log_starting_inv_rate;

        // NP TODO batching
        /*
        // PoW bits for the batching steps
        let mut batching_pow_bits = 0.;

        if ldt_parameters.batch_size > 1 {
            let prox_gaps_error_batching = parameters.security_assumption.prox_gaps_error(
                ldt_parameters.log_degree,
                parameters.starting_log_inv_rate,
                ldt_parameters.field.extension_bit_size(),
                ldt_parameters.batch_size,
            ); // We now start, the initial folding pow bits
            batching_pow_bits = pow_util(security_level, prox_gaps_error_batching);

            // Add the round for the batching
            protocol_builder = protocol_builder
                .start_round("batching_round")
                .verifier_message(VerifierMessage::new(
                    vec![RbRError::new("batching_error", prox_gaps_error_batching)],
                    batching_pow_bits,
                ))
                .end_round();
        } */

        // NP TODO prepare Merkle tree / MMCS with domain separation, if possible and necessary
        /* let mut current_merkle_tree = MerkleTree::new(
            starting_domain_log_size - log_starting_folding_factor,
            ldt_parameters.field,
            (1 << log_starting_folding_factor) * ldt_parameters.batch_size,
            false, // First tree is over the base
        ); */

        // Degree of next polynomial to send
        let mut current_log_degree = log_starting_degree - log_starting_folding_factor;
        let mut log_inv_rate = log_starting_inv_rate;

        // We now start, the initial folding pow bits
        let field_bits = F::bits();

        let starting_folding_prox_gaps_error = security_assumption.prox_gaps_error(
            current_log_degree,
            log_inv_rate,
            field_bits,
            1 << parameters.log_starting_folding_factor,
        );

        let starting_folding_pow_bits =
            compute_pow(security_level, starting_folding_prox_gaps_error);

        let mut round_parameters = Vec::with_capacity(num_rounds);

        for (next_folding_factor, next_rate) in log_folding_factors.into_iter().zip(log_inv_rates) {
            // This is the size of the new evaluation domain
            let new_evaluation_domain_size = current_log_degree + next_rate;

            // Send the new oracle
            // NP TODO Merkle tree / MMCS
            /* let next_merkle_tree = MerkleTree::new(
                new_evaluation_domain_size - folding_factor,
                ldt_parameters.field,
                1 << folding_factor,
                true,
            ); */

            // Compute the ood samples required
            let ood_samples = security_assumption.determine_ood_samples(
                security_level,
                current_log_degree,
                next_rate,
                field_bits,
            );

            // Add OOD rounds to protocol
            // NP re-introduce depending on FS/ProtocolBuilder/absorption of public parameters
            /* if ood_samples > 0 {
                let ood_error = parameters.security_assumption.ood_error(
                    current_log_degree,
                    next_rate,
                    field_bits,
                    ood_samples,
                );
            } */

            // Compute the number of queries required
            let num_queries = security_assumption.queries(protocol_security_level, log_inv_rate);

            // We need to compute the errors, to compute the according PoW
            let query_error = security_assumption.queries_error(log_inv_rate, num_queries);

            let num_terms = num_queries + ood_samples;
            let prox_gaps_error_1 = parameters.security_assumption.prox_gaps_error(
                current_log_degree,
                next_rate,
                field_bits,
                num_terms,
            );

            let prox_gaps_error_2 = security_assumption.prox_gaps_error(
                current_log_degree - next_folding_factor,
                next_rate,
                field_bits,
                1 << next_folding_factor,
            );

            // Now compute the PoW
            let pow_bits = compute_pow(
                security_level,
                query_error.min(prox_gaps_error_1).min(prox_gaps_error_2),
            );

            let round_config = RoundConfig {
                log_evaluation_domain_size: new_evaluation_domain_size,
                log_folding_factor: next_folding_factor,
                num_queries,
                pow_bits,
                ood_samples,
                log_inv_rate,
            };
            round_parameters.push(round_config);
            log_inv_rate = next_rate;
            current_log_degree -= next_folding_factor;
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
        let final_pow_bits = compute_pow(security_level, query_error);

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

    // Getters for all internal fields
    pub fn starting_domain_log_size(&self) -> usize {
        self.starting_domain_log_size
    }

    pub fn starting_folding_pow_bits(&self) -> f64 {
        self.starting_folding_pow_bits
    }

    pub(crate) fn round_configs(&self) -> &[RoundConfig] {
        &self.round_parameters
    }

    pub(crate) fn round_config(&self, i: usize) -> &RoundConfig {
        &self.round_parameters[i]
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

    pub fn final_pow_bits(&self) -> f64 {
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
        self.parameters.log_starting_folding_factor
    }

    pub fn log_folding_factors(&self) -> &[usize] {
        &self.parameters.log_folding_factors
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

    pub fn mmcs_config(&self) -> &M {
        &self.parameters.mmcs_config
    }
}
// NO TODO why is this here/necessary?/rename
/// Whereas `FriConfig` encompasses parameters the end user can set, `FriGenericConfig` is
/// set by the PCS calling FRI, and abstracts over implementation details of the PCS.
pub trait FriGenericConfig<F: Field> {
    type InputProof;
    type InputError: Debug;

    /// We can ask FRI to sample extra query bits (LSB) for our own purposes.
    /// They will be passed to our callbacks, but ignored (shifted off) by FRI.
    fn extra_query_index_bits(&self) -> usize;

    /// Fold a row, returning a single column.
    /// Right now the input row will always be 2 columns wide,
    /// but we may support higher folding arity in the future.
    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        beta: F,
        evals: impl Iterator<Item = F>,
    ) -> F;

    /// Same as applying fold_row to every row, possibly faster.
    fn fold_matrix<M: Matrix<F>>(&self, beta: F, m: M) -> Vec<F>;
}
