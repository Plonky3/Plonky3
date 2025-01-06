use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::Field;
use p3_matrix::Matrix;

#[derive(Debug, Clone, Copy)]
pub enum SoundnessType {
    Provable,
    Conjecture,
}

#[derive(Debug)]
pub struct StirConfig<M> {
    // The targeted security level of the whole construction
    pub security_level: usize,

    // The targeted security level of the protocol
    pub protocol_security_level: usize,

    pub soundness_type: SoundnessType,

    pub proof_of_work_bits: Vec<usize>,

    pub mmcs_config: M,

    // Folding and code-related parameters
    // Note: All degrees refer to degree bounds + 1
    pub starting_degree: usize,
    pub degrees: Vec<usize>,

    pub stopping_degree: usize,
    pub folding_factor: usize,

    pub starting_log_inv_rate: usize,
    pub log_inv_rates: Vec<usize>,

    pub num_rounds: usize,
    pub repetitions: Vec<usize>,
    pub ood_samples: usize,
}

impl<M> StirConfig<M> {
    pub fn new(
        security_level: usize,
        protocol_security_level: usize,
        mmcs_config: M,
        soundness_type: SoundnessType,
        starting_degree: usize,
        stopping_degree: usize,
        folding_factor: usize,
        starting_log_inv_rate: usize,
    ) -> Self {
        assert!(folding_factor.is_power_of_two());
        assert!(starting_degree.is_power_of_two());
        assert!(stopping_degree.is_power_of_two());

        let mut d = starting_degree;

        let mut degrees = vec![d];
        println!("d: {}", d);
        let mut num_rounds = 0;

        while d > stopping_degree {
            assert!(d % folding_factor == 0);
            d /= folding_factor;
            degrees.push(d);
            num_rounds += 1;
        }

        num_rounds -= 1;
        degrees.pop();

        let mut log_inv_rates = vec![starting_log_inv_rate];
        let log_folding = folding_factor.ilog2() as usize;
        log_inv_rates
            .extend((1..num_rounds + 1).map(|i| starting_log_inv_rate + i * (log_folding - 1)));

        // Computing repetitions
        let constant = match soundness_type {
            SoundnessType::Provable => 2,
            SoundnessType::Conjecture => 1,
        };

        let proof_of_work_bits: Vec<usize> = log_inv_rates
            .iter()
            .map(|&log_inv_rate| {
                Self::proof_of_work_bits(
                    constant,
                    log_inv_rate,
                    security_level,
                    protocol_security_level,
                )
            })
            .collect();

        let mut repetitions: Vec<usize> = log_inv_rates
            .iter()
            .map(|&log_inv_rate| Self::repetitions(constant, log_inv_rate, protocol_security_level))
            .collect();

        for i in 0..num_rounds {
            repetitions[i] = repetitions[i].min(degrees[i] / folding_factor);
        }

        assert_eq!(num_rounds + 1, log_inv_rates.len());
        assert_eq!(num_rounds + 1, repetitions.len());

        Self {
            security_level,
            protocol_security_level,
            soundness_type,
            proof_of_work_bits,
            mmcs_config,
            starting_degree,
            degrees,
            stopping_degree,
            folding_factor,
            starting_log_inv_rate,
            log_inv_rates,
            num_rounds,
            repetitions,
            ood_samples: 2,
        }
    }

    fn proof_of_work_bits(
        constant: usize,
        log_inv_rate: usize,
        security_level: usize,
        protocol_security_level: usize,
    ) -> usize {
        let repetitions = Self::repetitions(constant, log_inv_rate, protocol_security_level);

        let achieved_security_bits = (log_inv_rate as f64 / constant as f64) * repetitions as f64;
        let remaining_security_bits = security_level as f64 - achieved_security_bits;

        if remaining_security_bits <= 0. {
            0
        } else {
            remaining_security_bits.ceil() as usize
        }
    }

    fn repetitions(constant: usize, log_inv_rate: usize, protocol_security_level: usize) -> usize {
        ((constant * protocol_security_level) as f64 / log_inv_rate as f64).ceil() as usize
    }
}

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
