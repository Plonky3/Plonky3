//! Protocol configuration for WHIR polynomial commitments.

use alloc::vec::Vec;
use core::fmt::Display;

mod folding;
pub mod soundness;
pub mod whir;

pub use folding::{FoldingFactor, FoldingFactorError};
pub use soundness::SecurityAssumption;
pub use whir::{RoundConfig, WhirConfig};

/// Fallback proof-of-work difficulty when the user does not specify one.
///
/// 16 bits strikes a balance between prover cost and verifier DoS resistance.
/// - Higher values slow down the prover;
/// - Lower values weaken the PoW contribution to soundness.
pub const DEFAULT_MAX_POW: usize = 16;

/// Configuration parameters for WHIR proofs.
#[derive(Clone, Debug)]
pub struct ProtocolParameters {
    /// Initial logarithmic inverse rate for the first committed codeword.
    pub starting_log_inv_rate: usize,
    /// Log-inverse rates for the codewords committed after each intermediate
    /// WHIR round.
    pub round_log_inv_rates: Vec<usize>,
    /// The folding factor strategy.
    pub folding_factor: FoldingFactor,
    /// The type of soundness guarantee.
    pub soundness_type: SecurityAssumption,
    /// The security level in bits.
    pub security_level: usize,
    /// The number of bits required for proof-of-work (PoW).
    pub pow_bits: usize,
    /// Enable HVZK code-switching (Construction 9.7, eprint 2026/391 §9.4).
    ///
    /// When `true`, each non-final intermediate round:
    /// - Commits the target polynomial with ZK randomness (`Enc(f ∥ r')`).
    /// - Commits a mask oracle hiding the previous round's encoding randomness.
    /// - Uses a private zero-evader for OOD answers (Lemma 9.3).
    /// - Stores per-query STIR and OOD corrections in the proof so the
    ///   verifier can recover plain evaluations for the sumcheck constraint.
    ///
    /// The composed protocol is HVZK with error `ζ_{C'} + ζ_ze + ζ_{C_zk}`
    /// per round (Lemma 9.8), composing via Theorem 4.5.
    pub zk: bool,
}

impl ProtocolParameters {
    /// Testing parameters with ZK enabled.
    ///
    /// Security level 32 keeps tests fast. Suffix variable ordering required.
    pub fn new_testing_zk(num_variables: usize) -> Self {
        let folding_factor = FoldingFactor::Constant(4);
        Self {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: Self::default_round_log_inv_rates(
                num_variables,
                &folding_factor,
                1,
            ),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: true,
        }
    }

    /// Benchmark parameters with ZK enabled.
    ///
    /// Security level 100 with PoW for realistic overhead measurement.
    pub fn new_benchmark_zk(num_variables: usize) -> Self {
        let folding_factor = FoldingFactor::ConstantFromSecondRound(4, 4);
        Self {
            security_level: 100,
            pow_bits: 20,
            round_log_inv_rates: Self::default_round_log_inv_rates(
                num_variables,
                &folding_factor,
                1,
            ),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: true,
        }
    }

    fn default_round_log_inv_rates(
        num_variables: usize,
        folding_factor: &FoldingFactor,
        starting_log_inv_rate: usize,
    ) -> alloc::vec::Vec<usize> {
        let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_variables);
        let mut rates = alloc::vec::Vec::with_capacity(num_rounds);
        let mut rate = starting_log_inv_rate;
        for round in 0..num_rounds {
            rate += folding_factor.at_round(round) - 1;
            rates.push(rate);
        }
        rates
    }
}

impl Display for ProtocolParameters {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "Targeting {}-bits of security with {}-bits of PoW - soundness: {:?}",
            self.security_level, self.pow_bits, self.soundness_type
        )?;
        writeln!(
            f,
            "Starting rate: 2^-{}, round_log_inv_rates: {:?}, folding_factor: {:?}",
            self.starting_log_inv_rate, self.round_log_inv_rates, self.folding_factor,
        )
    }
}
