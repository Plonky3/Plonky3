//! Protocol configuration for WHIR polynomial commitments.

use alloc::vec::Vec;
use core::fmt::Display;

mod folding;
pub mod soundness;
pub mod whir;

pub use folding::{FoldingFactor, FoldingFactorError};
pub use soundness::SecurityAssumption;
pub use whir::{RoundConfig, WhirConfig, WhirConfigError};

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
