//! Protocol configuration for WHIR polynomial commitments.

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
    /// The logarithmic inverse rate for sampling.
    pub starting_log_inv_rate: usize,
    /// The value v such that that the size of the Reed Solomon domain on which
    /// our polynomial is evaluated gets divided by `2^v` at the first round.
    /// RS domain size at commitment = 2^(num_variables + starting_log_inv_rate)
    /// RS domain size after the first round = 2^(num_variables + starting_log_inv_rate - v)
    /// The default value is 1 (halving the domain size, which is the behavior of the consecutive rounds).
    pub rs_domain_initial_reduction_factor: usize,
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
            "Starting rate: 2^-{}, folding_factor: {:?}",
            self.starting_log_inv_rate, self.folding_factor,
        )
    }
}
