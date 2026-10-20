//! Parameters prover and verifier share, and the sponge every phase draws from.

use p3_binary_field::BinaryChallenger;
use p3_challenger::HashChallenger;
use p3_keccak::Keccak256Hash;

use crate::Val;

/// The Fiat-Shamir sponge every Flock phase shares.
pub type FlockChallenger = BinaryChallenger<Val, HashChallenger<u8, Keccak256Hash, 32>>;

/// Proof parameters shared by prover and verifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlockConfig {
    /// Bit commitment parameters.
    pub pcs: PcsParams,
    /// Grinding bits before every zerocheck challenge.
    pub pow_bits: usize,
}

/// WHIR parameters of the bit commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcsParams {
    /// Target in bits for every error term of the opening.
    pub security_bits: usize,
    /// Log inverse rate of the first codeword.
    pub log_inv_rate: usize,
    /// Variables each WHIR round folds.
    pub folding: usize,
}

impl Default for PcsParams {
    /// Each term at 128 bits, rate 1/2, list decoding up to the Johnson bound.
    ///
    /// The terms add up: the opening reaches about 126 bits, and Keccak-256 caps the hashes at 128.
    fn default() -> Self {
        Self {
            security_bits: 128,
            log_inv_rate: 1,
            folding: 4,
        }
    }
}
