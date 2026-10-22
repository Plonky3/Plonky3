//! Parameters prover and verifier share, and the sponge every phase draws from.

use p3_binary_pcs::whir::CommitmentParams;
use p3_binary_pcs::whir::preset::keccak64;

/// The Fiat-Shamir sponge every Flock phase shares.
pub type FlockChallenger = keccak64::Challenger;

/// Proof parameters shared by prover and verifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlockConfig {
    /// Bit commitment parameters.
    pub pcs: PcsParams,
    /// Grinding bits before every zerocheck challenge.
    pub pow_bits: usize,
}

/// WHIR parameters of the bit commitment.
pub type PcsParams = CommitmentParams;
