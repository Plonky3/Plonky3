//! Parameters prover and verifier share, and the sponge every phase draws from.

use p3_binary_pcs::whir::CommitmentParams;
use p3_binary_pcs::whir::preset::hash64;
use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;

/// A 256-bit hash a proof's transcript and Merkle tree both use.
pub trait FlockHash: CryptographicHasher<u8, [u8; 32]> + Default + Clone + Send + Sync {}

impl<H: CryptographicHasher<u8, [u8; 32]> + Default + Clone + Send + Sync> FlockHash for H {}

/// The Fiat-Shamir sponge every Flock phase shares, Keccak-256 by default.
pub type FlockChallenger<H = Keccak256Hash> = hash64::Challenger<H>;

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
