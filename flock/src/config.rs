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
    /// Bits the composed opening error must reach.
    pub security_bits: usize,
    /// Log inverse rate of the first codeword.
    pub log_inv_rate: usize,
    /// Variables the first WHIR round folds, one Merkle leaf holding `2^first_folding` elements.
    pub first_folding: usize,
    /// Bits the codeword domain loses after the first round.
    pub first_reduction: usize,
    /// Variables each later WHIR round folds.
    pub folding: usize,
    /// Grinding before every query phase.
    pub query_grinding: usize,
    /// Whether the initial commitment takes its own out-of-domain samples.
    ///
    /// Without them the outer protocol's soundness error grows by the list size (ePrint 2026/1329, Appendix C, Remark 11).
    pub commitment_ood: bool,
}

impl Default for PcsParams {
    /// 128 composed bits at rate 1/2, list decoding up to the Johnson bound.
    fn default() -> Self {
        Self {
            security_bits: 128,
            log_inv_rate: 1,
            first_folding: 6,
            first_reduction: 3,
            folding: 4,
            query_grinding: 16,
            commitment_ood: false,
        }
    }
}
