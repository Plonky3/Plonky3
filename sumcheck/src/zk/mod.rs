//! Honest-verifier zero-knowledge sumcheck for WHIR.
//!
//! Implements Construction 6.3 of eprint 2026/391 on top of both
//! stacked-binding modes — prefix and suffix.
//!
//! # Overview
//!
//! The plain WHIR sumcheck leaks the witness:
//! Each round polynomial coefficient is an affine function of the secret message.
//!
//! The HVZK variant blinds those coefficients with `k` small mask polynomials, one per sumcheck round.
//! Overhead: only `O(k * ell_zk)` extra field elements.
//!
//! # Protocol shape
//!
//! Let `k` be the folding factor and `ell_zk` the mask code message length.
//!
//! 1. Prover samples `k` univariate masks of degree `ell_zk - 1` over the extension field.
//! 2. The masks are batch encoded under a zero-knowledge code, MMCS-committed
//!    as one matrix, and absorbed into the Fiat-Shamir transcript.
//! 3. Prover sends `mu_tilde`, the sum of all mask evaluations over `{0,1}^k`.
//! 4. Verifier samples a combining challenge `eps` in the extension field.
//! 5. Per-round wire polynomial mixes:
//!    - the live mask,
//!    - past mask evaluations at sampled challenges,
//!    - future-mask endpoints,
//!    - the plain sumcheck contribution scaled by `eps`.
//!
//! # Round polynomial at a glance
//!
//! For round `j` with past challenges `gamma_1, ..., gamma_{j - 1}`:
//!
//! ```text
//!     h_j(X) = 2^{k - j}     * s_j(X)
//!            + 2^{k - j}     * sum_{l <  j} s_l(gamma_l)
//!            + 2^{k - j - 1} * sum_{l >  j} ( s_l(0) + s_l(1) )
//!            + eps           * plain_piece_j(X)
//! ```
//!
//! Verifier checks:
//!
//! ```text
//!     round 1  : h_1(0) + h_1(1) = eps * mu + mu_tilde
//!     round j>1: h_j(0) + h_j(1) = h_{j - 1}(gamma_{j - 1})
//! ```
//!
//! # Module layout
//!
//! - Proof record and mask oracle handle.
//! - Fiat-Shamir description of one masked batch, driven by every party.
//! - Prover-side sumcheck with mask sampling and round-polynomial assembly.
//! - Verifier-side replay with the dropped-coefficient reconstruction.
//! - Witness-free simulators, one per prelude, used to prove honest-verifier zero-knowledge.
//!
//! # Two preludes, and the scope of each
//!
//! A masked batch reaches its target one of two ways, and the description names which:
//!
//! ```text
//!     recorded claims  ->  the target is derived from the claims a verifier recorded
//!     inherited claim  ->  the target arrives as public input and is bound first
//! ```
//!
//! The composed WHIR pipeline plays the inherited one, at every batch, on both sides.
//!
//! The recorded-claims prelude is this crate's own standalone surface instead.
//! No in-tree pipeline plays it: its entry points serve callers who drive a masked batch directly, and it stays supported on that basis.
//!
//! Both preludes carry a witness-free simulator, so neither ships without its zero-knowledge argument.
//!
//! Scope, so a future reader can act on it: a third prelude owes a third simulator.
//! Retiring the recorded-claims prelude means retiring the public entry points that expose it, which is an API decision rather than a change inside this module.
//!
//! # Layout coverage
//!
//! Both stacked-binding modes are supported.
//! The masking layer is binding-agnostic; only the residual handoff differs between modes.
//! Resolves [Plonky3#1649](https://github.com/Plonky3/Plonky3/issues/1649).
//!
//! # Field constraints (Lemma 6.4)
//!
//! - Base field characteristic must not be `2`.
//! - Mask message length `ell_zk` must be at least `3`, so the mask (degree `ell_zk - 1`) covers the degree-2 plain round polynomial.
//!
//! Both are checked where the transcript description is built.
//!
//! A prover treats a violation as its own configuration bug.
//!
//! A verifier reports it.
//!
//! # Divergences from the paper
//!
//! ## `ell_zk >= 3`, against the paper's `ell_zk >= 2`
//!
//! The round polynomial here is a product of two multilinears, so it is quadratic in `X`.
//!
//! ```text
//!     h_size = max(ell_zk, 3)
//!
//!     ell_zk = 2  ->  mask covers slots 0, 1 only, and slot 2 carries eps * c_inf alone
//!     ell_zk = 3  ->  mask covers slot 2 as well
//! ```
//!
//! At `ell_zk = 2` the quadratic coefficient would ship unmasked, so the bound is deliberately one higher.
//!
//! ## The `aux * 2^{-j}` carry
//!
//! An inherited batch adds an auxiliary constant to its target, matching Definition 5.8.
//!
//! The carry is the normalisation that makes it behave like a constant function on the cube.
//!
//! ```text
//!     AUX * 2^{-k} at every point of {0,1}^k
//!
//!     round-j partial sum  ->  2^{k-j} * AUX * 2^{-k}  =  AUX * 2^{-j}
//!     cube total           ->  2^{k}   * AUX * 2^{-k}  =  AUX
//! ```
//!
//! Round `j` therefore adds `eps * AUX * 2^{-j}` to its constant slot, and the residual carries `eps * AUX * 2^{-k}`.
//!
//! ## Compiled hiding is computational, not perfect
//!
//! Lemma 6.4 is proved for an IOR whose masks are oracles, with a simulator answering queried positions.
//!
//! ```text
//!     paper     ->  mask oracle, only the queried positions are ever seen
//!     this code ->  the whole mask codeword committed under one MMCS root
//! ```
//!
//! A root determines the masks information-theoretically, so indistinguishability of the commits rests on the commitment hiding a high-entropy leaf set.
//!
//! Perfect indistinguishability still holds for everything the simulator itself emits.
//!
//! This is the standard IOR-to-argument compilation step, not a defect of the construction.
//!
//! # References
//!
//! - eprint 2026/391, Section 6 (Construction 6.3, Lemma 6.4, Lemma 6.5).

pub mod data;
pub mod prover;
pub mod simulator;
pub mod transcript;
pub mod verifier;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use data::{
    MaskOracle, ZkSumcheckData, ZkSumcheckHandoff, ZkVerifierHandoff, mask_residual,
    mask_residual_covectors, mask_residual_covectors_from_shape,
};
pub use prover::{ZkLayout, ZkPrefixProver, ZkProver, ZkSuffixProver, stack_codewords};
pub use simulator::{simulate_classic_unpacked, simulate_classic_unpacked_claim};
pub use transcript::{ZkPrelude, ZkProverTranscript, ZkSumcheckShape, ZkVerifierTranscript};
pub use verifier::ZkVerifier;
