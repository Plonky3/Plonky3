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
//! 2. Each mask is encoded under a zero-knowledge code, MMCS-committed, and absorbed into the Fiat-Shamir transcript.
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
//! - Transcript schema and mask oracle handle.
//! - Prover-side sumcheck with mask sampling and round-polynomial assembly.
//! - Verifier-side replay with the dropped-coefficient reconstruction.
//! - Witness-free simulator used to prove honest-verifier zero-knowledge.
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
//! Both are checked at constructor entry.
//!
//! # References
//!
//! - eprint 2026/391, Section 6 (Construction 6.3, Lemma 6.4, Lemma 6.5).

pub mod data;
pub mod prover;
pub mod simulator;
pub mod verifier;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use data::{MaskOracle, ZkSumcheckData};
pub use prover::{ZkLayout, ZkPrefixProver, ZkProver, ZkSuffixProver};
pub use simulator::simulate_classic_unpacked;
pub use verifier::ZkVerifier;
