//! HVZK base case (Construction 7.2 of eprint 2026/391).
//!
//! Last step of the hiding pipeline.
//! All folding is done, so the secrets are now small vectors.
//! One linear claim about them remains to prove, without revealing them.
//!
//! # The idea, with one secret
//!
//! The prover committed to `f` and claims `<f, W> = target` for a public `W`.
//!
//! ```text
//!     prover  : commits a fresh uniform vector g, sends mu = <g, W>
//!     verifier: sends a random challenge gamma
//!     prover  : reveals  f* = g + gamma * f
//!     verifier: checks   <f*, W> = mu + gamma * target
//! ```
//!
//! - The check holds by linearity exactly when both claims do.
//! - A cheating prover is stuck: `mu` is fixed before `gamma` is known.
//! - `f*` is a one-time-pad reveal: uniform, it leaks nothing about `f`
//!   (honest-verifier zero knowledge, Lemma 7.3).
//!
//! A reveal alone could be fabricated.
//! Spot checks tie it back to the commitments:
//!
//! ```text
//!     Enc(f*)(z) = Enc(g)(z) + gamma * Enc(f)(z)     at t random positions z
//! ```
//!
//! The right side opens the two committed codewords at `z`.
//! The left side re-encodes the reveal; encoding is linear.
//!
//! # The real protocol
//!
//! The pipeline arrives here with one source secret `f` and many mask
//! secrets `xi_i`, tied into one joint claim:
//!
//! ```text
//!     <f, W> + sum_i <xi_i, u_i> = target
//! ```
//!
//! The same three moves run once for all of them:
//!
//! - one fresh mask per secret, one shared `gamma`, one reveal per secret
//!   (the vector and its encoding randomness alike),
//! - one joint target check over all the reveals,
//! - spot checks per oracle: `t` source positions, `t_zk` per mask group.
//!
//! # Cost
//!
//! - One fresh mask per secret: a 2x blow-up.
//! - It touches only these terminal vectors.
//!   That is sublinear in the witness.
//!
//! # Mask groups
//!
//! - Masks committed together form one interleaved oracle.
//! - The carried commitments retain their original group-specific domains.
//! - Every fresh blind group is committed in one mixed-dimension MMCS tree.
//! - One global query vector is projected onto each shorter mask domain.
//! - Each projection remains independent and uniform for its group. Group-to-
//!   group correlation does not affect the soundness union bound.
//!
//! # Shared-query security argument
//!
//! Let `M` be the largest mask domain and `m_i` one group's domain. Both are
//! powers of two. MMCS maps a global query `q` to that group by
//! `q >> log2(M / m_i)`. Every group index has exactly `M / m_i` preimages,
//! so an independent uniform `q` projects to an independent uniform group
//! query. Consequently:
//!
//! - every mask exposes at most `t_zk` distinct positions, preserving its
//!   Reed-Solomon ZK query budget;
//! - a bad set in any group lifts to a global bad set of the same density, so
//!   its miss probability remains `(1 - delta_i)^t_zk`;
//! - correlations between different groups do not invalidate the union bound.
//!
//! This is a concrete extension of Construction 7.2 from one common `C_zk`
//! to heterogeneous power-of-two mask domains and requires protocol-level
//! review before the optimization is treated as production-ready.
//!
//! # Source oracle abstraction
//!
//! - The source `f` is usually virtual: the fold of the last committed
//!   oracle at the final sumcheck randomness.
//! - Opening and verifying source positions is delegated to caller closures.
//! - A directly-committed codeword is the trivial fold.

mod config;
mod error;
mod prover;
mod verifier;

use alloc::vec::Vec;

use crate::pcs::zk::mask::MaskGroupShape;

/// Largest mask-code domain represented by a mixed fresh-mask commitment.
fn max_mask_domain_size(groups: &[MaskGroupShape]) -> Option<usize> {
    groups.iter().map(|group| group.shape.domain_size).max()
}

/// Project global mixed-MMCS indices onto one mask group's domain.
///
/// All mask domains are powers of two. This is the same high-bit projection
/// specified by [`p3_commit::Mmcs`] for matrices shorter than the tallest one.
fn project_mask_positions(
    global_positions: &[usize],
    max_domain_size: usize,
    group_domain_size: usize,
) -> Vec<usize> {
    assert!(max_domain_size.is_power_of_two());
    assert!(group_domain_size.is_power_of_two());
    assert!(group_domain_size <= max_domain_size);
    let shift = max_domain_size.ilog2() - group_domain_size.ilog2();
    global_positions
        .iter()
        .map(|&position| position >> shift)
        .collect()
}

pub use config::{BaseCaseZkConfig, MaskGroupWitness, MaskProverData};
pub use error::BaseCaseZkError;
pub use prover::BaseCaseZkProver;
pub use verifier::BaseCaseZkVerifier;

#[cfg(test)]
mod tests;
