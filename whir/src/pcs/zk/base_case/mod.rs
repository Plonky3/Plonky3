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
//! - Each fixed group receives independent uniform draws. Different groups
//!   are correlated, but that does not worsen the query-round maximum bound.
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
//! - after the pre-query transcript fixes a bad group, accepting every group
//!   implies missing that selected group's bad set. Thus the query-round error
//!   is the maximum group miss probability, not a union bound.
//!
//! Groups with the same code, radius, domain, query count, and exact projected
//! query vector may additionally be analyzed as one wider interleaved
//! component. Its disagreement set is the union of the constituent row
//! disagreement sets; no alignment assumption is made. Without exact row
//! synchronization, their candidate-list factors must remain separate.
//!
//! # Heterogeneous Construction 7.2 contract
//!
//! At the ideal-oracle layer, let `M_0` be the source-code MCA error and `M_i`
//! the MCA error of mask group `i`. Let `L_0` and `L_i` bound the corresponding
//! old/fresh interleaved lists. The inherited two-round bounds have the shape
//!
//! ```text
//! epsilon_gamma <= M_0 + sum_i M_i + L_0 * product_i L_i / |EF|
//! epsilon_query <= max(source_miss, max_i group_miss_i)
//! ```
//!
//! MCA errors add because every group uses the same `gamma`; candidate lists
//! multiply because complete candidates form a Cartesian product. The query
//! maximum follows because a bad component can be selected from the fixed
//! pre-query transcript. This contract requires linear injective randomized
//! encodings, fresh independent encoding randomness, and radii strictly below
//! the corresponding code distances.
//!
//! The concrete Merkle and Fiat-Shamir compilation additionally relies on the
//! mixed MMCS binding the verifier-shaped, height-stratified row streams. The
//! configuration retains its conservative query surcharge until the complete
//! MCA/list/field accounting is reviewed.
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
