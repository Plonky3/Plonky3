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
//! - A group shares its evaluation domain, spot-check positions, and Merkle
//!   paths.
//! - The fresh blinds mirror that grouping.
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

pub use config::{BaseCaseZkConfig, MaskGroupWitness, MaskProverData};
pub use error::BaseCaseZkError;
pub use prover::BaseCaseZkProver;
pub use verifier::BaseCaseZkVerifier;

#[cfg(test)]
mod tests;
