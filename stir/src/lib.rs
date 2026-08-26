//! STIR: Reed-Solomon Proximity Testing with Fewer Queries.
//!
//! This crate implements the STIR polynomial commitment scheme, a univariate PCS
//! that achieves shorter proofs than FRI by replacing many direct proximity queries
//! with a small number of out-of-domain (OOD) samples combined with an answer
//! polynomial / shake polynomial argument.
//!
//! # Structure
//!
//! - [`config`]: Protocol parameters ([`StirParameters`], [`StirConfig`], [`StirRoundConfig`]).
//! - [`proof`]: Proof types ([`StirProof`], [`StirRoundProof`], [`StirQueryOpenings`]).
//! - [`utils`]: Polynomial arithmetic primitives (shake, ans, Horner eval, synthetic division).
//! - [`prover`]: The STIR prover ([`prover::prove_stir`]).
//! - [`verifier`]: The STIR verifier ([`verifier::verify_stir`], [`verifier::StirError`]).
//! - [`pcs`]: [`TwoAdicStirPcs`] implementing the [`p3_commit::Pcs`] trait.
//!
//! # Deviations from the STIR paper (eprint 2024/390)
//!
//! Several deliberate implementation choices differ from the construction stated in
//! the paper:
//!
//! - **Prover-assisted Ans/shake check.** The paper's verifier interpolates `Ans` itself.
//!   Here the prover sends `Ans` and a shake polynomial, and the verifier checks the
//!   identity at a transcript-derived random point. Its Schwartz–Zippel error is included
//!   explicitly in STIR's parameter validation.
//! - **Fixed `s` schedule.** OOD sample count is fixed per the paper's recommended schedule
//!   (`s = 1` for Johnson, `s = 2` for capacity); [`config::StirConfig::new`] does not search
//!   for the smallest valid `s`.
//! - **Proximity-gaps formulas from later work.** Johnson-bound proximity gaps use the
//!   tighter \[BCSS25\] bound rather than \[BCI+20\].
//! - **Round-0 joint queries-combination bound (CB only).**
//!   The capacity-bound initial eta extends the paper's prox-gap formula with a joint
//!   queries-combination × prox-gap term, using a closed-form upper bound on `t_0` in place
//!   of the missing `t_{-1}`.
//!   Without it, CB `queries_combination_error` at round 0 sags ~`log₂(t_0)` bits
//!   below `target_bits`. JB needs no such term — its list size stays small.
//! - **Union-bound buffer.** [`config::StirConfig::new`] adds an explicit
//!   `ceil(log2(6 · total_folds))` buffer to every per-round error term. Each round
//!   contributes up to six independent algebraic failure modes (query tier: query failure,
//!   OOD, random-combination,
//!   shake-check; folding tier: proximity-gaps, sumcheck), so the union bound over the full
//!   protocol is `≤ 6 · total_folds · 2^{-buffered_security_level}`. The paper's "+1 / +0"
//!   rule only delivers the claimed bits when `total_folds ≤ 2` and per-round terms are
//!   collapsed; the explicit log keeps deeper protocols tight across every term.

#![no_std]

extern crate alloc;

pub mod config;
pub mod pcs;
pub mod proof;
pub mod prover;
mod soundness;
pub mod utils;
pub mod verifier;

pub use config::{Stage, StirConfig, StirConfigError, StirParameters, StirRoundConfig};
pub use p3_security::whir::SecurityAssumption;
pub use pcs::TwoAdicStirPcs;
pub use proof::{StirProof, StirQueryOpenings, StirRoundProof};
pub use verifier::{StirError, StirVerifyOutputs};
