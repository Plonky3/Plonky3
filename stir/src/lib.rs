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
//! - [`proof`]: Proof types ([`StirProof`], [`StirRoundProof`], [`StirQueryProof`]).
//! - [`utils`]: Polynomial arithmetic primitives (shake, ans, Horner eval, synthetic division).
//! - [`prover`]: The STIR prover ([`prover::prove_stir`]).
//! - [`verifier`]: The STIR verifier ([`verifier::verify_stir`], [`verifier::StirError`]).
//! - [`pcs`]: [`TwoAdicStirPcs`] implementing the [`p3_commit::Pcs`] trait.

#![no_std]

extern crate alloc;

pub mod config;
pub mod pcs;
pub mod proof;
pub mod prover;
pub mod utils;
pub mod verifier;

pub use config::{StirConfig, StirParameters, StirRoundConfig};
pub use pcs::TwoAdicStirPcs;
pub use proof::{StirFinalQueryProof, StirProof, StirQueryProof, StirRoundProof};
pub use verifier::{StirError, StirVerifyOutputs};
