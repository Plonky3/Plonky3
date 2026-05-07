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
//!
//! # Deviations from the STIR paper (eprint 2024/390)
//!
//! Several deliberate implementation choices differ from the construction stated in
//! the paper:
//!
//! - **Subgroup-coordinate folding.** [`utils::fold_codeword`] uses subgroup x-coordinates
//!   `g^{j + l·new_height}` rather than the paper's coset coordinates `α·g^{j + l·new_height}`.
//!   Both prover and verifier follow the same convention end-to-end; soundness is preserved
//!   because the fold challenge is uniform.
//! - **Prover-assisted Ans/shake check.** The paper's verifier interpolates `Ans` itself.
//!   Here the prover sends `Ans` and a shake polynomial, and the verifier checks the
//!   identity at a transcript-derived random point. The Schwartz–Zippel error of this check
//!   is included in [`p3_commit::SecurityAssumption::stir_query_algebraic_bits`] via
//!   [`p3_commit::SecurityAssumption::shake_check_error`].
//! - **Fixed `s` schedule.** OOD sample count is fixed per the paper's recommended schedule
//!   (`s = 1` for Johnson, `s = 2` for capacity); [`config::StirConfig::new`] does not search
//!   for the smallest valid `s`.
//! - **Proximity-gaps formulas from later work.** Johnson-bound proximity gaps use the
//!   tighter \[BCSS25\] bound rather than \[BCI+20\]; see
//!   [`p3_commit::SecurityAssumption::prox_gaps_error`].
//! - **Union-bound buffer.** [`config::StirConfig::new`] adds an explicit
//!   `ceil(log2(total_folds))` buffer to every per-round error term (query failure plus
//!   the auxiliary terms bridged by PoW). The paper's "+1 / +0" rule only delivers the
//!   claimed bits when `total_folds ≤ 2`; the explicit log makes deeper protocols tight.
//! - **PoW placement.** Each round's query/OOD PoW grind is placed BEFORE OOD sampling, so
//!   it gates re-rolls of the OOD set via re-commitment (the paper's high-level protocol
//!   does not specify PoW). The folding PoW gates the fold challenge as in the paper.

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
