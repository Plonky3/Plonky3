//! WHIR: Reed-Solomon proximity testing with super-fast verification.
//!
//! An IOP of proximity for constrained Reed-Solomon codes that serves as
//! a multilinear polynomial commitment scheme.
//!
//! A hiding variant lives in the zero-knowledge PCS module.
//! Masked sumcheck batches, HVZK code-switching rounds, and a masked base
//! case compose into a commitment that reveals only the requested
//! evaluations.
//!
//! References:
//! - <https://eprint.iacr.org/2024/1586> (WHIR),
//! - <https://eprint.iacr.org/2026/391> (HVZK-WHIR).

#![no_std]

extern crate alloc;

pub mod fiat_shamir;
pub mod parameters;
pub mod pcs;
pub(crate) mod utils;

pub use p3_sumcheck::{self as sumcheck, constraints};
