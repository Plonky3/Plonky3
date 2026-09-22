//! Soundness analysis for STARK protocols.
//!
//! Decomposition is by error source, not by commitment scheme.
//!
//! A protocol's parameter type lives in its own crate.
//!
//! This crate supplies the error-bit math and the regime mirrors.
//!
//! The protocol assembles the regime and picks which error terms apply.
//!
//! It then composes them by union bound, or by minimum, or through the composite.
//!
//! # Layering
//!
//! ```text
//! protocol crate (p3-fri, p3-whir, …)
//!         │
//!         ▼
//!     p3-security ── p3-air, p3-field, p3-util, libm
//! ```
//!
//! Each regime mirrors the security-relevant subset of a runtime configuration.
//!
//! It also carries instance-shape metadata the runtime configuration does not hold.
//!
//! Trace length and opening counts are the usual examples.
//!
//! No conversion from the runtime configuration exists, and that is deliberate.
//!
//! Only the protocol crate can see both sides, so it assembles the regime explicitly.
//!
//! # References
//! - ethSTARK ([2021/582](https://eprint.iacr.org/2021/582))
//! - Proximity Gaps for Reed–Solomon Codes ([2020/654](https://eprint.iacr.org/2020/654))
//! - On the Security of STARKs with FRI ([2024/1553](https://eprint.iacr.org/2024/1553))
//! - On the Distribution of the Distances of Random Words ([2025/2010](https://eprint.iacr.org/2025/2010))
//! - BCHKS25 — Improved LDR proximity gaps ([2025/2055](https://eprint.iacr.org/2025/2055))
//! - DKT26 — Johnson-regime line MCA (Theorem 5.12, Appendix B.1–B.2)
//!   ([2026/2056](https://eprint.iacr.org/2026/2056))
//! - HVZK-WHIR ([2026/391](https://eprint.iacr.org/2026/391))

#![no_std]
#![allow(clippy::too_many_arguments)]

extern crate alloc;

pub mod assumption;
pub mod budget;
pub mod error;
pub mod fixed;
pub mod proximity;
pub mod report;
pub mod shape;

pub mod air;
pub mod deep;
pub mod grinding;
pub mod logup;
pub mod multilinear;

pub mod ldt;

pub mod binary;
pub mod bus;
pub mod fri;
pub mod whir;
pub mod word;

pub mod stark;

pub use assumption::SecurityAssumption;
pub use error::ErrorBits;
pub use grinding::GrindingSites;
pub use ldt::LowDegreeTest;
pub use multilinear::{
    BIT_RING_SWITCH_CLAIM_BATCHING_LABEL, BIT_RING_SWITCH_LABEL, COLUMN_BATCH_LABEL,
    bit_ring_switch_claim_batching_error, bit_ring_switch_claim_batching_term,
    bit_ring_switch_error, bit_ring_switch_tensors_error, bit_ring_switch_tensors_term,
    bit_ring_switch_term, column_batch_error, column_batch_term,
};
pub use report::{CandidateSet, Regime, RegimeReport, SecurityReport, SecurityTerm};
pub use shape::{InstanceShape, StarkAirParams};
