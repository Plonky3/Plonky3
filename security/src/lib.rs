//! Soundness analysis for STARK protocols.
//!
//! Decomposition is by **error source**, not by PCS. A protocol's params
//! type lives in its own crate; this crate provides the error-bit math and
//! the regime mirrors. The protocol assembles the regime, picks which
//! error terms apply, and composes them via [`ErrorBits::sum`] or
//! [`ErrorBits::min`] (or via [`stark::proven_security`] for the
//! AIR + DEEP + LDT composite).
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
//! Each `XxxRegime` mirrors the security-relevant subset of the
//! corresponding `XxxParameters` runtime config, **plus** instance-shape
//! metadata that doesn't live in the runtime config (trace length,
//! opening counts, …). There is intentionally no `From<XxxParameters>`
//! impl — the protocol crate is the only site with visibility into both
//! sides and assembles the regime explicitly.
//!
//! # References
//! - ethSTARK ([2021/582](https://eprint.iacr.org/2021/582))
//! - Proximity Gaps for Reed–Solomon Codes ([2020/654](https://eprint.iacr.org/2020/654))
//! - On the Security of STARKs with FRI ([2024/1553](https://eprint.iacr.org/2024/1553))
//! - On the Distribution of the Distances of Random Words ([2025/2010](https://eprint.iacr.org/2025/2010))
//! - BCHKS25 — Improved LDR proximity gaps ([2025/2055](https://eprint.iacr.org/2025/2055))

#![no_std]
#![allow(clippy::too_many_arguments)]

extern crate alloc;

pub mod assumption;
pub mod error;
pub mod proximity;
pub mod report;
pub mod shape;

pub mod air;
pub mod deep;
pub mod grinding;
pub mod logup;

pub mod ldt;

pub mod fri;
pub mod whir;

pub mod stark;

pub use assumption::SecurityAssumption;
pub use error::ErrorBits;
pub use ldt::LowDegreeTest;
pub use report::{Regime, RegimeReport, SecurityReport, SecurityTerm};
pub use shape::{InstanceShape, StarkAirParams};
