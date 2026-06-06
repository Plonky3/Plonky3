//! Zero-knowledge prover overlay for the stacked sumcheck.
//!
//! The masking layer is binding-agnostic.
//! A single generic prover handles both binding modes.
//! Only the residual handoff at the end of the per-round loop differs:
//!
//! ```text
//!     mode    | residual factor
//!     --------+--------------------------------------
//!     prefix  | low-to-high fold, packed product
//!     suffix  | high-to-low fold, unpacked product
//! ```
//!
//! That branch lives behind a small trait extension on the non-private layout.
//! See [`layout::ZkLayout`].
//!
//! # Module layout
//!
//! - Shared prelude helpers: mask sampling, auxiliary-target bookkeeping.
//! - Per-round assembly: round-context type and the polynomial it builds.
//! - Layout trait extension: per-mode residual handoff.
//! - Generic prover: single `into_sumcheck` implementation parameterised by layout.

mod common;
mod layout;
mod residual;
mod round;
mod zk_prover;

pub use common::stack_codewords;
pub use layout::ZkLayout;
pub use zk_prover::{ZkPrefixProver, ZkProver, ZkSuffixProver};
