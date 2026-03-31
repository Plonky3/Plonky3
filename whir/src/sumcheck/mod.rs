//! Sumcheck protocol implementation for multilinear polynomial verification.
//!
//! # The Sumcheck Protocol
//!
//! The sumcheck protocol verifies a claimed sum of the form:
//!
//! ```text
//! sum_{x in {0,1}^l} g(x) = t
//! ```
//!
//! Without it, the verifier would need to evaluate `g` at all `2^l` hypercube points.
//! The protocol reduces this to a single evaluation at a random point, using `l` rounds.
//!
//! In each round, the prover sends a low-degree univariate polynomial:
//!
//! ```text
//! h_i(X) = sum_{x_{i+1}, ..., x_l in {0,1}} g(r_1, ..., r_{i-1}, X, x_{i+1}, ..., x_l)
//! ```
//!
//! The verifier checks `h_i(0) + h_i(1) == claimed_sum`.
//! Then samples a random challenge `r_i` and updates the claim to `h_i(r_i)`.
//!
//! After `l` rounds, the verifier holds `(r_1, ..., r_l)` and queries `g` directly.

pub mod data;
pub mod error;
pub mod lagrange;
pub mod product_polynomial;
pub mod prover;
pub mod svo;

pub use data::{SumcheckData, verify_final_sumcheck_rounds};
pub use error::SumcheckError;
pub(crate) use lagrange::extrapolate_012;
