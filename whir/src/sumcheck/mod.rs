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
pub mod layout;
pub mod product_polynomial;
pub mod strategy;
pub mod svo;
pub mod table;
#[cfg(test)]
pub(crate) mod tests;

pub use data::{SumcheckData, verify_final_sumcheck_rounds};
pub use error::SumcheckError;
pub(crate) use lagrange::extrapolate_01inf;
use p3_field::Field;
pub use table::{OpeningProtocol, PointSchedule, TableShape, TableSpec};

/// A claimed evaluation together with layout-specific auxiliary data.
#[derive(Debug, Clone)]
pub struct Claim<F: Field, P, Data> {
    /// Point representation used to evaluate or later reconstruct this claim.
    pub(crate) point: P,
    /// Claimed value at `point`.
    pub(crate) eval: F,
    /// Extra strategy-specific prover or verifier metadata.
    pub(crate) data: Data,
}

impl<F: Field, P, Data> Claim<F, P, Data> {
    /// Returns the claimed value.
    pub const fn eval(&self) -> F {
        self.eval
    }

    /// Returns the point representation attached to this claim.
    pub const fn point(&self) -> &P {
        &self.point
    }
}
