//! Polynomial evaluation constraint systems for the WHIR protocol.
//!
//! Three constraint representations, each targeting a different protocol phase:
//!
//! - **Equality**: explicit multilinear points `p(z_i) = s_i`, batched via eq polynomials.
//! - **Select**: univariate evaluation claims, batched via the power-map select function.
//! - **Linear Sigma**: explicit linear sumcheck weights
//!   `sum_b a(b) * p(b) = sigma`, matching the linear Sigma-IOP layer.
//! - **Initial**: mutable builder that accumulates constraints during commitment.

/// Equality-based evaluation constraints with batched eq-polynomial combination.
pub mod eq;

/// Initial statement builder for the commitment phase.
pub mod initial;

/// Linear Sigma-IOP constraints over explicit hypercube weights.
pub mod linear;

/// Selection-based evaluation constraints using the power-map expansion.
pub mod select;

pub use eq::EqStatement;
pub use linear::{
    BatchedLinearSigmaOpeningClaim, BatchedLinearSigmaOracleValues, BatchedLinearSigmaProverOracle,
    BatchedLinearSigmaReductionProof, LinearSigmaConstraint, LinearSigmaOpeningClaim,
    LinearSigmaReductionError, LinearSigmaReductionProof, LinearSigmaStatement,
    prove_batched_linear_sigma_reduction, verify_batched_linear_sigma_reduction,
};
pub use select::SelectStatement;
