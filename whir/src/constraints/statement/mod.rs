//! Polynomial evaluation constraint systems for the WHIR protocol.
//!
//! Three constraint representations, each targeting a different protocol phase:
//!
//! - **Equality**: explicit multilinear points `p(z_i) = s_i`, batched via eq polynomials.
//! - **Select**: univariate evaluation claims, batched via the power-map select function.
//! - **Initial**: mutable builder that accumulates constraints during commitment.

/// Equality-based evaluation constraints with batched eq-polynomial combination.
pub mod eq;

/// Selection-based evaluation constraints using the power-map expansion.
pub mod select;

pub use eq::EqStatement;
pub use select::SelectStatement;
