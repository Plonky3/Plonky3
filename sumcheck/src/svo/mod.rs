//! Small-Value Optimization (SVO) for the sumcheck protocol.
//!
//! Optimizes sumcheck proving when the polynomial includes an equality polynomial factor.
//! Implements Algorithm 5 from "Speeding Up Sum-Check Proving" (ePrint 2025/1117).
//!
//! # Key Insight
//!
//! For polynomials of the form `g(X) = eq(w, X) * p(X)`, the equality polynomial can be decomposed as:
//!
//! ```text
//! eq(w, (r_{<i}, X, x')) = eq(w_{<i}, r_{<i}) * eq(w_i, X) * eq(w_{>i}, x')
//! ```
//!
//! This allows:
//! 1. Pre-computing smaller tables for the left and right components.
//! 2. Avoiding the full `2^l`-sized equality table.
//! 3. Reconstructing round polynomials from compact accumulators via Lagrange interpolation.

mod accumulator;
mod grid;
mod point;

pub(crate) use accumulator::{
    SvoAccumulators, calculate_accumulators_batch, calculate_product_accumulator,
};
pub(crate) use grid::evals_01inf_grid_prefix;
pub use point::SvoPoint;
