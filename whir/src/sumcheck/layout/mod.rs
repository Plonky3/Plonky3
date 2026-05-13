//! Stacked sumcheck layout.
//!
//! # Why stack
//!
//! - A WHIR commitment carries a fixed per-commitment overhead (FFT + Merkle).
//! - Committing each table separately multiplies that overhead by the table count.
//! - Stacking concatenates every table into one multilinear polynomial and
//!   commits once, so a single commitment covers all tables.
//! - WHIR natively supports multiple opening claims on the committed polynomial,
//!   so no extra batching sumcheck is needed on top.
//!
//! # Layout of the stacked polynomial
//!
//! - Sort source tables by arity, largest first.
//! - Lay columns out back-to-back on the boolean hypercube.
//! - Each column takes a contiguous slot of size `2^arity`.
//! - Pad with zeros up to the next power of two.
//! - The concatenation is itself a multilinear polynomial.
//!
//! Example: three tables with `(4, 3, 2)` variables and one column each.
//!
//! ```text
//!     +---- 16 ----+-- 8 --+-- 4 --+-- pad --+
//!     |    P_1     |  P_2  |  P_3  |  zeros  |
//!     +------------+-------+-------+---------+
//!     0           16      24      28        32
//! ```
//!
//! # Selectors: addressing a slot by a boolean prefix
//!
//! - Each column is reached by prefixing its local variables with a boolean
//!   selector that picks the slot.
//! - Local variables follow the selector bits.
//!
//! For the example above, with `P` the stacked polynomial:
//!
//! ```text
//!     P_1(x_1, x_2, x_3, x_4) = P(0,       x_1, x_2, x_3, x_4)
//!     P_2(x_1, x_2, x_3)      = P(1, 0,    x_1, x_2, x_3)
//!     P_3(x_1, x_2)           = P(1, 1, 0, x_1, x_2)
//! ```
//!
//! # Why selector lifts stay cheap in WHIR
//!
//! - The WHIR cost of adding an equality constraint `P(z) = y` scales as
//!   `O(2^k)`, where `k` counts the coordinates of `z` that are not in `{0, 1}`.
//! - Selector coordinates are always boolean, so they do not inflate `k`.
//! - Lifting a local claim into a stacked claim therefore adds no asymptotic
//!   cost beyond the original local coordinates.
//!
//! # Residual sumcheck: two binding modes
//!
//! After lifting, a batching challenge `alpha` collapses every recorded
//! opening into one residual claim. The residual sumcheck can bind variables
//! in two different orders:
//!
//! - Prefix-first binding: round one runs in SIMD-packed arithmetic, and the
//!   remaining rounds drive a product polynomial.
//! - Suffix-first binding: SVO accumulators are precomputed at claim-recording
//!   time and folded round by round with Lagrange weights.
//!
//! Both modes end at the same residual product polynomial; the binding order
//! only decides which fast-path tricks apply on the first rounds.
//!
//! # References
//!
//! - Stacking construction: "Minimal zkVM for Lean Ethereum", section 3.7.
//! - WHIR proximity argument: ePrint 2024/1586.
//! - SVO accumulators: ePrint 2025/1117, Algorithm 5.
//! - Jagged PCS: ePrint 2025/0917.

mod opening;
mod plan;
mod prover;
mod verifier;
mod witness;

pub use opening::{
    MultiClaim, Opening, ProverMultiClaim, ProverVirtualClaim, VerifierMultiClaim, VerifierOpening,
    VerifierVirtualClaim,
};
pub use prover::{Layout, PrefixProver, SuffixProver};
pub use verifier::Verifier;
pub use witness::{Selector, Table, TablePlacement, Witness};

use crate::sumcheck::strategy::VariableOrder;
pub use crate::sumcheck::table::TableShape;

/// Verifier-side metadata required to replay a committed layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutStrategy {
    /// Whether selector bits are reversed and appended after local bits.
    pub reverse_selectors: bool,
    /// Variable order used by the residual WHIR/sumcheck rounds.
    pub variable_order: VariableOrder,
}

impl LayoutStrategy {
    pub const fn new(reverse_selectors: bool, variable_order: VariableOrder) -> Self {
        Self {
            reverse_selectors,
            variable_order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_strategy_new_stores_constructor_arguments_verbatim() {
        // Invariant:
        //     LayoutStrategy::new copies its two arguments into the matching
        //     fields without mutation. The two reachable shapes — prefix
        //     binding without selector reversal, and suffix binding with
        //     selector reversal — are exercised exactly the same way.
        //
        // Fixture state:
        //     case A: (reverse_selectors = false, Prefix)
        //     case B: (reverse_selectors = true,  Suffix)
        let case_a = LayoutStrategy::new(false, VariableOrder::Prefix);
        let case_b = LayoutStrategy::new(true, VariableOrder::Suffix);

        assert!(!case_a.reverse_selectors);
        assert_eq!(case_a.variable_order, VariableOrder::Prefix);

        assert!(case_b.reverse_selectors);
        assert_eq!(case_b.variable_order, VariableOrder::Suffix);
    }
}
