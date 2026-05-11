//! Lagrange selector structures for constraint evaluation.

use p3_circuit::symbolic::RowSelectorsTargets;

use crate::Target;

/// Circuit version of Lagrange selectors for AIR constraint evaluation.
#[derive(Clone, Debug)]
pub struct RecursiveLagrangeSelectors {
    /// Row selector targets (is_first_row, is_last_row, is_transition)
    pub row_selectors: RowSelectorsTargets,
    /// Inverse of the vanishing polynomial: 1 / Z_H(point)
    ///
    /// The vanishing polynomial Z_H(x) = x^n - 1 for a domain of size n.
    /// This inverse is used to compute the quotient polynomial evaluation.
    pub inv_vanishing: Target,
}
