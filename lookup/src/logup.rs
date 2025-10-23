//! Core LogUp Implementation
//!
//! ## Mathematical Foundation
//!
//! LogUp transforms the standard lookup equation:
//! ```text
//! ∏(α - a_i)^(m_i) = ∏(α - b_j)^(m'_j)
//! ```
//!
//! Into an equivalent sum-based form using logarithmic derivatives:
//! ```text
//! ∑(m_i/(α - a_i)) = ∑(m'_j/(α - b_j))
//! ```
//!
//! Where:
//! - `α` is a random challenge
//! - `m_i, m'_j` are multiplicities (how many times each element appears)
//! - The transformation eliminates expensive exponentiation operations

use alloc::vec;
use alloc::vec::Vec;

use p3_air::{AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::lookup_traits::{Kind, Lookup, LookupError, LookupGadget, symbolic_to_expr};

/// Core LogUp gadget implementing lookup arguments via logarithmic derivatives.
///
/// The LogUp gadget transforms the multiplicative lookup constraint:
/// ```text
/// ∏(α - a_i)^(m_i) = ∏(α - b_j)^(m'_j)
/// ```
///
/// Into an equivalent additive constraint using logarithmic differentiation:
/// ```text
/// ∑(m_i/(α - a_i)) = ∑(m'_j/(α - b_j))
/// ```
///
/// This is implemented using a running sum auxiliary column `s` that accumulates:
/// ```text
/// s[i+1] = s[i] + ∑(m_a/(α - a)) - ∑(m_b/(α - b))
/// ```
///
/// Note that we do not differentiate between `a` and `b` in the implementation:
/// we simply have a list of `elements` with possibly negative `multiplicities`.
///
/// Constraints are defined as:
/// - **Initial Constraint**: `s[0] = 0`
/// - **Transition Constraint**: `s[i+1] = s[i] + contribution[i]`
/// - **Final Constraint**: `s[n-1] + contribution[n-1] = 0`
#[derive(Debug, Clone, Default)]
pub struct LogUpGadget;

impl LogUpGadget {
    /// Creates a new LogUp gadget instance.
    pub const fn new() -> Self {
        Self {}
    }

    /// Computes the numerator and denominator of the fraction:
    /// `∑(m_i / (α - combined_elements[i]))`, where
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j
    pub(crate) fn compute_combined_sum_terms<AB, E, M>(
        &self,
        elements: &[Vec<E>],
        multiplicities: &[M],
        alpha: &AB::ExprEF,
        beta: &AB::ExprEF,
    ) -> (AB::ExprEF, AB::ExprEF)
    where
        AB: PermutationAirBuilder,
        E: Into<AB::ExprEF> + Clone,
        M: Into<AB::ExprEF> + Clone,
    {
        if elements.is_empty() {
            return (AB::ExprEF::ZERO, AB::ExprEF::ONE);
        }

        let n = elements.len();

        // Precompute all (α - ∑e_{i, j} β^j) terms
        let terms = elements
            .iter()
            .map(|elts| {
                // Combine the elements in the tuple using beta.
                let combined_elt = elts.iter().fold(AB::ExprEF::ZERO, |acc, elt| {
                    elt.clone().into() + acc * beta.clone()
                });

                // Compute (α - combined_elt)
                alpha.clone() - combined_elt
            })
            .collect::<Vec<_>>();

        // Build prefix products: pref[i] = ∏_{j=0}^{i-1}(α - e_j)
        let mut pref = Vec::with_capacity(n + 1);
        pref.push(AB::ExprEF::ONE);
        for t in &terms {
            pref.push(pref.last().unwrap().clone() * t.clone());
        }

        // Build suffix products: suff[i] = ∏_{j=i}^{n-1}(α - e_j)
        let mut suff = vec![AB::ExprEF::ONE; n + 1];
        for i in (0..n).rev() {
            suff[i] = suff[i + 1].clone() * terms[i].clone();
        }

        // Common denominator is the product of all terms
        let common_denominator = pref[n].clone();

        // Compute numerator: ∑(m_i * ∏_{j≠i}(α - e_j))
        //
        // The product without i is: pref[i] * suff[i+1]
        let numerator = (0..n).fold(AB::ExprEF::ZERO, |acc, i| {
            acc + multiplicities[i].clone().into() * pref[i].clone() * suff[i + 1].clone()
        });

        (numerator, common_denominator)
    }

    /// Evaluates the transition and boundary constraints for a lookup argument.
    ///
    /// # Arguments:
    /// * builder - The AIR builder to construct expressions.
    /// * context - The lookup context containing:
    ///     * the kind of lookup (local or global),
    ///     * elements,
    ///     * multiplicities,
    ///     * and auxiliary column indices.
    /// * opt_expected_cumulated - Optional expected cumulative value for global lookups. For local lookups, this should be `None`.
    fn eval_update<AB>(
        &self,
        builder: &mut AB,
        context: Lookup<AB::F>,
        opt_expected_cumulated: Option<AB::ExprEF>,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        let Lookup {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns,
        } = context;

        assert!(
            element_exprs.len() == multiplicities_exprs.len(),
            "Mismatched lengths: elements and multiplicities must have same length"
        );
        assert_eq!(
            columns.len(),
            self.num_aux_cols(),
            "There is exactly one auxiliary column for LogUp"
        );
        let column = columns[0];

        // First, turn the symbolic expressions into builder expressions, for elements and multiplicities.
        let elements = element_exprs
            .iter()
            .map(|exprs| {
                exprs
                    .iter()
                    .map(|expr| symbolic_to_expr(builder, expr))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let multiplicities = multiplicities_exprs
            .iter()
            .map(|expr| symbolic_to_expr(builder, expr))
            .collect::<Vec<_>>();

        // Access the permutation (aux) table. It carries the running sum column `s`.
        let permutation = builder.permutation();

        let permutation_challenges = builder.permutation_randomness();

        assert!(
            permutation_challenges.len() >= self.num_challenges() * (column + 1),
            "Insufficient permutation challenges"
        );

        // Challenge for the running sum.
        let alpha = permutation_challenges[2 * column];
        // Challenge for combining the lookup tuples.
        let beta = permutation_challenges[2 * column + 1];

        let s = permutation.row_slice(0).unwrap();
        assert!(s.len() > column, "Permutation trace has insufficient width");

        // Read s[i] from the local row at the specified column.
        let s_local = s[column].into();
        // Read s[i+1] from the next row (or a zero-padded view on the last row).
        let s_next = permutation.row_slice(1).unwrap()[column].into();

        // Anchor s[0] = 0 at the start.
        //
        // Avoids a high-degree boundary constraint.
        // Telescoping is enforced by the last-row check (s[n−1] + contribution[n-1] = 0).
        // This keeps aux and main traces aligned in length.
        builder.when_first_row().assert_zero_ext(s_local.clone());

        // Build the fraction:  ∑ m_i/(α - combined_elements[i])  =  numerator / denominator .
        let (numerator, common_denominator) = self
            .compute_combined_sum_terms::<AB, AB::ExprEF, AB::ExprEF>(
                &elements,
                &multiplicities,
                &alpha.into(),
                &beta.into(),
            );

        if let Some(expected_cumulated) = opt_expected_cumulated {
            // If there is an `expected_cumulated`, we are in a global lookup update.
            assert!(
                matches!(kind, Kind::Global(_)),
                "Expected cumulated value provided for a non-global lookup"
            );

            // Transition constraint:
            builder.when_transition().assert_zero_ext(
                (s_next - s_local.clone()) * common_denominator.clone() - numerator.clone(),
            );

            // Final constraint:
            let final_val = (expected_cumulated - s_local) * common_denominator - numerator;
            builder.when_last_row().assert_zero_ext(final_val);
        } else {
            // If we don't have an `expected_cumulated`, we are in a local lookup update.
            assert!(
                matches!(kind, Kind::Local),
                "No expected cumulated value provided for a global lookup"
            );

            // If we are in a local lookup, the previous transition constraint doesn't have to be limited to transition rows:
            // - we are already ensuring that the first row is 0,
            // - at point `g^{n - 1}` (where `n` is the domain size), the next point is `g^0`, so that the constraint still holds
            // on the last row.
            builder.assert_zero_ext((s_next - s_local) * common_denominator - numerator);
        }
    }
}

impl LookupGadget for LogUpGadget {
    fn num_aux_cols(&self) -> usize {
        1
    }

    fn num_challenges(&self) -> usize {
        2
    }

    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(multiplicities[i] / (α - combined_elements[i])) = 0
    /// ```
    ///
    /// where `multiplicities` can be negative, and
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`.
    ///
    /// This is implemented using a running sum column that should sum to zero.
    fn eval_local_lookup<AB>(&self, builder: &mut AB, context: Lookup<AB::F>)
    where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        if let Kind::Global(_) = context.kind {
            panic!("Global lookups are not supported in local evaluation")
        }

        self.eval_update(builder, context, None);
    }

    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(multiplicities[i] / (α - combined_elements[i])) = `expected_cumulated`
    /// ```
    ///
    /// where `multiplicities` can be negative, and
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`.
    ///
    /// `expected_cumulated` is provided by the prover, and the sum of all `expected_cumulated` for this global interaction
    /// should be 0. The latter is checked as the final step, after all AIRS have been verified.
    ///
    /// This is implemented using a running sum column that should sum to `expected_cumulated`.
    fn eval_global_update<AB>(
        &self,
        builder: &mut AB,
        context: Lookup<AB::F>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        self.eval_update(builder, context, Some(expected_cumulated));
    }

    fn verify_global_final_value<EF: Field>(
        &self,
        all_expected_cumulative: &[EF],
    ) -> Result<(), LookupError> {
        let total = all_expected_cumulative.iter().cloned().sum::<EF>();

        if !total.is_zero() {
            return Err(LookupError::GlobalCumulativeMismatch);
        }

        Ok(())
    }

    /// We need to compute the degree of the transition constraint,
    /// as it is the constraint with highest degree:
    /// `(s[n + 1] - s[n]) * common_denominator - numerator = 0`
    ///
    /// But in `common_denominator`, each combined element e_i = ∑e_{i, j} β^j
    /// contributes (α - e_i). So we need to sum the degree of all
    /// combined elements to find the degree of the common denominator.
    ///
    /// `numerator = ∑(m_i * ∏_{j≠i}(α - e_j))`, where the e_j are the combined elements.
    /// So we have to compute the max of all m_i * ∏_{j≠i}(α - e_j).
    ///
    /// The constraint degree is then:
    /// `1 + max(deg(numerator), deg(common_denominator))`
    fn constraint_degree<F: Field>(&self, context: Lookup<F>) -> usize {
        assert!(context.multiplicities_exprs.len() == context.element_exprs.len());

        let n = context.multiplicities_exprs.len();

        // Compute degrees in a single pass.
        let mut degs = Vec::with_capacity(n);
        let mut deg_sum = 0;
        for elems in &context.element_exprs {
            let deg = elems
                .iter()
                .map(|elt| elt.degree_multiple())
                .max()
                .unwrap_or(0);
            degs.push(deg);
            deg_sum += deg;
        }

        // Compute 1 + degree(denominator).
        let deg_denom_constr = 1 + deg_sum;

        // Compute degree(numerator).
        let multiplicities = &context.multiplicities_exprs;
        let deg_num = (0..n)
            .map(|i| multiplicities[i].degree_multiple() + deg_sum - degs[i])
            .max()
            .unwrap_or(0);

        deg_denom_constr.max(deg_num)
    }
}
