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

use super::InteractionGadget;
use crate::error::LookupError;
use crate::interaction::{Interaction, InteractionKind, eval_symbolic};

/// Computes the numerator and denominator of the fraction:
/// `∑(m_i / (α - combined_elements[i]))`, where
/// `combined_elements[i] = ∑elements[i][n-j] * β^j`.
///
/// Uses prefix/suffix products for O(n) computation instead of O(n²).
pub(crate) fn compute_combined_sum_terms<AB, E, M>(
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

    // Precompute all (α - ∑(value[j] * β^j)) terms
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
#[derive(Debug, Clone, Copy, Default)]
pub struct LogUpGadget;

impl LogUpGadget {
    /// Creates a new LogUp gadget instance.
    pub const fn new() -> Self {
        Self
    }
}

impl InteractionGadget for LogUpGadget {
    /// Evaluates the transition and boundary constraints for a lookup argument.
    ///
    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(multiplicities[i] / (α - combined_elements[i])) = expected_value
    /// ```
    ///
    /// where `multiplicities` can be negative, and
    /// `combined_elements[i] = ∑elements[i][n-j] * β^j`.
    ///
    /// For local interactions (`is_global = false`), `expected_value = 0` and the constraint
    /// wraps around (the last row contributes to the first).
    ///
    /// For global interactions (`is_global = true`), `expected_value` is provided by the prover,
    /// and the sum of all `expected_value`s across AIRs for this global interaction should be 0.
    /// The latter is checked as the final step after all AIRs have been verified.
    ///
    /// This is implemented using a running sum column that accumulates the contributions.
    ///
    /// # Arguments
    /// * `builder` - The AIR builder to construct expressions
    /// * `interactions` - The interactions containing elements and multiplicities
    /// * `aux_column_index` - Which auxiliary column stores the running sum
    /// * `is_global` - Whether this is a global interaction (across multiple AIRs)
    /// * `expected_cumulative` - For global interactions, the expected cumulative value
    fn eval_constraints<AB, K>(
        &self,
        builder: &mut AB,
        interactions: &[Interaction<AB::F, K>],
        aux_column_index: usize,
        is_global: bool,
        expected_cumulative: Option<AB::ExprEF>,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
        K: InteractionKind,
    {
        // Validate parameters
        if is_global {
            assert!(
                expected_cumulative.is_some(),
                "Global interactions require an expected cumulative value"
            );
        } else {
            assert!(
                expected_cumulative.is_none(),
                "Local interactions should not have an expected cumulative value"
            );
        }

        if interactions.is_empty() {
            return;
        }

        // Extract elements and multiplicities from interactions
        let element_exprs: Vec<_> = interactions
            .iter()
            .map(|interaction| {
                interaction
                    .values
                    .iter()
                    .map(|expr| eval_symbolic(builder, expr).into())
                    .collect()
            })
            .collect();

        let multiplicity_exprs: Vec<_> = interactions
            .iter()
            .map(|interaction| eval_symbolic(builder, &interaction.multiplicity).into())
            .collect();

        // Access the permutation (auxiliary) table
        let permutation = builder.permutation();
        let permutation_challenges = builder.permutation_randomness();

        // We need 2 challenges per auxiliary column: α and β
        let required_challenges = 2 * (aux_column_index + 1);
        assert!(
            permutation_challenges.len() >= required_challenges,
            "Insufficient permutation challenges: need {}, have {}",
            required_challenges,
            permutation_challenges.len()
        );

        // Challenge for the running sum.
        let alpha = permutation_challenges[2 * aux_column_index];
        // Challenge for combining the lookup tuples.
        let beta = permutation_challenges[2 * aux_column_index + 1];

        // Get the running sum column
        let s = permutation.row_slice(0).unwrap();
        assert!(
            s.len() > aux_column_index,
            "Permutation trace has insufficient width"
        );

        // Read s[i] from the local row at the specified column.
        let s_local = s[aux_column_index].into();
        // Read s[i+1] from the next row (or a zero-padded view on the last row).
        let s_next = permutation.row_slice(1).unwrap()[aux_column_index].into();

        // Anchor s[0] = 0 at the start.
        //
        // Avoids a high-degree boundary constraint.
        // Telescoping is enforced by the last-row check (s[n−1] + contribution[n-1] = 0).
        // This keeps aux and main traces aligned in length.
        builder.when_first_row().assert_zero_ext(s_local.clone());

        // Build the fraction:  ∑ m_i/(α - combined_elements[i])  =  numerator / denominator .
        let (numerator, common_denominator) =
            compute_combined_sum_terms::<AB, AB::ExprEF, AB::ExprEF>(
                &element_exprs,
                &multiplicity_exprs,
                &alpha.into(),
                &beta.into(),
            );

        if is_global {
            let expected = expected_cumulative.unwrap();

            // Transition constraint (for non-last rows)
            builder.when_transition().assert_zero_ext(
                (s_next - s_local.clone()) * common_denominator.clone() - numerator.clone(),
            );

            // Final constraint (last row)
            let final_val = (expected - s_local) * common_denominator - numerator;
            builder.when_last_row().assert_zero_ext(final_val);
        } else {
            // If we are in a local lookup, the previous transition constraint doesn't have to be limited to transition rows:
            // - we are already ensuring that the first row is 0,
            // - at point `g^{n - 1}` (where `n` is the domain size), the next point is `g^0`, so that the constraint still holds
            // on the last row.
            builder.assert_zero_ext((s_next - s_local) * common_denominator - numerator);
        }
    }

    /// Computes the constraint degree for the transition constraint.
    ///
    /// The transition constraint is:
    /// ```text
    /// (s[i+1] - s[i]) * common_denominator - numerator = 0
    /// ```
    ///
    /// Where:
    /// - `common_denominator = ∏_j(α - combined_elements[j])`
    /// - `numerator = ∑_i(m_i * ∏_{j≠i}(α - combined_elements[j]))`
    /// - Each `combined_elements[i] = ∑elements[i][n-j] * β^j`
    ///
    /// The degree is `1 + max(deg(numerator), deg(common_denominator))`.
    fn constraint_degree<F: Field, K: InteractionKind>(
        &self,
        interactions: &[Interaction<F, K>],
    ) -> usize {
        if interactions.is_empty() {
            return 0;
        }

        let n = interactions.len();

        // Compute degree of each combined value
        let mut combined_degrees = Vec::with_capacity(n);
        let mut total_combined_degree = 0;

        for interaction in interactions {
            let deg = interaction
                .values
                .iter()
                .map(|expr| expr.degree_multiple())
                .max()
                .unwrap_or(0);
            combined_degrees.push(deg);
            total_combined_degree += deg;
        }

        // Degree of denominator constraint: 1 + deg(common_denominator)
        let deg_denom_constraint = 1 + total_combined_degree;

        // Degree of numerator: max over all interactions of
        // deg(multiplicity) + sum of combined degrees except this one
        let deg_numerator = (0..n)
            .map(|i| {
                let mult_deg = interactions[i].multiplicity.degree_multiple();
                mult_deg + total_combined_degree - combined_degrees[i]
            })
            .max()
            .unwrap_or(0);

        deg_denom_constraint.max(deg_numerator)
    }

    /// Verifies that the sum of all global cumulative values is zero.
    ///
    /// For global interactions, each AIR computes a cumulative value. This function
    /// ensures that the sum of all these values equals zero, which proves that all
    /// sends and receives balance globally across the entire STARK system.
    ///
    /// # Arguments
    /// * `all_expected_cumulative` - Cumulative values from all AIRs for this global interaction
    ///
    /// # Returns
    /// * `Ok(())` if the sum is zero (protocol satisfied)
    /// * `Err(LookupError)` if the sum is non-zero (protocol violated)
    fn verify_global_final_value<F: Field>(
        &self,
        all_expected_cumulative: &[F],
    ) -> Result<(), LookupError> {
        let total = all_expected_cumulative.iter().cloned().sum::<F>();

        if !total.is_zero() {
            return Err(LookupError::global_mismatch(
                "global",
                total,
                all_expected_cumulative.len(),
            ));
        }

        Ok(())
    }
}
