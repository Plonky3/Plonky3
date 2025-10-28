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

use super::{GadgetConstraintContext, InteractionGadget};
use crate::error::LookupError;
use crate::interaction::{Interaction, eval_symbolic};

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
    fn eval_constraints<AB>(
        &self,
        builder: &mut AB,
        context: GadgetConstraintContext<'_, AB::F, AB::ExprEF>,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        if context.interactions.is_empty() {
            return;
        }

        // Access permutation challenges
        let challenges = builder.permutation_randomness();
        let alpha_idx = 2 * context.aux_column_index;
        let beta_idx = alpha_idx + 1;
        if challenges.len() <= beta_idx {
            panic!(
                "Insufficient permutation challenges. Interaction {} requires at least {} challenges, but only {} are available.",
                context.aux_column_index,
                beta_idx + 1,
                challenges.len()
            );
        }
        let alpha = challenges[alpha_idx];
        let beta = challenges[beta_idx];

        // Access running sum column
        let perm_trace = builder.permutation();
        let s = perm_trace.row_slice(0).unwrap();
        let s_next_slice = perm_trace.row_slice(1).unwrap();
        if s.len() <= context.aux_column_index {
            panic!(
                "Permutation trace has insufficient width. Expected at least {} columns, found {}.",
                context.aux_column_index + 1,
                s.len()
            );
        }
        let s_local: AB::ExprEF = s[context.aux_column_index].into();
        let s_next: AB::ExprEF = s_next_slice[context.aux_column_index].into();

        // Prepare the numerator and denominator for the LogUp constraint.
        let (numerator, denominator) = {
            let element_exprs: Vec<Vec<AB::ExprEF>> = context
                .interactions
                .iter()
                .map(|int| {
                    int.values
                        .iter()
                        .map(|expr| eval_symbolic(builder, expr).into())
                        .collect()
                })
                .collect();

            let multiplicity_exprs: Vec<AB::ExprEF> = context
                .interactions
                .iter()
                .map(|int| eval_symbolic(builder, &int.multiplicity).into())
                .collect();

            compute_combined_sum_terms::<AB, _, _>(
                &element_exprs,
                &multiplicity_exprs,
                &alpha.into(),
                &beta.into(),
            )
        };

        // Apply the core constraints based on whether it's a local or global interaction.
        builder.when_first_row().assert_zero_ext(s_local.clone());

        if let Some(expected) = context.expected_cumulative_sum {
            // GLOBAL INTERACTION
            // Transition constraint for all rows except the last.
            builder.when_transition().assert_zero_ext(
                (s_next - s_local.clone()) * denominator.clone() - numerator.clone(),
            );

            // Final constraint: The last row's contribution must lead to the expected total.
            let final_check = (expected - s_local) * denominator - numerator;
            builder.when_last_row().assert_zero_ext(final_check);
        } else {
            // LOCAL INTERACTION
            // The running sum must wrap around to zero. This single constraint applies to all rows.
            builder.assert_zero_ext((s_next - s_local) * denominator - numerator);
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
    fn constraint_degree<F: Field>(&self, interactions: &[Interaction<F>]) -> usize {
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
