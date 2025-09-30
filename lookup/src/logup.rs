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

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{ExtensionBuilder, PermutationAirBuilder};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

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
/// With boundary constraints:
/// - `s[0] = ∑(m_a/(α - a)) - ∑(m_b/(α - b))` (correctly computed initial value)
/// - `s[n-1] = 0` (all terms cancel out if lookup is valid)
#[derive(Debug, Clone, Default)]
pub struct LogUpGadget<F> {
    _phantom: PhantomData<F>,
}

impl<F: Field> LogUpGadget<F> {
    /// Creates a new LogUp gadget instance.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Asserts lookup equality using LogUp with multiplicities support.
    ///
    /// This method enforces that elements from `a_elements` with their associated
    /// `a_multiplicities` form the same multiset as elements from `b_elements`
    /// with their `b_multiplicities`.
    ///
    /// # Arguments
    /// * `builder` - AIR builder for constraint generation
    /// * `a_elements` - Elements from the main execution trace
    /// * `a_multiplicities` - How many times each `a_element` should appear
    /// * `b_elements` - Elements from the lookup table
    /// * `b_multiplicities` - How many times each `b_element` should appear
    /// * `challenge` - Random challenge `α` for the LogUp argument
    ///
    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(a_multiplicities[i] / (α - a_elements[i])) = ∑_j(b_multiplicities[j] / (α - b_elements[j]))
    /// ```
    ///
    /// This is implemented using a running sum column that should sum to zero.
    pub fn assert_lookup_with_multiplicities<AB, AE, AM, BE, BM>(
        &self,
        builder: &mut AB,
        a_elements: &[AE],
        a_multiplicities: &[AM],
        b_elements: &[BE],
        b_multiplicities: &[BM],
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Copy,
        AM: Into<AB::ExprEF> + Copy,
        BE: Into<AB::ExprEF> + Copy,
        BM: Into<AB::ExprEF> + Copy,
    {
        self.assert_lookup_with_multiplicities_internal(
            builder,
            a_elements,
            a_multiplicities,
            b_elements,
            b_multiplicities,
            challenge,
        );
    }

    /// Internal implementation that doesn't require Copy on multiplicity expressions
    fn assert_lookup_with_multiplicities_internal<AB, AE, AM, BE, BM>(
        &self,
        builder: &mut AB,
        a_elements: &[AE],
        a_multiplicities: &[AM],
        b_elements: &[BE],
        b_multiplicities: &[BM],
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Copy,
        AM: Into<AB::ExprEF> + Clone,
        BE: Into<AB::ExprEF> + Copy,
        BM: Into<AB::ExprEF> + Clone,
    {
        assert_eq!(
            a_elements.len(),
            a_multiplicities.len(),
            "Mismatched lengths: a_elements and a_multiplicities must have same length"
        );
        assert_eq!(
            b_elements.len(),
            b_multiplicities.len(),
            "Mismatched lengths: b_elements and b_multiplicities must have same length"
        );

        let permutation = builder.permutation();
        let s_local = permutation.row_slice(0).unwrap()[0];
        let s_next = permutation.row_slice(1).unwrap()[0];

        // Boundary constraint for first row: s[0] = ∑(a_mults / (α - a_elems)) - ∑(b_mults / (α - b_elems))
        // This ensures the permutation trace starts with the correct initial value.
        self.add_initial_constraint(
            builder,
            s_local.into(),
            a_elements,
            a_multiplicities,
            b_elements,
            b_multiplicities,
            challenge,
        );

        // Boundary constraint: s[n-1] = 0. This is checked on the local value of the last row.
        builder.when_last_row().assert_zero_ext(s_local);

        // Add the transition constraint for the running sum update.
        self.add_logup_transition_constraint(
            builder,
            s_local.into(),
            s_next.into(),
            a_elements,
            a_multiplicities,
            b_elements,
            b_multiplicities,
            challenge,
        );
    }

    /// Convenience method for simple lookup without explicit multiplicities.
    ///
    /// This is equivalent to calling `assert_lookup_with_multiplicities` where
    /// all multiplicities are set to 1. It is recommended to use the more general
    /// function for all lookups to ensure consistency.
    pub fn assert_lookup<AB, AE, BE>(
        &self,
        builder: &mut AB,
        a_elements: &[AE],
        b_elements: &[BE],
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Copy,
        BE: Into<AB::ExprEF> + Copy,
    {
        // Create a direct lookup by passing one-vectors as multiplicities
        let a_ones: Vec<_> = (0..a_elements.len()).map(|_| AB::ExprEF::ONE).collect();
        let b_ones: Vec<_> = (0..b_elements.len()).map(|_| AB::ExprEF::ONE).collect();

        // Now call the generic function with proper signature matching
        self.assert_lookup_with_multiplicities_internal(
            builder, a_elements, &a_ones, b_elements, &b_ones, challenge,
        );
    }

    /// Implements the initial boundary constraint for the first row.
    ///
    /// The initial constraint enforces:
    /// ```text
    /// s[0] = ∑_j(a_mults[j] / (α - a_elems[j])) - ∑_j(b_mults[j] / (α - b_elems[j]))
    /// ```
    ///
    /// Since division is not allowed in polynomial constraints, we clear denominators
    /// by multiplying through by the common denominator, yielding a polynomial constraint.
    #[allow(clippy::too_many_arguments)]
    fn add_initial_constraint<AB, AE, AM, BE, BM>(
        &self,
        builder: &mut AB,
        s_local: AB::ExprEF,
        a_elements: &[AE],
        a_multiplicities: &[AM],
        b_elements: &[BE],
        b_multiplicities: &[BM],
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Copy,
        AM: Into<AB::ExprEF> + Clone,
        BE: Into<AB::ExprEF> + Copy,
        BM: Into<AB::ExprEF> + Clone,
    {
        // Convert challenge to expression
        let alpha: AB::ExprEF = challenge.into();

        // Compute the sum terms for the first row
        let (a_sum_numerator, a_common_denominator): (AB::ExprEF, AB::ExprEF) =
            self.compute_sum_terms::<AB, AE, AM>(a_elements, a_multiplicities, &alpha);
        let (b_sum_numerator, b_common_denominator): (AB::ExprEF, AB::ExprEF) =
            self.compute_sum_terms::<AB, BE, BM>(b_elements, b_multiplicities, &alpha);

        // The constraint is: s[0] * common_denom = a_terms * b_denom - b_terms * a_denom
        // Where common_denom = a_common_denominator * b_common_denominator
        let common_denominator = a_common_denominator.clone() * b_common_denominator.clone();

        // Left side: s[0] * common_denominator
        let lhs = s_local * common_denominator;

        // Right side: a_numerator * b_denominator - b_numerator * a_denominator
        let rhs = a_sum_numerator * b_common_denominator - b_sum_numerator * a_common_denominator;

        // Enforce: lhs - rhs = 0 on the first row only
        builder.when_first_row().assert_zero_ext(lhs - rhs);
    }

    /// Implements the core LogUp transition constraint.
    ///
    /// The transition constraint enforces the running sum update rule:
    /// ```text
    /// s[i+1] = s[i] + ∑_j(a_mults[j] / (α - a_elems[j])) - ∑_j(b_mults[j] / (α - b_elems[j]))
    /// ```
    ///
    /// Since division is not allowed in polynomial constraints, we clear denominators
    /// by multiplying through by the common denominator, yielding a polynomial constraint.
    #[allow(clippy::too_many_arguments)]
    fn add_logup_transition_constraint<AB, AE, AM, BE, BM>(
        &self,
        builder: &mut AB,
        s_local: AB::ExprEF,
        s_next: AB::ExprEF,
        a_elements: &[AE],
        a_multiplicities: &[AM],
        b_elements: &[BE],
        b_multiplicities: &[BM],
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Copy,
        AM: Into<AB::ExprEF> + Clone,
        BE: Into<AB::ExprEF> + Copy,
        BM: Into<AB::ExprEF> + Clone,
    {
        // Convert challenge to expression
        let alpha: AB::ExprEF = challenge.into();

        // Compute the sum terms for this row
        let (a_sum_numerator, a_common_denominator): (AB::ExprEF, AB::ExprEF) =
            self.compute_sum_terms::<AB, AE, AM>(a_elements, a_multiplicities, &alpha);
        let (b_sum_numerator, b_common_denominator): (AB::ExprEF, AB::ExprEF) =
            self.compute_sum_terms::<AB, BE, BM>(b_elements, b_multiplicities, &alpha);

        // The constraint is: (s_next - s_local) * common_denom = a_terms * b_denom - b_terms * a_denom
        // Where common_denom = a_common_denominator * b_common_denominator
        let common_denominator = a_common_denominator.clone() * b_common_denominator.clone();

        // Left side: (s[i+1] - s[i]) * common_denominator
        let lhs = (s_next - s_local) * common_denominator;

        // Right side: a_numerator * b_denominator - b_numerator * a_denominator
        let rhs = a_sum_numerator * b_common_denominator - b_sum_numerator * a_common_denominator;

        // Enforce: lhs - rhs = 0 on all transition rows
        builder.when_transition().assert_zero_ext(lhs - rhs);
    }

    /// Computes the numerator and common denominator for a set of fractional terms.
    ///
    /// For terms of the form `∑(m_i/(α - e_i))`, this computes:
    /// - Numerator: `∑(m_i * ∏_{j≠i}(α - e_j))`
    /// - Common denominator: `∏_i(α - e_i)`
    ///
    /// This allows us to express the sum as a single fraction without division
    /// operations in the constraint system.
    fn compute_sum_terms<AB, E, M>(
        &self,
        elements: &[E],
        multiplicities: &[M],
        alpha: &AB::ExprEF,
    ) -> (AB::ExprEF, AB::ExprEF)
    where
        AB: PermutationAirBuilder,
        E: Into<AB::ExprEF> + Copy,
        M: Into<AB::ExprEF> + Clone,
    {
        if elements.is_empty() {
            return (AB::ExprEF::ZERO, AB::ExprEF::ONE);
        }

        // Precompute all (α - e_i) terms
        let terms: Vec<AB::ExprEF> = elements.iter().map(|&e| alpha.clone() - e.into()).collect();

        // Compute common denominator: ∏(α - e_i)
        let common_denominator = terms.iter().cloned().reduce(|acc, x| acc * x).unwrap();

        // Compute numerator: ∑(m_i * ∏_{j≠i}(α - e_j))
        let mut numerator = AB::ExprEF::ZERO;
        for (i, multiplicity) in multiplicities.iter().enumerate() {
            let mult_expr: AB::ExprEF = multiplicity.clone().into();

            // Compute ∏_{j≠i}(α - e_j)
            let mut partial_product = AB::ExprEF::ONE;
            for (j, term) in terms.iter().enumerate() {
                if i != j {
                    partial_product *= term.clone();
                }
            }
            numerator += mult_expr * partial_product;
        }

        (numerator, common_denominator)
    }

    /// Computes the polynomial degree of the LogUp transition constraint.
    ///
    /// The degree depends on the number of elements in the lookup, as each element
    /// contributes a factor `(α - element)` to the constraint polynomial.
    ///
    /// # Formula
    /// For `n_a` elements from set A and `n_b` elements from set B, the degree is:
    /// ```text
    /// degree = 1 + n_a + n_b
    /// ```
    ///
    /// The `+1` accounts for the running sum column interaction.
    pub fn constraint_degree(&self, a_elements_count: usize, b_elements_count: usize) -> usize {
        // Running sum column contributes degree 1
        1 + a_elements_count + b_elements_count
    }
}

/// Trait extension providing LogUp functionality to any PermutationAirBuilder.
pub trait LookupBuilder: PermutationAirBuilder {
    /// Assert lookup with multiplicities using LogUp.
    ///
    /// # Arguments
    /// * `a_elements` - Elements from the main trace
    /// * `a_multiplicities` - Multiplicities for each element in A
    /// * `b_elements` - Elements from the lookup table
    /// * `b_multiplicities` - Multiplicities for each element in B
    /// * `challenge` - Random challenge for the lookup argument
    fn assert_lookup_with_multiplicities<AE, AM, BE, BM>(
        &mut self,
        a_elements: &[AE],
        a_multiplicities: &[AM],
        b_elements: &[BE],
        b_multiplicities: &[BM],
        challenge: Self::RandomVar,
    ) where
        AE: Into<Self::ExprEF> + Copy,
        AM: Into<Self::ExprEF> + Copy,
        BE: Into<Self::ExprEF> + Copy,
        BM: Into<Self::ExprEF> + Copy;

    /// Assert simple lookup without explicit multiplicities (all elements appear once).
    ///
    /// # Arguments
    /// * `a_elements` - Elements from the main trace
    /// * `b_elements` - Elements from the lookup table
    /// * `challenge` - Random challenge for the lookup argument
    fn assert_lookup<AE, BE>(
        &mut self,
        a_elements: &[AE],
        b_elements: &[BE],
        challenge: Self::RandomVar,
    ) where
        AE: Into<Self::ExprEF> + Copy,
        BE: Into<Self::ExprEF> + Copy;
}

impl<AB: PermutationAirBuilder> LookupBuilder for AB {
    fn assert_lookup_with_multiplicities<AE, AM, BE, BM>(
        &mut self,
        a_elements: &[AE],
        a_multiplicities: &[AM],
        b_elements: &[BE],
        b_multiplicities: &[BM],
        challenge: Self::RandomVar,
    ) where
        AE: Into<Self::ExprEF> + Copy,
        AM: Into<Self::ExprEF> + Copy,
        BE: Into<Self::ExprEF> + Copy,
        BM: Into<Self::ExprEF> + Copy,
    {
        let gadget = LogUpGadget::<Self::F>::new();
        gadget.assert_lookup_with_multiplicities(
            self,
            a_elements,
            a_multiplicities,
            b_elements,
            b_multiplicities,
            challenge,
        );
    }

    fn assert_lookup<AE, BE>(
        &mut self,
        a_elements: &[AE],
        b_elements: &[BE],
        challenge: Self::RandomVar,
    ) where
        AE: Into<Self::ExprEF> + Copy,
        BE: Into<Self::ExprEF> + Copy,
    {
        let gadget = LogUpGadget::<Self::F>::new();
        gadget.assert_lookup(self, a_elements, b_elements, challenge);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_baby_bear::BabyBear;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    /// Base field type for the test
    type F = BabyBear;
    /// Extension field type for the test
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_constraint_degree_calculation() {
        let gadget = LogUpGadget::<F>::new();

        // Test basic constraint degree calculation
        // degree = 1 (running sum) + n_a + n_b
        assert_eq!(gadget.constraint_degree(0, 0), 1); // 1 + 0 + 0
        assert_eq!(gadget.constraint_degree(2, 3), 6); // 1 + 2 + 3
        assert_eq!(gadget.constraint_degree(1, 1), 3); // 1 + 1 + 1
        assert_eq!(gadget.constraint_degree(5, 2), 8); // 1 + 5 + 2
    }

    /// A mock `AirBuilder` for testing purposes that simulates constraint evaluation.
    struct MockAirBuilder {
        /// Main trace matrix containing the execution trace data
        main: RowMajorMatrix<F>,
        /// Auxiliary trace matrix containing the LogUp running sum column
        permutation: RowMajorMatrix<EF>,
        /// Random challenges used in the LogUp argument
        challenges: Vec<EF>,
        /// Current row being evaluated during constraint checking
        current_row: usize,
        /// Total height (number of rows) in the trace
        height: usize,
    }

    impl MockAirBuilder {
        fn new(
            main: RowMajorMatrix<F>,
            permutation: RowMajorMatrix<EF>,
            challenges: Vec<EF>,
        ) -> Self {
            let height = main.height();
            Self {
                main,
                permutation,
                challenges,
                current_row: 0,
                height,
            }
        }

        // Helper to update the builder to the current row being evaluated
        fn for_row(&mut self, row: usize) {
            self.current_row = row;
        }

        // A mock windowed view for the trace matrices
        fn window<T: Clone + Send + Sync + Field>(
            &self,
            trace: &RowMajorMatrix<T>,
        ) -> RowMajorMatrix<T> {
            let mut view = Vec::new();
            // local row
            let local_row: Vec<T> = trace.row(self.current_row).unwrap().into_iter().collect();
            view.extend_from_slice(&local_row);
            // next row (if it exists)
            if self.current_row + 1 < self.height {
                let next_row: Vec<T> = trace
                    .row(self.current_row + 1)
                    .unwrap()
                    .into_iter()
                    .collect();
                view.extend_from_slice(&next_row);
            } else {
                // pad with zeros if we are on the last row
                view.extend(vec![T::ZERO; trace.width()]);
            }
            RowMajorMatrix::new(view, trace.width())
        }
    }

    impl AirBuilder for MockAirBuilder {
        type F = F;
        type Expr = F;
        type Var = F;
        type M = RowMajorMatrix<F>;

        fn main(&self) -> Self::M {
            self.window(&self.main)
        }

        fn is_first_row(&self) -> Self::Expr {
            F::from_bool(self.current_row == 0)
        }

        fn is_last_row(&self) -> Self::Expr {
            F::from_bool(self.current_row == self.height - 1)
        }

        fn is_transition_window(&self, size: usize) -> Self::Expr {
            assert!(size > 0);
            F::from_bool(self.current_row < self.height - (size - 1))
        }

        fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
            let val = x.into();
            assert_eq!(
                val,
                F::ZERO,
                "Constraint failed at row {}: {:?} != 0",
                self.current_row,
                val
            );
        }
    }

    impl ExtensionBuilder for MockAirBuilder {
        type EF = EF;
        type ExprEF = EF;
        type VarEF = EF;

        fn assert_zero_ext<I>(&mut self, x: I)
        where
            I: Into<Self::ExprEF>,
        {
            let val = x.into();
            if val != EF::ZERO {
                panic!(
                    "Extension constraint failed at row {}: {:?} != 0",
                    self.current_row, val
                );
            }
        }
    }

    impl PermutationAirBuilder for MockAirBuilder {
        type MP = RowMajorMatrix<EF>;
        type RandomVar = EF;

        fn permutation(&self) -> Self::MP {
            self.window(&self.permutation)
        }

        fn permutation_randomness(&self) -> &[Self::RandomVar] {
            &self.challenges
        }
    }

    /// An AIR designed to perform an 8-bit range check using the `LogUpGadget`.
    ///
    /// This AIR demonstrates how to use LogUp for range checking by ensuring that
    /// all values in the main trace are within a specified range (0 to 255 for 8-bit).
    struct RangeCheckAir {
        /// The random challenge `alpha` used in the LogUp fraction denominators
        challenge: EF,
    }

    impl<AB> Air<AB> for RangeCheckAir
    where
        AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>,
        AB::Var: Copy + Into<AB::ExprEF>,
        AB::ExprEF: From<AB::Var> + From<F>,
        F: Copy + Into<AB::ExprEF>,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let main_local = main.row_slice(0).unwrap();

            // The main trace is structured as:
            // col 0: values_to_check
            // col 1: range_table_values
            // col 2: multiplicities
            let val = main_local[0];
            let table_val = main_local[1];
            let mult = main_local[2];

            // For each row, we assert that the multiset {val} with multiplicity 1
            // is equivalent to the multiset {table_val} with its given multiplicity.
            //
            // The running sum `s` ensures that over the entire trace, the multisets match globally.
            LogUpGadget::<F>::new().assert_lookup_with_multiplicities(
                builder,
                &[val],
                &[F::ONE], // Each value from the main trace is "read" once per row.
                &[table_val],
                &[mult], // The multiplicity from the witness.
                self.challenge,
            );
        }
    }

    impl<F: Field> BaseAir<F> for RangeCheckAir {
        fn width(&self) -> usize {
            3 // values, table_values, multiplicities
        }
    }

    /// Computes the contribution to the LogUp running sum for a single row.
    ///
    /// The contribution is: `1/(α - val_read) - mult/(α - val_provided)`
    fn compute_logup_contribution(
        alpha: EF,
        val_read: F,
        val_provided: F,
        mult: F,
    ) -> EF {
        (alpha - EF::from(val_read)).inverse()
            - (alpha - EF::from(val_provided)).inverse() * EF::from(mult)
    }

    /// Helper to generate a complete witness for a range check test.
    fn generate_range_check_witness(
        values_to_check: &[F],
    ) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>, EF) {
        let alpha = EF::from(F::new(0x12345678));

        // Create a trace where each value to check gets its own row
        // paired with the same value from the table with multiplicity 1
        let trace_len = values_to_check.len();

        // Build main trace matrix: [value, value, 1]
        //
        // This means: read 'value' once, provide 'value' once
        let mut main_flat = Vec::with_capacity(trace_len * 3);
        for &val in values_to_check {
            // value being read
            main_flat.push(val);
            // value being provided (same as read)
            main_flat.push(val);
            // multiplicity = 1 (provide once)
            main_flat.push(F::ONE);
        }

        // Build auxiliary running sum column `s`.
        //
        // s[i] = sum of contributions from rows 0 to i (inclusive)
        // s[0] = contribution from row 0 (initial boundary constraint)
        // s[i+1] = s[i] + contribution from row i+1 (transition constraint)
        let mut s_col = Vec::with_capacity(trace_len);
        let mut current_s = EF::ZERO;

        for &val in values_to_check.iter() {
            // Contribution from current row
            let contribution = compute_logup_contribution(alpha, val, val, F::ONE);
            current_s += contribution;

            // s[i] includes the contribution from row i
            s_col.push(current_s);
        }

        // Assemble matrices
        let main_trace = RowMajorMatrix::new(main_flat, 3);
        let aux_trace = RowMajorMatrix::new(s_col, 1);

        (main_trace, aux_trace, alpha)
    }

    #[test]
    fn test_range_check_end_to_end_valid() {
        // SCENARIO: All values are valid 8-bit integers.
        let values_to_check = vec![
            F::new(10),
            F::new(255),
            F::new(0),
            F::new(42),
            F::new(10), // a duplicate
        ];

        // Generate the witness and challenges.
        let (main_trace, aux_trace, alpha) = generate_range_check_witness(&values_to_check);

        // Final value of the running sum MUST be zero for a valid lookup.
        let final_row: Vec<EF> = aux_trace
            .row(aux_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect();
        let final_s = final_row[0];
        assert_eq!(final_s, EF::ZERO);

        // Setup the AIR and builder.
        let air = RangeCheckAir { challenge: alpha };
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);

        // Evaluate constraints for every row.
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }
    }

    #[test]
    #[should_panic(expected = "Extension constraint failed")]
    fn test_range_check_end_to_end_invalid() {
        // SCENARIO: One value (256) is outside the 8-bit range [0, 255].
        // We create a trace where we read 256 but provide 255 from the table

        // Create an invalid witness: we read 256 but provide a different value (255)
        let mut main_flat = Vec::new();
        main_flat.extend([F::new(10), F::new(10), F::ONE]); // valid row
        main_flat.extend([F::new(256), F::new(255), F::ONE]); // Mismatch!
        main_flat.extend([F::new(42), F::new(42), F::ONE]); // valid row

        let main_trace = RowMajorMatrix::new(main_flat, 3);
        let alpha = EF::from(F::new(0x12345678));

        // Build auxiliary running sum column - this will be non-zero at the end
        let mut s_col = Vec::with_capacity(main_trace.height());
        let mut current_s = EF::ZERO;

        for i in 0..main_trace.height() {
            let row: Vec<F> = main_trace.row(i).unwrap().into_iter().collect();
            let val_read = row[0];
            let val_provided = row[1];
            let mult = row[2];

            let contribution = compute_logup_contribution(alpha, val_read, val_provided, mult);
            current_s += contribution;

            // s[i] includes the contribution from row i
            s_col.push(current_s);
        }

        let aux_trace = RowMajorMatrix::new(s_col, 1);

        // The final sum should be non-zero due to the mismatch
        let final_row: Vec<EF> = aux_trace
            .row(aux_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect();
        let final_s = final_row[0];
        assert_ne!(final_s, EF::ZERO);

        // Setup the AIR and builder
        let air = RangeCheckAir { challenge: alpha };
        let mut builder = MockAirBuilder::new(main_trace, aux_trace.clone(), vec![alpha]);

        // Evaluate constraints.
        //
        // This should fail on the last row due to the non-zero final sum
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }
    }

    #[test]
    fn test_compute_sum_terms_logic() {
        let gadget = LogUpGadget::<F>::new();
        let alpha = EF::from_u8(100);

        // Elements: [2, 5], Multiplicities: [3, 1]
        //
        // We want to compute: 3/(α - 2) + 1/(α - 5)
        let elements = [F::new(2), F::new(5)];
        let multiplicities = [F::new(3), F::ONE];

        // Expected Numerator: 3 * (α - 5) + 1 * (α - 2)
        let expected_numerator =
            (alpha - EF::from_u8(5)) * EF::from_u8(3) + (alpha - EF::from_u8(2));
        // Expected Denominator: (α - 2) * (α - 5)
        let expected_denominator = (alpha - EF::from_u8(2)) * (alpha - EF::from_u8(5));

        let (num, den) =
            gadget.compute_sum_terms::<MockAirBuilder, F, F>(&elements, &multiplicities, &alpha);

        assert_eq!(num, expected_numerator);
        assert_eq!(den, expected_denominator);
    }

    #[test]
    #[should_panic(expected = "Extension constraint failed at row 1")]
    fn test_inconsistent_witness_fails_transition() {
        // SCENARIO: The main trace is valid, but the prover messes up the running sum calculation.
        let values_to_check = vec![F::new(10), F::new(20), F::new(30)];
        let (main_trace, mut aux_trace, alpha) = generate_range_check_witness(&values_to_check);

        // The witness is valid so far. Let's corrupt it.
        // The transition from row 1 to 2 will be s_2 = s_1 + C_1.
        //
        // Let's set s_2 to a garbage value to make the transition fail.
        let corrupted_s_val = EF::from_u8(99);
        aux_trace.values[2] = corrupted_s_val;

        // Evaluate constraints.
        //
        // This should now fail at row 1 when checking the transition to row 2.
        let air = RangeCheckAir { challenge: alpha };
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }
    }

    #[test]
    #[should_panic(expected = "Extension constraint failed at row 0")]
    fn test_zero_multiplicity_is_not_counted() {
        // SCENARIO: We read `10`, but the lookup table provides it with multiplicity 0.
        // We add a second, valid row to ensure the transition and boundary constraints are tested.
        let mut main_flat = Vec::new();
        // Read 10, provide 10 with mult 0
        main_flat.extend([F::new(10), F::new(10), F::ZERO]);
        // A valid row
        main_flat.extend([F::new(20), F::new(20), F::ONE]);

        let main_trace = RowMajorMatrix::new(main_flat, 3);
        let alpha = EF::from_u8(123);

        // Build witness. The contribution from row 0 will be non-zero.
        let mut s_col = Vec::with_capacity(main_trace.height());
        let mut current_s = EF::ZERO;
        for i in 0..main_trace.height() {
            let row: Vec<F> = main_trace.row(i).unwrap().into_iter().collect();
            let contribution = compute_logup_contribution(alpha, row[0], row[1], row[2]);
            current_s += contribution;

            // s[i] includes the contribution from row i
            s_col.push(current_s);
        }
        let aux_trace = RowMajorMatrix::new(s_col, 1);

        // The final value in the `s` column (s_1) should be non-zero.
        let final_s = aux_trace
            .row(main_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect::<Vec<EF>>()[0];
        assert_ne!(final_s, EF::ZERO);

        // Evaluate constraints
        let air = RangeCheckAir { challenge: alpha };
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);
        // The initial boundary constraint will fail on row 0 since s[0] is incorrect.
        //
        // It will panic on the first row (row 0).
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }
    }

    #[test]
    fn test_empty_lookup_is_valid() {
        // SCENARIO: A lookup is triggered, but both the read and provided sets are empty.
        let main_trace = RowMajorMatrix::new(vec![], 3);
        let aux_trace = RowMajorMatrix::new(vec![], 1);
        let alpha = EF::from_u8(123);

        let air = RangeCheckAir { challenge: alpha };
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);

        // This should not panic, as there are no rows to evaluate.
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }

        // Also test the internal logic directly
        let gadget = LogUpGadget::<F>::new();
        let (num, den) = gadget.compute_sum_terms::<MockAirBuilder, F, F>(&[], &[], &alpha);
        assert_eq!(num, EF::ZERO);
        assert_eq!(den, EF::ONE);
    }
}
