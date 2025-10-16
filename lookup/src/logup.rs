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
use core::marker::PhantomData;

use p3_air::{ExtensionBuilder, PermutationAirBuilder};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

/// A context structure that encapsulates a single lookup relationship and its associated
/// permutation column.
#[derive(Debug, Clone)]
pub struct LookupContext<'a, AE, AM, BE, BM> {
    /// Elements being read (consumed from the table)
    a_elements: &'a [AE],
    /// Multiplicities for elements being read
    a_multiplicities: &'a [AM],
    /// Elements being provided (added to the table)
    b_elements: &'a [BE],
    /// Multiplicities for elements being provided
    b_multiplicities: &'a [BM],
    /// The column index in the permutation trace for this lookup's running sum
    column: usize,
}

impl<'a, AE, AM, BE, BM> LookupContext<'a, AE, AM, BE, BM> {
    /// Creates a new lookup context with the specified column.
    ///
    /// # Arguments
    /// * `a_elements` - Elements from the main execution trace
    /// * `a_multiplicities` - How many times each `a_element` should appear
    /// * `b_elements` - Elements from the lookup table
    /// * `b_multiplicities` - How many times each `b_element` should appear
    /// * `column` - The column index in the permutation trace for this lookup
    pub const fn new(
        a_elements: &'a [AE],
        a_multiplicities: &'a [AM],
        b_elements: &'a [BE],
        b_multiplicities: &'a [BM],
        column: usize,
    ) -> Self {
        Self {
            a_elements,
            a_multiplicities,
            b_elements,
            b_multiplicities,
            column,
        }
    }
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
/// Constraints are defined as:
/// - **Initial Constraint**: `s[0] = 0`
/// - **Transition Constraint**: `s[i+1] = s[i] + contribution[i]`
/// - **Final Constraint**: `s[n-1] + contribution[n-1] = 0`
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

    /// Internal implementation using `LookupContext`.
    ///
    /// This method enforces that elements from `a_elements` with their associated
    /// `a_multiplicities` form the same multiset as elements from `b_elements`
    /// with their `b_multiplicities`.
    ///
    /// # Arguments
    /// * `builder` - AIR builder for constraint generation
    /// * `context` - The lookup context containing all lookup data and column assignment
    /// * `challenge` - Random challenge `α` for the LogUp argument
    ///
    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(a_multiplicities[i] / (α - a_elements[i])) = ∑_j(b_multiplicities[j] / (α - b_elements[j]))
    /// ```
    ///
    /// This is implemented using a running sum column that should sum to zero.
    fn assert_lookup_internal<AB, AE, AM, BE, BM>(
        &self,
        builder: &mut AB,
        context: &LookupContext<AE, AM, BE, BM>,
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Copy,
        AM: Into<AB::ExprEF> + Clone,
        BE: Into<AB::ExprEF> + Copy,
        BM: Into<AB::ExprEF> + Clone,
    {
        // Ensure |A| and |m_A| match. This is required to form ∑ m_A/(α - a).
        assert_eq!(
            context.a_elements.len(),
            context.a_multiplicities.len(),
            "Mismatched lengths: a_elements and a_multiplicities must have same length"
        );
        // Ensure |B| and |m_B| match. This is required to form ∑ m_B/(α - b).
        assert_eq!(
            context.b_elements.len(),
            context.b_multiplicities.len(),
            "Mismatched lengths: b_elements and b_multiplicities must have same length"
        );

        // Access the permutation (aux) table. It carries the running sum column s.
        let permutation = builder.permutation();
        // Read s[i] from the local row at the specified column.
        let s_local = permutation.row_slice(0).unwrap()[context.column].into();
        // Read s[i+1] from the next row (or a zero-padded view on the last row).
        let s_next = permutation.row_slice(1).unwrap()[context.column].into();

        // Anchor s[0] = 0 (not ∑ m_A/(α−a) − ∑ m_B/(α−b)).
        //
        // Avoids a high-degree boundary constraint.
        // Telescoping is enforced by the last-row check (s[n−1] = 0).
        // Simpler, and keeps aux and main traces aligned in length.
        builder.when_first_row().assert_zero_ext(s_local.clone());

        // Convert the random challenge to an expression: α.
        let alpha: AB::ExprEF = challenge.into();

        // Build A's fraction:  ∑ m_A/(α - a)  =  a_num / a_den .
        let (a_num, a_den) = self.compute_sum_terms::<AB, AE, AM>(
            context.a_elements,
            context.a_multiplicities,
            &alpha,
        );
        // Build B's fraction:  ∑ m_B/(α - b)  =  b_num / b_den .
        let (b_num, b_den) = self.compute_sum_terms::<AB, BE, BM>(
            context.b_elements,
            context.b_multiplicities,
            &alpha,
        );
        // Common denominator: D = a_den ⋅ b_den. This clears all divisions.
        let common_denominator = a_den.clone() * b_den.clone();
        // Numerator difference: N = a_num⋅b_den − b_num⋅a_den.
        //
        // The equality of sums holds iff N = 0 after clearing denominators.
        let contribution_poly = a_num * b_den - b_num * a_den;

        // Transition constraint on rows 0..n-2:
        // (s[i+1] − s[i])⋅D − N = 0.
        builder.when_transition().assert_zero_ext(
            (s_next - s_local.clone()) * common_denominator.clone() - contribution_poly.clone(),
        );

        // Final constraint on the last row:
        // s[n−1]⋅D + N = 0  ⇔  s[n−1] = −N/D  ⇔ total sum cancels to zero.
        builder
            .when_last_row()
            .assert_zero_ext(s_local * common_denominator + contribution_poly);
    }

    /// Asserts lookup equality.
    ///
    /// # Arguments
    /// * `builder` - AIR builder for constraint generation
    /// * `context` - The lookup context containing all lookup data and column assignment
    /// * `challenge` - Random challenge `α` for the LogUp argument
    pub fn assert_lookup<AB, AE, AM, BE, BM>(
        builder: &mut AB,
        context: &LookupContext<AE, AM, BE, BM>,
        challenge: AB::RandomVar,
    ) where
        AB: PermutationAirBuilder<F = F>,
        AE: Into<AB::ExprEF> + Copy,
        AM: Into<AB::ExprEF> + Clone,
        BE: Into<AB::ExprEF> + Copy,
        BM: Into<AB::ExprEF> + Clone,
    {
        let gadget = Self::new();
        gadget.assert_lookup_internal(builder, context, challenge);
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

        let n = elements.len();

        // Precompute all (α - e_i) terms
        let terms: Vec<AB::ExprEF> = elements.iter().map(|&e| alpha.clone() - e.into()).collect();

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
        let mut numerator = AB::ExprEF::ZERO;
        for i in 0..n {
            let mult_expr: AB::ExprEF = multiplicities[i].clone().into();
            let product_without_i = pref[i].clone() * suff[i + 1].clone();
            numerator += mult_expr * product_without_i;
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
    /// Assert lookup.
    ///
    /// # Arguments
    /// * `context` - The lookup context containing all lookup data and column assignment
    /// * `challenge` - Random challenge for the lookup argument
    fn assert_lookup<AE, AM, BE, BM>(
        &mut self,
        context: &LookupContext<AE, AM, BE, BM>,
        challenge: Self::RandomVar,
    ) where
        AE: Into<Self::ExprEF> + Copy,
        AM: Into<Self::ExprEF> + Clone,
        BE: Into<Self::ExprEF> + Copy,
        BM: Into<Self::ExprEF> + Clone;
}

impl<AB: PermutationAirBuilder> LookupBuilder for AB {
    fn assert_lookup<AE, AM, BE, BM>(
        &mut self,
        context: &LookupContext<AE, AM, BE, BM>,
        challenge: Self::RandomVar,
    ) where
        AE: Into<Self::ExprEF> + Copy,
        AM: Into<Self::ExprEF> + Clone,
        BE: Into<Self::ExprEF> + Copy,
        BM: Into<Self::ExprEF> + Clone,
    {
        LogUpGadget::<Self::F>::assert_lookup(self, context, challenge);
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

    /// An AIR designed to perform range checks using the `LogUpGadget`.
    ///
    /// This AIR demonstrates how to use LogUp for range checking. It supports multiple
    /// independent lookups by specifying how many lookups to perform.
    ///
    /// For `num_lookups = 1`: Main trace has 3 columns [read, provide, mult]
    /// For `num_lookups = 2`: Main trace has 6 columns [read1, provide1, mult1, read2, provide2, mult2]
    struct RangeCheckAir {
        /// The random challenge `alpha` used in the LogUp fraction denominators
        challenge: EF,
        /// Number of independent lookups (default: 1)
        num_lookups: usize,
    }

    impl RangeCheckAir {
        fn new(challenge: EF) -> Self {
            Self {
                challenge,
                num_lookups: 1,
            }
        }

        fn with_multiple_lookups(challenge: EF, num_lookups: usize) -> Self {
            Self {
                challenge,
                num_lookups,
            }
        }
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

            // Perform each lookup independently using LookupContext
            for lookup_idx in 0..self.num_lookups {
                let offset = lookup_idx * 3;

                // Extract columns for this lookup: [read, provide, mult]
                let val = main_local[offset];
                let table_val = main_local[offset + 1];
                let mult = main_local[offset + 2];

                // Create arrays with longer lifetime for the context
                let a_elements = [val];
                let a_multiplicities = [F::ONE];
                let b_elements = [table_val];
                let b_multiplicities = [mult];

                // Create a context for this lookup with its dedicated column
                let context = LookupContext::new(
                    &a_elements,
                    &a_multiplicities,
                    &b_elements,
                    &b_multiplicities,
                    lookup_idx, // Each lookup gets its own column
                );

                // Assert the lookup using the context
                LogUpGadget::<F>::assert_lookup(builder, &context, self.challenge);
            }
        }
    }

    impl<F: Field> BaseAir<F> for RangeCheckAir {
        fn width(&self) -> usize {
            3 * self.num_lookups // [read, provide, mult] per lookup
        }
    }

    /// Computes the contribution to the LogUp running sum for a single row.
    ///
    /// The contribution is: `1/(α - val_read) - mult/(α - val_provided)`
    fn compute_logup_contribution(alpha: EF, val_read: F, val_provided: F, mult: F) -> EF {
        (alpha - EF::from(val_read)).inverse()
            - (alpha - EF::from(val_provided)).inverse() * EF::from(mult)
    }

    /// A simple builder for constructing LogUp lookup traces with arbitrary read/provide patterns.
    ///
    /// This makes it easy to create complex test scenarios with non-trivial permutations
    /// and varied multiplicities.
    ///
    /// # Example
    /// ```ignore
    /// let trace = LookupTraceBuilder::new()
    ///     .row(3, 1, 1)  // Read 3, Provide 1 with multiplicity 1
    ///     .row(1, 2, 2)  // Read 1, Provide 2 with multiplicity 2
    ///     .row(2, 3, 1)  // Read 2, Provide 3 with multiplicity 1
    ///     .build();
    /// ```
    struct LookupTraceBuilder {
        /// (read_value, provide_value, multiplicity)
        rows: Vec<(F, F, F)>,
        /// Random challenge for the lookup argument
        alpha: EF,
    }

    impl LookupTraceBuilder {
        /// Create a new trace builder with a random challenge.
        fn new() -> Self {
            Self {
                rows: Vec::new(),
                alpha: EF::from_u32(0x12345678),
            }
        }

        /// Add a row to the trace.
        ///
        /// # Arguments
        /// * `read` - The value being read (always with multiplicity 1)
        /// * `provide` - The value being provided to the lookup table
        /// * `mult` - The multiplicity of the provided value
        fn row(mut self, read: u32, provide: u32, mult: u32) -> Self {
            self.rows
                .push((F::new(read), F::new(provide), F::new(mult)));
            self
        }

        /// Build the main and auxiliary traces.
        ///
        /// Returns `(main_trace, aux_trace, alpha)` where:
        /// - `main_trace`: 3 columns [read_val, provide_val, multiplicity]
        /// - `aux_trace`: 1 column [running_sum]
        /// - `alpha`: The challenge used
        fn build(self) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>, EF) {
            assert!(!self.rows.is_empty(), "Must have at least one row");

            // Build main trace: flatten (read, provide, mult) tuples into a single vector
            let main_flat: Vec<F> = self
                .rows
                .iter()
                .flat_map(|&(read, provide, mult)| [read, provide, mult])
                .collect();
            let main_trace = RowMajorMatrix::new(main_flat, 3);

            // Build auxiliary trace: running sum column
            // s[0] = 0, s[i+1] = s[i] + contribution_from_row_i
            let mut running_sum = EF::ZERO;
            let s_col: Vec<EF> = core::iter::once(EF::ZERO)
                .chain(self.rows.iter().map(|&(read, provide, mult)| {
                    running_sum += compute_logup_contribution(self.alpha, read, provide, mult);
                    running_sum
                }))
                .take(self.rows.len()) // Keep trace length equal to number of rows
                .collect();

            let aux_trace = RowMajorMatrix::new(s_col, 1);

            (main_trace, aux_trace, self.alpha)
        }
    }

    #[test]
    fn test_range_check_end_to_end_valid() {
        // SCENARIO: Simple range check where each value reads and provides itself.
        //
        // Values to check: [10, 255, 0, 42, 10]
        // Each row contributes 1/(α-val) - 1/(α-val) = 0, so final sum is 0.
        let (main_trace, aux_trace, alpha) = LookupTraceBuilder::new()
            .row(10, 10, 1)
            .row(255, 255, 1)
            .row(0, 0, 1)
            .row(42, 42, 1)
            .row(10, 10, 1)
            .build();

        // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0
        let s_final = aux_trace
            .row(aux_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect::<Vec<EF>>()[0];
        let last_row_data = main_trace
            .row(main_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect::<Vec<F>>();
        let last_contribution =
            compute_logup_contribution(alpha, last_row_data[0], last_row_data[1], last_row_data[2]);

        assert_eq!(
            s_final + last_contribution,
            EF::ZERO,
            "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
        );

        // Setup the AIR and builder.
        let air = RangeCheckAir::new(alpha);
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
        let air = RangeCheckAir::new(alpha);
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
        let (main_trace, mut aux_trace, alpha) = LookupTraceBuilder::new()
            .row(10, 10, 1)
            .row(20, 20, 1)
            .row(30, 30, 1)
            .build();

        // The witness is valid so far. Let's corrupt it.
        // The transition from row 1 to 2 will be s_2 = s_1 + C_1.
        //
        // Let's set s_2 to a garbage value to make the transition fail.
        let corrupted_s_val = EF::from_u8(99);
        aux_trace.values[2] = corrupted_s_val;

        // Evaluate constraints.
        //
        // This should now fail at row 1 when checking the transition to row 2.
        let air = RangeCheckAir::new(alpha);
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
        let air = RangeCheckAir::new(alpha);
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

        let air = RangeCheckAir::new(alpha);
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

    #[test]
    fn test_nontrivial_permutation() {
        // SCENARIO: Complex permutation with varied multiplicities
        //
        // Multiset equality check:
        // - Reads: {3: 2, 5: 4, 7: 2}  (8 values total)
        // - Provides: {3: 2, 5: 4, 7: 2}  ✓
        //
        // ┌─────┬──────┬─────────┬──────┬────────────────────────────────────┐
        // │ Row │ Read │ Provide │ Mult │ Contribution                       │
        // ├─────┼──────┼─────────┼──────┼────────────────────────────────────┤
        // │  0  │  7   │    3    │  2   │  1/(α-7) - 2/(α-3)                │
        // │  1  │  3   │    5    │  4   │  1/(α-3) - 4/(α-5)                │
        // │  2  │  5   │    7    │  2   │  1/(α-5) - 2/(α-7)                │
        // │  3  │  3   │    3    │  0   │  1/(α-3) - 0 = 1/(α-3)            │
        // │  4  │  7   │    5    │  0   │  1/(α-7) - 0 = 1/(α-7)            │
        // │  5  │  5   │    5    │  0   │  1/(α-5) - 0 = 1/(α-5)            │
        // │  6  │  5   │    7    │  0   │  1/(α-5) - 0 = 1/(α-5)            │
        // │  7  │  5   │    --   │  0   │  1/(α-5) - 0 = 1/(α-5)            │
        // └─────┴──────┴─────────┴──────┴────────────────────────────────────┘
        //
        // Total contributions:
        // Reads:    [1/(α-7) + 1/(α-3) + 1/(α-5) + 1/(α-3) + 1/(α-7) + 1/(α-5) + 1/(α-5) + 1/(α-5)]
        //         = 2/(α-7) + 2/(α-3) + 4/(α-5)
        // Provides: [2/(α-3) + 4/(α-5) + 2/(α-7) + 0 + 0 + 0 + 0 + 0]
        //         = 2/(α-3) + 4/(α-5) + 2/(α-7)
        // Difference: 0

        let (main_trace, aux_trace, alpha) = LookupTraceBuilder::new()
            .row(7, 3, 2) // Read 7, provide {3, 3} to the table
            .row(3, 5, 4) // Read 3, provide {5, 5, 5, 5} to the table (4 fives total)
            .row(5, 7, 2) // Read 5, provide {7, 7} to the table
            .row(3, 3, 0) // Read 3, no provides (mult=0)
            .row(7, 5, 0) // Read 7, no provides (mult=0)
            .row(5, 5, 0) // Read 5, no provides (mult=0)
            .row(5, 7, 0) // Read 5, no provides (mult=0)
            .row(5, 5, 0) // Read 5, no provides (mult=0)
            .build();

        // The test must check the FINAL constraint: s[n-1] + c[n-1] = 0
        let s_final = aux_trace
            .row(aux_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect::<Vec<EF>>()[0];
        let last_row_data = main_trace
            .row(main_trace.height() - 1)
            .unwrap()
            .into_iter()
            .collect::<Vec<F>>();
        let last_contribution =
            compute_logup_contribution(alpha, last_row_data[0], last_row_data[1], last_row_data[2]);

        assert_eq!(
            s_final + last_contribution,
            EF::ZERO,
            "Total sum (s[n-1] + c[n-1]) must be zero for a valid lookup"
        );

        // Setup AIR and verify all constraints
        let air = RangeCheckAir::new(alpha);
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);

        // Evaluate constraints for every row
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }
    }

    #[test]
    fn test_multiple_lookups_different_columns() {
        // SCENARIO: Two independent lookups in the same AIR using different columns
        //
        // Lookup 1 (column 0): Simple range check [10, 20, 30]
        // Lookup 2 (column 1): Different values [5, 15, 25]

        let alpha = EF::from_u32(0x12345678);

        // Build main trace with 6 columns
        // Format: [read1, provide1, mult1, read2, provide2, mult2]
        let main_flat = vec![
            // Row 0
            F::new(10),
            F::new(10),
            F::ONE, // Lookup 1: read 10, provide 10
            F::new(5),
            F::new(5),
            F::ONE, // Lookup 2: read 5, provide 5
            // Row 1
            F::new(20),
            F::new(20),
            F::ONE, // Lookup 1: read 20, provide 20
            F::new(15),
            F::new(15),
            F::ONE, // Lookup 2: read 15, provide 15
            // Row 2
            F::new(30),
            F::new(30),
            F::ONE, // Lookup 1: read 30, provide 30
            F::new(25),
            F::new(25),
            F::ONE, // Lookup 2: read 25, provide 25
        ];
        let main_trace = RowMajorMatrix::new(main_flat, 6);

        // Build auxiliary trace with 2 columns (one per lookup)
        let mut s1_col = Vec::with_capacity(3); // Running sum for lookup 1
        let mut s2_col = Vec::with_capacity(3); // Running sum for lookup 2

        let mut s1 = EF::ZERO;
        let mut s2 = EF::ZERO;

        // Row 0: s[0] = 0 for both lookups
        s1_col.push(s1);
        s2_col.push(s2);

        // Row 1: Add contributions from row 0
        s1 += compute_logup_contribution(alpha, F::new(10), F::new(10), F::ONE);
        s2 += compute_logup_contribution(alpha, F::new(5), F::new(5), F::ONE);
        s1_col.push(s1);
        s2_col.push(s2);

        // Row 2: Add contributions from row 1
        s1 += compute_logup_contribution(alpha, F::new(20), F::new(20), F::ONE);
        s2 += compute_logup_contribution(alpha, F::new(15), F::new(15), F::ONE);
        s1_col.push(s1);
        s2_col.push(s2);

        // Interleave the two columns into a single flat vector
        let mut aux_flat = Vec::with_capacity(6);
        for i in 0..3 {
            aux_flat.push(s1_col[i]);
            aux_flat.push(s2_col[i]);
        }
        let aux_trace = RowMajorMatrix::new(aux_flat, 2); // 2 columns

        // Verify both final sums are zero (each lookup is valid)
        let row2_data: Vec<EF> = aux_trace.row(2).unwrap().into_iter().collect();
        let s1_final = row2_data[0];
        let s2_final = row2_data[1];

        let c1_final = compute_logup_contribution(alpha, F::new(30), F::new(30), F::ONE);
        let c2_final = compute_logup_contribution(alpha, F::new(25), F::new(25), F::ONE);

        assert_eq!(
            s1_final + c1_final,
            EF::ZERO,
            "Lookup 1 final sum must be zero"
        );
        assert_eq!(
            s2_final + c2_final,
            EF::ZERO,
            "Lookup 2 final sum must be zero"
        );

        // Setup AIR with 2 lookups and verify all constraints
        let air = RangeCheckAir::with_multiple_lookups(alpha, 2);
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);

        // Evaluate constraints for every row - both lookups should pass
        for i in 0..builder.height {
            builder.for_row(i);
            air.eval(&mut builder);
        }
    }
}
