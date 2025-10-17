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

use p3_air::{AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::lookup_traits::{Kind, Lookup, LookupGadget, symbolic_to_expr};

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
    fn compute_combined_sum_terms<AB, E, M>(
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
        let mut numerator = AB::ExprEF::ZERO;
        for i in 0..n {
            let mult_expr: AB::ExprEF = multiplicities[i].clone().into();
            let product_without_i = pref[i].clone() * suff[i + 1].clone();
            numerator += mult_expr * product_without_i;
        }

        (numerator, common_denominator)
    }

    fn eval_update<AB>(
        &self,
        builder: &mut AB,
        context: Lookup<AB::F>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        let Lookup {
            kind: _,
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

        // First, turn the symbolic expressions into builder expressions.
        let elements = element_exprs
            .iter()
            .map(|exprs| {
                exprs
                    .iter()
                    .map(|expr| symbolic_to_expr(builder, expr.clone().into()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let multiplicities = multiplicities_exprs
            .iter()
            .map(|expr| symbolic_to_expr(builder, expr.clone().into()))
            .collect::<Vec<_>>();

        // Access the permutation (aux) table. It carries the running sum column s.
        let permutation = builder.permutation();

        let permutation_challenges = builder.permutation_randomness();

        assert!(
            permutation_challenges.len() >= self.num_challenges() * column,
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

        // Anchor s[0] = 0 (not ∑ m_A/(α−a) − ∑ m_B/(α−b)).
        //
        // Avoids a high-degree boundary constraint.
        // Telescoping is enforced by the last-row check (s[n−1] = 0).
        // Simpler, and keeps aux and main traces aligned in length.
        builder.when_first_row().assert_zero_ext(s_local.clone());

        // Build A's fraction:  ∑ m_A/(α - a)  =  a_num / a_den .
        let (numerator, common_denominator): (AB::ExprEF, AB::ExprEF) = self
            .compute_combined_sum_terms::<AB, AB::ExprEF, AB::ExprEF>(
                &elements,
                &multiplicities,
                &alpha.into(),
                &beta.into(),
            );

        builder.when_transition().assert_zero_ext(
            (s_next - s_local.clone()) * common_denominator.clone() - numerator.clone(),
        );

        let final_val = s_local * common_denominator + numerator;
        builder
            .when_last_row()
            .assert_zero_ext(final_val - expected_cumulated);
    }
}

impl<F: Field> LookupGadget<F> for LogUpGadget<F> {
    fn num_aux_cols(&self) -> usize {
        1
    }

    fn num_challenges(&self) -> usize {
        2
    }

    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(a_multiplicities[i] / (α - a_elements[i])) = ∑_j(b_multiplicities[j] / (α - b_elements[j]))
    /// ```
    ///
    /// This is implemented using a running sum column that should sum to zero.
    fn eval_local_lookup<AB>(&self, builder: &mut AB, context: Lookup<AB::F>)
    where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        match context.kind {
            Kind::Global(_) => panic!("Global lookups are not supported in local evaluation"),
            _ => {}
        }

        self.eval_update(builder, context, AB::ExprEF::ZERO);
    }

    /// # Mathematical Details
    /// The constraint enforces:
    /// ```text
    /// ∑_i(a_multiplicities[i] / (α - a_elements[i])) = ∑_j(b_multiplicities[j] / (α - b_elements[j]))
    /// ```
    ///
    /// This is implemented using a running sum column that should sum to `expected_cumulated`.
    fn eval_global_update<AB: PermutationAirBuilder>(
        &self,
        builder: &mut AB,
        context: Lookup<AB::F>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        self.eval_update(builder, context, expected_cumulated);
    }

    fn eval_global_final_value<AB: PermutationAirBuilder>(
        &self,
        builder: &mut AB,
        all_expected_cumulative: &[AB::ExprEF],
    ) {
        let total = all_expected_cumulative
            .iter()
            .fold(AB::ExprEF::ZERO, |acc, x| acc + x.clone());

        builder.assert_zero_ext(total);
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
    /// The constraint degree if then:
    /// `1 + max(deg(numerator), deg(common_denominator))`
    fn constraint_degree(&self, context: Lookup<F>) -> usize {
        assert!(context.multiplicities_exprs.len() == context.element_exprs.len());

        let n = context.multiplicities_exprs.len();

        // Compute degrees in a single pass.
        let mut degs = Vec::with_capacity(n);
        let mut deg_sum = 0;
        for elems in context.element_exprs.iter() {
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

        core::cmp::max(deg_denom_constr, deg_num)
    }
}

impl<F: Field> LogUpGadget<F> {
    /// Creates a new LogUp gadget instance.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_baby_bear::BabyBear;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};

    use super::*;
    use crate::lookup_traits::{AirLookupHandler, Direction};

    /// Base field type for the test
    type F = BabyBear;
    /// Extension field type for the test
    type EF = BinomialExtensionField<F, 4>;

    fn create_symbolic_with_degree(degree: usize) -> SymbolicExpression<F> {
        let x = Rc::new(SymbolicExpression::Constant(F::ONE));
        let y = Rc::new(SymbolicExpression::Constant(F::TWO));
        SymbolicExpression::Mul {
            x,
            y,
            degree_multiple: degree,
        }
    }

    fn create_dummy_lookup(
        num_elements_per_tuple: Vec<usize>,
        degree_per_element: Vec<Vec<usize>>,
        degree_multiplicities: Vec<usize>,
    ) -> Lookup<F> {
        assert!(num_elements_per_tuple.len() == degree_per_element.len());
        assert!(num_elements_per_tuple.len() == degree_multiplicities.len());

        let element_exprs = num_elements_per_tuple
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                assert_eq!(num_elements_per_tuple[i], degree_per_element[i].len());
                (0..n)
                    .map(|j| create_symbolic_with_degree(degree_per_element[i][j]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let multiplicities_exprs = degree_multiplicities
            .iter()
            .map(|&deg| create_symbolic_with_degree(deg))
            .collect::<Vec<_>>();

        Lookup {
            kind: Kind::Local,
            element_exprs,
            multiplicities_exprs,
            columns: vec![0],
        }
    }

    #[test]
    fn test_constraint_degree_calculation() {
        let gadget = LogUpGadget::<F>::new();

        // Test basic constraint degree calculation
        // We have two lookup elements (each element is a single column):
        // - each element has degree 1
        // - each multiplicity has degree 1
        // - so the total degree should be 3 (1 + (1 + 1)).
        let lookup_deg_3 = create_dummy_lookup(vec![1, 1], vec![vec![1], vec![1]], vec![1, 1]);
        assert_eq!(gadget.constraint_degree(lookup_deg_3), 3);

        // We have two lookup elements (each element is a single column):
        // - each element has degree 1
        // - each multiplicity has degree 3
        // - so the total degree should be 4 (3 + 1).
        let lookup_degree_4 = create_dummy_lookup(vec![1, 1], vec![vec![1], vec![1]], vec![3, 3]);
        assert_eq!(gadget.constraint_degree(lookup_degree_4), 4);

        // We have two lookup elements (each element is a single column):
        // - one element has degree 2
        // - one element has degree 3
        // - each multiplicity has degree 1
        // - so the total degree should be 6 (3 + 3).
        let lookup_degree_6 = create_dummy_lookup(vec![1, 1], vec![vec![2], vec![3]], vec![1, 1]);
        assert_eq!(gadget.constraint_degree(lookup_degree_6), 6);

        // We have two lookup elements:
        // - first element is a tuple of 5 columns
        // - second element is a tuple of 6 columns
        // - first element has degree 2
        // - second element has degree 3
        // - first multiplicity has degree 5
        // - second multiplicity has degree 2
        // - so the total degree should be 7 (5 + 2).
        let degrees1 = vec![1, 3, 0, 0, 0]; // First element of degree 3.
        let degrees2 = vec![0, 1, 0, 2, 0, 1]; // Second element of degree 2.
        let lookup_degree_7 = create_dummy_lookup(vec![5, 6], vec![degrees1, degrees2], vec![5, 2]);
        assert_eq!(gadget.constraint_degree(lookup_degree_7), 7);
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

    impl PairBuilder for MockAirBuilder {
        fn preprocessed(&self) -> Self::M {
            RowMajorMatrix::new(vec![], 0)
        }
    }

    impl AirBuilderWithPublicValues for MockAirBuilder {
        type PublicVar = Self::F;

        fn public_values(&self) -> &[Self::PublicVar] {
            &[]
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
        /// Number of independent lookups (default: 1)
        num_lookups: usize,
        /// Number of registered lookups (for AirLookupHandler)
        cur_num_lookups: usize,
    }

    impl RangeCheckAir {
        fn new() -> Self {
            Self {
                num_lookups: 1,
                cur_num_lookups: 0,
            }
        }

        fn with_multiple_lookups(num_lookups: usize) -> Self {
            Self {
                num_lookups,
                cur_num_lookups: 0,
            }
        }
    }

    impl<AB> Air<AB> for RangeCheckAir
    where
        AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>
            + PairBuilder
            + AirBuilderWithPublicValues,
        AB::Var: Copy + Into<AB::ExprEF>,
        AB::ExprEF: From<AB::Var> + From<F>,
        F: Copy + Into<AB::ExprEF>,
    {
        fn eval(&self, _builder: &mut AB) {
            // There are no constraints, only lookups for the range checks.
        }
    }

    impl<F: Field> BaseAir<F> for RangeCheckAir {
        fn width(&self) -> usize {
            3 * self.num_lookups // [read, provide, mult] per lookup
        }
    }

    impl<AB> AirLookupHandler<AB> for RangeCheckAir
    where
        AB: PermutationAirBuilder<F = F, EF = EF, RandomVar = EF>
            + PairBuilder
            + AirBuilderWithPublicValues,
        AB::Var: Copy + Into<AB::ExprEF>,
        AB::ExprEF: From<AB::Var> + From<F>,
        F: Copy + Into<AB::ExprEF>,
    {
        fn add_lookup_columns(&mut self) -> Vec<usize> {
            let new_idx = self.cur_num_lookups;
            self.cur_num_lookups += 1;

            vec![new_idx]
        }

        fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
            let symbolic_air_builder =
                SymbolicAirBuilder::<F>::new(0, <Self as BaseAir<AB::F>>::width(self), 0);

            let symbolic_main = symbolic_air_builder.main();
            let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

            // Perform each lookup independently using LookupContext
            (0..self.num_lookups)
                .map(|lookup_idx| {
                    let offset = lookup_idx * 3;

                    // Extract columns for this lookup: [read, provide, mult]
                    let val = symbolic_main_local[offset];
                    let table_val = symbolic_main_local[offset + 1];
                    let mult = symbolic_main_local[offset + 2];

                    // Create arrays with longer lifetime for the context
                    let a_elements = vec![val.into()];
                    let a_multiplicities = SymbolicExpression::Constant(F::ONE);

                    let b_elements = vec![table_val.into()];
                    let b_multiplicities = mult.into();

                    let lookup_inputs = vec![
                        (a_elements, a_multiplicities, Direction::Send),
                        (b_elements, b_multiplicities, Direction::Receive),
                    ];

                    // Register the local lookup.
                    <Self as AirLookupHandler<AB>>::register_lookup(
                        self,
                        Kind::Local,
                        &lookup_inputs,
                    )
                })
                .collect::<Vec<_>>()
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
        /// Random challenge for combining tuple elements (not used in single-column tests)
        beta: EF,
    }

    impl LookupTraceBuilder {
        /// Create a new trace builder with a random challenge.
        fn new() -> Self {
            Self {
                rows: Vec::new(),
                alpha: EF::from_u32(0x12345678),
                beta: EF::from_u32(0x87654321),
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
        fn build(self) -> (RowMajorMatrix<F>, RowMajorMatrix<EF>, Vec<EF>) {
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

            (main_trace, aux_trace, vec![self.alpha, self.beta])
        }
    }

    #[test]
    fn test_range_check_end_to_end_valid() {
        // SCENARIO: Simple range check where each value reads and provides itself.
        //
        // Values to check: [10, 255, 0, 42, 10]
        // Each row contributes 1/(α-val) - 1/(α-val) = 0, so final sum is 0.
        let (main_trace, aux_trace, challenges) = LookupTraceBuilder::new()
            .row(10, 10, 1)
            .row(255, 255, 1)
            .row(0, 0, 1)
            .row(42, 42, 1)
            .row(10, 10, 1)
            .build();

        let alpha = challenges[0];
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
        let mut air = RangeCheckAir::new();
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges);

        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // Check that the lookup was created correctly.
        assert_eq!(lookups.len(), 1, "Should have one lookup defined");
        assert_eq!(
            lookups[0].columns,
            vec![0],
            "Lookup should use the first auxiliary column"
        );
        assert_eq!(
            lookups[0].element_exprs.len(),
            2,
            "Lookup should have two element tuples (read and provide)"
        );
        assert_eq!(lookups[0].kind, Kind::Local, "Lookup should be local");
        assert_eq!(air.cur_num_lookups, 1, "Should have one lookup registered");

        // Evaluate constraints for every row.
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
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
        let beta = EF::from(F::new(0x87654321));

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
        let mut air = RangeCheckAir::new();
        let mut builder = MockAirBuilder::new(main_trace, aux_trace.clone(), vec![alpha, beta]);

        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // Evaluate constraints.
        //
        // This should fail on the second row due to the non-zero final sum
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
        }
    }

    #[test]
    fn test_compute_sum_terms_logic() {
        let gadget = LogUpGadget::<F>::new();
        let alpha = EF::from_u8(100);
        let beta = EF::from_u8(100);

        // Elements: [2, 5], Multiplicities: [3, 1]
        //
        // We want to compute: 3/(α - 2) + 1/(α - 5)
        let elements = [vec![F::new(2)], vec![F::new(5)]];
        let multiplicities = [F::new(3), F::ONE];

        // Expected Numerator: 3 * (α - 5) + 1 * (α - 2)
        let expected_numerator =
            (alpha - EF::from_u8(5)) * EF::from_u8(3) + (alpha - EF::from_u8(2));
        // Expected Denominator: (α - 2) * (α - 5)
        let expected_denominator = (alpha - EF::from_u8(2)) * (alpha - EF::from_u8(5));

        let (num, den) = gadget.compute_combined_sum_terms::<MockAirBuilder, F, F>(
            &elements,
            &multiplicities,
            &alpha,
            &beta,
        );

        assert_eq!(num, expected_numerator);
        assert_eq!(den, expected_denominator);
    }

    #[test]
    #[should_panic(expected = "Extension constraint failed at row 1")]
    fn test_inconsistent_witness_fails_transition() {
        // SCENARIO: The main trace is valid, but the prover messes up the running sum calculation.
        let (main_trace, mut aux_trace, challenges) = LookupTraceBuilder::new()
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
        let mut air = RangeCheckAir::new();
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges);

        // Register the lookups.
        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // Evaluate the constraints.
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
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
        let mut air = RangeCheckAir::new();
        let zero = EF::ZERO;
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha, zero]);

        // Register the lookups.
        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // The initial boundary constraint will fail on row 0 since s[0] is incorrect.
        //
        // It will panic on the first row (row 0).
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
        }
    }

    #[test]
    fn test_empty_lookup_is_valid() {
        // SCENARIO: A lookup is triggered, but both the read and provided sets are empty.
        let main_trace = RowMajorMatrix::new(vec![], 3);
        let aux_trace = RowMajorMatrix::new(vec![], 1);
        let alpha = EF::from_u8(123);

        let mut air = RangeCheckAir::new();
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, vec![alpha]);

        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // This should not panic, as there are no rows to evaluate.
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
        }

        // Also test the internal logic directly
        let gadget = LogUpGadget::<F>::new();
        let (num, den) =
            gadget.compute_combined_sum_terms::<MockAirBuilder, F, F>(&[], &[], &alpha, &alpha);
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

        let (main_trace, aux_trace, challenges) = LookupTraceBuilder::new()
            .row(7, 3, 2) // Read 7, provide {3, 3} to the table
            .row(3, 5, 4) // Read 3, provide {5, 5, 5, 5} to the table (4 fives total)
            .row(5, 7, 2) // Read 5, provide {7, 7} to the table
            .row(3, 3, 0) // Read 3, no provides (mult=0)
            .row(7, 5, 0) // Read 7, no provides (mult=0)
            .row(5, 5, 0) // Read 5, no provides (mult=0)
            .row(5, 7, 0) // Read 5, no provides (mult=0)
            .row(5, 5, 0) // Read 5, no provides (mult=0)
            .build();

        let alpha = challenges[0];
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
        let mut air = RangeCheckAir::new();
        let mut builder = MockAirBuilder::new(main_trace, aux_trace, challenges);

        // Register the lookups.
        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // Evaluate constraints for every row
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
        }
    }

    #[test]
    fn test_multiple_lookups_different_columns() {
        // SCENARIO: Two independent lookups in the same AIR using different columns
        //
        // Lookup 1 (column 0): Simple range check [10, 20, 30]
        // Lookup 2 (column 1): Different values [5, 15, 25]

        let alpha = EF::from_u32(0x12345678);
        let beta = EF::from_u32(0x87654321);
        let gamma = EF::from_u32(0xABCDEF01);
        let delta = EF::from_u32(0x10FEDCBA);

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
        let mut air = RangeCheckAir::with_multiple_lookups(2);
        let mut builder =
            MockAirBuilder::new(main_trace, aux_trace, vec![alpha, beta, gamma, delta]);

        // Register lookups.
        let lookup_gadget = LogUpGadget::<F>::new();
        let lookups = <RangeCheckAir as AirLookupHandler<MockAirBuilder>>::get_lookups(&mut air);

        // Check that the lookup was created correctly.
        assert_eq!(lookups.len(), 2, "Should have two lookups defined");
        assert_eq!(
            (lookups[0].columns.clone(), lookups[1].columns.clone()),
            (vec![0], vec![1]),
            "Lookup should use the first two auxiliary column"
        );
        assert_eq!(
            lookups[0].element_exprs.len(),
            2,
            "Lookup should have two element tuples (read and provide)"
        );
        assert_eq!(lookups[0].kind, Kind::Local, "Lookup should be local");
        assert_eq!(air.cur_num_lookups, 2, "Should have two lookups registered");

        // Evaluate constraints for every row - both lookups should pass
        for i in 0..builder.height {
            builder.for_row(i);
            lookups.iter().for_each(|lookup| {
                lookup_gadget.eval_local_lookup(&mut builder, lookup.clone());
            });
        }
    }

    // TODO: Add tests for:
    // - tuple lookups (multiple columns per element)
    // - global lookups (need a more complex AIR setup)
}
