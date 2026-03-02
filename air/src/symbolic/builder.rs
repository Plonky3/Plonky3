use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::symbolic::SymbolicExpr;
use crate::symbolic::expression::BaseLeaf;
use crate::symbolic::expression_ext::SymbolicExpressionExt;
use crate::symbolic::variable::{BaseEntry, ExtEntry, SymbolicVariableExt};
use crate::{
    Air, AirBuilder, ExtensionBuilder, PermutationAirBuilder, SymbolicExpression, SymbolicVariable,
};

#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(air: &A, preprocessed_width: usize) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(air, preprocessed_width, 0, 0)
}

#[instrument(
    name = "infer base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_max_constraint_degree_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) = get_all_symbolic_constraints(
        air,
        preprocessed_width,
        permutation_width,
        num_permutation_challenges,
    );

    let base_degree = base_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);

    let extension_degree = extension_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);
    base_degree.max(extension_degree)
}

#[instrument(
    name = "evaluate base constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        air.num_public_values(),
        0,
        0,
    );
    air.eval(&mut builder);
    builder.base_constraints()
}

#[instrument(
    name = "evaluate extension constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
) -> Vec<SymbolicExpressionExt<F, EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        air.num_public_values(),
        permutation_width,
        num_permutation_challenges,
    );
    air.eval(&mut builder);
    builder.extension_constraints()
}

#[instrument(
    name = "evaluate all constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_all_symbolic_constraints<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<SymbolicExpressionExt<F, EF>>,
)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        air.num_public_values(),
        permutation_width,
        num_permutation_challenges,
    );
    air.eval(&mut builder);
    (builder.base_constraints(), builder.extension_constraints())
}

/// An [`AirBuilder`] for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F> = F> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    permutation: RowMajorMatrix<SymbolicVariableExt<F, EF>>,
    permutation_challenges: Vec<SymbolicVariableExt<F, EF>>,
    extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    pub fn new(
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_permutation_challenges: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width).map(move |index| {
                    SymbolicVariable::new(BaseEntry::Preprocessed { offset }, index)
                })
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width)
                    .map(move |index| SymbolicVariable::new(BaseEntry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(BaseEntry::Public, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width).map(move |index| {
                    SymbolicVariableExt::new(ExtEntry::Permutation { offset }, index)
                })
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| SymbolicVariableExt::new(ExtEntry::Challenge, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            extension_constraints: vec![],
        }
    }

    pub fn extension_constraints(&self) -> Vec<SymbolicExpressionExt<F, EF>> {
        self.extension_constraints.clone()
    }

    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.base_constraints.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<F>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn preprocessed(&self) -> Option<Self::M> {
        if self.preprocessed.values.is_empty() {
            None
        } else {
            Some(self.preprocessed.clone())
        }
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpr::Leaf(BaseLeaf::IsFirstRow)
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpr::Leaf(BaseLeaf::IsLastRow)
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpr::Leaf(BaseLeaf::IsTransition)
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type EF = EF;
    type ExprEF = SymbolicExpressionExt<F, EF>;
    type VarEF = SymbolicVariableExt<F, EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = SymbolicVariableExt<F, EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::BaseAir;

    type F = BabyBear;
    type EF = p3_field::extension::BinomialExtensionField<F, 4>;

    #[derive(Debug)]
    struct MockAir {
        constraints: Vec<SymbolicVariable<F>>,
        width: usize,
    }

    impl BaseAir<F> for MockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl Air<SymbolicAirBuilder<F>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F>) {
            for constraint in &self.constraints {
                builder.assert_zero(*constraint);
            }
        }
    }

    #[test]
    fn test_get_max_constraint_degree_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let max_degree = get_max_constraint_degree(&air, 3);
        assert_eq!(
            max_degree, 0,
            "No constraints should result in a degree of 0"
        );
    }

    #[test]
    fn test_get_max_constraint_degree_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0),
                SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1),
                SymbolicVariable::new(BaseEntry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let max_degree = get_max_constraint_degree(&air, 3);
        assert_eq!(max_degree, 1, "Max constraint degree should be 1");
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1 = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0);
        let c2 = SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1);

        let air = MockAir {
            constraints: vec![c1, c2],
            width: 4,
        };

        let constraints = get_symbolic_constraints(&air, 3);

        assert_eq!(constraints.len(), 2, "Should return exactly 2 constraints");

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Leaf(BaseLeaf::Variable(v)) if v.index == c1.index && v.entry == c1.entry)),
            "Expected constraint {c1:?} was not found"
        );

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Leaf(BaseLeaf::Variable(v)) if v.index == c2.index && v.entry == c2.entry)),
            "Expected constraint {c2:?} was not found"
        );
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        let builder = SymbolicAirBuilder::<F>::new(2, 4, 3, 0, 0);

        let expected_main = [
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 1),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 2),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 3),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 0),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 1),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 2),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 3),
        ];

        let builder_main = builder.main.values;

        assert_eq!(
            builder_main.len(),
            expected_main.len(),
            "Main matrix should have the expected length"
        );

        for (expected, actual) in expected_main.iter().zip(builder_main.iter()) {
            assert_eq!(expected.index, actual.index, "Index mismatch");
            assert_eq!(expected.entry, actual.entry, "Entry mismatch");
        }
    }

    #[test]
    fn test_symbolic_air_builder_is_first_last_row() {
        let builder = SymbolicAirBuilder::<F>::new(2, 4, 3, 0, 0);

        assert!(
            matches!(
                builder.is_first_row(),
                SymbolicExpression::Leaf(BaseLeaf::IsFirstRow)
            ),
            "First row condition did not match"
        );

        assert!(
            matches!(
                builder.is_last_row(),
                SymbolicExpression::Leaf(BaseLeaf::IsLastRow)
            ),
            "Last row condition did not match"
        );
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<F>::new(2, 4, 3, 0, 0);
        let expr = SymbolicExpression::Leaf(BaseLeaf::Constant(F::new(5)));
        builder.assert_zero(expr);

        let constraints = builder.base_constraints();
        assert_eq!(constraints.len(), 1, "One constraint should be recorded");

        assert!(
            constraints.iter().any(
                |x| matches!(x, SymbolicExpression::Leaf(BaseLeaf::Constant(val)) if *val == F::new(5))
            ),
            "Constraint should match the asserted one"
        );
    }

    #[test]
    fn test_is_transition_window_size_2() {
        // Window size 2 returns the transition selector.
        let builder = SymbolicAirBuilder::<F>::new(0, 2, 0, 0, 0);
        let expr = builder.is_transition_window(2);
        assert!(matches!(
            expr,
            SymbolicExpression::Leaf(BaseLeaf::IsTransition)
        ));
    }

    #[test]
    #[should_panic(expected = "uni-stark only supports a window size of 2")]
    fn test_is_transition_window_size_3_panics() {
        // Window size 3 is not supported and should panic.
        let builder = SymbolicAirBuilder::<F>::new(0, 2, 0, 0, 0);
        let _ = builder.is_transition_window(3);
    }

    #[test]
    fn test_main_returns_correct_dimensions() {
        // The main matrix has 2 rows (one per offset) and the given width.
        let builder = SymbolicAirBuilder::<F>::new(0, 3, 0, 0, 0);
        let main = builder.main();

        // 2 rows times 3 columns gives 6 entries.
        assert_eq!(main.values.len(), 6);

        // First row has offset 0.
        assert_eq!(main.values[0].entry, BaseEntry::Main { offset: 0 });
        assert_eq!(main.values[0].index, 0);
        assert_eq!(main.values[2].index, 2);

        // Second row has offset 1.
        assert_eq!(main.values[3].entry, BaseEntry::Main { offset: 1 });
        assert_eq!(main.values[3].index, 0);
    }

    #[test]
    fn test_preprocessed_returns_correct_dimensions() {
        // The preprocessed matrix has 2 rows and the given preprocessed width.
        let builder = SymbolicAirBuilder::<F>::new(2, 3, 0, 0, 0);
        let prep = builder.preprocessed().expect("should be Some");

        // 2 rows times 2 columns gives 4 entries.
        assert_eq!(prep.values.len(), 4);
        assert_eq!(prep.values[0].entry, BaseEntry::Preprocessed { offset: 0 });
        assert_eq!(prep.values[0].index, 0);
        assert_eq!(prep.values[1].index, 1);
        assert_eq!(prep.values[2].entry, BaseEntry::Preprocessed { offset: 1 });
    }

    #[test]
    fn test_preprocessed_returns_none_when_width_is_zero() {
        // A builder with zero preprocessed columns should report no preprocessed trace.
        let builder = SymbolicAirBuilder::<F>::new(0, 3, 0, 0, 0);
        assert!(builder.preprocessed().is_none());
    }

    #[test]
    fn test_public_values_correct_count_and_entries() {
        // All public value variables have the public entry kind.
        let builder = SymbolicAirBuilder::<F>::new(0, 2, 5, 0, 0);
        let pv = builder.public_values();
        assert_eq!(pv.len(), 5);
        for (i, var) in pv.iter().enumerate() {
            assert_eq!(var.entry, BaseEntry::Public);
            assert_eq!(var.index, i);
        }
    }

    #[test]
    fn test_assert_zero_ext_records_constraint() {
        // Asserting an extension constraint records it in the builder.
        let mut builder = SymbolicAirBuilder::<F, EF>::new(0, 2, 0, 2, 1);
        let expr = SymbolicExpressionExt::<F, EF>::from(F::new(7));
        builder.assert_zero_ext(expr);
        let ext_constraints = builder.extension_constraints();
        assert_eq!(ext_constraints.len(), 1);
    }

    #[test]
    fn test_extension_constraints_initially_empty() {
        // A fresh builder starts with no extension constraints.
        let builder = SymbolicAirBuilder::<F, EF>::new(0, 2, 0, 0, 0);
        assert!(builder.extension_constraints().is_empty());
    }

    #[test]
    fn test_permutation_returns_correct_dimensions() {
        // The permutation matrix has 2 rows and the given permutation width.
        let builder = SymbolicAirBuilder::<F, EF>::new(0, 2, 0, 3, 0);
        let perm = builder.permutation();

        // 2 rows times 3 columns gives 6 entries.
        assert_eq!(perm.values.len(), 6);
        assert_eq!(perm.values[0].entry, ExtEntry::Permutation { offset: 0 });
        assert_eq!(perm.values[0].index, 0);
        assert_eq!(perm.values[3].entry, ExtEntry::Permutation { offset: 1 });
    }

    #[test]
    fn test_permutation_randomness_correct_count() {
        // All challenge variables have the challenge entry kind.
        let builder = SymbolicAirBuilder::<F, EF>::new(0, 2, 0, 2, 4);
        let challenges = builder.permutation_randomness();
        assert_eq!(challenges.len(), 4);
        for (i, var) in challenges.iter().enumerate() {
            assert_eq!(var.entry, ExtEntry::Challenge);
            assert_eq!(var.index, i);
        }
    }

    #[derive(Debug)]
    struct ExtMockAir {
        width: usize,
    }

    impl BaseAir<F> for ExtMockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl Air<SymbolicAirBuilder<F, EF>> for ExtMockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F, EF>) {
            // Record one base constraint from the main trace.
            let main = builder.main();
            builder.assert_zero(main.values[0]);

            // Record one extension constraint from the permutation trace.
            let perm = builder.permutation();
            builder.assert_zero_ext(perm.values[0]);
        }
    }

    #[test]
    fn test_get_symbolic_constraints_extension() {
        // Only the extension constraint is returned.
        let air = ExtMockAir { width: 2 };
        let ext_constraints = get_symbolic_constraints_extension::<F, EF, _>(&air, 0, 3, 1);
        assert_eq!(ext_constraints.len(), 1);
    }

    #[test]
    fn test_get_all_symbolic_constraints() {
        // Both the base and extension constraint are returned.
        let air = ExtMockAir { width: 2 };
        let (base, ext) = get_all_symbolic_constraints::<F, EF, _>(&air, 0, 3, 1);
        assert_eq!(base.len(), 1);
        assert_eq!(ext.len(), 1);
    }

    #[test]
    fn test_get_max_constraint_degree_extension() {
        // The max degree covers both base and extension constraints.
        let air = ExtMockAir { width: 2 };
        let max_deg = get_max_constraint_degree_extension::<F, EF, _>(&air, 0, 3, 1);

        // Both constraints are single variables with degree 1.
        assert_eq!(max_deg, 1);
    }
}
