//! Symbolic AIR builder for constraint analysis.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::{
    Air, AirBuilder, AirBuilderWithPublicValues, Entry, ExtensionBuilder, PeriodicAirBuilder,
    PermutationAirBuilder, SymbolicExpression, SymbolicVariable,
};

/// Compute the maximum constraint degree of base field constraints.
#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    num_periodic_columns: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(
        air,
        preprocessed_width,
        num_public_values,
        0,
        0,
        num_periodic_columns,
    )
}

/// Compute the maximum constraint degree across both base and extension field constraints.
#[instrument(
    name = "infer base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_max_constraint_degree_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    num_periodic_columns: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) = get_all_symbolic_constraints(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_permutation_challenges,
        num_periodic_columns,
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

/// Evaluate the AIR symbolically and return the base field constraint expressions.
#[instrument(
    name = "evaluate base constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    num_periodic_columns: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        0,
        0,
        num_periodic_columns,
    );
    air.eval(&mut builder);
    builder.base_constraints()
}

/// Evaluate the AIR symbolically and return the extension field constraint expressions.
#[instrument(
    name = "evaluate extension constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    num_periodic_columns: usize,
) -> Vec<SymbolicExpression<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        permutation_width,
        num_permutation_challenges,
        num_periodic_columns,
    );
    air.eval(&mut builder);
    builder.extension_constraints()
}

/// Evaluate the AIR symbolically and return both base and extension field constraint expressions.
#[instrument(
    name = "evaluate all constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_all_symbolic_constraints<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    num_periodic_columns: usize,
) -> (Vec<SymbolicExpression<F>>, Vec<SymbolicExpression<EF>>)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        permutation_width,
        num_permutation_challenges,
        num_periodic_columns,
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
    periodic: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    permutation: RowMajorMatrix<SymbolicVariable<EF>>,
    permutation_challenges: Vec<SymbolicVariable<EF>>,
    extension_constraints: Vec<SymbolicExpression<EF>>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    /// Create a new `SymbolicAirBuilder` with the given dimensions.
    pub fn new(
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_permutation_challenges: usize,
        num_periodic_columns: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        let periodic = (0..num_periodic_columns)
            .map(|index| SymbolicVariable::new(Entry::Periodic, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width)
                    .map(move |index| SymbolicVariable::new(Entry::Permutation { offset }, index))
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| SymbolicVariable::new(Entry::Challenge, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            periodic,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            extension_constraints: vec![],
        }
    }

    /// Return the collected extension field constraints.
    pub fn extension_constraints(&self) -> Vec<SymbolicExpression<EF>> {
        self.extension_constraints.clone()
    }

    /// Return the collected base field constraints.
    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.base_constraints.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn preprocessed(&self) -> Option<Self::M> {
        Some(self.preprocessed.clone())
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("SymbolicAirBuilder only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues for SymbolicAirBuilder<F, EF> {
    type PublicVar = SymbolicVariable<F>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
{
    type EF = EF;
    type ExprEF = SymbolicExpression<EF>;
    type VarEF = SymbolicVariable<EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = SymbolicVariable<EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }
}

impl<F: Field, EF: ExtensionField<F>> PeriodicAirBuilder for SymbolicAirBuilder<F, EF> {
    type PeriodicVar = SymbolicVariable<F>;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_matrix::Matrix;

    use super::*;
    use crate::{AirWithPeriodicColumns, BaseAir, PeriodicAirBuilder};

    #[derive(Debug)]
    struct MockAir {
        constraints: Vec<SymbolicVariable<BabyBear>>,
        width: usize,
        periodic_columns: Vec<Vec<BabyBear>>,
    }

    impl BaseAir<BabyBear> for MockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl AirWithPeriodicColumns<BabyBear> for MockAir {
        fn periodic_columns(&self) -> &[Vec<BabyBear>] {
            &self.periodic_columns
        }
    }

    impl Air<SymbolicAirBuilder<BabyBear>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear>) {
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
            periodic_columns: vec![],
        };
        let max_degree = get_max_constraint_degree(&air, 3, 2, 0);
        assert_eq!(
            max_degree, 0,
            "No constraints should result in a degree of 0"
        );
    }

    #[test]
    fn test_get_max_constraint_degree_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(Entry::Main { offset: 0 }, 0),
                SymbolicVariable::new(Entry::Main { offset: 1 }, 1),
                SymbolicVariable::new(Entry::Main { offset: 2 }, 2),
            ],
            width: 4,
            periodic_columns: vec![],
        };
        let max_degree = get_max_constraint_degree(&air, 3, 2, 0);
        assert_eq!(max_degree, 1, "Max constraint degree should be 1");
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1 = SymbolicVariable::new(Entry::Main { offset: 0 }, 0);
        let c2 = SymbolicVariable::new(Entry::Main { offset: 1 }, 1);

        let air = MockAir {
            constraints: vec![c1, c2],
            width: 4,
            periodic_columns: vec![],
        };

        let constraints = get_symbolic_constraints(&air, 3, 2, 0);

        assert_eq!(constraints.len(), 2, "Should return exactly 2 constraints");

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Variable(v) if v.index == c1.index && v.entry == c1.entry)),
            "Expected constraint {c1:?} was not found"
        );

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Variable(v) if v.index == c2.index && v.entry == c2.entry)),
            "Expected constraint {c2:?} was not found"
        );
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0, 0);

        let expected_main = [
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 0 }, 0),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 0 }, 1),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 0 }, 2),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 0 }, 3),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 1 }, 0),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 1 }, 1),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 1 }, 2),
            SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 1 }, 3),
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
        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0, 0);

        assert!(
            matches!(builder.is_first_row(), SymbolicExpression::IsFirstRow),
            "First row condition did not match"
        );

        assert!(
            matches!(builder.is_last_row(), SymbolicExpression::IsLastRow),
            "Last row condition did not match"
        );
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0, 0);
        let expr = SymbolicExpression::Constant(BabyBear::new(5));
        builder.assert_zero(expr);

        let constraints = builder.base_constraints();
        assert_eq!(constraints.len(), 1, "One constraint should be recorded");

        assert!(
            constraints.iter().any(
                |x| matches!(x, SymbolicExpression::Constant(val) if *val == BabyBear::new(5))
            ),
            "Constraint should match the asserted one"
        );
    }

    #[test]
    fn test_symbolic_air_builder_with_periodic() {
        let num_periodic = 3;
        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0, num_periodic);

        let periodic_values = builder.periodic_values();
        assert_eq!(
            periodic_values.len(),
            num_periodic,
            "Should have {num_periodic} periodic columns"
        );

        // Check that periodic variables have correct Entry type and indices
        for (i, var) in periodic_values.iter().enumerate() {
            assert_eq!(var.entry, Entry::Periodic, "Should be a Periodic entry");
            assert_eq!(var.index, i, "Index should match position");
        }
    }

    #[test]
    fn test_periodic_columns_in_constraints() {
        // Create periodic columns for the MockAir
        let periodic_col = vec![BabyBear::new(1), BabyBear::new(2)];

        let air = MockAir {
            constraints: vec![],
            width: 4,
            periodic_columns: vec![periodic_col.clone()],
        };

        // Verify that the air returns correct periodic columns
        let columns = air.periodic_columns();
        assert_eq!(columns.len(), 1, "Should have 1 periodic column");
        assert_eq!(columns[0], periodic_col, "Periodic column should match");
    }

    #[test]
    fn test_periodic_columns_matrix() {
        // Create multiple periodic columns of same length
        let periodic_col1 = vec![BabyBear::new(1), BabyBear::new(2), BabyBear::new(3)];
        let periodic_col2 = vec![BabyBear::new(4), BabyBear::new(5), BabyBear::new(6)];

        let air = MockAir {
            constraints: vec![],
            width: 4,
            periodic_columns: vec![periodic_col1, periodic_col2],
        };

        let matrix = air.periodic_columns_matrix().expect("Should have a matrix");
        assert_eq!(matrix.height(), 3, "Matrix should have 3 rows");
        assert_eq!(matrix.width(), 2, "Matrix should have 2 columns");

        // Check values - row-major order: [row0_col0, row0_col1, row1_col0, row1_col1, ...]
        let expected = [
            BabyBear::new(1),
            BabyBear::new(4),
            BabyBear::new(2),
            BabyBear::new(5),
            BabyBear::new(3),
            BabyBear::new(6),
        ];
        assert_eq!(matrix.values, expected);
    }
}
