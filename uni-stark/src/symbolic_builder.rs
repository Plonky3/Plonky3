use alloc::vec;
use alloc::vec::Vec;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::Entry;
use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;

#[instrument(skip_all)]
pub fn get_log_num_quotient_chunks<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_log_quotient_degree_extension(air, preprocessed_width, num_public_values, 0, 0, is_zk)
}

#[instrument(name = "infer log of base and extension constraint degree", skip_all)]
pub fn get_log_quotient_degree_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree_extension::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_permutation_challenges,
    ) + is_zk)
        .max(2);

    // We bound the degree of the quotient polynomial by constraint_degree - 1,
    // then choose the number of quotient chunks as the smallest power of two
    // >= (constraint_degree - 1). This function returns log2(#chunks).
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(air, preprocessed_width, num_public_values, 0, 0)
}

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
    num_public_values: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder =
        SymbolicAirBuilder::new(preprocessed_width, air.width(), num_public_values, 0, 0);
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
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
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
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
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
    permutation: RowMajorMatrix<SymbolicVariable<EF>>,
    permutation_challenges: Vec<SymbolicVariable<EF>>,
    extension_constraints: Vec<SymbolicExpression<EF>>,
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
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            extension_constraints: vec![],
        }
    }

    pub fn extension_constraints(&self) -> Vec<SymbolicExpression<EF>> {
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

    fn main(&self) -> Self::M {
        self.main.clone()
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
            panic!("uni-stark only supports a window size of 2")
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

impl<F: Field, EF: ExtensionField<F>> PairBuilder for SymbolicAirBuilder<F, EF> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
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

#[cfg(test)]
mod tests {
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;

    use super::*;

    #[derive(Debug)]
    struct MockAir {
        constraints: Vec<SymbolicVariable<BabyBear>>,
        width: usize,
    }

    impl BaseAir<BabyBear> for MockAir {
        fn width(&self) -> usize {
            self.width
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
    fn test_get_log_num_quotient_chunks_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, 3, 2, 0);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_num_quotient_chunks_single_constraint() {
        let air = MockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, 3, 2, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_log_num_quotient_chunks_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(Entry::Main { offset: 0 }, 0),
                SymbolicVariable::new(Entry::Main { offset: 1 }, 1),
                SymbolicVariable::new(Entry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, 3, 2, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_max_constraint_degree_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let max_degree = get_max_constraint_degree(&air, 3, 2);
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
        };
        let max_degree = get_max_constraint_degree(&air, 3, 2);
        assert_eq!(max_degree, 1, "Max constraint degree should be 1");
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1 = SymbolicVariable::new(Entry::Main { offset: 0 }, 0);
        let c2 = SymbolicVariable::new(Entry::Main { offset: 1 }, 1);

        let air = MockAir {
            constraints: vec![c1, c2],
            width: 4,
        };

        let constraints = get_symbolic_constraints(&air, 3, 2);

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
        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0);

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
        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0);

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
        let mut builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3, 0, 0);
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
}
