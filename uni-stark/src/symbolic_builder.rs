use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, PairBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;
use crate::Entry;

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree =
        get_max_constraint_degree(air, preprocessed_width, num_public_values).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_symbolic_constraints(air, preprocessed_width, num_public_values)
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(preprocessed_width, air.width(), num_public_values);
    air.eval(&mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub(crate) fn new(preprocessed_width: usize, width: usize, num_public_values: usize) -> Self {
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
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            constraints: vec![],
        }
    }

    pub(crate) fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }
}

impl<F: Field> AirBuilder for SymbolicAirBuilder<F> {
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

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }
}

impl<F: Field> AirBuilderWithPublicValues for SymbolicAirBuilder<F> {
    type PublicVar = SymbolicVariable<F>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> PairBuilder for SymbolicAirBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

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
    fn test_get_log_quotient_degree_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let log_degree = get_log_quotient_degree(&air, 3, 2);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_quotient_degree_single_constraint() {
        let air = MockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_quotient_degree(&air, 3, 2);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_log_quotient_degree_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(Entry::Main { offset: 0 }, 0),
                SymbolicVariable::new(Entry::Main { offset: 1 }, 1),
                SymbolicVariable::new(Entry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let log_degree = get_log_quotient_degree(&air, 3, 2);
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

        assert!(constraints.contains(&SymbolicExpression::Variable(c1)),);
        assert!(constraints.contains(&SymbolicExpression::Variable(c2)),);
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        use p3_matrix::dense::RowMajorMatrix;

        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3);

        let expected_preprocessed = vec![
            SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, 0),
            SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, 1),
            SymbolicVariable::new(Entry::Preprocessed { offset: 1 }, 0),
            SymbolicVariable::new(Entry::Preprocessed { offset: 1 }, 1),
        ];

        let expected_main = vec![
            SymbolicVariable::new(Entry::Main { offset: 0 }, 0),
            SymbolicVariable::new(Entry::Main { offset: 0 }, 1),
            SymbolicVariable::new(Entry::Main { offset: 0 }, 2),
            SymbolicVariable::new(Entry::Main { offset: 0 }, 3),
            SymbolicVariable::new(Entry::Main { offset: 1 }, 0),
            SymbolicVariable::new(Entry::Main { offset: 1 }, 1),
            SymbolicVariable::new(Entry::Main { offset: 1 }, 2),
            SymbolicVariable::new(Entry::Main { offset: 1 }, 3),
        ];

        let expected_public = vec![
            SymbolicVariable::new(Entry::Public, 0),
            SymbolicVariable::new(Entry::Public, 1),
            SymbolicVariable::new(Entry::Public, 2),
        ];

        assert_eq!(
            builder.main,
            RowMajorMatrix::new(expected_main, 4),
            "Main matrix should match expected values"
        );

        assert_eq!(
            builder.preprocessed,
            RowMajorMatrix::new(expected_preprocessed, 2),
            "Preprocessed matrix should match expected values"
        );

        assert_eq!(
            builder.public_values, expected_public,
            "Public values should match expected values"
        );
    }

    #[test]
    fn test_symbolic_air_builder_is_first_last_row() {
        let builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3);
        assert_eq!(builder.is_first_row(), SymbolicExpression::IsFirstRow);
        assert_eq!(builder.is_last_row(), SymbolicExpression::IsLastRow);
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<BabyBear>::new(2, 4, 3);
        let expr = SymbolicExpression::Constant(BabyBear::new(5));
        builder.assert_zero(expr.clone());
        let constraints = builder.constraints();
        assert_eq!(constraints.len(), 1, "One constraint should be recorded");
        assert_eq!(
            constraints[0], expr,
            "Constraint should match the asserted one"
        );
    }
}
