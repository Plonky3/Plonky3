use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_matrix::Matrix;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;
use crate::Entry;

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, A>(air: &A, num_public_values: usize) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = get_max_constraint_degree(air, num_public_values).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(air: &A, num_public_values: usize) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_symbolic_constraints(air, num_public_values)
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    num_public_values: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let prep_width = air.preprocessed_trace().map(|t| t.width()).unwrap_or(0);
    let main_width = air.width();
    // TODO: replace zeros with actual permutation width after adding support in `uni-stark`.
    let perm_width = 0;
    let mut builder =
        SymbolicAirBuilder::new(prep_width, main_width, perm_width, num_public_values);
    air.eval(&mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    permutation: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub(crate) fn new(
        preprocessed_width: usize,
        main_width: usize,
        permutation_width: usize,
        num_public_values: usize,
    ) -> Self {
        let new_matrix = |width: usize, entry: Entry| {
            let values = [false, true]
                .into_iter()
                .flat_map(|is_next| {
                    (0..main_width).map(move |column| SymbolicVariable {
                        entry,
                        is_next,
                        column,
                        _phantom: PhantomData,
                    })
                })
                .collect();
            RowMajorMatrix::new(values, width)
        };
        let public_values = (0..num_public_values)
            .map(|i| SymbolicVariable {
                entry: Entry::PublicValue,
                is_next: false,
                column: i,
                _phantom: PhantomData,
            })
            .collect();
        Self {
            preprocessed: new_matrix(preprocessed_width, Entry::Preprocessed),
            main: new_matrix(main_width, Entry::Main),
            permutation: new_matrix(permutation_width, Entry::Permutation),
            // TODO replace zeros once we have SymbolicExpression::PublicValue
            public_values: public_values,
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
        self.public_values.as_slice()
    }
}
