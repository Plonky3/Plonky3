use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_matrix::Matrix;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;
use crate::{Entry, SymbolicExpressionExt};

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, EF, A>(air: &A, num_public_values: usize) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = get_max_constraint_degree(air, num_public_values).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the zerofier.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, EF, A>(air: &A, num_public_values: usize) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) =
        get_symbolic_constraints(air, num_public_values);
    base_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .chain(extension_constraints.iter().map(|c| c.degree_multiple()))
        .max()
        .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, EF, A>(
    air: &A,
    num_public_values: usize,
) -> (Vec<SymbolicExpression<F>>, Vec<SymbolicExpressionExt<EF>>)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
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
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F>> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    permutation: RowMajorMatrix<SymbolicVariable<EF>>,
    public_values: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    extension_constraints: Vec<SymbolicExpressionExt<EF>>,
}

fn new_matrix<F: Field>(width: usize, entry: Entry) -> RowMajorMatrix<SymbolicVariable<F>> {
    let values = [false, true]
        .into_iter()
        .flat_map(|is_next| {
            (0..width).map(move |column| SymbolicVariable {
                entry,
                is_next,
                column,
                _phantom: PhantomData,
            })
        })
        .collect();
    RowMajorMatrix::new(values, width)
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    pub(crate) fn new(
        preprocessed_width: usize,
        main_width: usize,
        permutation_width: usize,
        num_public_values: usize,
    ) -> Self {
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
            base_constraints: vec![],
            extension_constraints: vec![],
        }
    }

    pub(crate) fn constraints(
        self,
    ) -> (Vec<SymbolicExpression<F>>, Vec<SymbolicExpressionExt<EF>>) {
        (self.base_constraints, self.extension_constraints)
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
        self.public_values.as_slice()
    }
}

impl<F: Field, EF: ExtensionField<F>> PairBuilder for SymbolicAirBuilder<F, EF> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF> {
    type EF = EF;
    type ExprEF = SymbolicExpressionExt<EF>;
    type VarEF = SymbolicVariable<EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF> {
    type PermutationVar = SymbolicVariable<EF>;

    fn permutation(&self) -> Self::M {
        self.permutation.clone()
    }
}
