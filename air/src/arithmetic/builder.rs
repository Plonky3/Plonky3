use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

use super::operands::*;
use crate::{
    AirBuilder, AirLayout, BaseEntry, BaseLeaf, ConstraintLayout, ExtEntry, ExtensionBuilder,
    PermutationAirBuilder,
};
/// Symbolic AIR builder that records constraints.
#[derive(Debug)]
pub struct ArithmeticAirBuilder<F: Field, EF: ExtensionField<F> = F> {
    preprocessed: RowMajorMatrix<ArithmeticVariable<F>>,
    main: RowMajorMatrix<ArithmeticVariable<F>>,
    public_values: Vec<ArithmeticVariable<F>>,
    periodic: Vec<ArithmeticVariable<F>>,
    base_constraints: Vec<ArithmeticExpression<F>>,
    permutation: RowMajorMatrix<ArithmeticVariableExt<F, EF>>,
    permutation_challenges: Vec<ArithmeticVariableExt<F, EF>>,
    permutation_values: Vec<ArithmeticVariableExt<F, EF>>,
    extension_constraints: Vec<ArithmeticExpressionExt<F, EF>>,
    constraint_types: Vec<ConstraintType>,
}

impl<F: Field, EF: ExtensionField<F>> ArithmeticAirBuilder<F, EF> {
    pub fn new(layout: AirLayout) -> Self {
        let AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width,
            num_permutation_challenges,
            num_permutation_values,
            num_periodic_columns,
        } = layout;
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width).map(move |index| {
                    ArithmeticVariable::new(BaseEntry::Preprocessed { offset }, index)
                })
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..main_width)
                    .map(move |index| ArithmeticVariable::new(BaseEntry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| ArithmeticVariable::new(BaseEntry::Public, index))
            .collect();
        let periodic = (0..num_periodic_columns)
            .map(|index| ArithmeticVariable::new(BaseEntry::Periodic, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width).map(move |index| {
                    ArithmeticVariableExt::new(ExtEntry::Permutation { offset }, index)
                })
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| ArithmeticVariableExt::new(ExtEntry::Challenge, index))
            .collect();
        let permutation_values = (0..num_permutation_values)
            .map(|index| ArithmeticVariableExt::new(ExtEntry::PermutationValue, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            public_values,
            periodic,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            permutation_values,
            extension_constraints: vec![],
            constraint_types: vec![],
        }
    }

    /// Return the constraint layout mapping global indices to base/ext streams.
    pub fn constraint_layout(&self) -> ConstraintLayout {
        let mut base_indices = Vec::new();
        let mut ext_indices = Vec::new();
        for (idx, kind) in self.constraint_types.iter().enumerate() {
            match kind {
                ConstraintType::Base => base_indices.push(idx),
                ConstraintType::Ext => ext_indices.push(idx),
            }
        }
        ConstraintLayout {
            base_indices,
            ext_indices,
        }
    }

    pub fn extension_constraints(&self) -> Vec<ArithmeticExpressionExt<F, EF>> {
        self.extension_constraints.clone()
    }

    pub fn base_constraints(&self) -> Vec<ArithmeticExpression<F>> {
        self.base_constraints.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for ArithmeticAirBuilder<F, EF> {
    type F = F;
    type Expr = ArithmeticExpression<F>;
    type Var = ArithmeticVariable<F>;
    type PreprocessedWindow = RowMajorMatrix<Self::Var>;
    type MainWindow = RowMajorMatrix<Self::Var>;
    type PublicVar = ArithmeticVariable<F>;
    type PeriodicVar = ArithmeticVariable<F>;

    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        ArithmeticExpression::leaf(BaseLeaf::IsFirstRow, 1)
    }

    fn is_last_row(&self) -> Self::Expr {
        ArithmeticExpression::leaf(BaseLeaf::IsLastRow, 1)
    }

    fn is_transition(&self) -> Self::Expr {
        ArithmeticExpression::leaf(BaseLeaf::IsTransition, 0)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Base);
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for ArithmeticAirBuilder<F, EF>
where
    ArithmeticExpressionExt<F, EF>: Algebra<EF>,
{
    type EF = EF;
    type ExprEF = ArithmeticExpressionExt<F, EF>;
    type VarEF = ArithmeticVariableExt<F, EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Ext);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for ArithmeticAirBuilder<F, EF>
where
    ArithmeticExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = ArithmeticVariableExt<F, EF>;

    type PermutationVar = ArithmeticVariableExt<F, EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &self.permutation_values
    }
}

#[derive(Clone, Copy, Debug)]
enum ConstraintType {
    Base,
    Ext,
}
/// Owned arithmetic roots and their global assertion order.
#[derive(Debug)]
pub struct CapturedArithmeticConstraints<F: Field, EF: ExtensionField<F> = F> {
    pub base: Vec<ArithmeticExpression<F>>,
    pub extension: Vec<ArithmeticExpressionExt<F, EF>>,
    pub layout: ConstraintLayout,
}
impl<F: Field, EF: ExtensionField<F>> ArithmeticAirBuilder<F, EF> {
    /// Move recorded roots out of the builder, retaining zero and duplicate assertions.
    pub fn finish(self) -> CapturedArithmeticConstraints<F, EF> {
        let layout = self.constraint_layout();
        CapturedArithmeticConstraints {
            base: self.base_constraints,
            extension: self.extension_constraints,
            layout,
        }
    }
}
/// Capture arithmetic roots using an explicit layout and evaluation closure.
///
/// ```
/// use p3_air::{capture_arithmetic, AirBuilder, AirLayout};
/// use p3_baby_bear::BabyBear;
/// let captured = capture_arithmetic::<BabyBear, BabyBear>(AirLayout::default(), |b| {
///     b.assert_zero(BabyBear::new(0));
/// });
/// assert_eq!(captured.base.len(), 1);
/// ```
pub fn capture_arithmetic<F: Field, EF: ExtensionField<F>>(
    layout: AirLayout,
    eval: impl FnOnce(&mut ArithmeticAirBuilder<F, EF>),
) -> CapturedArithmeticConstraints<F, EF> {
    let mut builder = ArithmeticAirBuilder::new(layout);
    eval(&mut builder);
    builder.finish()
}
impl<F: Field, EF: ExtensionField<F>> crate::PassthroughNamedAirBuilder
    for ArithmeticAirBuilder<F, EF>
{
}
