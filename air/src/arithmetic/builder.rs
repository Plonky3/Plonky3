use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use thiserror::Error;

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
    /// Construct a recording builder with the supplied input dimensions.
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

    /// Clone extension constraints in emission order, including zero and duplicate assertions.
    pub fn extension_constraints(&self) -> Vec<ArithmeticExpressionExt<F, EF>> {
        self.extension_constraints.clone()
    }

    /// Clone base constraints in emission order, including zero and duplicate assertions.
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
    /// Constraints emitted through [`AirBuilder::assert_zero`], in emission order.
    pub base: Vec<ArithmeticExpression<F>>,
    /// Constraints emitted through [`ExtensionBuilder::assert_zero_ext`], in emission order.
    pub extension: Vec<ArithmeticExpressionExt<F, EF>>,
    /// Global assertion indices for the base and extension streams.
    pub layout: ConstraintLayout,
}
/// Reasons a publicly assembled arithmetic capture cannot be lowered.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ArithmeticLoweringError {
    /// A constraint stream and its index stream have different lengths.
    #[error(
        "arithmetic capture length mismatch: base has {base_constraints} constraints and {base_indices} indices; extension has {extension_constraints} constraints and {extension_indices} indices"
    )]
    LengthMismatch {
        /// Number of base constraints.
        base_constraints: usize,
        /// Number of base indices.
        base_indices: usize,
        /// Number of extension constraints.
        extension_constraints: usize,
        /// Number of extension indices.
        extension_indices: usize,
    },
    /// The next global index does not appear in exactly one stream.
    #[error(
        "invalid arithmetic capture index at {expected}: next base index is {base_index:?}, next extension index is {extension_index:?}"
    )]
    InvalidIndex {
        /// Next required global assertion index.
        expected: usize,
        /// Next unconsumed base index, if any.
        base_index: Option<usize>,
        /// Next unconsumed extension index, if any.
        extension_index: Option<usize>,
    },
}

impl<F: Field, EF: ExtensionField<F>> CapturedArithmeticConstraints<F, EF> {
    /// Lower every root to primitive symbolic operations, preserving arithmetic values.
    ///
    /// Retains stream order, zero and duplicate roots, and global assertion indices.
    /// Reuses pointer-identity caches across roots and base expressions lifted into
    /// extension operands. Independently allocated equal expressions need not share.
    /// Grouping and sharing may differ from legacy capture; serialized structure
    /// and protocol commitments are not guaranteed to match.
    ///
    /// Traversal bounds recursion and uses an explicit stack for deeper graphs.
    /// Destroying deeply nested source or result graphs still uses recursive `Arc` drops.
    ///
    /// # Errors
    ///
    /// Returns an error before lowering if the stream and layout lengths differ,
    /// or the index streams fail to form a strictly increasing, disjoint
    /// partition of all global assertion indices.
    ///
    /// ```
    /// use p3_air::{AirBuilder, AirLayout, capture_arithmetic};
    /// use p3_baby_bear::BabyBear;
    ///
    /// let captured = capture_arithmetic::<BabyBear, BabyBear>(AirLayout::default(), |b| {
    ///     b.assert_zero(BabyBear::new(0));
    /// });
    /// let symbolic = captured.lower().unwrap();
    /// assert_eq!(symbolic.base_constraints().len(), 1);
    /// assert_eq!(symbolic.constraint_layout().base_indices, [0]);
    /// ```
    pub fn lower(&self) -> Result<crate::SymbolicConstraints<F, EF>, ArithmeticLoweringError> {
        if self.base.len() != self.layout.base_indices.len()
            || self.extension.len() != self.layout.ext_indices.len()
        {
            return Err(ArithmeticLoweringError::LengthMismatch {
                base_constraints: self.base.len(),
                base_indices: self.layout.base_indices.len(),
                extension_constraints: self.extension.len(),
                extension_indices: self.layout.ext_indices.len(),
            });
        }
        let mut base_indices = self.layout.base_indices.iter().copied().peekable();
        let mut extension_indices = self.layout.ext_indices.iter().copied().peekable();
        for expected in 0..self.base.len() + self.extension.len() {
            match (
                base_indices.peek().copied(),
                extension_indices.peek().copied(),
            ) {
                (Some(base), extension) if base == expected && extension != Some(expected) => {
                    base_indices.next();
                }
                (base, Some(extension)) if extension == expected && base != Some(expected) => {
                    extension_indices.next();
                }
                (base_index, extension_index) => {
                    return Err(ArithmeticLoweringError::InvalidIndex {
                        expected,
                        base_index,
                        extension_index,
                    });
                }
            }
        }

        let mut base_memo = BTreeMap::new();
        let mut extension_memo = BTreeMap::new();
        let base = self
            .base
            .iter()
            .map(|root| root.lower_inner(&mut base_memo, &mut (), 0))
            .collect();
        let extension = self
            .extension
            .iter()
            .map(|root| root.lower_inner(&mut extension_memo, &mut base_memo, 0))
            .collect();
        Ok(crate::SymbolicConstraints::from_parts(
            base,
            extension,
            ConstraintLayout {
                base_indices: self.layout.base_indices.clone(),
                ext_indices: self.layout.ext_indices.clone(),
            },
        ))
    }
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
