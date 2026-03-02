use alloc::sync::Arc;

use p3_field::extension::BinomialExtensionField;
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

use crate::symbolic::expression::BaseLeaf;
use crate::symbolic::variable::SymbolicVariableExt;
use crate::symbolic::{SymLeaf, SymbolicExpr, SymbolicExpression, SymbolicVariable};

/// Leaf nodes for extension-field symbolic expressions.
///
/// These represent the atomic building blocks of extension-field AIR constraints:
/// lifted base-field sub-trees, extension-field variables, and extension-field constants.
#[derive(Clone, Debug)]
pub enum ExtLeaf<F, EF> {
    /// A lifted base-field expression (entire base sub-tree preserved).
    Base(SymbolicExpression<F>),

    /// An extension-field variable (permutation column or challenge).
    ExtVariable(SymbolicVariableExt<F, EF>),

    /// An extension-field constant.
    ExtConstant(EF),
}

/// A symbolic expression tree for extension-field AIR constraints.
///
/// This is a type alias for the generic [`SymbolicExpr`] parameterized with
/// extension-field [`ExtLeaf`] nodes.
pub type SymbolicExpressionExt<F, EF> = SymbolicExpr<ExtLeaf<F, EF>>;

impl<F: Field, EF: ExtensionField<F>> SymLeaf for ExtLeaf<F, EF> {
    type F = F;

    const ZERO: Self = Self::Base(SymbolicExpression::ZERO);
    const ONE: Self = Self::Base(SymbolicExpression::ONE);
    const TWO: Self = Self::Base(SymbolicExpression::TWO);
    const NEG_ONE: Self = Self::Base(SymbolicExpression::NEG_ONE);

    fn degree_multiple(&self) -> usize {
        match self {
            Self::Base(e) => e.degree_multiple(),
            Self::ExtVariable(v) => v.degree_multiple(),
            Self::ExtConstant(_) => 0,
        }
    }

    fn as_const(&self) -> Option<&F> {
        match self {
            Self::Base(SymbolicExpression::Leaf(BaseLeaf::Constant(c))) => Some(c),
            _ => None,
        }
    }

    fn from_const(c: F) -> Self {
        Self::Base(SymbolicExpression::from(c))
    }
}

impl<F: Field, EF> SymbolicExpressionExt<F, EF> {
    /// Try to lower this extension expression to a base-field expression.
    ///
    /// Returns `None` if the tree contains any extension-only nodes
    /// ([`ExtVariable`](ExtLeaf::ExtVariable) or [`ExtConstant`](ExtLeaf::ExtConstant)).
    pub fn to_base(&self) -> Option<SymbolicExpression<F>> {
        match self {
            Self::Leaf(ExtLeaf::Base(e)) => Some(e.clone()),
            Self::Leaf(ExtLeaf::ExtVariable(_) | ExtLeaf::ExtConstant(_)) => None,
            Self::Add {
                x,
                y,
                degree_multiple,
            } => Some(SymbolicExpr::Add {
                x: Arc::new(x.to_base()?),
                y: Arc::new(y.to_base()?),
                degree_multiple: *degree_multiple,
            }),
            Self::Sub {
                x,
                y,
                degree_multiple,
            } => Some(SymbolicExpr::Sub {
                x: Arc::new(x.to_base()?),
                y: Arc::new(y.to_base()?),
                degree_multiple: *degree_multiple,
            }),
            Self::Neg { x, degree_multiple } => Some(SymbolicExpr::Neg {
                x: Arc::new(x.to_base()?),
                degree_multiple: *degree_multiple,
            }),
            Self::Mul {
                x,
                y,
                degree_multiple,
            } => Some(SymbolicExpr::Mul {
                x: Arc::new(x.to_base()?),
                y: Arc::new(y.to_base()?),
                degree_multiple: *degree_multiple,
            }),
        }
    }
}

impl<F: Field, EF> From<SymbolicExpression<F>> for SymbolicExpressionExt<F, EF> {
    fn from(expr: SymbolicExpression<F>) -> Self {
        Self::Leaf(ExtLeaf::Base(expr))
    }
}

impl<F: Field, EF> From<SymbolicVariable<F>> for SymbolicExpressionExt<F, EF> {
    fn from(var: SymbolicVariable<F>) -> Self {
        Self::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Variable(var))))
    }
}

impl<F, EF> From<SymbolicVariableExt<F, EF>> for SymbolicExpressionExt<F, EF> {
    fn from(var: SymbolicVariableExt<F, EF>) -> Self {
        Self::Leaf(ExtLeaf::ExtVariable(var))
    }
}

impl<F: Field, EF> From<F> for SymbolicExpressionExt<F, EF> {
    fn from(f: F) -> Self {
        Self::Leaf(ExtLeaf::Base(SymbolicExpression::from(f)))
    }
}

/// Concrete [`From`] for [`BinomialExtensionField`] constants.
///
/// This avoids overlap with [`From<F>`] when `EF = F`, since
/// [`BinomialExtensionField<F, D>`] is always a distinct type from `F`.
impl<F, const D: usize> From<BinomialExtensionField<F, D>>
    for SymbolicExpressionExt<F, BinomialExtensionField<F, D>>
where
    F: Field,
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
    fn from(ef: BinomialExtensionField<F, D>) -> Self {
        Self::Leaf(ExtLeaf::ExtConstant(ef))
    }
}

impl<F: Field, EF: ExtensionField<F>> Algebra<F> for SymbolicExpressionExt<F, EF> {}

impl<F: Field, EF: ExtensionField<F>> Algebra<SymbolicExpression<F>>
    for SymbolicExpressionExt<F, EF>
{
}

impl<F: Field, EF: ExtensionField<F>> Algebra<SymbolicVariable<F>>
    for SymbolicExpressionExt<F, EF>
{
}

impl<F: Field, EF: ExtensionField<F>> Algebra<SymbolicVariableExt<F, EF>>
    for SymbolicExpressionExt<F, EF>
{
}

/// Concrete [`Algebra`] for [`BinomialExtensionField`] — avoids overlap with `Algebra<F>` when `EF = F`.
impl<F: Field, const D: usize> Algebra<BinomialExtensionField<F, D>>
    for SymbolicExpressionExt<F, BinomialExtensionField<F, D>>
where
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
}
