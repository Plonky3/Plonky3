use alloc::sync::Arc;

use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
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
            Self::ExtConstant(ef) if ef.is_in_basefield() => {
                Some(&ef.as_basis_coefficients_slice()[0])
            }
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

impl<F: Field> From<QuinticTrinomialExtensionField<F>>
    for SymbolicExpressionExt<F, QuinticTrinomialExtensionField<F>>
where
    QuinticTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn from(ef: QuinticTrinomialExtensionField<F>) -> Self {
        Self::Leaf(ExtLeaf::ExtConstant(ef))
    }
}

/// Concrete [`Algebra`] for [`QuinticTrinomialExtensionField`] — avoids overlap with `Algebra<F>`.
impl<F: Field> Algebra<QuinticTrinomialExtensionField<F>>
    for SymbolicExpressionExt<F, QuinticTrinomialExtensionField<F>>
where
    QuinticTrinomialExtensionField<F>: ExtensionField<F>,
{
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    use super::*;
    use crate::symbolic::SymbolicExpr;
    use crate::symbolic::variable::{BaseEntry, ExtEntry};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn ext_leaf_degree_multiple_base_variable() {
        // A base leaf with a trace variable inside has degree 1.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let leaf = ExtLeaf::<F, EF>::Base(SymbolicExpression::from(var));
        assert_eq!(leaf.degree_multiple(), 1);
    }

    #[test]
    fn ext_leaf_degree_multiple_base_constant() {
        // A base leaf with a constant inside has degree 0.
        let leaf = ExtLeaf::<F, EF>::Base(SymbolicExpression::from(F::new(42)));
        assert_eq!(leaf.degree_multiple(), 0);
    }

    #[test]
    fn ext_leaf_degree_multiple_ext_variable() {
        // A permutation variable has degree 1.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let leaf = ExtLeaf::ExtVariable(var);
        assert_eq!(leaf.degree_multiple(), 1);
    }

    #[test]
    fn ext_leaf_degree_multiple_ext_variable_challenge() {
        // A challenge variable has degree 0.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Challenge, 0);
        let leaf = ExtLeaf::ExtVariable(var);
        assert_eq!(leaf.degree_multiple(), 0);
    }

    #[test]
    fn ext_leaf_degree_multiple_ext_constant() {
        // An extension constant always has degree 0.
        let leaf = ExtLeaf::<F, EF>::ExtConstant(EF::ONE);
        assert_eq!(leaf.degree_multiple(), 0);
    }

    #[test]
    fn ext_leaf_as_const_base_constant() {
        // A base constant leaf can be viewed as a field constant.
        let leaf = ExtLeaf::<F, EF>::Base(SymbolicExpression::from(F::new(7)));
        assert_eq!(leaf.as_const(), Some(&F::new(7)));
    }

    #[test]
    fn ext_leaf_as_const_base_variable() {
        // A base variable leaf is not a constant.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let leaf = ExtLeaf::<F, EF>::Base(SymbolicExpression::from(var));
        assert!(leaf.as_const().is_none());
    }

    #[test]
    fn ext_leaf_as_const_ext_variable() {
        // An extension variable leaf is not a constant.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let leaf = ExtLeaf::ExtVariable(var);
        assert!(leaf.as_const().is_none());
    }

    #[test]
    fn ext_leaf_as_const_ext_constant_in_basefield() {
        // An extension constant that lies in the base field is recognized as a constant.
        let leaf = ExtLeaf::<F, EF>::ExtConstant(EF::ONE);
        assert_eq!(leaf.as_const(), Some(&F::ONE));
    }

    #[test]
    fn ext_leaf_as_const_ext_constant_zero() {
        // The extension zero element is recognized as the base zero.
        let leaf = ExtLeaf::<F, EF>::ExtConstant(EF::ZERO);
        assert_eq!(leaf.as_const(), Some(&F::ZERO));
    }

    #[test]
    fn ext_leaf_as_const_ext_constant_not_in_basefield() {
        // An extension constant with non-zero higher coefficients is not a base constant.
        let ef_val = EF::from_basis_coefficients_fn(|i| if i == 1 { F::ONE } else { F::ZERO });
        let leaf = ExtLeaf::<F, EF>::ExtConstant(ef_val);
        assert!(leaf.as_const().is_none());
    }

    #[test]
    fn ext_leaf_from_const() {
        // Creating a leaf from a base-field value produces a constant.
        let leaf = ExtLeaf::<F, EF>::from_const(F::new(13));
        assert_eq!(leaf.as_const(), Some(&F::new(13)));
    }

    #[test]
    fn to_base_leaf_base() {
        // A base-only leaf can be lowered to a base expression.
        let base_expr = SymbolicExpression::from(F::new(5));
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(base_expr);
        let lowered = ext_expr.to_base();

        assert!(lowered.is_some());
        assert!(matches!(
            lowered.unwrap(),
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == F::new(5)
        ));
    }

    #[test]
    fn to_base_leaf_ext_variable() {
        // An extension variable cannot be lowered to base.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(var);
        assert!(ext_expr.to_base().is_none());
    }

    #[test]
    fn to_base_leaf_ext_constant() {
        // An extension constant cannot be lowered to base.
        let ext_expr = SymbolicExpressionExt::<F, EF>::Leaf(ExtLeaf::ExtConstant(EF::TWO));
        assert!(ext_expr.to_base().is_none());
    }

    #[test]
    fn to_base_add_of_base_exprs() {
        // A sum of two base-only expressions can be lowered.
        let a = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let b = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let sum = a + b;
        let lowered = sum.to_base();

        match lowered {
            Some(SymbolicExpr::Add {
                x,
                y,
                degree_multiple,
            }) => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if *c == F::new(3)
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
            }
            _ => panic!("Expected a lowered Add node"),
        }
    }

    #[test]
    fn to_base_add_with_ext_child_returns_none() {
        // A sum with one extension-only child cannot be lowered.
        let base = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let ext_var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let sum = base + ext_var;
        assert!(sum.to_base().is_none());
    }

    #[test]
    fn to_base_sub_of_base_exprs() {
        // A difference of two base-only expressions can be lowered.
        let a = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let b = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            1,
        ));
        let diff = a - b;
        let lowered = diff.to_base();

        match lowered {
            Some(SymbolicExpr::Sub {
                x,
                y,
                degree_multiple,
            }) => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 1 && v.entry == BaseEntry::Main { offset: 0 }
                ));
            }
            _ => panic!("Expected a lowered Sub node"),
        }
    }

    #[test]
    fn to_base_sub_with_ext_child_returns_none() {
        // A difference with an extension-only child cannot be lowered.
        let base = SymbolicExpressionExt::<F, EF>::from(F::new(5));
        let ext_var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Challenge,
            0,
        ));
        let diff = base - ext_var;
        assert!(diff.to_base().is_none());
    }

    #[test]
    fn to_base_neg_of_base_expr() {
        // Negation of a base-only expression can be lowered.
        let var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let neg = -var;
        let lowered = neg.to_base();

        match lowered {
            Some(SymbolicExpr::Neg { x, degree_multiple }) => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
            }
            _ => panic!("Expected a lowered Neg node"),
        }
    }

    #[test]
    fn to_base_mul_of_base_exprs() {
        // A product of two base-only expressions can be lowered.
        let a = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let b = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            1,
        ));
        let prod = a * b;
        let lowered = prod.to_base();

        match lowered {
            Some(SymbolicExpr::Mul {
                x,
                y,
                degree_multiple,
            }) => {
                assert_eq!(degree_multiple, 2);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 1 && v.entry == BaseEntry::Main { offset: 0 }
                ));
            }
            _ => panic!("Expected a lowered Mul node"),
        }
    }

    #[test]
    fn to_base_mul_with_ext_child_returns_none() {
        // A product with an extension-only child cannot be lowered.
        let base = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let ext_var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let prod = base * ext_var;
        assert!(prod.to_base().is_none());
    }

    #[test]
    fn from_symbolic_expression() {
        // Converting a base expression lifts it into a base leaf.
        let base_expr = SymbolicExpression::from(F::new(99));
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(base_expr);
        assert!(matches!(
            ext_expr,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c)))) if c == F::new(99)
        ));
    }

    #[test]
    fn from_symbolic_variable() {
        // Converting a base variable lifts it into a base leaf.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 2);
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(var);
        assert!(matches!(
            ext_expr,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Variable(v))))
                if v.index == 2 && v.entry == BaseEntry::Main { offset: 0 }
        ));
    }

    #[test]
    fn from_symbolic_variable_ext() {
        // Converting an extension variable produces an extension variable leaf.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 1 }, 3);
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(var);
        assert!(matches!(
            ext_expr,
            SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                if v.index == 3 && v.entry == ExtEntry::Permutation { offset: 1 }
        ));
    }

    #[test]
    fn from_base_field() {
        // Converting a base field element produces a base constant leaf.
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(F::new(42));
        assert!(matches!(
            ext_expr,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c)))) if c == F::new(42)
        ));
    }

    #[test]
    fn from_binomial_extension_field() {
        // Converting an extension field element produces an extension constant leaf.
        let ef_val = EF::ONE + EF::ONE;
        let ext_expr = SymbolicExpressionExt::<F, EF>::from(ef_val);
        assert!(matches!(
            ext_expr,
            SymbolicExpr::Leaf(ExtLeaf::ExtConstant(c)) if c == ef_val
        ));
    }

    #[test]
    fn ext_add_constant_folding() {
        // Two base constants fold into one on addition.
        let a = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let b = SymbolicExpressionExt::<F, EF>::from(F::new(4));
        let result = a + b;
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c)))) if c == F::new(7)
        ));
    }

    #[test]
    fn ext_sub_constant_folding() {
        // Two base constants fold into one on subtraction.
        let a = SymbolicExpressionExt::<F, EF>::from(F::new(10));
        let b = SymbolicExpressionExt::<F, EF>::from(F::new(4));
        let result = a - b;
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c)))) if c == F::new(6)
        ));
    }

    #[test]
    fn ext_mul_constant_folding() {
        // Two base constants fold into one on multiplication.
        let a = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let b = SymbolicExpressionExt::<F, EF>::from(F::new(5));
        let result = a * b;
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c)))) if c == F::new(15)
        ));
    }

    #[test]
    fn ext_add_variables_degree_tracking() {
        // Adding two degree-1 variables gives degree 1 (the max).
        let a = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let b = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            1,
        ));
        let result = a + b;

        match result {
            SymbolicExpr::Add {
                x,
                y,
                degree_multiple,
            } => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 1 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
            }
            _ => panic!("Expected an Add node"),
        }
    }

    #[test]
    fn ext_mul_variables_degree_tracking() {
        // Multiplying two degree-1 variables gives degree 2 (the sum).
        let a = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let b = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            1,
        ));
        let result = a * b;

        match result {
            SymbolicExpr::Mul {
                x,
                y,
                degree_multiple,
            } => {
                assert_eq!(degree_multiple, 2);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 1 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
            }
            _ => panic!("Expected a Mul node"),
        }
    }

    #[test]
    fn ext_constant_zero_mul_folds_to_zero() {
        // Multiplying by the extension zero folds to the zero constant.
        let var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let zero = SymbolicExpressionExt::<F, EF>::from(EF::ZERO);
        let result = var * zero;
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c)))) if c == F::ZERO
        ));
    }

    #[test]
    fn ext_constant_one_mul_folds_to_identity() {
        // Multiplying by the extension one folds to the other operand.
        let var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let one = SymbolicExpressionExt::<F, EF>::from(EF::ONE);
        let result = var * one;
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
        ));
    }

    #[test]
    fn ext_constant_zero_add_folds_to_identity() {
        // Adding the extension zero folds to the other operand.
        let var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let zero = SymbolicExpressionExt::<F, EF>::from(EF::ZERO);
        let result = zero + var;
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
        ));
    }

    #[test]
    fn ext_constant_zero_sub_folds_to_neg() {
        // Subtracting from the extension zero folds to negation.
        let var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let zero = SymbolicExpressionExt::<F, EF>::from(EF::ZERO);
        let result = zero - var;
        match result {
            SymbolicExpr::Neg { x, degree_multiple } => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
            }
            _ => panic!("Expected a Neg node"),
        }
    }

    #[test]
    fn ext_constant_not_in_basefield_no_folding() {
        // A non-base-field extension constant does not fold with multiplication.
        let var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let non_base = SymbolicExpressionExt::<F, EF>::from(EF::from_basis_coefficients_fn(|i| {
            if i == 1 { F::ONE } else { F::ZERO }
        }));
        let result = var * non_base;
        match result {
            SymbolicExpr::Mul {
                x,
                y,
                degree_multiple,
            } => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtConstant(_))
                ));
            }
            _ => panic!("Expected a Mul node since the constant is not in the base field"),
        }
    }

    #[test]
    fn ext_mixed_base_and_ext_arithmetic() {
        // Mixing a base variable with an extension variable in a sum.
        let base_var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let ext_var = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            0,
        ));
        let result = base_var + ext_var;

        match &result {
            SymbolicExpr::Add {
                x,
                y,
                degree_multiple,
            } => {
                assert_eq!(*degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Variable(v))))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v))
                        if v.index == 0 && v.entry == ExtEntry::Permutation { offset: 0 }
                ));
            }
            _ => panic!("Expected an Add node"),
        }

        // The mixed result cannot be lowered to base.
        assert!(result.to_base().is_none());
    }
}
