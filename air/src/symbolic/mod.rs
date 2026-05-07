//! Symbolic expression types for AIR constraint representation.

mod builder;
mod expression;
pub(crate) mod expression_ext;
mod variable;

use alloc::sync::Arc;
use core::iter::{Product, Sum};
use core::ops;

pub use builder::*;
pub use expression::{BaseLeaf, SymbolicExpression};
pub use expression_ext::{ExtLeaf, SymbolicExpressionExt};
use p3_field::{Dup, ExtensionField, Field, PrimeCharacteristicRing};
pub use variable::{BaseEntry, ExtEntry, SymbolicVariable, SymbolicVariableExt};

/// Properties that leaf nodes must provide for the generic expression tree.
///
/// Both [`BaseLeaf`](expression::BaseLeaf) (base-field) and
/// [`ExtLeaf`](expression_ext::ExtLeaf) (extension-field) implement this trait,
/// enabling [`SymbolicExpr`] to handle constant folding, degree tracking, and
/// arithmetic generically.
pub trait SymLeaf: Clone + core::fmt::Debug {
    /// The base field type used for constant folding.
    type F: Field;

    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;

    /// Returns the degree multiple of this leaf.
    fn degree_multiple(&self) -> usize;

    /// Try to view this leaf as a base-field constant.
    fn as_const(&self) -> Option<&Self::F>;

    /// Create a leaf from a base-field constant.
    fn from_const(c: Self::F) -> Self;
}

/// A symbolic expression tree, generic over its leaf type `A`.
///
/// This enum captures the shared tree structure — Add/Sub/Neg/Mul nodes with
/// `Arc`-wrapped children and cached degree multiples — used by both base-field
/// and extension-field symbolic expressions.
///
/// Concrete types are provided via type aliases:
/// - [`SymbolicExpression<F>`] = `SymbolicExpr<BaseLeaf<F>>` (base-field constraints)
/// - [`SymbolicExpressionExt<F, EF>`] = `SymbolicExpr<ExtLeaf<F, EF>>` (extension-field constraints)
#[derive(Clone, Debug)]
pub enum SymbolicExpr<A> {
    /// A leaf node (variable, constant, selector, or lifted sub-expression).
    Leaf(A),

    /// Addition of two sub-expressions.
    Add {
        x: Arc<Self>,
        y: Arc<Self>,
        degree_multiple: usize,
    },

    /// Subtraction of two sub-expressions.
    Sub {
        x: Arc<Self>,
        y: Arc<Self>,
        degree_multiple: usize,
    },

    /// Negation of a sub-expression.
    Neg {
        x: Arc<Self>,
        degree_multiple: usize,
    },

    /// Multiplication of two sub-expressions.
    Mul {
        x: Arc<Self>,
        y: Arc<Self>,
        degree_multiple: usize,
    },
}

impl<A: SymLeaf> SymbolicExpr<A> {
    /// Returns the degree multiple of this expression.
    pub fn degree_multiple(&self) -> usize {
        match self {
            Self::Leaf(a) => a.degree_multiple(),
            Self::Add {
                degree_multiple, ..
            }
            | Self::Sub {
                degree_multiple, ..
            }
            | Self::Neg {
                degree_multiple, ..
            }
            | Self::Mul {
                degree_multiple, ..
            } => *degree_multiple,
        }
    }

    /// Try to view this expression as a base-field constant.
    fn as_const(&self) -> Option<&A::F> {
        match self {
            Self::Leaf(a) => a.as_const(),
            _ => None,
        }
    }

    /// Addition with constant folding and zero-identity elimination.
    fn sym_add(self, rhs: Self) -> Self {
        if let (Some(&a), Some(&b)) = (self.as_const(), rhs.as_const()) {
            return Self::Leaf(A::from_const(a + b));
        }
        if self.as_const().is_some_and(|c| c.is_zero()) {
            return rhs;
        }
        if rhs.as_const().is_some_and(|c| c.is_zero()) {
            return self;
        }
        let dm = self.degree_multiple().max(rhs.degree_multiple());
        Self::Add {
            x: Arc::new(self),
            y: Arc::new(rhs),
            degree_multiple: dm,
        }
    }

    /// Subtraction with constant folding and zero-identity elimination.
    fn sym_sub(self, rhs: Self) -> Self {
        if let (Some(&a), Some(&b)) = (self.as_const(), rhs.as_const()) {
            return Self::Leaf(A::from_const(a - b));
        }
        if self.as_const().is_some_and(|c| c.is_zero()) {
            return rhs.sym_neg();
        }
        if rhs.as_const().is_some_and(|c| c.is_zero()) {
            return self;
        }
        let dm = self.degree_multiple().max(rhs.degree_multiple());
        Self::Sub {
            x: Arc::new(self),
            y: Arc::new(rhs),
            degree_multiple: dm,
        }
    }

    /// Negation with constant folding.
    fn sym_neg(self) -> Self {
        if let Some(&c) = self.as_const() {
            return Self::Leaf(A::from_const(-c));
        }
        let dm = self.degree_multiple();
        Self::Neg {
            x: Arc::new(self),
            degree_multiple: dm,
        }
    }

    /// Multiplication with constant folding, zero-annihilation, and one-identity.
    fn sym_mul(self, rhs: Self) -> Self {
        if let (Some(&a), Some(&b)) = (self.as_const(), rhs.as_const()) {
            return Self::Leaf(A::from_const(a * b));
        }
        if self.as_const().is_some_and(|c| c.is_zero())
            || rhs.as_const().is_some_and(|c| c.is_zero())
        {
            return Self::Leaf(A::from_const(A::F::ZERO));
        }
        if self.as_const().is_some_and(|c| c.is_one()) {
            return rhs;
        }
        if rhs.as_const().is_some_and(|c| c.is_one()) {
            return self;
        }
        let dm = self.degree_multiple() + rhs.degree_multiple();
        Self::Mul {
            x: Arc::new(self),
            y: Arc::new(rhs),
            degree_multiple: dm,
        }
    }
}

impl<A: SymLeaf> PrimeCharacteristicRing for SymbolicExpr<A> {
    type PrimeSubfield = <A::F as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::Leaf(A::ZERO);
    const ONE: Self = Self::Leaf(A::ONE);
    const TWO: Self = Self::Leaf(A::TWO);
    const NEG_ONE: Self = Self::Leaf(A::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::Leaf(A::from_const(A::F::from_prime_subfield(f)))
    }
}

impl<A: SymLeaf> Dup for SymbolicExpr<A> {
    #[inline(always)]
    fn dup(&self) -> Self {
        self.clone()
    }
}

impl<A: SymLeaf> Default for SymbolicExpr<A> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<A: SymLeaf, T: Into<Self>> ops::Add<T> for SymbolicExpr<A> {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        self.sym_add(rhs.into())
    }
}

impl<A: SymLeaf, T: Into<Self>> ops::Sub<T> for SymbolicExpr<A> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        self.sym_sub(rhs.into())
    }
}

impl<A: SymLeaf> ops::Neg for SymbolicExpr<A> {
    type Output = Self;
    fn neg(self) -> Self {
        self.sym_neg()
    }
}

impl<A: SymLeaf, T: Into<Self>> ops::Mul<T> for SymbolicExpr<A> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        self.sym_mul(rhs.into())
    }
}

impl<A: SymLeaf, T: Into<Self>> ops::AddAssign<T> for SymbolicExpr<A> {
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<A: SymLeaf, T: Into<Self>> ops::SubAssign<T> for SymbolicExpr<A> {
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<A: SymLeaf, T: Into<Self>> ops::MulAssign<T> for SymbolicExpr<A> {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<A: SymLeaf, T: Into<Self>> Sum<T> for SymbolicExpr<A> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|a, b| a + b)
            .unwrap_or(Self::ZERO)
    }
}

impl<A: SymLeaf, T: Into<Self>> Product<T> for SymbolicExpr<A> {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|a, b| a * b)
            .unwrap_or(Self::ONE)
    }
}

impl<F: Field, T: Into<SymbolicExpression<F>>> ops::Add<T> for SymbolicVariable<F> {
    type Output = SymbolicExpression<F>;
    fn add(self, rhs: T) -> Self::Output {
        Self::Output::from(self) + rhs.into()
    }
}

impl<F: Field, T: Into<SymbolicExpression<F>>> ops::Sub<T> for SymbolicVariable<F> {
    type Output = SymbolicExpression<F>;
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from(self) - rhs.into()
    }
}

impl<F: Field, T: Into<SymbolicExpression<F>>> ops::Mul<T> for SymbolicVariable<F> {
    type Output = SymbolicExpression<F>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from(self) * rhs.into()
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<SymbolicExpressionExt<F, EF>>> ops::Add<T>
    for SymbolicVariableExt<F, EF>
{
    type Output = SymbolicExpressionExt<F, EF>;
    fn add(self, rhs: T) -> Self::Output {
        Self::Output::from(self) + rhs.into()
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<SymbolicExpressionExt<F, EF>>> ops::Sub<T>
    for SymbolicVariableExt<F, EF>
{
    type Output = SymbolicExpressionExt<F, EF>;
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from(self) - rhs.into()
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<SymbolicExpressionExt<F, EF>>> ops::Mul<T>
    for SymbolicVariableExt<F, EF>
{
    type Output = SymbolicExpressionExt<F, EF>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from(self) * rhs.into()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::symbolic::expression::BaseLeaf;
    use crate::symbolic::expression_ext::ExtLeaf;
    use crate::symbolic::variable::{BaseEntry, ExtEntry};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn symbolic_variable_add_produces_add_node() {
        // Adding a variable and a non-zero constant creates an addition node.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let expr = SymbolicExpression::from(F::new(5));
        let result = var + expr;
        match result {
            SymbolicExpr::Add {
                x,
                y,
                degree_multiple,
            } => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if *c == F::new(5)
                ));
            }
            _ => panic!("Expected an Add node"),
        }
    }

    #[test]
    fn symbolic_variable_sub_produces_sub_node() {
        // Subtracting two variables creates a subtraction node.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let other = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));
        let result = var - other;
        match result {
            SymbolicExpr::Sub {
                x,
                y,
                degree_multiple,
            } => {
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
            _ => panic!("Expected a Sub node"),
        }
    }

    #[test]
    fn symbolic_variable_mul_produces_mul_node() {
        // Multiplying two variables creates a multiplication node with summed degree.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let other = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));
        let result = var * other;
        match result {
            SymbolicExpr::Mul {
                x,
                y,
                degree_multiple,
            } => {
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
            _ => panic!("Expected a Mul node"),
        }
    }

    #[test]
    fn symbolic_variable_ext_add_produces_add_node() {
        // Adding an extension variable and a non-zero constant creates an addition node.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let expr = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let result = var + expr;
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
                    SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Leaf(BaseLeaf::Constant(c))))
                        if *c == F::new(3)
                ));
            }
            _ => panic!("Expected an Add node"),
        }
    }

    #[test]
    fn symbolic_variable_ext_sub_produces_sub_node() {
        // Subtracting two extension variables creates a subtraction node.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let other = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            1,
        ));
        let result = var - other;
        match result {
            SymbolicExpr::Sub {
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
            _ => panic!("Expected a Sub node"),
        }
    }

    #[test]
    fn symbolic_variable_ext_mul_produces_mul_node() {
        // Multiplying two extension variables creates a multiplication node with summed degree.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let other = SymbolicExpressionExt::<F, EF>::from(SymbolicVariableExt::<F, EF>::new(
            ExtEntry::Permutation { offset: 0 },
            1,
        ));
        let result = var * other;
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
}
