use alloc::sync::Arc;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::extension::BinomialExtensionField;
use p3_field::{Algebra, ExtensionField, Field, InjectiveMonomial, PrimeCharacteristicRing};

use crate::symbolic::SymbolicVariable;

impl<F, const D: usize> From<SymbolicExpression<F>>
    for SymbolicExpression<BinomialExtensionField<F, D>>
where
    F: Field,
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
    /// Generic implementation for ANY field F using a [`BinomialExtensionField`].
    /// This works for BabyBear, KoalaBear, Mersenne31, and any future field
    /// without modifying this crate.
    ///
    /// Since [`BinomialExtensionField<F, D>`] is always a distinct type from `F`,
    /// this implementation doesn't conflict with the blanket `From<T> for T`.
    fn from(expr: SymbolicExpression<F>) -> Self {
        match expr {
            SymbolicExpression::Variable(v) => {
                Self::Variable(SymbolicVariable::new(v.entry, v.index))
            }
            SymbolicExpression::IsFirstRow => Self::IsFirstRow,
            SymbolicExpression::IsLastRow => Self::IsLastRow,
            SymbolicExpression::IsTransition => Self::IsTransition,
            SymbolicExpression::Constant(c) => {
                // We convert the base constant 'c' into the extension field
                Self::Constant(BinomialExtensionField::<F, D>::from(c))
            }
            SymbolicExpression::Add {
                x,
                y,
                degree_multiple,
            } => Self::Add {
                x: Arc::new(Self::from((*x).clone())),
                y: Arc::new(Self::from((*y).clone())),
                degree_multiple,
            },
            SymbolicExpression::Sub {
                x,
                y,
                degree_multiple,
            } => Self::Sub {
                x: Arc::new(Self::from((*x).clone())),
                y: Arc::new(Self::from((*y).clone())),
                degree_multiple,
            },
            SymbolicExpression::Neg { x, degree_multiple } => Self::Neg {
                x: Arc::new(Self::from((*x).clone())),
                degree_multiple,
            },
            SymbolicExpression::Mul {
                x,
                y,
                degree_multiple,
            } => Self::Mul {
                x: Arc::new(Self::from((*x).clone())),
                y: Arc::new(Self::from((*y).clone())),
                degree_multiple,
            },
        }
    }
}

/// A symbolic expression tree representing AIR constraint computations over [`SymbolicVariable`]s.
///
/// This enum forms an Abstract Syntax Tree (AST) for constraint expressions.
///
/// Each node represents either:
/// - A leaf value (variable, constant, selector) or
/// - An arithmetic operation combining sub-expressions.
#[derive(Clone, Debug)]
pub enum SymbolicExpression<F> {
    /// A reference to a trace column or public input.
    ///
    /// Wraps a [`SymbolicVariable`] that identifies which column and row offset.
    Variable(SymbolicVariable<F>),

    /// Selector that is:
    /// - 1 on the first row,
    /// - 0 elsewhere.
    ///
    /// Evaluates to `L_0(x)`, the Lagrange basis polynomial for index 0.
    IsFirstRow,

    /// Selector that is:
    /// - 1 on the last row,
    /// - 0 elsewhere.
    ///
    /// Evaluates to `L_{n-1}(x)`, the Lagrange basis polynomial for the last index.
    IsLastRow,

    /// Selector that is:
    /// - 1 on all rows except the last,
    /// - 0 on the last row.
    ///
    /// Used for transition constraints that should not apply on the final row.
    IsTransition,

    /// A constant field element.
    Constant(F),

    /// Addition of two sub-expressions.
    Add {
        /// Left operand.
        x: Arc<Self>,
        /// Right operand.
        y: Arc<Self>,
        /// Cached degree multiple: `max(x.degree_multiple, y.degree_multiple)`.
        degree_multiple: usize,
    },

    /// Subtraction of two sub-expressions.
    Sub {
        /// Left operand (minuend).
        x: Arc<Self>,
        /// Right operand (subtrahend).
        y: Arc<Self>,
        /// Cached degree multiple: `max(x.degree_multiple, y.degree_multiple)`.
        degree_multiple: usize,
    },

    /// Negation of a sub-expression.
    Neg {
        /// The expression to negate.
        x: Arc<Self>,
        /// Cached degree multiple: same as `x.degree_multiple`.
        degree_multiple: usize,
    },

    /// Multiplication of two sub-expressions.
    Mul {
        /// Left operand.
        x: Arc<Self>,
        /// Right operand.
        y: Arc<Self>,
        /// Cached degree multiple: `x.degree_multiple + y.degree_multiple`.
        degree_multiple: usize,
    },
}

impl<F> SymbolicExpression<F> {
    /// Returns the degree multiple of this expression.
    ///
    /// The degree multiple represents how many times the trace length `n`
    /// appears in the expression's polynomial degree. This determines:
    /// - The quotient polynomial's degree
    /// - The required FRI blowup factor
    ///
    /// # Degree Rules
    ///
    /// Degree 0 (constants):
    /// - `Constant`
    /// - `Variable` with public values or challenges
    ///
    /// Degree 1 (linear in trace length):
    /// - `Variable` with trace columns (main, preprocessed, permutation)
    /// - `IsFirstRow`
    /// - `IsLastRow`
    /// - `IsTransition`
    ///
    /// Composite expressions:
    /// - `Add`, `Sub`: max of operands
    /// - `Neg`: same as operand
    /// - `Mul`: sum of operands
    pub const fn degree_multiple(&self) -> usize {
        match self {
            Self::Variable(v) => v.degree_multiple(),
            Self::IsFirstRow | Self::IsLastRow | Self::IsTransition => 1,
            Self::Constant(_) => 0,
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
}

impl<F: Field> Default for SymbolicExpression<F> {
    fn default() -> Self {
        Self::Constant(F::ZERO)
    }
}

impl<F: Field, EF: ExtensionField<F>> From<SymbolicVariable<F>> for SymbolicExpression<EF> {
    fn from(var: SymbolicVariable<F>) -> Self {
        Self::Variable(SymbolicVariable::new(var.entry, var.index))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<F> for SymbolicExpression<EF> {
    fn from(var: F) -> Self {
        Self::Constant(var.into())
    }
}

impl<F: Field> PrimeCharacteristicRing for SymbolicExpression<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        F::from_prime_subfield(f).into()
    }
}

impl<F: Field> Algebra<F> for SymbolicExpression<F> {}

impl<F: Field> Algebra<SymbolicVariable<F>> for SymbolicExpression<F> {}

// Note we cannot implement PermutationMonomial due to the degree_multiple part which makes
// operations non invertible.
impl<F: Field + InjectiveMonomial<N>, const N: u64> InjectiveMonomial<N> for SymbolicExpression<F> {}

impl<F: Field, T> Add<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs + rhs),
            (lhs, rhs) => Self::Add {
                degree_multiple: lhs.degree_multiple().max(rhs.degree_multiple()),
                x: Arc::new(lhs),
                y: Arc::new(rhs),
            },
        }
    }
}

impl<F: Field, T> AddAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<F: Field, T> Sum<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x + y)
            .unwrap_or(Self::ZERO)
    }
}

impl<F: Field, T: Into<Self>> Sub<T> for SymbolicExpression<F> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs - rhs),
            (lhs, rhs) => Self::Sub {
                degree_multiple: lhs.degree_multiple().max(rhs.degree_multiple()),
                x: Arc::new(lhs),
                y: Arc::new(rhs),
            },
        }
    }
}

impl<F: Field, T> SubAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<F: Field> Neg for SymbolicExpression<F> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Constant(c) => Self::Constant(-c),
            expr => Self::Neg {
                degree_multiple: expr.degree_multiple(),
                x: Arc::new(expr),
            },
        }
    }
}

impl<F: Field, T: Into<Self>> Mul<T> for SymbolicExpression<F> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        match (self, rhs.into()) {
            (Self::Constant(lhs), Self::Constant(rhs)) => Self::Constant(lhs * rhs),
            (lhs, rhs) => Self::Mul {
                degree_multiple: lhs.degree_multiple() + rhs.degree_multiple(),
                x: Arc::new(lhs),
                y: Arc::new(rhs),
            },
        }
    }
}

impl<F: Field, T> MulAssign<T> for SymbolicExpression<F>
where
    T: Into<Self>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<F: Field, T: Into<Self>> Product<T> for SymbolicExpression<F> {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|x, y| x * y)
            .unwrap_or(Self::ONE)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::symbolic::Entry;

    #[test]
    fn test_symbolic_expression_degree_multiple() {
        let constant_expr = SymbolicExpression::<BabyBear>::Constant(BabyBear::new(5));
        assert_eq!(
            constant_expr.degree_multiple(),
            0,
            "Constant should have degree 0"
        );

        let variable_expr =
            SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, 1));
        assert_eq!(
            variable_expr.degree_multiple(),
            1,
            "Main variable should have degree 1"
        );

        let preprocessed_var = SymbolicExpression::Variable(SymbolicVariable::new(
            Entry::Preprocessed { offset: 0 },
            2,
        ));
        assert_eq!(
            preprocessed_var.degree_multiple(),
            1,
            "Preprocessed variable should have degree 1"
        );

        let permutation_var = SymbolicExpression::Variable(SymbolicVariable::<BabyBear>::new(
            Entry::Permutation { offset: 0 },
            3,
        ));
        assert_eq!(
            permutation_var.degree_multiple(),
            1,
            "Permutation variable should have degree 1"
        );

        let public_var =
            SymbolicExpression::Variable(SymbolicVariable::<BabyBear>::new(Entry::Public, 4));
        assert_eq!(
            public_var.degree_multiple(),
            0,
            "Public variable should have degree 0"
        );

        let challenge_var =
            SymbolicExpression::Variable(SymbolicVariable::<BabyBear>::new(Entry::Challenge, 5));
        assert_eq!(
            challenge_var.degree_multiple(),
            0,
            "Challenge variable should have degree 0"
        );

        let is_first_row = SymbolicExpression::<BabyBear>::IsFirstRow;
        assert_eq!(
            is_first_row.degree_multiple(),
            1,
            "IsFirstRow should have degree 1"
        );

        let is_last_row = SymbolicExpression::<BabyBear>::IsLastRow;
        assert_eq!(
            is_last_row.degree_multiple(),
            1,
            "IsLastRow should have degree 1"
        );

        let is_transition = SymbolicExpression::<BabyBear>::IsTransition;
        assert_eq!(
            is_transition.degree_multiple(),
            1,
            "IsTransition should have degree 1"
        );

        let add_expr = SymbolicExpression::<BabyBear>::Add {
            x: Arc::new(variable_expr.clone()),
            y: Arc::new(preprocessed_var.clone()),
            degree_multiple: 1,
        };
        assert_eq!(
            add_expr.degree_multiple(),
            1,
            "Addition should take max degree of inputs"
        );

        let sub_expr = SymbolicExpression::<BabyBear>::Sub {
            x: Arc::new(variable_expr.clone()),
            y: Arc::new(preprocessed_var.clone()),
            degree_multiple: 1,
        };
        assert_eq!(
            sub_expr.degree_multiple(),
            1,
            "Subtraction should take max degree of inputs"
        );

        let neg_expr = SymbolicExpression::<BabyBear>::Neg {
            x: Arc::new(variable_expr.clone()),
            degree_multiple: 1,
        };
        assert_eq!(
            neg_expr.degree_multiple(),
            1,
            "Negation should keep the degree"
        );

        let mul_expr = SymbolicExpression::<BabyBear>::Mul {
            x: Arc::new(variable_expr),
            y: Arc::new(preprocessed_var),
            degree_multiple: 2,
        };
        assert_eq!(
            mul_expr.degree_multiple(),
            2,
            "Multiplication should sum degrees"
        );
    }

    #[test]
    fn test_addition_of_constants() {
        let a = SymbolicExpression::Constant(BabyBear::new(3));
        let b = SymbolicExpression::Constant(BabyBear::new(4));
        let result = a + b;
        match result {
            SymbolicExpression::Constant(val) => assert_eq!(val, BabyBear::new(7)),
            _ => panic!("Addition of constants did not simplify correctly"),
        }
    }

    #[test]
    fn test_subtraction_of_constants() {
        let a = SymbolicExpression::Constant(BabyBear::new(10));
        let b = SymbolicExpression::Constant(BabyBear::new(4));
        let result = a - b;
        match result {
            SymbolicExpression::Constant(val) => assert_eq!(val, BabyBear::new(6)),
            _ => panic!("Subtraction of constants did not simplify correctly"),
        }
    }

    #[test]
    fn test_negation() {
        let a = SymbolicExpression::Constant(BabyBear::new(7));
        let result = -a;
        match result {
            SymbolicExpression::Constant(val) => {
                assert_eq!(val, BabyBear::NEG_ONE * BabyBear::new(7));
            }
            _ => panic!("Negation did not work correctly"),
        }
    }

    #[test]
    fn test_multiplication_of_constants() {
        let a = SymbolicExpression::Constant(BabyBear::new(3));
        let b = SymbolicExpression::Constant(BabyBear::new(5));
        let result = a * b;
        match result {
            SymbolicExpression::Constant(val) => assert_eq!(val, BabyBear::new(15)),
            _ => panic!("Multiplication of constants did not simplify correctly"),
        }
    }

    #[test]
    fn test_degree_multiple_for_addition() {
        let a = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            1,
        ));
        let b = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            2,
        ));
        let result = a + b;
        match result {
            SymbolicExpression::Add {
                degree_multiple,
                x,
                y,
            } => {
                assert_eq!(degree_multiple, 1);
                assert!(
                    matches!(*x, SymbolicExpression::Variable(ref v) if v.index == 1 && matches!(v.entry, Entry::Main { offset: 0 }))
                );
                assert!(
                    matches!(*y, SymbolicExpression::Variable(ref v) if v.index == 2 && matches!(v.entry, Entry::Main { offset: 0 }))
                );
            }
            _ => panic!("Addition did not create an Add expression"),
        }
    }

    #[test]
    fn test_degree_multiple_for_multiplication() {
        let a = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            1,
        ));
        let b = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            2,
        ));
        let result = a * b;

        match result {
            SymbolicExpression::Mul {
                degree_multiple,
                x,
                y,
            } => {
                assert_eq!(degree_multiple, 2, "Multiplication should sum degrees");

                assert!(
                    matches!(*x, SymbolicExpression::Variable(ref v)
                        if v.index == 1 && matches!(v.entry, Entry::Main { offset: 0 })
                    ),
                    "Left operand should match `a`"
                );

                assert!(
                    matches!(*y, SymbolicExpression::Variable(ref v)
                        if v.index == 2 && matches!(v.entry, Entry::Main { offset: 0 })
                    ),
                    "Right operand should match `b`"
                );
            }
            _ => panic!("Multiplication did not create a `Mul` expression"),
        }
    }

    #[test]
    fn test_sum_operator() {
        let expressions = vec![
            SymbolicExpression::Constant(BabyBear::new(2)),
            SymbolicExpression::Constant(BabyBear::new(3)),
            SymbolicExpression::Constant(BabyBear::new(5)),
        ];
        let result: SymbolicExpression<BabyBear> = expressions.into_iter().sum();
        match result {
            SymbolicExpression::Constant(val) => assert_eq!(val, BabyBear::new(10)),
            _ => panic!("Sum did not produce correct result"),
        }
    }

    #[test]
    fn test_product_operator() {
        let expressions = vec![
            SymbolicExpression::Constant(BabyBear::new(2)),
            SymbolicExpression::Constant(BabyBear::new(3)),
            SymbolicExpression::Constant(BabyBear::new(4)),
        ];
        let result: SymbolicExpression<BabyBear> = expressions.into_iter().product();
        match result {
            SymbolicExpression::Constant(val) => assert_eq!(val, BabyBear::new(24)),
            _ => panic!("Product did not produce correct result"),
        }
    }

    #[test]
    fn test_default_is_zero() {
        // Default should produce ZERO constant.
        let expr: SymbolicExpression<BabyBear> = Default::default();

        // Verify it matches the zero constant.
        assert!(matches!(
            expr,
            SymbolicExpression::Constant(c) if c == BabyBear::ZERO
        ));
    }

    #[test]
    fn test_ring_constants() {
        // ZERO is a Constant variant wrapping the field's zero element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::ZERO,
            SymbolicExpression::Constant(c) if c == BabyBear::ZERO
        ));

        // ONE is a Constant variant wrapping the field's one element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::ONE,
            SymbolicExpression::Constant(c) if c == BabyBear::ONE
        ));

        // TWO is a Constant variant wrapping the field's two element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::TWO,
            SymbolicExpression::Constant(c) if c == BabyBear::TWO
        ));

        // NEG_ONE is a Constant variant wrapping the field's -1 element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::NEG_ONE,
            SymbolicExpression::Constant(c) if c == BabyBear::NEG_ONE
        ));
    }

    #[test]
    fn test_from_symbolic_variable() {
        // Create a main trace variable at column index 3.
        let var = SymbolicVariable::<BabyBear>::new(Entry::Main { offset: 0 }, 3);

        // Convert to expression.
        let expr: SymbolicExpression<BabyBear> = var.into();

        // Verify the variable is preserved with correct entry and index.
        match expr {
            SymbolicExpression::Variable(v) => {
                assert!(matches!(v.entry, Entry::Main { offset: 0 }));
                assert_eq!(v.index, 3);
            }
            _ => panic!("Expected Variable variant"),
        }
    }

    #[test]
    fn test_from_field_element() {
        // Convert a field element directly to expression.
        let field_val = BabyBear::new(42);
        let expr: SymbolicExpression<BabyBear> = field_val.into();

        // Verify it becomes a Constant with the same value.
        assert!(matches!(
            expr,
            SymbolicExpression::Constant(c) if c == field_val
        ));
    }

    #[test]
    fn test_from_prime_subfield() {
        // Create expression from prime subfield element.
        let prime_subfield_val = <BabyBear as PrimeCharacteristicRing>::PrimeSubfield::new(7);
        let expr = SymbolicExpression::<BabyBear>::from_prime_subfield(prime_subfield_val);

        // Verify it produces a constant with the converted value.
        assert!(matches!(
            expr,
            SymbolicExpression::Constant(c) if c == BabyBear::new(7)
        ));
    }

    #[test]
    fn test_assign_operators() {
        // Test AddAssign with constants (should simplify).
        let mut expr = SymbolicExpression::Constant(BabyBear::new(5));
        expr += SymbolicExpression::Constant(BabyBear::new(3));
        assert!(matches!(
            expr,
            SymbolicExpression::Constant(c) if c == BabyBear::new(8)
        ));

        // Test SubAssign with constants (should simplify).
        let mut expr = SymbolicExpression::Constant(BabyBear::new(10));
        expr -= SymbolicExpression::Constant(BabyBear::new(4));
        assert!(matches!(
            expr,
            SymbolicExpression::Constant(c) if c == BabyBear::new(6)
        ));

        // Test MulAssign with constants (should simplify).
        let mut expr = SymbolicExpression::Constant(BabyBear::new(6));
        expr *= SymbolicExpression::Constant(BabyBear::new(7));
        assert!(matches!(
            expr,
            SymbolicExpression::Constant(c) if c == BabyBear::new(42)
        ));
    }

    #[test]
    fn test_subtraction_creates_sub_node() {
        // Create two trace variables.
        let a = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            0,
        ));
        let b = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            1,
        ));

        // Subtract them.
        let result = a - b;

        // Should create Sub node (not simplified).
        match result {
            SymbolicExpression::Sub {
                x,
                y,
                degree_multiple,
            } => {
                // Both operands have degree 1, so max is 1.
                assert_eq!(degree_multiple, 1);

                // Verify left operand is main trace variable at index 0, offset 0.
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpression::Variable(v)
                        if v.index == 0 && matches!(v.entry, Entry::Main { offset: 0 })
                ));

                // Verify right operand is main trace variable at index 1, offset 0.
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpression::Variable(v)
                        if v.index == 1 && matches!(v.entry, Entry::Main { offset: 0 })
                ));
            }
            _ => panic!("Expected Sub variant"),
        }
    }

    #[test]
    fn test_negation_creates_neg_node() {
        // Create a trace variable.
        let var = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            0,
        ));

        // Negate it.
        let result = -var;

        // Should create Neg node (not simplified).
        match result {
            SymbolicExpression::Neg { x, degree_multiple } => {
                // Degree is preserved from operand.
                assert_eq!(degree_multiple, 1);

                // Verify operand is main trace variable at index 0, offset 0.
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpression::Variable(v)
                        if v.index == 0 && matches!(v.entry, Entry::Main { offset: 0 })
                ));
            }
            _ => panic!("Expected Neg variant"),
        }
    }

    #[test]
    fn test_empty_sum_returns_zero() {
        // Sum of empty iterator should be additive identity.
        let empty: Vec<SymbolicExpression<BabyBear>> = vec![];
        let result: SymbolicExpression<BabyBear> = empty.into_iter().sum();

        assert!(matches!(
            result,
            SymbolicExpression::Constant(c) if c == BabyBear::ZERO
        ));
    }

    #[test]
    fn test_empty_product_returns_one() {
        // Product of empty iterator should be multiplicative identity.
        let empty: Vec<SymbolicExpression<BabyBear>> = vec![];
        let result: SymbolicExpression<BabyBear> = empty.into_iter().product();

        assert!(matches!(
            result,
            SymbolicExpression::Constant(c) if c == BabyBear::ONE
        ));
    }

    #[test]
    fn test_mixed_degree_addition() {
        // Constant has degree 0.
        let constant = SymbolicExpression::Constant(BabyBear::new(5));

        // Variable has degree 1.
        let var = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            0,
        ));

        // Add them: max(0, 1) = 1.
        let result = constant + var;

        match result {
            SymbolicExpression::Add {
                x,
                y,
                degree_multiple,
            } => {
                // Degree is max(0, 1) = 1.
                assert_eq!(degree_multiple, 1);

                // Verify left operand is the constant 5.
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpression::Constant(c) if *c == BabyBear::new(5)
                ));

                // Verify right operand is main trace variable at index 0, offset 0.
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpression::Variable(v)
                        if v.index == 0 && matches!(v.entry, Entry::Main { offset: 0 })
                ));
            }
            _ => panic!("Expected Add variant"),
        }
    }

    #[test]
    fn test_chained_multiplication_degree() {
        // Create three variables, each with degree 1.
        let a = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            0,
        ));
        let b = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            1,
        ));
        let c = SymbolicExpression::Variable::<BabyBear>(SymbolicVariable::new(
            Entry::Main { offset: 0 },
            2,
        ));

        // a * b has degree 1 + 1 = 2.
        let ab = a * b;
        assert_eq!(ab.degree_multiple(), 2);

        // (a * b) * c has degree 2 + 1 = 3.
        let abc = ab * c;
        assert_eq!(abc.degree_multiple(), 3);
    }
}
