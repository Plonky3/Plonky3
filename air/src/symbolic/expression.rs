use p3_field::{Algebra, ExtensionField, Field, InjectiveMonomial};

use crate::symbolic::variable::{BaseEntry, SymbolicVariable};
use crate::symbolic::{SymLeaf, SymbolicExpr};
use crate::{AirBuilder, WindowAccess};

/// Leaf nodes for base-field symbolic expressions.
///
/// These represent the atomic building blocks of AIR constraint expressions:
/// trace column references, selectors, and field constants.
#[derive(Clone, Debug)]
pub enum BaseLeaf<F> {
    /// A reference to a trace column or public input.
    Variable(SymbolicVariable<F>),

    /// Selector evaluating to a non-zero value only on the first row.
    IsFirstRow,

    /// Selector evaluating to a non-zero value only on the last row.
    IsLastRow,

    /// Selector evaluating to zero only on the last row.
    IsTransition,

    /// A constant field element.
    Constant(F),
}

/// A symbolic expression tree for base-field AIR constraints.
///
/// This is a type alias for the generic [`SymbolicExpr`] parameterized with
/// base-field [`BaseLeaf`] nodes.
pub type SymbolicExpression<F> = SymbolicExpr<BaseLeaf<F>>;

impl<F: Field> SymLeaf for BaseLeaf<F> {
    type F = F;

    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);

    fn degree_multiple(&self) -> usize {
        match self {
            Self::Variable(v) => v.degree_multiple(),
            Self::IsFirstRow | Self::IsLastRow => 1,
            Self::IsTransition | Self::Constant(_) => 0,
        }
    }

    fn as_const(&self) -> Option<&F> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    fn from_const(c: F) -> Self {
        Self::Constant(c)
    }
}

impl<F: Field, EF: ExtensionField<F>> From<SymbolicVariable<F>> for SymbolicExpression<EF> {
    fn from(var: SymbolicVariable<F>) -> Self {
        Self::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
            var.entry, var.index,
        )))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<F> for SymbolicExpression<EF> {
    fn from(f: F) -> Self {
        Self::Leaf(BaseLeaf::Constant(f.into()))
    }
}

impl<F: Field> SymbolicExpression<F> {
    /// Evaluate this symbolic expression against a concrete [`AirBuilder`].
    ///
    /// # Overview
    ///
    /// - Walk the expression tree top-down.
    /// - Replace each leaf with the builder's concrete value.
    /// - Recurse into arithmetic nodes; combine in the builder's algebra.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///     leaf   → builder lookup (main / preprocessed / public / periodic / selector / constant)
    ///     x + y  → resolve(x) + resolve(y)
    ///     x - y  → resolve(x) - resolve(y)
    ///     x * y  → resolve(x) * resolve(y)
    ///     -x     → -resolve(x)
    /// ```
    ///
    /// # Panics
    ///
    /// - Row offset other than 0 or 1.
    /// - Column index out of bounds.
    pub fn resolve<AB>(&self, builder: &AB) -> AB::Expr
    where
        AB: AirBuilder<F = F>,
    {
        match self {
            Self::Leaf(leaf) => match leaf {
                BaseLeaf::Variable(v) => match v.entry {
                    // Main trace: offset 0 = current row, offset 1 = next row.
                    // Symbolic builders only emit two-row windows.
                    BaseEntry::Main { offset } => {
                        let main = builder.main();
                        match offset {
                            0 => main
                                .current(v.index)
                                .expect("main column index out of bounds")
                                .into(),
                            1 => main
                                .next(v.index)
                                .expect("main column index out of bounds")
                                .into(),
                            _ => panic!("expressions cannot span more than two rows"),
                        }
                    }
                    // Preprocessed trace: same shape, commitment-free trace.
                    BaseEntry::Preprocessed { offset } => {
                        let prep = builder.preprocessed();
                        match offset {
                            0 => prep
                                .current(v.index)
                                .expect("preprocessed column index out of bounds")
                                .into(),
                            1 => prep
                                .next(v.index)
                                .expect("preprocessed column index out of bounds")
                                .into(),
                            _ => panic!("expressions cannot span more than two rows"),
                        }
                    }
                    // Public input: direct slice lookup.
                    BaseEntry::Public => builder.public_values()[v.index].into(),
                    // Periodic column at the current row.
                    // Empty default slice → out-of-bounds panic on stray emissions.
                    BaseEntry::Periodic => builder.periodic_values()[v.index].into(),
                },
                // Boundary and transition selectors come straight from the builder.
                BaseLeaf::IsFirstRow => builder.is_first_row(),
                BaseLeaf::IsLastRow => builder.is_last_row(),
                BaseLeaf::IsTransition => builder.is_transition_window(2),
                // Lift the field constant into the builder's expression algebra.
                BaseLeaf::Constant(c) => AB::Expr::from(*c),
            },
            // Arithmetic: recurse on operands, combine in the builder's algebra.
            Self::Add { x, y, .. } => x.resolve(builder) + y.resolve(builder),
            Self::Sub { x, y, .. } => x.resolve(builder) - y.resolve(builder),
            Self::Neg { x, .. } => -x.resolve(builder),
            Self::Mul { x, y, .. } => x.resolve(builder) * y.resolve(builder),
        }
    }
}

impl<F: Field> Algebra<F> for SymbolicExpression<F> {}

impl<F: Field> Algebra<SymbolicVariable<F>> for SymbolicExpression<F> {}

// Note we cannot implement PermutationMonomial due to the degree_multiple part which makes
// operations non invertible.
impl<F: Field + InjectiveMonomial<N>, const N: u64> InjectiveMonomial<N> for SymbolicExpression<F> {}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;
    use crate::symbolic::BaseEntry;

    #[test]
    fn test_symbolic_expression_degree_multiple() {
        let constant_expr =
            SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::new(5)));
        assert_eq!(
            constant_expr.degree_multiple(),
            0,
            "Constant should have degree 0"
        );

        let variable_expr = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));
        assert_eq!(
            variable_expr.degree_multiple(),
            1,
            "Main variable should have degree 1"
        );

        let preprocessed_var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
            BaseEntry::Preprocessed { offset: 0 },
            2,
        )));
        assert_eq!(
            preprocessed_var.degree_multiple(),
            1,
            "Preprocessed variable should have degree 1"
        );

        let public_var = SymbolicExpression::Leaf(BaseLeaf::Variable(
            SymbolicVariable::<BabyBear>::new(BaseEntry::Public, 4),
        ));
        assert_eq!(
            public_var.degree_multiple(),
            0,
            "Public variable should have degree 0"
        );

        let is_first_row = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::IsFirstRow);
        assert_eq!(
            is_first_row.degree_multiple(),
            1,
            "IsFirstRow should have degree 1"
        );

        let is_last_row = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::IsLastRow);
        assert_eq!(
            is_last_row.degree_multiple(),
            1,
            "IsLastRow should have degree 1"
        );

        let is_transition = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::IsTransition);
        assert_eq!(
            is_transition.degree_multiple(),
            0,
            "IsTransition should have degree 0"
        );

        let add_expr = SymbolicExpr::<BaseLeaf<BabyBear>>::Add {
            x: Arc::new(variable_expr.clone()),
            y: Arc::new(preprocessed_var.clone()),
            degree_multiple: 1,
        };
        assert_eq!(
            add_expr.degree_multiple(),
            1,
            "Addition should take max degree of inputs"
        );

        let sub_expr = SymbolicExpr::<BaseLeaf<BabyBear>>::Sub {
            x: Arc::new(variable_expr.clone()),
            y: Arc::new(preprocessed_var.clone()),
            degree_multiple: 1,
        };
        assert_eq!(
            sub_expr.degree_multiple(),
            1,
            "Subtraction should take max degree of inputs"
        );

        let neg_expr = SymbolicExpr::<BaseLeaf<BabyBear>>::Neg {
            x: Arc::new(variable_expr.clone()),
            degree_multiple: 1,
        };
        assert_eq!(
            neg_expr.degree_multiple(),
            1,
            "Negation should keep the degree"
        );

        let mul_expr = SymbolicExpr::<BaseLeaf<BabyBear>>::Mul {
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
        let a = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(3)));
        let b = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(4)));
        let result = a + b;
        match result {
            SymbolicExpr::Leaf(BaseLeaf::Constant(val)) => assert_eq!(val, BabyBear::new(7)),
            _ => panic!("Addition of constants did not simplify correctly"),
        }
    }

    #[test]
    fn test_subtraction_of_constants() {
        let a = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(10)));
        let b = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(4)));
        let result = a - b;
        match result {
            SymbolicExpr::Leaf(BaseLeaf::Constant(val)) => assert_eq!(val, BabyBear::new(6)),
            _ => panic!("Subtraction of constants did not simplify correctly"),
        }
    }

    #[test]
    fn test_negation() {
        let a = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(7)));
        let result = -a;
        match result {
            SymbolicExpr::Leaf(BaseLeaf::Constant(val)) => {
                assert_eq!(val, BabyBear::NEG_ONE * BabyBear::new(7));
            }
            _ => panic!("Negation did not work correctly"),
        }
    }

    #[test]
    fn test_multiplication_of_constants() {
        let a = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(3)));
        let b = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(5)));
        let result = a * b;
        match result {
            SymbolicExpr::Leaf(BaseLeaf::Constant(val)) => assert_eq!(val, BabyBear::new(15)),
            _ => panic!("Multiplication of constants did not simplify correctly"),
        }
    }

    #[test]
    fn test_degree_multiple_for_addition() {
        let a = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));
        let b = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            2,
        )));
        let result = a + b;
        match result {
            SymbolicExpr::Add {
                degree_multiple,
                x,
                y,
            } => {
                assert_eq!(degree_multiple, 1);
                assert!(
                    matches!(&*x, SymbolicExpr::Leaf(BaseLeaf::Variable(v)) if v.index == 1 && matches!(v.entry, BaseEntry::Main { offset: 0 }))
                );
                assert!(
                    matches!(&*y, SymbolicExpr::Leaf(BaseLeaf::Variable(v)) if v.index == 2 && matches!(v.entry, BaseEntry::Main { offset: 0 }))
                );
            }
            _ => panic!("Addition did not create an Add expression"),
        }
    }

    #[test]
    fn test_degree_multiple_for_multiplication() {
        let a = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));
        let b = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            2,
        )));
        let result = a * b;

        match result {
            SymbolicExpr::Mul {
                degree_multiple,
                x,
                y,
            } => {
                assert_eq!(degree_multiple, 2, "Multiplication should sum degrees");

                assert!(
                    matches!(&*x, SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 1 && matches!(v.entry, BaseEntry::Main { offset: 0 })
                    ),
                    "Left operand should match `a`"
                );

                assert!(
                    matches!(&*y, SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 2 && matches!(v.entry, BaseEntry::Main { offset: 0 })
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
            SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(2))),
            SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(3))),
            SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(5))),
        ];
        let result: SymbolicExpression<BabyBear> = expressions.into_iter().sum();
        match result {
            SymbolicExpr::Leaf(BaseLeaf::Constant(val)) => assert_eq!(val, BabyBear::new(10)),
            _ => panic!("Sum did not produce correct result"),
        }
    }

    #[test]
    fn test_product_operator() {
        let expressions = vec![
            SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(2))),
            SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(3))),
            SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(4))),
        ];
        let result: SymbolicExpression<BabyBear> = expressions.into_iter().product();
        match result {
            SymbolicExpr::Leaf(BaseLeaf::Constant(val)) => assert_eq!(val, BabyBear::new(24)),
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
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ZERO
        ));
    }

    #[test]
    fn test_ring_constants() {
        // ZERO is a Constant variant wrapping the field's zero element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::ZERO,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ZERO
        ));
        // ONE is a Constant variant wrapping the field's one element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::ONE,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ONE
        ));
        // TWO is a Constant variant wrapping the field's two element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::TWO,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::TWO
        ));
        // NEG_ONE is a Constant variant wrapping the field's -1 element.
        assert!(matches!(
            SymbolicExpression::<BabyBear>::NEG_ONE,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::NEG_ONE
        ));
    }

    #[test]
    fn test_from_symbolic_variable() {
        // Create a main trace variable at column index 3.
        let var = SymbolicVariable::<BabyBear>::new(BaseEntry::Main { offset: 0 }, 3);
        // Convert to expression.
        let expr: SymbolicExpression<BabyBear> = var.into();
        // Verify the variable is preserved with correct entry and index.
        match expr {
            SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => {
                assert!(matches!(v.entry, BaseEntry::Main { offset: 0 }));
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
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == field_val
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
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::new(7)
        ));
    }

    #[test]
    fn test_assign_operators() {
        // Test AddAssign with constants (should simplify).
        let mut expr = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(5)));
        expr += SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(3)));
        assert!(matches!(
            expr,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::new(8)
        ));

        // Test SubAssign with constants (should simplify).
        let mut expr = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(10)));
        expr -= SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(4)));
        assert!(matches!(
            expr,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::new(6)
        ));

        // Test MulAssign with constants (should simplify).
        let mut expr = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(6)));
        expr *= SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(7)));
        assert!(matches!(
            expr,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::new(42)
        ));
    }

    #[test]
    fn test_subtraction_creates_sub_node() {
        // Create two trace variables.
        let a = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let b = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));

        // Subtract them.
        let result = a - b;

        // Should create Sub node (not simplified).
        match result {
            SymbolicExpr::Sub {
                x,
                y,
                degree_multiple,
            } => {
                // Both operands have degree 1, so max is 1.
                assert_eq!(degree_multiple, 1);

                // Verify left operand is main trace variable at index 0, offset 0.
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && matches!(v.entry, BaseEntry::Main { offset: 0 })
                ));

                // Verify right operand is main trace variable at index 1, offset 0.
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 1 && matches!(v.entry, BaseEntry::Main { offset: 0 })
                ));
            }
            _ => panic!("Expected Sub variant"),
        }
    }

    #[test]
    fn test_negation_creates_neg_node() {
        // Create a trace variable.
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));

        // Negate it.
        let result = -var;

        // Should create Neg node (not simplified).
        match result {
            SymbolicExpr::Neg { x, degree_multiple } => {
                // Degree is preserved from operand.
                assert_eq!(degree_multiple, 1);

                // Verify operand is main trace variable at index 0, offset 0.
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && matches!(v.entry, BaseEntry::Main { offset: 0 })
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
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ZERO
        ));
    }

    #[test]
    fn test_empty_product_returns_one() {
        // Product of empty iterator should be multiplicative identity.
        let empty: Vec<SymbolicExpression<BabyBear>> = vec![];
        let result: SymbolicExpression<BabyBear> = empty.into_iter().product();
        assert!(matches!(
            result,
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ONE
        ));
    }

    #[test]
    fn test_mixed_degree_addition() {
        // Constant has degree 0.
        let constant = SymbolicExpression::Leaf(BaseLeaf::Constant(BabyBear::new(5)));

        // Variable has degree 1.
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));

        // Add them: max(0, 1) = 1.
        let result = constant + var;

        match result {
            SymbolicExpr::Add {
                x,
                y,
                degree_multiple,
            } => {
                // Degree is max(0, 1) = 1.
                assert_eq!(degree_multiple, 1);

                // Verify left operand is the constant 5.
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if *c == BabyBear::new(5)
                ));

                // Verify right operand is main trace variable at index 0, offset 0.
                assert!(matches!(
                    y.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && matches!(v.entry, BaseEntry::Main { offset: 0 })
                ));
            }
            _ => panic!("Expected Add variant"),
        }
    }

    #[test]
    fn test_chained_multiplication_degree() {
        // Create three variables, each with degree 1.
        let a = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let b = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            1,
        )));
        let c = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            2,
        )));

        // a * b has degree 1 + 1 = 2.
        let ab = a * b;
        assert_eq!(ab.degree_multiple(), 2);

        // (a * b) * c has degree 2 + 1 = 3.
        let abc = ab * c;
        assert_eq!(abc.degree_multiple(), 3);
    }

    #[test]
    fn test_add_zero_identity_folding() {
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let zero = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::ZERO));

        // x + 0 should return x, not create an Add node.
        let result = var.clone() + zero.clone();
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Variable(_))),
            "x + 0 should fold to x"
        );

        // 0 + x should return x, not create an Add node.
        let result = zero + var;
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Variable(_))),
            "0 + x should fold to x"
        );
    }

    #[test]
    fn test_sub_zero_identity_folding() {
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let zero = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::ZERO));

        // x - 0 should return x, not create a Sub node.
        let result = var.clone() - zero.clone();
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Variable(_))),
            "x - 0 should fold to x"
        );

        // 0 - x should return -x, not create a Sub node.
        let result = zero - var;
        match result {
            SymbolicExpr::Neg { x, degree_multiple } => {
                assert_eq!(degree_multiple, 1);
                assert!(matches!(
                    x.as_ref(),
                    SymbolicExpr::Leaf(BaseLeaf::Variable(v))
                        if v.index == 0 && v.entry == BaseEntry::Main { offset: 0 }
                ));
            }
            _ => panic!("0 - x should fold to Neg(x)"),
        }
    }

    #[test]
    fn test_mul_zero_identity_folding() {
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let zero = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::ZERO));

        // x * 0 should return Constant(0), not create a Mul node.
        let result = var.clone() * zero.clone();
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ZERO),
            "x * 0 should fold to 0"
        );

        // 0 * x should return Constant(0), not create a Mul node.
        let result = zero * var;
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Constant(c)) if c == BabyBear::ZERO),
            "0 * x should fold to 0"
        );
    }

    #[test]
    fn test_mul_one_identity_folding() {
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let one = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::ONE));

        // x * 1 should return x, not create a Mul node.
        let result = var.clone() * one.clone();
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Variable(_))),
            "x * 1 should fold to x"
        );

        // 1 * x should return x, not create a Mul node.
        let result = one * var;
        assert!(
            matches!(result, SymbolicExpr::Leaf(BaseLeaf::Variable(_))),
            "1 * x should fold to x"
        );
    }

    #[test]
    fn test_identity_folding_preserves_degree() {
        let var = SymbolicExpression::Leaf(BaseLeaf::Variable(SymbolicVariable::<BabyBear>::new(
            BaseEntry::Main { offset: 0 },
            0,
        )));
        let zero = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::ZERO));
        let one = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::Constant(BabyBear::ONE));

        // x + 0 should preserve degree of x.
        let result = var.clone() + zero.clone();
        assert_eq!(result.degree_multiple(), 1);

        // x - 0 should preserve degree of x.
        let result = var.clone() - zero.clone();
        assert_eq!(result.degree_multiple(), 1);

        // 0 - x should preserve degree of x.
        let result = zero.clone() - var.clone();
        assert_eq!(result.degree_multiple(), 1);

        // x * 1 should preserve degree of x.
        let result = var.clone() * one;
        assert_eq!(result.degree_multiple(), 1);

        // x * 0 should have degree 0 (constant).
        let result = var * zero;
        assert_eq!(result.degree_multiple(), 0);
    }

    /// Minimal builder used to drive symbolic-expression resolution.
    ///
    /// Carries:
    /// - a 2-row main trace,
    /// - a public-value slice,
    /// - precomputed selector values for the current row,
    /// - a periodic-column row evaluated at the current step.
    struct ResolveTestBuilder {
        main: RowMajorMatrix<BabyBear>,
        public_values: Vec<BabyBear>,
        periodic_row: Vec<BabyBear>,
        is_first: BabyBear,
        is_last: BabyBear,
        is_transition: BabyBear,
    }

    impl AirBuilder for ResolveTestBuilder {
        type F = BabyBear;
        type Expr = BabyBear;
        type Var = BabyBear;
        type PreprocessedWindow = RowMajorMatrix<BabyBear>;
        type MainWindow = RowMajorMatrix<BabyBear>;
        type PublicVar = BabyBear;
        type PeriodicVar = BabyBear;

        fn main(&self) -> Self::MainWindow {
            self.main.clone()
        }

        fn preprocessed(&self) -> &Self::PreprocessedWindow {
            unimplemented!("no preprocessed columns in test builder")
        }

        fn is_first_row(&self) -> Self::Expr {
            self.is_first
        }

        fn is_last_row(&self) -> Self::Expr {
            self.is_last
        }

        fn is_transition_window(&self, _: usize) -> Self::Expr {
            self.is_transition
        }

        fn assert_zero<I: Into<Self::Expr>>(&mut self, _: I) {}

        fn public_values(&self) -> &[Self::PublicVar] {
            &self.public_values
        }

        fn periodic_values(&self) -> &[Self::PeriodicVar] {
            &self.periodic_row
        }
    }

    /// 2-row × 2-column trace, plus a 2-cell periodic row at the current step:
    ///
    /// ```text
    ///     main row 0 (current): [10, 20]
    ///     main row 1 (next):    [30, 40]
    ///     periodic_row (curr):  [7, 13]
    /// ```
    fn test_builder() -> ResolveTestBuilder {
        ResolveTestBuilder {
            main: RowMajorMatrix::new(
                vec![
                    BabyBear::new(10),
                    BabyBear::new(20), // current row
                    BabyBear::new(30),
                    BabyBear::new(40), // next row
                ],
                2, // width
            ),
            public_values: vec![BabyBear::new(99)],
            // Two periodic columns at the current row.
            // Distinct primes so any cross-stream mix-up is visible.
            periodic_row: vec![BabyBear::new(7), BabyBear::new(13)],
            is_first: BabyBear::ONE,
            is_last: BabyBear::ZERO,
            is_transition: BabyBear::ONE,
        }
    }

    #[test]
    fn resolve_main_current_row() {
        let b = test_builder();
        // Main column 0, offset 0 → current row value 10.
        let expr =
            SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0));
        assert_eq!(expr.resolve(&b), BabyBear::new(10));
    }

    #[test]
    fn resolve_main_next_row() {
        let b = test_builder();
        // Main column 1, offset 1 → next row value 40.
        let expr =
            SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1));
        assert_eq!(expr.resolve(&b), BabyBear::new(40));
    }

    #[test]
    fn resolve_public_value() {
        let b = test_builder();
        // Public value at index 0 → 99.
        let expr = SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Public, 0));
        assert_eq!(expr.resolve(&b), BabyBear::new(99));
    }

    #[test]
    fn resolve_constant() {
        let b = test_builder();
        let expr = SymbolicExpression::<BabyBear>::from(BabyBear::new(42));
        assert_eq!(expr.resolve(&b), BabyBear::new(42));
    }

    #[test]
    fn resolve_selectors() {
        let b = test_builder();

        let first = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::IsFirstRow);
        assert_eq!(first.resolve(&b), BabyBear::ONE, "is_first_row = 1");

        let last = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::IsLastRow);
        assert_eq!(last.resolve(&b), BabyBear::ZERO, "is_last_row = 0");

        let trans = SymbolicExpression::<BabyBear>::Leaf(BaseLeaf::IsTransition);
        assert_eq!(trans.resolve(&b), BabyBear::ONE, "is_transition = 1");
    }

    #[test]
    fn resolve_arithmetic() {
        let b = test_builder();

        // col0_curr = 10, col1_curr = 20.
        let col0 =
            SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0));
        let col1 =
            SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 1));

        // 10 + 20 = 30.
        let add = col0.clone() + col1.clone();
        assert_eq!(add.resolve(&b), BabyBear::new(30));

        // 10 - 20 = -10 (mod p).
        let sub = col0.clone() - col1.clone();
        assert_eq!(sub.resolve(&b), BabyBear::new(10) - BabyBear::new(20));

        // 10 * 20 = 200.
        let mul = col0.clone() * col1;
        assert_eq!(mul.resolve(&b), BabyBear::new(200));

        // -10 (mod p).
        let neg = -col0;
        assert_eq!(neg.resolve(&b), -BabyBear::new(10));
    }

    #[test]
    fn resolve_periodic_columns() {
        // Invariant: a periodic leaf reads from the builder's
        // periodic-value slice, in declared column order.
        //
        // Fixture:
        //
        //     periodic row (current step) : [7, 13]
        //     index 0 →  7
        //     index 1 → 13
        let b = test_builder();

        // Column 0 → 7.
        let p0 =
            SymbolicExpression::from(SymbolicVariable::<BabyBear>::new(BaseEntry::Periodic, 0));
        assert_eq!(p0.resolve(&b), BabyBear::new(7));

        // Column 1 → 13.
        let p1 =
            SymbolicExpression::from(SymbolicVariable::<BabyBear>::new(BaseEntry::Periodic, 1));
        assert_eq!(p1.resolve(&b), BabyBear::new(13));
    }

    #[test]
    fn resolve_periodic_combines_with_arithmetic() {
        // Invariant: periodic leaves compose under the same algebra
        // as main, public, and preprocessed leaves.
        //
        // Fixture:
        //
        //     periodic row : [7, 13]
        //     main row 0   : [10, 20]
        //
        //     expression   : main[0] * periodic[0] + periodic[1]
        //                  = 10 * 7 + 13
        //                  = 83
        let b = test_builder();

        let col0 =
            SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0));
        let p0 =
            SymbolicExpression::from(SymbolicVariable::<BabyBear>::new(BaseEntry::Periodic, 0));
        let p1 =
            SymbolicExpression::from(SymbolicVariable::<BabyBear>::new(BaseEntry::Periodic, 1));

        let expr = col0 * p0 + p1;
        assert_eq!(expr.resolve(&b), BabyBear::new(83));
    }
}
