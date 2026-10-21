use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

use super::expression::BinaryOperation::{self, Add, Mul, Sub};
use super::{
    ArithmeticExpression, ArithmeticExpressionExt, ArithmeticExtLeaf, ArithmeticNode, BaseOperand,
    ExtOperand,
};
use crate::{AirBuilder, ExtEntry, PermutationAirBuilder, SymbolicExpr, WindowAccess};

// Runtime arity cannot reproduce arbitrary const generic calls. Common small
// arities and SIMD-sized powers of two dispatch exactly; other arities use
// generic arithmetic (or the algebra's runtime batched LC helper).
fn sum<T: PrimeCharacteristicRing + Clone>(a: &[T]) -> T {
    match a.len() {
        0 => T::sum_array::<0>(a),
        1 => T::sum_array::<1>(a),
        2 => T::sum_array::<2>(a),
        3 => T::sum_array::<3>(a),
        4 => T::sum_array::<4>(a),
        5 => T::sum_array::<5>(a),
        6 => T::sum_array::<6>(a),
        7 => T::sum_array::<7>(a),
        8 => T::sum_array::<8>(a),
        16 => T::sum_array::<16>(a),
        32 => T::sum_array::<32>(a),
        64 => T::sum_array::<64>(a),
        _ => a.iter().cloned().sum(),
    }
}
fn dot<T: PrimeCharacteristicRing + Clone>(a: &[T], b: &[T]) -> T {
    match a.len() {
        0 => T::dot_product::<0>(a.try_into().unwrap(), b.try_into().unwrap()),
        1 => T::dot_product::<1>(a.try_into().unwrap(), b.try_into().unwrap()),
        2 => T::dot_product::<2>(a.try_into().unwrap(), b.try_into().unwrap()),
        3 => T::dot_product::<3>(a.try_into().unwrap(), b.try_into().unwrap()),
        4 => T::dot_product::<4>(a.try_into().unwrap(), b.try_into().unwrap()),
        5 => T::dot_product::<5>(a.try_into().unwrap(), b.try_into().unwrap()),
        6 => T::dot_product::<6>(a.try_into().unwrap(), b.try_into().unwrap()),
        7 => T::dot_product::<7>(a.try_into().unwrap(), b.try_into().unwrap()),
        8 => T::dot_product::<8>(a.try_into().unwrap(), b.try_into().unwrap()),
        16 => T::dot_product::<16>(a.try_into().unwrap(), b.try_into().unwrap()),
        32 => T::dot_product::<32>(a.try_into().unwrap(), b.try_into().unwrap()),
        64 => T::dot_product::<64>(a.try_into().unwrap(), b.try_into().unwrap()),
        _ => a
            .iter()
            .cloned()
            .zip(b.iter().cloned())
            .map(|(a, b)| a * b)
            .sum(),
    }
}
fn mixed<T: Algebra<C> + Clone, C: p3_field::Dup + Clone>(a: &[T], b: &[C]) -> T {
    match a.len() {
        0 => T::mixed_dot_product::<0>(a.try_into().unwrap(), b.try_into().unwrap()),
        1 => T::mixed_dot_product::<1>(a.try_into().unwrap(), b.try_into().unwrap()),
        2 => T::mixed_dot_product::<2>(a.try_into().unwrap(), b.try_into().unwrap()),
        3 => T::mixed_dot_product::<3>(a.try_into().unwrap(), b.try_into().unwrap()),
        4 => T::mixed_dot_product::<4>(a.try_into().unwrap(), b.try_into().unwrap()),
        5 => T::mixed_dot_product::<5>(a.try_into().unwrap(), b.try_into().unwrap()),
        6 => T::mixed_dot_product::<6>(a.try_into().unwrap(), b.try_into().unwrap()),
        7 => T::mixed_dot_product::<7>(a.try_into().unwrap(), b.try_into().unwrap()),
        8 => T::mixed_dot_product::<8>(a.try_into().unwrap(), b.try_into().unwrap()),
        16 => T::mixed_dot_product::<16>(a.try_into().unwrap(), b.try_into().unwrap()),
        32 => T::mixed_dot_product::<32>(a.try_into().unwrap(), b.try_into().unwrap()),
        64 => T::mixed_dot_product::<64>(a.try_into().unwrap(), b.try_into().unwrap()),
        _ => T::batched_linear_combination(a, b),
    }
}

#[derive(Clone)]
enum BaseValue<F, E> {
    Scalar(F),
    Expr(E),
}
impl<F: Field, E: Algebra<F>> BaseValue<F, E> {
    fn into_expr(self) -> E {
        match self {
            Self::Scalar(c) => c.into(),
            Self::Expr(e) => e,
        }
    }
}
fn base_binary<F: Field, E: Algebra<F>>(
    a: BaseValue<F, E>,
    b: BaseValue<F, E>,
    op: BinaryOperation,
) -> BaseValue<F, E> {
    use BaseValue::*;
    match (a, b) {
        (Scalar(a), Scalar(b)) => Scalar(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Expr(a), Scalar(b)) => Expr(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Scalar(a), Expr(b)) => Expr(match op {
            Add => b + a,
            Sub => -b + a,
            Mul => b * a,
        }),
        (Expr(a), Expr(b)) => Expr(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
    }
}
fn shared_base<F: Field, AB: AirBuilder<F = F>>(
    e: &Arc<ArithmeticExpression<F>>,
    b: &AB,
    memo: &mut BTreeMap<usize, AB::Expr>,
) -> AB::Expr {
    let key = Arc::as_ptr(e) as usize;
    if let Some(v) = memo.get(&key) {
        return v.clone();
    }
    let value = e.resolve_inner(b, memo);
    memo.insert(key, value.clone());
    value
}
fn base_operand<F: Field, AB: AirBuilder<F = F>>(
    o: &BaseOperand<F>,
    b: &AB,
    memo: &mut BTreeMap<usize, AB::Expr>,
) -> BaseValue<F, AB::Expr> {
    match o {
        BaseOperand::Scalar(c) => BaseValue::Scalar(*c),
        BaseOperand::Expr(e) => BaseValue::Expr(shared_base(e, b, memo)),
    }
}
impl<F: Field> ArithmeticExpression<F> {
    /// Evaluate using the target builder's arithmetic hooks and scalar operations.
    ///
    /// Sum/dot arities 0 through 8, 16, 32 and 64 dispatch to exact const
    /// specializations. Other arities use generic arithmetic; mixed dots use
    /// the target algebra's batched helper. Public nodes allow native emitters
    /// to recover exact const arities for all sizes. Shared operands are memoized.
    pub fn resolve<AB: AirBuilder<F = F>>(&self, builder: &AB) -> AB::Expr {
        self.resolve_inner(builder, &mut BTreeMap::new())
    }
    fn resolve_inner<AB: AirBuilder<F = F>>(
        &self,
        b: &AB,
        memo: &mut BTreeMap<usize, AB::Expr>,
    ) -> AB::Expr {
        let operand = |o: &BaseOperand<F>, m: &mut BTreeMap<usize, AB::Expr>| base_operand(o, b, m);
        match self.node() {
            ArithmeticNode::Leaf(l) => SymbolicExpr::Leaf(l.clone()).resolve(b),
            ArithmeticNode::Add(a, c) => {
                base_binary(operand(a, memo), operand(c, memo), Add).into_expr()
            }
            ArithmeticNode::Sub(a, c) => {
                base_binary(operand(a, memo), operand(c, memo), Sub).into_expr()
            }
            ArithmeticNode::Mul(a, c) => {
                base_binary(operand(a, memo), operand(c, memo), Mul).into_expr()
            }
            ArithmeticNode::Neg(a) => -operand(a, memo).into_expr(),
            ArithmeticNode::Square(a) => operand(a, memo).into_expr().square(),
            ArithmeticNode::Sum(a) => sum(&a
                .iter()
                .map(|a| operand(a, memo).into_expr())
                .collect::<Vec<_>>()),
            ArithmeticNode::Dot(p) | ArithmeticNode::MixedDot(p) => {
                let a: Vec<_> = p
                    .iter()
                    .map(|(a, _)| operand(a, memo).into_expr())
                    .collect();
                if matches!(self.node(), ArithmeticNode::MixedDot(_))
                    && p.iter().all(|(_, c)| matches!(c, BaseOperand::Scalar(_)))
                {
                    let c: Vec<_> = p
                        .iter()
                        .map(|(_, c)| match c {
                            BaseOperand::Scalar(c) => *c,
                            _ => unreachable!(),
                        })
                        .collect();
                    mixed(&a, &c)
                } else {
                    let c: Vec<_> = p
                        .iter()
                        .map(|(_, c)| operand(c, memo).into_expr())
                        .collect();
                    dot(&a, &c)
                }
            }
        }
    }
}

#[derive(Clone)]
enum ExtValue<F, E, EF, EE> {
    Base(BaseValue<F, E>),
    Scalar(EF),
    Expr(EE),
}
impl<F: Field, E: Algebra<F>, EF: ExtensionField<F>, EE: Algebra<E> + Algebra<EF> + Algebra<F>>
    ExtValue<F, E, EF, EE>
{
    fn into_expr(self) -> EE {
        match self {
            Self::Base(BaseValue::Scalar(c)) => c.into(),
            Self::Base(BaseValue::Expr(e)) => e.into(),
            Self::Scalar(c) => c.into(),
            Self::Expr(e) => e,
        }
    }
}
fn ext_binary<
    F: Field,
    E: Algebra<F>,
    EF: ExtensionField<F>,
    EE: Algebra<E> + Algebra<EF> + Algebra<F>,
>(
    a: ExtValue<F, E, EF, EE>,
    b: ExtValue<F, E, EF, EE>,
    op: BinaryOperation,
) -> ExtValue<F, E, EF, EE> {
    use ExtValue::*;
    match (a, b) {
        (Base(a), Base(b)) => Base(base_binary(a, b, op)),
        (Expr(a), Base(BaseValue::Scalar(b))) => Expr(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Base(BaseValue::Scalar(a)), Expr(b)) => Expr(match op {
            Add => b + a,
            Sub => -b + a,
            Mul => b * a,
        }),
        (Expr(a), Base(BaseValue::Expr(b))) => Expr(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Base(BaseValue::Expr(a)), Expr(b)) => Expr(match op {
            Add => b + a,
            Sub => -b + a,
            Mul => b * a,
        }),
        (Expr(a), Scalar(b)) => Expr(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Scalar(a), Expr(b)) => Expr(match op {
            Add => b + a,
            Sub => -b + a,
            Mul => b * a,
        }),
        (Base(BaseValue::Expr(a)), Scalar(b)) => {
            let b = EE::from(b);
            Expr(match op {
                Add => b + a,
                Sub => -b + a,
                Mul => b * a,
            })
        }
        (Scalar(a), Base(BaseValue::Expr(b))) => {
            let a = EE::from(a);
            Expr(match op {
                Add => a + b,
                Sub => a - b,
                Mul => a * b,
            })
        }
        (Scalar(a), Scalar(b)) => Scalar(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Scalar(a), Base(BaseValue::Scalar(b))) => Scalar(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
        (Base(BaseValue::Scalar(a)), Scalar(b)) => Scalar(match op {
            Add => b + a,
            Sub => -b + a,
            Mul => b * a,
        }),
        (Expr(a), Expr(b)) => Expr(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
        }),
    }
}
fn shared_ext<F: Field, EF: ExtensionField<F>, AB>(
    e: &Arc<ArithmeticExpressionExt<F, EF>>,
    b: &AB,
    base: &mut BTreeMap<usize, AB::Expr>,
    ext: &mut BTreeMap<usize, AB::ExprEF>,
) -> AB::ExprEF
where
    AB: PermutationAirBuilder<F = F, EF = EF>,
    AB::ExprEF: Algebra<F>,
{
    let key = Arc::as_ptr(e) as usize;
    if let Some(v) = ext.get(&key) {
        return v.clone();
    }
    let v = e.resolve_inner(b, base, ext);
    ext.insert(key, v.clone());
    v
}
fn ext_operand<F: Field, EF: ExtensionField<F>, AB>(
    o: &ExtOperand<F, EF>,
    b: &AB,
    base: &mut BTreeMap<usize, AB::Expr>,
    ext: &mut BTreeMap<usize, AB::ExprEF>,
) -> ExtValue<F, AB::Expr, EF, AB::ExprEF>
where
    AB: PermutationAirBuilder<F = F, EF = EF>,
    AB::ExprEF: Algebra<F>,
{
    match o {
        ExtOperand::BaseScalar(c) => ExtValue::Base(BaseValue::Scalar(*c)),
        ExtOperand::ExtensionScalar(c) => ExtValue::Scalar(*c),
        ExtOperand::Base(e) => ExtValue::Base(BaseValue::Expr(shared_base(e, b, base))),
        ExtOperand::Extension(e) => ExtValue::Expr(shared_ext(e, b, base, ext)),
    }
}
impl<F: Field, EF: ExtensionField<F>> ArithmeticExpressionExt<F, EF> {
    /// Evaluate with base/scalar arithmetic retained until extension arithmetic is needed.
    /// Uses the same bounded hook dispatch as [`ArithmeticExpression::resolve`].
    pub fn resolve<AB>(&self, builder: &AB) -> AB::ExprEF
    where
        AB: PermutationAirBuilder<F = F, EF = EF>,
        AB::ExprEF: Algebra<F>,
    {
        self.resolve_inner(builder, &mut BTreeMap::new(), &mut BTreeMap::new())
    }
    fn resolve_inner<AB>(
        &self,
        b: &AB,
        base: &mut BTreeMap<usize, AB::Expr>,
        ext: &mut BTreeMap<usize, AB::ExprEF>,
    ) -> AB::ExprEF
    where
        AB: PermutationAirBuilder<F = F, EF = EF>,
        AB::ExprEF: Algebra<F>,
    {
        match self.node() {
            ArithmeticNode::Leaf(l) => match l {
                ArithmeticExtLeaf::Base(e) => e.resolve_inner(b, base).into(),
                ArithmeticExtLeaf::Constant(c) => (*c).into(),
                ArithmeticExtLeaf::Variable(v) => match v.entry {
                    ExtEntry::Permutation { offset } => {
                        let p = b.permutation();
                        match offset {
                            0 => p.current(v.index),
                            1 => p.next(v.index),
                            _ => panic!("expressions cannot span more than two rows"),
                        }
                        .expect("permutation column index out of bounds")
                        .into()
                    }
                    ExtEntry::Challenge => b.permutation_randomness()[v.index].into(),
                    ExtEntry::PermutationValue => b.permutation_values()[v.index].clone().into(),
                },
            },
            ArithmeticNode::Add(a, c) => ext_binary(
                ext_operand(a, b, base, ext),
                ext_operand(c, b, base, ext),
                Add,
            )
            .into_expr(),
            ArithmeticNode::Sub(a, c) => ext_binary(
                ext_operand(a, b, base, ext),
                ext_operand(c, b, base, ext),
                Sub,
            )
            .into_expr(),
            ArithmeticNode::Mul(a, c) => ext_binary(
                ext_operand(a, b, base, ext),
                ext_operand(c, b, base, ext),
                Mul,
            )
            .into_expr(),
            ArithmeticNode::Neg(a) => -ext_operand(a, b, base, ext).into_expr(),
            ArithmeticNode::Square(a) => ext_operand(a, b, base, ext).into_expr().square(),
            ArithmeticNode::Sum(a) => sum(&a
                .iter()
                .map(|a| ext_operand(a, b, base, ext).into_expr())
                .collect::<Vec<_>>()),
            ArithmeticNode::Dot(p) | ArithmeticNode::MixedDot(p) => {
                let a: Vec<_> = p
                    .iter()
                    .map(|(a, _)| ext_operand(a, b, base, ext).into_expr())
                    .collect();
                if matches!(self.node(), ArithmeticNode::MixedDot(_)) {
                    if p.iter()
                        .all(|(_, c)| matches!(c, ExtOperand::BaseScalar(_)))
                    {
                        let c: Vec<_> = p
                            .iter()
                            .map(|(_, c)| match c {
                                ExtOperand::BaseScalar(c) => *c,
                                _ => unreachable!(),
                            })
                            .collect();
                        return mixed::<AB::ExprEF, F>(&a, &c);
                    }
                    if p.iter()
                        .all(|(_, c)| matches!(c, ExtOperand::ExtensionScalar(_)))
                    {
                        let c: Vec<_> = p
                            .iter()
                            .map(|(_, c)| match c {
                                ExtOperand::ExtensionScalar(c) => *c,
                                _ => unreachable!(),
                            })
                            .collect();
                        return mixed::<AB::ExprEF, EF>(&a, &c);
                    }
                    if p.iter().all(|(_, c)| matches!(c, ExtOperand::Base(_))) {
                        let c: Vec<_> = p
                            .iter()
                            .map(|(_, c)| match c {
                                ExtOperand::Base(e) => shared_base(e, b, base),
                                _ => unreachable!(),
                            })
                            .collect();
                        return mixed::<AB::ExprEF, AB::Expr>(&a, &c);
                    }
                }
                let c: Vec<_> = p
                    .iter()
                    .map(|(_, c)| ext_operand(c, b, base, ext).into_expr())
                    .collect();
                dot(&a, &c)
            }
        }
    }
}
