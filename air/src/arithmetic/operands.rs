use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops;

use p3_field::extension::{
    BinomialExtensionField, Complex, CubicTrinomialExtensionField, QuinticTrinomialExtensionField,
};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

use super::expression::*;
use crate::{BaseEntry, BaseLeaf, ExtEntry, SymLeaf, SymbolicVariable, SymbolicVariableExt};

/// Base-field arithmetic expression, retaining scalar operands.
pub type ArithmeticExpression<F> = ArithmeticExpr<BaseLeaf<F>>;
/// Extension-field arithmetic expression, retaining base and extension operands.
pub type ArithmeticExpressionExt<F, EF> = ArithmeticExpr<ArithmeticExtLeaf<F, EF>>;

/// Extension leaves keep a lifted base expression in its original domain.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ArithmeticExtLeaf<F: Field, EF: ExtensionField<F>> {
    /// An expression evaluated in the base algebra before lifting.
    Base(ArithmeticExpression<F>),
    /// A permutation column, challenge, or permutation value.
    Variable(SymbolicVariableExt<F, EF>),
    /// A true extension-field constant.
    Constant(EF),
}
/// A base scalar is distinct from an expression that happens to be constant.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum BaseOperand<F: Field> {
    /// A coefficient supplied as a raw field value.
    Scalar(F),
    /// A shared expression, including explicitly lifted constants.
    Expr(Arc<ArithmeticExpression<F>>),
}
/// Operand domains available to an extension expression.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ExtOperand<F: Field, EF: ExtensionField<F>> {
    /// A raw base-field coefficient.
    BaseScalar(F),
    /// A raw extension-field coefficient.
    ExtensionScalar(EF),
    /// A shared base expression, evaluated in the base algebra.
    Base(Arc<ArithmeticExpression<F>>),
    /// A shared extension expression.
    Extension(Arc<ArithmeticExpressionExt<F, EF>>),
}
impl<F: Field> ArithmeticLeaf for BaseLeaf<F> {
    type LegacyLeaf = Self;
    fn lower_leaf(&self) -> Self::LegacyLeaf {
        self.clone()
    }
    type Value = F;
    type Operand = BaseOperand<F>;
    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);
    fn constant(&self) -> Option<F> {
        self.as_const().copied()
    }
    fn from_constant(v: F) -> Self {
        Self::Constant(v)
    }
    fn degree(&self, trace: Option<(usize, &[usize])>, transition: usize) -> usize {
        match trace {
            Some((n, p)) => self.poly_degree(n, p),
            None => self.degree_multiple_with_transition(transition),
        }
    }
}
impl<F: Field, EF: ExtensionField<F>> ArithmeticLeaf for ArithmeticExtLeaf<F, EF> {
    type LegacyLeaf = crate::ExtLeaf<F, EF>;
    fn lower_leaf(&self) -> Self::LegacyLeaf {
        match self {
            Self::Base(e) => crate::ExtLeaf::Base(e.lower()),
            Self::Variable(v) => crate::ExtLeaf::ExtVariable(*v),
            Self::Constant(c) => crate::ExtLeaf::ExtConstant(*c),
        }
    }
    type Value = EF;
    type Operand = ExtOperand<F, EF>;
    const ZERO: Self = Self::Base(ArithmeticExpression::ZERO);
    const ONE: Self = Self::Base(ArithmeticExpression::ONE);
    const TWO: Self = Self::Base(ArithmeticExpression::TWO);
    const NEG_ONE: Self = Self::Base(ArithmeticExpression::NEG_ONE);
    fn constant(&self) -> Option<EF> {
        match self {
            Self::Base(e) => e.constant().map(EF::from),
            Self::Constant(c) => Some(*c),
            _ => None,
        }
    }
    fn from_constant(v: EF) -> Self {
        Self::Constant(v)
    }
    fn degree(&self, trace: Option<(usize, &[usize])>, transition: usize) -> usize {
        match self {
            Self::Base(e) => match trace {
                Some((n, p)) => e.poly_degree(n, p),
                None => e.degree_multiple_with_transition(transition),
            },
            Self::Variable(v) => {
                trace.map_or_else(|| v.degree_multiple(), |(n, _)| v.poly_degree(n))
            }
            Self::Constant(_) => 0,
        }
    }
}
fn shared_degree<L: ArithmeticLeaf>(
    e: &Arc<ArithmeticExpr<L>>,
    trace: Option<(usize, &[usize])>,
    transition: usize,
    memo: &mut BTreeMap<usize, usize>,
) -> usize {
    if trace.is_none() && transition == 0 {
        return e.degree_multiple();
    }
    let key = Arc::as_ptr(e) as usize;
    if let Some(d) = memo.get(&key) {
        return *d;
    }
    let d = e.degree_inner(trace, transition, memo);
    memo.insert(key, d);
    d
}
impl<F: Field> ArithmeticOperand for BaseOperand<F> {
    type Leaf = BaseLeaf<F>;
    fn lower(
        &self,
        memo: &mut BTreeMap<usize, crate::SymbolicExpression<F>>,
    ) -> crate::SymbolicExpression<F> {
        match self {
            Self::Scalar(c) => (*c).into(),
            Self::Expr(e) => lower_shared(e, memo),
        }
    }
    fn expression(self) -> ArithmeticExpression<F> {
        match self {
            Self::Scalar(c) => c.into(),
            Self::Expr(e) => (*e).clone(),
        }
    }
    fn from_expression(e: ArithmeticExpression<F>) -> Self {
        Self::Expr(Arc::new(e))
    }
    fn constant(&self) -> Option<F> {
        match self {
            Self::Scalar(c) => Some(*c),
            Self::Expr(e) => e.constant(),
        }
    }
    fn degree(
        &self,
        t: Option<(usize, &[usize])>,
        s: usize,
        m: &mut BTreeMap<usize, usize>,
    ) -> usize {
        match self {
            Self::Scalar(_) => 0,
            Self::Expr(e) => shared_degree(e, t, s, m),
        }
    }
}
impl<F: Field, EF: ExtensionField<F>> ArithmeticOperand for ExtOperand<F, EF> {
    type Leaf = ArithmeticExtLeaf<F, EF>;
    fn lower(
        &self,
        memo: &mut BTreeMap<usize, crate::SymbolicExpressionExt<F, EF>>,
    ) -> crate::SymbolicExpressionExt<F, EF> {
        match self {
            Self::BaseScalar(c) => (*c).into(),
            Self::ExtensionScalar(c) => crate::SymbolicExpr::Leaf(crate::ExtLeaf::ExtConstant(*c)),
            Self::Base(e) => e.lower().into(),
            Self::Extension(e) => lower_shared(e, memo),
        }
    }
    fn expression(self) -> ArithmeticExpressionExt<F, EF> {
        match self {
            Self::BaseScalar(c) => c.into(),
            Self::ExtensionScalar(c) => ArithmeticExpressionExt::from_extension(c),
            Self::Base(e) => (*e).clone().into(),
            Self::Extension(e) => (*e).clone(),
        }
    }
    fn from_expression(e: ArithmeticExpressionExt<F, EF>) -> Self {
        Self::Extension(Arc::new(e))
    }
    fn constant(&self) -> Option<EF> {
        match self {
            Self::BaseScalar(c) => Some((*c).into()),
            Self::ExtensionScalar(c) => Some(*c),
            Self::Base(e) => e.constant().map(EF::from),
            Self::Extension(e) => e.constant(),
        }
    }
    fn degree(
        &self,
        t: Option<(usize, &[usize])>,
        s: usize,
        m: &mut BTreeMap<usize, usize>,
    ) -> usize {
        match self {
            Self::BaseScalar(_) | Self::ExtensionScalar(_) => 0,
            Self::Base(e) => shared_degree(e, t, s, m),
            Self::Extension(e) => shared_degree(e, t, s, m),
        }
    }
}
impl<F: Field> From<F> for ArithmeticExpression<F> {
    fn from(v: F) -> Self {
        Self::leaf(BaseLeaf::Constant(v), 0)
    }
}
impl<F: Field> From<ArithmeticVariable<F>> for ArithmeticExpression<F> {
    fn from(v: ArithmeticVariable<F>) -> Self {
        Self::leaf(BaseLeaf::Variable(v.0), v.0.degree_multiple())
    }
}
impl<F: Field> From<F> for BaseOperand<F> {
    fn from(v: F) -> Self {
        Self::Scalar(v)
    }
}
impl<F: Field> From<ArithmeticExpression<F>> for BaseOperand<F> {
    fn from(v: ArithmeticExpression<F>) -> Self {
        Self::Expr(Arc::new(v))
    }
}
impl<F: Field> From<ArithmeticVariable<F>> for BaseOperand<F> {
    fn from(v: ArithmeticVariable<F>) -> Self {
        ArithmeticExpression::from(v).into()
    }
}
impl<F: Field, EF: ExtensionField<F>> From<F> for ArithmeticExpressionExt<F, EF> {
    fn from(v: F) -> Self {
        Self::leaf(ArithmeticExtLeaf::Base(v.into()), 0)
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticExpression<F>>
    for ArithmeticExpressionExt<F, EF>
{
    fn from(v: ArithmeticExpression<F>) -> Self {
        let d = v.degree_multiple();
        Self::leaf(ArithmeticExtLeaf::Base(v), d)
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticVariable<F>>
    for ArithmeticExpressionExt<F, EF>
{
    fn from(v: ArithmeticVariable<F>) -> Self {
        ArithmeticExpression::from(v).into()
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticVariableExt<F, EF>>
    for ArithmeticExpressionExt<F, EF>
{
    fn from(v: ArithmeticVariableExt<F, EF>) -> Self {
        Self::leaf(ArithmeticExtLeaf::Variable(v.0), v.0.degree_multiple())
    }
}
impl<F: Field, EF: ExtensionField<F>> From<F> for ExtOperand<F, EF> {
    fn from(v: F) -> Self {
        Self::BaseScalar(v)
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticExpression<F>> for ExtOperand<F, EF> {
    fn from(v: ArithmeticExpression<F>) -> Self {
        Self::Base(Arc::new(v))
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticExpressionExt<F, EF>> for ExtOperand<F, EF> {
    fn from(v: ArithmeticExpressionExt<F, EF>) -> Self {
        Self::Extension(Arc::new(v))
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticVariable<F>> for ExtOperand<F, EF> {
    fn from(v: ArithmeticVariable<F>) -> Self {
        ArithmeticExpression::from(v).into()
    }
}
impl<F: Field, EF: ExtensionField<F>> From<ArithmeticVariableExt<F, EF>> for ExtOperand<F, EF> {
    fn from(v: ArithmeticVariableExt<F, EF>) -> Self {
        ArithmeticExpressionExt::from(v).into()
    }
}
impl<F: Field, EF: ExtensionField<F>> ArithmeticExpressionExt<F, EF> {
    /// Construct a true extension constant without a conflicting generic `From<EF>`.
    pub const fn from_extension(v: EF) -> Self {
        Self::leaf(ArithmeticExtLeaf::Constant(v), 0)
    }
}
/// Copyable base variable whose operators produce arithmetic expressions.
#[derive(Clone, Copy, Debug)]
pub struct ArithmeticVariable<F>(pub SymbolicVariable<F>);
impl<F> ArithmeticVariable<F> {
    pub const fn new(entry: BaseEntry, index: usize) -> Self {
        Self(SymbolicVariable::new(entry, index))
    }
}
/// Copyable extension variable whose operators produce arithmetic expressions.
#[derive(Clone, Copy, Debug)]
pub struct ArithmeticVariableExt<F, EF>(pub SymbolicVariableExt<F, EF>);
impl<F, EF> ArithmeticVariableExt<F, EF> {
    pub const fn new(entry: ExtEntry, index: usize) -> Self {
        Self(SymbolicVariableExt::new(entry, index))
    }
}
impl<F: Field, T: Into<BaseOperand<F>>> ops::Add<T> for ArithmeticVariable<F> {
    type Output = ArithmeticExpression<F>;
    fn add(self, rhs: T) -> Self::Output {
        Self::Output::from(self) + rhs
    }
}
impl<F: Field, T: Into<BaseOperand<F>>> ops::Sub<T> for ArithmeticVariable<F> {
    type Output = ArithmeticExpression<F>;
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from(self) - rhs
    }
}
impl<F: Field, T: Into<BaseOperand<F>>> ops::Mul<T> for ArithmeticVariable<F> {
    type Output = ArithmeticExpression<F>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from(self) * rhs
    }
}
impl<F: Field, EF: ExtensionField<F>, T: Into<ExtOperand<F, EF>>> ops::Add<T>
    for ArithmeticVariableExt<F, EF>
{
    type Output = ArithmeticExpressionExt<F, EF>;
    fn add(self, rhs: T) -> Self::Output {
        Self::Output::from(self) + rhs
    }
}
impl<F: Field, EF: ExtensionField<F>, T: Into<ExtOperand<F, EF>>> ops::Sub<T>
    for ArithmeticVariableExt<F, EF>
{
    type Output = ArithmeticExpressionExt<F, EF>;
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from(self) - rhs
    }
}
impl<F: Field, EF: ExtensionField<F>, T: Into<ExtOperand<F, EF>>> ops::Mul<T>
    for ArithmeticVariableExt<F, EF>
{
    type Output = ArithmeticExpressionExt<F, EF>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from(self) * rhs
    }
}
impl<F: Field> Algebra<F> for ArithmeticExpression<F> {
    fn mixed_dot_product<const N: usize>(a: &[Self; N], b: &[F; N]) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[F]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(Into::into)
                .collect::<Vec<BaseOperand<F>>>(),
            true,
        )
    }
}
impl<F: Field> Algebra<ArithmeticVariable<F>> for ArithmeticExpression<F> {
    fn mixed_dot_product<const N: usize>(a: &[Self; N], b: &[ArithmeticVariable<F>; N]) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[ArithmeticVariable<F>]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(Into::into)
                .collect::<Vec<BaseOperand<F>>>(),
            true,
        )
    }
}
impl<F: Field, EF: ExtensionField<F>> Algebra<F> for ArithmeticExpressionExt<F, EF> {
    fn mixed_dot_product<const N: usize>(a: &[Self; N], b: &[F; N]) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[F]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(Into::into)
                .collect::<Vec<ExtOperand<F, EF>>>(),
            true,
        )
    }
}
impl<F: Field, EF: ExtensionField<F>> Algebra<ArithmeticExpression<F>>
    for ArithmeticExpressionExt<F, EF>
{
    fn mixed_dot_product<const N: usize>(a: &[Self; N], b: &[ArithmeticExpression<F>; N]) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[ArithmeticExpression<F>]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(Into::into)
                .collect::<Vec<ExtOperand<F, EF>>>(),
            true,
        )
    }
}
impl<F: Field, EF: ExtensionField<F>> Algebra<ArithmeticVariable<F>>
    for ArithmeticExpressionExt<F, EF>
{
    fn mixed_dot_product<const N: usize>(a: &[Self; N], b: &[ArithmeticVariable<F>; N]) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[ArithmeticVariable<F>]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(Into::into)
                .collect::<Vec<ExtOperand<F, EF>>>(),
            true,
        )
    }
}
impl<F: Field, EF: ExtensionField<F>> Algebra<ArithmeticVariableExt<F, EF>>
    for ArithmeticExpressionExt<F, EF>
{
    fn mixed_dot_product<const N: usize>(
        a: &[Self; N],
        b: &[ArithmeticVariableExt<F, EF>; N],
    ) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[ArithmeticVariableExt<F, EF>]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(Into::into)
                .collect::<Vec<ExtOperand<F, EF>>>(),
            true,
        )
    }
}
impl<F: Field, const D: usize> From<BinomialExtensionField<F, D>>
    for ArithmeticExpressionExt<F, BinomialExtensionField<F, D>>
where
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
    fn from(v: BinomialExtensionField<F, D>) -> Self {
        Self::from_extension(v)
    }
}
impl<F: Field, const D: usize> From<BinomialExtensionField<F, D>>
    for ExtOperand<F, BinomialExtensionField<F, D>>
where
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
    fn from(v: BinomialExtensionField<F, D>) -> Self {
        Self::ExtensionScalar(v)
    }
}
impl<F: Field, const D: usize> Algebra<BinomialExtensionField<F, D>>
    for ArithmeticExpressionExt<F, BinomialExtensionField<F, D>>
where
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
    fn mixed_dot_product<const N: usize>(
        a: &[Self; N],
        b: &[BinomialExtensionField<F, D>; N],
    ) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[BinomialExtensionField<F, D>]) -> Self {
        Self::dot(
            a,
            &b.iter().copied().map(Into::into).collect::<Vec<_>>(),
            true,
        )
    }
}
impl<F: Field> From<CubicTrinomialExtensionField<F>>
    for ArithmeticExpressionExt<F, CubicTrinomialExtensionField<F>>
where
    CubicTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn from(v: CubicTrinomialExtensionField<F>) -> Self {
        Self::from_extension(v)
    }
}
impl<F: Field> From<CubicTrinomialExtensionField<F>>
    for ExtOperand<F, CubicTrinomialExtensionField<F>>
where
    CubicTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn from(v: CubicTrinomialExtensionField<F>) -> Self {
        Self::ExtensionScalar(v)
    }
}
impl<F: Field> Algebra<CubicTrinomialExtensionField<F>>
    for ArithmeticExpressionExt<F, CubicTrinomialExtensionField<F>>
where
    CubicTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn mixed_dot_product<const N: usize>(
        a: &[Self; N],
        b: &[CubicTrinomialExtensionField<F>; N],
    ) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[CubicTrinomialExtensionField<F>]) -> Self {
        Self::dot(
            a,
            &b.iter().copied().map(Into::into).collect::<Vec<_>>(),
            true,
        )
    }
}
impl<F: Field> From<QuinticTrinomialExtensionField<F>>
    for ArithmeticExpressionExt<F, QuinticTrinomialExtensionField<F>>
where
    QuinticTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn from(v: QuinticTrinomialExtensionField<F>) -> Self {
        Self::from_extension(v)
    }
}
impl<F: Field> From<QuinticTrinomialExtensionField<F>>
    for ExtOperand<F, QuinticTrinomialExtensionField<F>>
where
    QuinticTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn from(v: QuinticTrinomialExtensionField<F>) -> Self {
        Self::ExtensionScalar(v)
    }
}
impl<F: Field> Algebra<QuinticTrinomialExtensionField<F>>
    for ArithmeticExpressionExt<F, QuinticTrinomialExtensionField<F>>
where
    QuinticTrinomialExtensionField<F>: ExtensionField<F>,
{
    fn mixed_dot_product<const N: usize>(
        a: &[Self; N],
        b: &[QuinticTrinomialExtensionField<F>; N],
    ) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[QuinticTrinomialExtensionField<F>]) -> Self {
        Self::dot(
            a,
            &b.iter().copied().map(Into::into).collect::<Vec<_>>(),
            true,
        )
    }
}
impl<F: Field> From<BinomialExtensionField<Complex<F>, 2>>
    for ArithmeticExpressionExt<F, BinomialExtensionField<Complex<F>, 2>>
where
    BinomialExtensionField<Complex<F>, 2>: ExtensionField<F>,
{
    fn from(v: BinomialExtensionField<Complex<F>, 2>) -> Self {
        Self::from_extension(v)
    }
}
impl<F: Field> From<BinomialExtensionField<Complex<F>, 2>>
    for ExtOperand<F, BinomialExtensionField<Complex<F>, 2>>
where
    BinomialExtensionField<Complex<F>, 2>: ExtensionField<F>,
{
    fn from(v: BinomialExtensionField<Complex<F>, 2>) -> Self {
        Self::ExtensionScalar(v)
    }
}
impl<F: Field> Algebra<BinomialExtensionField<Complex<F>, 2>>
    for ArithmeticExpressionExt<F, BinomialExtensionField<Complex<F>, 2>>
where
    BinomialExtensionField<Complex<F>, 2>: ExtensionField<F>,
{
    fn mixed_dot_product<const N: usize>(
        a: &[Self; N],
        b: &[BinomialExtensionField<Complex<F>, 2>; N],
    ) -> Self {
        Self::batched_linear_combination(a, b)
    }
    fn batched_linear_combination(a: &[Self], b: &[BinomialExtensionField<Complex<F>, 2>]) -> Self {
        Self::dot(
            a,
            &b.iter().copied().map(Into::into).collect::<Vec<_>>(),
            true,
        )
    }
}

fn lower_shared<L: ArithmeticLeaf>(
    e: &Arc<ArithmeticExpr<L>>,
    memo: &mut BTreeMap<usize, crate::SymbolicExpr<L::LegacyLeaf>>,
) -> crate::SymbolicExpr<L::LegacyLeaf> {
    let key = Arc::as_ptr(e) as usize;
    if let Some(v) = memo.get(&key) {
        return v.clone();
    }
    let v = e.lower_inner(memo);
    memo.insert(key, v.clone());
    v
}

impl<F: Field> super::expression::sealed::Leaf for BaseLeaf<F> {}
impl<F: Field, EF: ExtensionField<F>> super::expression::sealed::Leaf for ArithmeticExtLeaf<F, EF> {}
impl<F: Field> super::expression::sealed::Operand for BaseOperand<F> {}
impl<F: Field, EF: ExtensionField<F>> super::expression::sealed::Operand for ExtOperand<F, EF> {}
