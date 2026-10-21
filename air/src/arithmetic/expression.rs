use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::iter::{Product, Sum};
use core::ops;

use BinaryOperation::{Add, Mul, Sub};
use p3_field::{Dup, Field, PrimeCharacteristicRing};

/// Read-only arithmetic structure. Mixed dots retain the coefficient operand class.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ArithmeticNode<L, O> {
    /// Atomic input, selector, or constant.
    Leaf(L),
    /// Addition.
    Add(O, O),
    /// Subtraction.
    Sub(O, O),
    /// Multiplication.
    Mul(O, O),
    /// Additive inverse.
    Neg(O),
    /// Dedicated squaring operation.
    Square(O),
    /// Ordered n-ary sum.
    Sum(Arc<[O]>),
    /// Dot product whose two input arrays had the same expression type.
    Dot(Arc<[(O, O)]>),
    /// Linear combination with coefficients supplied through `Algebra<C>`.
    MixedDot(Arc<[(O, O)]>),
}

/// Internal leaf contract; implemented by the supplied base and extension leaves.
#[doc(hidden)]
pub trait ArithmeticLeaf: sealed::Leaf + Clone + core::fmt::Debug {
    type Value: Field;
    type LegacyLeaf: crate::SymLeaf;
    fn lower_leaf(&self) -> Self::LegacyLeaf;
    type Operand: ArithmeticOperand<Leaf = Self>;
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;
    fn constant(&self) -> Option<Self::Value>;
    fn from_constant(value: Self::Value) -> Self;
    fn degree(&self, trace: Option<(usize, &[usize])>, transition: usize) -> usize;
}

/// Internal operand contract; operands preserve their original arithmetic domain.
#[doc(hidden)]
pub trait ArithmeticOperand:
    sealed::Operand + Clone + core::fmt::Debug + From<ArithmeticExpr<Self::Leaf>>
{
    type Leaf: ArithmeticLeaf<Operand = Self>;
    fn expression(self) -> ArithmeticExpr<Self::Leaf>;
    fn lower(
        &self,
        memo: &mut BTreeMap<usize, crate::SymbolicExpr<<Self::Leaf as ArithmeticLeaf>::LegacyLeaf>>,
    ) -> crate::SymbolicExpr<<Self::Leaf as ArithmeticLeaf>::LegacyLeaf>;
    fn from_expression(value: ArithmeticExpr<Self::Leaf>) -> Self;
    fn constant(&self) -> Option<<Self::Leaf as ArithmeticLeaf>::Value>;
    fn degree(
        &self,
        trace: Option<(usize, &[usize])>,
        transition: usize,
        memo: &mut BTreeMap<usize, usize>,
    ) -> usize;
}

/// Immutable arithmetic expression with conservative cached degree.
///
/// Constructors are private, so cached metadata cannot be invalidated. Degree
/// arithmetic panics on overflow instead of silently underestimating a bound.
#[derive(Clone, Debug)]
pub struct ArithmeticExpr<L: ArithmeticLeaf> {
    node: ArithmeticNode<L, L::Operand>,
    degree: usize,
}

pub(crate) const fn add_degree(a: usize, b: usize) -> usize {
    a.checked_add(b).expect("arithmetic degree overflow")
}

impl<L: ArithmeticLeaf> ArithmeticExpr<L> {
    pub(crate) const fn leaf(value: L, degree: usize) -> Self {
        Self {
            node: ArithmeticNode::Leaf(value),
            degree,
        }
    }
    /// Inspect the operation without permitting metadata mutation.
    pub const fn node(&self) -> &ArithmeticNode<L, L::Operand> {
        &self.node
    }
    /// Conservative degree in multiples of the trace degree.
    pub const fn degree_multiple(&self) -> usize {
        self.degree
    }
    /// Conservative degree with an explicit transition-selector weight.
    pub fn degree_multiple_with_transition(&self, transition: usize) -> usize {
        self.degree_inner(None, transition, &mut BTreeMap::new())
    }
    /// Trace-size-aware conservative polynomial degree, including periodic columns.
    pub fn poly_degree(&self, trace_len: usize, periodic_periods: &[usize]) -> usize {
        self.degree_inner(Some((trace_len, periodic_periods)), 0, &mut BTreeMap::new())
    }
    pub(crate) fn degree_inner(
        &self,
        trace: Option<(usize, &[usize])>,
        transition: usize,
        memo: &mut BTreeMap<usize, usize>,
    ) -> usize {
        let deg =
            |o: &L::Operand, memo: &mut BTreeMap<usize, usize>| o.degree(trace, transition, memo);
        match &self.node {
            ArithmeticNode::Leaf(l) => l.degree(trace, transition),
            ArithmeticNode::Add(a, b) | ArithmeticNode::Sub(a, b) => deg(a, memo).max(deg(b, memo)),
            ArithmeticNode::Mul(a, b) => add_degree(deg(a, memo), deg(b, memo)),
            ArithmeticNode::Neg(a) => deg(a, memo),
            ArithmeticNode::Square(a) => add_degree(deg(a, memo), deg(a, memo)),
            ArithmeticNode::Sum(a) => a.iter().map(|a| deg(a, memo)).max().unwrap_or(0),
            ArithmeticNode::Dot(a) | ArithmeticNode::MixedDot(a) => a
                .iter()
                .map(|(a, b)| {
                    if a.constant().is_some_and(|v| v == L::Value::ZERO)
                        || b.constant().is_some_and(|v| v == L::Value::ZERO)
                    {
                        0
                    } else {
                        add_degree(deg(a, memo), deg(b, memo))
                    }
                })
                .max()
                .unwrap_or(0),
        }
    }
    pub(crate) fn constant(&self) -> Option<L::Value> {
        match &self.node {
            ArithmeticNode::Leaf(l) => l.constant(),
            _ => None,
        }
    }
    fn build(node: ArithmeticNode<L, L::Operand>) -> Self {
        let constant = match &node {
            ArithmeticNode::Sum(values) => values
                .iter()
                .try_fold(L::Value::ZERO, |sum, v| Some(sum + v.constant()?)),
            ArithmeticNode::Dot(values) | ArithmeticNode::MixedDot(values) => {
                values.iter().try_fold(L::Value::ZERO, |sum, (a, b)| {
                    if a.constant() == Some(L::Value::ZERO) || b.constant() == Some(L::Value::ZERO)
                    {
                        Some(sum)
                    } else {
                        Some(sum + a.constant()? * b.constant()?)
                    }
                })
            }
            _ => None,
        };
        if let Some(c) = constant {
            return Self::leaf(L::from_constant(c), 0);
        }
        let mut e = Self { node, degree: 0 };
        e.degree = e.degree_inner(None, 0, &mut BTreeMap::new());
        e
    }
    pub(crate) fn dot(a: &[Self], b: &[L::Operand], mixed: bool) -> Self {
        assert_eq!(a.len(), b.len(), "linear combination length mismatch");
        let pairs: Arc<[_]> = a
            .iter()
            .cloned()
            .map(L::Operand::from_expression)
            .zip(b.iter().cloned())
            .collect();
        Self::build(if mixed {
            ArithmeticNode::MixedDot(pairs)
        } else {
            ArithmeticNode::Dot(pairs)
        })
    }
    fn binary(self, rhs: L::Operand, op: BinaryOperation) -> Self {
        let a = L::Operand::from_expression(self);
        if let (Some(x), Some(y)) = (a.constant(), rhs.constant()) {
            return Self::leaf(
                L::from_constant(match op {
                    Add => x + y,
                    Sub => x - y,
                    Mul => x * y,
                }),
                0,
            );
        }
        if op == Mul
            && (a.constant() == Some(L::Value::ZERO) || rhs.constant() == Some(L::Value::ZERO))
        {
            return Self::ZERO;
        }
        if rhs.constant() == Some(L::Value::ZERO) && op != Mul {
            return a.expression();
        }
        if a.constant() == Some(L::Value::ZERO) && op == Add {
            return rhs.expression();
        }
        if a.constant() == Some(L::Value::ZERO) && op == Sub {
            return -rhs.expression();
        }
        if op == Mul {
            if rhs.constant() == Some(L::Value::ONE) {
                return a.expression();
            }
            if a.constant() == Some(L::Value::ONE) {
                return rhs.expression();
            }
        }
        Self::build(match op {
            Add => ArithmeticNode::Add(a, rhs),
            Sub => ArithmeticNode::Sub(a, rhs),
            Mul => ArithmeticNode::Mul(a, rhs),
        })
    }
}
impl<L: ArithmeticLeaf> PrimeCharacteristicRing for ArithmeticExpr<L> {
    type PrimeSubfield = <L::Value as PrimeCharacteristicRing>::PrimeSubfield;
    const ZERO: Self = Self::leaf(L::ZERO, 0);
    const ONE: Self = Self::leaf(L::ONE, 0);
    const TWO: Self = Self::leaf(L::TWO, 0);
    const NEG_ONE: Self = Self::leaf(L::NEG_ONE, 0);
    fn from_prime_subfield(value: Self::PrimeSubfield) -> Self {
        Self::leaf(L::from_constant(L::Value::from_prime_subfield(value)), 0)
    }
    fn square(&self) -> Self {
        if let Some(c) = self.constant() {
            return Self::leaf(L::from_constant(c.square()), 0);
        }
        Self::build(ArithmeticNode::Square(L::Operand::from_expression(
            self.clone(),
        )))
    }
    fn sum_array<const N: usize>(input: &[Self]) -> Self {
        assert_eq!(input.len(), N, "sum_array length mismatch");
        input.iter().cloned().sum()
    }
    fn dot_product<const N: usize>(a: &[Self; N], b: &[Self; N]) -> Self {
        Self::dot(
            a,
            &b.iter()
                .cloned()
                .map(L::Operand::from_expression)
                .collect::<Vec<_>>(),
            false,
        )
    }
}
impl<L: ArithmeticLeaf> Dup for ArithmeticExpr<L> {
    fn dup(&self) -> Self {
        self.clone()
    }
}
impl<L: ArithmeticLeaf> Default for ArithmeticExpr<L> {
    fn default() -> Self {
        Self::ZERO
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> ops::Add<T> for ArithmeticExpr<L> {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        self.binary(rhs.into(), Add)
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> ops::Sub<T> for ArithmeticExpr<L> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        self.binary(rhs.into(), Sub)
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> ops::Mul<T> for ArithmeticExpr<L> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        self.binary(rhs.into(), Mul)
    }
}
impl<L: ArithmeticLeaf> ops::Neg for ArithmeticExpr<L> {
    type Output = Self;
    fn neg(self) -> Self {
        self.constant().map_or_else(
            || Self::build(ArithmeticNode::Neg(L::Operand::from_expression(self))),
            |c| Self::leaf(L::from_constant(-c), 0),
        )
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> ops::AddAssign<T> for ArithmeticExpr<L> {
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs;
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> ops::SubAssign<T> for ArithmeticExpr<L> {
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs;
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> ops::MulAssign<T> for ArithmeticExpr<L> {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs;
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> Sum<T> for ArithmeticExpr<L> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        Self::build(ArithmeticNode::Sum(iter.map(Into::into).collect()))
    }
}
impl<L: ArithmeticLeaf, T: Into<L::Operand>> Product<T> for ArithmeticExpr<L> {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<L: ArithmeticLeaf> ArithmeticExpr<L> {
    /// Lower to primitive symbolic operations, preserving arithmetic values.
    ///
    /// Sum grouping and sharing can differ from legacy capture; this is not a
    /// serialization or protocol-commitment compatibility guarantee.
    pub fn lower(&self) -> crate::SymbolicExpr<L::LegacyLeaf> {
        self.lower_inner(&mut BTreeMap::new())
    }
    pub(crate) fn lower_inner(
        &self,
        memo: &mut BTreeMap<usize, crate::SymbolicExpr<L::LegacyLeaf>>,
    ) -> crate::SymbolicExpr<L::LegacyLeaf> {
        use crate::SymbolicExpr as S;
        match &self.node {
            ArithmeticNode::Leaf(l) => S::Leaf(l.lower_leaf()),
            ArithmeticNode::Add(a, b) => a.lower(memo) + b.lower(memo),
            ArithmeticNode::Sub(a, b) => a.lower(memo) - b.lower(memo),
            ArithmeticNode::Mul(a, b) => a.lower(memo) * b.lower(memo),
            ArithmeticNode::Neg(a) => -a.lower(memo),
            ArithmeticNode::Square(a) => {
                let v = Arc::new(a.lower(memo));
                S::Mul {
                    x: v.clone(),
                    y: v,
                    degree_multiple: self.degree,
                }
            }
            ArithmeticNode::Sum(a) => a.iter().map(|a| a.lower(memo)).sum(),
            ArithmeticNode::Dot(a) | ArithmeticNode::MixedDot(a) => {
                a.iter().map(|(a, b)| a.lower(memo) * b.lower(memo)).sum()
            }
        }
    }
}

// Only the supplied leaf/operand families may participate in this representation.
pub(super) mod sealed {
    pub trait Leaf {}
    pub trait Operand {}
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BinaryOperation {
    Add,
    Sub,
    Mul,
}
