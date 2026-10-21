use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::iter::{Product, Sum};
use core::ops;

use BinaryOperation::{Add, Mul, Sub};
use p3_field::{Dup, Field, PrimeCharacteristicRing};

/// Read-only arithmetic structure.
/// Mixed dots retain the coefficient operand class.
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
    type LowerBase: Default;
    fn lower_leaf(&self, base: &mut Self::LowerBase, depth: usize) -> Self::LegacyLeaf;
    type Operand: ArithmeticOperand<Leaf = Self>;
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;
    fn constant(&self) -> Option<Self::Value>;
    fn from_constant(value: Self::Value) -> Self;
    fn degree(&self, trace: Option<(usize, &[usize])>, transition: usize, depth: usize) -> usize;
}

/// Internal operand contract; operands preserve their original arithmetic domain.
#[doc(hidden)]
pub trait ArithmeticOperand:
    sealed::Operand + Clone + core::fmt::Debug + From<ArithmeticExpr<Self::Leaf>>
{
    type Leaf: ArithmeticLeaf<Operand = Self>;
    fn expression(self) -> ArithmeticExpr<Self::Leaf>;
    fn shared_expression(&self) -> Option<&Arc<ArithmeticExpr<Self::Leaf>>>;
    fn lower(
        &self,
        memo: &mut BTreeMap<
            usize,
            Arc<crate::SymbolicExpr<<Self::Leaf as ArithmeticLeaf>::LegacyLeaf>>,
        >,
        base: &mut <Self::Leaf as ArithmeticLeaf>::LowerBase,
        depth: usize,
    ) -> Arc<crate::SymbolicExpr<<Self::Leaf as ArithmeticLeaf>::LegacyLeaf>>;
    fn lower_value(
        &self,
        memo: &mut BTreeMap<
            usize,
            Arc<crate::SymbolicExpr<<Self::Leaf as ArithmeticLeaf>::LegacyLeaf>>,
        >,
        base: &mut <Self::Leaf as ArithmeticLeaf>::LowerBase,
        depth: usize,
    ) -> crate::SymbolicExpr<<Self::Leaf as ArithmeticLeaf>::LegacyLeaf>;
    fn from_expression(value: ArithmeticExpr<Self::Leaf>) -> Self;
    fn constant(&self) -> Option<<Self::Leaf as ArithmeticLeaf>::Value>;
    fn degree(
        &self,
        trace: Option<(usize, &[usize])>,
        transition: usize,
        memo: &mut BTreeMap<usize, usize>,
        depth: usize,
    ) -> usize;
}

/// Immutable arithmetic expression with conservative cached degree.
///
/// Constructors are private, so cached metadata cannot be invalidated.
/// Degree arithmetic panics on overflow instead of silently underestimating a bound.
#[derive(Clone, Debug)]
pub struct ArithmeticExpr<L: ArithmeticLeaf> {
    node: ArithmeticNode<L, L::Operand>,
    degree: usize,
}

pub(crate) const fn add_degree(a: usize, b: usize) -> usize {
    a.checked_add(b).expect("arithmetic degree overflow")
}

// Shallow expressions avoid an explicit traversal stack and its extra cache probes.
pub(super) const RECURSION_LIMIT: usize = 32;

// Complete children before evaluating their parent, keeping traversal frames on the heap.
// Keys refer to stable expressions borrowed from the source graph for the entire walk.
pub(super) fn evaluate_graph<'a, L: ArithmeticLeaf, V: Clone>(
    root: &'a ArithmeticExpr<L>,
    memo: &mut BTreeMap<usize, V>,
    skip_zero_products: bool,
    mut evaluate: impl FnMut(&ArithmeticExpr<L>, &mut BTreeMap<usize, V>) -> V,
) -> V {
    let root_key = core::ptr::from_ref(root) as usize;
    if let Some(value) = memo.get(&root_key) {
        return value.clone();
    }
    if matches!(root.node(), ArithmeticNode::Leaf(_)) {
        return evaluate(root, memo);
    }
    let mut pending = Vec::new();
    pending.push((root, false));
    while let Some((expression, ready)) = pending.pop() {
        let key = core::ptr::from_ref(expression) as usize;
        if ready {
            let value = evaluate(expression, memo);
            memo.insert(key, value);
        } else if !memo.contains_key(&key) {
            pending.push((expression, true));
            let mut push = |operand: &'a L::Operand| {
                if let Some(child) = operand.shared_expression() {
                    pending.push((child.as_ref(), false));
                }
            };
            match expression.node() {
                ArithmeticNode::Leaf(_) => {}
                ArithmeticNode::Add(a, b)
                | ArithmeticNode::Sub(a, b)
                | ArithmeticNode::Mul(a, b) => {
                    push(b);
                    push(a);
                }
                ArithmeticNode::Neg(a) | ArithmeticNode::Square(a) => push(a),
                ArithmeticNode::Sum(a) => a.iter().rev().for_each(push),
                ArithmeticNode::Dot(p) | ArithmeticNode::MixedDot(p) => {
                    for (a, b) in p.iter().rev() {
                        // Degree bounds ignore zero products before visiting either operand.
                        // Replay and lowering retain their normal operand traversal.
                        if skip_zero_products
                            && (a.constant() == Some(L::Value::ZERO)
                                || b.constant() == Some(L::Value::ZERO))
                        {
                            continue;
                        }
                        push(b);
                        push(a);
                    }
                }
            }
        }
    }
    memo[&root_key].clone()
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
        self.degree_inner(None, transition, &mut BTreeMap::new(), 0)
    }
    /// Trace-size-aware conservative polynomial degree, including periodic columns.
    pub fn poly_degree(&self, trace_len: usize, periodic_periods: &[usize]) -> usize {
        self.degree_inner(
            Some((trace_len, periodic_periods)),
            0,
            &mut BTreeMap::new(),
            0,
        )
    }
    pub(crate) fn degree_inner(
        &self,
        trace: Option<(usize, &[usize])>,
        transition: usize,
        memo: &mut BTreeMap<usize, usize>,
        depth: usize,
    ) -> usize {
        if (trace.is_none() && transition == 0) || depth < RECURSION_LIMIT {
            return self.degree_node(trace, transition, memo, depth);
        }
        evaluate_graph(self, memo, true, |e, memo| {
            e.degree_node(trace, transition, memo, depth)
        })
    }
    fn degree_node(
        &self,
        trace: Option<(usize, &[usize])>,
        transition: usize,
        memo: &mut BTreeMap<usize, usize>,
        depth: usize,
    ) -> usize {
        let deg = |o: &L::Operand, memo: &mut BTreeMap<usize, usize>| {
            o.degree(trace, transition, memo, depth + 1)
        };
        match &self.node {
            ArithmeticNode::Leaf(l) => l.degree(trace, transition, depth + 1),
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
        e.degree = e.degree_inner(None, 0, &mut BTreeMap::new(), 0);
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
    /// Sum grouping and sharing can differ from legacy capture; this is not a serialization or protocol-commitment compatibility guarantee.
    /// Traversal bounds recursion and uses an explicit stack for deeper graphs; destroying a deeply nested source or result graph still uses recursive `Arc` drops.
    pub fn lower(&self) -> crate::SymbolicExpr<L::LegacyLeaf> {
        self.lower_inner(&mut BTreeMap::new(), &mut L::LowerBase::default(), 0)
    }
    pub(crate) fn lower_inner(
        &self,
        memo: &mut BTreeMap<usize, Arc<crate::SymbolicExpr<L::LegacyLeaf>>>,
        base: &mut L::LowerBase,
        depth: usize,
    ) -> crate::SymbolicExpr<L::LegacyLeaf> {
        if depth < RECURSION_LIMIT || matches!(self.node, ArithmeticNode::Leaf(_)) {
            return self.lower_node(memo, base, depth);
        }
        (*evaluate_graph(self, memo, false, |e, memo| {
            Arc::new(e.lower_node(memo, base, depth))
        }))
        .clone()
    }
    fn lower_binary(
        x: Arc<crate::SymbolicExpr<L::LegacyLeaf>>,
        y: Arc<crate::SymbolicExpr<L::LegacyLeaf>>,
        op: BinaryOperation,
    ) -> crate::SymbolicExpr<L::LegacyLeaf> {
        use crate::SymbolicExpr as S;
        let constant = |v: &crate::SymbolicExpr<L::LegacyLeaf>| match v {
            S::Leaf(l) => crate::SymLeaf::as_const(l).copied(),
            _ => None,
        };
        let special = |v: &crate::SymbolicExpr<L::LegacyLeaf>| {
            constant(v).is_some_and(|c| c.is_zero() || c.is_one())
        };
        if special(&x) || special(&y) || (constant(&x).is_some() && constant(&y).is_some()) {
            return match op {
                Add => (*x).clone() + (*y).clone(),
                Sub => (*x).clone() - (*y).clone(),
                Mul => (*x).clone() * (*y).clone(),
            };
        }
        let degree_multiple = if op == Mul {
            add_degree(x.degree_multiple(), y.degree_multiple())
        } else {
            x.degree_multiple().max(y.degree_multiple())
        };
        match op {
            Add => S::Add {
                x,
                y,
                degree_multiple,
            },
            Sub => S::Sub {
                x,
                y,
                degree_multiple,
            },
            Mul => S::Mul {
                x,
                y,
                degree_multiple,
            },
        }
    }
    fn lower_node(
        &self,
        memo: &mut BTreeMap<usize, Arc<crate::SymbolicExpr<L::LegacyLeaf>>>,
        base: &mut L::LowerBase,
        depth: usize,
    ) -> crate::SymbolicExpr<L::LegacyLeaf> {
        use crate::SymbolicExpr as S;
        match &self.node {
            ArithmeticNode::Leaf(l) => S::Leaf(l.lower_leaf(base, depth + 1)),
            ArithmeticNode::Add(a, b) | ArithmeticNode::Sub(a, b) | ArithmeticNode::Mul(a, b) => {
                let op = match self.node {
                    ArithmeticNode::Add(..) => Add,
                    ArithmeticNode::Sub(..) => Sub,
                    _ => Mul,
                };
                Self::lower_binary(
                    a.lower(memo, base, depth + 1),
                    b.lower(memo, base, depth + 1),
                    op,
                )
            }
            ArithmeticNode::Neg(a) => {
                let x = a.lower(memo, base, depth + 1);
                if matches!(x.as_ref(), S::Leaf(l) if crate::SymLeaf::as_const(l).is_some()) {
                    return -(*x).clone();
                }
                let degree_multiple = x.degree_multiple();
                S::Neg { x, degree_multiple }
            }
            ArithmeticNode::Square(a) => {
                let v = a.lower(memo, base, depth + 1);
                S::Mul {
                    degree_multiple: self.degree,
                    x: v.clone(),
                    y: v,
                }
            }
            ArithmeticNode::Sum(a) => a.iter().map(|a| a.lower_value(memo, base, depth + 1)).sum(),
            ArithmeticNode::Dot(a) | ArithmeticNode::MixedDot(a) => a
                .iter()
                .map(|(a, b)| {
                    if a.constant() == Some(L::Value::ZERO) || b.constant() == Some(L::Value::ZERO)
                    {
                        S::ZERO
                    } else if a.constant() == Some(L::Value::ONE) {
                        b.lower_value(memo, base, depth + 1)
                    } else if b.constant() == Some(L::Value::ONE) {
                        a.lower_value(memo, base, depth + 1)
                    } else {
                        Self::lower_binary(
                            a.lower(memo, base, depth + 1),
                            b.lower(memo, base, depth + 1),
                            Mul,
                        )
                    }
                })
                .sum(),
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
