//! Memoized, iterative interpretation of symbolic expression DAGs.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use super::{BaseLeaf, ExtEntry, ExtLeaf, SymbolicExpr, SymbolicExpression, SymbolicExpressionExt};
use crate::{AirBuilder, PermutationAirBuilder, WindowAccess};

/// Interprets leaves and arithmetic in a [`SymbolicExpr`] DAG.
///
/// Outputs can be field values, degrees, dependency sets, or compiler node IDs.
/// The leaf lifetime allows visitors to retain references to embedded subtrees.
/// Arithmetic callbacks receive already computed child outputs. Each distinct
/// node is visited once per [`fold_symbolic_expressions`] call, so callbacks
/// should not rely on running once per incoming edge.
pub trait SymbolicExprVisitor<'a, A: 'a> {
    /// Cached interpretation of a node. Cloning should ideally be cheap.
    type Output: Clone;

    /// Interpret a leaf, which remains alive for `'a`.
    fn leaf(&mut self, leaf: &'a A) -> Self::Output;
    /// Interpret an addition.
    fn add(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output;
    /// Interpret a subtraction.
    fn sub(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output;
    /// Interpret a negation.
    fn neg(&mut self, x: &Self::Output) -> Self::Output;
    /// Interpret a multiplication.
    fn mul(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output;
}

/// Fold several roots using an explicit stack and one pointer-identity cache.
///
/// Children are interpreted before their parents, with left children first.
/// Shared nodes (including duplicate roots) are interpreted once across the
/// entire call. Structurally equal nodes at different addresses remain distinct.
/// Results follow root order, including duplicates; empty input returns no
/// results. Nodes remain borrowed throughout the call, so cached addresses
/// cannot be invalidated or reused.
///
/// This uses `O(n)` cached outputs and `O(h)` stack entries for `n` distinct
/// nodes and maximum depth `h`, with `O(n log n)` map overhead. Output cloning
/// and visitor work are additional costs. Traversal does not recurse; dropping
/// an owned expression or serializing it can still recurse.
///
/// ```
/// use p3_air::{SymbolicExpr, SymbolicExprVisitor, fold_symbolic_expressions};
///
/// struct Degree;
/// impl<'a> SymbolicExprVisitor<'a, usize> for Degree {
///     type Output = usize;
///     fn leaf(&mut self, degree: &'a usize) -> usize { *degree }
///     fn add(&mut self, x: &usize, y: &usize) -> usize { (*x).max(*y) }
///     fn sub(&mut self, x: &usize, y: &usize) -> usize { (*x).max(*y) }
///     fn neg(&mut self, x: &usize) -> usize { *x }
///     fn mul(&mut self, x: &usize, y: &usize) -> usize { x + y }
/// }
/// let root = SymbolicExpr::Leaf(2);
/// assert_eq!(fold_symbolic_expressions(&[&root, &root], &mut Degree), [2, 2]);
/// ```
pub fn fold_symbolic_expressions<'a, A: 'a, V: SymbolicExprVisitor<'a, A>>(
    roots: &[&'a SymbolicExpr<A>],
    visitor: &mut V,
) -> Vec<V::Output> {
    let mut cache = BTreeMap::<*const SymbolicExpr<A>, V::Output>::new();
    let mut stack = Vec::new();
    let mut outputs = Vec::with_capacity(roots.len());
    for &root in roots {
        stack.push((root, false));
        while let Some((node, ready)) = stack.pop() {
            let key = core::ptr::from_ref(node);
            if cache.contains_key(&key) {
                continue;
            }
            if !ready {
                stack.push((node, true));
                match node {
                    SymbolicExpr::Leaf(_) => {}
                    SymbolicExpr::Neg { x, .. } => stack.push((x, false)),
                    SymbolicExpr::Add { x, y, .. }
                    | SymbolicExpr::Sub { x, y, .. }
                    | SymbolicExpr::Mul { x, y, .. } => {
                        stack.push((y, false));
                        stack.push((x, false));
                    }
                }
                continue;
            }
            let child = |node: &SymbolicExpr<A>| &cache[&core::ptr::from_ref(node)];
            let output = match node {
                SymbolicExpr::Leaf(leaf) => visitor.leaf(leaf),
                SymbolicExpr::Add { x, y, .. } => visitor.add(child(x), child(y)),
                SymbolicExpr::Sub { x, y, .. } => visitor.sub(child(x), child(y)),
                SymbolicExpr::Neg { x, .. } => visitor.neg(child(x)),
                SymbolicExpr::Mul { x, y, .. } => visitor.mul(child(x), child(y)),
            };
            cache.insert(key, output);
        }
        outputs.push(cache[&core::ptr::from_ref(root)].clone());
    }
    outputs
}

// Both resolvers interpret arithmetic in their own builder algebra.
macro_rules! arithmetic_callbacks {
    () => {
        fn add(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output {
            x.clone() + y.clone()
        }
        fn sub(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output {
            x.clone() - y.clone()
        }
        fn neg(&mut self, x: &Self::Output) -> Self::Output {
            -x.clone()
        }
        fn mul(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output {
            x.clone() * y.clone()
        }
    };
}

struct BaseResolver<'b, AB>(&'b AB);

impl<'a, F: Field, AB: AirBuilder<F = F>> SymbolicExprVisitor<'a, BaseLeaf<F>>
    for BaseResolver<'_, AB>
{
    type Output = AB::Expr;

    fn leaf(&mut self, leaf: &'a BaseLeaf<F>) -> Self::Output {
        // Reuse the legacy leaf lookup and its validation; no subtree is cloned
        // and this call cannot recurse because its argument is a leaf.
        SymbolicExpr::Leaf(leaf.clone()).resolve(self.0)
    }

    arithmetic_callbacks!();
}

/// Resolve base-field roots with shared memoization and no recursive traversal.
///
/// Unlike [`SymbolicExpression::resolve`], this shares computed values across
/// roots and shared children. This is an opt-in path with allocation and map
/// overhead, which can cost more for small, unshared expressions.
///
/// # Panics
///
/// Panics for row offsets other than zero or one, or out-of-bounds indices,
/// just like [`SymbolicExpression::resolve`].
pub fn resolve_symbolic_expressions<F: Field, AB: AirBuilder<F = F>>(
    roots: &[&SymbolicExpression<F>],
    builder: &AB,
) -> Vec<AB::Expr> {
    fold_symbolic_expressions(roots, &mut BaseResolver(builder))
}

#[derive(Default)]
struct BaseRoots<'a, F> {
    roots: Vec<&'a SymbolicExpression<F>>,
}

impl<'a, F: 'a, EF: 'a> SymbolicExprVisitor<'a, ExtLeaf<F, EF>> for BaseRoots<'a, F> {
    type Output = ();

    fn leaf(&mut self, leaf: &'a ExtLeaf<F, EF>) {
        if let ExtLeaf::Base(base) = leaf {
            self.roots.push(base);
        }
    }
    fn add(&mut self, _: &(), _: &()) {}
    fn sub(&mut self, _: &(), _: &()) {}
    fn neg(&mut self, _: &()) {}
    fn mul(&mut self, _: &(), _: &()) {}
}

struct ExtensionResolver<'b, F, AB: AirBuilder> {
    builder: &'b AB,
    bases: BTreeMap<*const SymbolicExpression<F>, AB::Expr>,
}

impl<'a, F: Field, EF: ExtensionField<F>, AB: PermutationAirBuilder<F = F, EF = EF>>
    SymbolicExprVisitor<'a, ExtLeaf<F, EF>> for ExtensionResolver<'_, F, AB>
{
    type Output = AB::ExprEF;

    fn leaf(&mut self, leaf: &'a ExtLeaf<F, EF>) -> Self::Output {
        match leaf {
            ExtLeaf::Base(base) => self.bases[&core::ptr::from_ref(base)].clone().into(),
            ExtLeaf::ExtConstant(value) => (*value).into(),
            ExtLeaf::ExtVariable(variable) => match variable.entry {
                ExtEntry::Permutation { offset } => {
                    let permutation = self.builder.permutation();
                    match offset {
                        0 => permutation.current(variable.index),
                        1 => permutation.next(variable.index),
                        _ => panic!("expressions cannot span more than two rows"),
                    }
                    .expect("permutation column index out of bounds")
                    .into()
                }
                ExtEntry::Challenge => self.builder.permutation_randomness()[variable.index].into(),
                ExtEntry::PermutationValue => self.builder.permutation_values()[variable.index]
                    .clone()
                    .into(),
            },
        }
    }

    arithmetic_callbacks!();
}

/// Resolve extension-field roots with shared base and extension memoization.
///
/// All embedded base subtrees are collected first and folded together in
/// `AB::Expr`, preserving shared base children across different extension leaves
/// and roots. Only their results are lifted into `AB::ExprEF`. Extension
/// arithmetic then runs in `AB::ExprEF`, including true extension constants,
/// permutation columns, challenges, and expected permutation values.
///
/// Each call uses fresh caches. The result order includes duplicate roots.
///
/// # Panics
///
/// Panics for row offsets other than zero or one, out-of-bounds indices, or
/// permutation references on a builder without permutation data.
pub fn resolve_symbolic_expressions_ext<
    F: Field,
    EF: ExtensionField<F>,
    AB: PermutationAirBuilder<F = F, EF = EF>,
>(
    roots: &[&SymbolicExpressionExt<F, EF>],
    builder: &AB,
) -> Vec<AB::ExprEF> {
    let mut base_roots = BaseRoots { roots: Vec::new() };
    fold_symbolic_expressions(roots, &mut base_roots);
    let values = resolve_symbolic_expressions(&base_roots.roots, builder);
    let bases = base_roots
        .roots
        .into_iter()
        .map(core::ptr::from_ref)
        .zip(values)
        .collect();
    fold_symbolic_expressions(roots, &mut ExtensionResolver { builder, bases })
}

impl<F: Field, EF: ExtensionField<F>> SymbolicExpressionExt<F, EF> {
    /// Resolve this extension expression against a permutation-capable builder.
    ///
    /// Uses [`resolve_symbolic_expressions_ext`]; pass multiple roots to that
    /// function to reuse shared results across constraints.
    ///
    /// # Panics
    ///
    /// See [`resolve_symbolic_expressions_ext`].
    pub fn resolve<AB: PermutationAirBuilder<F = F, EF = EF>>(&self, builder: &AB) -> AB::ExprEF {
        resolve_symbolic_expressions_ext(&[self], builder)
            .pop()
            .unwrap()
    }
}
