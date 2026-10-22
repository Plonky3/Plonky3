//! AIR constraints at the out-of-domain point, as circuit values.
//!
//! The AIR is captured symbolically once, and each leaf maps to a lifted opening.
//!
//! Constraints fold in global assertion order, as the native folder does.

use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::symbolic::{
    BaseEntry, BaseLeaf, ConstraintLayout, ExtEntry, ExtLeaf, SymbolicExprVisitor,
    SymbolicExpression, SymbolicExpressionExt, fold_symbolic_expression_cached,
    fold_symbolic_expressions,
};
use p3_commit::LagrangeSelectors;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};

use crate::{Base, Ext};

/// The circuit value every symbolic leaf stands for at the out-of-domain point.
pub(crate) struct Leaves<'a, F, EF> {
    /// Main trace openings at `zeta`.
    pub(crate) trace_local: &'a [Ext<F, EF>],
    /// Main trace openings at the next point.
    pub(crate) trace_next: &'a [Ext<F, EF>],
    /// Preprocessed openings at `zeta`.
    pub(crate) preprocessed_local: &'a [Ext<F, EF>],
    /// Preprocessed openings at the next point.
    pub(crate) preprocessed_next: &'a [Ext<F, EF>],
    /// Periodic column values at `zeta`.
    pub(crate) periodic_values: &'a [Ext<F, EF>],
    /// Public values of the AIR.
    pub(crate) public_values: &'a [Base<F>],
    /// Lagrange selectors at `zeta`.
    pub(crate) selectors: &'a LagrangeSelectors<Ext<F, EF>>,
    /// Permutation trace openings at `zeta`.
    pub(crate) permutation_local: &'a [Ext<F, EF>],
    /// Permutation trace openings at the next point.
    pub(crate) permutation_next: &'a [Ext<F, EF>],
    /// Challenges of the permutation argument.
    pub(crate) permutation_challenges: &'a [Ext<F, EF>],
    /// Claimed values of the permutation argument.
    pub(crate) permutation_values: &'a [Ext<F, EF>],
}

/// Interprets base-field leaves; their values at `zeta` lie in the challenge field.
struct BaseVisitor<'v, 'a, F, EF>(&'v Leaves<'a, F, EF>);

impl<'e, F: Field, EF: ExtensionField<F>> SymbolicExprVisitor<'e, BaseLeaf<F>>
    for BaseVisitor<'_, '_, F, EF>
{
    type Output = Ext<F, EF>;

    fn leaf(&mut self, leaf: &'e BaseLeaf<F>) -> Ext<F, EF> {
        let leaves = self.0;
        match leaf {
            BaseLeaf::Variable(v) => match v.entry {
                BaseEntry::Main { offset: 0 } => leaves.trace_local[v.index].clone(),
                BaseEntry::Main { .. } => leaves.trace_next[v.index].clone(),
                BaseEntry::Preprocessed { offset: 0 } => leaves.preprocessed_local[v.index].clone(),
                BaseEntry::Preprocessed { .. } => leaves.preprocessed_next[v.index].clone(),
                BaseEntry::Periodic => leaves.periodic_values[v.index].clone(),
                BaseEntry::Public => Ext::from(leaves.public_values[v.index].clone()),
            },
            BaseLeaf::IsFirstRow => leaves.selectors.is_first_row.clone(),
            BaseLeaf::IsLastRow => leaves.selectors.is_last_row.clone(),
            BaseLeaf::IsTransition => leaves.selectors.is_transition.clone(),
            BaseLeaf::Constant(c) => Ext::from(Base::from(*c)),
        }
    }

    fn add(&mut self, x: &Ext<F, EF>, y: &Ext<F, EF>) -> Ext<F, EF> {
        x.clone() + y.clone()
    }

    fn sub(&mut self, x: &Ext<F, EF>, y: &Ext<F, EF>) -> Ext<F, EF> {
        x.clone() - y.clone()
    }

    fn neg(&mut self, x: &Ext<F, EF>) -> Ext<F, EF> {
        -x.clone()
    }

    fn mul(&mut self, x: &Ext<F, EF>, y: &Ext<F, EF>) -> Ext<F, EF> {
        x.clone() * y.clone()
    }
}

/// Interprets challenge-field leaves, folding embedded base expressions against one shared cache.
struct ExtVisitor<'v, 'a, 'e, F, EF> {
    leaves: &'v Leaves<'a, F, EF>,
    bases: HashMap<*const SymbolicExpression<F>, Ext<F, EF>>,
    stack: Vec<(&'e SymbolicExpression<F>, bool)>,
}

impl<'e, F: Field, EF: ExtensionField<F>> SymbolicExprVisitor<'e, ExtLeaf<F, EF>>
    for ExtVisitor<'_, '_, 'e, F, EF>
{
    type Output = Ext<F, EF>;

    fn leaf(&mut self, leaf: &'e ExtLeaf<F, EF>) -> Ext<F, EF> {
        let leaves = self.leaves;
        match leaf {
            ExtLeaf::Base(base) => fold_symbolic_expression_cached(
                base,
                &mut BaseVisitor(leaves),
                &mut self.bases,
                &mut self.stack,
            ),
            ExtLeaf::ExtVariable(v) => match v.entry {
                ExtEntry::Permutation { offset: 0 } => leaves.permutation_local[v.index].clone(),
                ExtEntry::Permutation { .. } => leaves.permutation_next[v.index].clone(),
                ExtEntry::Challenge => leaves.permutation_challenges[v.index].clone(),
                ExtEntry::PermutationValue => leaves.permutation_values[v.index].clone(),
            },
            ExtLeaf::ExtConstant(c) => Ext::from(*c),
        }
    }

    fn add(&mut self, x: &Ext<F, EF>, y: &Ext<F, EF>) -> Ext<F, EF> {
        x.clone() + y.clone()
    }

    fn sub(&mut self, x: &Ext<F, EF>, y: &Ext<F, EF>) -> Ext<F, EF> {
        x.clone() - y.clone()
    }

    fn neg(&mut self, x: &Ext<F, EF>) -> Ext<F, EF> {
        -x.clone()
    }

    fn mul(&mut self, x: &Ext<F, EF>, y: &Ext<F, EF>) -> Ext<F, EF> {
        x.clone() * y.clone()
    }
}

/// Every captured constraint at `zeta`, folded by `alpha` in global assertion order.
///
/// The fold is Horner's rule in `alpha`, so the first assertion gets the highest power.
pub(crate) fn fold_constraints<F: Field, EF: ExtensionField<F>>(
    base: &[SymbolicExpression<F>],
    ext: &[SymbolicExpressionExt<F, EF>],
    layout: &ConstraintLayout,
    leaves: &Leaves<'_, F, EF>,
    alpha: &Ext<F, EF>,
) -> Ext<F, EF> {
    let mut visitor = ExtVisitor {
        leaves,
        bases: HashMap::new(),
        stack: Vec::new(),
    };
    let base_values: Vec<Ext<F, EF>> = base
        .iter()
        .map(|root| {
            fold_symbolic_expression_cached(
                root,
                &mut BaseVisitor(leaves),
                &mut visitor.bases,
                &mut visitor.stack,
            )
        })
        .collect();
    let ext_roots: Vec<&SymbolicExpressionExt<F, EF>> = ext.iter().collect();
    let ext_values = fold_symbolic_expressions(&ext_roots, &mut visitor);

    let mut ordered = vec![Ext::ZERO; layout.total_constraints()];
    for (&position, value) in layout.base_indices.iter().zip(base_values) {
        ordered[position] = value;
    }
    for (&position, value) in layout.ext_indices.iter().zip(ext_values) {
        ordered[position] = value;
    }
    ordered.into_iter().fold(Ext::ZERO, |acc, constraint| {
        acc * alpha.clone() + constraint
    })
}
