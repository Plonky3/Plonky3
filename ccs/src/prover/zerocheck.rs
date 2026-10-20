//! Zerocheck prover of 0 = sum_x eq(tau, x) * sum_k alpha_k * g_k(x) over constraint rows.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{RoundPolyInterpolator, RoundProver};

use crate::relation::{Constraint, RowSelector};

/// Matrix output tables, folded once per round.
enum Tables<F, EF> {
    /// Unbound base-field tables.
    Base(Vec<Poly<F>>),
    /// Tables after the first fold.
    Ext(Vec<Poly<EF>>),
}

/// Constraint compiled for the round kernel.
struct Compiled<'a, F, A> {
    /// Selected rows.
    selector: RowSelector,
    /// Term coefficients in the tables' field.
    coeffs: Vec<Scale<A>>,
    /// Source constraint, read for its factors.
    source: &'a Constraint<F>,
}

/// Term coefficient, with plus and minus one singled out to skip a multiplication.
#[derive(Clone, Copy)]
enum Scale<A> {
    /// Coefficient one.
    One,
    /// Coefficient minus one.
    NegOne,
    /// General coefficient.
    By(A),
}

impl<A: Field> Scale<A> {
    /// Classify a coefficient once, outside the hot loop.
    fn new(c: A) -> Self {
        if c == A::ONE {
            Self::One
        } else if c == A::NEG_ONE {
            Self::NegOne
        } else {
            Self::By(c)
        }
    }

    /// Scale a product by this coefficient.
    #[inline(always)]
    fn apply(self, product: A) -> A {
        match self {
            Self::One => product,
            Self::NegOne => -product,
            Self::By(c) => product * c,
        }
    }
}

/// Folded selector indicators, each kept as its one entry that differs from the bulk value.
///
/// Folding keeps that shape, so a selector costs one point per round.
#[derive(Clone, Copy)]
struct Boundaries<EF> {
    /// Entry 0 of the folded first-row indicator.
    first: EF,
    /// Last entry of the folded last-row indicator.
    last: EF,
    /// Last entry of the folded transition indicator.
    transition: EF,
}

impl<EF: Field> Boundaries<EF> {
    /// Unbound indicators.
    const fn new() -> Self {
        Self {
            first: EF::ONE,
            last: EF::ONE,
            transition: EF::ZERO,
        }
    }

    /// Bind the leading variable of every indicator.
    fn fold(&mut self, r: EF) {
        // Special entry pairs with a zero, a zero, and a one respectively.
        self.first *= EF::ONE - r;
        self.last *= r;
        self.transition = EF::ONE - r + r * self.transition;
    }

    /// Pair holding the special entry, and the selector minus its bulk value there at a node.
    fn correction(&self, selector: RowSelector, half: usize, node: EF) -> Option<(usize, EF)> {
        // First row lives in pair 0; last row and transition in pair half - 1.
        match selector {
            RowSelector::All => None,
            RowSelector::First => Some((0, self.first * (EF::ONE - node))),
            RowSelector::Last => Some((half - 1, self.last * node)),
            RowSelector::Transition => Some((half - 1, (self.transition - EF::ONE) * node)),
        }
    }
}

/// Zerocheck prover with Gruen's split (ePrint 2024/108, Section 3), summing degree d instead of d + 1.
///
/// ```text
/// eq(tau, (r, X, x')) = eq(tau_<k, r) * eq(tau_k, X) * eq(tau_>k, x')
/// ```
pub(crate) struct ZerocheckProver<'a, F, EF> {
    /// Constraints with base-field coefficients.
    base: Vec<Compiled<'a, F, F>>,
    /// Constraints with extension coefficients.
    ext: Vec<Compiled<'a, F, EF>>,
    /// Constraint batching weights.
    alphas: Vec<EF>,
    /// Interpolation nodes 0 to d in the extension.
    ext_nodes: Vec<EF>,
    /// Base-field nodes, when they all exist.
    base_nodes: Option<Vec<F>>,
    /// Extends the degree-d sum to one more node.
    interpolator: RoundPolyInterpolator<EF>,
    /// Zerocheck point.
    tau: Vec<EF>,
    /// Variable the next round binds.
    round: usize,
    /// Matrix output tables.
    tables: Tables<F, EF>,
    /// Folded selector indicators.
    boundaries: Boundaries<EF>,
    /// Equality table over the coordinates after this round.
    eq_tail: Poly<EF>,
    /// Product of the bound equality factors.
    scale: EF,
}

impl<'a, F: Field, EF: ExtensionField<F>> ZerocheckProver<'a, F, EF> {
    /// Start a zerocheck over one output table per matrix.
    pub(crate) fn new(
        constraints: &'a [Constraint<F>],
        alphas: Vec<EF>,
        degree: usize,
        tau: Vec<EF>,
        operands: Vec<Vec<F>>,
    ) -> Self {
        // Coefficients in both fields.
        let base: Vec<Compiled<'a, F, F>> = constraints
            .iter()
            .map(|source| Compiled {
                selector: source.selector(),
                coeffs: source
                    .terms()
                    .iter()
                    .map(|t| Scale::new(*t.coefficient()))
                    .collect(),
                source,
            })
            .collect();
        let ext: Vec<Compiled<'a, F, EF>> = constraints
            .iter()
            .map(|source| Compiled {
                selector: source.selector(),
                coeffs: source
                    .terms()
                    .iter()
                    .map(|t| Scale::new(EF::from(*t.coefficient())))
                    .collect(),
                source,
            })
            .collect();

        // Nodes 0 .. d of the degree-d sum.
        let ext_nodes: Vec<EF> = (0..=degree).map(EF::interpolation_node).collect();

        // Stay in the base field when every node lives there; otherwise lift up front.
        let base_nodes: Option<Vec<F>> = ext_nodes
            .iter()
            .map(<EF as ExtensionField<F>>::as_base)
            .collect();
        let tables = if base_nodes.is_some() {
            Tables::Base(operands.into_iter().map(Poly::new).collect())
        } else {
            Tables::Ext(
                operands
                    .into_iter()
                    .map(|t| Poly::new(t.into_iter().map(EF::from).collect()))
                    .collect(),
            )
        };

        // Tail over every coordinate after the first; empty product for a single row.
        let eq_tail = match tau.split_first() {
            Some((_, tail)) => Poly::new_from_point(tail, EF::ONE),
            None => Poly::new(vec![EF::ONE]),
        };

        Self {
            base,
            ext,
            alphas,
            interpolator: RoundPolyInterpolator::new(degree),
            ext_nodes,
            base_nodes,
            tau,
            round: 0,
            tables,
            boundaries: Boundaries::new(),
            eq_tail,
            scale: EF::ONE,
        }
    }

    /// Matrix outputs at the bound point.
    ///
    /// # Panics
    ///
    /// - Some row variable is still free.
    pub(crate) fn operand_evals(&self) -> Vec<EF> {
        let single = "a table with free variables has no single value";
        match &self.tables {
            Tables::Base(tables) => tables
                .iter()
                .map(|t| EF::from(t.as_constant().expect(single)))
                .collect(),
            Tables::Ext(tables) => tables
                .iter()
                .map(|t| t.as_constant().expect(single))
                .collect(),
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> RoundProver<EF> for ZerocheckProver<'_, F, EF> {
    fn fold(&mut self, r: EF) {
        // Bind the leading variable of every table.
        match &mut self.tables {
            // First fold lifts into the extension.
            Tables::Base(tables) => {
                self.tables = Tables::Ext(tables.iter().map(|t| t.fix_prefix_var(r)).collect());
            }
            // Later folds are in place.
            Tables::Ext(tables) => tables.iter_mut().for_each(|t| t.fix_prefix_var_mut(r)),
        }

        // Fold the selector entries.
        self.boundaries.fold(r);

        // Absorb the bound equality factor.
        self.scale *= eq_linear(self.tau[self.round], r);
        self.round += 1;

        // Summing out an eq variable drops it, since its two factors add to one.
        if self.eq_tail.num_evals() > 1 {
            self.eq_tail.sum_prefix_var_mut();
        }
    }

    fn round_poly(&self) -> Vec<EF> {
        // Degree-d sum at nodes 0 .. d.
        let q = match &self.tables {
            Tables::Base(tables) => Round::<F, F, EF> {
                tables,
                constraints: &self.base,
                nodes: self
                    .base_nodes
                    .as_deref()
                    .expect("base tables are kept only when every node is a base element"),
                ext_nodes: &self.ext_nodes,
                eq_tail: self.eq_tail.as_slice(),
                alphas: &self.alphas,
                boundaries: self.boundaries,
            }
            .sum(),
            Tables::Ext(tables) => Round::<F, EF, EF> {
                tables,
                constraints: &self.ext,
                nodes: &self.ext_nodes,
                ext_nodes: &self.ext_nodes,
                eq_tail: self.eq_tail.as_slice(),
                alphas: &self.alphas,
                boundaries: self.boundaries,
            }
            .sum(),
        };

        // Extend to node d + 1, with q(1) given through the running sum.
        let degree = q.len() - 1;
        let mut transmitted = Vec::with_capacity(degree);
        transmitted.push(q[0]);
        transmitted.extend_from_slice(&q[2..]);
        let q_last = *self
            .interpolator
            .extend_evals(&transmitted, q[0] + q[1], degree + 1)
            .last()
            .expect("the extension holds at least one entry");

        // h(X) = scale * eq(tau_k, X) * q(X) at nodes 0, 2, 3, ..., d + 1.
        let tau_k = self.tau[self.round];
        let at =
            |node: usize, q: EF| self.scale * eq_linear(tau_k, EF::interpolation_node(node)) * q;
        let mut h = Vec::with_capacity(degree + 1);
        h.push(at(0, q[0]));
        h.extend((2..=degree).map(|node| at(node, q[node])));
        h.push(at(degree + 1, q_last));
        h
    }
}

/// Equality factor eq(t, X) = (1 - t) * (1 - X) + t * X.
fn eq_linear<EF: Field>(t: EF, x: EF) -> EF {
    EF::ONE - t + (t.double() - EF::ONE) * x
}

/// Inputs of one round's sum, in the tables' field.
struct Round<'r, 'a, F, A, EF> {
    /// Matrix output tables.
    tables: &'r [Poly<A>],
    /// Compiled constraints.
    constraints: &'r [Compiled<'a, F, A>],
    /// Interpolation nodes in the tables' field.
    nodes: &'r [A],
    /// Nodes in the extension.
    ext_nodes: &'r [EF],
    /// Equality weights of the free pairs.
    eq_tail: &'r [EF],
    /// Constraint batching weights.
    alphas: &'r [EF],
    /// Folded selector indicators.
    boundaries: Boundaries<EF>,
}

impl<F, A, EF> Round<'_, '_, F, A, EF>
where
    F: Field,
    A: Field,
    EF: ExtensionField<A>,
{
    /// Degree-d part q(t) of the round polynomial at every node.
    ///
    /// Bulk pass with selectors at their bulk value, then one boundary correction per selector.
    fn sum(&self) -> Vec<EF> {
        let half = self.tables[0].num_evals() / 2;
        let width = self.nodes.len();
        let num_tables = self.tables.len();
        let num_constraints = self.constraints.len();
        debug_assert_eq!(
            self.eq_tail.len(),
            half,
            "the tail covers exactly the free variables"
        );

        // Bulk value one: every-row and transition constraints.
        let bulk: Vec<usize> = (0..num_constraints)
            .filter(|&k| {
                matches!(
                    self.constraints[k].selector,
                    RowSelector::All | RowSelector::Transition
                )
            })
            .collect();

        // Phase 1: bulk pass.
        let (mut sums, _) = (0..half)
            .into_par_iter()
            .with_min_task_bytes(2 * num_tables * size_of::<A>())
            .par_fold_reduce(
                // Per split: sums per constraint and node, plus a scratch row.
                || {
                    (
                        vec![EF::ZERO; num_constraints * width],
                        vec![A::ZERO; num_tables * width],
                    )
                },
                |(mut sums, mut values), x| {
                    self.lines(x, half, &mut values);
                    let weight = self.eq_tail[x];
                    for &k in &bulk {
                        for t in 0..width {
                            sums[k * width + t] += weight * self.constraint(k, &values, t);
                        }
                    }
                    (sums, values)
                },
                // Add split sums.
                |(mut left, scratch), (right, _)| {
                    left.iter_mut().zip(right).for_each(|(l, r)| *l += r);
                    (left, scratch)
                },
            );

        // Phase 2: add eq_tail * g_k * (selector - bulk value) at the boundary pair.
        let mut values = vec![A::ZERO; num_tables * width];
        for k in 0..num_constraints {
            for t in 0..width {
                let Some((x, departure)) = self.boundaries.correction(
                    self.constraints[k].selector,
                    half,
                    self.ext_nodes[t],
                ) else {
                    continue;
                };
                self.lines(x, half, &mut values);
                sums[k * width + t] += self.eq_tail[x] * departure * self.constraint(k, &values, t);
            }
        }

        // Phase 3: batch the constraints per node.
        (0..width)
            .map(|t| {
                self.alphas
                    .iter()
                    .zip(sums.chunks_exact(width))
                    .map(|(&alpha, sums)| alpha * sums[t])
                    .sum()
            })
            .collect()
    }

    /// Every table on its line through one free pair, at every node.
    #[inline(always)]
    fn lines(&self, x: usize, half: usize, values: &mut [A]) {
        let width = self.nodes.len();

        // Consecutive integer nodes step by one addition; others take a multiplication.
        let consecutive = self.nodes.get(2).is_none_or(|&two| two == A::TWO);

        for (table, row) in self.tables.iter().zip(values.chunks_exact_mut(width)) {
            let (lo, hi) = (table.as_slice()[x], table.as_slice()[x + half]);
            let slope = hi - lo;
            row[0] = lo;
            row[1] = hi;
            if consecutive {
                // u(t + 1) = u(t) + slope.
                for t in 2..width {
                    row[t] = row[t - 1] + slope;
                }
            } else {
                // u(t) = lo + node_t * slope.
                for (value, &node) in row[2..].iter_mut().zip(&self.nodes[2..]) {
                    *value = lo + slope * node;
                }
            }
        }
    }

    /// Term sum of one constraint at one node, ignoring its selector.
    #[inline(always)]
    fn constraint(&self, k: usize, values: &[A], t: usize) -> A {
        let width = self.nodes.len();
        let constraint = &self.constraints[k];
        constraint
            .source
            .terms()
            .iter()
            .zip(&constraint.coeffs)
            .map(|(term, &c)| {
                // Start from the first factor: a lone factor costs no product.
                let (&first, rest) = term
                    .factors()
                    .split_first()
                    .expect("every term has a factor");
                let product = rest
                    .iter()
                    .fold(values[first * width + t], |p, &j| p * values[j * width + t]);
                c.apply(product)
            })
            .fold(A::ZERO, |acc, v| acc + v)
    }
}
