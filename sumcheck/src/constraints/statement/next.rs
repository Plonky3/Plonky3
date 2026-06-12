use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::strategy::VariableOrder;

/// A batched system of repeat-last Next evaluation constraints.
///
/// Each constraint asserts one of:
/// ```text
/// sum_{prefix, x} P(prefix, x) * Eq(selector_i, prefix) * Next(point_i, x)
/// sum_{x, suffix} P(x, suffix) * Next(point_i, x) * Eq(selector_i, suffix)
/// ```
#[derive(Clone, Debug)]
pub struct NextStatement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// Repeat-last Next constraints.
    constraints: Vec<NextConstraint<F>>,
    /// Claimed repeat-last Next evaluations.
    pub evaluations: Vec<F>,
}

/// One slot-local repeat-last Next constraint.
#[derive(Clone, Debug)]
struct NextConstraint<F> {
    /// Fixed slot selector. Empty for a full-space Next constraint.
    selector: Point<F>,
    /// Local point used by `Next(point, x_local)`.
    point: Point<F>,
    /// Whether selector variables appear before or after local variables.
    var_order: VariableOrder,
}

impl<F: Field> NextConstraint<F> {
    /// Evaluates this constraint's weight at a full challenge point.
    fn eval_at(&self, row: &Point<F>) -> F {
        let (selector_row, local_row) = match self.var_order {
            VariableOrder::Prefix => row.split_at(self.selector.num_variables()),
            VariableOrder::Suffix => {
                let (local_row, selector_row) = row.split_at(self.point.num_variables());
                (selector_row, local_row)
            }
        };
        let (_carry, done, omega) = Point::eval_next(self.point.as_slice(), local_row.as_slice());
        Point::eval_eq(self.selector.as_slice(), selector_row.as_slice()) * (done + omega)
    }

    /// Accumulates this dense Next weight into `out`.
    ///
    /// This is intentionally simple rather than WHIR-hot-path optimized. WHIR
    /// openings use SVO-specific prefix/suffix Next paths instead.
    pub(crate) fn accumulate(&self, out: &mut [F], scale: F) {
        assert_eq!(
            log2_strict_usize(out.len()),
            self.selector.num_variables() + self.point.num_variables()
        );

        let local_next = Poly::new_next_from_point(self.point.as_slice());
        let selector_eq = Poly::new_from_point(self.selector.as_slice(), scale);

        match self.var_order {
            VariableOrder::Prefix => {
                let local_rows = 1 << self.point.num_variables();
                out.chunks_mut(local_rows)
                    .zip(selector_eq.as_slice().iter())
                    .for_each(|(chunk, &selector_weight)| {
                        chunk
                            .iter_mut()
                            .zip(local_next.iter())
                            .for_each(|(out, &next_weight)| *out += selector_weight * next_weight);
                    });
            }
            VariableOrder::Suffix => {
                let selector_rows = 1 << self.selector.num_variables();
                out.chunks_mut(selector_rows)
                    .zip(local_next.iter())
                    .for_each(|(chunk, &next_weight)| {
                        chunk.iter_mut().zip(selector_eq.iter()).for_each(
                            |(out, &selector_weight)| {
                                *out += selector_weight * next_weight;
                            },
                        );
                    });
            }
        }
    }

    /// Packed dense variant of [`accumulate`](Self::accumulate).
    ///
    /// This materializes scalar dense weights before packing. It is kept as a
    /// generic constraint fallback; WHIR's optimized Next opening path does not
    /// call this in the current protocol flow.
    pub(crate) fn accumulate_packed<Base>(&self, out: &mut [F::ExtensionPacking], scale: F)
    where
        Base: Field,
        F: ExtensionField<Base>,
    {
        let total_vars = self.selector.num_variables() + self.point.num_variables();
        let k_pack = log2_strict_usize(Base::Packing::WIDTH);
        assert_eq!(log2_strict_usize(out.len()) + k_pack, total_vars);

        let mut dense = F::zero_vec(1 << total_vars);
        self.accumulate(&mut dense, scale);

        out.iter_mut()
            .zip(dense.chunks(Base::Packing::WIDTH))
            .for_each(|(out, chunk)| {
                *out += F::ExtensionPacking::from_ext_slice(chunk);
            });
    }
}

impl<F: Field> NextStatement<F> {
    /// Creates an empty `NextStatement<F>`.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            constraints: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true if no constraints have been added.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.constraints.is_empty() == self.evaluations.is_empty());
        self.constraints.is_empty()
    }

    /// Returns the number of constraints.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.constraints.len() == self.evaluations.len());
        self.constraints.len()
    }

    /// Iterates over this statement's weights evaluated at `row`.
    pub fn weights_at<'a>(&'a self, row: &'a Point<F>) -> impl Iterator<Item = F> + 'a {
        self.constraints
            .iter()
            .map(|constraint| constraint.eval_at(row))
    }

    /// Adds a repeat-last Next evaluation constraint.
    ///
    /// `selector` fixes the slot containing the local `point`. Use an empty
    /// selector for a full-space Next constraint. `var_order` tells whether the
    /// selector variables are before or after the local Next variables.
    pub fn add_evaluated_constraint(
        &mut self,
        selector: Point<F>,
        point: Point<F>,
        eval: F,
        var_order: VariableOrder,
    ) {
        assert_eq!(
            selector.num_variables() + point.num_variables(),
            self.num_variables()
        );
        self.constraints.push(NextConstraint {
            selector,
            point,
            var_order,
        });
        self.evaluations.push(eval);
    }

    /// Combines expected evaluation values with powers of `challenge`.
    pub fn combine_evals(&self, claimed_eval: &mut F, challenge: F, shift: usize) {
        *claimed_eval += dot_product(
            self.evaluations.iter().copied(),
            challenge.shifted_powers(challenge.exp_u64(shift as u64)),
        );
    }

    /// Combines all Next constraints into a dense hypercube weight polynomial.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine<Base>(
        &self,
        acc_weights: &mut Poly<F>,
        acc_sum: &mut F,
        challenge: F,
        shift: usize,
    ) where
        F: ExtensionField<Base>,
        Base: Field,
    {
        if self.constraints.is_empty() {
            return;
        }

        self.constraints
            .iter()
            .zip(self.evaluations.iter())
            .zip(challenge.shifted_powers(challenge.exp_u64(shift as u64)))
            .for_each(|((constraint, &eval), alpha)| {
                constraint.accumulate(acc_weights.as_mut_slice(), alpha);
                *acc_sum += alpha * eval;
            });
    }

    /// SIMD-packed variant of [`combine`](Self::combine).
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_packed<Base>(
        &self,
        weights: &mut Poly<F::ExtensionPacking>,
        sum: &mut F,
        challenge: F,
        shift: usize,
    ) where
        F: ExtensionField<Base>,
        Base: Field,
    {
        if self.constraints.is_empty() {
            return;
        }

        let k = self.num_variables();
        let k_pack = log2_strict_usize(Base::Packing::WIDTH);
        assert!(k >= k_pack);
        assert_eq!(weights.num_variables() + k_pack, k);

        self.constraints
            .iter()
            .zip(self.evaluations.iter())
            .zip(challenge.shifted_powers(challenge.exp_u64(shift as u64)))
            .for_each(|((constraint, &eval), alpha)| {
                constraint.accumulate_packed::<Base>(weights.as_mut_slice(), alpha);
                *sum += alpha * eval;
            });
    }
}
