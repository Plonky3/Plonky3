use alloc::vec::Vec;

use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, add_scaled_slice_in_place,
    dot_product,
};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::strategy::VariableOrder;

/// A batched group of repeat-last successor evaluation constraints.
///
/// # Overview
///
/// The successor map sends a hypercube row `x` to `x + 1`, with the all-ones row repeating itself.
/// Each constraint pins a slot with an equality selector and evaluates a local point through that successor view.
///
/// # Algorithm
///
/// A constraint asserts one weighted sum over the polynomial, depending on whether the selector sits before or after the local variables.
///
/// ```text
/// sum_{prefix, x} P(prefix, x) * Eq(selector_i, prefix) * Next(point_i, x)
/// sum_{x, suffix} P(x, suffix) * Next(point_i, x) * Eq(selector_i, suffix)
/// ```
#[derive(Clone, Debug)]
pub struct NextStatement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// The slot-local successor constraints in their stored order.
    constraints: Vec<NextConstraint<F>>,
    /// Claimed successor evaluation of each constraint, aligned with the constraints above.
    pub evaluations: Vec<F>,
}

impl<F: Field> NextStatement<F> {
    /// Creates an empty group over the given variable space.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: the multilinear arity that every later constraint must match.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            // Fix the arity now; constraints added later are checked against it.
            num_variables,
            // No constraints and no claimed values until they are added.
            constraints: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true when no constraints have been added.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        // Invariant: every constraint has exactly one claimed value, so the two lists empty together.
        debug_assert!(self.constraints.is_empty() == self.evaluations.is_empty());
        self.constraints.is_empty()
    }

    /// Returns the number of stored constraints.
    #[must_use]
    pub const fn len(&self) -> usize {
        // Invariant: one claimed value per constraint, so both lists share a length.
        debug_assert!(self.constraints.len() == self.evaluations.len());
        self.constraints.len()
    }

    /// Streams one weight per stored constraint, each evaluated at a single point.
    ///
    /// # Arguments
    ///
    /// - `row`: the point at which every constraint weight is evaluated.
    pub fn weights_at<'a>(&'a self, row: &'a Point<F>) -> impl Iterator<Item = F> + 'a {
        // Walk the constraints in stored order.
        self.constraints
            .iter()
            // Each one yields its selector-gated successor weight at the query point.
            .map(|constraint| constraint.eval_at(row))
    }

    /// Adds one slot-local successor evaluation constraint with its claimed value.
    ///
    /// # Arguments
    ///
    /// - `selector`: equality point that fixes the slot, empty for a full-space constraint.
    /// - `point`: local point evaluated through the successor map inside that slot.
    /// - `eval`: the claimed successor evaluation for this constraint.
    /// - `var_order`: whether the selector variables come before or after the local variables.
    ///
    /// # Panics
    ///
    /// Panics if the selector and local variable counts together differ from the group's arity.
    pub fn add_evaluated_constraint(
        &mut self,
        selector: Point<F>,
        point: Point<F>,
        eval: F,
        var_order: VariableOrder,
    ) {
        // The slot variables plus the local variables must fill the whole space.
        assert_eq!(
            selector.num_variables() + point.num_variables(),
            self.num_variables()
        );
        // Store the constraint and its claimed value at matching positions.
        self.constraints.push(NextConstraint {
            selector,
            point,
            var_order,
        });
        self.evaluations.push(eval);
    }

    /// Accumulates the challenge-weighted sum of this group's claimed evaluations.
    ///
    /// # Arguments
    ///
    /// - `claimed_eval`: scalar accumulator updated in place.
    /// - `challenge`: the batching challenge whose powers weight each claimed value.
    /// - `shift`: offset of the first challenge power, so this group follows earlier groups.
    pub fn combine_evals(&self, claimed_eval: &mut F, challenge: F, shift: usize) {
        // Dot the claimed values against the challenge powers starting at the shift.
        // The shift keeps this group's powers disjoint from earlier groups.
        *claimed_eval += dot_product(
            self.evaluations.iter().copied(),
            challenge.shifted_powers(challenge.exp_u64(shift as u64)),
        );
    }

    /// Folds every constraint into one dense weight polynomial and one expected sum.
    ///
    /// # Overview
    ///
    /// Each constraint contributes its dense successor weight, scaled by a power of the challenge.
    /// The first constraint is weighted by the challenge raised to the shift, and each later one by the next power.
    ///
    /// # Arguments
    ///
    /// - `acc_weights`: dense weight polynomial accumulator over the hypercube.
    /// - `acc_sum`: scalar accumulator for the expected sum.
    /// - `challenge`: the batching challenge whose powers weight each constraint.
    /// - `shift`: offset of the first challenge power, so this group follows earlier groups.
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
        // Nothing to fold when this group holds no constraints.
        if self.constraints.is_empty() {
            return;
        }

        // Pair each constraint with its claimed value and its challenge power.
        // The powers start at the shift, one step per constraint.
        self.constraints
            .iter()
            .zip(self.evaluations.iter())
            .zip(challenge.shifted_powers(challenge.exp_u64(shift as u64)))
            .for_each(|((constraint, &eval), alpha)| {
                // Add this constraint's scaled dense weight into the polynomial accumulator.
                constraint.accumulate(acc_weights.as_mut_slice(), alpha);
                // Mirror the same scaling on the scalar side.
                *acc_sum += alpha * eval;
            });
    }

    /// SIMD-packed variant that folds every constraint into a packed weight polynomial and an expected sum.
    ///
    /// # Arguments
    ///
    /// - `weights`: packed weight polynomial accumulator over the hypercube.
    /// - `sum`: scalar accumulator for the expected sum.
    /// - `challenge`: the batching challenge whose powers weight each constraint.
    /// - `shift`: offset of the first challenge power, so this group follows earlier groups.
    ///
    /// # Panics
    ///
    /// Panics if the variable count is below the packing width.
    /// Panics if the packed buffer arity plus the packing width differs from the group's arity.
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
        // Nothing to fold when this group holds no constraints.
        if self.constraints.is_empty() {
            return;
        }

        let k = self.num_variables();
        // Number of variables collapsed into each packed lane.
        let k_pack = log2_strict_usize(Base::Packing::WIDTH);
        // The full space must be at least one packed slot wide.
        assert!(k >= k_pack);
        // The packed buffer plus the lane count must span the whole hypercube.
        assert_eq!(weights.num_variables() + k_pack, k);

        // Pair each constraint with its claimed value and its challenge power.
        // The powers start at the shift, one step per constraint.
        self.constraints
            .iter()
            .zip(self.evaluations.iter())
            .zip(challenge.shifted_powers(challenge.exp_u64(shift as u64)))
            .for_each(|((constraint, &eval), alpha)| {
                // Add this constraint's scaled dense weight into the packed accumulator.
                constraint.accumulate_packed::<Base>(weights.as_mut_slice(), alpha);
                // Mirror the same scaling on the scalar side.
                *sum += alpha * eval;
            });
    }
}

/// One slot-local repeat-last successor constraint.
///
/// # Overview
///
/// A fixed equality selector picks a slot, and a local point is evaluated through the successor view inside that slot.
#[derive(Clone, Debug)]
struct NextConstraint<F> {
    /// Equality point that fixes the slot.
    ///
    /// Empty when the constraint spans the whole space with no slot restriction.
    selector: Point<F>,
    /// Local point evaluated through the successor map inside the chosen slot.
    point: Point<F>,
    /// Whether the selector variables come before or after the local variables.
    var_order: VariableOrder,
}

impl<F: Field> NextConstraint<F> {
    /// Evaluates this constraint's weight at one full point.
    ///
    /// # Overview
    ///
    /// The weight is the slot selector value times the successor value of the local point.
    ///
    /// # Arguments
    ///
    /// - `row`: the full point, holding both the selector coordinates and the local coordinates.
    ///
    /// # Returns
    ///
    /// The product of the selector's equality value and the local point's successor value.
    fn eval_at(&self, row: &Point<F>) -> F {
        // Split the full point into its selector half and its local half.
        // The variable order decides which half comes first.
        let (selector_row, local_row) = match self.var_order {
            // Selector first: take the leading selector coordinates, the rest are local.
            VariableOrder::Prefix => row.split_at(self.selector.num_variables()),
            // Local first: take the leading local coordinates, the rest select the slot.
            VariableOrder::Suffix => {
                let (local_row, selector_row) = row.split_at(self.point.num_variables());
                (selector_row, local_row)
            }
        };
        // Fold the local point through the successor map.
        // The full successor value is the settled contribution plus the repeating all-ones boundary.
        let (_carry, done, omega) = Point::eval_next(self.point.as_slice(), local_row.as_slice());
        // Gate by the slot selector: the equality value is 1 only inside the chosen slot.
        Point::eval_eq(self.selector.as_slice(), selector_row.as_slice()) * (done + omega)
    }

    /// Adds this constraint's dense successor weight, scaled, into a hypercube buffer.
    ///
    /// # Overview
    ///
    /// The weight is the outer product of the slot selector table and the local successor table.
    /// Each hypercube entry is the selector weight of its slot times the successor weight of its local position.
    ///
    /// # Arguments
    ///
    /// - `out`: dense buffer over the full hypercube, one entry per row, added to in place.
    /// - `scale`: batching weight folded into the selector table so every entry carries it.
    ///
    /// # Panics
    ///
    /// Panics if the buffer length is not `2` to the power of the combined variable count.
    ///
    /// # Performance
    ///
    /// This builds the full dense table directly and is meant for general use, not the tightest opening path.
    pub(crate) fn accumulate(&self, out: &mut [F], scale: F) {
        // The buffer must cover exactly the slot variables plus the local variables.
        assert_eq!(
            log2_strict_usize(out.len()),
            self.selector.num_variables() + self.point.num_variables()
        );

        // Local axis: successor weights of the local point over its own hypercube.
        let local_next = Poly::new_next_from_point(self.point.as_slice());
        // Slot axis: selector equality table, premultiplied by the batching scale.
        let selector_eq = Poly::new_from_point(self.selector.as_slice(), scale);

        // The two axes interleave differently depending on which side leads the layout.
        match self.var_order {
            // Selector leads: the buffer is a run of local blocks, one block per slot.
            //
            //     out = [ slot_0 block | slot_1 block | ... ]
            //            each block has one entry per local row
            VariableOrder::Prefix => {
                let local_rows = 1 << self.point.num_variables();
                // Walk each slot block alongside its single selector weight.
                out.chunks_mut(local_rows)
                    .zip(selector_eq.as_slice().iter())
                    .for_each(|(chunk, &selector_weight)| {
                        // Within the block, weight each local row by the successor table.
                        // The shared kernel runs this fused multiply-add over SIMD lanes.
                        add_scaled_slice_in_place(chunk, local_next.as_slice(), selector_weight);
                    });
            }
            // Local leads: the buffer is a run of slot blocks, one block per local row.
            //
            //     out = [ local_0 block | local_1 block | ... ]
            //            each block has one entry per slot
            VariableOrder::Suffix => {
                let selector_rows = 1 << self.selector.num_variables();
                // Walk each local block alongside its single successor weight.
                out.chunks_mut(selector_rows)
                    .zip(local_next.iter())
                    .for_each(|(chunk, &next_weight)| {
                        // Within the block, weight each slot by the selector table.
                        // The shared kernel runs this fused multiply-add over SIMD lanes.
                        add_scaled_slice_in_place(chunk, selector_eq.as_slice(), next_weight);
                    });
            }
        }
    }

    /// SIMD-packed variant that adds this constraint's dense successor weight into a packed buffer.
    ///
    /// # Overview
    ///
    /// The scalar dense table is built first, then folded into packed elements.
    /// Each packed element covers one group of consecutive hypercube entries.
    ///
    /// # Arguments
    ///
    /// - `out`: packed buffer over the hypercube, added to in place.
    /// - `scale`: batching weight folded into the dense table.
    ///
    /// # Panics
    ///
    /// Panics if the packed buffer length times the packing width is not `2` to the combined variable count.
    pub(crate) fn accumulate_packed<Base>(&self, out: &mut [F::ExtensionPacking], scale: F)
    where
        Base: Field,
        F: ExtensionField<Base>,
    {
        // Total hypercube dimension is the slot variables plus the local variables.
        let total_vars = self.selector.num_variables() + self.point.num_variables();
        // Each packed slot holds this many scalar entries.
        let k_pack = log2_strict_usize(Base::Packing::WIDTH);
        // The packed buffer plus the lane count must span the whole hypercube.
        assert_eq!(log2_strict_usize(out.len()) + k_pack, total_vars);

        // Build the full scalar weight table for this constraint.
        let mut dense = F::zero_vec(1 << total_vars);
        self.accumulate(&mut dense, scale);

        // Fold each consecutive run of scalar entries into one packed element.
        //
        //     dense: [ e0 e1 e2 e3 | e4 e5 e6 e7 | ... ]
        //     out:   [   packed_0   |   packed_1   | ... ]
        out.iter_mut()
            .zip(dense.chunks(Base::Packing::WIDTH))
            .for_each(|(out, chunk)| {
                *out += F::ExtensionPacking::from_ext_slice(chunk);
            });
    }
}
