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
    pub fn combine(&self, acc_weights: &mut Poly<F>, acc_sum: &mut F, challenge: F, shift: usize) {
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
        // Slice the full row into its selector and local halves; the variable
        // order decides which half leads.
        let row = row.as_slice();
        let sel = self.selector.num_variables();
        let loc = self.point.num_variables();
        let (selector_row, local_row) = match self.var_order {
            VariableOrder::Prefix => (&row[..sel], &row[sel..]),
            VariableOrder::Suffix => (&row[loc..], &row[..loc]),
        };
        // Fold the local point through the successor map, then gate by the slot
        // selector whose equality value is 1 only inside the chosen slot.
        let (_carry, done, omega) = Point::eval_next(self.point.as_slice(), local_row);
        Point::eval_eq(self.selector.as_slice(), selector_row) * (done + omega)
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

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn empty_group_reports_no_constraints() {
        // A fresh group over 3 variables holds nothing.
        let statement = NextStatement::<EF>::initialize(3);
        assert_eq!(statement.num_variables(), 3);
        assert!(statement.is_empty());
        assert_eq!(statement.len(), 0);
    }

    #[test]
    fn adding_a_constraint_grows_the_group() {
        // Each added constraint contributes one stored constraint and one claim.
        let mut statement = NextStatement::<EF>::initialize(2);
        statement.add_evaluated_constraint(
            Point::new(Vec::new()),
            Point::new(vec![EF::from_u64(1), EF::from_u64(2)]),
            EF::from_u64(9),
            VariableOrder::Prefix,
        );
        assert!(!statement.is_empty());
        assert_eq!(statement.len(), 1);
    }

    #[test]
    #[should_panic]
    fn adding_a_constraint_rejects_mismatched_arity() {
        // The selector and local variable counts must fill the declared space.
        // Here 1 + 1 = 2, but the group spans 3 variables.
        let mut statement = NextStatement::<EF>::initialize(3);
        statement.add_evaluated_constraint(
            Point::new(vec![EF::ONE]),
            Point::new(vec![EF::ONE]),
            EF::ZERO,
            VariableOrder::Prefix,
        );
    }

    #[test]
    fn weights_at_is_selector_eq_times_successor_value() {
        // Invariant: a slot-local weight at a full point factors as
        //   eq(selector, selector_row) * (successor value of the local point).
        //
        // Both factors use independent references:
        // - the successor value comes from the dense successor table interpolated
        //   at the local row, a different route than the closed-form weight, and
        // - the selector gate is the textbook equality product computed by hand.
        let mut rng = SmallRng::seed_from_u64(0);
        let selector = Point::<EF>::rand(&mut rng, 1);
        let local = Point::<EF>::rand(&mut rng, 2);

        let mut statement = NextStatement::<EF>::initialize(3);
        statement.add_evaluated_constraint(
            selector.clone(),
            local.clone(),
            EF::ZERO,
            VariableOrder::Prefix,
        );

        // Prefix order: the selector variable leads the full row.
        let row = Point::<EF>::rand(&mut rng, 3);
        let (selector_row, local_row) = row.split_at(1);
        let successor = Poly::new_next_from_point(local.as_slice()).eval_base(&local_row);
        let gate = selector
            .iter()
            .zip(selector_row.iter())
            .map(|(&si, &xi)| si * xi + (EF::ONE - si) * (EF::ONE - xi))
            .product::<EF>();
        let expected = gate * successor;

        assert_eq!(
            statement.weights_at(&row).collect::<Vec<_>>(),
            vec![expected]
        );
    }

    #[test]
    fn combine_builds_the_scaled_successor_table() {
        // Invariant: combining one full-space constraint writes the dense
        // successor weight table scaled by the constraint's challenge power,
        // and the scalar side picks up the same power times the claim.
        let mut rng = SmallRng::seed_from_u64(1);
        let local = Point::<EF>::rand(&mut rng, 4);
        let claim = EF::from_u64(7);

        let mut statement = NextStatement::<EF>::initialize(4);
        statement.add_evaluated_constraint(
            Point::new(Vec::new()),
            local.clone(),
            claim,
            VariableOrder::Prefix,
        );

        let alpha = EF::from_u64(3);
        let shift = 2;
        let scale = alpha.exp_u64(shift as u64);

        let mut weights = Poly::<EF>::zero(4);
        let mut sum = EF::ZERO;
        statement.combine(&mut weights, &mut sum, alpha, shift);

        // Naive golden: the full-space successor table scaled by alpha^shift.
        let table = Poly::new_next_from_point(local.as_slice());
        for (&got, &want) in weights.as_slice().iter().zip(table.iter()) {
            assert_eq!(got, scale * want);
        }
        assert_eq!(sum, scale * claim);
    }

    #[test]
    fn combine_evals_accumulates_challenge_powers() {
        // Two claims weighted by gamma^1 and gamma^2 with gamma = 2.
        let mut statement = NextStatement::<EF>::initialize(2);
        let point = Point::new(vec![EF::ONE, EF::ZERO]);
        statement.add_evaluated_constraint(
            Point::new(Vec::new()),
            point.clone(),
            EF::from_u64(3),
            VariableOrder::Prefix,
        );
        statement.add_evaluated_constraint(
            Point::new(Vec::new()),
            point,
            EF::from_u64(5),
            VariableOrder::Prefix,
        );

        let mut acc = EF::ZERO;
        statement.combine_evals(&mut acc, EF::from_u64(2), 1);
        // gamma^1 * 3 + gamma^2 * 5 = 2 * 3 + 4 * 5 = 26.
        assert_eq!(acc, EF::from_u64(26));
    }

    #[test]
    fn combine_packed_agrees_with_combine() {
        // The packed weight table unpacks to the same scalars as the dense path.
        let mut rng = SmallRng::seed_from_u64(2);
        let k = 6;
        let local = Point::<EF>::rand(&mut rng, k);

        let mut statement = NextStatement::<EF>::initialize(k);
        statement.add_evaluated_constraint(
            Point::new(Vec::new()),
            local,
            EF::from_u64(5),
            VariableOrder::Prefix,
        );

        let alpha = EF::from_u64(2);
        let mut scalar = Poly::<EF>::zero(k);
        let mut scalar_sum = EF::ZERO;
        statement.combine(&mut scalar, &mut scalar_sum, alpha, 0);

        let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
        let mut packed = Poly::<<EF as ExtensionField<F>>::ExtensionPacking>::zero(k - k_pack);
        let mut packed_sum = EF::ZERO;
        statement.combine_packed::<F>(&mut packed, &mut packed_sum, alpha, 0);

        // Unpack the SIMD lanes back into hypercube order and compare.
        let unpacked = <<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<
            F,
            EF,
        >>::to_ext_iter(packed.as_slice().iter().copied())
        .collect::<Vec<_>>();
        assert_eq!(scalar.as_slice(), &unpacked[..]);
        assert_eq!(packed_sum, scalar_sum);
    }
}
