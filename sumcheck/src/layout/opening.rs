//! Opened claims on stacked tables.

use alloc::vec::Vec;

use p3_field::{Field, add_scaled_slice_in_place};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::Claim;
use crate::svo::{
    SvoAccumulators, SvoPoint, calculate_product_accumulator, evals_01inf_grid_prefix,
};

/// Multi-opening claim over an SVO point.
pub type ProverMultiClaim<F, EF> =
    MultiClaim<EF, SvoPoint<F, EF>, EqSvoPartials<EF>, NextSvoPartials<EF>>;
/// Virtual claim carrying precomputed SVO accumulators.
pub type ProverVirtualClaim<EF> = Claim<EF, Point<EF>, SvoAccumulators<EF>>;

/// Opening on the verifier side: column index plus claimed evaluation.
pub type VerifierOpening<EF> = Opening<EF, ()>;
/// Multi-opening claim over a plain point on the verifier side.
pub type VerifierMultiClaim<EF> = MultiClaim<EF, Point<EF>, (), ()>;
/// Virtual evaluation claim on the stacked polynomial (verifier side).
pub type VerifierVirtualClaim<EF> = Claim<EF, Point<EF>, ()>;

/// Equality-weight partial-evaluation table produced for one round of the small-value-optimization preprocessing.
///
/// Holds the multilinear residual left active after that round has been folded.
#[derive(Debug, Clone)]
pub struct EqPartials<EF: Field> {
    /// Active equality-weight residual for this round.
    pub(crate) poly: Poly<EF>,
}

impl<EF: Field> EqPartials<EF> {
    /// Wraps an already-built per-round residual.
    pub const fn new(poly: Poly<EF>) -> Self {
        Self { poly }
    }

    /// Builds an all-zero residual.
    ///
    /// # Arguments
    ///
    /// - `num_variables` — arity of the multilinear residual to allocate.
    pub fn zero(num_variables: usize) -> Self {
        Self {
            // Start from the additive identity so callers can fold contributions in afterward.
            poly: Poly::zero(num_variables),
        }
    }

    /// Folds another round's residual into this one, weighted by a challenge power.
    ///
    /// # Arguments
    ///
    /// - `other` — source residual to add in.
    /// - `scale` — multiplier applied to every source coefficient.
    ///
    /// # Panics
    ///
    /// - The two residuals must share the same arity.
    pub fn accumulate(&mut self, other: &Self, scale: EF) {
        // Invariant: only residuals over the same variable space can be added coefficient-wise.
        assert_eq!(self.poly.num_variables(), other.poly.num_variables());

        // Coefficient-wise fused multiply-add: out[i] += scale * other[i].
        // The shared kernel packs the slices and runs the add over SIMD lanes.
        add_scaled_slice_in_place(self.poly.as_mut_slice(), other.poly.as_slice(), scale);
    }

    /// Returns the active equality-weight residual.
    pub const fn poly(&self) -> &Poly<EF> {
        &self.poly
    }

    /// Adds prefix-layout equality accumulator contributions for one round.
    ///
    /// The active SVO variables are the low bits, so the equality weight is a
    /// product over those variables and can be folded with the multilinear
    /// product accumulator.
    ///
    /// # Arguments
    ///
    /// - `p_active`: the opening point restricted to the active SVO variables of this round.
    /// - `acc0`: running accumulator for the round polynomial evaluated at `0`.
    /// - `acc_inf`: running accumulator for the round polynomial evaluated at `inf`.
    ///
    /// # Panics
    ///
    /// - If `p_active` is empty.
    /// - If the stored payload does not have one variable per active coordinate.
    /// - If `acc0` or `acc_inf` does not have length `3^(active_len - 1)`.
    pub(crate) fn accumulate_prefix(&self, p_active: &[EF], acc0: &mut [EF], acc_inf: &mut [EF]) {
        // One field element per active coordinate fixed this round.
        let active_len = p_active.len();
        // A round always folds at least one active coordinate.
        assert!(active_len > 0);
        // The cached payload must span exactly the active coordinates.
        assert_eq!(self.poly().num_variables(), active_len);

        // Each ternary grid third over the remaining active-1 coordinates has 3^(active_len-1) rows.
        let stride = 3usize.pow((active_len - 1) as u32);
        assert_eq!(acc0.len(), stride);
        assert_eq!(acc_inf.len(), stride);

        // Build the multilinear equality weight as a product over the active point coordinates.
        let eq_active = Poly::new_from_point(p_active, EF::ONE);
        // Fold the equality weight against the payload, keeping only the 0 and inf grid thirds.
        let [term0, term_inf] =
            calculate_product_accumulator(active_len, eq_active.as_slice(), self.poly().as_slice());

        // Add the 0-evaluation contribution of this opening into the running accumulator.
        acc0.iter_mut()
            .zip(term0.iter())
            .for_each(|(out, &value)| *out += value);
        // Add the inf-evaluation (leading-coefficient) contribution likewise.
        acc_inf
            .iter_mut()
            .zip(term_inf.iter())
            .for_each(|(out, &value)| *out += value);
    }

    /// Adds suffix-layout equality accumulator contributions for one round.
    ///
    /// The active SVO variables are the high bits, so both the equality weight
    /// and the payload are expanded to the `{0, 1, inf}` grid and multiplied
    /// pointwise on the `0` and `inf` thirds.
    ///
    /// # Arguments
    ///
    /// - `p_active`: the opening point restricted to the active SVO variables of this round.
    /// - `acc0`: running accumulator for the round polynomial evaluated at `0`.
    /// - `acc_inf`: running accumulator for the round polynomial evaluated at `inf`.
    ///
    /// # Panics
    ///
    /// - If `p_active` is empty.
    /// - If the stored payload does not have one variable per active coordinate.
    /// - If `acc0` or `acc_inf` does not have length `3^(active_len - 1)`.
    pub(crate) fn accumulate_suffix(&self, p_active: &[EF], acc0: &mut [EF], acc_inf: &mut [EF]) {
        // One field element per active coordinate fixed this round.
        let active_len = p_active.len();
        // A round always folds at least one active coordinate.
        assert!(active_len > 0);
        // The cached payload must span exactly the active coordinates.
        assert_eq!(self.poly().num_variables(), active_len);

        // Each ternary grid third over the remaining active-1 coordinates has 3^(active_len-1) rows.
        let stride = 3usize.pow((active_len - 1) as u32);
        assert_eq!(acc0.len(), stride);
        assert_eq!(acc_inf.len(), stride);

        // Expand the equality weight from the 2^l hypercube to the 3^l ternary grid.
        let eq_grid = evals_01inf_grid_prefix(Poly::new_from_point(p_active, EF::ONE).as_slice());
        // Expand the cached payload to the same ternary grid.
        let acc_grid = evals_01inf_grid_prefix(self.poly().as_slice());

        // The grid spans the full ternary cube so the 0-face and inf-face slices are well defined.
        debug_assert_eq!(eq_grid.len(), 3 * stride);
        debug_assert_eq!(acc_grid.len(), 3 * stride);

        // The first third of the grid fixes the leading active coordinate to 0.
        acc0.iter_mut()
            .zip(eq_grid[..stride].iter().zip(acc_grid[..stride].iter()))
            .for_each(|(out, (&eq, &eval))| *out += eq * eval);
        // The last third fixes the leading active coordinate to inf (its leading coefficient).
        acc_inf
            .iter_mut()
            .zip(
                eq_grid[2 * stride..]
                    .iter()
                    .zip(acc_grid[2 * stride..].iter()),
            )
            .for_each(|(out, (&eq, &eval))| *out += eq * eval);
    }
}

/// Equality-weight preprocessing payload for one ordinary opening.
///
/// Carries one residual table per round of the small-value-optimization preprocessing.
#[derive(Debug, Clone)]
pub struct EqSvoPartials<EF: Field> {
    /// One residual table per preprocessing round, in round order.
    pub(crate) rounds: Vec<EqPartials<EF>>,
}

impl<EF: Field> EqSvoPartials<EF> {
    /// Wraps the per-round residual tables.
    pub const fn new(rounds: Vec<EqPartials<EF>>) -> Self {
        Self { rounds }
    }

    /// Returns the per-round residual tables in round order.
    pub fn rounds(&self) -> &[EqPartials<EF>] {
        &self.rounds
    }
}

/// Preprocessing residuals for one round of a repeat-last successor opening.
///
/// The repeat-last successor view evaluates a column at the index one past each point.
///
/// Its weight splits into three carry-state components held here.
#[derive(Debug, Clone)]
pub struct NextPartials<EF: Field> {
    /// Residual paired with the carry-has-finished state.
    pub(crate) done: Poly<EF>,
    /// Residual paired with the carry-still-propagating state.
    pub(crate) carry: Poly<EF>,
    /// Residual paired with the repeat-of-the-last-coordinate state.
    pub(crate) omega: Poly<EF>,
}

impl<EF: Field> NextPartials<EF> {
    /// Wraps already-built residuals for the three carry-state components.
    pub const fn new(done: Poly<EF>, carry: Poly<EF>, omega: Poly<EF>) -> Self {
        Self { done, carry, omega }
    }

    /// Builds an all-zero residual for each of the three components.
    ///
    /// # Arguments
    ///
    /// - `num_variables` — arity of each residual to allocate.
    pub fn zero(num_variables: usize) -> Self {
        Self {
            // Each carry-state component starts from the additive identity.
            done: Poly::zero(num_variables),
            carry: Poly::zero(num_variables),
            omega: Poly::zero(num_variables),
        }
    }

    /// Builds the three repeat-last-successor state tables for a point.
    ///
    /// # Overview
    ///
    /// The repeat-last successor map sends Boolean row `x` to row `x + 1`, with the
    /// last row mapping to itself.
    /// Its weight against a point splits into three tables indexed by Boolean rows:
    ///
    /// - a "done" table equal to the equality weight shifted up by one row,
    /// - a "carry" table holding the wrap-in weight at the all-zeros row,
    /// - an "omega" table holding the wrap-out weight at the all-ones row.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///     rows:   0      1      2    ...   2^n - 1
    ///     done:   0    eq[0]  eq[1]  ...   eq[2^n-2]
    ///     carry:  B      0      0    ...      0
    ///     omega:  0      0      0    ...      B
    ///
    ///     B = product of all point coordinates (the all-ones corner weight)
    /// ```
    pub(crate) fn from_point(point: &[EF]) -> Self {
        // Number of point coordinates, one per Boolean variable.
        let num_variables = point.len();
        // Number of Boolean rows over those variables.
        let num_rows = 1 << num_variables;

        // The all-ones corner weight is the product of all coordinates.
        let boundary = point.iter().copied().product::<EF>();
        // Carry enters only at the all-zeros row (the row that wraps in).
        let mut carry = EF::zero_vec(num_rows);
        // Omega exits only at the all-ones row (the row that repeats).
        let mut omega = EF::zero_vec(num_rows);
        carry[0] = boundary;
        omega[num_rows - 1] = boundary;

        // Equality weights of the point over all Boolean rows.
        let eq = Poly::new_from_point(point, EF::ONE);
        // Done table is the equality table shifted up by one row, with row 0 left at zero.
        let mut done = EF::zero_vec(num_rows);
        if num_rows > 1 {
            done[1..].copy_from_slice(&eq.as_slice()[..num_rows - 1]);
        }

        Self::new(Poly::new(done), Poly::new(carry), Poly::new(omega))
    }

    /// Folds another round's residuals into this one, weighted by a challenge power.
    ///
    /// # Arguments
    ///
    /// - `other` — source residuals to add in.
    /// - `scale` — multiplier applied to every source coefficient.
    ///
    /// # Panics
    ///
    /// - Each component must share its arity with the matching source component.
    pub fn accumulate(&mut self, other: &Self, scale: EF) {
        // Invariant: components are added only across matching variable spaces.
        assert_eq!(self.done.num_variables(), other.done.num_variables());
        assert_eq!(self.carry.num_variables(), other.carry.num_variables());
        assert_eq!(self.omega.num_variables(), other.omega.num_variables());

        // Each component is a coefficient-wise fused multiply-add out[i] += scale * other[i].
        // The shared kernel packs the slices and runs the add over SIMD lanes.
        add_scaled_slice_in_place(self.done.as_mut_slice(), other.done.as_slice(), scale);
        add_scaled_slice_in_place(self.carry.as_mut_slice(), other.carry.as_slice(), scale);
        add_scaled_slice_in_place(self.omega.as_mut_slice(), other.omega.as_slice(), scale);
    }

    /// Returns the carry-has-finished residual.
    pub const fn done(&self) -> &Poly<EF> {
        &self.done
    }

    /// Returns the carry-still-propagating residual.
    pub const fn carry(&self) -> &Poly<EF> {
        &self.carry
    }

    /// Returns the repeat-of-the-last-coordinate residual.
    pub const fn omega(&self) -> &Poly<EF> {
        &self.omega
    }

    /// Adds suffix-layout successor accumulator contributions for one round.
    ///
    /// # Overview
    ///
    /// - The stored payloads are the active-variable successor data for this round.
    /// - Both the successor state tables at the active point and the payloads are expanded to the `{0, 1, inf}` grid.
    /// - The round polynomial values at `0` and `inf` are accumulated by summing the three state-times-data products.
    ///
    /// # Arguments
    ///
    /// - `p_active`: the opening point restricted to the active SVO variables of this round.
    /// - `acc0`: running accumulator for the round polynomial evaluated at `0`.
    /// - `acc_inf`: running accumulator for the round polynomial evaluated at `inf`.
    ///
    /// # Panics
    ///
    /// - If `p_active` is empty.
    /// - If any stored payload does not span the active coordinates.
    /// - If `acc0` or `acc_inf` does not have length `3^(active_len - 1)`.
    pub(crate) fn accumulate_suffix(&self, p_active: &[EF], acc0: &mut [EF], acc_inf: &mut [EF]) {
        // One field element per active coordinate fixed this round.
        let active_len = p_active.len();
        // A round always folds at least one active coordinate.
        assert!(active_len > 0);
        // All three payloads must span exactly the active coordinates.
        assert_eq!(self.done().num_variables(), active_len);
        assert_eq!(self.carry().num_variables(), active_len);
        assert_eq!(self.omega().num_variables(), active_len);

        // Each ternary grid third over the remaining active-1 coordinates has 3^(active_len-1) rows.
        let stride = 3usize.pow((active_len - 1) as u32);
        assert_eq!(acc0.len(), stride);
        assert_eq!(acc_inf.len(), stride);

        // Build the three successor state tables for the active point.
        // TODO: carry and omega polys are sparse.
        let active = Self::from_point(p_active);

        // Expand every state and data table from the hypercube to the ternary grid.
        let carry_grid = evals_01inf_grid_prefix(active.carry().as_slice());
        let done_grid = evals_01inf_grid_prefix(active.done().as_slice());
        let omega_grid = evals_01inf_grid_prefix(active.omega().as_slice());
        let done_data_grid = evals_01inf_grid_prefix(self.done().as_slice());
        let carry_data_grid = evals_01inf_grid_prefix(self.carry().as_slice());
        let omega_data_grid = evals_01inf_grid_prefix(self.omega().as_slice());

        // Every grid spans the full ternary cube so the 0-face and inf-face slices are well defined.
        debug_assert_eq!(carry_grid.len(), 3 * stride);
        debug_assert_eq!(done_grid.len(), 3 * stride);
        debug_assert_eq!(omega_grid.len(), 3 * stride);
        debug_assert_eq!(done_data_grid.len(), 3 * stride);
        debug_assert_eq!(carry_data_grid.len(), 3 * stride);
        debug_assert_eq!(omega_data_grid.len(), 3 * stride);

        // First grid third: leading active coordinate fixed to 0; sum the three state-data products.
        acc0.iter_mut()
            .zip(
                done_grid[..stride]
                    .iter()
                    .zip(done_data_grid[..stride].iter()),
            )
            .zip(
                carry_grid[..stride]
                    .iter()
                    .zip(carry_data_grid[..stride].iter()),
            )
            .zip(
                omega_grid[..stride]
                    .iter()
                    .zip(omega_data_grid[..stride].iter()),
            )
            .for_each(
                |(((out, (&done, &done_data)), (&carry, &carry_data)), (&omega, &omega_data))| {
                    *out += done * done_data + carry * carry_data + omega * omega_data;
                },
            );

        // Last grid third: leading active coordinate fixed to inf; same three-term product.
        acc_inf
            .iter_mut()
            .zip(
                done_grid[2 * stride..]
                    .iter()
                    .zip(done_data_grid[2 * stride..].iter()),
            )
            .zip(
                carry_grid[2 * stride..]
                    .iter()
                    .zip(carry_data_grid[2 * stride..].iter()),
            )
            .zip(
                omega_grid[2 * stride..]
                    .iter()
                    .zip(omega_data_grid[2 * stride..].iter()),
            )
            .for_each(
                |(((out, (&done, &done_data)), (&carry, &carry_data)), (&omega, &omega_data))| {
                    *out += done * done_data + carry * carry_data + omega * omega_data;
                },
            );
    }

    /// Adds prefix-layout successor accumulator contributions for one round.
    ///
    /// # Overview
    ///
    /// - The stored payloads are the active-variable successor data for this round.
    /// - In prefix layout each active state factors into a product of one state table and one data payload.
    /// - Three product accumulators (equality, done, omega) sum their `0` and `inf` contributions into the running accumulators.
    ///
    /// # Arguments
    ///
    /// - `p_active`: the opening point restricted to the active SVO variables of this round.
    /// - `acc0`: running accumulator for the round polynomial evaluated at `0`.
    /// - `acc_inf`: running accumulator for the round polynomial evaluated at `inf`.
    ///
    /// # Panics
    ///
    /// - If `p_active` is empty.
    /// - If any stored payload does not span the active coordinates.
    /// - If `acc0` or `acc_inf` does not have length `3^(active_len - 1)`.
    pub(crate) fn accumulate_prefix(&self, p_active: &[EF], acc0: &mut [EF], acc_inf: &mut [EF]) {
        // One field element per active coordinate fixed this round.
        let active_len = p_active.len();
        // A round always folds at least one active coordinate.
        assert!(active_len > 0);
        // All three payloads must span exactly the active coordinates.
        assert_eq!(self.done().num_variables(), active_len);
        assert_eq!(self.carry().num_variables(), active_len);
        assert_eq!(self.omega().num_variables(), active_len);

        // Each ternary grid third over the remaining active-1 coordinates has 3^(active_len-1) rows.
        let stride = 3usize.pow((active_len - 1) as u32);
        assert_eq!(acc0.len(), stride);
        assert_eq!(acc_inf.len(), stride);

        // Successor state tables for the active point.
        let active = Self::from_point(p_active);
        // Plain equality weights of the active point.
        let eq_active = Poly::new_from_point(p_active, EF::ONE);

        // Each active state term is a product of one weight table and one data payload.
        let terms = [
            // Equality weight times the shifted-done data payload.
            calculate_product_accumulator(active_len, eq_active.as_slice(), self.done().as_slice()),
            // Done state times the carry-into-next data payload.
            calculate_product_accumulator(
                active_len,
                active.done().as_slice(),
                self.carry().as_slice(),
            ),
            // Omega boundary state times the boundary data payload.
            calculate_product_accumulator(
                active_len,
                active.omega().as_slice(),
                self.omega().as_slice(),
            ),
        ];

        // Fold every term's 0 and inf contributions into the running accumulators.
        for [term0, term_inf] in terms {
            acc0.iter_mut()
                .zip(term0.iter())
                .for_each(|(out, &value)| *out += value);
            acc_inf
                .iter_mut()
                .zip(term_inf.iter())
                .for_each(|(out, &value)| *out += value);
        }
    }
}

/// Preprocessing payload for one repeat-last successor opening.
///
/// Carries the three-component residuals for every round of the small-value-optimization preprocessing.
#[derive(Debug, Clone)]
pub struct NextSvoPartials<EF: Field> {
    /// One three-component residual set per preprocessing round, in round order.
    pub(crate) rounds: Vec<NextPartials<EF>>,
}

impl<EF: Field> NextSvoPartials<EF> {
    /// Wraps the per-round residual sets.
    pub const fn new(rounds: Vec<NextPartials<EF>>) -> Self {
        Self { rounds }
    }

    /// Returns the per-round residual sets in round order.
    pub fn rounds(&self) -> &[NextPartials<EF>] {
        &self.rounds
    }
}

/// Single opening of one polynomial at a shared evaluation point.
///
/// # Virtual openings
///
/// A polynomial index of `None` represents a virtual opening detached from any source column.
///
/// Strategies create these internally as transient claims during accumulator batching.
#[derive(Debug, Clone)]
pub struct Opening<EF: Field, Data> {
    /// Source column index, or `None` for a virtual opening.
    pub(crate) poly_idx: Option<usize>,
    /// Value of the polynomial at the shared claim point.
    pub(crate) eval: EF,
    /// Strategy-specific payload attached to this opening.
    pub(crate) data: Data,
}

impl<EF: Field, Data> Opening<EF, Data> {
    /// Returns the evaluation.
    pub const fn eval(&self) -> EF {
        self.eval
    }

    /// Returns the source column index, or `None` for a virtual opening.
    pub const fn poly_idx(&self) -> Option<usize> {
        self.poly_idx
    }

    /// Returns the strategy-specific payload.
    pub const fn data(&self) -> &Data {
        &self.data
    }

    /// Builds an opening on a concrete column carrying a strategy payload.
    ///
    /// # Arguments
    ///
    /// - `poly_idx` — source column index inside the owning table.
    /// - `eval`     — value of the opened view at the shared claim point.
    /// - `data`     — preprocessing payload attached to this opening.
    pub const fn new_with_data(poly_idx: usize, eval: EF, data: Data) -> Self {
        Self {
            // A concrete column index marks this as non-virtual.
            poly_idx: Some(poly_idx),
            eval,
            data,
        }
    }
}

impl<EF: Field> Opening<EF, ()> {
    /// Builds an opening for a concrete table column.
    ///
    /// # Arguments
    ///
    /// - `poly_idx` — source column index inside the owning table.
    /// - `eval`     — value of that column at the shared claim point.
    pub const fn new(poly_idx: usize, eval: EF) -> Self {
        Self {
            poly_idx: Some(poly_idx),
            eval,
            data: (),
        }
    }
}

/// A batch of openings that share one evaluation point.
///
/// Current openings evaluate a column at the point.
///
/// Next openings evaluate the repeat-last successor view at the same point.
///
/// ```text
///     point     ── shared by every opening
///     current   [evaluate-column opening_0, ...]
///     next      [repeat-last opening_0, ...]
/// ```
///
/// # Alpha-ordering contract
///
/// - Each recorded opening consumes one power of the batching challenge.
/// - The canonical ordering is insertion order, walked as:
///     - placements, in witness-layout order,
///     - claims inside each placement, in recording order,
///     - current openings inside each claim, in recording order,
///     - next openings inside each claim, in recording order.
/// - Prover and verifier walk the same nested loop.
/// - That forces the challenge-power-to-opening mapping to agree when the transcripts mirror each other.
#[derive(Debug, Clone)]
pub struct MultiClaim<EF: Field, Point, EqData, NextData> {
    /// Shared evaluation point of every opening in the batch.
    pub(super) point: Point,
    /// Openings that evaluate a column at the shared point.
    pub(super) current_openings: Vec<Opening<EF, EqData>>,
    /// Openings that evaluate the repeat-last successor view at the shared point.
    pub(super) next_openings: Vec<Opening<EF, NextData>>,
}

impl<EF: Field, Point, EqData, NextData> MultiClaim<EF, Point, EqData, NextData> {
    /// Builds a batch whose openings all share one evaluation point.
    ///
    /// # Arguments
    ///
    /// - `point`            — evaluation point shared by every opening.
    /// - `current_openings` — openings that evaluate a column at the point.
    /// - `next_openings`    — openings that evaluate the repeat-last successor view at the point.
    pub const fn new(
        point: Point,
        current_openings: Vec<Opening<EF, EqData>>,
        next_openings: Vec<Opening<EF, NextData>>,
    ) -> Self {
        Self {
            point,
            current_openings,
            next_openings,
        }
    }

    /// Returns the shared evaluation point.
    pub const fn point(&self) -> &Point {
        &self.point
    }

    /// Returns the total number of openings across both groups.
    pub const fn len(&self) -> usize {
        // Total consumed challenge powers equals current plus next openings.
        self.current_openings.len() + self.next_openings.len()
    }

    /// Returns whether the batch holds no openings in either group.
    pub const fn is_empty(&self) -> bool {
        // Empty only when neither group contributes an opening.
        self.current_openings.is_empty() && self.next_openings.is_empty()
    }

    /// Returns the openings that evaluate a column at the shared point.
    pub fn current_openings(&self) -> &[Opening<EF, EqData>] {
        &self.current_openings
    }

    /// Returns the openings that evaluate the repeat-last successor view at the shared point.
    pub fn next_openings(&self) -> &[Opening<EF, NextData>] {
        &self.next_openings
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;

    // Brute-force reference: materialize each successor state table row by row from the closed form.
    fn next_partials_from_point_reference(point: &[F]) -> [Poly<F>; 3] {
        let num_variables = point.len();
        let num_rows = 1 << num_variables;
        let mut carry = F::zero_vec(num_rows);
        let mut done = F::zero_vec(num_rows);
        let mut omega = F::zero_vec(num_rows);

        // Evaluate the closed-form successor decomposition at every Boolean row.
        for row_idx in 0..num_rows {
            let row = Point::hypercube(row_idx, num_variables);
            let (c, d, o) = Point::eval_next(point, row.as_slice());
            carry[row_idx] = c;
            done[row_idx] = d;
            omega[row_idx] = o;
        }

        [Poly::new(carry), Poly::new(done), Poly::new(omega)]
    }

    #[test]
    fn next_partials_from_point_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Invariant: the fast sparse construction equals the brute-force per-row tables.
        // Fixture state: variable counts 0..=8, with 16 random points each.
        for num_variables in 0..=8 {
            for _ in 0..16 {
                let point = Point::<F>::rand(&mut rng, num_variables);
                // Fast path: closed-form sparse construction of the three state tables.
                let actual = NextPartials::from_point(point.as_slice());
                // Slow path: one closed-form evaluation per Boolean row.
                let [carry, done, omega] = next_partials_from_point_reference(point.as_slice());

                assert_eq!(actual.carry(), &carry);
                assert_eq!(actual.done(), &done);
                assert_eq!(actual.omega(), &omega);
            }
        }
    }

    #[test]
    fn opening_new_sets_poly_idx_eval_and_unit_data() {
        // Fixture: concrete opening at column 3 with value 42.
        let opening: Opening<F, ()> = Opening::new(3, F::from_u64(42));

        // Constructor must wrap the index and attach unit payload.
        //
        //     poly_idx  = Some(3)
        //     eval      = 42
        //     data      = ()
        assert_eq!(opening.poly_idx(), Some(3));
        assert_eq!(opening.eval(), F::from_u64(42));
        assert_eq!(opening.data(), &());
    }

    #[test]
    fn opening_struct_literal_supports_virtual_form() {
        // Virtual openings are built via struct literal because they require
        // poly_idx = None, which the public constructor does not expose.
        let opening: Opening<F, ()> = Opening {
            poly_idx: None,
            eval: F::from_u64(7),
            data: (),
        };

        // Accessor must surface the None sentinel.
        assert_eq!(opening.poly_idx(), None);
        assert_eq!(opening.eval(), F::from_u64(7));
    }

    #[test]
    fn opening_accessors_reflect_non_unit_data() {
        // Fixture: Data = Vec<F> — used by strategies carrying per-round partials.
        let data = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let opening: Opening<F, Vec<F>> = Opening {
            poly_idx: Some(0),
            eval: F::from_u64(99),
            data: data.clone(),
        };

        // Accessor returns the payload untouched (same length, same values).
        assert_eq!(opening.data(), &data);
    }

    #[test]
    fn opening_clone_copies_every_field() {
        // Regression: derived Clone must not drop or swap any field.
        let original: Opening<F, Vec<F>> = Opening {
            poly_idx: Some(5),
            eval: F::from_u64(11),
            data: vec![F::from_u64(10)],
        };
        let cloned = original.clone();

        assert_eq!(cloned.poly_idx(), original.poly_idx());
        assert_eq!(cloned.eval(), original.eval());
        assert_eq!(cloned.data(), original.data());
    }

    #[test]
    fn multi_claim_new_preserves_shared_point() {
        // Fixture: two openings at an arbitrary point value (u32 chosen for simplicity).
        let openings = vec![
            Opening::<F, ()>::new(0, F::from_u64(1)),
            Opening::<F, ()>::new(1, F::from_u64(2)),
        ];
        let claim = MultiClaim::<F, u32, (), ()>::new(100, openings, Vec::new());

        // Constructor forwards the point and the current openings verbatim.
        // No next openings were supplied, so that group stays empty.
        //
        //     point              = 100
        //     current_openings   = [col 0, col 1]  → len 2
        //     next_openings      = []              → len 0
        assert_eq!(*claim.point(), 100);
        assert_eq!(claim.current_openings().len(), 2);
        assert_eq!(claim.next_openings().len(), 0);
    }

    #[test]
    fn multi_claim_len_matches_openings_count() {
        // Cover empty, singleton, and multi-opening batches with one loop.
        for n in [0usize, 1, 4] {
            let openings: Vec<Opening<F, ()>> = (0..n)
                .map(|i| Opening::new(i, F::from_u64(i as u64)))
                .collect();
            let claim = MultiClaim::<F, u32, (), ()>::new(0, openings, Vec::new());

            // Invariant: reported length equals constructed size.
            assert_eq!(claim.len(), n);
        }
    }

    #[test]
    fn multi_claim_is_empty_is_true_iff_no_openings() {
        // Empty claim: is_empty must be true.
        let empty: MultiClaim<F, u32, (), ()> = MultiClaim::new(0, vec![], vec![]);
        assert!(empty.is_empty());

        // Non-empty claim: is_empty must be false.
        let filled =
            MultiClaim::<F, u32, (), ()>::new(0, vec![Opening::new(0, F::from_u64(1))], vec![]);
        assert!(!filled.is_empty());
    }

    #[test]
    fn multi_claim_current_openings_returns_insertion_order() {
        // Build openings in a non-trivial poly_idx order.
        //
        //     insertion: [col 2, col 0, col 1]
        //     openings(): must be the same slice, same order.
        let expected = vec![
            Opening::<F, ()>::new(2, F::from_u64(20)),
            Opening::<F, ()>::new(0, F::from_u64(0)),
            Opening::<F, ()>::new(1, F::from_u64(10)),
        ];
        let claim = MultiClaim::<F, u32, (), ()>::new(0, expected.clone(), Vec::new());

        for (i, got) in claim.current_openings().iter().enumerate() {
            assert_eq!(got.poly_idx(), expected[i].poly_idx());
            assert_eq!(got.eval(), expected[i].eval());
        }
    }

    #[test]
    fn multi_claim_clone_preserves_point_and_openings() {
        // Invariant: derived Clone copies the point and both opening groups.
        //
        // Fixture state:
        //     point            = 77
        //     current_openings = [col 1, col 2]
        //     next_openings    = [col 3]
        let claim = MultiClaim::<F, u32, (), ()>::new(
            77,
            vec![
                Opening::new(1, F::from_u64(5)),
                Opening::new(2, F::from_u64(6)),
            ],
            vec![Opening::new(3, F::from_u64(7))],
        );
        let cloned = claim.clone();

        // Point and total length survive the clone unchanged.
        assert_eq!(*cloned.point(), *claim.point());
        assert_eq!(cloned.len(), claim.len());
        // Every current opening matches its source index and value.
        for (a, b) in cloned
            .current_openings()
            .iter()
            .zip(claim.current_openings())
        {
            assert_eq!(a.poly_idx(), b.poly_idx());
            assert_eq!(a.eval(), b.eval());
        }
        // Every next opening matches its source index and value.
        for (a, b) in cloned.next_openings().iter().zip(claim.next_openings()) {
            assert_eq!(a.poly_idx(), b.poly_idx());
            assert_eq!(a.eval(), b.eval());
        }
    }
}
