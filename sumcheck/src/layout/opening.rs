//! Opened claims on stacked tables.

use alloc::vec::Vec;

use p3_field::{Field, add_scaled_slice_in_place};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::Claim;
use crate::svo::{SvoAccumulators, SvoPoint};

/// Multi-opening claim over an SVO point.
pub type ProverMultiClaim<F, EF> =
    MultiClaim<EF, SvoPoint<F, EF>, EqSvoPartials<EF>, NextSvoPartials<EF>>;
/// Prover opening that evaluates a column at the sampled point.
pub type ProverEqOpening<EF> = Opening<EF, EqSvoPartials<EF>>;
/// Prover opening that evaluates the repeat-last successor view at the sampled point.
pub type ProverNextOpening<EF> = Opening<EF, NextSvoPartials<EF>>;
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

    use super::*;

    type F = BabyBear;

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
