//! Opened claims on stacked tables.

use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::Claim;
use crate::svo::{SvoAccumulators, SvoPoint};

/// Multi-opening claim over an SVO point.
pub type ProverMultiClaim<F, EF> =
    MultiClaim<EF, SvoPoint<F, EF>, EqSvoPartials<EF>, NextSvoPartials<EF>>;
/// Ordinary prover opening payload.
pub type ProverEqOpening<EF> = Opening<EF, EqSvoPartials<EF>>;
/// Repeat-last Next prover opening payload.
pub type ProverNextOpening<EF> = Opening<EF, NextSvoPartials<EF>>;
/// Virtual claim carrying precomputed SVO accumulators.
pub type ProverVirtualClaim<EF> = Claim<EF, Point<EF>, SvoAccumulators<EF>>;

/// Opening on the verifier side: column index plus claimed evaluation.
pub type VerifierOpening<EF> = Opening<EF, ()>;
/// Multi-opening claim over a plain point on the verifier side.
pub type VerifierMultiClaim<EF> = MultiClaim<EF, Point<EF>, (), ()>;
/// Virtual evaluation claim on the stacked polynomial (verifier side).
pub type VerifierVirtualClaim<EF> = Claim<EF, Point<EF>, ()>;

/// Per-round ordinary Eq data for one SVO round.
#[derive(Debug, Clone)]
pub struct EqPartials<EF: Field> {
    pub(crate) poly: Poly<EF>,
}

impl<EF: Field> EqPartials<EF> {
    /// Builds the active Eq data for one SVO round.
    pub const fn new(poly: Poly<EF>) -> Self {
        Self { poly }
    }

    /// Builds zero active Eq data over `num_variables`.
    pub fn zero(num_variables: usize) -> Self {
        Self {
            poly: Poly::zero(num_variables),
        }
    }

    /// Adds `scale * other` into this round's polynomial.
    pub fn accumulate(&mut self, other: &Self, scale: EF) {
        assert_eq!(self.poly.num_variables(), other.poly.num_variables());

        self.poly
            .as_mut_slice()
            .iter_mut()
            .zip(other.poly.iter())
            .for_each(|(out, &f)| *out += scale * f);
    }

    /// Returns the active Eq data polynomial.
    pub const fn poly(&self) -> &Poly<EF> {
        &self.poly
    }
}

/// Per-round ordinary Eq SVO partial evaluations for one opening.
#[derive(Debug, Clone)]
pub struct EqSvoPartials<EF: Field> {
    pub(crate) rounds: Vec<EqPartials<EF>>,
}

impl<EF: Field> EqSvoPartials<EF> {
    /// Builds Eq SVO partials from per-round data.
    pub const fn new(rounds: Vec<EqPartials<EF>>) -> Self {
        Self { rounds }
    }

    /// Returns all per-round partials.
    pub fn rounds(&self) -> &[EqPartials<EF>] {
        &self.rounds
    }
}

/// Active-data polynomials used by one Next SVO round.
#[derive(Debug, Clone)]
pub struct NextPartials<EF: Field> {
    pub(crate) done: Poly<EF>,
    pub(crate) carry: Poly<EF>,
    pub(crate) omega: Poly<EF>,
}

impl<EF: Field> NextPartials<EF> {
    /// Builds the active data triple for one SVO round.
    pub const fn new(done: Poly<EF>, carry: Poly<EF>, omega: Poly<EF>) -> Self {
        Self { done, carry, omega }
    }

    /// Builds a zero active data triple over `num_variables`.
    pub fn zero(num_variables: usize) -> Self {
        Self {
            done: Poly::zero(num_variables),
            carry: Poly::zero(num_variables),
            omega: Poly::zero(num_variables),
        }
    }

    /// Adds `scale * other` into each component.
    pub fn accumulate(&mut self, other: &Self, scale: EF) {
        assert_eq!(self.done.num_variables(), other.done.num_variables());
        assert_eq!(self.carry.num_variables(), other.carry.num_variables());
        assert_eq!(self.omega.num_variables(), other.omega.num_variables());

        self.done
            .as_mut_slice()
            .iter_mut()
            .zip(other.done.iter())
            .for_each(|(out, &f)| *out += scale * f);
        self.carry
            .as_mut_slice()
            .iter_mut()
            .zip(other.carry.iter())
            .for_each(|(out, &f)| *out += scale * f);
        self.omega
            .as_mut_slice()
            .iter_mut()
            .zip(other.omega.iter())
            .for_each(|(out, &f)| *out += scale * f);
    }

    /// Data multiplied by the active `done` carry-state polynomial.
    pub const fn done(&self) -> &Poly<EF> {
        &self.done
    }

    /// Data multiplied by the active `carry` carry-state polynomial.
    pub const fn carry(&self) -> &Poly<EF> {
        &self.carry
    }

    /// Data multiplied by the active `omega` repeat-last polynomial.
    pub const fn omega(&self) -> &Poly<EF> {
        &self.omega
    }
}

/// Per-round Method 4 data for one Next opening.
#[derive(Debug, Clone)]
pub struct NextSvoPartials<EF: Field> {
    pub(crate) rounds: Vec<NextPartials<EF>>,
}

impl<EF: Field> NextSvoPartials<EF> {
    /// Builds Method 4 partials from per-round data.
    pub const fn new(rounds: Vec<NextPartials<EF>>) -> Self {
        Self { rounds }
    }

    /// Returns all per-round partials.
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

    /// Builds a concrete opening with strategy-specific payload.
    pub const fn new_with_data(poly_idx: usize, eval: EF, data: Data) -> Self {
        Self {
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
/// ```text
///     point     ── shared by every opening
///     current   [ordinary opening_0, ...]
///     next      [next opening_0, ...]
/// ```
///
/// # Alpha-ordering contract
///
/// - Each recorded opening consumes one power of the batching challenge.
/// - The canonical ordering is insertion order, walked as:
///     - placements, in witness-layout order,
///     - claims inside each placement, in recording order,
///     - current openings inside each claim, in the order they entered `eval`,
///     - next openings inside each claim, in the order they entered `eval`.
/// - Prover and verifier walk the same three-loop nest, so the alpha-to-claim
///   mapping is forced to agree when the transcripts mirror each other.
#[derive(Debug, Clone)]
pub struct MultiClaim<EF: Field, Point, EqData, NextData> {
    /// Shared evaluation point of every opening in the batch.
    pub(super) point: Point,
    /// Ordinary openings attached to the shared point.
    pub(super) current_openings: Vec<Opening<EF, EqData>>,
    /// Repeat-last Next openings attached to the shared point.
    pub(super) next_openings: Vec<Opening<EF, NextData>>,
}

impl<EF: Field, Point, EqData, NextData> MultiClaim<EF, Point, EqData, NextData> {
    /// Builds a batch sharing `point`, holding the given openings.
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

    /// Returns the number of openings.
    pub const fn len(&self) -> usize {
        self.current_openings.len() + self.next_openings.len()
    }

    /// Returns whether the batch holds no openings.
    pub const fn is_empty(&self) -> bool {
        self.current_openings.is_empty() && self.next_openings.is_empty()
    }

    /// Returns ordinary openings.
    pub fn current_openings(&self) -> &[Opening<EF, EqData>] {
        &self.current_openings
    }

    /// Returns repeat-last Next openings.
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

        // Constructor must forward the point and the openings vector verbatim.
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
        // Regression: derived Clone must copy both the point and the Vec contents.
        let claim = MultiClaim::<F, u32, (), ()>::new(
            77,
            vec![
                Opening::new(1, F::from_u64(5)),
                Opening::new(2, F::from_u64(6)),
            ],
            vec![Opening::new(3, F::from_u64(7))],
        );
        let cloned = claim.clone();

        assert_eq!(*cloned.point(), *claim.point());
        assert_eq!(cloned.len(), claim.len());
        for (a, b) in cloned
            .current_openings()
            .iter()
            .zip(claim.current_openings())
        {
            assert_eq!(a.poly_idx(), b.poly_idx());
            assert_eq!(a.eval(), b.eval());
        }
        for (a, b) in cloned.next_openings().iter().zip(claim.next_openings()) {
            assert_eq!(a.poly_idx(), b.poly_idx());
            assert_eq!(a.eval(), b.eval());
        }
    }
}
