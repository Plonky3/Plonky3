//! Shared helpers for stacked-polynomial traversal.
//!
//! Both prover modes walk the same nested structure:
//!
//! ```text
//!     placements → columns → claims → filtered openings
//! ```
//!
//! The traversal below packages that walk behind a single named record,
//! so each mode uses one shared routine instead of a hand-written loop.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use crate::sumcheck::layout::opening::{MultiClaim, Opening};
use crate::sumcheck::layout::witness::TablePlacement;

/// Per-opening context surfaced during stacked-polynomial traversal.
///
/// Bundles every quantity the traversal exposes so call sites take one
/// named parameter instead of seven positional ones.
pub(super) struct OpeningVisit<'a, Claim, Opening, EF> {
    /// Source table index.
    pub(super) table_idx: usize,
    /// Claim index within the owning table's claim list.
    pub(super) claim_idx: usize,
    /// Opening index within the owning claim's opening list.
    pub(super) opening_idx: usize,
    /// Borrow of the multi-claim that owns the current opening.
    pub(super) claim: &'a Claim,
    /// Borrow of the current opening.
    pub(super) opening: &'a Opening,
    /// Current power of the batching challenge.
    pub(super) alpha: EF,
    /// Slot range inside the stacked polynomial, in scalar units.
    pub(super) slot: core::ops::Range<usize>,
}

/// Visits every recorded opening in stacked-polynomial order.
///
/// # Visit order
///
/// - Outer loop: placement in insertion order.
/// - Middle loop: column index within that placement.
/// - Inner loop: claims, then openings tied to the current column.
///
/// # Invariants
///
/// - Alpha powers advance by one step per visited opening.
/// - Slot ranges advance once per column, regardless of opening count.
/// - Openings within one claim may surface out of insertion order when
///   columns are opened non-monotonically; the opening-index field keeps
///   alignment with the claim's opening list.
pub(super) fn traverse_openings<F, EF, Point, Data>(
    placements: &[TablePlacement],
    slot_bits: impl Fn(usize) -> usize,
    claim_map: &[Vec<MultiClaim<F, EF, Point, Data>>],
    alpha: EF,
    mut f: impl FnMut(OpeningVisit<'_, MultiClaim<F, EF, Point, Data>, Opening<EF, Data>, EF>),
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Challenge powers produced lazily; one step consumed per visited opening.
    let mut alphas = alpha.powers();
    // Running offset into the stacked polynomial, in scalar units.
    let mut off = 0usize;

    for placement in placements {
        // Each column of this table owns a slot of size 2^num_variables_table.
        let slot_size = 1usize << slot_bits(placement.idx());

        for poly_idx in 0..placement.num_polys() {
            for (claim_idx, claim) in claim_map[placement.idx()].iter().enumerate() {
                for (opening_idx, opening) in claim.openings().iter().enumerate() {
                    // Filter: only visit openings tied to the current column.
                    if opening.poly_idx() == Some(poly_idx) {
                        f(OpeningVisit {
                            table_idx: placement.idx(),
                            claim_idx,
                            opening_idx,
                            claim,
                            opening,
                            alpha: alphas.next().unwrap(),
                            slot: off..off + slot_size,
                        });
                    }
                }
            }
            // Move to the next slot after exhausting its claim list.
            off += slot_size;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::sumcheck::layout::witness::Selector;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    // Condensed visit record captured by tests.
    type VisitRecord = (usize, usize, usize, EF, (usize, usize));

    // Builds a placement with the given column count.
    fn make_placement(table_idx: usize, num_polys: usize) -> TablePlacement {
        let selectors = (0..num_polys).map(|i| Selector::new(1, i % 2)).collect();
        TablePlacement::new(table_idx, selectors)
    }

    // Collects every visit into a flat list of observable quantities.
    fn collect_visits<Point: Clone, Data>(
        placements: &[TablePlacement],
        slot_bits: impl Fn(usize) -> usize,
        claim_map: &[Vec<MultiClaim<F, EF, Point, Data>>],
        alpha: EF,
    ) -> Vec<VisitRecord> {
        let mut out = Vec::new();
        traverse_openings(placements, slot_bits, claim_map, alpha, |v| {
            out.push((
                v.table_idx,
                v.claim_idx,
                v.opening_idx,
                v.alpha,
                (v.slot.start, v.slot.end),
            ));
        });
        out
    }

    #[test]
    fn traverse_openings_ascending_single_claim() {
        // Invariant:
        //     Every opening is visited exactly once, in column-major order.
        //     Alpha advances by one step per visited opening.
        //
        // Fixture state:
        //     one placement, 2 columns, slot_size = 2^3 = 8
        //     one claim opening columns (0, 1) in ascending order
        //     alpha = 2
        //
        // Expected visits:
        //
        //     column 0 → opening 0, alpha^0 = 1, slot 0..8
        //     column 1 → opening 1, alpha^1 = 2, slot 8..16
        let placements = vec![make_placement(0, 2)];
        let claim = MultiClaim::<F, EF, (), ()>::new(
            (),
            vec![
                Opening::new(0, EF::from_u64(11)),
                Opening::new(1, EF::from_u64(13)),
            ],
        );
        let claim_map = vec![vec![claim]];

        // Collect every visit: expect two records for the two openings.
        let got = collect_visits(&placements, |_| 3, &claim_map, EF::from_u64(2));

        // Check: order, opening_idx, alpha power, and slot bounds all match.
        assert_eq!(
            got,
            vec![
                (0, 0, 0, EF::from_u64(1), (0, 8)),
                (0, 0, 1, EF::from_u64(2), (8, 16)),
            ],
        );
    }

    #[test]
    fn traverse_openings_non_ascending_preserves_opening_idx() {
        // Invariant:
        //     Columns are visited in poly-index order (0, 1, ...), but the
        //     opening-index field reports the original insertion slot.
        //
        // Fixture state:
        //     insertion order: [(poly=1, .), (poly=0, .)]
        //     slot_size = 2^2 = 4
        //     alpha = 3
        //
        // Expected visits:
        //
        //     column 0 → opening_idx = 1   (second-inserted opening)
        //     column 1 → opening_idx = 0   (first-inserted opening)
        let placements = vec![make_placement(0, 2)];
        let claim = MultiClaim::<F, EF, (), ()>::new(
            (),
            vec![
                Opening::new(1, EF::from_u64(99)),
                Opening::new(0, EF::from_u64(77)),
            ],
        );
        let claim_map = vec![vec![claim]];

        // Collect every visit: one record per opening yielded by the traversal.
        let got = collect_visits(&placements, |_| 2, &claim_map, EF::from_u64(3));

        // Check: column-major order, opening_idx reflects insertion, alpha advances.
        assert_eq!(
            got,
            vec![
                (0, 0, 1, EF::from_u64(1), (0, 4)),
                (0, 0, 0, EF::from_u64(3), (4, 8)),
            ],
        );
    }

    #[test]
    fn traverse_openings_advances_offset_across_placements() {
        // Invariant:
        //     Slot offset carries over across placements; a new placement
        //     starts immediately after the previous one ends.
        //
        // Fixture state:
        //     placement 0 (table 0): 1 column, slot_size = 4
        //     placement 1 (table 1): 1 column, slot_size = 4
        //     alpha = 5
        //
        // Expected slot layout:
        //
        //     placement 0 column 0 → slot 0..4
        //     placement 1 column 0 → slot 4..8
        let placements = vec![make_placement(0, 1), make_placement(1, 1)];
        let claim0 = MultiClaim::<F, EF, (), ()>::new((), vec![Opening::new(0, EF::from_u64(5))]);
        let claim1 = MultiClaim::<F, EF, (), ()>::new((), vec![Opening::new(0, EF::from_u64(7))]);
        let claim_map = vec![vec![claim0], vec![claim1]];

        // Collect every visit: expect two records, one per placement.
        let got = collect_visits(&placements, |_| 2, &claim_map, EF::from_u64(5));

        // Check: offset stays continuous; alpha advances across placements.
        assert_eq!(
            got,
            vec![
                (0, 0, 0, EF::from_u64(1), (0, 4)),
                (1, 0, 0, EF::from_u64(5), (4, 8)),
            ],
        );
    }

    #[test]
    fn traverse_openings_skips_virtual_openings() {
        // Invariant:
        //     An opening tagged None carries no column association; the
        //     column-filter inside the traversal must drop it.
        //
        // Fixture state:
        //     one placement, one column, one claim with one virtual opening.
        //
        // Expected: zero visits.
        let placements = vec![make_placement(0, 1)];
        let claim = MultiClaim::<F, EF, (), ()>::new(
            (),
            vec![Opening {
                poly_idx: None,
                eval: EF::from_u64(42),
                data: (),
            }],
        );
        let claim_map = vec![vec![claim]];

        // Collect every visit: virtual-only claim must produce none.
        let got = collect_visits(&placements, |_| 1, &claim_map, EF::from_u64(2));

        // Check: the traversal emitted nothing.
        assert!(got.is_empty(), "virtual-only claim must yield no visits");
    }

    #[test]
    fn traverse_openings_sums_alpha_power_sequence() {
        // Invariant:
        //     Over N visits the alpha field takes the values a^0, a^1, ..., a^{N-1}.
        //
        // Fixture state:
        //     N = 5 columns in one placement; one claim opens all of them.
        //     alpha = 4.
        const N: usize = 5;
        let placements = vec![make_placement(0, N)];
        // Build one opening per column so the traversal yields exactly N visits.
        let openings = (0..N)
            .map(|i| Opening::new(i, EF::from_u64(i as u64)))
            .collect();
        let claim = MultiClaim::<F, EF, (), ()>::new((), openings);
        let claim_map = vec![vec![claim]];

        // Collect every visit; 5 visits expected.
        let alpha = EF::from_u64(4);
        let got = collect_visits(&placements, |_| 1, &claim_map, alpha);

        // Check: alpha sequence matches the running a^i.
        let mut expected = EF::ONE;
        for v in &got {
            assert_eq!(v.3, expected);
            expected *= alpha;
        }
        // Check: visit count equals opening count.
        assert_eq!(got.len(), N);
    }
}
