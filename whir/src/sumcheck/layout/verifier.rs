//! Verifier-side reconstruction of the stacked layout and its claims.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, dot_product};
use p3_multilinear_util::point::Point;

use crate::constraints::Constraint;
use crate::constraints::statement::EqStatement;
use crate::sumcheck::Claim;
use crate::sumcheck::layout::LayoutStrategy;
use crate::sumcheck::layout::opening::{VerifierMultiClaim, VerifierOpening, VerifierVirtualClaim};
use crate::sumcheck::layout::plan::{LayoutShape, plan_layout};
use crate::sumcheck::layout::witness::{Selector, TablePlacement};
use crate::sumcheck::table::TableShape;

/// Verifier-side layout and claim registry.
#[derive(Debug, Clone)]
pub struct Verifier<F: Field, EF: ExtensionField<F>> {
    /// Per-table placement metadata inside the stacked polynomial.
    placements: Vec<TablePlacement>,
    /// Side-map: source-table index → position into `placements`.
    ///
    /// Lets lookups by source table run in O(1) instead of a linear scan over placements.
    placement_by_table: Vec<usize>,
    /// Number of variables of the stacked polynomial.
    k: usize,
    /// Concrete claims recorded per source table.
    claim_map: Vec<Vec<VerifierMultiClaim<EF>>>,
    /// Virtual claims sampled directly on the stacked polynomial.
    virtual_claims: Vec<VerifierVirtualClaim<EF>>,
    /// Whether selector bitstrings are reversed and laid out after local bits.
    strategy: LayoutStrategy,
    /// Marker to tie the challenger's field type
    _marker: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>> Verifier<F, EF> {
    /// Reconstructs the verifier-side layout from table shapes.
    ///
    /// # Algorithm
    ///
    /// - Sort table indices by arity ascending.
    /// - Iterate reversed, so largest tables get placed first.
    /// - Each column occupies one slot of size `2^arity`.
    /// - Stacked arity equals `log2` of total stacked size, rounded up.
    pub fn new(tables: &[TableShape], strategy: LayoutStrategy) -> Self {
        // Delegate slot assignment to the shared planner (same routine as the prover).
        let shapes: Vec<LayoutShape> = tables
            .iter()
            .map(|t| LayoutShape {
                arity: t.num_variables(),
                width: t.width(),
            })
            .collect();
        let (k, mut placements) = plan_layout(&shapes);
        if strategy.reverse_selectors {
            placements
                .iter_mut()
                .for_each(TablePlacement::reverse_selectors);
        }

        // Build the side-map: source-table index → index into `placements`.
        let mut placement_by_table = vec![0usize; tables.len()];
        for (i, p) in placements.iter().enumerate() {
            placement_by_table[p.idx()] = i;
        }

        Self {
            placements,
            placement_by_table,
            k,
            // One (empty) concrete-claim list per source table.
            claim_map: (0..tables.len()).map(|_| Vec::new()).collect(),
            // No virtual claims recorded yet.
            virtual_claims: Vec::new(),
            strategy,
            _marker: PhantomData,
        }
    }

    /// Returns the arity of the source table at the given index.
    pub fn num_variables_table(&self, table_idx: usize) -> usize {
        // Look up this table's placement; every column shares the same selector bit-width.
        let placement = self.placement(table_idx);
        let selector_vars = placement
            .selectors()
            .first()
            .map(Selector::num_variables)
            .unwrap_or(0);
        // Table arity is the stacked arity minus the selector bits that address the slot.
        self.k - selector_vars
    }

    /// Records concrete opening claims for one table.
    ///
    /// # Arguments
    ///
    /// - `table_idx`  — source table index.
    /// - `polys`      — column indices that were opened; must be non-empty.
    /// - `evals`      — claimed evaluations, aligned with the column list.
    /// - `challenger` — Fiat–Shamir transcript.
    ///
    /// # Fiat–Shamir
    ///
    /// - Samples the opening point internally from the challenger.
    /// - Absorbs the evaluations into the transcript.
    /// - Mirrors exactly the prover's `eval` absorption order.
    ///
    /// # Panics
    ///
    /// - Columns list must be non-empty.
    /// - Column list and evaluation list must have equal length.
    /// - Every column index must be in range for this table.
    pub fn add_claim<Ch>(
        &mut self,
        table_idx: usize,
        polys: &[usize],
        evals: &[EF],
        challenger: &mut Ch,
    ) where
        Ch: p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    {
        let placement = self.placement(table_idx);
        // Preconditions.
        assert!(
            !polys.is_empty(),
            "opening schedule must name at least one column"
        );
        assert_eq!(polys.len(), evals.len());
        assert!(polys.iter().all(|&i| i < placement.num_polys()));

        // Sample the local-frame opening point from the transcript.
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            self.num_variables_table(table_idx),
        );

        // Absorb the evals into the transcript; mirrors the prover's eval.
        challenger.observe_algebra_slice(evals);

        // Pair each column index with its claimed evaluation into an opening.
        let openings = polys
            .iter()
            .zip(evals.iter())
            .map(|(&poly_idx, &eval)| VerifierOpening::new(poly_idx, eval))
            .collect();

        // Store the batch under this table's claim list.
        self.claim_map[table_idx].push(VerifierMultiClaim::new(point, openings));
    }

    /// Records a virtual evaluation claim on the full stacked polynomial.
    ///
    /// # Fiat–Shamir
    ///
    /// - Samples the opening point from the challenger.
    /// - Absorbs the evaluation into the transcript.
    /// - Mirrors exactly the prover's `add_virtual_eval` absorption order.
    pub fn add_virtual_eval<Ch>(&mut self, eval: EF, challenger: &mut Ch)
    where
        Ch: p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    {
        // Sample a challenge point covering every stacked variable.
        let point = Point::expand_from_univariate(challenger.sample_algebra_element(), self.k);
        // Absorb the evaluation into the transcript.
        challenger.observe_algebra_element(eval);
        // Record the claim with unit payload (verifier side carries no extras).
        self.virtual_claims.push(Claim {
            point,
            eval,
            data: (),
        });
    }

    /// Computes the batched claimed sum across concrete and virtual openings.
    ///
    /// ```text
    ///     sum = sum_{i}  alpha^i * eval_i
    /// ```
    ///
    /// # Traversal order
    ///
    /// - Concrete openings first, walked in stacked-polynomial order.
    /// - Virtual claims continue the alpha sequence after the concrete ones.
    pub fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        let mut alphas = alpha.powers();

        // Concrete openings: three loops, no filter. Alphas match insertion order.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
            }
        }

        // Virtual claims: continue the alpha sequence right after the concrete ones.
        sum += dot_product::<EF, _, _>(
            self.virtual_claims.iter().map(VerifierVirtualClaim::eval),
            alpha.powers().skip(self.num_claims()),
        );

        sum
    }

    /// Builds the verifier-side equality constraint batching every claim.
    ///
    /// # Contributions
    ///
    /// - Concrete opening: equality at the selector-lifted claim point.
    /// - Virtual claim: equality at the full stacked point.
    pub fn constraint(&self, alpha: EF) -> Constraint<F, EF> {
        // Output statement spans the full stacked variable space.
        let mut eq_statement = EqStatement::initialize(self.k);

        // Concrete contributions: walk each opening in insertion order and lift
        // its claim point through the selector for that opening's column.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    // The opening's column tells us which selector to lift through.
                    let col = opening.poly_idx().unwrap();
                    let lifted = if self.strategy.reverse_selectors {
                        placement.selectors()[col].lift_suffix(claim.point())
                    } else {
                        placement.selectors()[col].lift_prefix(claim.point())
                    };
                    eq_statement.add_evaluated_constraint(lifted, opening.eval());
                }
            }
        }

        // Virtual contributions: claim points already span the full stacked space.
        for claim in &self.virtual_claims {
            eq_statement.add_evaluated_constraint(claim.point.clone(), claim.eval);
        }

        // Wrap the assembled statement into an alpha-batched constraint.
        Constraint::new_eq_only(alpha, eq_statement)
    }

    /// Returns the placement metadata for the given source table.
    ///
    /// O(1) via the side-map built at construction.
    fn placement(&self, table_idx: usize) -> &TablePlacement {
        &self.placements[self.placement_by_table[table_idx]]
    }

    /// Returns the number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(VerifierMultiClaim::len))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::sumcheck::strategy::VariableOrder;
    use crate::sumcheck::tests::{EF, F, challenger};

    const fn prefix_strategy() -> LayoutStrategy {
        LayoutStrategy::new(false, VariableOrder::Prefix)
    }

    #[test]
    fn table_shape_new_stores_dimensions_and_derives_getters() {
        // Invariant:
        //     A shape of log-rows = k and width = w exposes:
        //         k()     = k
        //         width() = w
        //
        // Fixture state:
        //     k = 3, width = 2 → height = 8.
        let shape = TableShape::new(3, 2);

        // Check each derived quantity.
        assert_eq!(shape.num_variables(), 3);
        assert_eq!(shape.width(), 2);
    }

    #[test]
    #[should_panic]
    fn table_shape_new_rejects_zero_width() {
        // Invariant:
        //     A shape with zero columns is disallowed.
        let _ = TableShape::new(3, 0);
    }

    #[test]
    fn verifier_new_places_largest_table_first() {
        // Invariant:
        //     new() sorts tables by arity ascending and places them reversed,
        //     so the first placement corresponds to the largest table.
        //
        // Fixture state:
        //     shapes: [arity 9 (t0), arity 10 (t1)]
        //     expected placement order: [t1 (arity 10), t0 (arity 9)]
        let shapes = vec![TableShape::new(9, 2), TableShape::new(10, 2)];
        let verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());

        // First placement must point back at the larger table.
        assert_eq!(verifier.placements[0].idx(), 1);
        // Second placement must point back at the smaller table.
        assert_eq!(verifier.placements[1].idx(), 0);
    }

    #[test]
    fn verifier_num_variables_table_derives_from_selector_bits() {
        // Invariant:
        //     num_variables_table equals the stacked arity minus the selector bits.
        //
        // Fixture state:
        //     shapes: [arity 9 × 2 cols, arity 10 × 2 cols]
        //     stacked rows = 2^9 * 2 + 2^10 * 2 = 3 * 2^10 → k = ceil(log2) = 11
        //     table 0 (arity 9) → selector bits = 2 → num_variables_table = 9
        //     table 1 (arity 10) → selector bits = 1 → num_variables_table = 10
        let shapes = vec![TableShape::new(9, 2), TableShape::new(10, 2)];
        let verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());

        // Check derivation per table.
        assert_eq!(verifier.num_variables_table(0), 9);
        assert_eq!(verifier.num_variables_table(1), 10);
    }

    #[test]
    fn verifier_add_claim_increments_num_claims() {
        // Invariant:
        //     num_claims sums the opening count across every recorded claim.
        //
        // Fixture state:
        //     fresh verifier:                      0 openings
        //     after add_claim(table 0, [0, 1]):    2 openings
        //     after add_claim(table 1, [0]):       3 openings
        let shapes = vec![TableShape::new(9, 2), TableShape::new(10, 2)];
        let mut verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());
        assert_eq!(verifier.num_claims(), 0);

        // Fresh Fiat-Shamir transcript; add_claim samples the point and absorbs.
        let mut ch = challenger();

        // Add two openings on table 0; count jumps from 0 to 2.
        verifier.add_claim(0, &[0, 1], &[EF::from_u64(7), EF::from_u64(11)], &mut ch);
        assert_eq!(verifier.num_claims(), 2);

        // Add one opening on table 1; count jumps from 2 to 3.
        verifier.add_claim(1, &[0], &[EF::from_u64(13)], &mut ch);
        assert_eq!(verifier.num_claims(), 3);
    }

    #[test]
    fn verifier_sum_weights_concrete_then_virtual_by_alpha_powers() {
        // Invariant:
        //     sum(alpha) = eval_0 * alpha^0 + eval_1 * alpha^1 + virtual * alpha^2
        //     for two concrete openings followed by one virtual claim.
        //
        // Fixture state:
        //     concrete openings: column 0 → 7, column 1 → 11 on table 0
        //     virtual claim:     value 13
        //     alpha:             5
        let shapes = vec![TableShape::new(9, 2)];
        let mut verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());

        // Fresh transcript; points are sampled inside add_claim / add_virtual_eval.
        let mut ch = challenger();

        // Concrete claim: two columns of table 0.
        verifier.add_claim(0, &[0, 1], &[EF::from_u64(7), EF::from_u64(11)], &mut ch);

        // Virtual claim: covers the full stacked space.
        verifier.add_virtual_eval(EF::from_u64(13), &mut ch);

        // Manual batched sum with alpha = 5.
        let alpha = EF::from_u64(5);
        let expected =
            EF::from_u64(7) + alpha * EF::from_u64(11) + alpha.exp_u64(2) * EF::from_u64(13);

        // Check: the helper must match the hand-rolled formula.
        assert_eq!(verifier.sum(alpha), expected);
    }

    #[test]
    fn verifier_sum_empty_returns_zero() {
        // Invariant:
        //     With no claims recorded, sum(alpha) is zero for any alpha.
        let shapes = vec![TableShape::new(9, 1)];
        let verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());

        // Pick a non-trivial alpha to rule out accidental zero cancellation.
        let alpha = EF::from_u64(42);
        assert_eq!(verifier.sum(alpha), EF::ZERO);
    }

    #[test]
    fn verifier_sum_skips_untouched_tables() {
        // Invariant:
        //     A table with no recorded claims contributes zero to the sum
        //     and does not consume alpha powers.
        //
        // Fixture state:
        //     two tables; only table 1 records an opening.
        //     alpha: 3, eval: 9 → expected sum = 9 (alpha^0 only).
        let shapes = vec![TableShape::new(9, 2), TableShape::new(10, 2)];
        let mut verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());

        // Record a single opening on table 1; table 0 stays empty.
        let mut ch = challenger();
        verifier.add_claim(1, &[0], &[EF::from_u64(9)], &mut ch);

        // Check: only one opening → sum equals its eval scaled by alpha^0.
        assert_eq!(verifier.sum(EF::from_u64(3)), EF::from_u64(9));
    }
}
