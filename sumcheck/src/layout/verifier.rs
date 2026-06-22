//! Verifier-side reconstruction of the stacked layout and its claims.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, dot_product};
use p3_multilinear_util::point::Point;

use crate::constraints::statement::{EqStatement, NextStatement};
use crate::constraints::{Constraint, Statements};
use crate::layout::LayoutStrategy;
use crate::layout::opening::{VerifierMultiClaim, VerifierOpening, VerifierVirtualClaim};
use crate::layout::plan::{LayoutShape, plan_layout};
use crate::layout::witness::{Selector, TablePlacement};
use crate::strategy::VariableOrder;
use crate::table::{OpeningEvals, OpeningRequest, TableShape};
use crate::{Claim, SumcheckError};

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

    /// Return the layout strategy this verifier was constructed with.
    ///
    /// Downstream protocols read it to dispatch on the binding direction
    /// without threading the strategy through their own state.
    pub const fn strategy(&self) -> LayoutStrategy {
        self.strategy
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
    /// - `batch`      — current and next columns opened at this point.
    /// - `evals`      — claimed evaluations split the same way as the columns.
    /// - `challenger` — Fiat-Shamir transcript.
    ///
    /// # Fiat-Shamir
    ///
    /// - Samples the opening point internally from the transcript.
    /// - Absorbs the evaluations, current group first then next.
    /// - Mirrors exactly the prover-side absorption order.
    ///
    /// # Errors
    ///
    /// - Returns [`SumcheckError::OpeningShapeMismatch`] when the proof's
    ///   evaluations do not match the requested column shape.
    ///
    /// # Panics
    ///
    /// - At least one current or next column must be requested.
    /// - Every column index must be in range for this table.
    pub fn add_claim<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        evals: &OpeningEvals<EF>,
        challenger: &mut Ch,
    ) -> Result<(), SumcheckError>
    where
        Ch: p3_challenger::FieldChallenger<F> + p3_challenger::GrindingChallenger<Witness = F>,
    {
        let placement = self.placement(table_idx);
        // Split the request into its two column groups.
        let current = batch.current();
        let next = batch.next();
        // An empty request would silently record nothing. The schedule is built
        // by the verifier, so an empty request is a caller bug, not a bad proof.
        assert!(
            !batch.is_empty(),
            "opening schedule must name at least one column"
        );
        // The evaluations come from the proof, so a shape mismatch is a malformed
        // proof and must be rejected rather than aborting the verifier.
        if !batch.has_same_shape(evals) {
            return Err(SumcheckError::OpeningShapeMismatch {
                table_idx,
                expected_current: current.len(),
                expected_next: next.len(),
                actual_current: evals.current().len(),
                actual_next: evals.next().len(),
            });
        }
        // Every requested column must address an existing slot in this table.
        assert!(
            current
                .iter()
                .all(|&poly_idx| poly_idx < placement.num_polys())
        );
        assert!(
            next.iter()
                .all(|&poly_idx| poly_idx < placement.num_polys())
        );

        // Sample the local-frame opening point from the transcript.
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            self.num_variables_table(table_idx),
        );

        // Absorb the evals in the same current-then-next order as the prover.
        challenger.observe_algebra_slice(evals.current());
        challenger.observe_algebra_slice(evals.next());

        // Pair each current column with its claimed evaluation.
        let current_openings = current
            .iter()
            .copied()
            .zip(evals.current().iter().copied())
            .map(|(poly_idx, eval)| VerifierOpening::new(poly_idx, eval))
            .collect();
        // Pair each next column with its claimed evaluation.
        let next_openings = next
            .iter()
            .copied()
            .zip(evals.next().iter().copied())
            .map(|(poly_idx, eval)| VerifierOpening::new(poly_idx, eval))
            .collect();

        // Store the batch under this table's claim list.
        self.claim_map[table_idx].push(VerifierMultiClaim::new(
            point,
            current_openings,
            next_openings,
        ));

        Ok(())
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
        let mut concrete = EF::ZERO;
        let mut alphas = alpha.powers();

        // Walk every concrete opening in the canonical insertion order.
        //     placements -> claims -> current openings -> next openings
        // Each opening consumes the next power of alpha, matching the prover.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                // Current group first.
                for opening in claim.current_openings() {
                    concrete += opening.eval() * alphas.next().unwrap();
                }
                // Next group second, continuing the same power sequence.
                for opening in claim.next_openings() {
                    concrete += opening.eval() * alphas.next().unwrap();
                }
            }
        }

        // Virtual claims: continue the alpha sequence right after the concrete ones.
        let virtuals = dot_product::<EF, _, _>(
            self.virtual_claims.iter().map(VerifierVirtualClaim::eval),
            alpha.shifted_powers(alpha.exp_u64(self.num_claims() as u64)),
        );

        concrete + virtuals
    }

    /// Builds the alpha-batched constraint over every recorded claim.
    ///
    /// # Overview
    ///
    /// - A current opening contributes an equality at the selector-lifted claim point.
    /// - A next opening contributes a repeat-last successor weight at that slot.
    /// - A virtual claim contributes an equality at the full stacked point.
    ///
    /// # Why the split
    ///
    /// - The emitted statements preserve the same mixed insertion order the batched sum walks.
    /// - That keeps each statement aligned with the alpha power assigned to its opening.
    pub fn constraint(&self, alpha: EF) -> Constraint<F, EF> {
        // Accumulate statements over the full stacked variable space.
        // The push order mirrors the batched-sum walk, so alpha powers stay aligned.
        let mut statements = Vec::new();

        // Concrete contributions, walked in canonical insertion order.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                // Current group: one equality statement per claim's current openings.
                let mut eq_statement = EqStatement::initialize(self.k);
                for opening in claim.current_openings() {
                    // The column selects which slot lift to apply.
                    // Recorded concrete openings always bind a column, never virtual.
                    let col = opening
                        .poly_idx()
                        .expect("concrete claims only hold column-bound openings");
                    // Lift the slot-local claim point into the full stacked space.
                    // Suffix and prefix layouts place the folded variables at opposite ends.
                    let lifted = if self.strategy.reverse_selectors {
                        placement.selectors()[col].lift_suffix(claim.point())
                    } else {
                        placement.selectors()[col].lift_prefix(claim.point())
                    };
                    eq_statement.add_evaluated_constraint(lifted, opening.eval());
                }
                // Emit only if this claim actually had current openings.
                if !eq_statement.is_empty() {
                    statements.push(Statements::Eq(eq_statement));
                }

                // Next group: one repeat-last successor statement per claim's next openings.
                let mut next_statement = NextStatement::initialize(self.k);
                for opening in claim.next_openings() {
                    // Recorded concrete openings always bind a column, never virtual.
                    let col = opening
                        .poly_idx()
                        .expect("concrete claims only hold column-bound openings");
                    // The successor weight depends on which end the folded variables sit.
                    let var_order = if self.strategy.reverse_selectors {
                        VariableOrder::Suffix
                    } else {
                        VariableOrder::Prefix
                    };
                    // Feed the slot selector, the claim point, the eval, and the layout.
                    next_statement.add_evaluated_constraint(
                        placement.selectors()[col].point(),
                        claim.point().clone(),
                        opening.eval(),
                        var_order,
                    );
                }
                // Emit only if this claim actually had next openings.
                if !next_statement.is_empty() {
                    statements.push(Statements::Next(next_statement));
                }
            }
        }

        // Virtual contributions: claim points already span the full stacked space.
        let mut virtual_statement = EqStatement::initialize(self.k);
        for claim in &self.virtual_claims {
            virtual_statement.add_evaluated_constraint(claim.point.clone(), claim.eval);
        }
        // Emit the virtual block only when at least one virtual claim exists.
        if !virtual_statement.is_empty() {
            statements.push(Statements::Eq(virtual_statement));
        }

        // Wrap the assembled statements into an alpha-batched constraint.
        Constraint::new(alpha, self.k, statements)
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
    use crate::strategy::VariableOrder;
    use crate::table::OpeningBatch;
    use crate::tests::{EF, F, challenger};

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
        let request = OpeningBatch::new(vec![0, 1], Vec::new());
        let evals = OpeningBatch::new(vec![EF::from_u64(7), EF::from_u64(11)], Vec::new());
        verifier.add_claim(0, &request, &evals, &mut ch).unwrap();
        assert_eq!(verifier.num_claims(), 2);

        // Add one opening on table 1; count jumps from 2 to 3.
        let request = OpeningBatch::new(vec![0], Vec::new());
        let evals = OpeningBatch::new(vec![EF::from_u64(13)], Vec::new());
        verifier.add_claim(1, &request, &evals, &mut ch).unwrap();
        assert_eq!(verifier.num_claims(), 3);
    }

    #[test]
    fn verifier_add_claim_rejects_shape_mismatch_without_panicking() {
        // Invariant:
        //     The evaluations come from the proof, so a shape mismatch against
        //     the requested columns is rejected as an error, never a panic.
        //
        // Fixture state:
        //     request names two direct columns, but only one evaluation is given.
        let shapes = vec![TableShape::new(9, 2)];
        let mut verifier: Verifier<F, EF> = Verifier::new(&shapes, prefix_strategy());
        let mut ch = challenger();

        let request = OpeningBatch::new(vec![0, 1], Vec::new());
        let evals = OpeningBatch::new(vec![EF::from_u64(7)], Vec::new());
        let err = verifier
            .add_claim(0, &request, &evals, &mut ch)
            .unwrap_err();

        assert_eq!(
            err,
            SumcheckError::OpeningShapeMismatch {
                table_idx: 0,
                expected_current: 2,
                expected_next: 0,
                actual_current: 1,
                actual_next: 0,
            }
        );
        // The rejected claim left the registry untouched.
        assert_eq!(verifier.num_claims(), 0);
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
        let request = OpeningBatch::new(vec![0, 1], Vec::new());
        let evals = OpeningBatch::new(vec![EF::from_u64(7), EF::from_u64(11)], Vec::new());
        verifier.add_claim(0, &request, &evals, &mut ch).unwrap();

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
        let request = OpeningBatch::new(vec![0], Vec::new());
        let evals = OpeningBatch::new(vec![EF::from_u64(9)], Vec::new());
        verifier.add_claim(1, &request, &evals, &mut ch).unwrap();

        // Check: only one opening → sum equals its eval scaled by alpha^0.
        assert_eq!(verifier.sum(EF::from_u64(3)), EF::from_u64(9));
    }
}
