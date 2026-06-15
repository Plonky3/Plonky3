//! Shared per-table claim state for the stacked-sumcheck provers.
//!
//! - Both binding modes record the same openings.
//! - Both batch them the same way under one challenge.
//! - The committed polynomial is the only mode-specific datum.
//! - The prefix prover keeps that polynomial separately and embeds this state.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, dot_product};

use crate::Claim;
use crate::layout::witness::{Table, TablePlacement};
use crate::layout::{ProverMultiClaim, ProverVirtualClaim};

/// Opening claims recorded against one stacked polynomial, shared by both binding modes.
///
/// - Holds the layout context needed to batch the recorded claims under one challenge.
/// - Both provers embed this and add only their mode-specific state on top.
#[derive(Debug, Clone)]
pub struct StackedClaims<F: Field, EF: ExtensionField<F>> {
    /// Source tables behind the stacked polynomial.
    pub(crate) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(crate) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(crate) num_variables: usize,
    /// Number of preprocessing rounds consumed before residual sumcheck.
    pub(crate) folding: usize,
    /// Concrete claims recorded per source table.
    ///
    /// # Invariants
    ///
    /// - Every opening stored here is tied to a concrete source column.
    /// - Virtual openings never enter this map.
    /// - Claims are appended in insertion order.
    pub(crate) claim_map: Vec<Vec<ProverMultiClaim<F, EF>>>,
    /// Virtual claims sampled directly on the stacked polynomial.
    pub(crate) virtual_claims: Vec<ProverVirtualClaim<EF>>,
}

impl<F: Field, EF: ExtensionField<F>> StackedClaims<F, EF> {
    /// Creates empty claim state for a freshly committed stacked polynomial.
    ///
    /// # Arguments
    ///
    /// - `tables` — source tables behind the stacked polynomial.
    /// - `placements` — per-table slot assignment inside the stacked polynomial.
    /// - `num_variables` — arity of the stacked polynomial.
    /// - `folding` — preprocessing rounds consumed before residual sumcheck.
    pub(crate) fn new(
        tables: Vec<Table<F>>,
        placements: Vec<TablePlacement>,
        num_variables: usize,
        folding: usize,
    ) -> Self {
        // One empty claim list per source table.
        // Virtual claims live in their own bucket.
        let claim_map = (0..tables.len()).map(|_| Vec::new()).collect();
        Self {
            tables,
            placements,
            num_variables,
            folding,
            claim_map,
            virtual_claims: Vec::new(),
        }
    }

    /// Returns the number of preprocessing rounds.
    pub(crate) const fn folding(&self) -> usize {
        self.folding
    }

    /// Returns the number of variables of the stacked polynomial.
    pub(crate) const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the number of variables of table `id`.
    pub(crate) fn num_variables_table(&self, id: usize) -> usize {
        self.tables[id].num_variables()
    }

    /// Returns the total number of concrete openings recorded so far.
    pub(crate) fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(ProverMultiClaim::len))
            .sum()
    }

    /// Walks concrete claims in placement order.
    ///
    /// - Outer: placements, in the order the witness laid them out.
    /// - Inner: the claims recorded against each placement's source table.
    pub(crate) fn concrete_claims(&self) -> impl Iterator<Item = &ProverMultiClaim<F, EF>> {
        self.placements
            .iter()
            .flat_map(|placement| self.claim_map[placement.idx()].iter())
    }

    /// Computes the batched claimed sum from concrete and virtual openings.
    ///
    /// # Identity
    ///
    /// ```text
    ///     sum = sum_{i}  alpha^i * eval_i
    /// ```
    ///
    /// # Alpha ordering
    ///
    /// Powers of `alpha` are handed out in insertion order:
    ///
    /// - Outer: placements, in the order the witness laid them out.
    /// - Middle: claims recorded against that placement's source table.
    /// - Inner: openings inside each claim, in the order they were recorded.
    ///
    /// # Virtual claims
    ///
    /// - Virtual evaluations continue the same alpha sequence.
    /// - They start at `alpha^n`, with `n` the total concrete opening count.
    ///
    /// # Verifier agreement
    ///
    /// The verifier walks its claim registry with the same three-loop order,
    /// so both sides assign the same `alpha^i` to the same claim point.
    pub(crate) fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        let mut alphas = alpha.powers();

        // Walk every concrete opening in the canonical insertion order.
        //     placements -> claims -> current openings -> next openings
        // Each opening consumes the next power of alpha, matching the verifier.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                // Current group first.
                for opening in claim.current_openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
                // Next group second, continuing the same power sequence.
                for opening in claim.next_openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
            }
        }

        // Virtual claims continue the alpha sequence right after the concrete ones.
        sum += dot_product::<EF, _, _>(
            self.virtual_claims.iter().map(Claim::eval),
            alpha.shifted_powers(alpha.exp_u64(self.num_claims() as u64)),
        );

        sum
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;

    use super::*;
    use crate::layout::opening::{EqSvoPartials, NextSvoPartials, Opening};
    use crate::strategy::VariableOrder;
    use crate::svo::SvoPoint;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    // A single zero column of the given arity, enough to drive the table getters.
    fn table(arity: usize) -> Table<F> {
        Table::new(vec![Poly::<F>::zero(arity)])
    }

    // Placement pointing back at source table `idx`.
    // Selectors stay empty: only the table index is read by these methods.
    fn placement(idx: usize) -> TablePlacement {
        TablePlacement::new(idx, vec![])
    }

    // Concrete claim with the given current-group and next-group opening evaluations.
    // The shared point and per-round payloads are empty.
    // `sum`, `num_claims`, and `concrete_claims` read only the opening evals.
    fn claim(current: &[u64], next: &[u64]) -> ProverMultiClaim<F, EF> {
        // Empty SVO point: zero folded variables.
        let point = SvoPoint::<F, EF>::new_unpacked(0, &Point::new(vec![]), VariableOrder::Prefix);
        // Current openings carry the eq-weight payload type.
        let current_openings = current
            .iter()
            .map(|&e| Opening::new_with_data(0, EF::from_u64(e), EqSvoPartials::new(vec![])))
            .collect();
        // Next openings carry the successor payload type.
        let next_openings = next
            .iter()
            .map(|&e| Opening::new_with_data(0, EF::from_u64(e), NextSvoPartials::new(vec![])))
            .collect();
        ProverMultiClaim::new(point, current_openings, next_openings)
    }

    #[test]
    fn new_starts_empty_and_records_shape() {
        // Invariant:
        //     new allocates one empty claim bucket per table and stores the shape verbatim.
        //
        // Fixture state:
        //     two tables (arity 3, arity 5), stacked arity 6, folding 2.
        let sc = StackedClaims::<F, EF>::new(
            vec![table(3), table(5)],
            vec![placement(1), placement(0)],
            6,
            2,
        );

        // The scalar shape fields come back exactly as supplied.
        assert_eq!(sc.folding(), 2);
        assert_eq!(sc.num_variables(), 6);

        // Per-table arities are read straight from the source tables.
        assert_eq!(sc.num_variables_table(0), 3);
        assert_eq!(sc.num_variables_table(1), 5);

        // No openings recorded yet.
        assert_eq!(sc.num_claims(), 0);
        assert!(sc.concrete_claims().next().is_none());

        // One bucket per source table, and no virtual claims.
        assert_eq!(sc.claim_map.len(), 2);
        assert!(sc.virtual_claims.is_empty());
    }

    #[test]
    fn num_claims_counts_openings_in_both_groups() {
        // Invariant:
        //     num_claims sums the openings of both groups across every recorded claim.
        //
        // Fixture state:
        //     one table; one claim of 2 current + 1 next, then a claim of 1 current.
        let mut sc = StackedClaims::<F, EF>::new(vec![table(2)], vec![placement(0)], 2, 0);
        sc.claim_map[0].push(claim(&[3, 5], &[7]));
        sc.claim_map[0].push(claim(&[11], &[]));

        // 3 openings + 1 opening = 4 consumed challenge powers.
        assert_eq!(sc.num_claims(), 4);
    }

    #[test]
    fn concrete_claims_walks_in_placement_order() {
        // Invariant:
        //     concrete_claims yields claims in placement order, not source-table order.
        //
        // Fixture state:
        //     placements list table 1 before table 0 (largest-first layout).
        //     bucket 0 holds a claim with current eval 100.
        //     bucket 1 holds a claim with current eval 200.
        let mut sc = StackedClaims::<F, EF>::new(
            vec![table(2), table(2)],
            vec![placement(1), placement(0)],
            3,
            0,
        );
        sc.claim_map[0].push(claim(&[100], &[]));
        sc.claim_map[1].push(claim(&[200], &[]));

        // Placement order is [table 1, table 0], so bucket 1's claim comes first.
        let evals: Vec<EF> = sc
            .concrete_claims()
            .map(|c| c.current_openings()[0].eval())
            .collect();
        assert_eq!(evals, vec![EF::from_u64(200), EF::from_u64(100)]);
    }

    #[test]
    fn sum_weights_openings_by_alpha_powers() {
        // Invariant:
        //     sum = Σ_i alpha^i * eval_i, walked as
        //     placements -> claims -> current openings -> next openings,
        //     with virtual claims continuing at alpha^(concrete opening count).
        //
        // Fixture state (alpha = 2):
        //     claim A: current [3, 5], next [7]  -> powers 0, 1, 2
        //     claim B: current [11]              -> power 3
        //     virtual: eval 13                   -> power 4 = num_claims
        //
        //     3*1 + 5*2 + 7*4 + 11*8 + 13*16 = 337
        let mut sc = StackedClaims::<F, EF>::new(vec![table(2)], vec![placement(0)], 2, 0);
        sc.claim_map[0].push(claim(&[3, 5], &[7]));
        sc.claim_map[0].push(claim(&[11], &[]));
        sc.virtual_claims.push(Claim {
            point: Point::new(vec![]),
            eval: EF::from_u64(13),
            data: Vec::new(),
        });

        // Concrete openings fix the virtual claim's starting power at alpha^4.
        assert_eq!(sc.num_claims(), 4);
        assert_eq!(sc.sum(EF::from_u64(2)), EF::from_u64(337));
    }
}
