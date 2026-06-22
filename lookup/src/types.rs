//! Core data types for lookup arguments.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use num_bigint::BigUint;
use p3_air::symbolic::AirLayout;
use p3_air::{Air, SymbolicExpression};
use p3_field::{ExtensionField, Field, PrimeField};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::builder::{SymbolicInteraction, SymbolicLocalInteraction};
use crate::protocol::LookupProtocol;
use crate::symbolic::InteractionSymbolicBuilder;

/// Whether a lookup is confined to one AIR or shared across AIRs on a named bus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// Intra-AIR lookup.
    ///
    /// - Draws a fresh challenge pair, shared with no other lookup.
    /// - So its terminal contribution cannot cancel against another lookup.
    /// - The cross-AIR terminal sum forces it to balance on its own.
    Local,
    /// Cross-AIR lookup on a named bus.
    ///
    /// - Lookups sharing a bus name share one challenge pair.
    /// - Senders and receivers cancel only in the cross-AIR terminal sum.
    Global(String),
}

/// A single lookup argument: element tuples, multiplicities, and one
/// auxiliary column in the permutation trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Lookup<F: Field> {
    /// Local or global (with bus name).
    pub kind: Kind,
    /// Element tuples. Each inner `Vec` is one `(key0, key1, ...)` tuple.
    pub elements: Vec<Vec<SymbolicExpression<F>>>,
    /// Signed multiplicity per element tuple. Same length as `elements`.
    pub multiplicities: Vec<SymbolicExpression<F>>,
    /// Static per-row upper bound on the magnitude of this lookup's count.
    ///
    /// - One term of the multiplicity height-bound soundness check `sum_i w_i * h_i < p`.
    /// - Carried straight from the emitting global interaction.
    /// - Sound only if the AIR constrains the count to respect it on every row.
    /// - Typically `1` for a query and `0` for a table entry being provided.
    /// - Always `0` for intra-AIR lookups, which never cross AIRs.
    pub count_weight: u32,
    /// Slot index for this lookup within the AIR.
    ///
    /// - Selects the challenge pair at offsets `2*column` and `2*column + 1`.
    /// - Owns fraction column `column + 1` of the permutation trace.
    /// - Column `0` of the permutation trace is the shared accumulator.
    pub column: usize,
}

/// All lookups for one AIR, with column indices assigned.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent, bound = "")]
pub struct Lookups<F: Field>(Vec<Lookup<F>>);

impl<F: Field> Lookups<F> {
    /// Extract lookups from an AIR by running symbolic evaluation.
    pub fn from_air<EF, A>(air: &A) -> Self
    where
        EF: ExtensionField<F>,
        A: Air<InteractionSymbolicBuilder<F, EF>>,
        F: Clone + Send + Sync,
    {
        let mut builder = InteractionSymbolicBuilder::<F, EF>::new(AirLayout::from_air(air));
        air.eval(&mut builder);
        Self::from_interactions(builder.global_interactions(), builder.local_interactions())
    }

    /// Build from raw symbolic interactions.
    ///
    /// Local interactions first, then global — matching the LogUp column order.
    fn from_interactions(
        global: &[SymbolicInteraction<F>],
        local: &[SymbolicLocalInteraction<F>],
    ) -> Self {
        let mut lookups = Vec::with_capacity(local.len() + global.len());
        let mut col = 0;

        for i in local {
            let (elements, multiplicities) =
                i.tuples.iter().map(|(f, c)| (f.clone(), c.clone())).unzip();
            lookups.push(Lookup {
                kind: Kind::Local,
                elements,
                multiplicities,
                // Intra-AIR lookups balance within one AIR, so they sit out the cross-AIR bound.
                count_weight: 0,
                column: col,
            });
            col += 1;
        }

        for i in global {
            lookups.push(Lookup {
                kind: Kind::Global(i.bus_name.clone()),
                elements: vec![i.fields.clone()],
                multiplicities: vec![i.count.clone()],
                // Preserve the emitting interaction's weight for the height-bound check.
                count_weight: i.count_weight,
                column: col,
            });
            col += 1;
        }

        Self(lookups)
    }

    /// Fold global lookups on the same bus into shared fraction columns.
    ///
    /// One folded column carries several interactions as a single rational sum:
    /// ```text
    ///     f_col[r] = sum_j  m_j[r] / (prefix_bus - combine(payload_j[r]))
    /// ```
    ///
    /// - All folded interactions sit on one bus.
    /// - So they share one challenge pair, and the per-row sum is well defined.
    /// - A folded column never exceeds the fraction-pin degree `max_degree`.
    /// - Local lookups are grouped by their author already, so they pass through.
    ///
    /// # Soundness
    ///
    /// - Folding only moves a fraction from its own column into a shared one.
    /// - The per-bus rational sum, and every committed terminal, is unchanged.
    /// - So the cross-AIR balance and the per-bus separation still hold.
    /// - Both sides fold in emission order, so they agree on the layout.
    ///
    /// # Performance
    ///
    /// - Folding raises only the merged column's fraction-pin degree.
    /// - The cap holds that degree within the AIR's spare quotient budget.
    /// - Fewer columns then cost strictly less to commit and open.
    ///
    /// # Arguments
    ///
    /// - `gadget` — reports the exact fraction-pin degree of a candidate column.
    /// - `max_degree` — the largest degree that leaves the quotient cost unchanged.
    #[must_use]
    pub fn pack_same_bus<LG: LookupProtocol>(self, gadget: &LG, max_degree: usize) -> Self {
        // Locals keep their leading columns, untouched.
        let mut packed: Vec<Lookup<F>> = Vec::with_capacity(self.0.len());
        // Globals bucket by bus name, in first-appearance order.
        let mut buses: Vec<(String, Vec<Lookup<F>>)> = Vec::new();

        // Sort each lookup: locals straight through, globals into their bus bucket.
        for lookup in self.0 {
            match &lookup.kind {
                Kind::Local => packed.push(lookup),
                Kind::Global(name) => match buses.iter_mut().find(|(n, _)| n == name) {
                    // Bus already seen: append to its bucket.
                    Some((_, members)) => members.push(lookup),
                    None => {
                        // First sighting of this bus: clone the name, then open a bucket.
                        let bus = name.clone();
                        buses.push((bus, vec![lookup]));
                    }
                },
            }
        }

        // Fill columns greedily within each bus.
        // Seal a column when the next interaction would exceed the degree budget.
        for (_name, members) in buses {
            let mut current: Option<Lookup<F>> = None;
            for member in members {
                match current.take() {
                    // A lone interaction always fits: its degree is the unpacked degree.
                    None => current = Some(member),
                    Some(mut cur) => {
                        // Trial-merge: append the candidate's tuples to the open column.
                        let pivot = cur.elements.len();
                        cur.elements.extend(member.elements.iter().cloned());
                        cur.multiplicities
                            .extend(member.multiplicities.iter().cloned());

                        if gadget.constraint_degree(&cur) <= max_degree {
                            // Within budget: keep the merge.
                            // Each tuple holds its own per-row query bound, so the bounds add.
                            cur.count_weight = cur.count_weight.saturating_add(member.count_weight);
                            current = Some(cur);
                        } else {
                            // Over budget: undo the trial, seal the column, start a new one.
                            cur.elements.truncate(pivot);
                            cur.multiplicities.truncate(pivot);
                            packed.push(cur);
                            current = Some(member);
                        }
                    }
                }
            }
            // Flush the last open column for this bus.
            if let Some(cur) = current {
                packed.push(cur);
            }
        }

        // Slot i owns fraction column i + 1, so indices must stay contiguous.
        for (i, lookup) in packed.iter_mut().enumerate() {
            lookup.column = i;
        }

        Self(packed)
    }

    /// Sum of this AIR's lookup weights.
    ///
    /// Forms the `w_i` term in the height-bound check `sum_i w_i * h_i < p`.
    #[must_use]
    pub fn total_count_weight(&self) -> u64 {
        self.0.iter().map(|l| u64::from(l.count_weight)).sum()
    }
}

/// Enforce the LogUp multiplicity height-bound `sum_i w_i * h_i < p`.
///
/// # Why this exists
///
/// - LogUp proves a multiset identity over a base field of characteristic `p`.
/// - A provided entry's multiplicity equals how many queries hit it.
/// - Counted honestly, that multiplicity never exceeds `sum_i w_i * h_i`.
/// - This holds only because each AIR constrains every query count to its weight.
/// - Holding the sum below `p` rules out any multiplicity wrapping modulo `p`.
/// - A wrap would let a prover forge multiplicities and break soundness.
/// - Only public weights and heights are read, so both parties agree on the result.
///
/// # Arguments
///
/// - `lookups` — one lookup set per AIR.
/// - `heights` — one trace height per AIR, aligned with `lookups`.
///
/// # Panics
///
/// - When the two slices have different lengths, which is an orchestration bug.
///
/// # Errors
///
/// - When the weighted sum reaches the field characteristic.
pub fn check_multiplicity_height_bound<F: PrimeField>(
    lookups: &[Lookups<F>],
    heights: &[usize],
) -> Result<(), LookupError> {
    assert_eq!(
        lookups.len(),
        heights.len(),
        "lookups and heights must be aligned per AIR"
    );

    // Accumulate `sum_i w_i * h_i` in 128 bits.
    // A saturating add only caps far past any field size, where the bound already fails.
    let weighted_height_sum = lookups.iter().zip(heights).fold(0u128, |acc, (air, &h)| {
        let term = u128::from(air.total_count_weight()).saturating_mul(h as u128);
        acc.saturating_add(term)
    });

    // Compare against the exact characteristic `p`, valid for any prime size.
    if BigUint::from(weighted_height_sum) >= F::order() {
        return Err(LookupError::MultiplicityHeightBoundExceeded {
            weighted_height_sum,
            field_bits: F::bits(),
        });
    }
    Ok(())
}

impl<F: Field> Deref for Lookups<F> {
    type Target = [Lookup<F>];

    fn deref(&self) -> &[Lookup<F>] {
        &self.0
    }
}

impl<F: Field> AsRef<[Lookup<F>]> for Lookups<F> {
    fn as_ref(&self) -> &[Lookup<F>] {
        &self.0
    }
}

/// Single committed scalar summarising one AIR's lookup contribution.
///
/// # Semantics
///
/// - Holds the row-wise sum of every lookup's rational contribution.
/// - Committed once per AIR that declares at least one lookup.
/// - Absent (no commitment) when the AIR declares no lookup.
///
/// # Verification
///
/// - The verifier sums the present terminals across every AIR in a batch.
/// - A non-zero total signals an unbalanced lookup somewhere in the proof.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LookupTerminal<F>(pub F);

/// Lookup verification error.
#[derive(Debug, Error)]
pub enum LookupError {
    /// Cross-AIR sum of committed lookup terminals is non-zero.
    ///
    /// - Any imbalance makes the sum non-zero with overwhelming probability.
    /// - The collapsed terminal names no bus on its own.
    /// - Debug builds pinpoint the bus, tuple, and row via the prover's replay.
    #[error("cross-AIR lookup terminal sum is non-zero")]
    TerminalSumNonZero,
    /// The auxiliary lookup commitment is present when no AIR declares a lookup, or absent when one does.
    ///
    /// - The commitment must be present exactly when some AIR declares a lookup.
    #[error("lookup commitment presence does not match lookup configuration")]
    CommitmentMismatch,
    /// An AIR's committed terminal presence disagrees with whether it declares a lookup.
    ///
    /// - An AIR with at least one lookup commits exactly one terminal.
    /// - An AIR with no lookup commits none.
    #[error(
        "air {air}: lookup terminal presence mismatch: expected_present={expected_present}, got_present={got_present}"
    )]
    TerminalPresenceMismatch {
        air: usize,
        expected_present: bool,
        got_present: bool,
    },
    /// The auxiliary local and next openings of an AIR have different lengths.
    #[error("air {air}: permutation local/next length mismatch")]
    PermutationLengthMismatch { air: usize },
    /// The auxiliary opening width of an AIR does not match the expected width.
    #[error("air {air}: permutation width mismatch: expected {expected}")]
    PermutationWidthMismatch { air: usize, expected: usize },
    /// The LogUp multiplicity height-bound `sum_i w_i * h_i < p` is violated.
    ///
    /// - A table-entry multiplicity could then wrap modulo the field characteristic.
    /// - A prover could exploit that wrap to forge multiplicities and break soundness.
    #[error(
        "LogUp multiplicity height-bound exceeded: weighted height sum {weighted_height_sum} reaches the ~2^{field_bits} field characteristic"
    )]
    MultiplicityHeightBoundExceeded {
        weighted_height_sum: u128,
        field_bits: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};

    use super::*;
    use crate::logup::LogUpGadget;

    type F = BabyBear;

    /// Build one AIR's lookups from a list of global interaction weights.
    ///
    /// Field contents are irrelevant to the weight accounting, so they stay empty.
    fn global_lookups(weights: &[u32]) -> Lookups<F> {
        let global: Vec<SymbolicInteraction<F>> = weights
            .iter()
            .map(|&count_weight| SymbolicInteraction {
                bus_name: String::from("bus"),
                fields: vec![],
                count: SymbolicExpression::from(F::ONE),
                count_weight,
            })
            .collect();
        Lookups::from_interactions(&global, &[])
    }

    /// One global interaction carrying a single degree-1 main-column payload.
    fn global_payload(bus: &str, weight: u32) -> SymbolicInteraction<F> {
        // A main-column payload has degree 1.
        // So a lone tuple pins at degree 2, and a fold of g tuples pins at degree g + 1.
        let col = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        SymbolicInteraction {
            bus_name: String::from(bus),
            fields: vec![SymbolicExpression::from(col)],
            count: SymbolicExpression::from(F::ONE),
            count_weight: weight,
        }
    }

    /// Unpacked lookups for several payload interactions on one bus.
    fn same_bus(bus: &str, count: usize) -> Lookups<F> {
        // Each interaction is identical; only their count drives the packing tests.
        let global: Vec<SymbolicInteraction<F>> =
            (0..count).map(|_| global_payload(bus, 1)).collect();
        Lookups::from_interactions(&global, &[])
    }

    /// Column indices must stay `0..len` for the fraction-column mapping to hold.
    fn assert_contiguous_columns(lookups: &Lookups<F>) {
        // Slot at position i must own column i, with no gaps.
        for (i, l) in lookups.iter().enumerate() {
            assert_eq!(l.column, i, "columns must be contiguous after packing");
        }
    }

    #[test]
    fn from_interactions_carries_global_weight_and_zeroes_local() {
        // Layout: one local lookup, then a query (weight 1) and a table entry (weight 0).
        // Local interactions are emitted first, matching the LogUp column order.
        let global: Vec<SymbolicInteraction<F>> = vec![
            SymbolicInteraction {
                bus_name: String::from("bus"),
                fields: vec![],
                count: SymbolicExpression::from(F::ONE),
                count_weight: 1,
            },
            SymbolicInteraction {
                bus_name: String::from("bus"),
                fields: vec![],
                count: SymbolicExpression::from(F::ONE),
                count_weight: 0,
            },
        ];
        let local: Vec<SymbolicLocalInteraction<F>> = vec![SymbolicLocalInteraction {
            tuples: vec![(vec![], SymbolicExpression::from(F::ONE))],
        }];

        let lookups = Lookups::from_interactions(&global, &local);

        //     index 0: local        → weight 0 (never crosses AIRs)
        //     index 1: global query → weight 1
        //     index 2: global table → weight 0
        assert_eq!(lookups[0].count_weight, 0);
        assert_eq!(lookups[1].count_weight, 1);
        assert_eq!(lookups[2].count_weight, 0);

        // Only the query contributes to this AIR's weight.
        assert_eq!(lookups.total_count_weight(), 1);
    }

    #[test]
    fn total_count_weight_sums_every_lookup() {
        // Three queries and one table entry → weight 3.
        let lookups = global_lookups(&[1, 0, 1, 1]);
        assert_eq!(lookups.total_count_weight(), 3);
    }

    #[test]
    fn height_bound_accepts_sum_below_characteristic() {
        // Two AIRs, one query each, heights 2^20 → sum 2^21, far below the ~2^31 order.
        let lookups = [global_lookups(&[1]), global_lookups(&[1])];
        assert!(check_multiplicity_height_bound(&lookups, &[1 << 20, 1 << 20]).is_ok());

        // Zero-weight AIRs never trip the bound, whatever their height.
        let tables = [global_lookups(&[0]), global_lookups(&[0])];
        assert!(check_multiplicity_height_bound(&tables, &[1 << 30, 1 << 30]).is_ok());
    }

    #[test]
    fn height_bound_rejects_sum_reaching_characteristic() {
        // BabyBear order ~2.01e9 (~2^30.9).
        // Two AIRs, one query each, heights 2^30 → sum 2^31 > order → wrap risk.
        //
        //     weighted sum: 2^30 + 2^30 = 2^31 = 2147483648
        //     BabyBear p  :              ~2013265921
        //     2^31 >= p → reject
        let lookups = [global_lookups(&[1]), global_lookups(&[1])];
        let err = check_multiplicity_height_bound(&lookups, &[1 << 30, 1 << 30]).unwrap_err();

        match err {
            LookupError::MultiplicityHeightBoundExceeded {
                weighted_height_sum,
                field_bits,
            } => {
                assert_eq!(weighted_height_sum, 1u128 << 31);
                assert_eq!(field_bits, F::bits());
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn height_bound_threshold_is_the_characteristic_exactly() {
        // Pin the `>= p` boundary: rejection starts exactly at the characteristic.
        // Raw row counts (not powers of two) let the sum land on p and p - 1 precisely.
        let p = F::ORDER_U32 as usize;
        let lookups = [global_lookups(&[1]), global_lookups(&[1])];

        //     sum == p     → reject: a multiplicity of p is indistinguishable from 0
        assert!(check_multiplicity_height_bound(&lookups, &[p - 1, 1]).is_err());

        //     sum == p - 1 → accept: still the largest representable honest count
        assert!(check_multiplicity_height_bound(&lookups, &[p - 2, 1]).is_ok());
    }

    #[test]
    #[should_panic = "aligned per AIR"]
    fn height_bound_panics_on_length_mismatch() {
        // One AIR of lookups but two heights is an orchestration bug, not bad input.
        let lookups = [global_lookups(&[1])];
        let _ = check_multiplicity_height_bound(&lookups, &[1, 2]);
    }

    #[test]
    fn pack_merges_same_bus_within_budget() {
        // Four width-1 interactions, budget 3.
        // A fold of g tuples pins at degree g + 1.
        // So g = 2 fits (degree 3) but g = 3 does not (degree 4).
        // Greedy packs two per column: ceil(4 / 2) = 2 columns.
        let packed = same_bus("a", 4).pack_same_bus(&LogUpGadget, 3);

        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].elements.len(), 2);
        assert_eq!(packed[1].elements.len(), 2);
        assert_contiguous_columns(&packed);
    }

    #[test]
    fn pack_minimal_budget_leaves_every_interaction_alone() {
        // Budget 2 admits only a lone tuple (degree 2).
        // So no two interactions ever share a column.
        let packed = same_bus("a", 4).pack_same_bus(&LogUpGadget, 2);

        assert_eq!(packed.len(), 4);
        assert!(packed.iter().all(|l| l.elements.len() == 1));
        assert_contiguous_columns(&packed);
    }

    #[test]
    fn pack_never_merges_across_buses() {
        // Two buses, generous budget.
        // A column may fold one bus, never two.
        let global: Vec<SymbolicInteraction<F>> = vec![
            global_payload("a", 1),
            global_payload("b", 1),
            global_payload("a", 1),
            global_payload("b", 1),
        ];
        let packed = Lookups::from_interactions(&global, &[]).pack_same_bus(&LogUpGadget, 16);

        // Each bus collapses to one column.
        // The two buses never share a column.
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].kind, Kind::Global(String::from("a")));
        assert_eq!(packed[1].kind, Kind::Global(String::from("b")));
        assert_eq!(packed[0].elements.len(), 2);
        assert_eq!(packed[1].elements.len(), 2);
    }

    #[test]
    fn pack_keeps_locals_leading_and_untouched() {
        // One local, then three same-bus globals, budget 3.
        // The local keeps column 0.
        // The three globals fold into two columns: two tuples, then one.
        let local = vec![SymbolicLocalInteraction {
            tuples: vec![(vec![], SymbolicExpression::from(F::ONE))],
        }];
        let global = vec![
            global_payload("a", 1),
            global_payload("a", 1),
            global_payload("a", 1),
        ];
        let packed = Lookups::from_interactions(&global, &local).pack_same_bus(&LogUpGadget, 3);

        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0].kind, Kind::Local);
        assert_eq!(packed[0].elements.len(), 1);
        assert_eq!(packed[1].elements.len(), 2);
        assert_eq!(packed[2].elements.len(), 1);
        assert_contiguous_columns(&packed);
    }

    #[test]
    fn pack_preserves_total_count_weight() {
        // Folding adds the per-interaction weights.
        // So the AIR's height-bound term stays invariant.
        let global: Vec<SymbolicInteraction<F>> = [1, 0, 1, 1]
            .iter()
            .map(|&w| global_payload("a", w))
            .collect();
        let unpacked = Lookups::from_interactions(&global, &[]);
        let before = unpacked.total_count_weight();

        let packed = unpacked.pack_same_bus(&LogUpGadget, 5);
        assert_eq!(packed.total_count_weight(), before);
        assert_eq!(before, 3);
    }

    #[test]
    fn pack_sweep_preserves_invariants() {
        // Sweep interaction count against degree budget.
        // A width-1 column holds at most (budget - 1) tuples.
        // So the packed column count is exactly ceil(n / (budget - 1)).
        for n in 1..=16usize {
            for budget in 2..=8usize {
                let packed = same_bus("a", n).pack_same_bus(&LogUpGadget, budget);

                // Column count matches the closed form.
                let per_column = budget - 1;
                let expected_columns = n.div_ceil(per_column);
                assert_eq!(
                    packed.len(),
                    expected_columns,
                    "n={n} budget={budget}: column count"
                );

                // Folding moves tuples between columns but never drops one.
                let total_tuples: usize = packed.iter().map(|l| l.elements.len()).sum();
                assert_eq!(total_tuples, n, "n={n} budget={budget}: tuples conserved");

                // No packed column exceeds the degree budget.
                for l in packed.iter() {
                    assert!(
                        LogUpGadget.constraint_degree(l) <= budget,
                        "n={n} budget={budget}: degree within budget"
                    );
                }
                assert_contiguous_columns(&packed);
            }
        }
    }

    #[test]
    fn lookups_serde_round_trip() {
        // A representative LogUp set: one local lookup followed by two same-bus globals.
        let local = vec![SymbolicLocalInteraction {
            tuples: vec![(
                vec![SymbolicExpression::from(SymbolicVariable::<F>::new(
                    BaseEntry::Main { offset: 0 },
                    0,
                ))],
                SymbolicExpression::from(F::ONE),
            )],
        }];
        let global = vec![global_payload("bus", 1), global_payload("bus", 0)];
        let lookups = Lookups::from_interactions(&global, &local);

        let json = serde_json::to_string(&lookups).unwrap();
        let decoded: Lookups<F> = serde_json::from_str(&json).unwrap();

        // Structural equality: the decoded set re-serializes identically.
        assert_eq!(serde_json::to_string(&decoded).unwrap(), json);

        // Spot-check the decoded fields against the original.
        assert_eq!(decoded.len(), lookups.len());
        assert_eq!(decoded.total_count_weight(), lookups.total_count_weight());
        for (a, b) in decoded.iter().zip(lookups.iter()) {
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.column, b.column);
            assert_eq!(a.count_weight, b.count_weight);
        }
    }

    #[test]
    fn lookup_terminal_serde_round_trip() {
        let terminal = LookupTerminal(F::new(123));
        let json = serde_json::to_string(&terminal).unwrap();
        let decoded: LookupTerminal<F> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.0, terminal.0);
    }
}
