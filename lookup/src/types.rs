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
use crate::symbolic::InteractionSymbolicBuilder;

/// Whether a lookup is confined to one AIR or shared across AIRs on a named bus.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Default)]
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

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};

    use super::*;

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
}
