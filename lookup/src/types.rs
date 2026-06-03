//! Core data types for lookup arguments.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, SymbolicExpression};
use p3_field::{ExtensionField, Field};
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
                column: col,
            });
            col += 1;
        }

        for i in global {
            lookups.push(Lookup {
                kind: Kind::Global(i.bus_name.clone()),
                elements: vec![i.fields.clone()],
                multiplicities: vec![i.count.clone()],
                column: col,
            });
            col += 1;
        }

        Self(lookups)
    }
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
}
