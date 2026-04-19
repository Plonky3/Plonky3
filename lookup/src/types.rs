//! Core data types for lookup arguments.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, SymbolicExpression};
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};

use crate::builder::{SymbolicInteraction, SymbolicLocalInteraction};
use crate::symbolic::InteractionSymbolicBuilder;

/// Whether a lookup is local to one AIR or shared across AIRs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// Intra-AIR lookup. Running sum must return to zero.
    Local,
    /// Cross-AIR lookup on a named bus. Running sums are verified globally.
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
    /// Auxiliary column index in the permutation trace.
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

/// Prover-provided data for one global lookup interaction.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LookupData<F> {
    /// Bus name.
    pub name: String,
    /// Auxiliary column index.
    pub aux_column: usize,
    /// Cumulative sum computed by the prover.
    pub cumulative_sum: F,
}

/// Lookup verification error.
#[derive(Debug)]
pub enum LookupError {
    /// Global cumulative sums do not balance to zero.
    GlobalCumulativeMismatch(Option<String>),
}
