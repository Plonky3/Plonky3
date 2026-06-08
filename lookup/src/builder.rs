//! Trait for builders that record bus interactions.

use alloc::string::String;
use alloc::vec::Vec;

use p3_air::{AirBuilder, DebugConstraintBuilder, SymbolicExpression};
use p3_field::{ExtensionField, Field};

use crate::count::Count;

/// One message sent on a named bus during symbolic evaluation.
///
/// Rendered per trace row as a tuple of field expressions plus a signed
/// multiplicity telling how many times the message is emitted.
///
/// ```text
///     bus_name = "memory"
///     fields   = [addr, value]
///     count    = is_active        (evaluates to 0 or 1 per row)
///     weight   = 1
/// ```
#[derive(Clone, Debug)]
pub struct SymbolicInteraction<F> {
    /// Name of the bus this message belongs to.
    ///
    /// All AIRs sharing this string form one logical channel.
    pub bus_name: String,

    /// Message payload as symbolic expressions over the trace.
    pub fields: Vec<SymbolicExpression<F>>,

    /// Signed multiplicity: positive for a send, negative for a receive.
    pub count: SymbolicExpression<F>,

    /// Per-row upper bound on the count's magnitude.
    ///
    /// - Feeds the height check `sum_i weight_i * height_i < p`.
    /// - Sound only if the AIR constrains the count to respect it on every row.
    /// - `1` for a unit query, `0` for a provided table entry.
    pub count_weight: u32,
}

/// One intra-AIR lookup recorded during symbolic evaluation.
///
/// Several `(fields, count)` pairs are bundled into a single running sum
/// that must return to zero over the trace. No cross-AIR communication.
///
/// ```text
///     tuples = [(query_fields, +1), (table_fields, -mult)]
/// ```
#[derive(Clone, Debug)]
pub struct SymbolicLocalInteraction<F> {
    /// Pairs of message expressions and their signed multiplicities.
    pub tuples: Vec<(Vec<SymbolicExpression<F>>, SymbolicExpression<F>)>,
}

/// Opt-in extension to the AIR builder for AIRs that speak on buses.
///
/// - AIRs that emit messages bound their builder on this trait.
/// - Builders that do not care about interactions never implement it.
pub trait InteractionBuilder: AirBuilder {
    /// Record one global (cross-AIR) message on a named bus.
    ///
    /// # Arguments
    ///
    /// - `bus_name` — shared identifier linking all AIRs on the same bus.
    /// - `fields` — message payload.
    /// - `count` — signed multiplicity carrying its own per-row magnitude bound.
    ///
    /// # Soundness
    ///
    /// - A constant count fixes its bound automatically.
    /// - A variable count must declare a bound the AIR constrains it to respect.
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Count<Self::Expr>>,
    );

    /// Record one intra-AIR lookup, both sides in one call.
    ///
    /// Each tuple is `(message_fields, signed_count)`. All tuples collapse
    /// into a single running sum that must return to zero over the trace.
    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Self::Expr)>,
    );

    /// Global interactions pushed so far.
    fn num_global_interactions(&self) -> usize {
        0
    }

    /// Local interactions pushed so far.
    fn num_local_interactions(&self) -> usize {
        0
    }
}

impl<F: Field, EF: ExtensionField<F>> InteractionBuilder for DebugConstraintBuilder<'_, F, EF> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        _count: impl Into<Count<Self::Expr>>,
    ) {
        // Drain the iterator so any side effects inside a wrapping adapter still fire.
        //
        // Matching the semantics a recording builder would give.
        fields.into_iter().for_each(drop);
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Self::Expr)>,
    ) {
        // Same rationale as the global path:
        //
        // Swallow the iterator to keep caller-observable behavior consistent across builder implementations.
        tuples.into_iter().for_each(drop);
    }
}
