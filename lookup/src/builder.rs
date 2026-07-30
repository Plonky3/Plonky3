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

/// One `(message_fields, count)` pair of a local lookup.
pub type LocalTuple<F> = (Vec<SymbolicExpression<F>>, Count<SymbolicExpression<F>>);

/// One intra-AIR lookup recorded during symbolic evaluation.
///
/// Several `(fields, count)` pairs are bundled into a single running sum
/// that must return to zero over the trace. No cross-AIR communication.
///
/// ```text
///     tuples = [(query_fields, Count::bounded(+1, 1)), (table_fields, Count::provided(-mult))]
/// ```
#[derive(Clone, Debug)]
pub struct SymbolicLocalInteraction<F> {
    /// Pairs of message expressions and their signed, weight-bounded multiplicities.
    pub tuples: Vec<LocalTuple<F>>,
}

/// One mutually-exclusive group of cross-AIR messages on a named bus.
///
/// - Every branch sits on the same bus.
/// - Every branch shares that bus's single challenge pair.
/// - At most one branch is active on any row, selected by its flag.
/// - Only the active branch contributes a fraction.
/// - Every other branch vanishes.
/// - The group occupies a single auxiliary column.
/// - That column's degree is the per-branch maximum, not the sum.
///
/// # Soundness
///
/// - The flags must be boolean.
/// - The flags must sum to at most one on every row.
/// - The AIR must enforce both rules.
/// - This record only carries the flags.
/// - Non-exclusive flags make the fraction match no single message.
#[derive(Clone, Debug)]
pub struct SymbolicExclusiveInteraction<F> {
    /// Name of the bus this group sends on.
    pub bus_name: String,

    /// One branch per mutually-exclusive case.
    pub branches: Vec<SymbolicExclusiveBranch<F>>,
}

/// One branch of a mutually-exclusive group.
#[derive(Clone, Debug)]
pub struct SymbolicExclusiveBranch<F> {
    /// Boolean selector: `1` when this branch is the active one, else `0`.
    pub flag: SymbolicExpression<F>,

    /// Signed multiplicity contributed when this branch is active.
    pub count: SymbolicExpression<F>,

    /// Message payload for this branch.
    pub fields: Vec<SymbolicExpression<F>>,

    /// Per-row upper bound on the magnitude of the count.
    pub count_weight: u32,
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
    /// Each tuple is `(message_fields, count)`. All tuples collapse into a
    /// single running sum that must return to zero over the trace.
    ///
    /// # Soundness
    ///
    /// - A constant count fixes its bound automatically.
    /// - A variable count must declare a bound the AIR constrains it to respect.
    /// - The provided (table) side of a query should use `Count::provided`, keeping
    ///   it out of the multiplicity height-bound check.
    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    );

    /// Record one mutually-exclusive group of messages on a named bus.
    ///
    /// - Each branch is a `(flag, count, fields)` triple.
    /// - The flags select at most one active branch per row.
    /// - The group occupies a single auxiliary column.
    /// - Its constraint degree is the per-branch maximum, not the sum.
    ///
    /// # Arguments
    ///
    /// - `bus_name` — shared identifier linking all AIRs on the same bus.
    /// - `branches` — one `(flag, count, fields)` triple per exclusive case.
    ///
    /// # Soundness
    ///
    /// - The flags must be boolean and mutually exclusive.
    /// - The AIR must enforce that.
    /// - Otherwise the multiplexed fraction matches no single message.
    ///
    /// Recording builders override this.
    /// Other builders inherit a no-op default.
    fn push_exclusive_interaction(
        &mut self,
        _bus_name: &str,
        branches: impl IntoIterator<Item = (Self::Expr, Count<Self::Expr>, Vec<Self::Expr>)>,
    ) {
        // Drain the iterator so side effects in a wrapping adapter still fire.
        branches.into_iter().for_each(drop);
    }

    /// Global interactions pushed so far.
    fn num_global_interactions(&self) -> usize {
        0
    }

    /// Local interactions pushed so far.
    fn num_local_interactions(&self) -> usize {
        0
    }

    /// Exclusive interactions pushed so far.
    fn num_exclusive_interactions(&self) -> usize {
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
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
        // Same rationale as the global path:
        //
        // Swallow the iterator to keep caller-observable behavior consistent across builder implementations.
        tuples.into_iter().for_each(drop);
    }
}
