//! Core lookup types and traits.
//!
//! These types define the data structures for lookup arguments in STARKs.
//! They were previously in `p3-air` but live here to keep the `Air` trait
//! free of lookup concerns.

use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Neg;

use p3_air::{Air, PermutationAirBuilder, SymbolicExpression};
use p3_field::Field;
use serde::{Deserialize, Serialize};

/// Numeric identifier for a global bus.
///
/// All lookups sharing the same `BusIndex` participate in the same
/// permutation argument and share the same (alpha, beta) challenges.
pub type BusIndex = u16;

/// Defines errors that can occur during lookup verification.
#[derive(Debug)]
pub enum LookupError {
    /// Error indicating that the global cumulative sum is incorrect.
    GlobalCumulativeMismatch {
        /// The bus index of the failing global lookup.
        bus_index: BusIndex,
        /// Optional human-readable name for diagnostics.
        name: Option<String>,
    },
}

/// Specifies whether a lookup is local to an AIR or part of a global interaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A lookup where all entries are contained within a single AIR.
    Local,
    /// A lookup that spans multiple AIRs, identified by a numeric bus index.
    ///
    /// The `bus_index` determines challenge sharing and cumulative-sum grouping.
    /// The `name` is retained for diagnostics only.
    Global {
        /// Numeric bus identifier.
        bus_index: BusIndex,
        /// Human-readable name (for error messages / debugging).
        name: String,
    },
}

impl Kind {
    /// Convenience constructor for a global lookup.
    pub fn global(bus_index: BusIndex, name: impl Into<String>) -> Self {
        Self::Global {
            bus_index,
            name: name.into(),
        }
    }
}

/// Indicates the direction of data flow in a lookup interaction.
#[derive(Clone, Copy)]
pub enum Direction {
    /// Indicates that elements are being sent (contributed) to the lookup.
    Send,
    /// Indicates that elements are being received (removed) from the lookup.
    Receive,
}

impl Direction {
    /// Helper method to compute the signed multiplicity based on the direction.
    pub fn multiplicity<T: Neg<Output = T>>(&self, mult: T) -> T {
        match self {
            Self::Send => -mult,
            Self::Receive => mult,
        }
    }
}

/// Data required for global lookup arguments in a multi-STARK proof.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LookupData<F> {
    /// Numeric bus identifier for this global lookup.
    pub bus_index: BusIndex,
    /// Name of the global lookup interaction (for diagnostics).
    pub name: String,
    /// Index of the auxiliary column (if there are multiple auxiliary columns, this is the first one)
    pub aux_idx: usize,
    /// Expected cumulated value for a global lookup argument.
    pub expected_cumulated: F,
}

/// A type alias for a lookup input tuple.
///
/// The tuple contains:
/// - a vector of symbolic expressions representing the elements involved in the lookup,
/// - a symbolic expression representing the multiplicity of the lookup,
/// - a direction indicating whether the elements are being sent or received.
///
/// # Example
/// ```ignored
/// use p3_lookup::{LookupInput, Direction};
/// use p3_air::SymbolicExpression;
///
/// let lookup_input: LookupInput<F> = (
///     vec![SymbolicExpression::Constant(F::ONE)],
///     SymbolicExpression::Constant(F::ONE),
///     Direction::Send
/// );
/// ```
pub type LookupInput<F> = (Vec<SymbolicExpression<F>>, SymbolicExpression<F>, Direction);

/// A structure that holds the lookup data necessary to generate lookup contexts.
/// It is shared between the prover and the verifier.
#[derive(Clone, Debug)]
pub struct Lookup<F: Field> {
    /// Type of lookup: local or global
    pub kind: Kind,
    /// Elements being read (consumed from the table). Each `Vec<SymbolicExpression<F>>`
    /// actually represents a tuple of elements that are bundled together to make one lookup.
    pub element_exprs: Vec<Vec<SymbolicExpression<F>>>,
    /// Multiplicities for the elements.
    /// Note that Lagrange selectors may not be normalized, so they cannot be used as proper
    /// boolean filters in the multiplicities.
    pub multiplicities_exprs: Vec<SymbolicExpression<F>>,
    /// The column index in the permutation trace for this lookup's running sum
    pub columns: Vec<usize>,
}

impl<F: Field> Lookup<F> {
    /// Creates a new lookup with the specified column.
    ///
    /// # Arguments
    /// * `elements` - Elements from either the main execution trace or a lookup table.
    /// * `multiplicities` - How many times each `element` should appear
    /// * `column` - The column index in the permutation trace for this lookup
    pub const fn new(
        kind: Kind,
        element_exprs: Vec<Vec<SymbolicExpression<F>>>,
        multiplicities_exprs: Vec<SymbolicExpression<F>>,
        columns: Vec<usize>,
    ) -> Self {
        Self {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns,
        }
    }

    /// Counts how many global lookup interactions are present in `lookups`.
    pub fn global_count(lookups: &[Self]) -> usize {
        lookups
            .iter()
            .filter(|lookup| matches!(lookup.kind, Kind::Global { .. }))
            .count()
    }
}

/// Trait for evaluating lookup constraints.
pub trait LookupEvaluator {
    /// Returns the number of auxiliary columns needed by this lookup protocol.
    ///
    /// For example:
    /// - LogUp needs 1 column (running sum)
    fn num_aux_cols(&self) -> usize;

    /// Returns the number of challenges for each lookup argument.
    ///
    /// For example, for LogUp, this is 2:
    /// - one challenge for combining the lookup tuples,
    /// - one challenge for the running sum.
    fn num_challenges(&self) -> usize;

    /// Evaluates a local lookup argument based on the provided context.
    ///
    /// For example, in LogUp:
    /// - this checks that the running sum is updated correctly.
    /// - it checks that the final value of the running sum is 0.
    fn eval_local_lookup<AB>(&self, builder: &mut AB, context: &Lookup<AB::F>)
    where
        AB: PermutationAirBuilder;

    /// Evaluates a global lookup update based on the provided context, and the expected cumulated value.
    /// This evaluation is carried out at the AIR level. We still need to check that the permutation argument holds
    /// over all AIRs involved in the interaction.
    ///
    /// For example, in LogUp:
    /// - this checks that the running sum is updated correctly.
    /// - it checks that the local final value of the running sum is equal to the value provided by the prover.
    fn eval_global_update<AB>(
        &self,
        builder: &mut AB,
        context: &Lookup<AB::F>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder;

    /// Evaluates the lookup constraints for all provided contexts.
    ///
    /// For each context:
    /// - if it is a local lookup, evaluates it with `eval_local_lookup`.
    /// - if it is a global lookup, evaluates it with `eval_global_update`, reading the expected
    ///   cumulated value from the builder's `permutation_values()`.
    fn eval_lookups<AB>(&self, builder: &mut AB, contexts: &[Lookup<AB::F>])
    where
        AB: PermutationAirBuilder,
    {
        let mut pv_idx = 0;
        for context in contexts.iter() {
            match &context.kind {
                Kind::Local => {
                    self.eval_local_lookup(builder, context);
                }
                Kind::Global { .. } => {
                    let expected = builder.permutation_values()[pv_idx].clone();
                    pv_idx += 1;
                    self.eval_global_update(builder, context, expected.into());
                }
            }
        }
        assert_eq!(
            pv_idx,
            builder.permutation_values().len(),
            "permutation values count mismatch"
        );
    }
}

/// Extension trait for AIRs that use lookups.
///
/// This decouples lookup definition from the core [`Air`] trait, so AIRs
/// that don't use lookups don't need to know about lookup types at all.
pub trait LookupAir<F: Field> {
    /// Allocate auxiliary columns for a new lookup and return their indices.
    ///
    /// Default implementation returns an empty vector, indicating no lookup columns.
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        Vec::new()
    }

    /// Return all lookups registered by this AIR.
    ///
    /// Default implementation returns an empty vector, indicating no lookups.
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        Vec::new()
    }

    /// Register a lookup to be used in this AIR.
    ///
    /// This is a convenience method that constructs a [`Lookup`] from inputs
    /// and allocates auxiliary columns via [`add_lookup_columns`](Self::add_lookup_columns).
    fn register_lookup(&mut self, kind: Kind, lookup_inputs: &[LookupInput<F>]) -> Lookup<F> {
        let (element_exprs, multiplicities_exprs) = lookup_inputs
            .iter()
            .map(|(elems, mult, dir)| {
                let multiplicity = dir.multiplicity(mult.clone());
                (elems.clone(), multiplicity)
            })
            .unzip();

        Lookup {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns: self.add_lookup_columns(),
        }
    }
}

/// Extension of [`Air`] that adds lookup constraint evaluation.
///
/// This trait is blanket-implemented for every type that implements [`Air`],
/// so any AIR automatically supports `eval_with_lookups`. It lives in
/// `p3-lookup` (rather than `p3-air`) to keep the core `Air` trait free of
/// lookup concerns.
pub trait AirWithLookups<AB: PermutationAirBuilder>: Air<AB> {
    /// Evaluate both AIR constraints and lookup constraints.
    ///
    /// First evaluates the core AIR constraints via [`Air::eval`], then
    /// evaluates any lookup constraints via the provided [`LookupEvaluator`].
    ///
    /// For AIRs without lookups, pass an empty `lookups` slice and the
    /// evaluator will be skipped entirely.
    fn eval_with_lookups(
        &self,
        builder: &mut AB,
        lookups: &[Lookup<AB::F>],
        lookup_evaluator: &impl LookupEvaluator,
    ) {
        self.eval(builder);

        if !lookups.is_empty() {
            lookup_evaluator.eval_lookups(builder, lookups);
        }
    }
}

impl<AB: PermutationAirBuilder, A: Air<AB>> AirWithLookups<AB> for A {}
