//! Lookup Arguments for STARKs

use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Neg;

use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::{AirBuilderWithPublicValues, PairBuilder, PermutationAirBuilder, SymbolicExpression};

/// Defines errors that can occur during lookup verification.
#[derive(Debug)]
pub enum LookupError {
    /// Error indicating that the global cumulative sum is incorrect.
    GlobalCumulativeMismatch(Option<String>),
}

/// Specifies whether a lookup is local to an AIR or part of a global interaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A lookup where all entries are contained within a single AIR.
    Local,
    /// A lookup that spans multiple AIRs, identified by a unique interaction name.
    ///
    /// The interaction name is used to identify all elements that are part of the same interaction.
    Global(String),
}

/// Indicates the direction of data flow in a global lookup.
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
pub struct LookupData<F: Clone> {
    /// Name of the global lookup interaction.
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
/// use p3_air::lookup::{LookupInput, Direction};
/// use p3_air::SymbolicExpression;
///
/// let lookup_input: LookupInput<SymbolicExpression<F>> = (
///     vec![SymbolicExpression::Constant(F::ONE)],
///     SymbolicExpression::Constant(F::ONE),
///     Direction::Send
/// );
/// ```
pub type LookupInput<F> = (Vec<SymbolicExpression<F>>, SymbolicExpression<F>, Direction);

/// A structure that holds the lookup data necessary to generate lookup contexts
/// via [`LookupTraceBuilder`]. It is shared between the prover and the verifier.
#[derive(Clone, Debug)]
pub struct Lookup<F: Field> {
    /// Type of lookup: local or global
    pub kind: Kind,
    /// Elements being read (consumed from the table). Each `Vec<SymbolicExpression<F>>`
    /// actually represents a tuple of elements that are bundled together to make one lookup.
    pub element_exprs: Vec<Vec<SymbolicExpression<F>>>,
    /// Multiplicities for the elements.
    /// Note that Lagrange selectors may not be normalized, and so cannot be used as proper
    /// filters in the multiplicities.
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
}

/// Trait for evaluating lookup constraints.
/// This is the core interface needed by [`Air::eval_with_lookups`](crate::Air::eval_with_lookups).
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
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues;

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
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues;

    /// Evalutes the lookup constraints for all provided contexts.
    ///
    /// For each context:
    /// - if it is a local lookup, evaluates it with `eval_local_lookup`.
    /// - if it is a global lookup, evaluates it with `eval_global_update`, using the expected cumulated value from `lookup_data`.
    fn eval_lookups<AB>(
        &self,
        builder: &mut AB,
        contexts: &[Lookup<AB::F>],
        // Assumed to be sorted by auxiliary_index.
        lookup_data: &[LookupData<AB::ExprEF>],
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    {
        let mut lookup_data_iter = lookup_data.iter();
        for context in contexts.iter() {
            match &context.kind {
                Kind::Local => {
                    self.eval_local_lookup(builder, context);
                }
                Kind::Global(_) => {
                    // Find the expected cumulated value for this context.
                    let LookupData {
                        name: _,
                        aux_idx,
                        expected_cumulated,
                    } = lookup_data_iter
                        .next()
                        .expect("Expected cumulated value missing");

                    if *aux_idx != context.columns[0] {
                        panic!("Expected cumulated values not sorted by auxiliary index");
                    }
                    self.eval_global_update(builder, context, expected_cumulated.clone());
                }
            }
        }
        assert!(
            lookup_data_iter.next().is_none(),
            "Too many expected cumulated values provided"
        );
    }
}
