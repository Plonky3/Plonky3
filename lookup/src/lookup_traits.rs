use alloc::{string::String, vec::Vec};
use itertools::Itertools;
use p3_air::{Air, PermutationAirBuilder};
use p3_field::Field;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};

/// A trait for lookup argument.
pub trait LookupGadget<F: Field> {
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
    fn eval_local_lookup<AB, AE, AM>(&self, builder: &mut AB, context: LookupContext<AE, AM>)
    where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Clone,
        AM: Into<AB::ExprEF> + Clone;

    /// Evaluates a global lookup update based on the provided context, and the expected cumulated value.
    /// This evaluation is carried out at the AIR level. We still need to check that the permutation argument holds
    /// over all AIRs involved in the interaction.
    ///
    /// For example, in LogUp:
    /// - this checks that the running sum is updated correctly.
    /// - it checks that the local final value of the running sum is equal to the value provided by the prover.
    fn eval_global_update<AB, AE, AM>(
        &self,
        builder: &mut AB,
        context: LookupContext<AE, AM>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder,
        AE: Into<AB::ExprEF> + Clone,
        AM: Into<AB::ExprEF> + Clone;

    /// Evaluates the final cumulated value over all AIRs involved in the interaction,
    /// and checks that it is equal to the expected final value.
    ///
    /// For example, in LogUp:
    /// - it sums all expected cumulated values provided by each AIR within one interaction,
    /// - checks that the sum is equal to 0.
    fn eval_global_final_value<AB: PermutationAirBuilder>(
        &self,
        builder: &mut AB,
        all_expected_cumulated: &[AB::ExprEF],
    );

    /// Computes the polynomial degree of a lookup transition constraint.
    fn constraint_degree<AE, AM>(&self, context: LookupContext<AE, AM>) -> usize;
}

/// A context structure that encapsulates a single lookup relationship and its associated
/// permutation column.
#[derive(Debug, Clone)]
pub struct LookupContext<'a, AE, AM> {
    /// Whether the lookup is local or global.
    pub kind: Kind,
    /// Elements being read (consumed from the table). Each `Vec<AE>` actually represents a tuple of elements that are bundled together to make one lookup.
    pub elements: &'a [Vec<AE>],
    /// Multiplicities for the elements.
    pub multiplicities: &'a [AM],
    /// The column index in the permutation trace for this lookup's running sum
    pub columns: &'a [usize],
}

impl<'a, AE, AM> LookupContext<'a, AE, AM> {
    /// Creates a new lookup context with the specified column.
    ///
    /// # Arguments
    /// * `elements` - Elements from the either the main execution trace or a lookup table.
    /// * `multiplicities` - How many times each `element` should appear
    /// * `column` - The column index in the permutation trace for this lookup
    pub const fn new(
        kind: Kind,
        elements: &'a [Vec<AE>],
        multiplicities: &'a [AM],
        columns: &'a [usize],
    ) -> Self {
        Self {
            kind,
            elements,
            multiplicities,
            columns,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Kind {
    Local,
    // The string corresponds to the interactions's name. It is used to identify all elements that are part of the same interaction.
    Global(String),
}

/// A structure that holds the lookup data necessary to generate a `LookupContext`. It is shared between the prover and the verifier.
pub struct Lookup<'a, F: Field> {
    /// Type of lookup: local or global
    pub kind: Kind,
    /// Elements being read (consumed from the table). Each `Vec<AE>` actually represents a tuple of elements that are bundled together to make one lookup.
    pub element_exprs: &'a [Vec<SymbolicExpression<F>>],
    /// Multiplicities for the elements.
    pub multiplicities: Vec<SymbolicExpression<F>>,
    /// The column index in the permutation trace for this lookup's running sum
    pub columns: Vec<usize>,
}

// TODO: Actually no, we need a structure which contains the lookup, but also the actual auxiliary permutation column and challenges. Maybe we don't even need the actual lookup structure here?

pub enum Direction {
    Send,
    Receive,
}

pub trait AirLookupHandler<F: Field>: Air<SymbolicAirBuilder<F>> {
    /// Register a lookup to be used in this AIR.
    fn register_lookup<'a>(
        &mut self,
        direction: &[Direction],
        kind: Kind,
        elements: &'a [Vec<SymbolicExpression<F>>],
        multiplicities: &[SymbolicExpression<F>],
    ) -> Lookup<'a, F> {
        let muls = elements
            .iter()
            .zip_eq(multiplicities)
            .zip_eq(direction)
            .map(|((_, mult), dir)| match dir {
                Direction::Send => mult.clone(),
                Direction::Receive => -mult.clone(),
            })
            .collect::<Vec<_>>();

        Lookup {
            kind,
            element_exprs: elements,
            multiplicities: muls,
            columns: self.add_lookup_columns(),
        }
    }

    /// Updates the number of auxiliary columns to account for a new lookup column, and returns its index.
    fn add_lookup_columns(&mut self) -> Vec<usize>;

    /// Register all lookups for the current AIR and return them.
    fn get_lookups(&mut self) -> Vec<Lookup<'_, F>>;
}

// TODO: Create ProverLookup and VerifierLookup structures that contain the permutation columns, and implement `PermutationAirBuilder` for them.
