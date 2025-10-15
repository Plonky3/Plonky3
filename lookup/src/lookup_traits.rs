use alloc::{rc::Rc, string::String, vec::Vec};
use itertools::Itertools;
use p3_air::{Air, AirBuilderWithPublicValues, PairBuilder, PermutationAirBuilder};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_uni_stark::{Entry, SymbolicAirBuilder, SymbolicExpression};

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
    fn eval_local_lookup<AB>(&self, builder: &mut AB, context: Lookup<AB::F>)
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
        context: Lookup<AB::F>,
        expected_cumulated: AB::ExprEF,
    ) where
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues;

    /// Evaluates the final cumulated value over all AIRs involved in the interaction,
    /// and checks that it is equal to the expected final value.
    ///
    /// For example, in LogUp:
    /// - it sums all expected cumulated values provided by each AIR within one interaction,
    /// - checks that the sum is equal to 0.
    fn eval_global_final_value<
        AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    >(
        &self,
        builder: &mut AB,
        all_expected_cumulated: &[AB::ExprEF],
    );

    /// Computes the polynomial degree of a lookup transition constraint.
    fn constraint_degree<AB: PermutationAirBuilder>(&self, context: Lookup<AB::F>) -> usize;
}

// TODO: Add unit test.
// TODO: Can we reuse this function from recursion?
pub fn symbolic_to_expr<AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues>(
    builder: &mut AB,
    symbolic: Rc<SymbolicExpression<AB::F>>,
) -> AB::ExprEF {
    let turn_into_expr = |values: &Vec<AB::Var>| {
        values
            .iter()
            .map(|v| AB::Expr::from(v.clone()))
            .collect::<Vec<_>>()
    };
    let main = builder.main();
    let local_values = &turn_into_expr(&main.row_slice(0).unwrap().to_vec());
    let next_values = &turn_into_expr(&main.row_slice(1).unwrap().to_vec());

    let public_values = builder
        .public_values()
        .iter()
        .map(|v| AB::ExprEF::from((*v).into()))
        .collect::<Vec<_>>();

    match symbolic.as_ref() {
        SymbolicExpression::Constant(c) => AB::ExprEF::from(AB::EF::from(*c)),
        SymbolicExpression::Variable(v) => {
            let get_val = |offset: usize,
                           index: usize,
                           local_vals: &[AB::Expr],
                           next_vals: &[AB::Expr]| match offset {
                0 => AB::ExprEF::from(local_vals[index].clone()),
                1 => AB::ExprEF::from(next_vals[index].clone()),
                _ => panic!("Cannot have expressions involving more than two rows."),
            };

            match v.entry {
                Entry::Preprocessed { .. } => {
                    unimplemented!()
                }
                Entry::Main { offset } => get_val(offset, v.index, local_values, next_values),
                Entry::Public => public_values[v.index].clone(),
                _ => unimplemented!(),
            }
        }
        SymbolicExpression::Add {
            x,
            y,
            degree_multiple: _,
        } => {
            let x_expr = symbolic_to_expr(builder, x.clone());
            let y_expr = symbolic_to_expr(builder, y.clone());
            x_expr + y_expr
        }
        SymbolicExpression::Mul {
            x,
            y,
            degree_multiple: _,
        } => {
            let x_expr = symbolic_to_expr(builder, x.clone());
            let y_expr = symbolic_to_expr(builder, y.clone());
            x_expr * y_expr
        }
        SymbolicExpression::Sub {
            x,
            y,
            degree_multiple: _,
        } => {
            let x_expr = symbolic_to_expr(builder, x.clone());
            let y_expr = symbolic_to_expr(builder, y.clone());
            x_expr - y_expr
        }
        SymbolicExpression::Neg {
            x,
            degree_multiple: _,
        } => {
            let x_expr = symbolic_to_expr(builder, x.clone());
            -x_expr
        }
        SymbolicExpression::IsFirstRow => builder.is_first_row().into(),
        SymbolicExpression::IsLastRow => builder.is_last_row().into(),
        SymbolicExpression::IsTransition => builder.is_transition().into(),
    }
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
    pub multiplicities_exprs: Vec<SymbolicExpression<F>>,
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
            multiplicities_exprs: muls,
            columns: self.add_lookup_columns(),
        }
    }

    /// Updates the number of auxiliary columns to account for a new lookup column, and returns its index.
    fn add_lookup_columns(&mut self) -> Vec<usize>;

    /// Register all lookups for the current AIR and return them.
    fn get_lookups(&mut self) -> Vec<Lookup<'_, F>>;
}

// TODO: Create ProverLookup and VerifierLookup structures that contain the permutation columns, and implement `PermutationAirBuilder` for them.
