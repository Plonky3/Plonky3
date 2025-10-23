use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Neg;

use p3_air::{Air, AirBuilderWithPublicValues, PairBuilder, PermutationAirBuilder};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_uni_stark::{Entry, SymbolicExpression};

/// Defines errors that can occur during lookup verification.
#[derive(Debug)]
pub enum LookupError {
    /// Error indicating that the global cumulative sum is incorrect.
    GlobalCumulativeMismatch,
}

/// A trait for lookup argument.
pub trait LookupGadget {
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
    fn verify_global_final_value<EF: Field>(
        &self,
        all_expected_cumulated: &[EF],
    ) -> Result<(), LookupError>;

    /// Computes the polynomial degree of a lookup transition constraint.
    fn constraint_degree<F: Field>(&self, context: Lookup<F>) -> usize;
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

/// A type alias for a lookup input tuple. It contains:
/// - a vector of symbolic expressions representing the elements involved in the lookup,
/// - a symbolic expression representing the multiplicity of the lookup,
/// - a direction indicating whether the elements are being sent or received.
pub type LookupInput<F> = (Vec<SymbolicExpression<F>>, SymbolicExpression<F>, Direction);

/// A structure that holds the lookup data necessary to generate a `LookupContext`. It is shared between the prover and the verifier.
#[derive(Clone, Debug)]
pub struct Lookup<F: Field> {
    /// Type of lookup: local or global
    pub kind: Kind,
    /// Elements being read (consumed from the table). Each `Vec<SymbolicExpression<F>>` actually represents a tuple of elements that are bundled together to make one lookup.
    pub element_exprs: Vec<Vec<SymbolicExpression<F>>>,
    /// Multiplicities for the elements.
    pub multiplicities_exprs: Vec<SymbolicExpression<F>>,
    /// The column index in the permutation trace for this lookup's running sum
    pub columns: Vec<usize>,
}

impl<F: Field> Lookup<F> {
    /// Creates a new lookup with the specified column.
    ///
    /// # Arguments
    /// * `elements` - Elements from the either the main execution trace or a lookup table.
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

/// A trait for an AIR that handles lookup arguments.
pub trait AirLookupHandler<AB>: Air<AB>
where
    AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    AB::Var: Copy + Into<AB::ExprEF>,
    AB::ExprEF: From<AB::Var> + From<AB::F>,
{
    /// Register a lookup to be used in this AIR.
    /// This method can be used before proving or verifying, as the resulting data is shared between the prover and the verifier.
    fn register_lookup(
        &mut self,
        kind: Kind,
        lookup_inputs: &[LookupInput<AB::F>],
    ) -> Lookup<AB::F> {
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

    /// Updates the number of auxiliary columns to account for a new lookup column, and returns its index (or indices).
    fn add_lookup_columns(&mut self) -> Vec<usize>;

    /// Register all lookups for the current AIR and return them.
    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>>;
}

/// Takes a symbolic expression and converts it into an expression in the context of the provided AirBuilder.
pub fn symbolic_to_expr<AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues>(
    builder: &mut AB,
    symbolic: &SymbolicExpression<AB::F>,
) -> AB::ExprEF {
    let turn_into_expr = |values: &[AB::Var]| {
        values
            .iter()
            .map(|v| AB::Expr::from(v.clone()))
            .collect::<Vec<_>>()
    };
    let main = builder.main();
    let local_values = &turn_into_expr(&main.row_slice(0).unwrap());
    let next_values = &turn_into_expr(&main.row_slice(1).unwrap());

    let public_values = builder
        .public_values()
        .iter()
        .map(|v| AB::ExprEF::from((*v).into()))
        .collect::<Vec<_>>();

    match symbolic {
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
                Entry::Main { offset } => get_val(offset, v.index, local_values, next_values),
                Entry::Public => public_values[v.index].clone(),
                _ => unimplemented!(),
            }
        }
        SymbolicExpression::Add { x, y, .. } => {
            let x_expr = symbolic_to_expr(builder, x);
            let y_expr = symbolic_to_expr(builder, y);
            x_expr + y_expr
        }
        SymbolicExpression::Mul { x, y, .. } => {
            let x_expr = symbolic_to_expr(builder, x);
            let y_expr = symbolic_to_expr(builder, y);
            x_expr * y_expr
        }
        SymbolicExpression::Sub { x, y, .. } => {
            let x_expr = symbolic_to_expr(builder, x);
            let y_expr = symbolic_to_expr(builder, y);
            x_expr - y_expr
        }
        SymbolicExpression::Neg { x, .. } => {
            let x_expr = symbolic_to_expr(builder, x);
            -x_expr
        }
        SymbolicExpression::IsFirstRow => builder.is_first_row().into(),
        SymbolicExpression::IsLastRow => builder.is_last_row().into(),
        SymbolicExpression::IsTransition => builder.is_transition().into(),
    }
}
