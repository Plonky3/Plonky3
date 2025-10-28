//! Inter-AIR communication via interactions.

use alloc::vec::Vec;

use p3_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder, PermutationAirBuilder};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_uni_stark::{Entry, SymbolicExpression};

use crate::InteractionCollector;

/// A single interaction between AIRs.
#[derive(Debug, Clone)]
pub struct Interaction<F: Field> {
    /// The values in this interaction (e.g., lookup tuple elements).
    pub values: Vec<SymbolicExpression<F>>,

    /// The multiplicity (count) of this interaction.
    ///
    /// - Positive: This AIR **receives** (reads from the table)
    /// - Negative: This AIR **sends** (writes to the table)
    pub multiplicity: SymbolicExpression<F>,
}

impl<F: Field> Interaction<F> {
    /// Creates a new interaction.
    pub const fn new(
        values: Vec<SymbolicExpression<F>>,
        multiplicity: SymbolicExpression<F>,
    ) -> Self {
        Self {
            values,
            multiplicity,
        }
    }

    /// Returns the arity (number of values) in this interaction.
    pub const fn arity(&self) -> usize {
        self.values.len()
    }
}

/// Extension trait for AIR builders that support interactions.
///
/// Provides `send` and `receive` methods for defining inter-AIR communication.
pub trait MessageBuilder<F: Field> {
    /// Sends an interaction (provides data to a table).
    ///
    /// The multiplicity will be negated internally (send = negative contribution).
    fn send(&mut self, interaction: Interaction<F>);

    /// Receives an interaction (reads data from a table).
    ///
    /// The multiplicity is used as-is (receive = positive contribution).
    fn receive(&mut self, interaction: Interaction<F>);
}

/// Trait for AIRs that define interactions with other AIRs.
///
/// This trait separates interaction discovery from constraint evaluation,
/// keeping `Air::eval` focused purely on constraints.
///
/// # Example
/// ```ignore
/// use crate::builder::InteractionCollector;
///
/// impl<F: Field> InteractionDiscovery<F> for MyAir {
///     fn discover_interactions(&self, collector: &mut InteractionCollector<F>) {
///         let main = collector.main();
///         let local = main.row_slice(0).unwrap();
///
///         // Define sends and receives
///         collector.send(Interaction::new(...));
///         collector.receive(Interaction::new(...));
///     }
/// }
/// ```
pub trait InteractionDiscovery<F: Field> {
    /// Discovers and registers all interactions for this AIR.
    ///
    /// This method should use the collector's `send` and `receive` methods
    /// to define all interactions that this AIR participates in.
    fn discover_interactions(&self, collector: &mut InteractionCollector<F>);
}

/// Evaluates a symbolic expression in the context of an AIR builder.
///
/// Converts `SymbolicExpression<F>` to the builder's expression type `AB::Expr`.
pub fn eval_symbolic<AB>(builder: &AB, expr: &SymbolicExpression<AB::F>) -> AB::Expr
where
    AB: AirBuilder + PairBuilder + AirBuilderWithPublicValues + PermutationAirBuilder,
{
    match expr {
        SymbolicExpression::Variable(v) => match v.entry {
            Entry::Main { offset } => match offset {
                0 => builder.main().row_slice(0).unwrap()[v.index].clone().into(),
                1 => builder.main().row_slice(1).unwrap()[v.index].clone().into(),
                _ => panic!("Cannot have expressions involving more than two rows."),
            },
            Entry::Public => builder.public_values()[v.index].into(),
            _ => unimplemented!("Entry type {:?} not supported in interactions", v.entry),
        },
        SymbolicExpression::IsFirstRow => builder.is_first_row(),
        SymbolicExpression::IsLastRow => builder.is_last_row(),
        SymbolicExpression::IsTransition => builder.is_transition_window(2),
        SymbolicExpression::Constant(c) => AB::Expr::from(*c),
        SymbolicExpression::Add { x, y, .. } => {
            eval_symbolic(builder, x) + eval_symbolic(builder, y)
        }
        SymbolicExpression::Sub { x, y, .. } => {
            eval_symbolic(builder, x) - eval_symbolic(builder, y)
        }
        SymbolicExpression::Neg { x, .. } => -eval_symbolic(builder, x),
        SymbolicExpression::Mul { x, y, .. } => {
            eval_symbolic(builder, x) * eval_symbolic(builder, y)
        }
    }
}
