//! Inter-AIR communication via interactions.

use alloc::vec::Vec;
use core::hash::Hash;

use p3_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder, PermutationAirBuilder};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_uni_stark::{Entry, SymbolicExpression};

/// A marker trait for interaction identifiers.
///
/// This trait allows users to define their own taxonomy of interactions.
///
/// # Examples
///
/// ```ignore
/// // Simple: just use integers
/// type MyKind = u32;
///
/// // Complex: custom enum for a VM
/// #[derive(Copy, Clone, Eq, PartialEq, Hash)]
/// enum VmInteractionKind {
///     Memory,
///     Program,
///     RangeCheck,
/// }
/// ```
pub trait InteractionKind: 'static + Copy + Clone + Send + Sync + Eq + Hash {}

impl<T: 'static + Copy + Clone + Send + Sync + Eq + Hash> InteractionKind for T {}

/// A single interaction between AIRs.
#[derive(Debug, Clone)]
pub struct Interaction<F: Field, K: InteractionKind> {
    /// The values in this interaction (e.g., lookup tuple elements).
    pub values: Vec<SymbolicExpression<F>>,

    /// The multiplicity (count) of this interaction.
    ///
    /// - Positive: This AIR **receives** (reads from the table)
    /// - Negative: This AIR **sends** (writes to the table)
    pub multiplicity: SymbolicExpression<F>,

    /// The kind of interaction, used to group interactions together.
    ///
    /// All interactions with the same `kind` will be checked together
    /// in the proving/verification process.
    pub kind: K,
}

impl<F: Field, K: InteractionKind> Interaction<F, K> {
    /// Creates a new interaction.
    pub const fn new(
        values: Vec<SymbolicExpression<F>>,
        multiplicity: SymbolicExpression<F>,
        kind: K,
    ) -> Self {
        Self {
            values,
            multiplicity,
            kind,
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
pub trait AirBuilderWithInteractions<K: InteractionKind>: AirBuilder
where
    Self::F: Field,
{
    /// Sends an interaction (provides data to a table).
    ///
    /// The multiplicity will be negated internally (send = negative contribution).
    fn send(&mut self, interaction: Interaction<Self::F, K>);

    /// Receives an interaction (reads data from a table).
    ///
    /// The multiplicity is used as-is (receive = positive contribution).
    fn receive(&mut self, interaction: Interaction<Self::F, K>);
}

/// Evaluates a symbolic expression in the context of an AIR builder.
///
/// Converts `SymbolicExpression<F>` to the builder's expression type `AB::Expr`.
pub fn eval_symbolic<AB>(builder: &AB, expr: &SymbolicExpression<AB::F>) -> AB::Expr
where
    AB: AirBuilder + PairBuilder + AirBuilderWithPublicValues + PermutationAirBuilder,
{
    match expr {
        SymbolicExpression::Variable(v) => {
            let get_val = |offset: usize, index: usize| match offset {
                0 => builder.main().row_slice(0).unwrap()[index].clone().into(),
                1 => builder.main().row_slice(1).unwrap()[index].clone().into(),
                _ => panic!("Cannot have expressions involving more than two rows."),
            };

            match v.entry {
                Entry::Main { offset } => get_val(offset, v.index),
                Entry::Public => builder.public_values()[v.index].into(),
                _ => unimplemented!("Entry type {:?} not supported in interactions", v.entry),
            }
        }
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
