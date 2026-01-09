//! Symbolic expression types for AIR constraint representation.

mod expression;
mod variable;

pub use expression::SymbolicExpression;
pub use variable::{Entry, SymbolicVariable};
