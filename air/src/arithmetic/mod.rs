//! Opt-in arithmetic-preserving AIR capture.
//!
//! Unlike primitive symbolic capture, these types retain square, sum, dot and
//! mixed-coefficient operations. Lowering preserves arithmetic values, but makes
//! no guarantee about serialized structure or protocol commitments.
mod builder;
mod evaluation;
mod expression;
mod operands;
pub use builder::*;
pub use expression::*;
pub use operands::*;
