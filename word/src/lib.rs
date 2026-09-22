//! Word-level relations over XOR, AND, shifts, and integer multiplication.
//!
//! A constraint system writes every relation out once.
//!
//! A component instead declares one word gadget against slots of its own.
//!
//! A composition instantiates several gadgets a checked number of times each.
//!
//! It lays their words out packed by instance.

#![no_std]

extern crate alloc;

mod component;
mod constraint;
mod index;
mod shift;
mod system;
mod word;

pub use component::{Component, ComponentCall, ComponentError, Composition, CompositionError};
pub use constraint::{
    AndConstraint, EvaluationError, IntegerMulConstraint, Operand, ZeroConstraint,
};
pub use index::{IndexError, Segment, ValueIndex};
pub use shift::{Shift, ShiftError, ShiftKind, ShiftSequenceError, ShiftedValue};
pub use system::{
    ConstraintKind, ConstraintSystem, ConstraintTerm, OperandRole, ShapeError, SystemError,
    VerificationError,
};
pub use word::{Word, Word32, Word64};
