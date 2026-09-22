//! Word-level relations over XOR, AND, shifts, and integer multiplication.
//!
//! A [`ConstraintSystem`] writes every relation out once. A [`Component`] instead
//! declares one word gadget against component-local slots, and a [`Composition`]
//! instantiates several such gadgets a checked number of times each, laying their
//! words out packed by instance.

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
