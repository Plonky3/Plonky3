//! Word-level relations over XOR, AND, shifts, and integer multiplication.

#![no_std]

extern crate alloc;

mod constraint;
mod index;
mod shift;
mod system;
mod word;

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
