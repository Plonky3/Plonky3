//! Word-level relation proving, from a packed witness to one authenticated trace opening.
//!
//! A key proves a [`Statement`], which is either a flat constraint system or a
//! composition of components instantiated many times. The composed form compiles
//! each component once, so its stored wiring is the size of the components rather
//! than the size of the statement, and it reaches the same proof either way.

#![no_std]

extern crate alloc;

mod columns;
mod keys;
mod proof;
mod shift;
mod statement;
mod witness;

pub use columns::OperationColumns;
pub use keys::{
    CompiledKey, CompiledKeyLayout, CompiledSegment, ConstraintReference, KeyCompileError,
    LayoutComponent, LayoutFootprint,
};
pub use proof::{WordProof, WordProofError, WordProofKey};
pub use shift::{
    ShiftClaim, ShiftOpeningClaim, ShiftReductionError, ShiftReductionKey, ShiftReductionProof,
};
pub use statement::{Statement, StatementShape};
pub use witness::{Packed, PackedWitness, PackedWord, WitnessError};
