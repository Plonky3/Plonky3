//! Packed witness data and shift metadata for word-level proving backends.

#![no_std]

extern crate alloc;

mod columns;
mod keys;
mod proof;
mod shift;
mod witness;

pub use columns::OperationColumns;
pub use keys::{
    CompiledKey, CompiledKeyLayout, CompiledSegment, ConstraintReference, KeyCompileError,
    LayoutComponent,
};
pub use proof::{ProvedStatement, WordProof, WordProofError, WordProofKey};
pub use shift::{
    ShiftClaim, ShiftOpeningClaim, ShiftReductionError, ShiftReductionKey, ShiftReductionProof,
};
pub use witness::{Packed, PackedWitness, PackedWord, WitnessError};
