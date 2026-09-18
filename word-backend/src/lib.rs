//! Packed witness data and shift metadata for word-level proving backends.

#![no_std]

extern crate alloc;

mod columns;
mod keys;
mod witness;

pub use columns::OperationColumns;
pub use keys::{
    CompiledKey, CompiledKeyLayout, CompiledSegment, ConstraintReference, KeyCompileError,
    LayoutComponent,
};
pub use witness::{PackedWitness, PackedWord, WitnessError};
