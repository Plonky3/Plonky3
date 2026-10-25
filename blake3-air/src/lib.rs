//! AIRs for the Blake-3 permutation.
//!
//! [`Blake3Air`] assumes the field characteristic is greater than `3 * 2^16`.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod constants;
mod generation;

pub use air::*;
pub use columns::*;
pub use constants::{BITS_PER_LIMB, U32_LIMBS};
pub use generation::*;
