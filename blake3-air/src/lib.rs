//! AIRs for the Blake-3 permutation.
//!
//! [`Blake3Air`] assumes the field size is between 2^20 and 2^32.
//! [`Blake3BinaryAir`] works over fields of characteristic 2.

#![no_std]

extern crate alloc;

mod air;
mod binary;
mod columns;
mod constants;
mod generation;

pub use air::*;
pub use binary::*;
pub use columns::*;
pub use constants::{BITS_PER_LIMB, U32_LIMBS};
pub use generation::*;
