//! Bit-sliced `GF(2)`, in three layers:
//!
//! ```text
//!     underlier   a block of bits, 8 to 512 wide, with no field structure
//!     packing     that block read as one GF(2) element per bit
//!     transpose   a square matrix of packings, turned on its diagonal
//! ```
//!
//! An element carries one bit, so holding it in a byte wastes seven eighths of a register.
//! The packing fixes the lane order, and everything else here follows it.

mod packing;
mod transpose;
mod underlier;

pub use packing::{
    PackedGf2, PackedGf2x8, PackedGf2x16, PackedGf2x32, PackedGf2x64, PackedGf2x128, PackedGf2x256,
    PackedGf2x512,
};
pub use underlier::{Divisible, M128, M256, M512, Underlier, Word};
