//! Bit-sliced `GF(2)`, built on a layer of fixed-width blocks of bits.
//!
//! The block is the join point: everything above it is generic over the width, so a new
//! width is added by naming a block rather than by writing another kernel.

mod underlier;

pub use underlier::{Divisible, M128, M256, M512, Underlier, Word};
