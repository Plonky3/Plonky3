//! The NEON packings, on the `PMULL` carryless multiply of 128-bit registers.

pub(crate) mod gf64;
mod ghash128;

pub use ghash128::PackedGhash128;
