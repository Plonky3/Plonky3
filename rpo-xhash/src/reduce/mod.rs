//! Field-specific modular reduction.
//!
//! Each sub-module provides `reduce(x: u64) -> u32` (or u64 for Goldilocks)
//! that maps the output of the FFT/MDS integer pipeline back to the field.

pub mod goldilocks;
pub mod m31;
pub mod monty31;
