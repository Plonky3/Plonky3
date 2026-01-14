//! Matrix transpose operations.
//!
//! This module provides high-performance transpose implementations

mod rectangular;
mod square;

pub use rectangular::transpose;
pub(crate) use square::transpose_in_place_square;
