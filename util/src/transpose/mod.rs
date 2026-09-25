//! Matrix transpose operations.
//!
//! This module provides high-performance transpose implementations

mod portable;
mod rectangular;
mod rows;
mod square;

pub use rectangular::transpose;
pub use rows::transpose_rows;
pub(crate) use square::transpose_in_place_square;
