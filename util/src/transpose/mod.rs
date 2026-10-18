//! Matrix transpose operations.
//!
//! This module provides high-performance transpose implementations

mod portable;
mod rectangular;
mod rows;
mod square;

pub use rectangular::{transpose, transpose_words};
pub use rows::{transpose_rows, transpose_rows_words};
pub(crate) use square::transpose_in_place_square;
