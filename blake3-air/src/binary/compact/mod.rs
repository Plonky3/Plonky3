//! A round-per-row alternative to the binary compression-per-row AIR.

mod air;
mod columns;
mod generation;

pub use air::*;
pub use columns::*;
pub use generation::*;

/// Logical rows per compression: seven rounds followed by one independent padding round.
pub const BLAKE3_BINARY_COMPACT_ROWS_PER_COMPRESSION: usize = 8;

/// Row within each eight-row compression group where its output is available.
pub const BLAKE3_BINARY_COMPACT_OUTPUT_ROW: usize = 6;
