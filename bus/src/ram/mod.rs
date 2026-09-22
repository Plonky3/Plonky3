//! Authenticated mutable read-write memory over the binary-native bus.
//!
//! A cell holds whatever the last write put there, which the read-only argument cannot say.
//!
//! # Two orders
//!
//! The accesses exist twice: as the machine's chips issued them, and sorted by cell then reading.
//!
//! Only the sorted copy is committed here, since the machine's chips are already the other.
//!
//! They produce each access on a channel this copy consumes, and balancing it is the permutation.
//!
//! # Boundaries
//!
//! A proof that stands alone starts from empty memory and exports nothing.
//!
//! A proof continuing an execution inherits a committed image and hands one on.
//!
//! [`RamBoundary`] picks between them, and the two differ at a cell's first access.
//!
//! # What the machine owes
//!
//! Three obligations sit outside the constraints, and [`RamAir`] spells them out.

mod air;
mod error;
mod statement;
mod witness;

#[cfg(test)]
mod tests;

pub use air::RamAir;
pub use error::RamError;
pub use statement::{
    MAX_RAM_BIT_WIDTH, MIN_RAM_ACCESS_COUNT, RamBoundary, RamLayout, RamStatement,
};
pub use witness::{RamAccess, RamTrace};
