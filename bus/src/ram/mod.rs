//! Authenticated mutable read-write memory over the binary-native bus.
//!
//! A machine's memory is mutable: a cell holds whatever the last write put there.
//!
//! The read-only argument next door cannot say that, because it fixes one value per cell.
//!
//! Its counting proof also needs a read to carry the same value on both sides of the bus.
//!
//! A write breaks that, so mutable memory needs a different argument.
//!
//! # Two orders
//!
//! The accesses exist twice: as the machine's chips issued them, and sorted by cell then time.
//!
//! Only the sorted copy is committed here, because the machine's chips already are the other.
//!
//! They produce each access on a named channel and this copy consumes it.
//!
//! Balancing that channel is the permutation, and the enclosing plan already runs it.
//!
//! Sorting is what makes every memory rule local.
//!
//! Each rule below reads one row against the row above it.
//!
//! # The clock
//!
//! A clock reading belongs to the machine, and nothing here can check it against real time.
//!
//! A machine has to constrain its own readings to rise along its execution.
//!
//! What this argument adds is that two accesses to one cell cannot share a reading.
//!
//! Readings at different cells may collide freely, because no read's answer depends on that.
//!
//! An earlier draft kept a second committed copy whose reading was its row position.
//!
//! It bound nothing: a multiset claim lets the machine attach those readings in any order.
//!
//! Dropping it removed one operation, one cell, two clocks and one value per access.
//!
//! # The comparison
//!
//! Sorting needs a greater-than, which a field does not have.
//!
//! Each row witnesses the gap up from the row above it and adds that gap back.
//!
//! The carry has to vanish at the top, which is what makes the comparison unsigned.
//!
//! The gap has to be nonzero, which is what makes it strict rather than merely non-decreasing.
//!
//! # Nothing new on the wire
//!
//! The permutation is not a new reduction, only the plan's ordinary multiset balance.
//!
//! Static indexed tables keep the lookup path they already had.
//!
//! An immutable entry has no operation and no reading, so its tuple cannot pass for this one.
//!
//! # Boundaries
//!
//! A proof that stands alone starts from empty memory and exports nothing.
//!
//! A proof continuing an execution inherits a committed image and hands one on.
//!
//! Those are different statements, and confusing them is the easiest mistake to make here.
//!
//! # Against the references
//!
//! leanVM's memory is write-once, so it needs no ordering and keeps no sorted copy.
//!
//! Its offline check is the per-cell read counter our read-only module already uses.
//!
//! binius64 has no memory argument at all.
//!
//! Where leanVM does compare magnitudes it puts integers in a generator's exponent.
//!
//! A comparison is then a range check on a logarithm, settled by two table lookups.
//!
//! That is cheaper when a column costs a whole field element, which here it does not.
//!
//! This backend stores a bit column as a bit, and digits need no range table.
//!
//! binius64 states a whole sixty-four-bit borrow chain as one relation over packed lanes.
//!
//! The shape below is already that adder, so it should collapse once the word backend can.
//!
//! # Cost
//!
//! A statement commits one operation column, its cell and clock digits, and its value.
//!
//! On top of those sits a comparison witness a little wider than the larger digit count.
//!
//! Most of those columns hold a single bit, which the Boolean commitment path stores cheaply.
//!
//! A continuing proof's image channels take one leaf per access, not one per cell.
//!
//! A conditional declaration still holds its leaf on rows where it contributes nothing.
//!
//! Narrowing it is impossible, since the witness decides which rows open and close a run.
//!
//! # What remains the caller's
//!
//! Committing the trace, running the plan, and checking the constraints, as for any table.
//!
//! Constraining the machine's own clock readings, which nothing here can do for it.
//!
//! A continuing proof's two image tables are the caller's as well.
//!
//! This module declares their tuples and the plan balances them against what is committed.

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
