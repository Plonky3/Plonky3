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
//! The accesses are written down twice.
//!
//! One copy runs as the machine issued them, the other sorted by cell and then by time.
//!
//! Proving the copies hold the same accesses makes every memory rule local to the sorted one.
//!
//! Both copies live in one committed trace, so a single commitment binds them together.
//!
//! No later step can check one order against a trace the other did not come from.
//!
//! # The clock
//!
//! Time is the row position in the issuing order.
//!
//! An increment chain proves the clock counts up from zero, and the shape check caps how far.
//!
//! Distinct times give the sorted order one answer instead of several.
//!
//! # The comparison
//!
//! Sorting needs a greater-than, which a field does not have.
//!
//! Each sorted row witnesses the gap up from the row above it and adds that gap back.
//!
//! The carry has to vanish at the top, which is what makes the comparison unsigned.
//!
//! The gap has to be nonzero, which is what makes it strict rather than merely non-decreasing.
//!
//! # Nothing new on the wire
//!
//! The permutation is not a new reduction.
//!
//! The issuing order produces each access on a named channel and the sorted order consumes it.
//!
//! Balancing that channel is the whole proof, and the enclosing plan already does it.
//!
//! The machine's own chips produce their accesses on a second channel this memory consumes.
//!
//! That ties the issuing order to the computation rather than to an invented history.
//!
//! Static indexed tables keep the lookup path they already had.
//!
//! An immutable entry has no operation and no time, so its tuple cannot pass for this one.
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
//! Most columns hold a single bit, which the Boolean commitment path stores cheaply.
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
pub use statement::{MAX_RAM_BIT_WIDTH, RamBoundary, RamLayout, RamStatement};
pub use witness::{RamAccess, RamTrace};
