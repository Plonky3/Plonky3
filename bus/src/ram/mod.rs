//! Authenticated mutable read-write memory over the binary-native bus.
//!
//! A machine's memory is mutable: the value at an address is whatever the last write left there.
//! The read-only argument next door cannot say that. It seeds one immutable value per address and
//! proves every read matches it, which is exactly the wrong shape once a cell can change.
//!
//! This module proves the mutable statement with the classical offline-memory-checking
//! construction: write the accesses down twice, once in the order the machine issued them and
//! once sorted by address and then by time, prove the two orders hold the same accesses, and read
//! the memory semantics off the sorted order, where they are local.
//!
//! # The two orders
//!
//! Both orders live in one committed matrix, so one commitment binds both and no composition step
//! can authenticate one order against a matrix the other did not come from.
//!
//! The execution order is timestamped by row index. An explicit increment chain proves the clock
//! runs `0, 1, ..., n - 1` and refuses its carry out, so the timestamps are pairwise distinct and
//! the sorted order has a unique answer.
//!
//! The sorted order is compared row against row with a ripple adder over bit columns: the later
//! row's key is the earlier row's key plus a nonzero difference that does not carry out. That is
//! an unsigned, non-wrapping, strict increase, which is what makes address groups contiguous and
//! within-group time strictly forward.
//!
//! # Nothing here is a second protocol
//!
//! The permutation is not a new reduction. The execution order pushes each access on a named bus
//! and the sorted order pulls it, so the plan's existing multiset balance proves the two orders
//! agree. The machine's own chips push their accesses on a second named bus that this memory
//! pulls, which is what binds the execution order to the computation.
//!
//! Static indexed tables keep their own argument: an immutable table has no timestamps to sort
//! and no writes to order, so it stays on the lookup path, and [`RamStatement::check_against`]
//! refuses a channel whose payload width belongs to one.
//!
//! # Relationship to the references
//!
//! leanVM solves a different problem. Its memory is write-once, so the value at an address is
//! invariant and its offline memory check collapses to a per-address read counter in the orbit of
//! a multiplicative generator — which is the construction our read-only module already uses. With
//! a mutable cell a write's old and new values differ, the two bus sides no longer cancel at a
//! fixed `(address, value)` pair, and that counting argument stops working. Ordering is what
//! replaces it, which is why this module has a sorted trace and leanVM has none.
//!
//! binius64 has no memory argument at all.
//!
//! Where leanVM does compare magnitudes it avoids bit columns: an integer lives in the exponent of
//! a generator, so `t < t'` becomes a range check on a discrete logarithm, discharged by two
//! lookups against the address table. That is cheaper in a field-element-per-column cost model.
//! We decompose into bits instead, because this backend commits Boolean columns at one bit each,
//! and because bit columns keep the argument self-contained: no range table, no second channel
//! per comparison, and no dependence on the orbit length of a generator.
//!
//! binius64's word-level comparison gates point at the eventual optimisation here. It expresses a
//! whole 64-bit borrow chain as one constraint over packed lanes rather than 64 Boolean columns.
//! Once the word backend can state a packed-lane relation, the comparison witness below should
//! collapse the same way; the constraint shape is already the same full adder.
//!
//! # Cost
//!
//! A statement over `A` address bits, `T` timestamp bits, and `V` value components commits
//! `2(1 + A + T + V) + T + 3 max(A, T) + 2` columns per access, plus one more for a segment. Most
//! of those are single bits, which this backend's Boolean commitment path stores at one bit each
//! rather than one field element each.
//!
//! On the bus, the access and permutation channels take one leaf per access per declaration. A
//! segment's two image channels take one leaf per access as well, not one per address group,
//! because a conditional declaration still occupies its leaf on the rows where it is inactive.
//! That is not an oversight: which rows open and close a group is witness-dependent, so the
//! declaration cannot be narrowed to a short block the way a fixed boundary flush can be.
//! [`RamStatement::leaf_contribution`] reports the totals.
//!
//! # What is still the caller's
//!
//! Committing the trace, running the enclosing bus plan, and checking this AIR are the caller's,
//! as they are for every other table. In segment mode the incoming and outgoing image tables are
//! the caller's too: this module declares the tuples, and the plan balances them against whatever
//! committed images the machine supplies.

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
