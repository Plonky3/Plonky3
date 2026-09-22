//! Errors for the mutable read-write memory argument.

use alloc::string::String;

use thiserror::Error;

/// Invalid statements, layouts, or witnesses for mutable read-write memory.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RamError {
    /// A one-row trace has no transition, so nothing compares a row against the row above it.
    #[error("mutable memory covers {access_count} accesses, expected at least {minimum}")]
    TooFewAccesses {
        /// Access count supplied by the statement.
        access_count: usize,
        /// Fewest accesses a statement may cover.
        minimum: usize,
    },
    /// The permutation reuses the enclosing plan's product tree.
    ///
    /// That tree aligns one block per declaration, so a block height is a power of two.
    #[error("mutable memory covers {access_count} accesses, which is not a power of two")]
    NonPowerOfTwoAccessCount {
        /// Access count supplied by the statement.
        access_count: usize,
    },
    /// Addresses are compared bit by bit, so the width must fit a machine word.
    #[error("mutable memory has {address_bits} address bits, expected 1 to {maximum}")]
    AddressBits {
        /// Width supplied by the statement.
        address_bits: usize,
        /// Largest supported width.
        maximum: usize,
    },
    /// Timestamps are compared bit by bit, so the width must fit a machine word.
    #[error("mutable memory has {timestamp_bits} timestamp bits, expected 1 to {maximum}")]
    TimestampBits {
        /// Width supplied by the statement.
        timestamp_bits: usize,
        /// Largest supported width.
        maximum: usize,
    },
    /// An access with no value components carries no memory state.
    #[error("mutable memory requires at least one value component")]
    EmptyValue,
    /// Two roles on one channel would let a tuple of one cancel a tuple of the other.
    #[error("mutable memory reuses bus {name} for two different roles")]
    DuplicateBus {
        /// Caller-owned name used for more than one role.
        name: String,
    },
    /// A named bus carries no declarations in the enclosing plan.
    #[error("mutable memory bus {name} does not exist in this plan")]
    UnknownBus {
        /// Caller-owned name that the plan does not define.
        name: String,
    },
    /// The named channel's tuple cannot hold this memory's payload.
    ///
    /// A static indexed table keeps its own narrower lookup tuple.
    ///
    /// This is what refuses an attempt to reuse a table channel for mutable memory.
    #[error("mutable memory bus {name} has payload width {actual}, expected {expected}")]
    PayloadWidth {
        /// Caller-owned name of the mismatched bus.
        name: String,
        /// Width fixed by the memory statement.
        expected: usize,
        /// Width supplied by the bus plan.
        actual: usize,
    },
    /// The derived trace width overflows a machine word.
    #[error("mutable memory column layout overflows usize")]
    LayoutOverflow,
    /// A witness carries a different number of accesses than the statement covers.
    #[error("mutable memory witness has {actual} accesses, expected {expected}")]
    AccessCount {
        /// Access count fixed by the statement.
        expected: usize,
        /// Access count supplied by the witness.
        actual: usize,
    },
    /// One access carries the wrong number of value components.
    #[error("mutable memory access {index} has {actual} value components, expected {expected}")]
    ValueWidth {
        /// Position of the malformed access in execution order.
        index: usize,
        /// Width fixed by the statement.
        expected: usize,
        /// Width supplied by the witness.
        actual: usize,
    },
    /// One access names an address the statement cannot represent.
    #[error("mutable memory access {index} names cell {address}, wider than {address_bits} digits")]
    AddressRange {
        /// Position of the malformed access in execution order.
        index: usize,
        /// Address supplied by the witness.
        address: u64,
        /// Width fixed by the statement.
        address_bits: usize,
    },
    /// One access names a clock reading the statement cannot represent.
    #[error("mutable memory access {index} is timed {time}, wider than {timestamp_bits} digits")]
    TimeRange {
        /// Position of the malformed access in the caller's list.
        index: usize,
        /// Clock reading supplied by the witness.
        time: u64,
        /// Width fixed by the statement.
        timestamp_bits: usize,
    },
    /// Two accesses to one cell share a clock reading, so they have no order between them.
    #[error("mutable memory has two accesses to cell {address} at reading {time}")]
    RepeatedTime {
        /// Position of the later access in address-then-reading order.
        index: usize,
        /// Cell both accesses touch.
        address: u64,
        /// Reading both accesses claim.
        time: u64,
    },
    /// A continuing proof touches a cell it never opened against the inherited image.
    #[error(
        "mutable memory access {index} is the first at its cell but does not open it with a read"
    )]
    UnopenedSegmentGroup {
        /// Position of the offending access in address-then-timestamp order.
        index: usize,
    },
    /// The transcript's challenge field has no room for a challenge.
    #[error("mutable memory needs a challenge field larger than one element")]
    TrivialChallengeField,
    /// A read returns something other than what the last write left at its cell.
    #[error("mutable memory access {index} reads a value the access before it did not leave")]
    ReadContinuity {
        /// Position of the offending access in address-then-timestamp order.
        index: usize,
    },
}
