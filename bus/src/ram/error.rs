//! Errors for the mutable read-write memory argument.

use alloc::string::String;

use thiserror::Error;

/// Invalid statements, layouts, or witnesses for mutable read-write memory.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RamError {
    /// A memory with no accesses has nothing to prove and no product tree.
    #[error("mutable memory requires at least one access")]
    EmptyTrace,
    /// The permutation reduction reuses the enclosing bus plan's product tree.
    ///
    /// That tree addresses one aligned block per declaration, so a block height must be a power of two.
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
    /// Distinct execution timestamps need one clock value per access.
    ///
    /// A clock that wrapped would repeat a timestamp, and two accesses at one address would then
    /// have no defined order, so a read could be matched against either of them.
    #[error(
        "mutable memory covers {access_count} accesses, but {timestamp_bits} timestamp bits count only {capacity}"
    )]
    TimestampCapacity {
        /// Access count supplied by the statement.
        access_count: usize,
        /// Timestamp width supplied by the statement.
        timestamp_bits: usize,
        /// Distinct clock values that width can represent.
        capacity: u128,
    },
    /// An access with no value components carries no memory state.
    #[error("mutable memory requires at least one value component")]
    EmptyValue,
    /// Two roles cannot share one named bus without their tuples cancelling.
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
    /// The named bus tuple cannot hold this memory's payload.
    ///
    /// A static indexed table keeps its own narrower lookup tuple, so this also refuses a plan that
    /// tried to reuse a table channel as a mutable-memory channel.
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
    #[error(
        "mutable memory access {index} uses address {address}, which needs more than {address_bits} bits"
    )]
    AddressRange {
        /// Position of the malformed access in execution order.
        index: usize,
        /// Address supplied by the witness.
        address: u64,
        /// Width fixed by the statement.
        address_bits: usize,
    },
    /// A segment witness reads an address it never opened against the incoming image.
    #[error(
        "mutable memory segment access {index} is the first at its address but is not a read that opens the group"
    )]
    UnopenedSegmentGroup {
        /// Position of the offending access in address-then-timestamp order.
        index: usize,
    },
    /// The transcript's challenge field has no room for a challenge.
    #[error("mutable memory needs a challenge field larger than one element")]
    TrivialChallengeField,
    /// A read returns a value other than the one the last write left at its address.
    #[error(
        "mutable memory access {index} reads a value the preceding access at its address did not leave"
    )]
    ReadContinuity {
        /// Position of the offending access in address-then-timestamp order.
        index: usize,
    },
}
