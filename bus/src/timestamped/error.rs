//! Errors for the timestamped read-write memory.

use alloc::string::String;

use thiserror::Error;

use crate::{BusNameError, ReadOnlyMemoryError};

/// Invalid handles or plans for the timestamped read-write memory.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TimestampedMemoryError {
    /// A channel name leaves the alphabet the bus separator binds.
    #[error("timestamped memory channel {name} is not a well-formed name: {error}")]
    BusName {
        /// Caller-owned name the alphabet refused.
        name: String,
        /// Why the alphabet refused it.
        error: BusNameError,
    },
    /// Two roles on one channel would let a tuple of one cancel a tuple of the other.
    #[error("timestamped memory reuses bus {name} for two different roles")]
    DuplicateBus {
        /// Caller-owned name used for more than one role.
        name: String,
    },
    /// A cell with no value components carries no memory state.
    #[error("timestamped memory requires at least one value component")]
    EmptyValue,
    /// A gap of up to `2^32` ticks could be a multiple of the clock order.
    #[error(
        "timestamped memory clock orbit has {orbit_bits} bits, expected at least {minimum_bits}"
    )]
    ClockOrbitTooShort {
        /// Bits of the order of the clock tick.
        orbit_bits: usize,
        /// Fewest bits that keep every gap nonzero.
        minimum_bits: usize,
    },
    /// A range table refused its handle.
    #[error("timestamped memory range table: {0}")]
    RangeTable(#[from] ReadOnlyMemoryError),
    /// A named bus carries no declarations in the enclosing plan.
    #[error("timestamped memory bus {name} does not exist in this plan")]
    UnknownBus {
        /// Caller-owned name that the plan does not define.
        name: String,
    },
    /// The named channel's tuple does not have this memory's payload width.
    #[error("timestamped memory bus {name} has payload width {actual}, expected {expected}")]
    PayloadWidth {
        /// Caller-owned name of the mismatched bus.
        name: String,
        /// Width fixed by the memory.
        expected: usize,
        /// Width supplied by the bus plan.
        actual: usize,
    },
    /// The memory tuple count overflows a machine word.
    #[error("timestamped memory tuple count overflows usize")]
    TupleCountOverflow,
    /// A forged cycle of this many accesses could wrap the clock orbit.
    #[error("timestamped memory pulls {pulls} tuples, each climbing up to 2^{gap_bits} ticks")]
    TooManyAccesses {
        /// Pull declarations on the memory channel.
        pulls: usize,
        /// Bits of the largest gap one access proves.
        gap_bits: usize,
    },
    /// The transcript's challenge field has no room for a challenge.
    #[error("timestamped memory needs a challenge field larger than one element")]
    TrivialChallengeField,
}
