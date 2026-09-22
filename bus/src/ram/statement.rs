//! Verifier-derived shape of one mutable read-write memory.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_field::Field;
use p3_security::SecurityTerm;

use super::RamError;
use crate::BusPlan;

/// Largest cell number or clock width this argument decomposes.
///
/// Both become explicit bit columns, so a machine word is the natural ceiling.
///
/// It also keeps every witness index a plain unsigned integer.
pub const MAX_RAM_BIT_WIDTH: usize = 64;

/// How one proof's memory relates to the executions on either side of it.
///
/// These are two statements, not two settings of one.
///
/// They constrain the first access to a cell differently.
///
/// Only one of them says anything about the memory this proof leaves behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RamBoundary {
    /// One self-contained proof.
    ///
    /// Memory starts empty, so a first read of a cell returns zero.
    ///
    /// A first write to a cell is free, since it only overwrites that zero.
    ///
    /// Nothing about the memory left behind leaves the proof.
    ///
    /// An execution that continues past this proof must not choose this.
    ///
    /// Its next part would be free to start from any memory it liked.
    SingleProof,
    /// One part of a longer execution, bounded by two committed memory images.
    ///
    /// The first access to a cell has to be a read, declared on the inherited channel.
    ///
    /// The last access to a cell declares the value it leaves on the handed-on channel.
    ///
    /// Neither image is checked here.
    ///
    /// Both are ordinary named channels the enclosing plan balances like any other.
    ///
    /// Forcing that opening read is what binds the inherited value.
    ///
    /// A part of an execution cannot invent the memory it starts from.
    Segment {
        /// Channel carrying one entry per cell this part of the execution inherits.
        incoming: String,
        /// Channel carrying one entry per cell this part of the execution hands on.
        outgoing: String,
    },
}

impl RamBoundary {
    /// Whether this proof hands a memory image to whatever follows it.
    #[must_use]
    pub const fn is_segment(&self) -> bool {
        // Only a continuing proof allocates and constrains the closing marker.
        matches!(self, Self::Segment { .. })
    }

    /// Names of the image channels, inherited first and handed-on second.
    #[must_use]
    pub fn image_buses(&self) -> Option<[&str; 2]> {
        // A self-contained proof declares no image at either edge.
        match self {
            Self::SingleProof => None,
            Self::Segment { incoming, outgoing } => Some([incoming, outgoing]),
        }
    }
}

/// Public shape of one mutable read-write memory.
///
/// A verifier derives every field below without seeing a witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamStatement {
    /// Channel on which the machine's chips issue their memory accesses.
    ///
    /// Each chip produces one access and this memory consumes it.
    ///
    /// Balance then proves the issuing order holds exactly what the machine issued.
    pub access_bus: String,
    /// Channel carrying the permutation between the two orders.
    ///
    /// The issuing order produces every access and the sorted order consumes it.
    ///
    /// Balance then proves the two orders hold the same accesses.
    pub order_bus: String,
    /// Number of accesses, which is also the height of the trace.
    pub access_count: usize,
    /// Number of bits in a cell number.
    pub address_bits: usize,
    /// Number of bits in a clock reading.
    pub timestamp_bits: usize,
    /// Number of field components in one stored value.
    pub value_width: usize,
    /// How this proof's memory relates to the executions around it.
    pub boundary: RamBoundary,
}

impl RamStatement {
    /// Number of payload slots in one access.
    ///
    /// The operation marker leads, then the cell digits, the clock digits, and the value.
    #[must_use]
    pub const fn access_payload_width(&self) -> usize {
        // One slot per field element, and a digit keeps a slot to itself.
        1 + self.address_bits + self.timestamp_bits + self.value_width
    }

    /// Number of payload slots in one memory-image entry.
    ///
    /// An image is a snapshot, so an entry has no operation and no clock reading.
    #[must_use]
    pub const fn image_payload_width(&self) -> usize {
        // A cell number and its value are all an entry needs.
        self.address_bits + self.value_width
    }

    /// Checks every public dimension and every channel name for self-consistency.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported shape.
    ///
    /// Returns an error for one channel name used in two roles.
    pub fn validate(&self) -> Result<(), RamError> {
        // A memory with no accesses has no product tree and no first row to constrain.
        if self.access_count == 0 {
            return Err(RamError::EmptyTrace);
        }

        // The plan's product tree aligns each block, so a block height is a power of two.
        if !self.access_count.is_power_of_two() {
            return Err(RamError::NonPowerOfTwoAccessCount {
                access_count: self.access_count,
            });
        }
        if self.address_bits == 0 || self.address_bits > MAX_RAM_BIT_WIDTH {
            return Err(RamError::AddressBits {
                address_bits: self.address_bits,
                maximum: MAX_RAM_BIT_WIDTH,
            });
        }
        if self.timestamp_bits == 0 || self.timestamp_bits > MAX_RAM_BIT_WIDTH {
            return Err(RamError::TimestampBits {
                timestamp_bits: self.timestamp_bits,
                maximum: MAX_RAM_BIT_WIDTH,
            });
        }

        // The clock counts up from zero once per access, and it must not wrap.
        //
        // A wrap repeats a reading, and two accesses at one cell would then have no order.
        //
        // A read could be matched against the later write instead of the earlier one.
        let capacity = 1u128 << self.timestamp_bits;
        if self.access_count as u128 > capacity {
            return Err(RamError::TimestampCapacity {
                access_count: self.access_count,
                timestamp_bits: self.timestamp_bits,
                capacity,
            });
        }
        if self.value_width == 0 {
            return Err(RamError::EmptyValue);
        }

        // Two roles on one channel would let a tuple of one cancel a tuple of the other.
        let mut names = Vec::with_capacity(4);
        names.push(self.access_bus.as_str());
        names.push(self.order_bus.as_str());
        names.extend(self.boundary.image_buses().into_iter().flatten());
        for (position, name) in names.iter().enumerate() {
            if names[..position].contains(name) {
                return Err(RamError::DuplicateBus {
                    name: (*name).to_string(),
                });
            }
        }

        Ok(())
    }

    /// Checks this statement against the channels a plan actually defines.
    ///
    /// Everything declaring on a channel fixes its width, so a mismatch means the wrong channel.
    ///
    /// A static indexed table keeps a narrower lookup tuple of its own.
    ///
    /// That is what refuses an attempt to serve one through this argument instead.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement or a missing channel.
    ///
    /// Returns an error for a channel whose payload is the wrong width.
    pub fn check_against(&self, bus_plan: &BusPlan) -> Result<(), RamError> {
        // Dimensions come first, so no width is compared against a nonsense statement.
        self.validate()?;

        // The access and permutation channels carry a whole access each.
        for name in [self.access_bus.as_str(), self.order_bus.as_str()] {
            check_payload_width(bus_plan, name, self.access_payload_width())?;
        }

        // An image channel carries only a cell number and a value.
        for name in self.boundary.image_buses().into_iter().flatten() {
            check_payload_width(bus_plan, name, self.image_payload_width())?;
        }

        Ok(())
    }

    /// Produced and consumed leaves this memory adds to the plan, in that order.
    ///
    /// The issuing order produces one access and the sorted order consumes it.
    ///
    /// This memory also consumes one access from the machine's own chips.
    ///
    /// A continuing proof adds an inherited entry and a handed-on entry per row.
    ///
    /// The plan's fingerprint error grows with the larger of the two totals.
    ///
    /// That is what says, in bits, what a memory of a given size costs.
    #[must_use]
    pub const fn leaf_contribution(&self) -> [usize; 2] {
        // A conditional declaration keeps its leaf even where it contributes nothing.
        let images = if self.boundary.is_segment() {
            self.access_count
        } else {
            0
        };
        [self.access_count + images, 2 * self.access_count + images]
    }

    /// Soundness of this memory's claims, at the field the transcript samples from.
    ///
    /// This memory runs no random experiment of its own.
    ///
    /// Its two cross-order claims are ordinary multiset claims on the plan's product tree.
    ///
    /// Every constraint it adds is deterministic.
    ///
    /// What it contributes is leaves, which the plan's union bound already counts.
    ///
    /// So the honest report is the plan's own term rather than a second one on top.
    ///
    /// A separate term would charge the same fingerprint twice.
    ///
    /// The field size is read off the challenge field rather than supplied.
    ///
    /// Only the field challenges are actually drawn from bounds this error.
    ///
    /// A caller passing a base-field width would report a bound the protocol never had.
    ///
    /// # Errors
    ///
    /// Returns an error for a statement the plan does not support.
    ///
    /// Returns an error for a challenge field with no room for a challenge.
    pub fn security_term<EF: Field>(&self, bus_plan: &BusPlan) -> Result<SecurityTerm, RamError> {
        self.check_against(bus_plan)?;

        // The reported bit count rounds up, so one bit comes off to floor it.
        let field_bits = NonZeroUsize::new(EF::order().bits().saturating_sub(1) as usize)
            .ok_or(RamError::TrivialChallengeField)?;
        Ok(bus_plan.security_term(field_bits))
    }
}

/// Checks that one named channel exists and carries the expected payload width.
fn check_payload_width(bus_plan: &BusPlan, name: &str, expected: usize) -> Result<(), RamError> {
    // A name is the only stable identity a plan exposes, so lookup goes by name.
    let domain = bus_plan
        .domains()
        .iter()
        .find(|domain| domain.name == name)
        .ok_or_else(|| RamError::UnknownBus {
            name: name.to_string(),
        })?;
    if domain.payload_width != expected {
        return Err(RamError::PayloadWidth {
            name: name.to_string(),
            expected,
            actual: domain.payload_width,
        });
    }
    Ok(())
}

/// Column offsets of the single trace holding both access orders.
///
/// One trace for both orders is not a space optimisation.
///
/// It makes them share a height and a commitment.
///
/// No later step can then check one order against a trace the other did not come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamLayout {
    /// Number of cell digits, repeated here so a layout describes itself.
    pub address_bits: usize,
    /// Number of clock digits.
    pub timestamp_bits: usize,
    /// Number of value components.
    pub value_width: usize,
    /// Width of the shared comparison witness, the wider of the two digit counts.
    pub compare_bits: usize,
    /// Operation marker of the issuing-order access.
    pub execution_write: usize,
    /// First issuing-order cell digit, least significant first.
    pub execution_address: usize,
    /// First issuing-order clock digit, least significant first.
    pub execution_timestamp: usize,
    /// First issuing-order value component.
    pub execution_value: usize,
    /// First carry of the increment that produces the next row's clock reading.
    ///
    /// The chain holds one carry per clock digit and no carry out.
    ///
    /// The statement's capacity check already forbids a clock that could reach one.
    pub execution_carry: usize,
    /// Operation marker of the sorted-order access.
    pub memory_write: usize,
    /// First sorted-order cell digit, least significant first.
    pub memory_address: usize,
    /// First sorted-order clock digit, least significant first.
    pub memory_timestamp: usize,
    /// First sorted-order value component.
    pub memory_value: usize,
    /// Whether this sorted row touches the same cell as the row before it.
    pub same_address: usize,
    /// Digits of the gap up from the row before this one.
    pub compare_delta: usize,
    /// Carries of the adder that puts that gap back.
    pub compare_carry: usize,
    /// Running flags proving the gap is not zero.
    pub compare_nonzero: usize,
    /// Whether this row is the last access to its cell, in a continuing proof only.
    pub group_end: usize,
    /// Total trace width.
    pub width: usize,
}

impl RamLayout {
    /// Derives every offset from a validated statement.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement.
    ///
    /// Returns an error for a width that overflows a machine word.
    pub fn new(statement: &RamStatement) -> Result<Self, RamError> {
        statement.validate()?;
        let RamStatement {
            address_bits,
            timestamp_bits,
            value_width,
            ..
        } = *statement;

        // Exactly one of the two comparisons runs on a sorted row, so they share one witness.
        //
        // Whichever runs leaves any column above its own digit count unconstrained.
        let compare_bits = address_bits.max(timestamp_bits);

        // Every offset is a running total, checked once so no later index wraps in silence.
        let mut next = 0usize;
        let mut take = |count: usize| -> Result<usize, RamError> {
            let start = next;
            next = next.checked_add(count).ok_or(RamError::LayoutOverflow)?;
            Ok(start)
        };

        let execution_write = take(1)?;
        let execution_address = take(address_bits)?;
        let execution_timestamp = take(timestamp_bits)?;
        let execution_value = take(value_width)?;
        let execution_carry = take(timestamp_bits)?;
        let memory_write = take(1)?;
        let memory_address = take(address_bits)?;
        let memory_timestamp = take(timestamp_bits)?;
        let memory_value = take(value_width)?;
        let same_address = take(1)?;
        let compare_delta = take(compare_bits)?;
        let compare_carry = take(compare_bits + 1)?;
        let compare_nonzero = take(compare_bits)?;
        let group_end = take(usize::from(statement.boundary.is_segment()))?;
        let width = next;

        Ok(Self {
            address_bits,
            timestamp_bits,
            value_width,
            compare_bits,
            execution_write,
            execution_address,
            execution_timestamp,
            execution_value,
            execution_carry,
            memory_write,
            memory_address,
            memory_timestamp,
            memory_value,
            same_address,
            compare_delta,
            compare_carry,
            compare_nonzero,
            group_end,
            width,
        })
    }

    /// Columns of one issuing-order access, in payload order.
    ///
    /// The order is operation, cell digits, clock digits, then value components.
    pub fn execution_access_columns(&self) -> impl Iterator<Item = usize> + '_ {
        // One iterator drives both the declaration and every test that checks it.
        access_columns(
            self.execution_write,
            self.execution_address,
            self.execution_timestamp,
            self.execution_value,
            self.address_bits,
            self.timestamp_bits,
            self.value_width,
        )
    }

    /// Columns of one sorted-order access, in payload order.
    pub fn memory_access_columns(&self) -> impl Iterator<Item = usize> + '_ {
        // Both orders share a payload order, so their fingerprints are comparable.
        access_columns(
            self.memory_write,
            self.memory_address,
            self.memory_timestamp,
            self.memory_value,
            self.address_bits,
            self.timestamp_bits,
            self.value_width,
        )
    }

    /// Columns of one memory-image entry, in payload order.
    ///
    /// The cell digits lead, then the value components, read off the sorted order.
    pub fn image_columns(&self) -> impl Iterator<Item = usize> + '_ {
        // An image entry names a value held at a cell and nothing else.
        (self.memory_address..self.memory_address + self.address_bits)
            .chain(self.memory_value..self.memory_value + self.value_width)
    }
}

/// Builds one access tuple's column order from its component offsets.
fn access_columns(
    write: usize,
    address: usize,
    timestamp: usize,
    value: usize,
    address_bits: usize,
    timestamp_bits: usize,
    value_width: usize,
) -> impl Iterator<Item = usize> {
    // The operation marker leads, so whoever reads an access sees what it is first.
    core::iter::once(write)
        .chain(address..address + address_bits)
        .chain(timestamp..timestamp + timestamp_bits)
        .chain(value..value + value_width)
}
