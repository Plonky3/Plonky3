//! Verifier-derived shape of one mutable read-write memory.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_field::Field;
use p3_security::SecurityTerm;

use super::RamError;
use crate::BusPlan;

/// Largest address or timestamp width this argument decomposes.
///
/// Both widths become explicit bit columns compared by a ripple adder, so a machine word is the
/// natural ceiling and keeps every witness index a `u64`.
pub const MAX_RAM_BIT_WIDTH: usize = 64;

/// How one proof's memory relates to the executions on either side of it.
///
/// The two variants are not two configurations of one protocol.
///
/// They enforce different constraints at the first access to an address, and only one of them
/// exports anything about the memory this proof leaves behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RamBoundary {
    /// One self-contained proof.
    ///
    /// Memory starts all-zero: the first access to an address reads zero unless it is a write.
    ///
    /// The final image is not exported, so nothing outside this proof can depend on it.
    ///
    /// A machine that continues past this proof must not use this variant, because a later
    /// segment would be free to start from any memory it liked.
    SingleProof,
    /// One segment of a longer execution, bounded by two committed memory images.
    ///
    /// The first access to an address is forced to be a read, and that read is declared on
    /// `incoming`. The last access to an address declares its post-state on `outgoing`.
    ///
    /// Neither image is checked here. Both are ordinary named buses, so the enclosing bus plan
    /// balances them against whatever committed image tables the machine supplies, using the same
    /// multiset machinery every other table uses.
    ///
    /// The incoming read is what makes the inherited value binding: a segment cannot invent the
    /// state it starts from, because the value it reads has to match an entry of the committed
    /// incoming image.
    Segment {
        /// Bus carrying one `(address, value)` tuple per address this segment inherits.
        incoming: String,
        /// Bus carrying one `(address, value)` tuple per address this segment hands on.
        outgoing: String,
    },
}

impl RamBoundary {
    /// Whether this proof exports a memory image to a following segment.
    #[must_use]
    pub const fn is_segment(&self) -> bool {
        // Only the segment variant allocates and constrains the group-end marker.
        matches!(self, Self::Segment { .. })
    }

    /// Names of the image buses this boundary declares, in incoming-then-outgoing order.
    #[must_use]
    pub fn image_buses(&self) -> Option<[&str; 2]> {
        // A single proof declares no image at either edge.
        match self {
            Self::SingleProof => None,
            Self::Segment { incoming, outgoing } => Some([incoming, outgoing]),
        }
    }
}

/// Public shape of one mutable read-write memory.
///
/// Every field here is verifier-derived: none of it depends on the witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamStatement {
    /// Bus on which the machine's chips issue their memory accesses.
    ///
    /// The chips push one access tuple each and this memory pulls it, so the plan's balance proves
    /// the execution-order trace holds exactly the accesses the machine issued.
    pub access_bus: String,
    /// Bus carrying the permutation between the two access orders.
    ///
    /// The execution-order trace pushes every access and the address-sorted trace pulls it, so the
    /// plan's balance proves the two orders hold the same multiset.
    pub order_bus: String,
    /// Number of accesses, equal to the height of both traces.
    pub access_count: usize,
    /// Number of bits in an address.
    pub address_bits: usize,
    /// Number of bits in an execution timestamp.
    pub timestamp_bits: usize,
    /// Number of field components in one stored value.
    pub value_width: usize,
    /// Relationship between this proof's memory and the executions around it.
    pub boundary: RamBoundary,
}

impl RamStatement {
    /// Number of payload slots in one access tuple.
    ///
    /// The operation marker precedes the address bits, the timestamp bits, and the value.
    #[must_use]
    pub const fn access_payload_width(&self) -> usize {
        // Every slot is one field element, and bit columns stay one bit per column.
        1 + self.address_bits + self.timestamp_bits + self.value_width
    }

    /// Number of payload slots in one memory-image tuple.
    ///
    /// An image entry has no operation and no timestamp: it is a value held at an address.
    #[must_use]
    pub const fn image_payload_width(&self) -> usize {
        // The image is a snapshot, so only the address and the value identify an entry.
        self.address_bits + self.value_width
    }

    /// Checks every public dimension and every named channel for self-consistency.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported shape or a bus name used for two roles.
    pub fn validate(&self) -> Result<(), RamError> {
        // A memory with no accesses has no product tree and no first row to constrain.
        if self.access_count == 0 {
            return Err(RamError::EmptyTrace);
        }

        // The permutation rides on the enclosing plan's product tree, whose blocks are aligned.
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

        // The execution clock counts 0 to access_count - 1 and must not wrap.
        //
        // A wrapped clock repeats a timestamp, and two accesses at one address would then have no
        // defined order, so a read could be matched against the later write instead of the earlier.
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

        // Two roles sharing one channel would let a tuple of one role cancel a tuple of the other.
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

    /// Checks this statement against the channels a bus plan actually defines.
    ///
    /// The payload width of a named bus is fixed by every AIR that declares on it, so a mismatch
    /// means the caller pointed this memory at a channel built for something else.
    ///
    /// A static indexed table keeps a narrower lookup tuple, so this is also what refuses an
    /// attempt to serve one through the mutable-memory protocol instead of the table lookup.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement, a missing channel, or a payload-width mismatch.
    pub fn check_against(&self, bus_plan: &BusPlan) -> Result<(), RamError> {
        // Dimensions come first so a width comparison is never made against a nonsense statement.
        self.validate()?;

        // Access and order channels carry the full access tuple.
        for name in [self.access_bus.as_str(), self.order_bus.as_str()] {
            check_payload_width(bus_plan, name, self.access_payload_width())?;
        }

        // Image channels carry only the address and the value.
        for name in self.boundary.image_buses().into_iter().flatten() {
            check_payload_width(bus_plan, name, self.image_payload_width())?;
        }

        Ok(())
    }

    /// Push and pull leaves this memory adds to the enclosing plan, in that order.
    ///
    /// The execution order pushes one access tuple on the permutation channel; the sorted order
    /// pulls it back and this memory also pulls one tuple on the access channel. A segment adds
    /// one pull and one push per address group, which is at most one of each per access.
    ///
    /// The plan's fingerprint error grows with the larger of the two totals, so this is what a
    /// machine designer needs in order to see what a memory of a given size costs in bits.
    #[must_use]
    pub const fn leaf_contribution(&self) -> [usize; 2] {
        // Image declarations are row-conditional, so they occupy a leaf on every row regardless.
        let images = if self.boundary.is_segment() {
            self.access_count
        } else {
            0
        };
        [self.access_count + images, 2 * self.access_count + images]
    }

    /// Soundness of this memory's claims, at the challenge field the transcript samples from.
    ///
    /// This memory runs no random experiment of its own. Its two cross-trace claims are ordinary
    /// multiset claims on the enclosing plan's product tree, and every constraint it adds is
    /// deterministic. What it contributes is leaves, and the plan's union bound already counts
    /// them, so the honest report here is the plan's own term rather than a second one to add on
    /// top. Reporting a separate term would double-charge the same fingerprint.
    ///
    /// The field size is taken from `EF` rather than supplied, because the only size that bounds
    /// this error is the one challenges are actually drawn from. A caller passing a base-field
    /// width would report a bound the protocol never had.
    ///
    /// # Errors
    ///
    /// Returns an error for a statement the plan does not support, or a challenge field with no
    /// room for a challenge at all.
    pub fn security_term<EF: Field>(&self, bus_plan: &BusPlan) -> Result<SecurityTerm, RamError> {
        self.check_against(bus_plan)?;

        // `order().bits()` rounds up, so one bit comes off to reach `floor(log2 |EF|)`.
        let field_bits = NonZeroUsize::new(EF::order().bits().saturating_sub(1) as usize)
            .ok_or(RamError::TrivialChallengeField)?;
        Ok(bus_plan.security_term(field_bits))
    }
}

/// Checks that one named channel exists and carries the expected payload width.
fn check_payload_width(bus_plan: &BusPlan, name: &str, expected: usize) -> Result<(), RamError> {
    // Names are the only stable identity a plan exposes, so lookup is by name.
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
/// Both orders live in one committed matrix. That is not a space optimisation: it is what makes
/// the two orders share a height and a commitment, so no composition step can authenticate one
/// order against a matrix the other order did not come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamLayout {
    /// Number of address bits, repeated here so a layout is self-describing.
    pub address_bits: usize,
    /// Number of timestamp bits.
    pub timestamp_bits: usize,
    /// Number of value components.
    pub value_width: usize,
    /// Width of the shared strict-comparison witness, the wider of the two bit widths.
    pub compare_bits: usize,
    /// Operation marker of the execution-order access.
    pub execution_write: usize,
    /// First execution-order address bit, little-endian.
    pub execution_address: usize,
    /// First execution-order timestamp bit, little-endian.
    pub execution_timestamp: usize,
    /// First execution-order value component.
    pub execution_value: usize,
    /// First carry bit of the clock increment that produces the next row's timestamp.
    ///
    /// The chain holds one carry per timestamp bit. There is no carry out: the statement's
    /// capacity check already forbids a clock that could reach it.
    pub execution_carry: usize,
    /// Operation marker of the address-sorted access.
    pub memory_write: usize,
    /// First address-sorted address bit, little-endian.
    pub memory_address: usize,
    /// First address-sorted timestamp bit, little-endian.
    pub memory_timestamp: usize,
    /// First address-sorted value component.
    pub memory_value: usize,
    /// Whether this sorted row holds the same address as the row before it.
    pub same_address: usize,
    /// Difference bits witnessing the strict increase into this row.
    pub compare_delta: usize,
    /// Carry bits of the ripple adder checking that difference.
    pub compare_carry: usize,
    /// Running-or flags proving the difference is not zero.
    pub compare_nonzero: usize,
    /// Whether this sorted row is the last access at its address, in segment mode only.
    pub group_end: usize,
    /// Total trace width.
    pub width: usize,
}

impl RamLayout {
    /// Derives every offset from a validated statement.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement or a width that overflows a machine word.
    pub fn new(statement: &RamStatement) -> Result<Self, RamError> {
        statement.validate()?;
        let RamStatement {
            address_bits,
            timestamp_bits,
            value_width,
            ..
        } = *statement;

        // Exactly one of the two strict comparisons is active on any sorted row, so both share one
        // difference witness. A row comparing addresses leaves the high columns unconstrained.
        let compare_bits = address_bits.max(timestamp_bits);

        // Every offset is a running total, checked once so no later index can silently wrap.
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

    /// Columns of one access tuple in payload order, for the execution-order trace.
    ///
    /// The order is operation, address bits, timestamp bits, value components.
    pub fn execution_access_columns(&self) -> impl Iterator<Item = usize> + '_ {
        // One iterator drives both the AIR declaration and every test that checks it.
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

    /// Columns of one access tuple in payload order, for the address-sorted trace.
    pub fn memory_access_columns(&self) -> impl Iterator<Item = usize> + '_ {
        // Both orders use one payload order so their fingerprints are comparable.
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

    /// Columns of one memory-image tuple in payload order.
    ///
    /// The order is address bits then value components, taken from the address-sorted trace.
    pub fn image_columns(&self) -> impl Iterator<Item = usize> + '_ {
        // An image entry names a value held at an address and nothing else.
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
    // The operation marker leads so a verifier reading a tuple sees the opcode first.
    core::iter::once(write)
        .chain(address..address + address_bits)
        .chain(timestamp..timestamp + timestamp_bits)
        .chain(value..value + value_width)
}
