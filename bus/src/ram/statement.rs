//! Verifier-derived shape of one mutable read-write memory.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_field::Field;
use p3_security::SecurityTerm;

use super::RamError;
use crate::{BusName, BusPlan};

/// Largest cell number or clock width this argument decomposes into digits.
pub const MAX_RAM_BIT_WIDTH: usize = 64;

/// Fewest accesses a statement may cover, since a one-row trace has no transition.
pub const MIN_RAM_ACCESS_COUNT: usize = 2;

/// How one proof's memory relates to the executions on either side of it.
///
/// They differ at a cell's first access, and only one exports the memory left behind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RamBoundary {
    /// One self-contained proof, whose memory starts empty and is not exported.
    ///
    /// A first read of a cell returns zero, and a first write is free.
    ///
    /// A continuing execution must not choose it, or its next part could start from anything.
    SingleProof,
    /// One part of a longer execution, bounded by two committed memory images.
    ///
    /// A cell's first access declares on the inherited channel, its last on the handed-on one.
    ///
    /// Balance binds the start value, since the opening row's own value must match an entry.
    Segment {
        /// Channel carrying one entry per cell this part of the execution inherits.
        incoming: String,
        /// Channel carrying one entry per cell this part of the execution hands on.
        outgoing: String,
    },
}

impl RamBoundary {
    /// Whether this proof hands a memory image on.
    #[must_use]
    pub const fn is_segment(&self) -> bool {
        matches!(self, Self::Segment { .. })
    }

    /// Names of the image channels, inherited first and handed-on second.
    #[must_use]
    pub fn image_buses(&self) -> Option<[&str; 2]> {
        match self {
            Self::SingleProof => None,
            Self::Segment { incoming, outgoing } => Some([incoming, outgoing]),
        }
    }
}

/// Public shape of one mutable read-write memory, derived without seeing a witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamStatement {
    /// Channel the machine's chips issue their accesses on, which this memory consumes.
    pub access_bus: String,
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
    /// Number of payload slots in one access: operation, cell, clock, then value.
    #[must_use]
    pub const fn access_payload_width(&self) -> usize {
        1 + self.address_bits + self.timestamp_bits + self.value_width
    }

    /// Number of payload slots in one image entry, which is a cell and a value.
    #[must_use]
    pub const fn image_payload_width(&self) -> usize {
        self.address_bits + self.value_width
    }

    /// Checks every public dimension and every channel name for self-consistency.
    ///
    /// # Errors
    ///
    /// - An unsupported shape.
    /// - A channel name outside the alphabet, or one used in two roles.
    pub fn validate(&self) -> Result<(), RamError> {
        if self.access_count < MIN_RAM_ACCESS_COUNT {
            return Err(RamError::TooFewAccesses {
                access_count: self.access_count,
                minimum: MIN_RAM_ACCESS_COUNT,
            });
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
        if self.value_width == 0 {
            return Err(RamError::EmptyValue);
        }

        // Two roles on one channel would let a tuple of one cancel a tuple of the other.
        let mut names = Vec::with_capacity(3);
        names.push(self.access_bus.as_str());
        names.extend(self.boundary.image_buses().into_iter().flatten());
        for (position, name) in names.iter().enumerate() {
            // Checking the name here means no declaration ever has to check it again.
            BusName::try_new(name).map_err(|error| RamError::BusName {
                name: (*name).to_string(),
                error,
            })?;
            if names[..position].contains(name) {
                return Err(RamError::DuplicateBus {
                    name: (*name).to_string(),
                });
            }
        }

        Ok(())
    }

    /// Checks this statement against the channels a plan defines.
    ///
    /// A static indexed table keeps a narrower tuple, so this refuses one served here.
    ///
    /// # Errors
    ///
    /// - A malformed statement, or a channel the plan does not define.
    /// - A channel whose payload is the wrong width.
    pub fn check_against(&self, bus_plan: &BusPlan) -> Result<(), RamError> {
        self.validate()?;

        check_payload_width(bus_plan, &self.access_bus, self.access_payload_width())?;

        for name in self.boundary.image_buses().into_iter().flatten() {
            check_payload_width(bus_plan, name, self.image_payload_width())?;
        }

        Ok(())
    }

    /// Produced and consumed leaves this memory adds to the plan, in that order.
    ///
    /// It consumes one access per row, and a continuing proof adds an entry each way per row.
    #[must_use]
    pub const fn leaf_contribution(&self) -> [usize; 2] {
        // A conditional declaration keeps its leaf even where it contributes nothing.
        let images = if self.boundary.is_segment() {
            self.access_count
        } else {
            0
        };
        [images, self.access_count + images]
    }

    /// Soundness of this memory's claims, at the field the transcript samples from.
    ///
    /// This memory runs no random experiment, so the honest report is the plan's own term.
    ///
    /// The size is read off that field, so no base-field width can slip in.
    ///
    /// # Errors
    ///
    /// - A statement the plan does not support.
    /// - A challenge field with no room for a challenge.
    pub fn security_term<EF: Field>(&self, bus_plan: &BusPlan) -> Result<SecurityTerm, RamError> {
        self.check_against(bus_plan)?;

        // The reported bit count rounds up, so one bit comes off to floor it.
        let field_bits = NonZeroUsize::new(EF::order().bits().saturating_sub(1) as usize)
            .ok_or(RamError::TrivialChallengeField)?;
        Ok(bus_plan.security_term(field_bits))
    }
}

/// Reads back a channel name the statement already checked.
pub(super) fn channel(name: &str) -> BusName<'_> {
    BusName::try_new(name).expect("a validated statement holds only well-formed channel names")
}

/// Checks that one named channel exists and carries the expected payload width.
fn check_payload_width(bus_plan: &BusPlan, name: &str, expected: usize) -> Result<(), RamError> {
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

/// Column offsets of the trace, whose rows are the accesses sorted by cell then reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamLayout {
    /// Number of cell digits.
    pub address_bits: usize,
    /// Number of clock digits.
    pub timestamp_bits: usize,
    /// Number of value components.
    pub value_width: usize,
    /// Width of the shared comparison witness, the wider of the two digit counts.
    pub compare_bits: usize,
    /// Operation marker, set on a write.
    pub operation: usize,
    /// First cell digit, least significant first.
    pub address: usize,
    /// First clock digit, least significant first.
    pub timestamp: usize,
    /// First value component.
    pub value: usize,
    /// Whether this row touches the same cell as the row before it.
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
    /// - A malformed statement.
    /// - A width that overflows a machine word.
    pub fn new(statement: &RamStatement) -> Result<Self, RamError> {
        statement.validate()?;
        let RamStatement {
            address_bits,
            timestamp_bits,
            value_width,
            ..
        } = *statement;

        // Exactly one of the two comparisons runs on a row, so they share one witness.
        let compare_bits = address_bits.max(timestamp_bits);

        let mut next = 0usize;
        let mut take = |count: usize| -> Result<usize, RamError> {
            let start = next;
            next = next.checked_add(count).ok_or(RamError::LayoutOverflow)?;
            Ok(start)
        };

        let operation = take(1)?;
        let address = take(address_bits)?;
        let timestamp = take(timestamp_bits)?;
        let value = take(value_width)?;
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
            operation,
            address,
            timestamp,
            value,
            same_address,
            compare_delta,
            compare_carry,
            compare_nonzero,
            group_end,
            width,
        })
    }

    /// Columns of one access: operation, cell digits, clock digits, then value components.
    pub fn access_columns(&self) -> impl Iterator<Item = usize> + '_ {
        core::iter::once(self.operation)
            .chain(self.address..self.address + self.address_bits)
            .chain(self.timestamp..self.timestamp + self.timestamp_bits)
            .chain(self.value..self.value + self.value_width)
    }

    /// Columns of one image entry: the cell digits, then the value components.
    pub fn image_columns(&self) -> impl Iterator<Item = usize> + '_ {
        (self.address..self.address + self.address_bits)
            .chain(self.value..self.value + self.value_width)
    }
}
