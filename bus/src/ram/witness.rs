//! Witness generation for one mutable read-write memory.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

use super::{RamBoundary, RamError, RamLayout, RamStatement};

/// One memory access, in the order the machine issued it.
///
/// There is no timestamp field: the execution timestamp of an access is its position in the list,
/// and that is exactly what the clock constraint enforces. A caller cannot hand out two accesses
/// with one timestamp, because the representation has no way to say it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamAccess<F> {
    /// Whether this access replaces the stored value.
    pub write: bool,
    /// Cell this access touches.
    pub address: u64,
    /// Value read, or value written, one field element per component.
    pub value: Vec<F>,
}

impl<F: Field> RamAccess<F> {
    /// One read of `address` returning `value`.
    pub fn read(address: u64, value: impl IntoIterator<Item = F>) -> Self {
        // Reads carry the value they observed so the permutation can match them.
        Self {
            write: false,
            address,
            value: value.into_iter().collect(),
        }
    }

    /// One write of `value` to `address`.
    pub fn write(address: u64, value: impl IntoIterator<Item = F>) -> Self {
        // A write's value is the new state, which later reads at this address inherit.
        Self {
            write: true,
            address,
            value: value.into_iter().collect(),
        }
    }
}

/// Row-major trace of one mutable read-write memory.
///
/// Both access orders live in this one matrix, so committing it commits both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamTrace<F> {
    /// Row-major values.
    values: Vec<F>,
    /// Columns per row.
    width: usize,
}

impl<F> RamTrace<F> {
    /// Columns per row.
    #[must_use]
    pub const fn width(&self) -> usize {
        // The width is the AIR's width, so a caller never has to recompute it.
        self.width
    }

    /// Rows in the trace.
    #[must_use]
    pub const fn height(&self) -> usize {
        // A zero-width trace is unreachable: every statement has at least an operation column.
        self.values.len() / self.width
    }

    /// Row-major values, ready for a dense matrix.
    #[must_use]
    pub fn values(&self) -> &[F] {
        // Borrowing keeps a caller from copying a wide trace to inspect it.
        &self.values
    }

    /// Consumes the trace and returns its row-major values.
    #[must_use]
    pub fn into_values(self) -> Vec<F> {
        // A dense matrix takes ownership, so hand the allocation over rather than cloning it.
        self.values
    }

    /// One row of the trace.
    #[must_use]
    pub fn row(&self, row: usize) -> &[F] {
        // Row slicing is what every test and every debug dump wants.
        &self.values[row * self.width..(row + 1) * self.width]
    }

    /// One row of the trace, mutably.
    ///
    /// Only this crate's own tests use it, to build the malformed traces the constraints have to
    /// reject. A caller builds a trace from its accesses instead, so that every derived column
    /// stays consistent with them.
    #[cfg(test)]
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [F] {
        &mut self.values[row * self.width..(row + 1) * self.width]
    }
}

impl RamStatement {
    /// Builds the committed trace for one execution-ordered access list.
    ///
    /// The address-sorted order, every comparison witness, the clock, and the group markers are
    /// all derived here, so a caller supplies only the accesses themselves.
    ///
    /// Memory semantics are checked while the trace is built. That check is not what makes the
    /// proof sound — the AIR is — but it turns a witness bug into a named error instead of a
    /// constraint failure with no explanation.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement, a witness that does not match its dimensions,
    /// an out-of-range address, a read that breaks continuity, or, in segment mode, an address
    /// group that is not opened by a read.
    pub fn build_trace<F: Field>(
        &self,
        accesses: &[RamAccess<F>],
    ) -> Result<RamTrace<F>, RamError> {
        let layout = RamLayout::new(self)?;

        // Every dimension is public, so a mismatch is the caller's, not the prover's.
        if accesses.len() != self.access_count {
            return Err(RamError::AccessCount {
                expected: self.access_count,
                actual: accesses.len(),
            });
        }
        for (index, access) in accesses.iter().enumerate() {
            if access.value.len() != self.value_width {
                return Err(RamError::ValueWidth {
                    index,
                    expected: self.value_width,
                    actual: access.value.len(),
                });
            }

            // An address wider than the statement has no bit decomposition to commit.
            if u128::from(access.address) >= 1u128 << self.address_bits {
                return Err(RamError::AddressRange {
                    index,
                    address: access.address,
                    address_bits: self.address_bits,
                });
            }
        }

        let mut values = vec![F::ZERO; self.access_count * layout.width];

        // Execution order is the order the accesses arrived, timestamped by position.
        for (index, access) in accesses.iter().enumerate() {
            let row = &mut values[index * layout.width..(index + 1) * layout.width];
            write_access(
                row,
                [
                    layout.execution_write,
                    layout.execution_address,
                    layout.execution_timestamp,
                    layout.execution_value,
                ],
                &layout,
                access,
                index as u64,
            );

            // The clock adds one to this row's timestamp to reach the next row's.
            let mut carry = 1u64;
            row[layout.execution_carry] = F::ONE;
            for bit in 1..self.timestamp_bits {
                carry &= (index as u64 >> (bit - 1)) & 1;
                row[layout.execution_carry + bit] = F::from_bool(carry == 1);
            }
        }

        // Address order breaks ties by timestamp, which is the access index and therefore unique.
        let mut order = (0..self.access_count).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&index| (accesses[index].address, index));

        for (sorted, &index) in order.iter().enumerate() {
            let access = &accesses[index];
            let previous = sorted.checked_sub(1).map(|earlier| order[earlier]);
            let same_address =
                previous.is_some_and(|earlier| accesses[earlier].address == access.address);

            // The opening row of a group owes something different in each boundary mode.
            if !same_address {
                match self.boundary {
                    RamBoundary::SingleProof => {
                        if !access.write && access.value.iter().any(|&value| value != F::ZERO) {
                            return Err(RamError::ReadContinuity { index: sorted });
                        }
                    }
                    RamBoundary::Segment { .. } => {
                        if access.write {
                            return Err(RamError::UnopenedSegmentGroup { index: sorted });
                        }
                    }
                }
            } else if !access.write {
                // Inside a group a read returns whatever the preceding access left.
                let earlier = &accesses[previous.expect("a continued group has a previous row")];
                if access.value != earlier.value {
                    return Err(RamError::ReadContinuity { index: sorted });
                }
            }

            let row = &mut values[sorted * layout.width..(sorted + 1) * layout.width];
            write_access(
                row,
                [
                    layout.memory_write,
                    layout.memory_address,
                    layout.memory_timestamp,
                    layout.memory_value,
                ],
                &layout,
                access,
                index as u64,
            );
            row[layout.same_address] = F::from_bool(same_address);

            // One difference witness serves whichever of the two comparisons is active.
            if let Some(earlier) = previous {
                let (left, right, bits) = if same_address {
                    (earlier as u64, index as u64, self.timestamp_bits)
                } else {
                    (accesses[earlier].address, access.address, self.address_bits)
                };
                fill_comparison(row, &layout, left, right, bits);
            }
        }

        // A group ends where the next group begins, and the last row always ends one.
        if self.boundary.is_segment() {
            for sorted in 0..self.access_count {
                let ends = sorted + 1 == self.access_count || {
                    let next = order[sorted + 1];
                    accesses[next].address != accesses[order[sorted]].address
                };
                values[sorted * layout.width + layout.group_end] = F::from_bool(ends);
            }
        }

        Ok(RamTrace {
            values,
            width: layout.width,
        })
    }
}

/// Writes one access tuple into a row at the given component offsets.
fn write_access<F: Field>(
    row: &mut [F],
    offsets: [usize; 4],
    layout: &RamLayout,
    access: &RamAccess<F>,
    clock: u64,
) {
    // Offsets arrive in operation, address, timestamp, value order, one per component family.
    let [write, address, timestamp, value] = offsets;

    // Bit columns are little-endian in both orders, which is what the comparisons assume.
    row[write] = F::from_bool(access.write);
    for bit in 0..layout.address_bits {
        row[address + bit] = F::from_bool((access.address >> bit) & 1 == 1);
    }
    for bit in 0..layout.timestamp_bits {
        row[timestamp + bit] = F::from_bool((clock >> bit) & 1 == 1);
    }
    row[value..value + layout.value_width].copy_from_slice(&access.value);
}

/// Fills the ripple-adder witness proving `right > left` over `bits` unsigned bits.
///
/// The difference is added back to `left`, so the carry chain here is the one the AIR checks.
/// Columns above `bits` stay zero: the wider of the two comparisons owns them, and the AIR only
/// asserts they are Boolean.
fn fill_comparison<F: Field>(
    row: &mut [F],
    layout: &RamLayout,
    left: u64,
    right: u64,
    bits: usize,
) {
    debug_assert!(
        right > left,
        "a strict increase needs a positive difference"
    );
    let delta = right - left;

    let mut carry = 0u64;
    let mut nonzero = 0u64;
    for bit in 0..bits {
        let addend = (left >> bit) & 1;
        let difference = (delta >> bit) & 1;

        row[layout.compare_delta + bit] = F::from_bool(difference == 1);
        row[layout.compare_carry + bit] = F::from_bool(carry == 1);

        // Majority of the three inputs is the full adder's carry out.
        carry = (addend & difference) | (addend & carry) | (difference & carry);
        nonzero |= difference;
        row[layout.compare_nonzero + bit] = F::from_bool(nonzero == 1);
    }

    // A strict increase never wraps, so the final carry is zero and the column stays at it.
    debug_assert_eq!(
        carry, 0,
        "a strict increase within the bit width cannot wrap"
    );
    debug_assert_eq!(nonzero, 1, "a strict increase has a nonzero difference");
}
