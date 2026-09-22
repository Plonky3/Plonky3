//! Witness generation for one mutable read-write memory.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;

use super::{RamBoundary, RamError, RamLayout, RamStatement};

/// One memory access, in the order the machine issued it.
///
/// There is no clock field, because an access is timed by its position in the list.
///
/// That is exactly what the clock constraint goes on to enforce.
///
/// A caller cannot hand out two accesses at one time, since there is no way to say it.
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
    /// One read of a cell, returning what it held.
    pub fn read(address: u64, value: impl IntoIterator<Item = F>) -> Self {
        // A read carries the value it saw, so the permutation has something to match.
        Self {
            write: false,
            address,
            value: value.into_iter().collect(),
        }
    }

    /// One write of a value to a cell.
    pub fn write(address: u64, value: impl IntoIterator<Item = F>) -> Self {
        // A write carries the new value, which later reads of that cell inherit.
        Self {
            write: true,
            address,
            value: value.into_iter().collect(),
        }
    }
}

/// Row-major trace of one mutable read-write memory.
///
/// Both access orders live here, so committing this commits both.
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
        // A zero-width trace cannot happen, since every statement has an operation column.
        self.values.len() / self.width
    }

    /// Row-major values, ready for a dense matrix.
    #[must_use]
    pub fn values(&self) -> &[F] {
        // Borrowing saves a caller from copying a wide trace just to look at it.
        &self.values
    }

    /// Consumes the trace and returns its row-major values.
    #[must_use]
    pub fn into_values(self) -> Vec<F> {
        // A dense matrix takes ownership, so hand the allocation over instead of cloning.
        self.values
    }

    /// One row of the trace.
    #[must_use]
    pub fn row(&self, row: usize) -> &[F] {
        // Slicing by row is what every test and every dump wants.
        &self.values[row * self.width..(row + 1) * self.width]
    }

    /// One row of the trace, mutably.
    ///
    /// Only this crate's own tests use it, to build the traces the constraints have to reject.
    ///
    /// A caller builds a trace from its accesses instead.
    ///
    /// Every derived column then stays consistent with them.
    #[cfg(test)]
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [F] {
        &mut self.values[row * self.width..(row + 1) * self.width]
    }
}

impl RamStatement {
    /// Builds the committed trace from the accesses a machine issued.
    ///
    /// The sorted order, the comparison witness, the clock, and the markers are derived here.
    ///
    /// A caller supplies only the accesses themselves.
    ///
    /// Memory rules are checked while the trace is built.
    ///
    /// That check is not what makes the proof sound, since the constraints are.
    ///
    /// It turns a witness bug into a named error rather than an unexplained failure.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement or a witness of the wrong shape.
    ///
    /// Returns an error for a cell number too large to write down.
    ///
    /// Returns an error for a read that breaks continuity.
    ///
    /// Returns an error when a continuing proof leaves a cell's first access unopened.
    pub fn build_trace<F: Field>(
        &self,
        accesses: &[RamAccess<F>],
    ) -> Result<RamTrace<F>, RamError> {
        let layout = RamLayout::new(self)?;

        // Every dimension is public, so a mismatch is the caller's mistake.
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

            // A cell number wider than the statement has no digits to commit.
            if u128::from(access.address) >= 1u128 << self.address_bits {
                return Err(RamError::AddressRange {
                    index,
                    address: access.address,
                    address_bits: self.address_bits,
                });
            }
        }

        let mut values = vec![F::ZERO; self.access_count * layout.width];

        // The issuing order is the order they arrived, timed by position.
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

            // The clock adds one to this row's reading to reach the next row's.
            let mut carry = 1u64;
            row[layout.execution_carry] = F::ONE;
            for bit in 1..self.timestamp_bits {
                carry &= (index as u64 >> (bit - 1)) & 1;
                row[layout.execution_carry + bit] = F::from_bool(carry == 1);
            }
        }

        // Sorting goes by cell, then by time, and a time belongs to exactly one access.
        let mut order = (0..self.access_count).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&index| (accesses[index].address, index));

        for (sorted, &index) in order.iter().enumerate() {
            let access = &accesses[index];
            let previous = sorted.checked_sub(1).map(|earlier| order[earlier]);
            let same_address =
                previous.is_some_and(|earlier| accesses[earlier].address == access.address);

            // A cell's opening row owes something different at each kind of boundary.
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
                // Within a cell's run, a read returns whatever the access before it left.
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

            // One gap witness serves whichever of the two comparisons is running.
            if let Some(earlier) = previous {
                let (left, right, bits) = if same_address {
                    (earlier as u64, index as u64, self.timestamp_bits)
                } else {
                    (accesses[earlier].address, access.address, self.address_bits)
                };
                fill_comparison(row, &layout, left, right, bits);
            }
        }

        // A cell's run ends where the next begins, and the last row always ends one.
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

/// Writes one access into a row at the given component offsets.
fn write_access<F: Field>(
    row: &mut [F],
    offsets: [usize; 4],
    layout: &RamLayout,
    access: &RamAccess<F>,
    clock: u64,
) {
    // Offsets arrive in operation, cell, clock, then value order.
    let [write, address, timestamp, value] = offsets;

    // Digits run least significant first in both orders, which the comparisons assume.
    row[write] = F::from_bool(access.write);
    for bit in 0..layout.address_bits {
        row[address + bit] = F::from_bool((access.address >> bit) & 1 == 1);
    }
    for bit in 0..layout.timestamp_bits {
        row[timestamp + bit] = F::from_bool((clock >> bit) & 1 == 1);
    }
    row[value..value + layout.value_width].copy_from_slice(&access.value);
}

/// Fills the adder witness showing the later key exceeds the earlier one.
///
/// The gap is added back to the earlier key, so this is the carry chain the constraints check.
///
/// Columns above the compared width stay zero, because the wider comparison owns them.
///
/// Nothing reads them here beyond checking each holds a bit.
fn fill_comparison<F: Field>(
    row: &mut [F],
    layout: &RamLayout,
    left: u64,
    right: u64,
    bits: usize,
) {
    debug_assert!(right > left, "a strict increase needs a positive gap");
    let delta = right - left;

    let mut carry = 0u64;
    let mut nonzero = 0u64;
    for bit in 0..bits {
        let addend = (left >> bit) & 1;
        let difference = (delta >> bit) & 1;

        row[layout.compare_delta + bit] = F::from_bool(difference == 1);
        row[layout.compare_carry + bit] = F::from_bool(carry == 1);

        // The majority of the three inputs is the adder's carry out.
        carry = (addend & difference) | (addend & carry) | (difference & carry);
        nonzero |= difference;
        row[layout.compare_nonzero + bit] = F::from_bool(nonzero == 1);
    }

    // A strict increase never wraps, so the last carry is zero and that column stays so.
    debug_assert_eq!(carry, 0, "a strict increase within the width cannot wrap");
    debug_assert_eq!(nonzero, 1, "a strict increase has a nonzero gap");
}
