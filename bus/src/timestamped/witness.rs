//! Checked clock-gap digits and the corresponding range-table witness.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field};

use super::{CLOCK_GAP_BITS, CLOCK_RANGE_BITS, ClockGap, ClockRangeAir, RangeRead, high_step};

/// The two range-table indices of a gap in `1..=2^32`.
///
/// A gap is `(low + 1) + 2^16 * high`, including `2^32` when both digits are `65535`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClockGapDigits {
    low: u16,
    high: u16,
}

impl ClockGapDigits {
    /// Checks a strict clock gap and keeps its range-table indices.
    ///
    /// Returns `None` outside `1..=2^32`.
    #[must_use]
    pub const fn new(gap: u64) -> Option<Self> {
        if gap == 0 || gap > 1 << CLOCK_GAP_BITS {
            return None;
        }
        Some(Self {
            low: (gap - 1) as u16,
            high: ((gap - 1) >> CLOCK_RANGE_BITS) as u16,
        })
    }

    /// Index in the low-factor table.
    #[must_use]
    pub const fn low(self) -> usize {
        self.low as usize
    }

    /// Index in the high-factor table.
    #[must_use]
    pub const fn high(self) -> usize {
        self.high as usize
    }

    /// The factors `(lo, hi)` proving this gap with clock field `C` and bus field `F`.
    #[must_use]
    pub fn factors<C: Field, F: ExtensionField<C>>(self) -> (F, F) {
        (
            F::from(C::GENERATOR.exp_u64(self.low as u64 + 1)),
            F::from(high_step::<C>().exp_u64(self.high as u64)),
        )
    }
}

/// Builds the range reads and final [`ClockRangeAir`] columns of one memory.
///
/// Share one builder across every table and slot using the same pair of range buses.
/// Padding accesses also consume reads: use the digits of gap one for their factors.
/// Counts and their inverses advance multiplicatively from one.
/// As with the read-only memory bus, the total reads of an entry must stay below the order of `F::GENERATOR`.
#[derive(Clone, Debug)]
pub struct ClockRangeWitness<C, F> {
    columns: [Vec<F>; 4],
    count_inverses: [Vec<F>; 2],
    inverse_step: F,
    marker: PhantomData<fn() -> C>,
}

impl<C: Field, F: ExtensionField<C>> ClockRangeWitness<C, F> {
    /// Initializes both fixed tables with count one and no reads.
    #[must_use]
    pub fn new() -> Self {
        let height = ClockRangeAir::<C, F>::HEIGHT;
        let mut columns = core::array::from_fn(|_| vec![F::ONE; height]);
        for (column, first, step) in [
            (ClockRangeAir::<C, F>::LOW, C::GENERATOR, C::GENERATOR),
            (ClockRangeAir::<C, F>::HIGH, C::ONE, high_step::<C>()),
        ] {
            let mut factor = first;
            for entry in &mut columns[column] {
                *entry = F::from(factor);
                factor *= step;
            }
        }
        Self {
            columns,
            count_inverses: core::array::from_fn(|_| vec![F::ONE; height]),
            inverse_step: F::GENERATOR.inverse(),
            marker: PhantomData,
        }
    }

    /// Consumes one read of each factor, indexing directly by the checked digits.
    #[must_use]
    pub fn read(&mut self, digits: ClockGapDigits) -> ClockGap<F> {
        let mut read = |side: usize, index: usize| {
            let (value_column, count_column) = [
                (ClockRangeAir::<C, F>::LOW, ClockRangeAir::<C, F>::LOW_FINAL),
                (
                    ClockRangeAir::<C, F>::HIGH,
                    ClockRangeAir::<C, F>::HIGH_FINAL,
                ),
            ][side];
            let value = self.columns[value_column][index];
            let count = &mut self.columns[count_column][index];
            let count_inverse = &mut self.count_inverses[side][index];
            let result = RangeRead {
                value,
                count: *count,
                count_inverse: *count_inverse,
            };
            *count *= F::GENERATOR;
            *count_inverse *= self.inverse_step;
            result
        };
        ClockGap {
            low: read(0, digits.low()),
            high: read(1, digits.high()),
        }
    }

    /// Finishes as column-major `[low factor, low final count, high factor, high final count]`.
    #[must_use]
    pub fn into_columns(self) -> [Vec<F>; 4] {
        self.columns
    }
}

impl<C: Field, F: ExtensionField<C>> Default for ClockRangeWitness<C, F> {
    fn default() -> Self {
        Self::new()
    }
}
