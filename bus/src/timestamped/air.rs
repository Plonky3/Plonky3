//! Boundary and range tables of one timestamped read-write memory.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};

use super::{
    CLOCK_RANGE_BITS, PrivateRegion, PublicImage, TimestampedMemory, TimestampedMemoryError,
    TimestampedMemoryInteractionBuilder, high_step,
};
use crate::{BusActivation, BusDirection, BusInteractionBuilder};

/// Where the initial value of every cell comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimestampedSeed<F> {
    /// Every cell starts at zero.
    Zero,
    /// Cells start from an image the verifier knows.
    ///
    /// The image is a periodic column, so nothing is committed for it.
    ///
    /// The verifier evaluates it with [`PublicImage::evaluate`].
    ///
    /// The block covers exactly the image's cells, from cell zero.
    Public(PublicImage<F>),
    /// Cells of one region start from committed columns only the prover knows.
    ///
    /// The block covers exactly the region's cells.
    Private(PrivateRegion),
}

/// Seed and close block of one timestamped memory, one cell per row.
///
/// Columns: the cell address, its last access time, its final value, then any private initial value.
///
/// Row `i` holds cell `first + i` at address `F::GENERATOR^(first + i)`, so no two rows name the same cell.
///
/// The last time and final value are committed by the prover.
///
/// Balance forces them to match what the last access left.
///
/// A public or private seed pins the height to the cells it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimestampedBoundaryAir<C, F: Field> {
    /// Memory this block seeds and closes.
    memory: TimestampedMemory<C, F>,
    /// Where the initial values come from.
    seed: TimestampedSeed<F>,
}

impl<C: Field, F: ExtensionField<C>> TimestampedBoundaryAir<C, F> {
    /// Column of the cell address.
    pub const ADDRESS: usize = 0;
    /// Column of the last access time.
    pub const LAST: usize = 1;
    /// First column of the final value.
    pub const FINAL: usize = 2;

    /// Builds the block of one memory with one seed source.
    ///
    /// # Errors
    ///
    /// A public image whose words have a different width than the memory's values.
    pub fn new(
        memory: TimestampedMemory<C, F>,
        seed: TimestampedSeed<F>,
    ) -> Result<Self, TimestampedMemoryError> {
        if let TimestampedSeed::Public(image) = &seed
            && image.value_width() != memory.value_width
        {
            return Err(TimestampedMemoryError::ImageWidth {
                expected: memory.value_width,
                actual: image.value_width(),
            });
        }
        Ok(Self { memory, seed })
    }

    /// Memory this block seeds and closes.
    #[must_use]
    pub const fn memory(&self) -> &TimestampedMemory<C, F> {
        &self.memory
    }

    /// Where the initial values come from.
    #[must_use]
    pub const fn seed(&self) -> &TimestampedSeed<F> {
        &self.seed
    }

    /// Address of cell `cell`, which is `F::GENERATOR^cell`.
    #[must_use]
    pub fn cell_address(cell: usize) -> F {
        F::GENERATOR.exp_u64(cell as u64)
    }

    /// Index of the cell on row zero.
    #[must_use]
    pub const fn first_cell(&self) -> usize {
        match &self.seed {
            TimestampedSeed::Zero | TimestampedSeed::Public(_) => 0,
            TimestampedSeed::Private(region) => region.first_cell(),
        }
    }

    /// Index of the cell on the last row, when the seed pins the height.
    #[must_use]
    pub const fn last_cell(&self) -> Option<usize> {
        match &self.seed {
            TimestampedSeed::Zero => None,
            TimestampedSeed::Public(image) => Some((1 << image.log_cells()) - 1),
            TimestampedSeed::Private(region) => Some(region.last_cell()),
        }
    }

    /// First column of the committed initial value, for a private seed.
    #[must_use]
    pub const fn initial_column(&self) -> Option<usize> {
        match &self.seed {
            TimestampedSeed::Private(_) => Some(Self::FINAL + self.memory.value_width),
            _ => None,
        }
    }

    /// Initial value of the cell on the current row.
    fn initial<AB: AirBuilder<F = F>>(&self, builder: &AB) -> Vec<AB::Expr> {
        let width = self.memory.value_width;
        match &self.seed {
            TimestampedSeed::Zero => vec![AB::Expr::ZERO; width],
            TimestampedSeed::Public(_) => builder.periodic_values()[..width]
                .iter()
                .map(|&value| value.into())
                .collect(),
            TimestampedSeed::Private(_) => {
                let first = Self::FINAL + width;
                builder.main().current_slice()[first..first + width]
                    .iter()
                    .map(|&column| column.into())
                    .collect()
            }
        }
    }
}

impl<C: Field, F: ExtensionField<C>> BaseAir<F> for TimestampedBoundaryAir<C, F> {
    fn width(&self) -> usize {
        let private = match &self.seed {
            TimestampedSeed::Private(_) => self.memory.value_width,
            _ => 0,
        };
        Self::FINAL + self.memory.value_width + private
    }

    fn num_periodic_columns(&self) -> usize {
        match &self.seed {
            TimestampedSeed::Public(image) => image.value_width(),
            _ => 0,
        }
    }

    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
        match &self.seed {
            TimestampedSeed::Public(image) => Cow::Owned(image.columns()),
            _ => Cow::Borrowed(&[]),
        }
    }
}

impl<C, F, AB> Air<AB> for TimestampedBoundaryAir<C, F>
where
    C: Field,
    F: ExtensionField<C>,
    AB: BusInteractionBuilder<F = F>,
{
    fn eval(&self, builder: &mut AB) {
        let (address, next_address, last, final_value) = {
            let main = builder.main();
            let row = main.current_slice();
            let next = main.next_slice();
            (
                row[Self::ADDRESS],
                next[Self::ADDRESS],
                row[Self::LAST],
                row[Self::FINAL..Self::FINAL + self.memory.value_width]
                    .iter()
                    .map(|&column| column.into())
                    .collect::<Vec<AB::Expr>>(),
            )
        };

        // Cell addresses walk the generator orbit from the first cell, so they are distinct and fixed.
        builder
            .when_first_row()
            .assert_eq(address, Self::cell_address(self.first_cell()));
        builder
            .when_transition()
            .assert_eq(next_address, address.into() * F::GENERATOR);

        // A seeded block ends on its last cell, which pins its height.
        if let Some(last_cell) = self.last_cell() {
            builder
                .when_last_row()
                .assert_eq(address, Self::cell_address(last_cell));
        }

        let initial = self.initial(builder);
        builder.timestamped_boundary(
            &self.memory,
            address.into(),
            initial,
            last.into(),
            final_value,
        );
    }
}

/// Both fixed range tables of one timestamped memory, `2^16` rows.
///
/// Columns: the low factor, its final count, the high factor, its final count.
///
/// Row `j` holds `g^(j + 1)` and `g^(-2^16 * j)`.
///
/// Each entry is seeded at count one and closed at its final count, as read-only memory expects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockRangeAir<C, F: Field> {
    /// Memory whose gaps this table ranges.
    memory: TimestampedMemory<C, F>,
}

impl<C: Field, F: ExtensionField<C>> ClockRangeAir<C, F> {
    /// Column of the low factor.
    pub const LOW: usize = 0;
    /// Column of the low factor's final count.
    pub const LOW_FINAL: usize = 1;
    /// Column of the high factor.
    pub const HIGH: usize = 2;
    /// Column of the high factor's final count.
    pub const HIGH_FINAL: usize = 3;
    /// Number of rows, which the constraints pin.
    pub const HEIGHT: usize = 1 << CLOCK_RANGE_BITS;

    /// Builds the range tables of one memory.
    #[must_use]
    pub const fn new(memory: TimestampedMemory<C, F>) -> Self {
        Self { memory }
    }

    /// Low factor on row `row`, which is `g^(row + 1)`.
    #[must_use]
    pub fn low(row: usize) -> F {
        F::from(C::GENERATOR.exp_u64(row as u64 + 1))
    }

    /// High factor on row `row`, which is `g^(-2^16 * row)`.
    #[must_use]
    pub fn high(row: usize) -> F {
        F::from(high_step::<C>().exp_u64(row as u64))
    }
}

impl<C: Field, F: ExtensionField<C>, F2> BaseAir<F2> for ClockRangeAir<C, F> {
    fn width(&self) -> usize {
        4
    }
}

impl<C, F, AB> Air<AB> for ClockRangeAir<C, F>
where
    C: Field,
    F: ExtensionField<C>,
    AB: BusInteractionBuilder<F = F>,
{
    fn eval(&self, builder: &mut AB) {
        let (row, next) = {
            let main = builder.main();
            let row: [AB::Var; 4] = core::array::from_fn(|column| main.current_slice()[column]);
            let next: [AB::Var; 4] = core::array::from_fn(|column| main.next_slice()[column]);
            (row, next)
        };
        let tick = F::from(C::GENERATOR);

        // Row zero holds `g^1` and `g^0`.
        builder.when_first_row().assert_eq(row[Self::LOW], tick);
        builder.when_first_row().assert_one(row[Self::HIGH]);

        // Each row steps both factors by a fixed power of `g`.
        let mut transition = builder.when_transition();
        transition.assert_eq(next[Self::LOW], row[Self::LOW].into() * tick);
        transition.assert_eq(
            next[Self::HIGH],
            row[Self::HIGH].into() * F::from(high_step::<C>()),
        );

        // The last row holds `g^(2^16)`, which pins the height to exactly `2^16`.
        builder.when_last_row().assert_eq(
            row[Self::LOW],
            F::from(C::GENERATOR.exp_power_of_2(CLOCK_RANGE_BITS)),
        );

        // Each entry enters its read-only table at count one and leaves at its final count.
        for (bus, value, final_count) in [
            (&self.memory.low, Self::LOW, Self::LOW_FINAL),
            (&self.memory.high, Self::HIGH, Self::HIGH_FINAL),
        ] {
            builder.push_bus_interaction(
                bus.name(),
                BusDirection::Push,
                [row[value].into(), AB::Expr::ONE],
                BusActivation::Always,
            );
            builder.push_bus_interaction(
                bus.name(),
                BusDirection::Pull,
                [row[value], row[final_count]],
                BusActivation::Always,
            );
        }
    }
}
