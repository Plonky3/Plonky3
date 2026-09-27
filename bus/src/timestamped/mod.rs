//! Read-write memory for tables whose rows come in no particular order.
//!
//! The clock is a power of a fixed generator `g` of a clock field `C`.
//!
//! Advancing it is a multiplication by a constant, which is linear in characteristic two.
//!
//! # Tuples
//!
//! Each access to cell `a` at time `now` turns value `old` into value `new`:
//!
//! ```text
//!     pull (a, prev, old)        what the last access left
//!     push (a, now,  new)        what this access leaves
//! ```
//!
//! A boundary block seeds every cell once and closes it once:
//!
//! ```text
//!     seed    push (a, g^0,  initial)
//!     close   pull (a, last, final)
//! ```
//!
//! # Seeds
//!
//! The seed's initial value comes from one of three sources.
//!
//! - Zero: every cell starts at zero.
//! - Public: a sparse image the verifier knows, read as periodic columns and never committed.
//! - Private: committed columns for one region, opened like any other column.
//!
//! The verifier evaluates a public image with [`PublicImage::evaluate`].
//!
//! Its cost is one term per image word, not one per cell.
//!
//! # Slots
//!
//! A row at clock `t` may make several accesses.
//!
//! Slot `k` of that row accesses at time `now = g^k * t`.
//!
//! A table with `K` slots advances its clock by at least `g^K` per row, so no two accesses share a time.
//!
//! # Strict gap
//!
//! An access with `prev = now` would pull exactly what it pushes and could claim any value.
//!
//! Every access therefore proves `now = prev * g^d` with `1 <= d <= 2^32`:
//!
//! ```text
//!     prev * lo = now * hi
//!     lo  in { g^(j + 1)       : j < 2^16 }
//!     hi  in { g^(-2^16 * j)   : j < 2^16 }
//! ```
//!
//! Both tables are fixed and read through the read-only memory lookup.
//!
//! # Assumptions
//!
//! - Clock `0` marks a padding row, and every other clock is a power of `g`.
//! - Real clocks start at `g^1` or later, since the seed sits at `g^0`.
//! - Two consecutive accesses to one cell are at most `2^32` ticks apart.
//! - The whole run, and every forged cycle, stays below the order of `g`.
//!
//! # Soundness
//!
//! Padding has `now = 0`, so the gap check forces `prev = 0` there.
//!
//! Its tuples sit at time zero and can only cancel other padding tuples.
//!
//! On a real cell, balance chains the seed, the accesses and the close into one path.
//!
//! Along that path every step raises the clock exponent by `1` to `2^32`.
//!
//! A leftover cycle would need its gaps to sum to a multiple of the order of `g`.
//!
//! [`TimestampedMemory::check_against`] bounds the accesses so that no cycle can reach it.
//!
//! The path is then the time order, and each read sees the last write.

mod air;
mod error;
mod image;

#[cfg(test)]
mod tests;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::num::NonZeroUsize;

pub use air::{ClockRangeAir, TimestampedBoundaryAir, TimestampedSeed};
pub use error::TimestampedMemoryError;
pub use image::{PrivateRegion, PublicImage};
use num_bigint::BigUint;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_security::SecurityTerm;

use crate::{
    BusActivation, BusDirection, BusInteractionBuilder, BusName, BusPlan, ReadOnlyMemoryBus,
    ReadOnlyMemoryInteractionBuilder,
};

/// Bits of each gap digit, so each range table has `2^16` entries.
pub const CLOCK_RANGE_BITS: usize = 16;

/// Bits of the largest gap one access may prove, which is `2^32`.
pub const CLOCK_GAP_BITS: usize = 2 * CLOCK_RANGE_BITS;

/// Channels and clock of one timestamped read-write memory.
///
/// `C` is the clock field, whose generator ticks the clock.
///
/// `F` is the bus field, which contains `C`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimestampedMemory<C, F: Field> {
    /// Channel carrying the memory tuples.
    memory: String,
    /// Read-only table of low gap factors `g^(j + 1)`.
    low: ReadOnlyMemoryBus<F>,
    /// Read-only table of high gap factors `g^(-2^16 * j)`.
    high: ReadOnlyMemoryBus<F>,
    /// Number of field components in one stored value.
    value_width: usize,
    /// Bind the checked clock orbit to this handle.
    marker: PhantomData<fn() -> C>,
}

impl<C: Field, F: ExtensionField<C>> TimestampedMemory<C, F> {
    /// Names the memory channel and its two range tables.
    ///
    /// # Errors
    ///
    /// - A malformed name, or one name used for two roles.
    /// - An empty value.
    /// - A clock orbit too short to keep a gap of up to `2^32` nonzero.
    /// - A bus field too small for the read-only range lookups.
    pub fn new(
        memory: &str,
        low: &str,
        high: &str,
        value_width: usize,
    ) -> Result<Self, TimestampedMemoryError> {
        let names = [memory, low, high];
        for (position, name) in names.iter().enumerate() {
            BusName::try_new(name).map_err(|error| TimestampedMemoryError::BusName {
                name: (*name).to_string(),
                error,
            })?;

            // Two roles on one channel would let a tuple of one cancel a tuple of the other.
            if names[..position].contains(name) {
                return Err(TimestampedMemoryError::DuplicateBus {
                    name: (*name).to_string(),
                });
            }
        }
        if value_width == 0 {
            return Err(TimestampedMemoryError::EmptyValue);
        }

        // A gap `d <= 2^32` must never be a multiple of the order of `g`.
        let orbit = clock_orbit::<C>();
        if orbit <= BigUint::from(1u64) << CLOCK_GAP_BITS {
            return Err(TimestampedMemoryError::ClockOrbitTooShort {
                orbit_bits: orbit.bits() as usize,
                minimum_bits: CLOCK_GAP_BITS + 1,
            });
        }

        Ok(Self {
            memory: memory.to_string(),
            low: ReadOnlyMemoryBus::new(low)?,
            high: ReadOnlyMemoryBus::new(high)?,
            value_width,
            marker: PhantomData,
        })
    }

    /// Channel carrying the memory tuples.
    #[must_use]
    pub fn memory_bus(&self) -> BusName<'_> {
        BusName::new(&self.memory)
    }

    /// Read-only table of low gap factors.
    #[must_use]
    pub const fn low_bus(&self) -> &ReadOnlyMemoryBus<F> {
        &self.low
    }

    /// Read-only table of high gap factors.
    #[must_use]
    pub const fn high_bus(&self) -> &ReadOnlyMemoryBus<F> {
        &self.high
    }

    /// Number of field components in one stored value.
    #[must_use]
    pub const fn value_width(&self) -> usize {
        self.value_width
    }

    /// The clock tick `g`, embedded in the bus field.
    #[must_use]
    pub fn tick() -> F {
        F::from(C::GENERATOR)
    }

    /// Time of slot `slot` on a row at clock `g^0`, which is `g^slot`.
    #[must_use]
    pub fn slot_offset(slot: usize) -> F {
        F::from(C::GENERATOR.exp_u64(slot as u64))
    }

    /// The two range factors `(lo, hi)` proving a gap of `gap` ticks.
    ///
    /// Returns `None` for a gap outside `1..=2^32`.
    #[must_use]
    pub fn gap_factors(gap: u64) -> Option<(F, F)> {
        if gap == 0 || gap > 1 << CLOCK_GAP_BITS {
            return None;
        }

        // Write `d = (j + 1) + 2^16 * m` with both digits below `2^16`.
        let low_digit = (gap - 1) & ((1 << CLOCK_RANGE_BITS) - 1);
        let high_digit = (gap - 1) >> CLOCK_RANGE_BITS;
        let low = C::GENERATOR.exp_u64(low_digit + 1);
        let high = high_step::<C>().exp_u64(high_digit);
        Some((F::from(low), F::from(high)))
    }

    /// Checks this memory against the channels a plan defines.
    ///
    /// # Errors
    ///
    /// - A channel the plan does not define, or one of the wrong payload width.
    /// - So many memory tuples that a forged cycle could wrap the clock orbit.
    pub fn check_against(&self, bus_plan: &BusPlan) -> Result<(), TimestampedMemoryError> {
        // A memory tuple is a cell, a time, then the value.
        let memory = check_payload_width(bus_plan, &self.memory, 2 + self.value_width)?;

        // A range read is the factor itself as address, then its read count.
        check_payload_width(bus_plan, self.low.name().as_str(), 2)?;
        check_payload_width(bus_plan, self.high.name().as_str(), 2)?;

        // Every access and every close pulls once, so pulls bound the accesses.
        let pulls = bus_plan
            .blocks(BusDirection::Pull)
            .iter()
            .filter(|block| block.bus == memory)
            .try_fold(0usize, |total, block| {
                total.checked_add(1usize << block.log_height)
            })
            .ok_or(TimestampedMemoryError::TupleCountOverflow)?;

        // A cycle of `n` accesses climbs at most `n * 2^32` ticks, which must stay below the orbit.
        if BigUint::from(pulls) << CLOCK_GAP_BITS >= clock_orbit::<C>() {
            return Err(TimestampedMemoryError::TooManyAccesses {
                pulls,
                gap_bits: CLOCK_GAP_BITS,
            });
        }
        Ok(())
    }

    /// Soundness of this memory's claims, at the field the transcript samples from.
    ///
    /// The range lookups and memory tuples are declarations on the plan.
    ///
    /// The plan's own term therefore charges their fingerprints and product reduction.
    ///
    /// Everything else here is deterministic.
    ///
    /// # Errors
    ///
    /// - A plan this memory does not fit.
    /// - A challenge field with no room for a challenge.
    pub fn security_term<EF: Field>(
        &self,
        bus_plan: &BusPlan,
    ) -> Result<SecurityTerm, TimestampedMemoryError> {
        self.check_against(bus_plan)?;

        // The reported bit count rounds up, so one bit comes off to floor it.
        let field_bits = NonZeroUsize::new(EF::order().bits().saturating_sub(1) as usize)
            .ok_or(TimestampedMemoryError::TrivialChallengeField)?;
        Ok(bus_plan.security_term(field_bits))
    }
}

/// One read of a fixed range table.
#[derive(Clone, Debug)]
pub struct RangeRead<E> {
    /// Factor read from the table, which is also its address.
    pub value: E,
    /// Read-only count of that entry before this read.
    pub count: E,
    /// Inverse of that count, which rules out a zero count.
    pub count_inverse: E,
}

/// Witness of one strict clock gap `prev * lo = now * hi`.
#[derive(Clone, Debug)]
pub struct ClockGap<E> {
    /// Read of the low factor `g^(j + 1)`.
    pub low: RangeRead<E>,
    /// Read of the high factor `g^(-2^16 * m)`.
    pub high: RangeRead<E>,
}

/// One access to one cell, from one slot of one row.
#[derive(Clone, Debug)]
pub struct TimestampedAccess<E> {
    /// Cell touched by this access.
    pub address: E,
    /// Time of the last access to that cell.
    pub previous: E,
    /// Value the last access left.
    pub old: Vec<E>,
    /// Value this access leaves.
    pub new: Vec<E>,
    /// Proof that `previous` is strictly earlier than this access.
    pub gap: ClockGap<E>,
}

/// AIR interface for timestamped memory declarations.
pub trait TimestampedMemoryInteractionBuilder: BusInteractionBuilder
where
    Self::F: Field,
{
    /// Declares one access from slot `slot` of a row at clock `clock`.
    ///
    /// The access happens at time `g^slot * clock`.
    ///
    /// A read passes `new` equal to `old`.
    ///
    /// # Panics
    ///
    /// Panics when `old` or `new` does not hold `value_width` components.
    fn timestamped_access<C>(
        &mut self,
        memory: &TimestampedMemory<C, Self::F>,
        clock: Self::Expr,
        slot: usize,
        access: TimestampedAccess<Self::Expr>,
    ) where
        C: Field,
        Self::F: ExtensionField<C>,
    {
        assert_eq!(access.old.len(), memory.value_width, "old value width");
        assert_eq!(access.new.len(), memory.value_width, "new value width");
        let TimestampedAccess {
            address,
            previous,
            old,
            new,
            gap: ClockGap { low, high },
        } = access;
        let now = clock * TimestampedMemory::<C, Self::F>::slot_offset(slot);

        // Strict order: `now = prev * g^d` with `1 <= d <= 2^32`.
        self.assert_eq(
            previous.clone() * low.value.clone(),
            now.clone() * high.value.clone(),
        );
        for (bus, read) in [(&memory.low, low), (&memory.high, high)] {
            self.read_only_memory(bus, read.value, read.count, read.count_inverse, []);
        }

        let pull = [address.clone(), previous].into_iter().chain(old);
        self.push_bus_interaction(
            memory.memory_bus(),
            BusDirection::Pull,
            pull,
            BusActivation::Always,
        );
        let push = [address, now].into_iter().chain(new);
        self.push_bus_interaction(
            memory.memory_bus(),
            BusDirection::Push,
            push,
            BusActivation::Always,
        );
    }

    /// Seeds one cell at time `g^0` and closes it at time `last`.
    ///
    /// This is the hook every seed source goes through.
    ///
    /// The caller owes distinct cell addresses across all boundary rows.
    ///
    /// # Panics
    ///
    /// Panics when `initial` or `final_value` does not hold `value_width` components.
    fn timestamped_boundary<C>(
        &mut self,
        memory: &TimestampedMemory<C, Self::F>,
        address: Self::Expr,
        initial: Vec<Self::Expr>,
        last: Self::Expr,
        final_value: Vec<Self::Expr>,
    ) where
        C: Field,
        Self::F: ExtensionField<C>,
    {
        assert_eq!(initial.len(), memory.value_width, "initial value width");
        assert_eq!(final_value.len(), memory.value_width, "final value width");

        let seed = [address.clone(), Self::Expr::ONE]
            .into_iter()
            .chain(initial);
        self.push_bus_interaction(
            memory.memory_bus(),
            BusDirection::Push,
            seed,
            BusActivation::Always,
        );
        let close = [address, last].into_iter().chain(final_value);
        self.push_bus_interaction(
            memory.memory_bus(),
            BusDirection::Pull,
            close,
            BusActivation::Always,
        );
    }
}

impl<T> TimestampedMemoryInteractionBuilder for T
where
    T: BusInteractionBuilder,
    T::F: Field,
{
}

/// Order of the clock tick `g`, which is `|C| - 1`.
fn clock_orbit<C: Field>() -> BigUint {
    C::order() - BigUint::from(1u8)
}

/// Step between two high factors, which is `g^(-2^16)`.
fn high_step<C: Field>() -> C {
    C::GENERATOR.exp_power_of_2(CLOCK_RANGE_BITS).inverse()
}

/// Checks that one named channel exists with the expected payload width, and returns its index.
fn check_payload_width(
    bus_plan: &BusPlan,
    name: &str,
    expected: usize,
) -> Result<usize, TimestampedMemoryError> {
    let index = bus_plan
        .domains()
        .iter()
        .position(|domain| domain.name == name)
        .ok_or_else(|| TimestampedMemoryError::UnknownBus {
            name: name.to_string(),
        })?;
    let actual = bus_plan.domains()[index].payload_width;
    if actual != expected {
        return Err(TimestampedMemoryError::PayloadWidth {
            name: name.to_string(),
            expected,
            actual,
        });
    }
    Ok(index)
}
