//! Bit-sliced `GF(4)` AIR arithmetic, sixty-four trace rows at a time.
//!
//! A binary trace often holds only bits, and the interpolation nodes of a degree-3 zerocheck
//! round are the four elements of `GF(4)`. Every value the AIR computes there needs two bits,
//! so one pair of words carries a value for each of sixty-four rows:
//!
//! ```text
//!     lane i   =  (bit i of low)  +  (bit i of high) * g        g the generator, g^2 = g + 1
//!     a + b    =  (a.low ^ b.low, a.high ^ b.high)
//!     a * b    =  three ANDs and four XORs, every lane at once
//! ```
//!
//! The AIR then runs once per sixty-four rows. Each constraint leaves the lanes through a
//! lane-weighted sum, looked up a byte at a time, and meets its batching power there:
//!
//! ```text
//!     sum_lane w(lane) * sum_i alpha^(n-1-i) * C_i(lane)
//!         =  sum_i alpha^(n-1-i) * (sum over the set lanes of C_i's planes of w)
//! ```
//!
//! As with [`crate::subfield::SubfieldVar`], a trace-field value outside `GF(4)` poisons every
//! result it reaches, and a poisoned evaluation must be discarded.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_air::{Air, AirBuilder, RowWindow};
use p3_field::{Algebra, Field, HasSubfield, PrimeCharacteristicRing};
use p3_lookup::{Count, IndexedLookupBuilder, InteractionBuilder, TraceWindow};

use crate::folder::eval_boundary_io;
use crate::selectors::BoundaryEvals;

/// Rows one sliced value carries, one per bit of a word.
pub const SLICED_LANES: usize = u64::BITS as usize;

/// Bits of a lane mask that index one partial-sum table.
const TABLE_BITS: usize = u8::BITS as usize;

/// Entries of one partial-sum table.
const TABLE_ENTRIES: usize = 1 << TABLE_BITS;

/// Partial-sum tables covering the lanes of one plane.
const TABLES_PER_PLANE: usize = SLICED_LANES / TABLE_BITS;

/// Whether `S` is `GF(4)` and its generator obeys `g^2 = g + 1`, the rule the sliced product uses.
///
/// Every generator of a four-element field does, since its minimal polynomial is `x^2 + x + 1`.
#[must_use]
pub(crate) fn is_gf4<S: Field>() -> bool {
    S::order().to_u64_digits() == [4] && S::GENERATOR.square() == S::GENERATOR + S::ONE
}

/// The coordinates of `s` on the basis `{1, g}`, where `g` is the generator of `S`.
///
/// # Returns
///
/// `None` when `s` is none of `0, 1, g, g + 1`, which cannot happen once [`is_gf4`] holds.
#[inline]
#[must_use]
pub(crate) fn gf4_coordinates<S: Field>(s: S) -> Option<(bool, bool)> {
    let generator = S::GENERATOR;
    if s == S::ZERO {
        Some((false, false))
    } else if s == S::ONE {
        Some((true, false))
    } else if s == generator {
        Some((false, true))
    } else if s == generator + S::ONE {
        Some((true, true))
    } else {
        None
    }
}

/// A word with every bit equal to `bit`.
#[inline]
const fn broadcast_bit(bit: bool) -> u64 {
    (bit as u64).wrapping_neg()
}

/// Sixty-four AIR expression values in `GF(4) = S`, one per lane.
///
/// ```text
///     F value x   ->  narrow   ->  the same element in every lane,  or poisoned
///     a op b      ->  GF(4) op ->  poisoned when either operand is
/// ```
///
/// While unpoisoned, lane `i` lifted into `F` is what the same expression computes over `F` on
/// the lane's row.
pub struct SlicedGf4<F, S> {
    /// The coordinate on `1` of every lane.
    low: u64,
    /// The coordinate on the generator of every lane.
    high: u64,
    /// Whether some `F` value outside `S` reached this result.
    poisoned: bool,
    /// The trace field the AIR believes it computes over, and the subfield it runs in.
    _fields: PhantomData<fn() -> (F, S)>,
}

impl<F, S> SlicedGf4<F, S> {
    /// The value with the given coordinate planes, reached by no out-of-subfield input.
    #[inline]
    #[must_use]
    pub const fn from_planes(low: u64, high: u64) -> Self {
        Self {
            low,
            high,
            poisoned: false,
            _fields: PhantomData,
        }
    }

    /// The same element of `S`, given by its coordinates, in every lane.
    #[inline]
    #[must_use]
    pub const fn broadcast(low: bool, high: bool) -> Self {
        Self::from_planes(broadcast_bit(low), broadcast_bit(high))
    }

    /// A value an out-of-subfield input reached; its planes carry no meaning.
    const POISONED: Self = Self {
        low: 0,
        high: 0,
        poisoned: true,
        _fields: PhantomData,
    };

    /// Combine two operands' planes into one value, poisoned when either operand is.
    #[inline]
    const fn with_flags(low: u64, high: u64, lhs: bool, rhs: bool) -> Self {
        Self {
            low,
            high,
            poisoned: lhs | rhs,
            _fields: PhantomData,
        }
    }

    /// The coordinate planes `(low, high)`.
    ///
    /// They are the intended values only while [`Self::is_poisoned`] is false.
    #[inline]
    #[must_use]
    pub const fn planes(self) -> (u64, u64) {
        (self.low, self.high)
    }

    /// Whether an `F` value outside the subfield has reached this value.
    #[inline]
    #[must_use]
    pub const fn is_poisoned(self) -> bool {
        self.poisoned
    }

    /// Multiply every lane by the element of `S` with coordinates `(low, high)`.
    ///
    /// ```text
    ///     (c0 + c1 g)(a0 + a1 g) = (c0 a0 + c1 a1) + (c0 a1 + c1 a0 + c1 a1) g
    /// ```
    #[inline]
    #[must_use]
    pub(crate) const fn scale(self, low: bool, high: bool) -> Self {
        let (c0, c1) = (broadcast_bit(low), broadcast_bit(high));
        Self::with_flags(
            (c0 & self.low) ^ (c1 & self.high),
            (c0 & self.high) ^ (c1 & (self.low ^ self.high)),
            self.poisoned,
            false,
        )
    }
}

impl<F, S: Field> SlicedGf4<F, S> {
    /// Lane `lane` as an element of `S`.
    ///
    /// # Panics
    ///
    /// Panics if `lane` is not below [`SLICED_LANES`].
    #[must_use]
    pub fn lane(self, lane: usize) -> S {
        assert!(lane < SLICED_LANES, "lane out of range");
        let low = S::from_bool((self.low >> lane) & 1 == 1);
        let high = S::from_bool((self.high >> lane) & 1 == 1);
        low + high * S::GENERATOR
    }
}

impl<F: HasSubfield<S>, S: Field> SlicedGf4<F, S> {
    /// Narrow a trace-field value into every lane, poisoning it when it lies outside `S`.
    ///
    /// Every `F` value that enters this type goes through here.
    #[inline]
    #[must_use]
    pub fn narrow(x: F) -> Self {
        x.as_subfield()
            .and_then(gf4_coordinates)
            .map_or(Self::POISONED, |(low, high)| Self::broadcast(low, high))
    }
}

impl<F, S> Clone for SlicedGf4<F, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, S> Copy for SlicedGf4<F, S> {}

impl<F, S> fmt::Debug for SlicedGf4<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicedGf4")
            .field("low", &format_args!("{:#018x}", self.low))
            .field("high", &format_args!("{:#018x}", self.high))
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl<F, S> Default for SlicedGf4<F, S> {
    #[inline]
    fn default() -> Self {
        Self::from_planes(0, 0)
    }
}

impl<F, S> Add for SlicedGf4<F, S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::with_flags(
            self.low ^ rhs.low,
            self.high ^ rhs.high,
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F, S> AddAssign for SlicedGf4<F, S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F, S> Sub for SlicedGf4<F, S> {
    type Output = Self;

    /// Characteristic two: subtracting is adding.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::with_flags(
            self.low ^ rhs.low,
            self.high ^ rhs.high,
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F, S> SubAssign for SlicedGf4<F, S> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, S> Neg for SlicedGf4<F, S> {
    type Output = Self;

    /// Characteristic two: every value is its own negation.
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl<F, S> Mul for SlicedGf4<F, S> {
    type Output = Self;

    /// ```text
    ///     (a0 + a1 g)(b0 + b1 g) = (a0 b0 + a1 b1) + ((a0 + a1)(b0 + b1) + a0 b0) g
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let low_product = self.low & rhs.low;
        Self::with_flags(
            low_product ^ (self.high & rhs.high),
            ((self.low ^ self.high) & (rhs.low ^ rhs.high)) ^ low_product,
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F, S> MulAssign for SlicedGf4<F, S> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, S: Field> Sum for SlicedGf4<F, S> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F, S: Field> Product for SlicedGf4<F, S> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F, S: Field> PrimeCharacteristicRing for SlicedGf4<F, S> {
    // The prime subfield embeds the same way into every field of its characteristic.
    type PrimeSubfield = S::PrimeSubfield;

    const ZERO: Self = Self::from_planes(0, 0);
    const ONE: Self = Self::broadcast(true, false);
    // A four-element field has characteristic two.
    const TWO: Self = Self::ZERO;
    const NEG_ONE: Self = Self::ONE;

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        gf4_coordinates(S::from_prime_subfield(f))
            .map_or(Self::POISONED, |(low, high)| Self::broadcast(low, high))
    }

    #[inline]
    fn double(&self) -> Self {
        Self::with_flags(0, 0, self.poisoned, false)
    }

    /// ```text
    ///     (a0 + a1 g)^2 = a0 + a1 (g + 1) = (a0 + a1) + a1 g
    /// ```
    #[inline]
    fn square(&self) -> Self {
        Self::with_flags(self.low ^ self.high, self.high, self.poisoned, false)
    }

    /// ```text
    ///     x^2 - x = x^2 + x = a1
    /// ```
    #[inline]
    fn bool_check(&self) -> Self {
        Self::with_flags(self.high, 0, self.poisoned, false)
    }
}

impl<F: HasSubfield<S>, S: Field> From<F> for SlicedGf4<F, S> {
    #[inline]
    fn from(x: F) -> Self {
        Self::narrow(x)
    }
}

impl<F: HasSubfield<S>, S: Field> Add<F> for SlicedGf4<F, S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        self + Self::narrow(rhs)
    }
}

impl<F: HasSubfield<S>, S: Field> AddAssign<F> for SlicedGf4<F, S> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        *self += Self::narrow(rhs);
    }
}

impl<F: HasSubfield<S>, S: Field> Sub<F> for SlicedGf4<F, S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        self - Self::narrow(rhs)
    }
}

impl<F: HasSubfield<S>, S: Field> SubAssign<F> for SlicedGf4<F, S> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::narrow(rhs);
    }
}

impl<F: HasSubfield<S>, S: Field> Mul<F> for SlicedGf4<F, S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        self * Self::narrow(rhs)
    }
}

impl<F: HasSubfield<S>, S: Field> MulAssign<F> for SlicedGf4<F, S> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::narrow(rhs);
    }
}

impl<F: HasSubfield<S>, S: Field> Algebra<F> for SlicedGf4<F, S> {}

/// Byte-indexed partial sums of sixty-four lane weights.
///
/// The weighted sum of a sliced value's lanes takes one lookup per byte of each plane:
///
/// ```text
///     sum_lane w(lane) * value(lane)
///         = sum_byte low_table[byte][low byte]  +  sum_byte high_table[byte][high byte]
///
///     low_table[byte][m]  = sum of w(8 byte + i) over the set bits i of m
///     high_table[byte][m] = g * low_table[byte][m]
/// ```
#[derive(Clone, Debug)]
pub struct LaneSums<R> {
    /// The low-plane tables, one per byte of a plane, then the high-plane tables.
    tables: Box<[[R; TABLE_ENTRIES]; 2 * TABLES_PER_PLANE]>,
}

impl<R: Field> LaneSums<R> {
    /// Tabulate the partial sums of `weights`, with `generator` the image of `g`.
    ///
    /// # Panics
    ///
    /// Panics if `weights` does not hold one weight per lane.
    #[must_use]
    pub fn new(weights: &[R], generator: R) -> Self {
        assert_eq!(weights.len(), SLICED_LANES, "one weight per lane");
        let mut tables = vec![[R::ZERO; TABLE_ENTRIES]; 2 * TABLES_PER_PLANE];
        let (low, high) = tables.split_at_mut(TABLES_PER_PLANE);
        for (table, weights) in low.iter_mut().zip(weights.as_chunks::<TABLE_BITS>().0) {
            // Each mask adds its lowest set lane to the mask without it.
            for mask in 1..TABLE_ENTRIES {
                table[mask] = table[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
            }
        }
        for (high, low) in high.iter_mut().zip(low.iter()) {
            for (high, &low) in high.iter_mut().zip(low) {
                *high = generator * low;
            }
        }
        let tables = tables
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("the table count is fixed"));
        Self { tables }
    }

    /// The lane-weighted sum of the value with coordinate planes `low` and `high`.
    #[inline]
    #[must_use]
    pub fn sum(&self, low: u64, high: u64) -> R {
        let mut sum = R::ZERO;
        for byte in 0..TABLES_PER_PLANE {
            let shift = byte * TABLE_BITS;
            sum += self.tables[byte][usize::from((low >> shift) as u8)]
                + self.tables[TABLES_PER_PLANE + byte][usize::from((high >> shift) as u8)];
        }
        sum
    }
}

/// One AIR evaluation over sixty-four rows, lane-weighted and alpha-batched.
#[derive(Clone, Copy, Debug)]
pub struct SlicedEvaluation<R> {
    /// `sum_i alpha^(n-1-i) * sum_lane w(lane) * C_i(lane)`.
    pub value: R,
    /// Whether any constraint value was poisoned, which makes `value` meaningless.
    pub poisoned: bool,
}

/// AIR folder over sixty-four rows of bit-sliced `GF(4)` values.
///
/// Every asserted constraint is summed across the lanes with the weights of a [`LaneSums`], then
/// weighted by its descending alpha power. Lookup declarations are dropped, as in
/// [`crate::folder::MultilinearFolder`]: a stage that declares lookups never runs here.
#[derive(Debug)]
pub struct SlicedFolder<'a, F, S, R> {
    /// Two-row main window holding the current and shifted-by-one rows.
    main_window: RowWindow<'a, SlicedGf4<F, S>>,
    /// Two-row preprocessed window; zero-width when the AIR has no preprocessed columns.
    preprocessed_window: RowWindow<'a, SlicedGf4<F, S>>,
    /// Periodic column values, one per declared periodic column.
    periodic_values: &'a [SlicedGf4<F, S>],
    /// Boundary-selector values shared by all selector accessors.
    boundary: BoundaryEvals<SlicedGf4<F, S>>,
    /// Public inputs forwarded to the AIR, always in the trace field.
    public_values: &'a [F],
    /// Descending alpha powers, one per asserted constraint, boundary pins included.
    alpha_powers: &'a [R],
    /// The lane weights every constraint is summed with.
    lanes: &'a LaneSums<R>,
    /// Running lane-weighted, alpha-batched sum.
    accumulator: R,
    /// Number of constraints asserted so far, which is the next position in `alpha_powers`.
    constraint_index: usize,
    /// Whether any asserted value was poisoned.
    poisoned: bool,
}

impl<'a, F, S, R: Field> SlicedFolder<'a, F, S, R> {
    /// Build a folder for one evaluation over sixty-four rows.
    ///
    /// # Arguments
    ///
    /// - `local`, `next`: main column values at the current and shifted-by-one rows.
    /// - `boundary`: selector values at the same rows.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `alpha_powers`: `alpha^(n - 1 - i)` for each of the `n` constraints, pins included.
    /// - `lanes`: the weights summing each constraint across the lanes.
    #[inline]
    #[must_use]
    pub fn new(
        local: &'a [SlicedGf4<F, S>],
        next: &'a [SlicedGf4<F, S>],
        boundary: BoundaryEvals<SlicedGf4<F, S>>,
        public_values: &'a [F],
        alpha_powers: &'a [R],
        lanes: &'a LaneSums<R>,
    ) -> Self {
        Self {
            main_window: RowWindow::from_two_rows(local, next),
            preprocessed_window: RowWindow::from_two_rows(&[], &[]),
            periodic_values: &[],
            boundary,
            public_values,
            alpha_powers,
            lanes,
            accumulator: R::ZERO,
            constraint_index: 0,
            poisoned: false,
        }
    }

    /// Attach the two-row preprocessed window read by the AIR.
    #[inline]
    #[must_use]
    pub fn with_preprocessed(
        mut self,
        current: &'a [SlicedGf4<F, S>],
        next: &'a [SlicedGf4<F, S>],
    ) -> Self {
        self.preprocessed_window = RowWindow::from_two_rows(current, next);
        self
    }

    /// Attach the periodic column values read by the AIR.
    #[inline]
    #[must_use]
    pub const fn with_periodic(mut self, values: &'a [SlicedGf4<F, S>]) -> Self {
        self.periodic_values = values;
        self
    }

    /// Run the AIR, then its public boundary pins, and return the batched sum.
    ///
    /// # Panics
    ///
    /// Panics if the alpha powers do not number one per asserted constraint.
    #[inline]
    #[must_use]
    pub fn eval_air<A>(mut self, air: &A) -> SlicedEvaluation<R>
    where
        A: Air<Self>,
        Self: AirBuilder,
    {
        air.eval(&mut self);
        eval_boundary_io(&mut self, air.public_boundary_io());
        assert_eq!(
            self.constraint_index,
            self.alpha_powers.len(),
            "attached alpha powers must match the number of asserted constraints"
        );
        SlicedEvaluation {
            value: self.accumulator,
            poisoned: self.poisoned,
        }
    }
}

impl<'a, F, S, R> AirBuilder for SlicedFolder<'a, F, S, R>
where
    F: HasSubfield<S>,
    S: Field,
    R: Field,
{
    type F = F;
    type Expr = SlicedGf4<F, S>;
    type Var = SlicedGf4<F, S>;
    type MainWindow = RowWindow<'a, SlicedGf4<F, S>>;
    type PreprocessedWindow = RowWindow<'a, SlicedGf4<F, S>>;
    // Public values stay in the trace field and narrow into the lanes on read.
    type PublicVar = F;
    type PeriodicVar = SlicedGf4<F, S>;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        self.main_window
    }

    #[inline]
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.boundary.first
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.boundary.last
    }

    #[inline]
    fn is_transition(&self) -> Self::Expr {
        self.boundary.transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.poisoned |= x.poisoned;
        // A constraint past the last power is only counted; the count check rejects it.
        if let Some(&power) = self.alpha_powers.get(self.constraint_index) {
            self.accumulator += power * self.lanes.sum(x.low, x.high);
        }
        self.constraint_index += 1;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}

/// Lookup declarations are dropped: a stage that declares lookups never runs this folder.
impl<'a, F, S, R> InteractionBuilder for SlicedFolder<'a, F, S, R>
where
    F: HasSubfield<S>,
    S: Field,
    R: Field,
{
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        _fields: impl IntoIterator<Item = E>,
        _count: impl Into<Count<Self::Expr>>,
    ) {
    }

    fn push_local_interaction(
        &mut self,
        _tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
    }
}

/// Indexed reads are dropped, as every other lookup declaration is.
impl<'a, F, S, R> IndexedLookupBuilder for SlicedFolder<'a, F, S, R>
where
    F: HasSubfield<S>,
    S: Field,
    R: Field,
{
    fn push_indexed_read(
        &mut self,
        _table: &str,
        _position: usize,
        payload: impl IntoIterator<Item = usize>,
    ) {
        payload.into_iter().for_each(drop);
    }

    fn push_indexed_table(
        &mut self,
        _name: &str,
        _window: TraceWindow,
        columns: impl IntoIterator<Item = usize>,
    ) {
        columns.into_iter().for_each(drop);
    }

    fn num_indexed_reads(&self) -> usize {
        0
    }

    fn num_indexed_tables(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests;
