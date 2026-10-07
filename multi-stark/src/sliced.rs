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
//! Where the target has a kernel for it, the powers are summed per lane first instead, eight
//! constraints at a time, and each lane meets its weight once; see `PreparedPowers`.
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
use p3_bus::{BusActivation, BusDirection, BusInteractionRecorder, BusName, RecordToken};
use p3_field::{Algebra, Field, HasSubfield, PrimeCharacteristicRing};
use p3_lookup::{Count, IndexedLookupBuilder, InteractionBuilder, TraceWindow};

use crate::folder::eval_boundary_io;
use crate::selectors::BoundaryEvals;

mod quadratic;

pub(crate) use quadratic::SLICED_CELLS;
pub use quadratic::{BitLaneSums, SlicedBit, SlicedQuadratic, SlicedQuadraticFolder};

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
///
/// `S` must have four elements: the product and the coordinates assume `GF(4)`.
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
    pub(crate) const fn from_planes(low: u64, high: u64) -> Self {
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
    pub(crate) const fn broadcast(low: bool, high: bool) -> Self {
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
    /// The planes and the flag accumulate on their own: a whole value carried around the loop
    /// drags its padding bytes through memory with it.
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let (mut low, mut high, mut poisoned) = (0, 0, false);
        for x in iter {
            low ^= x.low;
            high ^= x.high;
            poisoned |= x.poisoned;
        }
        Self::with_flags(low, high, poisoned, false)
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

/// Fewest constraints for the ordinary sliced folder to take the prepared kernel.
///
/// Each evaluation pays the kernel a fixed cost, clearing its sums and reading them out
/// through 128 plane sums and a 128-term product. An AIR asserting fewer constraints saves
/// less than that over summing them one at a time.
const MIN_CONSTRAINTS: usize = 640;

/// Fewest constraints for the four-cell single-plane folder to take the prepared kernel.
///
/// The higher cutoff reflects its four prepared states. On the quadratic folder's measured
/// all-quadratic path, the kernel was slower at 1,280 constraints and faster at 2,560.
const MIN_BIT_CONSTRAINTS: usize = 2_560;

/// Descending alpha powers laid out for the sliced folders to sum constraints eight at a time.
///
/// Every constraint's lanes are weighted the same way, so the weights can wait until the end:
///
/// ```text
///     sum_i alpha_i * sum_lane w(lane) * C_i(lane)  =  sum_lane w(lane) * A(lane)
///     A(lane)  =  sum_i low_i(lane) * alpha_i  +  high_i(lane) * g alpha_i
/// ```
///
/// Each `A(lane)` is a sum of powers picked by the lane's bits, a sum over `F_2` of their
/// coordinates. Where the target has a kernel for it, eight constraints' bits fill a byte per
/// lane, and an `8 x 8` bit matrix per coordinate byte adds the eight powers they pick.
/// A lane's coordinates then leave through its weight once per evaluation.
#[derive(Debug)]
pub(crate) struct PreparedPowers<R>(kernel::Prepared<R>);

impl<R: Field> PreparedPowers<R> {
    /// Lay out each AIR's alpha powers and their products with `generator`, the image of `g`.
    ///
    /// The coordinates are read from `R`'s byte encoding, which must be linear over `F_2`:
    /// sixteen bytes, and the encoding of a sum the XOR of the encodings. Preparing checks
    /// the size, characteristic two, that zero encodes to zero, that a run of sums encodes to
    /// the XOR of their terms, and that each element of the basis it solves for encodes to its
    /// unit vector. That rejects a broken encoding it meets but does not prove a sound one.
    ///
    /// # Returns
    ///
    /// One layout per AIR, or `None` where the folder sums constraint by constraint instead: the
    /// target has no kernel, `R`'s encoding fails the checks, or the AIR asserts too few
    /// constraints to repay the kernel's fixed cost per evaluation.
    #[must_use]
    pub(crate) fn per_air(alpha_powers: &[Vec<R>], generator: R) -> Vec<Option<Self>> {
        // The two-plane folder's crossover has only been measured for 128-bit fields.
        if R::NUM_BYTES != 16 {
            return alpha_powers.iter().map(|_| None).collect();
        }
        kernel::Prepared::per_air(alpha_powers, generator, MIN_CONSTRAINTS)
            .into_iter()
            .map(|prepared| prepared.map(Self))
            .collect()
    }

    /// Lay out powers for the single-plane folder, using its measured activation threshold.
    ///
    /// This also accepts a characteristic-two field with a linear 24-byte encoding.
    #[must_use]
    pub(crate) fn per_air_bits(alpha_powers: &[Vec<R>]) -> Vec<Option<Self>> {
        kernel::Prepared::per_air(alpha_powers, R::ZERO, MIN_BIT_CONSTRAINTS)
            .into_iter()
            .map(|prepared| prepared.map(Self))
            .collect()
    }
}

pub(crate) use kernel::PreparedSums;

/// The kernel's running sums of one evaluation, beside the layout of the powers they add.
#[derive(Debug)]
struct KernelSums<'a, R> {
    /// The alpha powers laid out for the kernel.
    prepared: &'a PreparedPowers<R>,
    /// The per-lane sums of the constraints asserted so far.
    sums: PreparedSums,
}

/// One AIR evaluation over sixty-four rows, lane-weighted and alpha-batched.
#[derive(Clone, Copy, Debug)]
pub struct SlicedEvaluation<R> {
    /// `sum_i alpha^(n-1-i) * sum_lane w(lane) * C_i(lane)`, one such sum per evaluation for a
    /// folder that carries several side by side.
    pub value: R,
    /// Whether any constraint value was poisoned, which makes `value` meaningless.
    pub poisoned: bool,
}

/// AIR folder over sixty-four rows of bit-sliced `GF(4)` values.
///
/// Every asserted constraint is summed across the lanes with the weights of a [`LaneSums`], then
/// weighted by its descending alpha power. Lookup declarations are dropped, as in
/// [`crate::folder::MultilinearFolder`]: a stage that declares lookups never runs here. A bus
/// declaration leaves only its Booleanity check, asserted like any other constraint; its tuple
/// is left to the bus protocol.
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
    /// Running lane-weighted, alpha-batched sum, one constraint at a time.
    ///
    /// Beside the kernel only debug builds keep it, to check the kernel's sum against.
    accumulator: R,
    /// The kernel's sums, when the kernel sums the constraints instead.
    kernel: Option<KernelSums<'a, R>>,
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
            kernel: None,
            constraint_index: 0,
            poisoned: false,
        }
    }

    /// Sum the constraints with the kernel, from `prepared`, the layout of the alpha powers.
    ///
    /// # Panics
    ///
    /// Panics if `prepared` does not hold one power per alpha power.
    #[inline]
    #[must_use]
    pub(crate) fn with_prepared_powers(mut self, prepared: &'a PreparedPowers<R>) -> Self {
        assert_eq!(
            prepared.0.len(),
            self.alpha_powers.len(),
            "the prepared powers must be the attached alpha powers"
        );
        self.kernel = Some(KernelSums {
            prepared,
            sums: PreparedSums::new(),
        });
        self
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
        let value = match &mut self.kernel {
            Some(kernel) => {
                let value = kernel.sums.finish(kernel.prepared, self.lanes);
                debug_assert_eq!(
                    value, self.accumulator,
                    "the kernel must sum what the lane tables sum"
                );
                value
            }
            None => self.accumulator,
        };
        SlicedEvaluation {
            value,
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

    // An AIR asserts in its hottest loops, and a call per constraint would cost more than
    // either sum.
    #[inline(always)]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.poisoned |= x.poisoned;
        // A constraint past the last power is only counted; the count check rejects it.
        if let Some(kernel) = &mut self.kernel {
            kernel
                .sums
                .add(kernel.prepared, self.constraint_index, x.low, x.high);
        }
        // Summed one at a time, a value that vanishes on every lane adds nothing, as a
        // selector-gated one mostly does.
        if (self.kernel.is_none() || cfg!(debug_assertions))
            && let Some(&power) = self.alpha_powers.get(self.constraint_index)
            && (x.low | x.high) != 0
        {
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

/// Bus tuples are dropped: a declaration's Booleanity check is asserted before it is recorded.
impl<'a, F, S, R> BusInteractionRecorder for SlicedFolder<'a, F, S, R>
where
    F: HasSubfield<S>,
    S: Field,
    R: Field,
{
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        _token: RecordToken,
        _bus: BusName<'_>,
        _direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        _activation: BusActivation<Self::Expr>,
    ) {
        // The bus protocol evaluates its retained symbolic profile after commitment.
        fields.into_iter().for_each(|field| {
            let _ = field.into();
        });
    }
}

/// The byte-sliced constraint sums, on a target with `8 x 8` bit-matrix products and wide
/// registers.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod kernel {
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::arch::x86_64::*;

    use p3_field::Field;
    use p3_maybe_rayon::prelude::*;

    use super::{
        BitLaneSums, LaneSums, PreparedPowers, SLICED_LANES, TABLE_BITS, TABLES_PER_PLANE,
    };

    /// Constraints one block gathers, one per bit of a lane byte.
    const BLOCK: usize = 8;

    /// Coordinates over `F_2` of an element of a field the kernel takes.
    const COORDINATES: usize = 128;

    /// Bytes of an element's encoding, one coordinate per bit.
    const BYTES: usize = COORDINATES / 8;

    /// Additional coordinates of a cubic extension over a 64-bit base field.
    const EXTRA_BYTES: usize = 8;
    const WIDE_BYTES: usize = BYTES + EXTRA_BYTES;
    const WIDE_COORDINATES: usize = WIDE_BYTES * 8;

    /// Words one register holds.
    const REGISTER_WORDS: usize = 8;

    // `UNIT`, `REVERSED`, `interleave` and `gather` repeat the constants of the ring-switch
    // products kernel in `p3_sumcheck`'s `ring_switch::bits::products`, where `gather` is
    // `transpose_words`. That kernel keeps them private, so a fix to either copy belongs in both.

    /// Quadword whose byte `i` is `1 << i`, the input that transposes a matrix operand.
    const UNIT: u64 = 0x8040_2010_0804_0201;

    /// Quadword whose byte `i` is `1 << (7 - i)`, the transpose with its bytes reversed.
    const REVERSED: u64 = 0x0102_0408_1020_4080;

    /// The in-lane byte shuffle interleaving a 128-bit lane's two words, byte `b` of each
    /// together.
    ///
    /// Byte pair `b` of a lane holding words `2k` and `2k + 1` is then their bytes `b`.
    const fn interleave() -> [i64; 8] {
        let mut quadwords = [0i64; 8];
        let mut q = 0;
        while q < 8 {
            let mut word = 0u64;
            let mut byte = 0;
            while byte < 8 {
                // Byte `2b` of the lane takes byte `b` of its first word, byte `2b + 1` of its
                // second.
                let b = 8 * (q % 2) + byte;
                word |= ((b / 2 + 8 * (b % 2)) as u64) << (8 * byte);
                byte += 1;
            }
            quadwords[q] = word as i64;
            q += 1;
        }
        quadwords
    }

    /// See [`interleave`].
    const INTERLEAVE: [i64; 8] = interleave();

    /// The byte-pair permute taking pair `b` of lane `k` to pair `k` of quadword `b`.
    ///
    /// After [`INTERLEAVE`], byte `j` of quadword `b` is then byte `b` of word `j`.
    const fn gather() -> [i64; 8] {
        let mut quadwords = [0i64; 8];
        let mut b = 0;
        while b < 8 {
            let mut word = 0u64;
            let mut k = 0;
            while k < 4 {
                word |= ((8 * k + b) as u64) << (16 * k);
                k += 1;
            }
            quadwords[b] = word as i64;
            b += 1;
        }
        quadwords
    }

    /// See [`gather`].
    const GATHER: [i64; 8] = gather();

    /// Descending alpha powers as the bit matrices that add them, eight constraints a block.
    #[derive(Debug)]
    pub(super) struct Prepared<R> {
        /// Per block, the matrices of its constraints' powers, then of their products with `g`.
        ///
        /// Matrix `p` of a plane takes a lane byte whose bit `k` is constraint `7 - k`'s lane
        /// bit, and returns coordinate byte `p` of the sum of the powers it picks.
        blocks: Vec<u64>,
        /// Number of powers.
        len: usize,
        /// `basis[b]`: the element whose encoding sets coordinate `b` alone, one for every AIR.
        basis: Arc<[R]>,
    }

    impl<R: Field> Prepared<R> {
        /// Lay out each AIR's powers against one basis of `R`, see [`PreparedPowers::per_air`].
        pub(super) fn per_air(
            alpha_powers: &[Vec<R>],
            generator: R,
            min_constraints: usize,
        ) -> Vec<Option<Self>> {
            let basis = alpha_powers
                .iter()
                .any(|powers| powers.len() >= min_constraints)
                .then(coordinate_basis::<R>)
                .flatten()
                .map(Arc::from);
            alpha_powers
                .iter()
                .map(|powers| {
                    let basis = basis.as_ref()?;
                    (powers.len() >= min_constraints)
                        .then(|| Self::new(powers, generator, Arc::clone(basis)))
                })
                .collect()
        }

        /// Lay out `alpha_powers` and their products with `generator`, whatever their number.
        ///
        /// `basis` must be what [`coordinate_basis`] returns for `R`.
        pub(super) fn new(alpha_powers: &[R], generator: R, basis: Arc<[R]>) -> Self {
            let bytes = R::NUM_BYTES;
            assert!(matches!(bytes, BYTES | WIDE_BYTES));
            assert_eq!(basis.len(), bytes * 8);
            let mut blocks = vec![0; alpha_powers.len().div_ceil(BLOCK) * 2 * bytes];
            blocks
                .par_chunks_mut(2 * bytes)
                .zip(alpha_powers.par_chunks(BLOCK))
                .for_each(|(block, powers)| {
                    // SAFETY: this module is compiled only where the build enables every target
                    // feature the kernels name. Each matrix occupies exactly `bytes` words.
                    unsafe {
                        if bytes == BYTES {
                            let low = core::array::from_fn(|j| {
                                powers.get(j).map_or(0, |&p| coordinates(p))
                            });
                            let high = core::array::from_fn(|j| {
                                powers.get(j).map_or(0, |&p| coordinates(generator * p))
                            });
                            block[..bytes].copy_from_slice(&block_matrices(&low));
                            block[bytes..].copy_from_slice(&block_matrices(&high));
                        } else {
                            let low = core::array::from_fn(|j| {
                                powers.get(j).map_or([0; 3], |&p| coordinates_192(p))
                            });
                            let high = core::array::from_fn(|j| {
                                powers
                                    .get(j)
                                    .map_or([0; 3], |&p| coordinates_192(generator * p))
                            });
                            block[..bytes].copy_from_slice(&block_matrices_192(&low));
                            block[bytes..].copy_from_slice(&block_matrices_192(&high));
                        }
                    }
                });
            Self {
                blocks,
                len: alpha_powers.len(),
                basis,
            }
        }

        /// Number of powers laid out.
        pub(super) const fn len(&self) -> usize {
            self.len
        }
    }

    /// The coordinates of `value`, its 16-byte encoding read as a little-endian word.
    fn coordinates<R: Field>(value: R) -> u128 {
        let mut bytes = [0u8; BYTES];
        for (slot, byte) in bytes.iter_mut().zip(value.into_bytes()) {
            *slot = byte;
        }
        u128::from_le_bytes(bytes)
    }

    /// The elements whose encodings are the unit vectors, element `b` setting coordinate `b`.
    ///
    /// # Returns
    ///
    /// `None` unless `R` has characteristic two and a 16- or 24-byte encoding that passes the
    /// checks of [`basis_under`].
    pub(super) fn coordinate_basis<R: Field>() -> Option<Box<[R]>> {
        if R::ONE + R::ONE != R::ZERO {
            return None;
        }
        match R::NUM_BYTES {
            BYTES => basis_under(coordinates::<R>).map(|basis| -> Box<[R]> { basis }),
            WIDE_BYTES => basis_under_192(coordinates_192::<R>).map(|basis| -> Box<[R]> { basis }),
            _ => None,
        }
    }

    /// The elements whose images under `encode` are the unit vectors.
    ///
    /// The first 128 powers of a generator of a field of 128 coordinates span it. Reducing their
    /// images to the identity, and adding the powers alongside, leaves each unit vector beside
    /// the element it encodes, provided `encode` is linear over `F_2`.
    ///
    /// Linearity is spot-checked, not proved: zero must encode to zero, each power plus the next
    /// to the XOR of their images, and every element found to its unit vector.
    ///
    /// # Returns
    ///
    /// `None` when a check fails or the powers span fewer than 128 coordinates.
    pub(super) fn basis_under<R: Field>(
        encode: impl Fn(R) -> u128,
    ) -> Option<Box<[R; COORDINATES]>> {
        if encode(R::ZERO) != 0 {
            return None;
        }
        let mut rows = R::GENERATOR
            .powers()
            .take(COORDINATES)
            .map(|power| (encode(power), power))
            .collect::<Vec<_>>();
        if !rows
            .windows(2)
            .all(|pair| encode(pair[0].1 + pair[1].1) == pair[0].0 ^ pair[1].0)
        {
            return None;
        }
        for bit in 0..COORDINATES {
            let pivot = (bit..COORDINATES).find(|&row| (rows[row].0 >> bit) & 1 == 1)?;
            rows.swap(bit, pivot);
            let (unit, element) = rows[bit];
            for (row, (bits, value)) in rows.iter_mut().enumerate() {
                if row != bit && (*bits >> bit) & 1 == 1 {
                    *bits ^= unit;
                    *value += element;
                }
            }
        }
        // An encoding other than the coordinate vector leaves some element off its unit vector.
        let basis = rows
            .into_iter()
            .map(|(_, element)| element)
            .collect::<Vec<_>>();
        if !basis
            .iter()
            .enumerate()
            .all(|(bit, &element)| encode(element) == 1 << bit)
        {
            return None;
        }
        basis.into_boxed_slice().try_into().ok()
    }

    /// A 24-byte encoding as three little-endian coordinate words.
    fn coordinates_192<R: Field>(value: R) -> [u64; 3] {
        let mut words = [0; 3];
        for (i, byte) in value.into_bytes().into_iter().enumerate() {
            words[i / 8] |= u64::from(byte) << (8 * (i % 8));
        }
        words
    }

    /// The same admission and basis recovery as [`basis_under`], over three coordinate words.
    pub(super) fn basis_under_192<R: Field>(
        encode: impl Fn(R) -> [u64; 3],
    ) -> Option<Box<[R; WIDE_COORDINATES]>> {
        if encode(R::ZERO) != [0; 3] {
            return None;
        }
        let mut rows = R::GENERATOR
            .powers()
            .take(WIDE_COORDINATES)
            .map(|power| (encode(power), power))
            .collect::<Vec<_>>();
        if !rows.windows(2).all(|pair| {
            encode(pair[0].1 + pair[1].1) == core::array::from_fn(|i| pair[0].0[i] ^ pair[1].0[i])
        }) {
            return None;
        }
        for bit in 0..WIDE_COORDINATES {
            let limb = bit / 64;
            let mask = 1u64 << (bit % 64);
            let pivot = (bit..WIDE_COORDINATES).find(|&row| rows[row].0[limb] & mask != 0)?;
            rows.swap(bit, pivot);
            let (unit, element) = rows[bit];
            for (row, (bits, value)) in rows.iter_mut().enumerate() {
                if row != bit && bits[limb] & mask != 0 {
                    for (word, pivot) in bits.iter_mut().zip(unit) {
                        *word ^= pivot;
                    }
                    *value += element;
                }
            }
        }
        let basis = rows
            .into_iter()
            .map(|(_, element)| element)
            .collect::<Vec<_>>();
        if !basis.iter().enumerate().all(|(bit, &element)| {
            let mut unit = [0; 3];
            unit[bit / 64] = 1 << (bit % 64);
            encode(element) == unit
        }) {
            return None;
        }
        basis.into_boxed_slice().try_into().ok()
    }

    /// The matrices adding the eight powers with coordinates `powers`, one per coordinate byte.
    ///
    /// ```text
    ///     matrix p, byte 7 - i, bit k  =  coordinate 8p + i of power 7 - k
    /// ```
    ///
    /// Quadword `p` of the gathered bytes holds byte `p` of every power, power `j` at byte `j`,
    /// and the reversed transpose moves bit `8p + i` of power `7 - k` to bit `k` of byte `7 - i`.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    fn block_matrices(powers: &[u128; BLOCK]) -> [u64; BYTES] {
        let mut gathered = [0u64; BYTES];
        for (j, power) in powers.iter().enumerate() {
            for (row, &byte) in gathered.iter_mut().zip(&power.to_le_bytes()) {
                *row |= u64::from(byte) << (8 * j);
            }
        }
        let mut matrices = [0u64; BYTES];
        for (matrices, gathered) in matrices
            .as_chunks_mut::<REGISTER_WORDS>()
            .0
            .iter_mut()
            .zip(gathered.as_chunks::<REGISTER_WORDS>().0)
        {
            let transposed = _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(REVERSED as i64),
                load(gathered),
            );
            store(matrices, transposed);
        }
        matrices
    }

    /// The 24 coordinate-byte matrices of eight 192-bit powers.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    fn block_matrices_192(powers: &[[u64; 3]; BLOCK]) -> [u64; WIDE_BYTES] {
        let mut gathered = [0u64; WIDE_BYTES];
        for (j, power) in powers.iter().enumerate() {
            for (i, row) in gathered.iter_mut().enumerate() {
                let byte = (power[i / 8] >> (8 * (i % 8))) as u8;
                *row |= u64::from(byte) << (8 * j);
            }
        }
        let mut matrices = [0; WIDE_BYTES];
        for (matrix, gathered) in matrices
            .as_chunks_mut::<REGISTER_WORDS>()
            .0
            .iter_mut()
            .zip(gathered.as_chunks::<REGISTER_WORDS>().0)
        {
            store(
                matrix,
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(REVERSED as i64),
                    load(gathered),
                ),
            );
        }
        matrices
    }

    /// Blocks whose planes wait at once.
    ///
    /// A block is added once the next one fills. Its words have left the store buffer by then,
    /// where one wide load of them would otherwise wait for eight narrow stores to land.
    const WAITING: usize = 2;

    /// Words of the ring each plane waits in, one block after another.
    const RING: usize = WAITING * BLOCK;

    /// The cubic field's extra rows retain the head's register alignment when allocated.
    #[derive(Debug)]
    #[repr(C, align(64))]
    struct ExtraSums([[u64; REGISTER_WORDS]; EXTRA_BYTES]);

    /// Running sums of one evaluation's constraints against a [`PreparedPowers`] layout.
    ///
    /// Constraint `i` enters as its two planes, the constraints in order and each once, and
    /// `sum_i alpha_i * sum_lane w(lane) * C_i(lane)` leaves. A plane that vanishes on a whole
    /// block costs nothing, so values in `F_2` enter with a zero high plane.
    ///
    /// Word `w` of row `p` of the sums holds coordinate byte `p` of lanes `8w .. 8w + 7`, lane
    /// `l` at byte `l`. The planes of the blocks not yet added wait beside them, one word per
    /// constraint.
    #[derive(Debug)]
    #[repr(C, align(64))]
    pub(crate) struct PreparedSums {
        /// Coordinate bytes of every lane's sum, one register's worth per coordinate byte.
        sums: [[u64; REGISTER_WORDS]; BYTES],
        /// The waiting low planes, then the waiting high planes, constraint `i` at word
        /// `i % RING` of each ring.
        planes: [[u64; RING]; 2],
        /// Whether a block has reached the sums, which are all zero until one does.
        carried: bool,
        /// Only active 192-bit evaluations allocate the extra coordinate rows. This pointer
        /// fits the original struct's tail padding, preserving the 128-bit stack footprint.
        extra: Option<Box<ExtraSums>>,
    }

    impl PreparedSums {
        /// No constraint yet.
        pub(crate) const fn new() -> Self {
            Self {
                sums: [[0; REGISTER_WORDS]; BYTES],
                planes: [[0; RING]; 2],
                carried: false,
                extra: None,
            }
        }

        #[cfg(test)]
        pub(super) const fn has_extra_sums(&self) -> bool {
            self.extra.is_some()
        }

        /// Add constraint `index` with planes `low` and `high`.
        ///
        /// Filling a block adds the one before it. A constraint past the prepared powers adds
        /// nothing, and the folder's count check rejects it.
        #[inline]
        pub(crate) fn add<R: Field>(
            &mut self,
            prepared: &PreparedPowers<R>,
            index: usize,
            low: u64,
            high: u64,
        ) {
            self.add_planes(prepared, index, [low, high]);
        }

        /// Add a Boolean constraint for [`Self::finish_bits`], without recording a high plane.
        #[inline]
        pub(crate) fn add_bits<R: Field>(
            &mut self,
            prepared: &PreparedPowers<R>,
            index: usize,
            bits: u64,
        ) {
            self.add_planes(prepared, index, [bits]);
        }

        #[inline]
        fn add_planes<R: Field, const N: usize>(
            &mut self,
            prepared: &PreparedPowers<R>,
            index: usize,
            planes: [u64; N],
        ) {
            if index >= prepared.0.len {
                return;
            }
            for (ring, plane) in self.planes.iter_mut().zip(planes) {
                ring[index % RING] = plane;
            }
            if index % BLOCK == BLOCK - 1 && index >= BLOCK {
                // SAFETY: this module is compiled only where the build enables every target
                // feature the kernel names.
                unsafe { self.flush::<R, N>(&prepared.0, index / BLOCK - 1) };
            }
        }

        /// The lane-weighted, alpha-batched sum, once every prepared constraint has been added.
        ///
        /// The last full block still waits for a next one, and so does a partial block after it.
        /// When no block carried a set bit, the sum is zero without reading the lanes out.
        pub(crate) fn finish<R: Field>(
            &mut self,
            prepared: &PreparedPowers<R>,
            lanes: &LaneSums<R>,
        ) -> R {
            self.finish_with::<R, 2>(prepared, |plane| plane_sum(lanes, plane))
        }

        /// [`Self::finish`], for values added through [`Self::add_bits`].
        pub(crate) fn finish_bits<R: Field>(
            &mut self,
            prepared: &PreparedPowers<R>,
            lanes: &BitLaneSums<R>,
        ) -> R {
            self.finish_with::<R, 1>(prepared, |plane| lanes.sum(plane))
        }

        /// Finish the byte-sliced sums, contracting every coordinate plane through `plane_sum`.
        fn finish_with<R: Field, const N: usize>(
            &mut self,
            prepared: &PreparedPowers<R>,
            plane_sum: impl FnMut(u64) -> R,
        ) -> R {
            let prepared = &prepared.0;
            let full = prepared.len / BLOCK;
            // SAFETY: this module is compiled only where the build enables every target feature
            // the kernel names.
            unsafe {
                if full > 0 {
                    self.flush::<R, N>(prepared, full - 1);
                }
                if !prepared.len.is_multiple_of(BLOCK) {
                    self.flush::<R, N>(prepared, full);
                }
            }
            if !self.carried {
                return R::ZERO;
            }
            // SAFETY: this module is compiled only where the build enables every target feature
            // the kernel names.
            unsafe {
                match R::NUM_BYTES {
                    BYTES => self.contract(
                        prepared
                            .basis
                            .as_ref()
                            .try_into()
                            .expect("128 coordinate basis elements"),
                        plane_sum,
                    ),
                    WIDE_BYTES => self.contract_192(
                        prepared
                            .basis
                            .as_ref()
                            .try_into()
                            .expect("192 coordinate basis elements"),
                        plane_sum,
                    ),
                    _ => unreachable!("only 128- and 192-bit fields are prepared"),
                }
            }
        }

        /// Add block `block` to the sums through its matrices, and clear its planes.
        ///
        /// A plane on which the whole block vanishes adds nothing.
        #[target_feature(enable = "avx512f,avx512bw,gfni")]
        fn flush<R: Field, const N: usize>(&mut self, prepared: &Prepared<R>, block: usize) {
            let at = block % WAITING;
            let planes: [_; N] =
                core::array::from_fn(|plane| load(&self.planes[plane].as_chunks::<BLOCK>().0[at]));
            if planes
                .iter()
                .all(|&words| _mm512_test_epi64_mask(words, words) == 0)
            {
                return;
            }
            for ring in &mut self.planes[..N] {
                ring.as_chunks_mut::<BLOCK>().0[at] = [0; BLOCK];
            }
            let bytes = R::NUM_BYTES;
            let matrices = &prepared.blocks[block * 2 * bytes..(block + 1) * 2 * bytes];
            let active = planes.map(|words| _mm512_test_epi64_mask(words, words) != 0);
            let lanes = core::array::from_fn(|i| {
                if active[i] {
                    lane_bytes(planes[i])
                } else {
                    _mm512_setzero_si512()
                }
            });
            accumulate_bytes(
                &mut self.sums,
                core::array::from_fn(|plane| {
                    matrices[plane * bytes..plane * bytes + BYTES]
                        .try_into()
                        .unwrap()
                }),
                &lanes,
                active,
            );
            if bytes == WIDE_BYTES {
                let extra = self
                    .extra
                    .get_or_insert_with(|| Box::new(ExtraSums([[0; REGISTER_WORDS]; EXTRA_BYTES])));
                accumulate_bytes(
                    &mut extra.0,
                    core::array::from_fn(|plane| {
                        matrices[plane * bytes + BYTES..(plane + 1) * bytes]
                            .try_into()
                            .unwrap()
                    }),
                    &lanes,
                    active,
                );
            }
            self.carried = true;
        }

        /// `sum_lane w(lane) * A(lane)` from the coordinate bytes of every lane's sum.
        ///
        /// ```text
        ///     sum_lane w(lane) * A(lane)  =  sum_b basis[b] * (sum of w over the lanes with bit b)
        /// ```
        #[target_feature(enable = "avx512f,avx512bw")]
        fn contract<R: Field>(
            &self,
            basis: &[R; COORDINATES],
            mut plane_sum: impl FnMut(u64) -> R,
        ) -> R {
            let mut sums = [R::ZERO; COORDINATES];
            for (bytes, sums) in self.sums.iter().zip(sums.as_chunks_mut::<8>().0.iter_mut()) {
                let bytes = load(bytes);
                for (bit, sum) in sums.iter_mut().enumerate() {
                    let plane = _mm512_test_epi8_mask(bytes, _mm512_set1_epi8((1u8 << bit) as i8));
                    *sum = plane_sum(plane);
                }
            }
            R::dot_product(basis, &sums)
        }

        /// Contract all three coordinate words of a cubic field after at least one active block.
        #[target_feature(enable = "avx512f,avx512bw")]
        fn contract_192<R: Field>(
            &self,
            basis: &[R; WIDE_COORDINATES],
            mut plane_sum: impl FnMut(u64) -> R,
        ) -> R {
            let extra = self
                .extra
                .as_ref()
                .expect("an active 192-bit sum has its extra rows");
            let mut sums = [R::ZERO; WIDE_COORDINATES];
            for (bytes, sums) in self
                .sums
                .iter()
                .chain(&extra.0)
                .zip(sums.as_chunks_mut::<8>().0.iter_mut())
            {
                let bytes = load(bytes);
                for (bit, sum) in sums.iter_mut().enumerate() {
                    let plane = _mm512_test_epi8_mask(bytes, _mm512_set1_epi8((1u8 << bit) as i8));
                    *sum = plane_sum(plane);
                }
            }
            R::dot_product(basis, &sums)
        }
    }

    /// Accumulate a bounded group of coordinate rows without keeping all 24 sums in registers.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    fn accumulate_bytes<const B: usize, const N: usize>(
        words: &mut [[u64; REGISTER_WORDS]; B],
        matrices: [&[u64; B]; N],
        lanes: &[__m512i; N],
        active: [bool; N],
    ) {
        let mut sums = words.map(|row| load(&row));
        for plane in 0..N {
            if !active[plane] {
                continue;
            }
            for (sum, &matrix) in sums.iter_mut().zip(matrices[plane]) {
                let picked = _mm512_gf2p8affine_epi64_epi8::<0>(
                    lanes[plane],
                    _mm512_set1_epi64(matrix as i64),
                );
                *sum = _mm512_xor_si512(*sum, picked);
            }
        }
        for (words, sum) in words.iter_mut().zip(sums) {
            store(words, sum);
        }
    }

    /// Eight constraints' planes as lane bytes: bit `k` of byte `l` is lane `l` of word `7 - k`.
    ///
    /// Byte `b` of word `j` holds lanes `8b .. 8b + 7` of constraint `j`. Gathering byte `b` of
    /// every word into quadword `b` leaves an `8 x 8` bit block per quadword, which the affine
    /// instruction transposes with the unit input.
    #[target_feature(enable = "avx512f,avx512bw,gfni")]
    fn lane_bytes(words: __m512i) -> __m512i {
        let pairs = _mm512_shuffle_epi8(words, constant(INTERLEAVE));
        let bytes = _mm512_permutexvar_epi16(constant(GATHER), pairs);
        _mm512_gf2p8affine_epi64_epi8::<0>(_mm512_set1_epi64(UNIT as i64), bytes)
    }

    /// The lane-weighted sum of the lanes set in `plane`, each weighted once.
    #[inline]
    fn plane_sum<R: Field>(lanes: &LaneSums<R>, plane: u64) -> R {
        const { assert!(SLICED_LANES == TABLES_PER_PLANE * TABLE_BITS) };
        let mut sum = R::ZERO;
        for byte in 0..TABLES_PER_PLANE {
            sum += lanes.tables[byte][usize::from((plane >> (byte * TABLE_BITS)) as u8)];
        }
        sum
    }

    /// Eight words as one register, the first lowest.
    #[target_feature(enable = "avx512f")]
    fn load(words: &[u64; REGISTER_WORDS]) -> __m512i {
        // SAFETY: the words span the register's sixty-four bytes, and the load takes any
        // alignment.
        unsafe { _mm512_loadu_si512(words.as_ptr().cast()) }
    }

    /// One register as eight words, the lowest first.
    #[target_feature(enable = "avx512f")]
    fn store(words: &mut [u64; REGISTER_WORDS], value: __m512i) {
        // SAFETY: the words span the register's sixty-four bytes, and the store takes any
        // alignment.
        unsafe { _mm512_storeu_si512(words.as_mut_ptr().cast(), value) }
    }

    /// A register of eight quadwords, the first lowest.
    #[target_feature(enable = "avx512f")]
    fn constant(quadwords: [i64; 8]) -> __m512i {
        let [q0, q1, q2, q3, q4, q5, q6, q7] = quadwords;
        _mm512_set_epi64(q7, q6, q5, q4, q3, q2, q1, q0)
    }
}

/// No byte-sliced kernel on this target, so nothing is ever prepared.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
mod kernel {
    use alloc::vec::Vec;
    use core::convert::Infallible;
    use core::marker::PhantomData;

    use p3_field::Field;

    use super::{BitLaneSums, LaneSums, PreparedPowers};

    /// A layout no value can take, so every constraint is summed on its own.
    #[derive(Debug)]
    pub(super) struct Prepared<R>(Infallible, PhantomData<fn() -> R>);

    // The kernel's methods run intrinsics, so they cannot be `const`.
    // Keeping these the same stops constness from leaking into callers on some targets only.
    #[allow(clippy::missing_const_for_fn)]
    impl<R> Prepared<R> {
        /// Refuses every AIR, the target having no kernel to prepare for.
        pub(super) fn per_air(
            alpha_powers: &[Vec<R>],
            _generator: R,
            _min_constraints: usize,
        ) -> Vec<Option<Self>> {
            alpha_powers.iter().map(|_| None).collect()
        }

        /// Never called: no value of this type exists.
        pub(super) fn len(&self) -> usize {
            match self.0 {}
        }
    }

    /// Nothing to hold without the kernel.
    #[derive(Debug)]
    pub(crate) struct PreparedSums;

    // These keep the kernel's signatures, whose methods run intrinsics and write through
    // `self`.
    #[allow(clippy::missing_const_for_fn, clippy::needless_pass_by_ref_mut)]
    impl PreparedSums {
        /// No constraint yet.
        pub(crate) const fn new() -> Self {
            Self
        }

        /// Never called: no prepared layout exists.
        pub(crate) fn add<R>(
            &mut self,
            prepared: &PreparedPowers<R>,
            _index: usize,
            _low: u64,
            _high: u64,
        ) {
            match prepared.0.0 {}
        }

        /// Never called: no prepared layout exists.
        pub(crate) fn add_bits<R>(
            &mut self,
            prepared: &PreparedPowers<R>,
            _index: usize,
            _bits: u64,
        ) {
            match prepared.0.0 {}
        }

        /// Never called: no prepared layout exists.
        pub(crate) fn finish<R: Field>(
            &mut self,
            prepared: &PreparedPowers<R>,
            _lanes: &LaneSums<R>,
        ) -> R {
            match prepared.0.0 {}
        }

        /// Never called: no prepared layout exists.
        pub(crate) fn finish_bits<R: Field>(
            &mut self,
            prepared: &PreparedPowers<R>,
            _lanes: &BitLaneSums<R>,
        ) -> R {
            match prepared.0.0 {}
        }
    }
}

#[cfg(test)]
mod tests;
