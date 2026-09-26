//! Bit-sliced homogeneous parts of `GF(2)` AIR constraints, sixty-four rows at a time.
//!
//! Over bit inputs and bit constants, an expression of degree at most two splits into its
//! quadratic, linear, and constant parts, each one bit per lane:
//!
//! ```text
//!     x      =  Q + L + K
//!     a + b  =  (Qa + Qb,  La + Lb,  Ka + Kb)
//!     a * b  =  (La Lb + Qa Kb + Ka Qb,  La Kb + Ka Lb,  Ka Kb)
//! ```
//!
//! The product drops the parts of degree three and four. Dropping every monomial of degree three
//! and up is a ring homomorphism, so every part of a constraint of degree at most two is exact.
//!
//! As in [`SlicedGf4`](super::SlicedGf4), a trace-field value outside `GF(2)` poisons every
//! result it reaches, and a poisoned evaluation must be discarded.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_air::{Air, AirBuilder, RowWindow};
use p3_field::{Algebra, Field, PrimeCharacteristicRing};
use p3_lookup::{Count, IndexedLookupBuilder, InteractionBuilder, TraceWindow};

use super::{SLICED_LANES, SlicedEvaluation, TABLE_BITS, TABLE_ENTRIES, TABLES_PER_PLANE};
use crate::folder::eval_boundary_io;
use crate::selectors::BoundaryEvals;

/// Sixty-four bits of one AIR input, one per lane.
///
/// An input is linear: its quadratic and constant parts are zero.
#[repr(transparent)]
pub struct SlicedBit<F> {
    /// The input's bit in every lane.
    bits: u64,
    /// The trace field the AIR believes it computes over.
    _field: PhantomData<fn() -> F>,
}

impl<F> SlicedBit<F> {
    /// The input with the given bit in every lane.
    #[inline]
    #[must_use]
    pub const fn new(bits: u64) -> Self {
        Self {
            bits,
            _field: PhantomData,
        }
    }

    /// The input's bit in every lane.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.bits
    }
}

impl<F> Clone for SlicedBit<F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for SlicedBit<F> {}

impl<F> Default for SlicedBit<F> {
    #[inline]
    fn default() -> Self {
        Self::new(0)
    }
}

impl<F> fmt::Debug for SlicedBit<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SlicedBit({:#018x})", self.bits)
    }
}

/// Sixty-four values of an expression of degree at most two, split into homogeneous parts.
///
/// Lane `i` of each part is that part of the expression evaluated at lane `i`'s inputs.
///
/// `F` must have characteristic two, as a field holding `GF(4)` does.
pub struct SlicedQuadratic<F> {
    /// The quadratic part of every lane.
    quadratic: u64,
    /// The linear part of every lane.
    linear: u64,
    /// The constant part: every bit clear, or every bit set.
    constant: u64,
    /// Whether some `F` value outside `GF(2)` reached this value.
    poisoned: bool,
    /// The trace field the AIR believes it computes over.
    _field: PhantomData<fn() -> F>,
}

impl<F> SlicedQuadratic<F> {
    /// The value with the given parts, reached by no value outside `GF(2)`.
    #[inline]
    #[must_use]
    pub(crate) const fn from_parts(quadratic: u64, linear: u64, constant: u64) -> Self {
        Self {
            quadratic,
            linear,
            constant,
            poisoned: false,
            _field: PhantomData,
        }
    }

    /// A value a trace-field value outside `GF(2)` reached; its parts carry no meaning.
    const POISONED: Self = Self {
        quadratic: 0,
        linear: 0,
        constant: 0,
        poisoned: true,
        _field: PhantomData,
    };

    /// Combine two operands' parts into one value, poisoned when either operand is.
    #[inline]
    const fn with_flags(quadratic: u64, linear: u64, constant: u64, lhs: bool, rhs: bool) -> Self {
        Self {
            quadratic,
            linear,
            constant,
            poisoned: lhs | rhs,
            _field: PhantomData,
        }
    }

    /// The quadratic part of every lane.
    #[inline]
    #[must_use]
    pub const fn quadratic(self) -> u64 {
        self.quadratic
    }

    /// The whole value of every lane, the sum of its three parts.
    #[inline]
    #[must_use]
    pub const fn value(self) -> u64 {
        self.quadratic ^ self.linear ^ self.constant
    }

    /// Whether a trace-field value outside `GF(2)` has reached this value.
    #[inline]
    #[must_use]
    pub const fn is_poisoned(self) -> bool {
        self.poisoned
    }

    /// The sum with an input, whose only part is linear.
    #[inline]
    const fn add_input(self, bits: u64) -> Self {
        Self::with_flags(
            self.quadratic,
            self.linear ^ bits,
            self.constant,
            self.poisoned,
            false,
        )
    }

    /// The product with an input, whose only part is linear.
    #[inline]
    const fn mul_input(self, bits: u64) -> Self {
        Self::with_flags(
            self.linear & bits,
            self.constant & bits,
            0,
            self.poisoned,
            false,
        )
    }
}

impl<F: Field> SlicedQuadratic<F> {
    /// Narrow a trace-field constant into every lane, poisoning it when it lies outside `GF(2)`.
    ///
    /// Every `F` value that enters this type goes through here.
    #[inline]
    #[must_use]
    pub fn narrow(x: F) -> Self {
        if x == F::ZERO {
            Self::ZERO
        } else if x == F::ONE {
            Self::ONE
        } else {
            Self::POISONED
        }
    }
}

impl<F> Clone for SlicedQuadratic<F> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for SlicedQuadratic<F> {}

impl<F> fmt::Debug for SlicedQuadratic<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlicedQuadratic")
            .field("quadratic", &format_args!("{:#018x}", self.quadratic))
            .field("linear", &format_args!("{:#018x}", self.linear))
            .field("constant", &format_args!("{:#018x}", self.constant))
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl<F> Default for SlicedQuadratic<F> {
    #[inline]
    fn default() -> Self {
        Self::from_parts(0, 0, 0)
    }
}

impl<F> From<SlicedBit<F>> for SlicedQuadratic<F> {
    #[inline]
    fn from(x: SlicedBit<F>) -> Self {
        Self::from_parts(0, x.bits, 0)
    }
}

impl<F> Add for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::with_flags(
            self.quadratic ^ rhs.quadratic,
            self.linear ^ rhs.linear,
            self.constant ^ rhs.constant,
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F> AddAssign for SlicedQuadratic<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F> Sub for SlicedQuadratic<F> {
    type Output = Self;

    /// Characteristic two: subtracting is adding.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl<F> SubAssign for SlicedQuadratic<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F> Neg for SlicedQuadratic<F> {
    type Output = Self;

    /// Characteristic two: every value is its own negation.
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl<F> Mul for SlicedQuadratic<F> {
    type Output = Self;

    /// ```text
    ///     (Qa + La + Ka)(Qb + Lb + Kb) = (La Lb + Qa Kb + Ka Qb) + (La Kb + Ka Lb) + Ka Kb
    /// ```
    ///
    /// up to the parts of degree three and four.
    // Clippy's nursery lint reads the cross terms `Qa Kb` and `Ka Qb` as typos for `Qa Qb`.
    #[allow(clippy::suspicious_operation_groupings)]
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::with_flags(
            (self.linear & rhs.linear)
                ^ (self.quadratic & rhs.constant)
                ^ (self.constant & rhs.quadratic),
            (self.linear & rhs.constant) ^ (self.constant & rhs.linear),
            self.constant & rhs.constant,
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F> MulAssign for SlicedQuadratic<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> Sum for SlicedQuadratic<F> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: Field> Product for SlicedQuadratic<F> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: Field> PrimeCharacteristicRing for SlicedQuadratic<F> {
    // The prime subfield embeds the same way into every field of its characteristic.
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self::from_parts(0, 0, 0);
    const ONE: Self = Self::from_parts(0, 0, u64::MAX);
    // `F` has characteristic two.
    const TWO: Self = Self::ZERO;
    const NEG_ONE: Self = Self::ONE;

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::narrow(F::from_prime_subfield(f))
    }

    #[inline]
    fn double(&self) -> Self {
        Self::with_flags(0, 0, 0, self.poisoned, false)
    }

    /// ```text
    ///     (Q + L + K)^2 = Q^2 + L^2 + K^2  ->  (L^2, 0, K)        Q^2 dropped, K a bit
    /// ```
    ///
    /// `L^2` is `L` on a bit.
    #[inline]
    fn square(&self) -> Self {
        Self::with_flags(self.linear, 0, self.constant, self.poisoned, false)
    }

    /// ```text
    ///     x^2 - x  ->  (L^2 + Q, L, K + K) = (L + Q, L, 0)         on bits
    /// ```
    #[inline]
    fn bool_check(&self) -> Self {
        Self::with_flags(
            self.linear ^ self.quadratic,
            self.linear,
            0,
            self.poisoned,
            false,
        )
    }
}

impl<F: Field> From<F> for SlicedQuadratic<F> {
    #[inline]
    fn from(x: F) -> Self {
        Self::narrow(x)
    }
}

impl<F: Field> Add<F> for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        self + Self::narrow(rhs)
    }
}

impl<F: Field> AddAssign<F> for SlicedQuadratic<F> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        *self += Self::narrow(rhs);
    }
}

impl<F: Field> Sub<F> for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        self - Self::narrow(rhs)
    }
}

impl<F: Field> SubAssign<F> for SlicedQuadratic<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::narrow(rhs);
    }
}

impl<F: Field> Mul<F> for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        self * Self::narrow(rhs)
    }
}

impl<F: Field> MulAssign<F> for SlicedQuadratic<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::narrow(rhs);
    }
}

impl<F: Field> Algebra<F> for SlicedQuadratic<F> {}

impl<F> Add<SlicedBit<F>> for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: SlicedBit<F>) -> Self {
        self.add_input(rhs.bits)
    }
}

impl<F> AddAssign<SlicedBit<F>> for SlicedQuadratic<F> {
    #[inline]
    fn add_assign(&mut self, rhs: SlicedBit<F>) {
        *self = *self + rhs;
    }
}

impl<F> Sub<SlicedBit<F>> for SlicedQuadratic<F> {
    type Output = Self;

    /// Characteristic two: subtracting is adding.
    #[inline]
    fn sub(self, rhs: SlicedBit<F>) -> Self {
        self.add_input(rhs.bits)
    }
}

impl<F> SubAssign<SlicedBit<F>> for SlicedQuadratic<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: SlicedBit<F>) {
        *self = *self - rhs;
    }
}

impl<F> Mul<SlicedBit<F>> for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SlicedBit<F>) -> Self {
        self.mul_input(rhs.bits)
    }
}

impl<F> MulAssign<SlicedBit<F>> for SlicedQuadratic<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: SlicedBit<F>) {
        *self = *self * rhs;
    }
}

impl<F: Field> Algebra<SlicedBit<F>> for SlicedQuadratic<F> {}

impl<F> Add for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn add(self, rhs: Self) -> SlicedQuadratic<F> {
        SlicedQuadratic::from(self).add_input(rhs.bits)
    }
}

impl<F> Sub for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    /// Characteristic two: subtracting is adding.
    #[inline]
    fn sub(self, rhs: Self) -> SlicedQuadratic<F> {
        SlicedQuadratic::from(self).add_input(rhs.bits)
    }
}

impl<F> Mul for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn mul(self, rhs: Self) -> SlicedQuadratic<F> {
        SlicedQuadratic::from(self).mul_input(rhs.bits)
    }
}

impl<F> Add<SlicedQuadratic<F>> for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn add(self, rhs: SlicedQuadratic<F>) -> SlicedQuadratic<F> {
        rhs.add_input(self.bits)
    }
}

impl<F> Sub<SlicedQuadratic<F>> for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    /// Characteristic two: subtracting is adding.
    #[inline]
    fn sub(self, rhs: SlicedQuadratic<F>) -> SlicedQuadratic<F> {
        rhs.add_input(self.bits)
    }
}

impl<F> Mul<SlicedQuadratic<F>> for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn mul(self, rhs: SlicedQuadratic<F>) -> SlicedQuadratic<F> {
        rhs.mul_input(self.bits)
    }
}

impl<F: Field> Add<F> for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn add(self, rhs: F) -> SlicedQuadratic<F> {
        SlicedQuadratic::narrow(rhs).add_input(self.bits)
    }
}

impl<F: Field> Sub<F> for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn sub(self, rhs: F) -> SlicedQuadratic<F> {
        SlicedQuadratic::narrow(rhs).add_input(self.bits)
    }
}

impl<F: Field> Mul<F> for SlicedBit<F> {
    type Output = SlicedQuadratic<F>;

    #[inline]
    fn mul(self, rhs: F) -> SlicedQuadratic<F> {
        SlicedQuadratic::narrow(rhs).mul_input(self.bits)
    }
}

/// Byte-indexed partial sums of sixty-four lane weights, for values with one bit per lane.
///
/// ```text
///     sum_lane w(lane) * bit(lane)  =  sum_byte table[byte][bits' byte]
///
///     table[byte][m] = sum of w(8 byte + i) over the set bits i of m
/// ```
#[derive(Clone, Debug)]
pub struct BitLaneSums<R> {
    /// One table per byte of a word.
    tables: Box<[[R; TABLE_ENTRIES]; TABLES_PER_PLANE]>,
}

impl<R: Field> BitLaneSums<R> {
    /// Tabulate the partial sums of `weights`.
    ///
    /// # Panics
    ///
    /// Panics if `weights` does not hold one weight per lane.
    #[must_use]
    pub fn new(weights: &[R]) -> Self {
        assert_eq!(weights.len(), SLICED_LANES, "one weight per lane");
        let mut tables = vec![[R::ZERO; TABLE_ENTRIES]; TABLES_PER_PLANE];
        for (table, weights) in tables.iter_mut().zip(weights.as_chunks::<TABLE_BITS>().0) {
            // Each mask adds its lowest set lane to the mask without it.
            for mask in 1..TABLE_ENTRIES {
                table[mask] = table[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
            }
        }
        let tables = tables
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("the table count is fixed"));
        Self { tables }
    }

    /// The lane-weighted sum of the lanes whose bit is set in `bits`.
    #[inline]
    #[must_use]
    pub fn sum(&self, bits: u64) -> R {
        let mut sum = R::ZERO;
        for (byte, table) in self.tables.iter().enumerate() {
            sum += table[usize::from((bits >> (byte * TABLE_BITS)) as u8)];
        }
        sum
    }
}

/// AIR folder over sixty-four rows, each constraint reduced to one bit per lane.
///
/// Every asserted constraint contributes its quadratic part, or its whole value when the folder
/// is built for whole values. The bits are summed across the lanes with the weights of a
/// [`BitLaneSums`], then weighted by the constraint's descending alpha power. Lookup declarations
/// are dropped, as in [`SlicedFolder`](super::SlicedFolder).
#[derive(Debug)]
pub struct SlicedQuadraticFolder<'a, F, R> {
    /// Two-row main window holding the current and shifted-by-one rows.
    main_window: RowWindow<'a, SlicedBit<F>>,
    /// Two-row preprocessed window; zero-width when the AIR has no preprocessed columns.
    preprocessed_window: RowWindow<'a, SlicedBit<F>>,
    /// Periodic column values, one per declared periodic column.
    periodic_values: &'a [SlicedBit<F>],
    /// Boundary-selector values shared by all selector accessors.
    boundary: BoundaryEvals<SlicedBit<F>>,
    /// Public inputs forwarded to the AIR, always in the trace field.
    public_values: &'a [F],
    /// Descending alpha powers, one per asserted constraint, boundary pins included.
    alpha_powers: &'a [R],
    /// The lane weights every constraint is summed with.
    lanes: &'a BitLaneSums<R>,
    /// Every bit set when a constraint contributes its whole value, clear for its quadratic part.
    whole: u64,
    /// Running lane-weighted, alpha-batched sum.
    accumulator: R,
    /// Number of constraints asserted so far, which is the next position in `alpha_powers`.
    constraint_index: usize,
    /// Whether any asserted value was poisoned.
    poisoned: bool,
}

impl<'a, F, R: Field> SlicedQuadraticFolder<'a, F, R> {
    /// Build a folder for one evaluation over sixty-four rows.
    ///
    /// # Arguments
    ///
    /// - `local`, `next`: main column inputs at the current and shifted-by-one rows.
    /// - `boundary`: selector inputs at the same rows.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `alpha_powers`: `alpha^(n - 1 - i)` for each of the `n` constraints, pins included.
    /// - `lanes`: the weights summing each constraint across the lanes.
    /// - `whole`: whether each constraint contributes its whole value rather than its quadratic
    ///   part.
    #[inline]
    #[must_use]
    pub fn new(
        local: &'a [SlicedBit<F>],
        next: &'a [SlicedBit<F>],
        boundary: BoundaryEvals<SlicedBit<F>>,
        public_values: &'a [F],
        alpha_powers: &'a [R],
        lanes: &'a BitLaneSums<R>,
        whole: bool,
    ) -> Self {
        Self {
            main_window: RowWindow::from_two_rows(local, next),
            preprocessed_window: RowWindow::from_two_rows(&[], &[]),
            periodic_values: &[],
            boundary,
            public_values,
            alpha_powers,
            lanes,
            whole: u64::from(whole).wrapping_neg(),
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
        current: &'a [SlicedBit<F>],
        next: &'a [SlicedBit<F>],
    ) -> Self {
        self.preprocessed_window = RowWindow::from_two_rows(current, next);
        self
    }

    /// Attach the periodic column values read by the AIR.
    #[inline]
    #[must_use]
    pub const fn with_periodic(mut self, values: &'a [SlicedBit<F>]) -> Self {
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

    /// Add the lane-weighted sum of `bits`, scaled by `power`, to the running sum.
    #[inline]
    fn accumulate(&mut self, power: R, bits: u64) {
        self.accumulator += power * self.lanes.sum(bits);
    }
}

impl<'a, F, R> AirBuilder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
    R: Field,
{
    type F = F;
    type Expr = SlicedQuadratic<F>;
    type Var = SlicedBit<F>;
    type MainWindow = RowWindow<'a, SlicedBit<F>>;
    type PreprocessedWindow = RowWindow<'a, SlicedBit<F>>;
    // Public values stay in the trace field and narrow into the lanes on read.
    type PublicVar = F;
    type PeriodicVar = SlicedBit<F>;

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
        self.boundary.first.into()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.boundary.last.into()
    }

    #[inline]
    fn is_transition(&self) -> Self::Expr {
        self.boundary.transition.into()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.poisoned |= x.poisoned;
        let bits = x.quadratic ^ ((x.linear ^ x.constant) & self.whole);
        // A constraint past the last power is only counted; the count check rejects it.
        // A value that vanishes on every lane adds nothing.
        if let Some(&power) = self.alpha_powers.get(self.constraint_index)
            && bits != 0
        {
            self.accumulate(power, bits);
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
impl<'a, F, R> InteractionBuilder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
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
impl<'a, F, R> IndexedLookupBuilder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
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
