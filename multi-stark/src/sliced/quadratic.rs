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

use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, Field, PrimeCharacteristicRing};

mod folder;

pub use folder::{BitLaneSums, SlicedQuadraticFolder};

/// Evaluations one AIR walk carries side by side, each over its own sixty-four lanes.
pub const SLICED_CELLS: usize = 4;

/// One word per evaluation: bit `i` of word `c` is lane `i` of evaluation `c`.
pub type CellWords = [u64; SLICED_CELLS];

/// The words `f(a[c], b[c])`, one per evaluation.
#[inline(always)]
fn zip(a: CellWords, b: CellWords, f: impl Fn(u64, u64) -> u64) -> CellWords {
    core::array::from_fn(|cell| f(a[cell], b[cell]))
}

/// The words `a & b`.
#[inline(always)]
fn and(a: CellWords, b: CellWords) -> CellWords {
    zip(a, b, |a, b| a & b)
}

/// The words `a ^ b`.
#[inline(always)]
fn xor(a: CellWords, b: CellWords) -> CellWords {
    zip(a, b, |a, b| a ^ b)
}

/// The words of one AIR input, one per evaluation.
///
/// An input is linear: its quadratic and constant parts are zero.
#[repr(transparent)]
pub struct SlicedBit<F> {
    /// The input's bits, one word per evaluation.
    bits: CellWords,
    /// The trace field the AIR believes it computes over.
    _field: PhantomData<fn() -> F>,
}

impl<F> SlicedBit<F> {
    /// The input with the given bits, one word per evaluation.
    #[inline]
    #[must_use]
    pub const fn new(bits: CellWords) -> Self {
        Self {
            bits,
            _field: PhantomData,
        }
    }

    /// The input's bits, one word per evaluation.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> CellWords {
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
        Self::new([0; SLICED_CELLS])
    }
}

impl<F> fmt::Debug for SlicedBit<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SlicedBit({:#018x?})", self.bits)
    }
}

/// An expression of degree at most two, split into homogeneous parts, over the lanes of every
/// evaluation.
///
/// Bit `i` of a part's word `c` is that part of the expression at lane `i` of evaluation `c`.
///
/// `F` must have characteristic two, as a field holding `GF(4)` does.
pub struct SlicedQuadratic<F> {
    /// The quadratic part, one word per evaluation.
    quadratic: CellWords,
    /// The linear part, one word per evaluation.
    linear: CellWords,
    /// The constant part, the same in every lane: every bit clear, or every bit set.
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
    pub(crate) const fn from_parts(quadratic: CellWords, linear: CellWords, constant: u64) -> Self {
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
        quadratic: [0; SLICED_CELLS],
        linear: [0; SLICED_CELLS],
        constant: 0,
        poisoned: true,
        _field: PhantomData,
    };

    /// Combine two operands' parts into one value, poisoned when either operand is.
    #[inline]
    const fn with_flags(
        quadratic: CellWords,
        linear: CellWords,
        constant: u64,
        lhs: bool,
        rhs: bool,
    ) -> Self {
        Self {
            quadratic,
            linear,
            constant,
            poisoned: lhs | rhs,
            _field: PhantomData,
        }
    }

    /// The quadratic part, one word per evaluation.
    #[inline]
    #[must_use]
    pub const fn quadratic(self) -> CellWords {
        self.quadratic
    }

    /// The whole value, the sum of the three parts, one word per evaluation.
    #[inline]
    #[must_use]
    pub fn value(self) -> CellWords {
        xor(
            xor(self.quadratic, self.linear),
            [self.constant; SLICED_CELLS],
        )
    }

    /// Whether a trace-field value outside `GF(2)` has reached this value.
    #[inline]
    #[must_use]
    pub const fn is_poisoned(self) -> bool {
        self.poisoned
    }

    /// The sum of two values, part by part.
    #[inline]
    fn add_parts(self, rhs: Self) -> Self {
        Self::with_flags(
            xor(self.quadratic, rhs.quadratic),
            xor(self.linear, rhs.linear),
            self.constant ^ rhs.constant,
            self.poisoned,
            rhs.poisoned,
        )
    }

    /// The product of two values:
    ///
    /// ```text
    ///     (Qa + La + Ka)(Qb + Lb + Kb) = (La Lb + Qa Kb + Ka Qb) + (La Kb + Ka Lb) + Ka Kb
    /// ```
    ///
    /// up to the parts of degree three and four.
    #[inline]
    fn mul_parts(self, rhs: Self) -> Self {
        let (lhs_constant, rhs_constant) =
            ([self.constant; SLICED_CELLS], [rhs.constant; SLICED_CELLS]);
        Self::with_flags(
            xor(
                and(self.linear, rhs.linear),
                xor(
                    and(self.quadratic, rhs_constant),
                    and(lhs_constant, rhs.quadratic),
                ),
            ),
            xor(
                and(self.linear, rhs_constant),
                and(lhs_constant, rhs.linear),
            ),
            self.constant & rhs.constant,
            self.poisoned,
            rhs.poisoned,
        )
    }

    /// The sum with an input, whose only part is linear.
    #[inline]
    fn add_input(self, bits: CellWords) -> Self {
        Self::with_flags(
            self.quadratic,
            xor(self.linear, bits),
            self.constant,
            self.poisoned,
            false,
        )
    }

    /// The product with an input, whose only part is linear.
    #[inline]
    fn mul_input(self, bits: CellWords) -> Self {
        Self::with_flags(
            and(self.linear, bits),
            and([self.constant; SLICED_CELLS], bits),
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
            .field("quadratic", &format_args!("{:#018x?}", self.quadratic))
            .field("linear", &format_args!("{:#018x?}", self.linear))
            .field("constant", &format_args!("{:#018x}", self.constant))
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl<F> Default for SlicedQuadratic<F> {
    #[inline]
    fn default() -> Self {
        Self::from_parts([0; SLICED_CELLS], [0; SLICED_CELLS], 0)
    }
}

impl<F> From<SlicedBit<F>> for SlicedQuadratic<F> {
    #[inline]
    fn from(x: SlicedBit<F>) -> Self {
        Self::from_parts([0; SLICED_CELLS], x.bits, 0)
    }
}

impl<F> Add for SlicedQuadratic<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add_parts(rhs)
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
        self.add_parts(rhs)
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

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_parts(rhs)
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

    const ZERO: Self = Self::from_parts([0; SLICED_CELLS], [0; SLICED_CELLS], 0);
    const ONE: Self = Self::from_parts([0; SLICED_CELLS], [0; SLICED_CELLS], u64::MAX);
    // `F` has characteristic two.
    const TWO: Self = Self::ZERO;
    const NEG_ONE: Self = Self::ONE;

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::narrow(F::from_prime_subfield(f))
    }

    #[inline]
    fn double(&self) -> Self {
        Self::with_flags(
            [0; SLICED_CELLS],
            [0; SLICED_CELLS],
            0,
            self.poisoned,
            false,
        )
    }

    /// ```text
    ///     (Q + L + K)^2 = Q^2 + L^2 + K^2  ->  (L^2, 0, K)        Q^2 dropped, K a bit
    /// ```
    ///
    /// `L^2` is `L` on a bit.
    #[inline]
    fn square(&self) -> Self {
        Self::with_flags(
            self.linear,
            [0; SLICED_CELLS],
            self.constant,
            self.poisoned,
            false,
        )
    }

    /// ```text
    ///     x^2 - x  ->  (L^2 + Q, L, K + K) = (L + Q, L, 0)         on bits
    /// ```
    #[inline]
    fn bool_check(&self) -> Self {
        Self::with_flags(
            xor(self.linear, self.quadratic),
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

#[cfg(test)]
mod tests;
