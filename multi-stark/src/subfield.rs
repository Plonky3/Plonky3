//! AIR arithmetic carried out inside a small subfield of the trace field.
//!
//! When every trace cell lies in a small subfield `S`, the AIR's expressions can be computed
//! in `S` and only the batched result lifted:
//!
//! ```text
//!     expression values : S          (SubfieldVar)
//!     batched constraint: EF         (SubfieldAcc = sum_i alpha^(n-1-i) * C_i)
//! ```
//!
//! The AIR still sees an algebra over the trace field `F`. Its constants and public values are
//! `F` elements, and each one narrows into `S` on its way in. A value that does not fit poisons
//! the result: the flag is carried through every operation. A poisoned value is not the one the
//! AIR asked for, and whoever reads it must discard it and compute over `F` instead.

use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, Field, HasSubfield, PrimeCharacteristicRing};

/// An AIR expression value held in the small subfield `S` of the trace field `F`.
///
/// ```text
///     F value x   ->  narrow   ->  S value s with F::from(s) = x,  or poisoned
///     a op b      ->  S op     ->  poisoned when either operand is
/// ```
///
/// While unpoisoned, lifting a result into `F` gives what the same expression computes over `F`.
pub struct SubfieldVar<F, S> {
    /// The value, meaningful only while the flag is clear.
    value: S,
    /// Whether some `F` value outside the subfield reached this result.
    poisoned: bool,
    /// The trace field the AIR believes it computes over.
    _field: PhantomData<fn() -> F>,
}

impl<F, S> SubfieldVar<F, S> {
    /// A subfield value that no out-of-subfield input has reached.
    #[inline]
    pub const fn new(value: S) -> Self {
        Self {
            value,
            poisoned: false,
            _field: PhantomData,
        }
    }

    /// Combine two operands' results into one, poisoned when either operand is.
    #[inline]
    const fn with_flags(value: S, lhs: bool, rhs: bool) -> Self {
        Self {
            value,
            poisoned: lhs | rhs,
            _field: PhantomData,
        }
    }

    /// The subfield value.
    ///
    /// It is the image of the intended `F` value only while [`Self::is_poisoned`] is false.
    #[inline]
    pub const fn value(self) -> S
    where
        S: Copy,
    {
        self.value
    }

    /// Whether an `F` value outside the subfield has reached this value.
    #[inline]
    pub const fn is_poisoned(self) -> bool
    where
        S: Copy,
    {
        self.poisoned
    }
}

impl<F: HasSubfield<S>, S: Field> SubfieldVar<F, S> {
    /// Narrow a trace-field value into the subfield, poisoning it when it lies outside.
    ///
    /// Every `F` value that enters this type goes through here.
    #[inline]
    pub fn narrow(x: F) -> Self {
        x.as_subfield().map_or(
            Self {
                value: S::ZERO,
                poisoned: true,
                _field: PhantomData,
            },
            Self::new,
        )
    }
}

impl<F, S: Copy> Clone for SubfieldVar<F, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, S: Copy> Copy for SubfieldVar<F, S> {}

impl<F, S: fmt::Debug> fmt::Debug for SubfieldVar<F, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubfieldVar")
            .field("value", &self.value)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl<F, S: Field> Default for SubfieldVar<F, S> {
    #[inline]
    fn default() -> Self {
        Self::new(S::ZERO)
    }
}

impl<F, S: Field> Add for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::with_flags(self.value + rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<F, S: Field> AddAssign for SubfieldVar<F, S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F, S: Field> Sub for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::with_flags(self.value - rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<F, S: Field> SubAssign for SubfieldVar<F, S> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, S: Field> Neg for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::with_flags(-self.value, self.poisoned, false)
    }
}

impl<F, S: Field> Mul for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::with_flags(self.value * rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<F, S: Field> MulAssign for SubfieldVar<F, S> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, S: Field> Sum for SubfieldVar<F, S> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F, S: Field> Product for SubfieldVar<F, S> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F, S: Field> PrimeCharacteristicRing for SubfieldVar<F, S> {
    // The prime subfield embeds the same way into every field of its characteristic.
    type PrimeSubfield = S::PrimeSubfield;

    const ZERO: Self = Self::new(S::ZERO);
    const ONE: Self = Self::new(S::ONE);
    const TWO: Self = Self::new(S::TWO);
    const NEG_ONE: Self = Self::new(S::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::new(S::from_prime_subfield(f))
    }

    #[inline]
    fn double(&self) -> Self {
        Self::with_flags(self.value.double(), self.poisoned, false)
    }

    #[inline]
    fn square(&self) -> Self {
        Self::with_flags(self.value.square(), self.poisoned, false)
    }

    #[inline]
    fn bool_check(&self) -> Self {
        Self::with_flags(self.value.bool_check(), self.poisoned, false)
    }
}

impl<F: HasSubfield<S>, S: Field> From<F> for SubfieldVar<F, S> {
    #[inline]
    fn from(x: F) -> Self {
        Self::narrow(x)
    }
}

impl<F: HasSubfield<S>, S: Field> Add<F> for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        self + Self::narrow(rhs)
    }
}

impl<F: HasSubfield<S>, S: Field> AddAssign<F> for SubfieldVar<F, S> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        *self += Self::narrow(rhs);
    }
}

impl<F: HasSubfield<S>, S: Field> Sub<F> for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        self - Self::narrow(rhs)
    }
}

impl<F: HasSubfield<S>, S: Field> SubAssign<F> for SubfieldVar<F, S> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::narrow(rhs);
    }
}

impl<F: HasSubfield<S>, S: Field> Mul<F> for SubfieldVar<F, S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        self * Self::narrow(rhs)
    }
}

impl<F: HasSubfield<S>, S: Field> MulAssign<F> for SubfieldVar<F, S> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::narrow(rhs);
    }
}

impl<F: HasSubfield<S>, S: Field> Algebra<F> for SubfieldVar<F, S> {}

/// A batched constraint value over the challenge field `EF`, built from subfield expressions.
///
/// The batching weights are `EF` elements. Each constraint is a [`SubfieldVar`], so weighting
/// it multiplies by an element of `S`, which the field can apply without a general product:
///
/// ```text
///     acc += alpha^k * C    with alpha^k in EF and C in S
/// ```
///
/// The poison flag of every constraint folded in is carried along.
pub struct SubfieldAcc<EF, S> {
    /// The accumulated value, meaningful only while the flag is clear.
    value: EF,
    /// Whether a poisoned subfield value reached this accumulator.
    poisoned: bool,
    /// The subfield the folded expressions were computed in.
    _subfield: PhantomData<fn() -> S>,
}

impl<EF, S> SubfieldAcc<EF, S> {
    /// A challenge-field value that no poisoned input has reached.
    #[inline]
    pub const fn new(value: EF) -> Self {
        Self {
            value,
            poisoned: false,
            _subfield: PhantomData,
        }
    }

    /// Combine two operands' results into one, poisoned when either operand is.
    #[inline]
    const fn with_flags(value: EF, lhs: bool, rhs: bool) -> Self {
        Self {
            value,
            poisoned: lhs | rhs,
            _subfield: PhantomData,
        }
    }

    /// The accumulated value.
    ///
    /// It is the intended value only while [`Self::is_poisoned`] is false.
    #[inline]
    pub const fn value(self) -> EF
    where
        EF: Copy,
    {
        self.value
    }

    /// Whether a poisoned subfield value has reached this accumulator.
    #[inline]
    pub const fn is_poisoned(self) -> bool
    where
        EF: Copy,
    {
        self.poisoned
    }
}

impl<EF: Copy, S> Clone for SubfieldAcc<EF, S> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<EF: Copy, S> Copy for SubfieldAcc<EF, S> {}

impl<EF: fmt::Debug, S> fmt::Debug for SubfieldAcc<EF, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubfieldAcc")
            .field("value", &self.value)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

impl<EF: Field, S> Default for SubfieldAcc<EF, S> {
    #[inline]
    fn default() -> Self {
        Self::new(EF::ZERO)
    }
}

impl<EF: Field, S> Add for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::with_flags(self.value + rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<EF: Field, S> AddAssign for SubfieldAcc<EF, S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<EF: Field, S> Sub for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::with_flags(self.value - rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<EF: Field, S> SubAssign for SubfieldAcc<EF, S> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<EF: Field, S> Neg for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::with_flags(-self.value, self.poisoned, false)
    }
}

impl<EF: Field, S> Mul for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::with_flags(self.value * rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<EF: Field, S> MulAssign for SubfieldAcc<EF, S> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<EF: Field, S> Sum for SubfieldAcc<EF, S> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<EF: Field, S> Product for SubfieldAcc<EF, S> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<EF: Field, S> PrimeCharacteristicRing for SubfieldAcc<EF, S> {
    type PrimeSubfield = EF::PrimeSubfield;

    const ZERO: Self = Self::new(EF::ZERO);
    const ONE: Self = Self::new(EF::ONE);
    const TWO: Self = Self::new(EF::TWO);
    const NEG_ONE: Self = Self::new(EF::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::new(EF::from_prime_subfield(f))
    }
}

impl<F, EF: HasSubfield<S>, S: Field> From<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    #[inline]
    fn from(x: SubfieldVar<F, S>) -> Self {
        Self::with_flags(EF::from(x.value), x.poisoned, false)
    }
}

impl<F, EF: HasSubfield<S>, S: Field> Add<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: SubfieldVar<F, S>) -> Self {
        Self::with_flags(
            self.value + EF::from(rhs.value),
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F, EF: HasSubfield<S>, S: Field> AddAssign<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    #[inline]
    fn add_assign(&mut self, rhs: SubfieldVar<F, S>) {
        *self = *self + rhs;
    }
}

impl<F, EF: HasSubfield<S>, S: Field> Sub<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SubfieldVar<F, S>) -> Self {
        Self::with_flags(
            self.value - EF::from(rhs.value),
            self.poisoned,
            rhs.poisoned,
        )
    }
}

impl<F, EF: HasSubfield<S>, S: Field> SubAssign<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    #[inline]
    fn sub_assign(&mut self, rhs: SubfieldVar<F, S>) {
        *self = *self - rhs;
    }
}

impl<F, EF: HasSubfield<S>, S: Field> Mul<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SubfieldVar<F, S>) -> Self {
        Self::with_flags(self.value * rhs.value, self.poisoned, rhs.poisoned)
    }
}

impl<F, EF: HasSubfield<S>, S: Field> MulAssign<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {
    #[inline]
    fn mul_assign(&mut self, rhs: SubfieldVar<F, S>) {
        *self = *self * rhs;
    }
}

impl<F, EF: HasSubfield<S>, S: Field> Algebra<SubfieldVar<F, S>> for SubfieldAcc<EF, S> {}

#[cfg(test)]
mod tests;
