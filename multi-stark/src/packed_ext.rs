//! Local wrapper granting a packed extension-field value arithmetic against
//! the base field directly.

use core::fmt;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Algebra, Field, PrimeCharacteristicRing};

/// Wraps a packed extension-field value so it supports arithmetic against
/// the base field `F` directly, i.e. implements `Algebra<F>`.
///
/// `P` (typically `EF::ExtensionPacking`) only implements `Algebra<EF>` and
/// `Algebra<F::Packing>` — never `Algebra<F>` — because a direct `From<F>`
/// impl on `P` itself would conflict under orphan/coherence rules with the
/// existing `From<F::Packing>` impl: a field's fallback (non-SIMD) packing
/// can have `F::Packing = F`, so the compiler cannot prove the two impls
/// are disjoint. This type is local to this crate, so implementing
/// `Algebra<F>` for it here is not subject to that conflict.
///
/// Only the trait items required to satisfy `p3_air::AirBuilder`'s bounds
/// are overridden; every other [`PrimeCharacteristicRing`] method falls
/// back to its default (mathematically correct, not necessarily as fast as
/// `P`'s own overrides for that operation).
#[repr(transparent)]
pub struct PackedExt<F, P>(pub P, PhantomData<fn() -> F>);

impl<F, P> PackedExt<F, P> {
    #[inline]
    pub const fn new(p: P) -> Self {
        Self(p, PhantomData)
    }
}

impl<F, P: Copy> Clone for PackedExt<F, P> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, P: Copy> Copy for PackedExt<F, P> {}

impl<F, P: fmt::Debug> fmt::Debug for PackedExt<F, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<F, P: Default> Default for PackedExt<F, P> {
    #[inline]
    fn default() -> Self {
        Self::new(P::default())
    }
}

impl<F, P: Add<Output = P>> Add for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }
}

impl<F, P: AddAssign> AddAssign for PackedExt<F, P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<F, P: Sub<Output = P>> Sub for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.0 - rhs.0)
    }
}

impl<F, P: SubAssign> SubAssign for PackedExt<F, P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<F, P: Neg<Output = P>> Neg for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.0)
    }
}

impl<F, P: Mul<Output = P>> Mul for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.0 * rhs.0)
    }
}

impl<F, P: MulAssign> MulAssign for PackedExt<F, P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<F, P: PrimeCharacteristicRing + Copy> Sum for PackedExt<F, P> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|x| x.0).sum())
    }
}

impl<F, P: PrimeCharacteristicRing + Copy> Product for PackedExt<F, P> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|x| x.0).product())
    }
}

impl<F, P: PrimeCharacteristicRing + Copy> PrimeCharacteristicRing for PackedExt<F, P> {
    type PrimeSubfield = P::PrimeSubfield;

    const ZERO: Self = Self::new(P::ZERO);
    const ONE: Self = Self::new(P::ONE);
    const TWO: Self = Self::new(P::TWO);
    const NEG_ONE: Self = Self::new(P::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::new(P::from_prime_subfield(f))
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> From<F> for PackedExt<F, P> {
    #[inline]
    fn from(f: F) -> Self {
        Self::new(P::from(F::Packing::from(f)))
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> Add<F> for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: F) -> Self {
        self + Self::from(rhs)
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> AddAssign<F> for PackedExt<F, P> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        *self += Self::from(rhs);
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> Sub<F> for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: F) -> Self {
        self - Self::from(rhs)
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> SubAssign<F> for PackedExt<F, P> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::from(rhs);
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> Mul<F> for PackedExt<F, P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        self * Self::from(rhs)
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> MulAssign<F> for PackedExt<F, P> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::from(rhs);
    }
}

impl<F: Field, P: Algebra<F::Packing> + Copy> Algebra<F> for PackedExt<F, P> {}
