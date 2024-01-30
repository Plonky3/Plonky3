use core::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{AbstractExtensionField, AbstractField, ExtensionField, Field};

/// The restriction of scalars from a field `EF` to a subfield `F`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Hash)]
pub struct Res<F, EF>(EF, PhantomData<F>);

impl<F: Field, EF: AbstractExtensionField<F>> Res<F, EF> {
    /// Returns the underlying field element.
    pub fn into_inner(self) -> EF {
        self.0
    }

    pub const fn from_inner(e: EF) -> Self {
        Self(e, PhantomData)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> AbstractField for Res<F, EF> {
    type F = F;

    fn from_f(f: F) -> Self {
        Self::from_inner(EF::from_base(f))
    }

    fn zero() -> Self {
        Self::from_inner(EF::zero())
    }

    fn one() -> Self {
        Self::from_inner(EF::one())
    }

    fn two() -> Self {
        Self::from_inner(EF::two())
    }

    fn from_bool(b: bool) -> Self {
        Self::from_inner(EF::from_bool(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::from_inner(EF::from_canonical_u8(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::from_inner(EF::from_canonical_u16(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::from_inner(EF::from_canonical_u32(n).into())
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::from_inner(EF::from_canonical_u64(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::from_inner(EF::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::from_inner(EF::from_wrapped_u32(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::from_inner(EF::from_wrapped_u64(n))
    }

    fn neg_one() -> Self {
        Self::from_inner(EF::neg_one())
    }

    fn generator() -> Self {
        Self::from_inner(EF::generator())
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> AbstractExtensionField<F> for Res<F, EF> {
    const D: usize = EF::D;

    fn from_base(b: F) -> Self {
        Self::from_inner(EF::from_base(b))
    }

    fn from_base_fn<Fun: FnMut(usize) -> F>(mut f: Fun) -> Self {
        Self::from_inner(EF::from_base_fn(|i| f(i)))
    }

    fn from_base_slice(bs: &[F]) -> Self {
        Self::from_inner(EF::from_base_slice(bs))
    }

    fn as_base_slice(&self) -> &[F] {
        self.0.as_base_slice()
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> From<F> for Res<F, EF> {
    fn from(f: F) -> Self {
        Res(EF::from_base(f), PhantomData)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Add for Res<F, EF> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::from_inner(self.0 + other.0)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> AddAssign for Res<F, EF> {
    fn add_assign(&mut self, other: Self) {
        self.0.add_assign(other.0);
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Mul for Res<F, EF> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::from_inner(self.0 * other.0)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> MulAssign for Res<F, EF> {
    fn mul_assign(&mut self, other: Self) {
        self.0.mul_assign(other.0);
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> MulAssign<F> for Res<F, EF> {
    fn mul_assign(&mut self, other: F) {
        self.0.mul_assign(other);
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Sub for Res<F, EF> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::from_inner(self.0 - other.0)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Neg for Res<F, EF> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_inner(-self.0)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> SubAssign for Res<F, EF> {
    fn sub_assign(&mut self, other: Self) {
        self.0.sub_assign(other.0);
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Add<F> for Res<F, EF> {
    type Output = Self;

    fn add(self, other: F) -> Self {
        Self::from_inner(self.0 + other)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> AddAssign<F> for Res<F, EF> {
    fn add_assign(&mut self, other: F) {
        self.0.add_assign(other);
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Sub<F> for Res<F, EF> {
    type Output = Self;

    fn sub(self, other: F) -> Self {
        Self::from_inner(self.0 - other)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> SubAssign<F> for Res<F, EF> {
    fn sub_assign(&mut self, other: F) {
        self.0.sub_assign(other);
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> Mul<F> for Res<F, EF> {
    type Output = Self;

    fn mul(self, other: F) -> Self {
        Self::from_inner(self.0 * other)
    }
}

impl<F: Field, EF: ExtensionField<F>> Div for Res<F, EF> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self::from_inner(self.0 / other.0)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> core::iter::Product for Res<F, EF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<F: Field, EF: AbstractExtensionField<F>> core::iter::Sum for Res<F, EF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}
