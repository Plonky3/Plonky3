use alloc::format;
use alloc::string::String;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use super::HasFrobenuis;
use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{field_to_array, AbstractExtensionField, AbstractField, ExtensionField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BinomialExtensionField<F: BinomiallyExtendable<D>, const D: usize> {
    value: [F; D],
}

impl<F: BinomiallyExtendable<D>, const D: usize> Default for BinomialExtensionField<F, D> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> From<F> for BinomialExtensionField<F, D> {
    fn from(x: F) -> Self {
        Self {
            value: field_to_array::<F, D>(x),
        }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> ExtensionField<F>
    for BinomialExtensionField<F, D>
{
}

impl<F: BinomiallyExtendable<D>, const D: usize> HasFrobenuis<F> for BinomialExtensionField<F, D> {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(n^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= D {
            // x |-> x^(n^D) is the identity, so x^(n^count) ==
            // x^(n^(count % D))
            return self.repeated_frobenius(count % D);
        }
        let arr: &[F] = self.as_base_slice();

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((n-1)/D)
        let mut z0 = F::DTH_ROOT;
        for _ in 1..count {
            z0 *= F::DTH_ROOT;
        }

        let mut res = [F::ZERO; D];
        for (i, z) in z0.powers().take(D).enumerate() {
            res[i] = arr[i] * z;
        }

        Self::from_base_slice(&res)
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> AbstractField for BinomialExtensionField<F, D> {
    type F = Self;

    const ZERO: Self = Self {
        value: [F::ZERO; D],
    };
    const ONE: Self = Self {
        value: field_to_array::<F, D>(F::ONE),
    };
    const TWO: Self = Self {
        value: field_to_array::<F, D>(F::TWO),
    };
    const NEG_ONE: Self = Self {
        value: field_to_array::<F, D>(F::NEG_ONE),
    };

    fn from_bool(b: bool) -> Self {
        F::from_bool(b).into()
    }

    fn from_canonical_u8(n: u8) -> Self {
        F::from_canonical_u8(n).into()
    }

    fn from_canonical_u16(n: u16) -> Self {
        F::from_canonical_u16(n).into()
    }

    fn from_canonical_u32(n: u32) -> Self {
        F::from_canonical_u32(n).into()
    }

    /// Convert from `u64`. Undefined behavior if the input is outside the canonical range.
    fn from_canonical_u64(n: u64) -> Self {
        F::from_canonical_u64(n).into()
    }

    /// Convert from `usize`. Undefined behavior if the input is outside the canonical range.
    fn from_canonical_usize(n: usize) -> Self {
        F::from_canonical_usize(n).into()
    }

    fn from_wrapped_u32(n: u32) -> Self {
        F::from_wrapped_u32(n).into()
    }

    fn from_wrapped_u64(n: u64) -> Self {
        F::from_wrapped_u64(n).into()
    }

    fn generator() -> Self {
        Self {
            value: F::ext_multiplicative_group_generator(),
        }
    }

    #[inline(always)]
    fn square(&self) -> Self {
        // Specialising mul reduces the computation of c1 from 2 muls
        // and one add to one mul and a shift
        *self * *self
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Field for BinomialExtensionField<F, D> {
    type Packing = Self;
    // Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Writing 'a' for self, we need to compute a^(r-1):
        // r = n^D-1/n-1 = n^(D-1)+n^(D-2)+...+n
        let mut f = Self::ONE;
        for _ in 1..D {
            f = (f * *self).frobenius();
        }

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product.

        let a = self.value;
        let b = f.value;
        let mut g = F::ZERO;
        for i in 1..D {
            g += a[i] * b[D - i];
        }
        g *= F::W;
        g += a[0] * b[0];
        debug_assert_eq!(Self::from(g), *self * f);

        Some(f * g.inverse())
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Display for BinomialExtensionField<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let linear_part = format!("{} + {}*X", self.value[0], self.value[1]); // ok, since D >= 2
        let nonlin_part: String = self.value[2..]
            .iter()
            .zip(2..)
            .map(|(x, i)| format!(" + {x}*X^{i}"))
            .collect();
        write!(f, "{}", linear_part + &nonlin_part)
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Debug for BinomialExtensionField<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Neg for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(F::neg),
        }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Add for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self { value: res }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Add<F> for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] += rhs;
        Self { value: res }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> AddAssign for BinomialExtensionField<F, D> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> AddAssign<F> for BinomialExtensionField<F, D> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Sum for BinomialExtensionField<F, D> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Sub for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r -= rhs_val;
        }
        Self { value: res }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Sub<F> for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> SubAssign for BinomialExtensionField<F, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> SubAssign<F> for BinomialExtensionField<F, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Mul for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;

        let mut res = [F::ZERO; D];

        for i in 0..D {
            for j in 0..D {
                if i + j >= D {
                    res[i + j - D] += F::W * a[i] * b[j];
                } else {
                    res[i + j] += a[i] * b[j];
                }
            }
        }

        Self { value: res }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Mul<F> for BinomialExtensionField<F, D> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self {
            value: self.value.map(|x| x * rhs),
        }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Product for BinomialExtensionField<F, D> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Div for BinomialExtensionField<F, D> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> DivAssign for BinomialExtensionField<F, D> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> MulAssign for BinomialExtensionField<F, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> MulAssign<F> for BinomialExtensionField<F, D> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> AbstractExtensionField<F>
    for BinomialExtensionField<F, D>
{
    const D: usize = F::D;

    fn from_base(b: F) -> Self {
        Self::from(b)
    }

    fn from_base_slice(bs: &[F]) -> Self {
        Self {
            value: bs.try_into().expect("slice has wrong length"),
        }
    }

    fn as_base_slice(&self) -> &[F] {
        &self.value
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Distribution<BinomialExtensionField<F, D>>
    for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BinomialExtensionField<F, D> {
        let mut res = [F::ZERO; D];
        for r in res.iter_mut() {
            *r = Standard.sample(rng);
        }
        BinomialExtensionField::<F, D>::from_base_slice(&res)
    }
}
