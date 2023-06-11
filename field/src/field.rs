use crate::packed::PackedField;
use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;
use itertools::Itertools;

/// A generalization of `Field` which permits things like
/// - an actual field element
/// - a symbolic expression which would evaluate to a field element
/// - a vector of field elements
pub trait AbstractField:
    Sized
    + Default
    + Clone
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + Sum
    + Product
    + Debug
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;

    fn multiplicative_group_generator() -> Self;

    #[must_use]
    fn square(&self) -> Self {
        self.clone() * self.clone()
    }
}

/// An `AbstractField` which abstracts the given field `F`.
pub trait AbstractionOf<F: Field>:
    AbstractField
    + From<F>
    + Add<F, Output = Self>
    + AddAssign<F>
    + Sub<F, Output = Self>
    + SubAssign<F>
    + Mul<F, Output = Self>
    + MulAssign<F>
    + Sum<F>
    + Product<F>
{
}

impl<F: Field> AbstractionOf<F> for F {}

/// An element of a finite field.
pub trait Field:
    AbstractField + 'static + Copy + Div<Self, Output = Self> + Eq + Hash + Send + Sync + Display
{
    type Packing: PackedField<Scalar = Self>;

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// `x += y * s`, where `s` is a scalar.
    // TODO: Use PackedField
    // TODO: Move out of Field?
    fn add_scaled_slice_in_place(x: &mut [Self], y: &[Self], s: Self) {
        x.iter_mut()
            .zip_eq(y)
            .for_each(|(x_i, y_i)| *x_i += *y_i * s);
    }

    /// self * 2^exp
    #[must_use]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        *self * Self::TWO.exp_u64(exp)
    }

    /// self / 2^exp
    #[must_use]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        *self / Self::TWO.exp_u64(exp)
    }

    /// The multiplicative inverse of this field element, if it exists.
    #[must_use]
    fn try_inverse(&self) -> Option<Self>;

    #[must_use]
    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    #[must_use]
    fn exp_u64(&self, power: u64) -> Self {
        let mut current = *self;
        let mut product = Self::ONE;

        for j in 0..bits_u64(power) {
            if (power >> j & 1) != 0 {
                product *= current;
            }
            current = current.square();
        }
        product
    }

    #[must_use]
    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res = res.square();
        }
        res
    }

    #[must_use]
    fn powers(&self) -> Powers<Self> {
        Powers {
            base: *self,
            current: Self::ONE,
        }
    }
}

pub trait PrimeField: Field + Ord {
    fn from_canonical_u8(n: u8) -> Self;
    fn from_canonical_u16(n: u8) -> Self;
    fn from_canonical_u32(n: u32) -> Self;
    fn from_canonical_u64(n: u64) -> Self;
    fn from_canonical_usize(n: usize) -> Self;

    fn from_wrapped_u32(n: u32) -> Self;
    fn from_wrapped_u64(n: u64) -> Self;
}

/// A prime field of order less than `2^64`.
pub trait PrimeField64: PrimeField {
    const ORDER_U64: u64;

    fn as_canonical_u64(&self) -> u64;
}

/// A prime field of order less than `2^32`.
pub trait PrimeField32: PrimeField64 {
    const ORDER_U32: u32;

    fn as_canonical_u32(&self) -> u32;
}

impl<F: PrimeField32> PrimeField64 for F {
    const ORDER_U64: u64 = <F as PrimeField32>::ORDER_U32 as u64;

    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }
}

pub trait AbstractExtensionField<Base>:
    AbstractField
    + Add<Base, Output = Self>
    + AddAssign<Base>
    + Sub<Base, Output = Self>
    + SubAssign<Base>
    + Mul<Base, Output = Self>
    + MulAssign<Base>
{
    const D: usize;

    fn from_base(b: Base) -> Self;

    fn from_base_slice(bs: &[Base]) -> Self;

    fn as_base_slice(&self) -> &[Base];
}

pub trait ExtensionField<Base: Field>: Field + AbstractExtensionField<Base> {}

impl<Base: Field, Ext: Field + AbstractExtensionField<Base>> ExtensionField<Base> for Ext {}

impl<F: Field> AbstractExtensionField<F> for F {
    const D: usize = 1;

    fn from_base(b: F) -> Self {
        b
    }

    fn from_base_slice(bs: &[F]) -> Self {
        assert_eq!(bs.len(), 1);
        bs[0]
    }

    fn as_base_slice(&self) -> &[F] {
        slice::from_ref(self)
    }
}

/// A field which supplies information like the two-adicity of its multiplicative group, and methods
/// for obtaining two-adic roots of unity.
pub trait TwoAdicField: Field {
    /// The number of factors of two in this field's multiplicative group.
    const TWO_ADICITY: usize;

    /// Generator of a multiplicative subgroup of order `2^TWO_ADICITY`.
    fn power_of_two_generator() -> Self;

    /// Returns a primitive root of order `2^bits`.
    #[must_use]
    fn primitive_root_of_unity(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        let base = Self::power_of_two_generator();
        base.mul_2exp_u64((Self::TWO_ADICITY - bits) as u64)
    }
}

/// An iterator over the powers of a certain base element `b`: `b^0, b^1, b^2, ...`.
#[derive(Clone)]
pub struct Powers<F> {
    pub base: F,
    pub current: F,
}

impl<F: MulAssign + Clone> Iterator for Powers<F> {
    type Item = F;

    fn next(&mut self) -> Option<F> {
        let result = self.current.clone();
        self.current *= self.base.clone();
        Some(result)
    }
}

fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
