use crate::packed::PackedField;
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use itertools::Itertools;

/// A finite field.
pub trait Field:
    'static
    + Copy
    + Default
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sum
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Product
    + Div<Self, Output = Self>
    + Eq
    + Send
    + Sync
    + Debug
    + Display
{
    type Packing: PackedField<Scalar = Self>;

    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const NEG_ONE: Self;

    /// The number of factors of two in this field's multiplicative group.
    const TWO_ADICITY: usize;

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// `x += y * s`, where `s` is a scalar.
    // TODO: Use PackedField
    fn add_scaled_slice_in_place(x: &mut [Self], y: &[Self], s: Self) {
        x.iter_mut()
            .zip_eq(y)
            .for_each(|(x_i, y_i)| *x_i += *y_i * s);
    }

    fn square(&self) -> Self {
        *self * *self
    }

    /// self * 2^exp
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        *self * Self::TWO.exp_u64(exp)
    }

    /// self / 2^exp
    fn div_2exp_u64(&self, exp: u64) -> Self {
        *self / Self::TWO.exp_u64(exp)
    }

    /// The multiplicative inverse of this field element, if it exists.
    fn try_inverse(&self) -> Option<Self>;

    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

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
}

pub trait FieldExtension<Base: Field>:
    Field + Add<Base, Output = Self> + Mul<Base, Output = Self>
{
    const D: usize;

    fn to_base_array(&self) -> [Base; Self::D];

    fn from_base_array(arr: [Base; Self::D]) -> Self;

    fn from_base(b: Base) -> Self;
}

impl<F: Field> FieldExtension<F> for F {
    const D: usize = 1;

    fn to_base_array(&self) -> [F; Self::D] {
        [*self]
    }

    fn from_base_array(arr: [F; Self::D]) -> Self {
        arr[0]
    }

    fn from_base(b: F) -> Self {
        b
    }
}

pub trait PrimeField: Field {}

fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
