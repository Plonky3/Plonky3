use crate::packed::PackedField;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use itertools::Itertools;

pub trait SmoothSubgroupField: Field {
    fn smooth_factors(&self) -> Vec<(u32, u32)>;
}

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

    /// The multiplicative inverse of this field element, if it exists.
    fn try_inverse(&self) -> Option<Self>;

    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        self.exp_u64(POWER)
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

pub trait FieldExtension<Base: Field>: Field {
    const D: usize;

    fn to_base_array(&self) -> [Base; Self::D];

    fn from_base_array(arr: [Base; Self::D]) -> Self;

    fn from_base(b: Base) -> Self;

    fn add_base(&self, x: Base) -> Self {
        *self + Self::from_base(x)
    }

    fn mul_base(&self, x: Base) -> Self {
        *self * Self::from_base(x)
    }
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

pub trait PrimeField: Field {
    const NEG_ONE: Self;
}

/// A `Field` with highly 2-adic multiplicative subgroups.
pub trait TwoAdicField: Field {
    const TWO_ADICITY: usize;
}

fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
