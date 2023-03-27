use crate::packed::PackedField;
use alloc::vec::Vec;
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign};
use itertools::Itertools;

pub trait SmoothSubgroupField: Field {
    fn smooth_factors(&self) -> Vec<(u32, u32)>;
}

/// A finite field;
pub trait Field:
    'static
    + Copy
    + Default
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sum
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Product
    + Send
    + Sync
    + Debug
    + Display
{
    type Packing: PackedField<Scalar = Self>;

    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;

    // fn add_arrs<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> [Self; N] {
    //     core::array::from_fn(|i| lhs[i] + rhs[i])
    // }
    //
    // fn add_slices(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
    //     lhs.iter().zip_eq(rhs).map(|(x, y)| *x + *y).collect()
    // }

    /// `x += y * s`, where `s` is a scalar.
    // TODO: Use PackedField
    fn add_scaled_slice_in_place(x: &mut [Self], y: &[Self], s: Self) {
        x.iter_mut()
            .zip_eq(y)
            .for_each(|(x_i, y_i)| *x_i += *y_i * s);
    }

    // fn mul_arrs<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> [Self; N] {
    //     core::array::from_fn(|i| lhs[i] * rhs[i])
    // }
    //
    // fn mul_slices(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
    //     lhs.iter().zip_eq(rhs).map(|(x, y)| *x * *y).collect()
    // }

    fn square(&self) -> Self {
        *self * *self
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

pub trait FieldExtension: Field {
    type Base: Field;
    const D: usize;

    fn to_base_array(&self) -> [Self::Base; Self::D];

    fn from_base_array(arr: [Self::Base; Self::D]) -> Self;

    fn from_base(b: Self::Base) -> Self;

    fn add_base(&self, x: Self::Base) -> Self {
        *self + Self::from_base(x)
    }

    fn mul_base(&self, x: Self::Base) -> Self {
        *self * Self::from_base(x)
    }
}

pub trait PrimeField: Field {
    const NEG_ONE: Self;
}

/// A `Field` with highly 2-adic multiplicative subgroups.
pub trait TwoAdicField: Field {
    const TWO_ADICITY: usize;
}

/// A `Field` with somewhat smooth multiplicative subgroups.
pub trait SemiSmoothField: Field {
    /// A list of "small" factors in the field's multiplicative subgroup, including duplicates.
    fn semi_smooth_factors() -> Vec<u32>;
}

fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
