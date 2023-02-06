use alloc::vec::Vec;
use core::ops::{Add, AddAssign, Mul, MulAssign};
use itertools::Itertools;

pub trait Field:
    'static
    + Copy
    + Default
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;

    fn add_arrs<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> [Self; N] {
        core::array::from_fn(|i| lhs[i] + rhs[i])
    }

    fn add_slices(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        lhs.iter().zip_eq(rhs).map(|(x, y)| *x + *y).collect()
    }

    /// `x += y * s`, where `s` is a scalar.
    fn add_scaled_slice_in_place(x: &mut [Self], y: &[Self], s: Self) {
        x.iter_mut()
            .zip_eq(y)
            .for_each(|(x_i, y_i)| *x_i += *y_i * s);
    }

    fn mul_arrs<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> [Self; N] {
        core::array::from_fn(|i| lhs[i] * rhs[i])
    }

    fn mul_slices(lhs: &[Self], rhs: &[Self]) -> Vec<Self> {
        lhs.iter().zip_eq(rhs).map(|(x, y)| *x * *y).collect()
    }
}

// TODO: Instead have batch methods on Field?
pub trait FieldVec<const N: usize, F: Field>:
    'static + Copy + Add<Self, Output = Self> + Mul<Self, Output = Self> + Mul<F, Output = Self>
{
}
