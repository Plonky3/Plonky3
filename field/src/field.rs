use crate::packed::PackedField;
use core::fmt::{Debug, Display};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use itertools::Itertools;

/// A finite field.
///
/// This `Field` trait represents both a field with a specified
/// extension `Field`/`DistinguishedSubfield` of degree `EXT_DEGREE`,
/// *and* elements of the extension field.
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
    // This name is a bit long; could probably just call it `Subfield`.
    type DistinguishedSubfield: Field;
    const EXT_DEGREE: usize;

    // Lift a subfield element to this Field; could also be called
    // `from_subfield()`.
    fn lift(x: Self::DistinguishedSubfield) -> Self;

    // Return self as an element of the DistinguishedSubfield if it is
    // indeed an element of the subfield, otherwise None.
    fn try_lower(&self) -> Option<Self::DistinguishedSubfield>;

    // Considering Field as a dimension EXT_DEGREE vector space over
    // DistinguishedSubfield, an element of Field has a
    // *NON-CANONICAL* representation as a tuple of EXT_DEGREE
    // elements of DistinguishedSubfield.
    //
    // The two functions below could provide facility to `map` or
    // `fold` over the elements of such a vector; they should be
    // *avoided* where possible and used with caution if necessary. The
    // intention is that they would replace existing cases where
    // `to_base_array()` is called.
    //
    // map should return an iterator I guess, since can't return a slice
    /*
    fn map_components<Fn>(&self, f: Fn) -> &[Self::DistinguishedSubfield];
    fn fold_components<Fn>(
        &self,
        f: Fn,
        init: Self::DistinguishedSubfield,
    ) -> Self::DistinguishedSubfield;
    */

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

// TODO: Would be good to be able to share these specialisations
// between Goldilocks, Mersenne31, etc.
/*
pub trait PrimeField: Field {
    type DistinguishedSubfield = Self;
    const EXT_DEGREE: usize = 1;

    fn lift(x: Self::DistinguishedSubfield) -> Self { x }
    fn try_lower(&self) -> Option<Self::DistinguishedSubfield> { Some(*self) }
}
*/

// This obviously belongs in a utils module.
fn bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
