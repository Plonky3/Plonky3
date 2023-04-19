use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::Field;
use p3_field::packed::PackedField;

pub trait AirTypes {
    type F: Field;

    type Var: 'static
        + Copy
        + Add<Self::Var, Output = Self::Exp>
        + Add<Self::Exp, Output = Self::Exp>
        + Add<Self::F, Output = Self::Exp>
        + Sub<Self::Var, Output = Self::Exp>
        + Sub<Self::Exp, Output = Self::Exp>
        + Sub<Self::F, Output = Self::Exp>
        + Neg<Output = Self::Exp>
        + Mul<Self::Var, Output = Self::Exp>
        + Mul<Self::Exp, Output = Self::Exp>
        + Mul<Self::F, Output = Self::Exp>;

    type Exp: Clone
        + From<Self::Var>
        + Add<Self::Var, Output = Self::Exp>
        + Add<Self::Exp, Output = Self::Exp>
        + Add<Self::F, Output = Self::Exp>
        + AddAssign<Self::Var>
        + AddAssign<Self::Exp>
        + AddAssign<Self::F>
        + Sum<Self::Var>
        + Sum<Self::Exp>
        + Sub<Self::Var, Output = Self::Exp>
        + Sub<Self::Exp, Output = Self::Exp>
        + Sub<Self::F, Output = Self::Exp>
        + SubAssign<Self::Var>
        + SubAssign<Self::Exp>
        + SubAssign<Self::F>
        + Neg<Output = Self::Exp>
        + Mul<Self::Var, Output = Self::Exp>
        + Mul<Self::Exp, Output = Self::Exp>
        + Mul<Self::F, Output = Self::Exp>
        + MulAssign<Self::Var>
        + MulAssign<Self::Exp>
        + MulAssign<Self::F>
        + Product<Self::Var>
        + Product<Self::Exp>;
}

impl<P: PackedField> AirTypes for P {
    type F = P::Scalar;
    type Var = P;
    type Exp = P;
}
