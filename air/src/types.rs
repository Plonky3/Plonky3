use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::Field;

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
    // TODO: Sum, Product?

    type Exp: Clone
        + From<Self::Var>
        + Add<Self::Var, Output = Self::Exp>
        + Add<Self::Exp, Output = Self::Exp>
        + Add<Self::F, Output = Self::Exp>
        + AddAssign<Self::Var>
        + AddAssign<Self::Exp>
        + AddAssign<Self::F>
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
        + MulAssign<Self::F>;
}

impl<F: Field> AirTypes for F {
    type F = Self;
    type Var = Self;
    type Exp = Self;
}
