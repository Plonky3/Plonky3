use crate::field::{AbstractField, Field};
use alloc::rc::Rc;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Debug)]
pub enum SymbolicField<F: Field, Var: Clone + Debug> {
    Variable(Var),
    Constant(F),
    Add(Rc<Self>, Rc<Self>),
    Sub(Rc<Self>, Rc<Self>),
    Neg(Rc<Self>),
    Mul(Rc<Self>, Rc<Self>),
}

impl<F: Field, Var: Clone + Debug> From<F> for SymbolicField<F, Var> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field, Var: Clone + Debug> AbstractField<F> for SymbolicField<F, Var> {
    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);
    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Self::Constant(F::MULTIPLICATIVE_GROUP_GENERATOR);
}

impl<F: Field, Var: Clone + Debug> Add for SymbolicField<F, Var> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Add(Rc::new(self), Rc::new(rhs))
    }
}

impl<F: Field, Var: Clone + Debug> Add<F> for SymbolicField<F, Var> {
    type Output = Self;

    fn add(self, rhs: F) -> Self {
        self + Self::from(rhs)
    }
}

impl<F: Field, Var: Clone + Debug> AddAssign for SymbolicField<F, Var> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field, Var: Clone + Debug> AddAssign<F> for SymbolicField<F, Var> {
    fn add_assign(&mut self, rhs: F) {
        *self += Self::from(rhs);
    }
}

impl<F: Field, Var: Clone + Debug> Sum for SymbolicField<F, Var> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::Constant(F::ZERO))
    }
}

impl<F: Field, Var: Clone + Debug> Sub for SymbolicField<F, Var> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::Sub(Rc::new(self), Rc::new(rhs))
    }
}

impl<F: Field, Var: Clone + Debug> Sub<F> for SymbolicField<F, Var> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self {
        self - Self::from(rhs)
    }
}

impl<F: Field, Var: Clone + Debug> SubAssign for SymbolicField<F, Var> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field, Var: Clone + Debug> SubAssign<F> for SymbolicField<F, Var> {
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::from(rhs);
    }
}

impl<F: Field, Var: Clone + Debug> Neg for SymbolicField<F, Var> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::Neg(Rc::new(self))
    }
}

impl<F: Field, Var: Clone + Debug> Mul for SymbolicField<F, Var> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Rc::new(self), Rc::new(rhs))
    }
}

impl<F: Field, Var: Clone + Debug> Mul<F> for SymbolicField<F, Var> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        self * Self::from(rhs)
    }
}

impl<F: Field, Var: Clone + Debug> MulAssign for SymbolicField<F, Var> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F: Field, Var: Clone + Debug> MulAssign<F> for SymbolicField<F, Var> {
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::from(rhs);
    }
}

impl<F: Field, Var: Clone + Debug> Product for SymbolicField<F, Var> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::Constant(F::ONE))
    }
}
