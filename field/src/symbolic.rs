use crate::field::{Field, FieldLike};
use alloc::rc::Rc;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Debug)]
pub enum SymbolicField<F: Field, V: Clone + Debug> {
    Variable(V),
    Constant(F),
    Add(Rc<Self>, Rc<Self>),
    Sub(Rc<Self>, Rc<Self>),
    Neg(Rc<Self>),
    Mul(Rc<Self>, Rc<Self>),
}

impl<F: Field, V: Clone + Debug> From<F> for SymbolicField<F, V> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field, V: Clone + Debug> FieldLike<F> for SymbolicField<F, V> {
    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);
}

impl<F: Field, V: Clone + Debug> Add for SymbolicField<F, V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Add(Rc::new(self), Rc::new(rhs))
    }
}

impl<F: Field, V: Clone + Debug> Add<F> for SymbolicField<F, V> {
    type Output = Self;

    fn add(self, rhs: F) -> Self {
        self + Self::from(rhs)
    }
}

impl<F: Field, V: Clone + Debug> AddAssign for SymbolicField<F, V> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F: Field, V: Clone + Debug> AddAssign<F> for SymbolicField<F, V> {
    fn add_assign(&mut self, rhs: F) {
        *self += Self::from(rhs);
    }
}

impl<F: Field, V: Clone + Debug> Sum for SymbolicField<F, V> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::Constant(F::ZERO))
    }
}

impl<F: Field, V: Clone + Debug> Sub for SymbolicField<F, V> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::Sub(Rc::new(self), Rc::new(rhs))
    }
}

impl<F: Field, V: Clone + Debug> Sub<F> for SymbolicField<F, V> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self {
        self - Self::from(rhs)
    }
}

impl<F: Field, V: Clone + Debug> SubAssign for SymbolicField<F, V> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F: Field, V: Clone + Debug> SubAssign<F> for SymbolicField<F, V> {
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::from(rhs);
    }
}

impl<F: Field, V: Clone + Debug> Neg for SymbolicField<F, V> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::Neg(Rc::new(self))
    }
}

impl<F: Field, V: Clone + Debug> Mul for SymbolicField<F, V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Rc::new(self), Rc::new(rhs))
    }
}

impl<F: Field, V: Clone + Debug> Mul<F> for SymbolicField<F, V> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        self * Self::from(rhs)
    }
}

impl<F: Field, V: Clone + Debug> MulAssign for SymbolicField<F, V> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F: Field, V: Clone + Debug> MulAssign<F> for SymbolicField<F, V> {
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::from(rhs);
    }
}

impl<F: Field, V: Clone + Debug> Product for SymbolicField<F, V> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::Constant(F::ONE))
    }
}
