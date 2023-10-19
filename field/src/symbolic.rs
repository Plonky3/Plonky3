use alloc::rc::Rc;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::{AbstractField, Field};

#[derive(Clone, Debug)]
pub enum SymbolicField<F: Field, Var> {
    Variable(Var),
    Constant(F),
    Add(Rc<Self>, Rc<Self>),
    Sub(Rc<Self>, Rc<Self>),
    Neg(Rc<Self>),
    Mul(Rc<Self>, Rc<Self>),
}

impl<F: Field, Var> Default for SymbolicField<F, Var> {
    fn default() -> Self {
        Self::Constant(F::zero())
    }
}

impl<F: Field, Var> From<F> for SymbolicField<F, Var> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field, Var: Clone + Debug> AbstractField for SymbolicField<F, Var> {
    type F = F;

    fn zero() -> Self {
        Self::Constant(F::zero())
    }
    fn one() -> Self {
        Self::Constant(F::one())
    }
    fn two() -> Self {
        Self::Constant(F::two())
    }
    fn neg_one() -> Self {
        Self::Constant(F::neg_one())
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f.into()
    }

    fn from_bool(b: bool) -> Self {
        Self::Constant(F::from_bool(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::Constant(F::from_canonical_u8(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::Constant(F::from_canonical_u16(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::Constant(F::from_canonical_u32(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::Constant(F::from_canonical_u64(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::Constant(F::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::Constant(F::from_wrapped_u32(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::Constant(F::from_wrapped_u64(n))
    }

    fn generator() -> Self {
        Self::Constant(F::generator())
    }
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
        iter.reduce(|x, y| x + y).unwrap_or(Self::zero())
    }
}

impl<F: Field, Var: Clone + Debug> Sum<F> for SymbolicField<F, Var> {
    fn sum<I: Iterator<Item = F>>(iter: I) -> Self {
        iter.map(|x| Self::from(x)).sum()
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
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl<F: Field, Var: Clone + Debug> Product<F> for SymbolicField<F, Var> {
    fn product<I: Iterator<Item = F>>(iter: I) -> Self {
        iter.map(|x| Self::from(x)).product()
    }
}
