use crate::AirTypes;
use alloc::rc::Rc;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::Field;

pub struct Symbolic<F: Field> {
    _phantom: PhantomData<F>,
}

/// A wrapper around `Rc` to get around the orphan rule.
#[derive(Clone)]
pub struct AirRc<T> {
    rc: Rc<T>,
}

impl<T> AirRc<T> {
    fn new(value: T) -> Self {
        Self { rc: Rc::new(value) }
    }
}

impl<T> Into<Rc<T>> for AirRc<T> {
    fn into(self) -> Rc<T> {
        self.rc
    }
}

impl<F: Field> AirTypes for Symbolic<F> {
    type F = F;
    type Var = SymbolicVar<F>;
    type Exp = AirRc<SymbolicExp<F>>;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SymbolicVar<F: Field> {
    TraceVar {
        row_offset: usize,
        column: usize,
        _phantom: PhantomData<F>,
    },
    // TODO: Preprocessed?
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicExp<F: Field> {
    Var(SymbolicVar<F>),
    Constant(F),
    Add(Rc<SymbolicExp<F>>, Rc<SymbolicExp<F>>),
    Sub(Rc<SymbolicExp<F>>, Rc<SymbolicExp<F>>),
    Neg(Rc<SymbolicExp<F>>),
    Mul(Rc<SymbolicExp<F>>, Rc<SymbolicExp<F>>),
}

impl<F: Field> From<SymbolicVar<F>> for SymbolicExp<F> {
    fn from(var: SymbolicVar<F>) -> Self {
        Self::Var(var)
    }
}

impl<F: Field> From<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    fn from(var: SymbolicVar<F>) -> Self {
        AirRc::new(SymbolicExp::Var(var))
    }
}

impl<F: Field> From<F> for SymbolicExp<F> {
    fn from(value: F) -> Self {
        Self::Constant(value)
    }
}

impl<F: Field> From<F> for AirRc<SymbolicExp<F>> {
    fn from(value: F) -> Self {
        AirRc::new(SymbolicExp::Constant(value))
    }
}

impl<F: Field> Add<Self> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        AirRc::new(SymbolicExp::Add(self.into(), rhs.into()))
    }
}

impl<F: Field> Add<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn add(self, rhs: SymbolicVar<F>) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl<F: Field> Add<F> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl<F: Field> AddAssign<Self> for AirRc<SymbolicExp<F>> {
    fn add_assign(&mut self, rhs: Self) {
        *self = AirRc::new(SymbolicExp::Add(self.clone().into(), rhs.into()));
    }
}

impl<F: Field> AddAssign<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    fn add_assign(&mut self, rhs: SymbolicVar<F>) {
        *self += Self::from(rhs);
    }
}

impl<F: Field> AddAssign<F> for AirRc<SymbolicExp<F>> {
    fn add_assign(&mut self, rhs: F) {
        *self += Self::from(rhs);
    }
}

impl<F: Field> Sum for AirRc<SymbolicExp<F>> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y)
            .unwrap_or(AirRc::new(SymbolicExp::Constant(F::ZERO)))
    }
}

impl<F: Field> Sub<Self> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        AirRc::new(SymbolicExp::Sub(self.into(), rhs.into()))
    }
}

impl<F: Field> Sub<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn sub(self, rhs: SymbolicVar<F>) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl<F: Field> Sub<F> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl<F: Field> SubAssign<Self> for AirRc<SymbolicExp<F>> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = AirRc::new(SymbolicExp::Sub(self.clone().into(), rhs.into()));
    }
}

impl<F: Field> SubAssign<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    fn sub_assign(&mut self, rhs: SymbolicVar<F>) {
        *self -= Self::from(rhs);
    }
}

impl<F: Field> SubAssign<F> for AirRc<SymbolicExp<F>> {
    fn sub_assign(&mut self, rhs: F) {
        *self -= Self::from(rhs);
    }
}

impl<F: Field> Neg for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        AirRc::new(SymbolicExp::Neg(self.into()))
    }
}

impl<F: Field> Mul<Self> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        AirRc::new(SymbolicExp::Mul(self.into(), rhs.into()))
    }
}

impl<F: Field> Mul<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn mul(self, rhs: SymbolicVar<F>) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl<F: Field> Mul<F> for AirRc<SymbolicExp<F>> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl<F: Field> MulAssign<Self> for AirRc<SymbolicExp<F>> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = AirRc::new(SymbolicExp::Mul(self.clone().into(), rhs.into()));
    }
}

impl<F: Field> MulAssign<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    fn mul_assign(&mut self, rhs: SymbolicVar<F>) {
        *self *= Self::from(rhs);
    }
}

impl<F: Field> MulAssign<F> for AirRc<SymbolicExp<F>> {
    fn mul_assign(&mut self, rhs: F) {
        *self *= Self::from(rhs);
    }
}

impl<F: Field> Product for AirRc<SymbolicExp<F>> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y)
            .unwrap_or(AirRc::new(SymbolicExp::Constant(F::ONE)))
    }
}

impl<F: Field> Add<Self> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn add(self, rhs: Self) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) + rhs
    }
}

impl<F: Field> Add<AirRc<SymbolicExp<F>>> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn add(self, rhs: AirRc<SymbolicExp<F>>) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) + rhs
    }
}

impl<F: Field> Add<F> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn add(self, rhs: F) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) + rhs
    }
}

impl<F: Field> Sum<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    fn sum<I: Iterator<Item = SymbolicVar<F>>>(iter: I) -> Self {
        iter.map(|x| AirRc::new(SymbolicExp::from(x))).sum()
    }
}

impl<F: Field> Sub<Self> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn sub(self, rhs: Self) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) - rhs
    }
}

impl<F: Field> Sub<AirRc<SymbolicExp<F>>> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn sub(self, rhs: AirRc<SymbolicExp<F>>) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) - rhs
    }
}

impl<F: Field> Sub<F> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn sub(self, rhs: F) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) - rhs
    }
}

impl<F: Field> Neg for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn neg(self) -> Self::Output {
        -AirRc::new(SymbolicExp::from(self))
    }
}

impl<F: Field> Mul<Self> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn mul(self, rhs: Self) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) * rhs
    }
}

impl<F: Field> Mul<AirRc<SymbolicExp<F>>> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn mul(self, rhs: AirRc<SymbolicExp<F>>) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) * rhs
    }
}

impl<F: Field> Mul<F> for SymbolicVar<F> {
    type Output = AirRc<SymbolicExp<F>>;

    fn mul(self, rhs: F) -> Self::Output {
        AirRc::new(SymbolicExp::from(self)) * rhs
    }
}

impl<F: Field> Product<SymbolicVar<F>> for AirRc<SymbolicExp<F>> {
    fn product<I: Iterator<Item = SymbolicVar<F>>>(iter: I) -> Self {
        iter.map(|x| AirRc::new(SymbolicExp::from(x))).product()
    }
}
