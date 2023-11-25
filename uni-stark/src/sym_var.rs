use core::marker::PhantomData;
use core::ops::{Add, Mul, Sub};

use p3_field::{Field, SymbolicField};

#[derive(Copy, Clone, Debug)]
pub struct BasicSymVar<F: Field> {
    pub row_offset: usize,
    pub column: usize,
    pub(crate) _phantom: PhantomData<F>,
}

impl<F: Field> From<BasicSymVar<F>> for SymbolicField<F, BasicSymVar<F>> {
    fn from(value: BasicSymVar<F>) -> Self {
        SymbolicField::Variable(value)
    }
}

impl<F: Field> Add for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn add(self, rhs: Self) -> Self::Output {
        SymbolicField::from(self) + SymbolicField::from(rhs)
    }
}

impl<F: Field> Add<F> for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn add(self, rhs: F) -> Self::Output {
        SymbolicField::from(self) + SymbolicField::from(rhs)
    }
}

impl<F: Field> Add<SymbolicField<F, Self>> for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn add(self, rhs: SymbolicField<F, Self>) -> Self::Output {
        SymbolicField::from(self) + rhs
    }
}

impl<F: Field> Add<BasicSymVar<F>> for SymbolicField<F, BasicSymVar<F>> {
    type Output = Self;

    fn add(self, rhs: BasicSymVar<F>) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl<F: Field> Sub for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        SymbolicField::from(self) - SymbolicField::from(rhs)
    }
}

impl<F: Field> Sub<F> for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn sub(self, rhs: F) -> Self::Output {
        SymbolicField::from(self) - SymbolicField::from(rhs)
    }
}

impl<F: Field> Sub<SymbolicField<F, Self>> for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn sub(self, rhs: SymbolicField<F, Self>) -> Self::Output {
        SymbolicField::from(self) - rhs
    }
}

impl<F: Field> Sub<BasicSymVar<F>> for SymbolicField<F, BasicSymVar<F>> {
    type Output = Self;

    fn sub(self, rhs: BasicSymVar<F>) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl<F: Field> Mul for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        SymbolicField::from(self) * SymbolicField::from(rhs)
    }
}

impl<F: Field> Mul<F> for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn mul(self, rhs: F) -> Self::Output {
        SymbolicField::from(self) * SymbolicField::from(rhs)
    }
}

impl<F: Field> Mul<SymbolicField<F, Self>> for BasicSymVar<F> {
    type Output = SymbolicField<F, Self>;

    fn mul(self, rhs: SymbolicField<F, Self>) -> Self::Output {
        SymbolicField::from(self) * rhs
    }
}

impl<F: Field> Mul<BasicSymVar<F>> for SymbolicField<F, BasicSymVar<F>> {
    type Output = Self;

    fn mul(self, rhs: BasicSymVar<F>) -> Self::Output {
        self * Self::from(rhs)
    }
}
