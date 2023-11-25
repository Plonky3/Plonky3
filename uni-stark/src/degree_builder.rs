use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_air::{Air, AirBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::dense::RowMajorMatrix;

pub fn get_max_constraint_degree<F: Field, A: Air<DegreeBuilder<F>>>(air: &A) -> usize {
    let mut builder = DegreeBuilder::new(air.width());
    air.eval(&mut builder);
    builder.max_degree
}

/// The degree of a constraint, in the form `x n`, where `n` is a variable representing the trace length.
#[derive(Copy, Clone, Default, Debug)]
pub struct ConstraintDegree<F: Field> {
    degree: usize,
    _phantom: PhantomData<F>,
}

impl<F: Field> ConstraintDegree<F> {
    fn new(degree: usize) -> Self {
        Self {
            degree,
            _phantom: PhantomData,
        }
    }

    fn constant() -> Self {
        Self::new(0)
    }

    fn linear() -> Self {
        Self::new(1)
    }
}

impl<F: Field> AbstractField for ConstraintDegree<F> {
    type F = F;

    fn zero() -> Self {
        Self::constant()
    }

    fn one() -> Self {
        Self::constant()
    }

    fn two() -> Self {
        Self::constant()
    }

    fn neg_one() -> Self {
        Self::constant()
    }

    fn from_f(_f: Self::F) -> Self {
        Self::constant()
    }

    fn from_bool(_b: bool) -> Self {
        Self::constant()
    }

    fn from_canonical_u8(_n: u8) -> Self {
        Self::constant()
    }

    fn from_canonical_u16(_n: u16) -> Self {
        Self::constant()
    }

    fn from_canonical_u32(_n: u32) -> Self {
        Self::constant()
    }

    fn from_canonical_u64(_n: u64) -> Self {
        Self::constant()
    }

    fn from_canonical_usize(_n: usize) -> Self {
        Self::constant()
    }

    fn from_wrapped_u32(_n: u32) -> Self {
        Self::constant()
    }

    fn from_wrapped_u64(_n: u64) -> Self {
        Self::constant()
    }

    fn generator() -> Self {
        // TODO: Probably shouldn't be in AbstractField, only Field.
        todo!()
    }
}

impl<F: Field> From<F> for ConstraintDegree<F> {
    fn from(_value: F) -> Self {
        Self::constant()
    }
}

impl<F: Field> Add for ConstraintDegree<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.degree.max(rhs.degree))
    }
}

impl<F: Field> Add<F> for ConstraintDegree<F> {
    type Output = Self;

    fn add(self, _rhs: F) -> Self::Output {
        self
    }
}

impl<F: Field> AddAssign for ConstraintDegree<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> Sum for ConstraintDegree<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::constant())
    }
}

impl<F: Field> Sub for ConstraintDegree<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.degree.max(rhs.degree))
    }
}

impl<F: Field> Sub<F> for ConstraintDegree<F> {
    type Output = Self;

    fn sub(self, _rhs: F) -> Self::Output {
        self
    }
}

impl<F: Field> SubAssign for ConstraintDegree<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> Neg for ConstraintDegree<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}

impl<F: Field> Mul for ConstraintDegree<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Self::new(self.degree + rhs.degree)
    }
}

impl<F: Field> Mul<F> for ConstraintDegree<F> {
    type Output = Self;

    fn mul(self, _rhs: F) -> Self::Output {
        self
    }
}

impl<F: Field> MulAssign for ConstraintDegree<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> Product for ConstraintDegree<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::constant())
    }
}

pub struct DegreeBuilder<F: Field> {
    main: RowMajorMatrix<ConstraintDegree<F>>,
    max_degree: usize,
}

impl<F: Field> DegreeBuilder<F> {
    fn new(width: usize) -> Self {
        let values = (0..2)
            .flat_map(|_row_offset| (0..width).map(move |_column| ConstraintDegree::linear()))
            .collect();
        Self {
            main: RowMajorMatrix::new(values, width),
            max_degree: 0,
        }
    }
}

impl<F: Field> AirBuilder for DegreeBuilder<F> {
    type F = F;
    type Expr = ConstraintDegree<F>;
    type Var = ConstraintDegree<F>;
    type M = RowMajorMatrix<ConstraintDegree<F>>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        ConstraintDegree::linear()
    }

    fn is_last_row(&self) -> Self::Expr {
        ConstraintDegree::linear()
    }

    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        ConstraintDegree::constant()
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.max_degree = self.max_degree.max(x.into().degree);
    }
}
