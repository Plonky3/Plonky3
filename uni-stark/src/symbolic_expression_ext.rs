use crate::SymbolicExpression;
use crate::SymbolicVariable;
use alloc::rc::Rc;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field};

#[derive(Clone, Debug)]
pub struct SymbolicExpressionExt<EF>(pub SymbolicExpression<EF>)
where
    EF: Field;

impl<EF: Field> SymbolicExpressionExt<EF> {
    pub fn degree_multiple(&self) -> usize {
        self.0.degree_multiple()
    }
}

impl<EF: Field> Default for SymbolicExpressionExt<EF> {
    fn default() -> Self {
        Self(SymbolicExpression::zero())
    }
}

impl<EF: Field> From<SymbolicVariable<EF>> for SymbolicExpressionExt<EF> {
    fn from(value: SymbolicVariable<EF>) -> Self {
        Self(value.into())
    }
}

impl<F, EF> From<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn from(value: SymbolicExpression<F>) -> Self {
        Self::from_base(value)
    }
}

impl<EF> AbstractField for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    type F = EF;

    fn zero() -> Self {
        Self(SymbolicExpression::zero())
    }

    fn one() -> Self {
        Self(SymbolicExpression::one())
    }

    fn two() -> Self {
        Self(SymbolicExpression::two())
    }

    fn neg_one() -> Self {
        Self(SymbolicExpression::neg_one())
    }

    fn from_f(f: Self::F) -> Self {
        Self(SymbolicExpression::from_f(f))
    }

    fn from_bool(b: bool) -> Self {
        Self(SymbolicExpression::from_bool(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self(SymbolicExpression::from_canonical_u8(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self(SymbolicExpression::from_canonical_u16(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self(SymbolicExpression::from_canonical_u32(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self(SymbolicExpression::from_canonical_u64(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self(SymbolicExpression::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self(SymbolicExpression::from_wrapped_u32(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self(SymbolicExpression::from_wrapped_u64(n))
    }

    fn generator() -> Self {
        Self(SymbolicExpression::generator())
    }
}

fn map_rc<F, EF>(rc: Rc<SymbolicExpression<F>>) -> Rc<SymbolicExpression<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    Rc::new(SymbolicExpressionExt::from_base((*rc).clone()).0)
}

impl<F, EF> AbstractExtensionField<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    const D: usize = EF::D;

    fn from_base(b: SymbolicExpression<F>) -> Self {
        match b {
            SymbolicExpression::Variable(v) => Self(SymbolicExpression::Variable(v.to_extension())),
            SymbolicExpression::IsFirstRow => Self(SymbolicExpression::IsFirstRow),
            SymbolicExpression::IsLastRow => Self(SymbolicExpression::IsLastRow),
            SymbolicExpression::IsTransition => Self(SymbolicExpression::IsTransition),
            SymbolicExpression::Constant(c) => Self(SymbolicExpression::Constant(EF::from_base(c))),
            SymbolicExpression::Add {
                x,
                y,
                degree_multiple,
            } => Self(SymbolicExpression::Add {
                x: map_rc(x),
                y: map_rc(y),
                degree_multiple,
            }),
            SymbolicExpression::Sub {
                x,
                y,
                degree_multiple,
            } => Self(SymbolicExpression::Sub {
                x: map_rc(x),
                y: map_rc(y),
                degree_multiple,
            }),
            SymbolicExpression::Neg { x, degree_multiple } => Self(SymbolicExpression::Neg {
                x: map_rc(x),
                degree_multiple,
            }),
            SymbolicExpression::Mul {
                x,
                y,
                degree_multiple,
            } => Self(SymbolicExpression::Mul {
                x: map_rc(x),
                y: map_rc(y),
                degree_multiple,
            }),
        }
    }

    fn from_base_slice(bs: &[SymbolicExpression<F>]) -> Self {
        bs.iter()
            .enumerate()
            .map(|(i, b)| Self::from_base(b.clone()) * EF::monomial(i))
            .sum()
    }

    fn from_base_fn<FN: FnMut(usize) -> SymbolicExpression<F>>(mut f: FN) -> Self {
        (0..EF::D)
            .map(|i| Self::from_base(f(i)) * EF::monomial(i))
            .map(Self::from_base)
            .sum()
    }

    fn as_base_slice(&self) -> &[SymbolicExpression<F>] {
        unimplemented!("as_base_slice")
    }
}

impl<EF> Add for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<F, EF> Add<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Output = Self;

    fn add(self, rhs: SymbolicExpression<F>) -> Self::Output {
        self + Self::from_base(rhs)
    }
}

impl<EF> AddAssign for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<F, EF> AddAssign<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn add_assign(&mut self, rhs: SymbolicExpression<F>) {
        *self = self.clone() + rhs;
    }
}

impl<EF> Sum for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::zero())
    }
}

impl<EF> Sub for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F, EF> Sub<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Output = Self;

    fn sub(self, rhs: SymbolicExpression<F>) -> Self::Output {
        self - Self::from_base(rhs)
    }
}

impl<EF> SubAssign for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F, EF> SubAssign<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn sub_assign(&mut self, rhs: SymbolicExpression<F>) {
        *self = self.clone() - rhs;
    }
}

impl<EF> Neg for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<EF> Mul for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<EF: Field> Mul<EF> for SymbolicExpressionExt<EF> {
    type Output = Self;

    fn mul(self, rhs: EF) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<F, EF> Mul<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Output = Self;

    fn mul(self, rhs: SymbolicExpression<F>) -> Self::Output {
        self * Self::from_base(rhs)
    }
}

impl<EF> MulAssign for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, EF> MulAssign<SymbolicExpression<F>> for SymbolicExpressionExt<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn mul_assign(&mut self, rhs: SymbolicExpression<F>) {
        *self = self.clone() * rhs;
    }
}

impl<EF> Product for SymbolicExpressionExt<EF>
where
    EF: Field,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}
