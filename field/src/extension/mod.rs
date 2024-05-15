use core::{
    array,
    fmt::{self, Debug, Display},
    hash::{self, Hash},
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use alloc::{format, string::ToString};
use itertools::Itertools;
use num_bigint::BigUint;
use rand::distributions::{Distribution, Standard};
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{AbstractField, Field, Packable};

mod binomial;
mod complex;

pub use binomial::*;
pub use complex::*;

pub trait AbstractFieldArray<AF: AbstractField>:
    AsRef<[AF]> + AsMut<[AF]> + Sized + Clone + Debug
{
    fn from_fn(f: impl FnMut(usize) -> AF) -> Self;
}
impl<AF: AbstractField, const N: usize> AbstractFieldArray<AF> for [AF; N] {
    fn from_fn(f: impl FnMut(usize) -> AF) -> Self {
        array::from_fn(f)
    }
}

// We need an auxiliary trait for this, otherwise when ExtensionAlgebra tries to typecheck its supertrait,
// it will have a cycle trying to resolve Self::Base
pub trait HasBase {
    type Base: Field;
}

pub trait AbstractExtensionAlgebra: HasBase + 'static + Sized + Send + Sync + Debug {
    const D: usize;
    type Repr<AF: AbstractField<F = Self::Base>>: AbstractFieldArray<AF>;

    const GEN: Self::Repr<Self::Base>;

    fn mul<AF: AbstractField<F = Self::Base>>(
        a: AbstractExtension<AF, Self>,
        b: AbstractExtension<AF, Self>,
    ) -> AbstractExtension<AF, Self>;

    fn square<AF: AbstractField<F = Self::Base>>(
        a: AbstractExtension<AF, Self>,
    ) -> AbstractExtension<AF, Self> {
        a.clone() * a
    }
    fn repeated_frobenius(
        a: AbstractExtension<Self::Base, Self>,
        count: usize,
    ) -> AbstractExtension<Self::Base, Self>;
    fn inverse(a: AbstractExtension<Self::Base, Self>) -> AbstractExtension<Self::Base, Self>;
}

pub trait ExtensionAlgebra:
    AbstractExtensionAlgebra<Repr<<Self as HasBase>::Base> = Self::FieldRepr>
{
    // The `AbstractFieldArray<F>` bound isn't necessary, but helps the methods dispatch more easily
    type FieldRepr: AbstractFieldArray<Self::Base> + 'static + Copy + Send + Sync + Eq + Hash;
}
impl<A: AbstractExtensionAlgebra> ExtensionAlgebra for A
where
    Self::Repr<Self::Base>: 'static + Copy + Send + Sync + Eq + Hash,
{
    type FieldRepr = Self::Repr<Self::Base>;
}

#[derive(Debug)]
pub struct AbstractExtension<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>>(
    pub A::Repr<AF>,
);

pub type Extension<A> = AbstractExtension<<A as HasBase>::Base, A>;

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Clone
    for AbstractExtension<AF, A>
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<A: ExtensionAlgebra> Copy for Extension<A> {}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Default
    for AbstractExtension<AF, A>
{
    fn default() -> Self {
        Self(A::Repr::<AF>::from_fn(|_| AF::default()))
    }
}

impl<A: ExtensionAlgebra> PartialEq for Extension<A> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl<A: ExtensionAlgebra> Eq for Extension<A> {}

impl<A: ExtensionAlgebra> Display for Extension<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            let s = self
                .0
                .as_ref()
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, x)| match (i, x.is_one()) {
                    (0, _) => format!("{x}"),
                    (1, true) => "X".to_string(),
                    (1, false) => format!("{x} X"),
                    (_, true) => format!("X^{i}"),
                    (_, false) => format!("{x} X^{i}"),
                })
                .join(" + ");
            write!(f, "{}", s)
        }
    }
}

impl<A: ExtensionAlgebra> Hash for Extension<A> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for i in 0..A::D {
            self[i].hash(state);
        }
    }
}

impl<A: ExtensionAlgebra> Packable for Extension<A> {}

impl<A: ExtensionAlgebra> Serialize for Extension<A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_tuple(A::D)?;
        for i in 0..A::D {
            seq.serialize_element(&self[i])?;
        }
        seq.end()
    }
}

struct FieldArrayVisitor<A: ExtensionAlgebra>(PhantomData<A>);
impl<'de, A: ExtensionAlgebra> Visitor<'de> for FieldArrayVisitor<A> {
    type Value = Extension<A>;
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_fmt(format_args!("extension of length {}", A::D))
    }
    fn visit_seq<Seq>(self, mut seq: Seq) -> Result<Self::Value, Seq::Error>
    where
        Seq: SeqAccess<'de>,
    {
        let mut x = Self::Value::default();
        for i in 0..A::D {
            x[i] = seq
                .next_element()?
                .ok_or(serde::de::Error::invalid_length(i, &self))?;
        }
        Ok(x)
    }
}

impl<'de, A: ExtensionAlgebra> Deserialize<'de> for Extension<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(A::D, FieldArrayVisitor(PhantomData))
    }
}

impl<A: ExtensionAlgebra> Distribution<Extension<A>> for Standard
where
    Standard: Distribution<A::Base>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Extension<A> {
        AbstractExtension::from_base_fn(|_| Standard.sample(rng))
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> AbstractExtension<AF, A> {
    pub fn from_base(x: AF) -> Self {
        let mut me = Self::default();
        me[0] = x;
        me
    }
    pub fn from_base_fn(f: impl FnMut(usize) -> AF) -> Self {
        Self(A::Repr::<AF>::from_fn(f))
    }
    pub fn as_base_slice(&self) -> &[AF] {
        self.0.as_ref()
    }
    pub fn map<AF2: AbstractField<F = AF::F>>(
        self,
        mut f: impl FnMut(AF) -> AF2,
    ) -> AbstractExtension<AF2, A> {
        // could do this without `clone` if we used `replace_with`
        AbstractExtension::<AF2, A>::from_base_fn(|i| f(self[i].clone()))
    }
}

impl<A: ExtensionAlgebra> Extension<A> {
    pub fn frobenius(self) -> Self {
        self.repeated_frobenius(1)
    }
    pub fn repeated_frobenius(self, count: usize) -> Self {
        A::repeated_frobenius(self, count)
    }

    pub fn ext_powers_packed(
        &self,
    ) -> impl Iterator<Item = AbstractExtension<<A::Base as Field>::Packing, A>> {
        (0..0).map(|_| todo!())
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Index<usize>
    for AbstractExtension<AF, A>
{
    type Output = AF;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0.as_ref()[i]
    }
}
impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> IndexMut<usize>
    for AbstractExtension<AF, A>
{
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0.as_mut()[i]
    }
}

macro_rules! forward_methods_to_base {
    (
        $f:ident ( $( $arg_name:ident : $arg_ty:ty ),* ) ;
        $( $rest:tt )*
    ) => {
        fn $f($($arg_name:$arg_ty),*) -> Self {
            Self::from_base(AF::$f($($arg_name),*))
        }
        forward_methods_to_base!($($rest)*);
    };
    () => {};
}

impl<AF: AbstractField, A: ExtensionAlgebra<Base = AF::F>> AbstractField
    for AbstractExtension<AF, A>
{
    type F = Extension<A>;

    fn from_f(f: Self::F) -> Self {
        f.map(AF::from_f)
    }

    fn generator() -> Self {
        Self::from_f(AbstractExtension(A::GEN))
    }

    forward_methods_to_base!(
        zero();
        one();
        two();
        neg_one();
        from_bool(x: bool);
        from_canonical_u8(x: u8);
        from_canonical_u16(x: u16);
        from_canonical_u32(x: u32);
        from_canonical_u64(x: u64);
        from_canonical_usize(x: usize);
        from_wrapped_u32(x: u32);
        from_wrapped_u64(x: u64);
    );

    fn double(&self) -> Self {
        self.clone().map(|x| x.double())
    }
}

impl<A: ExtensionAlgebra> Field for Extension<A> {
    type Packing = Self;

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(A::inverse(*self))
        }
    }

    fn order() -> BigUint {
        A::Base::order().pow(A::D as u32)
    }

    fn halve(&self) -> Self {
        self.map(|x| x.halve())
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Neg
    for AbstractExtension<AF, A>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Add
    for AbstractExtension<AF, A>
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..A::D {
            self[i] += rhs[i].clone();
        }
        self
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> AddAssign
    for AbstractExtension<AF, A>
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Sum
    for AbstractExtension<AF, A>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from_base(AF::zero()), |acc, x| acc + x)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Sub
    for AbstractExtension<AF, A>
{
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..A::D {
            self[i] -= rhs[i].clone();
        }
        self
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> SubAssign
    for AbstractExtension<AF, A>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Mul
    for AbstractExtension<AF, A>
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        A::mul(self, rhs)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Mul<AF>
    for AbstractExtension<AF, A>
{
    type Output = Self;
    fn mul(self, rhs: AF) -> Self::Output {
        self.map(|x| x * rhs.clone())
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> MulAssign
    for AbstractExtension<AF, A>
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<Base = AF::F>> Product
    for AbstractExtension<AF, A>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from_base(AF::one()), |acc, x| acc * x)
    }
}

impl<A: ExtensionAlgebra> Div<Self> for AbstractExtension<A::Base, A> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn foo<A: ExtensionAlgebra>(ext: Extension<A>, base: A::Base) {
        let x = ext * base;
    }
    fn foo2<A: ExtensionAlgebra>(
        ext: AbstractExtension<<A::Base as Field>::Packing, A>,
        base: <A::Base as Field>::Packing,
    ) {
        let x = ext * base;
    }
}
