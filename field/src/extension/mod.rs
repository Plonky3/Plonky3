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

pub trait AbstractExtensionAlgebra<F: Field>: 'static + Sized + Debug {
    const D: usize;
    type Repr<AF: AbstractField<F = F>>: AbstractFieldArray<AF>;

    const GEN: Self::Repr<F>;

    fn mul<AF: AbstractField<F = F>>(
        a: Extension<AF, Self>,
        b: Extension<AF, Self>,
    ) -> Extension<AF, Self>;

    fn square<AF: AbstractField<F = F>>(a: Extension<AF, Self>) -> Extension<AF, Self> {
        a.clone() * a
    }
    fn repeated_frobenius(a: Extension<F, Self>, count: usize) -> Extension<F, Self>;
    fn inverse(a: Extension<F, Self>) -> Extension<F, Self>;
}

pub trait ExtensionAlgebra<F: Field>:
    AbstractExtensionAlgebra<F, Repr<F> = Self::FieldRepr>
{
    // The `AbstractFieldArray<F>` bound isn't necessary, but helps the methods dispatch more easily
    type FieldRepr: AbstractFieldArray<F> + 'static + Copy + Send + Sync + Eq + Hash;
}
impl<F: Field, A: AbstractExtensionAlgebra<F>> ExtensionAlgebra<F> for A
where
    Self::Repr<F>: 'static + Copy + Send + Sync + Eq + Hash,
{
    type FieldRepr = Self::Repr<F>;
}

#[derive(Debug)]
pub struct Extension<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>>(pub A::Repr<AF>);

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Clone for Extension<AF, A> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> Copy for Extension<F, A> {}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Default for Extension<AF, A> {
    fn default() -> Self {
        Self(A::Repr::<AF>::from_fn(|_| AF::default()))
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> PartialEq for Extension<F, A> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl<F: Field, A: ExtensionAlgebra<F>> Eq for Extension<F, A> {}

impl<F: Field, A: ExtensionAlgebra<F>> Display for Extension<F, A> {
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

impl<F: Field, A: ExtensionAlgebra<F>> Hash for Extension<F, A> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for i in 0..A::D {
            self[i].hash(state);
        }
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> Packable for Extension<F, A> {}

impl<F: Field, A: ExtensionAlgebra<F>> Serialize for Extension<F, A> {
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

struct FieldArrayVisitor<F: Field, A: ExtensionAlgebra<F>>(PhantomData<(F, A)>);
impl<'de, F: Field, A: ExtensionAlgebra<F>> Visitor<'de> for FieldArrayVisitor<F, A> {
    type Value = Extension<F, A>;
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

impl<'de, F: Field, A: ExtensionAlgebra<F>> Deserialize<'de> for Extension<F, A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(A::D, FieldArrayVisitor(PhantomData))
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> Distribution<Extension<F, A>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Extension<F, A> {
        Extension::from_base_fn(|_| Standard.sample(rng))
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Extension<AF, A> {
    pub fn from_base(x: AF) -> Self {
        let mut me = Self::default();
        me[0] = x;
        me
    }
    pub fn from_base_fn(f: impl FnMut(usize) -> AF) -> Self {
        Self(A::Repr::<AF>::from_fn(f))
    }
    pub fn map<AF2: AbstractField<F = AF::F>>(
        self,
        mut f: impl FnMut(AF) -> AF2,
    ) -> Extension<AF2, A> {
        // could do this without `clone` if we used `replace_with`
        Extension::<AF2, A>::from_base_fn(|i| f(self[i].clone()))
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> Extension<F, A> {
    pub fn frobenius(self) -> Self {
        self.repeated_frobenius(1)
    }
    pub fn repeated_frobenius(self, count: usize) -> Self {
        A::repeated_frobenius(self, count)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Index<usize> for Extension<AF, A> {
    type Output = AF;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0.as_ref()[i]
    }
}
impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> IndexMut<usize> for Extension<AF, A> {
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

impl<AF: AbstractField, A: ExtensionAlgebra<AF::F>> AbstractField for Extension<AF, A> {
    type F = Extension<AF::F, A>;

    fn from_f(f: Self::F) -> Self {
        f.map(AF::from_f)
    }

    fn generator() -> Self {
        Self::from_f(Extension(A::GEN))
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

impl<F: Field, A: ExtensionAlgebra<F>> Field for Extension<F, A> {
    type Packing = Self;

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(A::inverse(*self))
        }
    }

    fn order() -> BigUint {
        F::order().pow(A::D as u32)
    }

    fn halve(&self) -> Self {
        self.map(|x| x.halve())
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Neg for Extension<AF, A> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Add for Extension<AF, A> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for i in 0..A::D {
            self[i] += rhs[i].clone();
        }
        self
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> AddAssign for Extension<AF, A> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Sum for Extension<AF, A> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from_base(AF::zero()), |acc, x| acc + x)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Sub for Extension<AF, A> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        for i in 0..A::D {
            self[i] -= rhs[i].clone();
        }
        self
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> SubAssign for Extension<AF, A> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Mul for Extension<AF, A> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        A::mul(self, rhs)
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Mul<AF> for Extension<AF, A> {
    type Output = Self;
    fn mul(self, rhs: AF) -> Self::Output {
        self.map(|x| x * rhs.clone())
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> MulAssign for Extension<AF, A> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<AF: AbstractField, A: AbstractExtensionAlgebra<AF::F>> Product for Extension<AF, A> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from_base(AF::one()), |acc, x| acc * x)
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> Div for Extension<F, A> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: Field, A: ExtensionAlgebra<F>> Div<F> for Extension<F, A> {
    type Output = Self;
    fn div(self, rhs: F) -> Self::Output {
        self * rhs.inverse()
    }
}
