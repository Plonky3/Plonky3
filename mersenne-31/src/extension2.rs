use core::ops::Mul;

use p3_field::{define_ext, impl_ext_af, impl_ext_f, AbstractField, Field};

use crate::Mersenne31;

define_ext!(pub CM31([Mersenne31; 2]));

impl<AF: AbstractField<F = Mersenne31>> Mul for CM31<AF> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let ([a, b], [c, d]) = (self.0, rhs.0);
        Self([
            a.clone() * c.clone() - b.clone() * d.clone(),
            a.clone() * d.clone() + b.clone() * c.clone(),
        ])
    }
}

impl<AF: AbstractField<F = Mersenne31>> AbstractField for CM31<AF> {
    impl_ext_af!();
    type F = CM31<Mersenne31>;
    fn generator() -> Self {
        Self([Mersenne31::new(12), Mersenne31::new(1)].map(AF::from_f))
    }
}

impl Field for CM31<Mersenne31> {
    impl_ext_f!(Mersenne31, 2);
    fn try_inverse(&self) -> Option<Self> {
        let [a, b] = self.0;
        (a.square() + b.square())
            .try_inverse()
            .map(|s| Self([a * s, -b * s]))
    }
}

/*
define_extension!(Mersenne31, pub struct CM31<AF>([AF; 2]));
impl_castable_extension!(Mersenne31, CM31<AF>, CM31<AF>, 1);
impl_castable_extension!(Mersenne31, CM31<AF>, AF, 2);
*/

/*

*/

// define_extension!(Mersenne31, pub struct QM31<AF>([CM31<AF>; 2]));

// define_extension!(QM31(2) > CM31(2) > Mersenne31);

/*
#[derive(
    Copy, Clone, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
pub struct CM31<AF>([AF; 2]);

#[derive(
    Copy, Clone, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
)]
#[repr(transparent)]
pub struct QM31<AF>([CM31<AF>; 2]);

impl<AF: AbstractField<F = Mersenne31>> CastableExtension<AF> for QM31<AF> {
    fn cast(&mut self) -> &mut [AF] {
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast(), 4) }
    }
}
impl<AF: AbstractField<F = Mersenne31>> CastableExtension<CM31<AF>> for QM31<AF> {
    fn cast(&mut self) -> &mut [CM31<AF>] {
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast(), 2) }
    }
}
impl<AF: AbstractField<F = Mersenne31>> CastableExtension<QM31<AF>> for QM31<AF> {
    fn cast(&mut self) -> &mut [QM31<AF>] {
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast(), 1) }
    }
}

impl<AF, Rhs> Add<Rhs> for QM31<AF>
where
    Rhs: AbstractField,
    Self: CastableExtension<Rhs>,
{
    type Output = Self;
    fn add(mut self, rhs: Rhs) -> Self::Output {
        for x in self.cast() {
            *x += rhs.clone();
        }
        self
    }
}
*/

/*
define_extension!(pub struct CM31([Mersenne31; 2]));
define_extension!(pub struct QM31([CM31<Mersenne31>; 2]));

impl<AF> CastableExtension<CM31<AF>> for CM31<AF> {
    fn cast_to_base<AF1: AbstractField<F = Self>, AF2: AbstractField<F = CM31<AF>>>(me: &AF1)
            -> &[AF2] {

    }
}
*/

/*
impl<AF: AbstractField<F = Mersenne31>> Add for CM31<AF> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl<AF: AbstractField<F = Mersenne31>> Add<AF> for CM31<AF> {
    type Output = Self;
    fn add(self, rhs: AF) -> Self::Output {
        todo!()
    }
}

enum CastResult<'a, Sub, Super> {
    Same(Super),
    Smaller(&'a [Sub]),
}

trait HasCastableSubfield<Sub>: Sized {
    fn cast_to_subfield(&mut self) -> &mut [Sub];
}

impl<AF, Rhs> Add<Rhs> for QM31<AF>
where
    Rhs: AbstractField,
    QM31<CM31<Mersenne31>>: HasCastableSubfield<Rhs::F>,
{
    type Output = Self;
    fn add(mut self, rhs: Rhs) -> Self::Output {
        for x in self.cast_to_subfield() {
            *x += rhs.clone();
        }
        self
    }
}
*/

/*
impl<AF: AbstractField<F = CM31<Mersenne31>>> Add<AF> for QM31<AF> {
    type Output = Self;
    fn add(self, rhs: AF) -> Self::Output {
        todo!()
    }
}
impl<AF: AbstractField<F = Mersenne31>> Add<AF> for QM31<AF> {
    type Output = Self;
    fn add(self, rhs: AF) -> Self::Output {
        todo!()
    }
}
*/

/*
impl<AF: AbstractField<F = Mersenne31>> Add<AF> for CM31<AF> {
    type Output = Self;
    fn add(self, rhs: AF) -> Self::Output {
        todo!()
    }
}
*/

/*
make_extension!(pub struct CM31([Mersenne31; 2]));

impl<AF: AbstractField<F = Mersenne31>> AbstractField for CM31<AF> {
    impl_ext_af!();
    type F = CM31<Mersenne31>;
    fn generator() -> Self {
        Self([Mersenne31::new(12), Mersenne31::new(1)].map(AF::from_f))
    }
}

impl<AF: AbstractField<F = Mersenne31>> Mul for CM31<AF> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let ([a, b], [c, d]) = (self.0, rhs.0);
        Self([
            a.clone() * c.clone() - b.clone() * d.clone(),
            a.clone() * d.clone() + b.clone() * c.clone(),
        ])
    }
}

impl Field for CM31<Mersenne31> {
    impl_ext_f!(Mersenne31, 2);
    fn try_inverse(&self) -> Option<Self> {
        let [a, b] = self.0;
        (a.square() + b.square())
            .try_inverse()
            .map(|s| Self([a * s, -b * s]))
    }
}
*/

/*
impl<AF: AbstractField<F = Mersenne31>> Add<AF> for CM31<AF> {
    type Output = CM31<AF>;
    fn add(self, rhs: AF) -> Self::Output {
        todo!()
    }
}

impl<AF: AbstractField<F = ()>> Add<AF> for CM31<AF> {
    type Output = CM31<AF>;
    fn add(self, rhs: AF) -> Self::Output {
        todo!()
    }
}
*/

/*
impl<AF: AbstractField<F = Mersenne31>> AbstractExtensionField<AF> for CM31<AF> {
    const D: usize = 2;

    fn from_base(b: AF) -> Self {
        todo!()
    }

    fn from_base_slice(bs: &[AF]) -> Self {
        todo!()
    }

    fn from_base_fn<F: FnMut(usize) -> AF>(f: F) -> Self {
        todo!()
    }

    fn as_base_slice(&self) -> &[AF] {
        todo!()
    }
}
*/
