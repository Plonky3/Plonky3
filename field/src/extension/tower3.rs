pub trait CastableExtension<Base> {
    fn cast(&mut self) -> &mut [Base];
}

#[macro_export]
macro_rules! make_ext {
    ($name:ty[$d:literal] >> $bottom:ty) => {
        #[derive(
            Copy, Clone, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name<AF>([AF; $d]);

        $crate::impl_castable_extension!($bottom, $name<AF>, AF, $d);
        $crate::impl_castable_extension!($bottom, $name<AF>, $name<AF>, 1);

        $crate::impl_ext!($name);
    };
    ($name:ident($d:literal) > $base:ident($bd:literal) >> $bottom:ty) => {
        #[derive(
            Copy, Clone, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name<AF>([$base<AF>; $d]);

        // todo: get $d from AbstractExtensionField
        $crate::impl_castable_extension!($bottom, $name<AF>, AF, $d * $bd);
        $crate::impl_castable_extension!($bottom, $name<AF>, $base<AF>, $d);
        $crate::impl_castable_extension!($bottom, $name<AF>, $name<AF>, 1);
    };
}

#[macro_export]
macro_rules! impl_castable_extension {
    ($bottom:ty, $sup:ty, $sub:ty, $d:expr) => {
        impl<AF: AbstractField<F = $bottom>> CastableExtension<$sub> for $sup {
            fn cast(&mut self) -> &mut [$sub] {
                unsafe { slice::from_raw_parts_mut((self as *mut Self).cast(), $d) }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ext {
    ($name:ident) => {
        impl<AF, Rhs> core::ops::Add<Rhs> for $name<AF>
        where
            Rhs: AbstractField,
            Self: CastableExtension<Rhs>,
        {
            type Output = Self;
            fn add(mut self, rhs: Rhs) -> Self::Output {
                self.cast()[0] += rhs;
                self
            }
        }
    };
}

#[macro_export]
macro_rules! define_extension {
    ($bottom:path, $vis:vis struct $name:ident < AF > ( [ $base:ty; $d:literal ] )) => {
        #[derive(
            Copy, Clone, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        pub struct $name<AF>([$base; $d]);

        impl $crate::Packable for $name<$bottom> {}

        impl<AF> From<AF> for $name<AF> {
            fn from(x: AF) -> Self {
                Self($crate::field_to_array(x))
            }
        }

        impl core::fmt::Display for $name<$bottom> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                use itertools::Itertools;
                if <Self as $crate::Field>::is_zero(self) {
                    write!(f, "0")
                } else {
                    let str = self
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| !<$base as $crate::Field>::is_zero(x))
                        .map(|(i, x)| match (i, <$base as $crate::Field>::is_one(x)) {
                            (0, _) => alloc::format!("{x}"),
                            (1, true) => "X".into(),
                            (1, false) => alloc::format!("{x} X"),
                            (_, true) => alloc::format!("X^{i}"),
                            (_, false) => alloc::format!("{x} X^{i}"),
                        })
                        .join(" + ");
                    write!(f, "{}", str)
                }
            }
        }

        impl<AF> core::ops::Neg for $name<AF>
        where
            $base: core::ops::Neg<Output = $base>,
        {
            type Output = Self;
            fn neg(self) -> Self::Output {
                Self(self.0.map(<$base>::neg))
            }
        }

        impl<AF, Rhs> core::ops::Add<Rhs> for $name<AF>
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

        impl<AF, Rhs> core::ops::AddAssign<Rhs> for $name<AF>
        where
            Rhs: AbstractField,
            Self: CastableExtension<Rhs>,
        {
            fn add_assign(&mut self, rhs: Rhs) {
                *self = self.clone() + rhs;
            }
        }

        impl<AF, Rhs> core::ops::Sub<Rhs> for $name<AF>
        where
            Rhs: AbstractField,
            Self: CastableExtension<Rhs>,
        {
            type Output = Self;
            fn sub(mut self, rhs: Rhs) -> Self::Output {
                for x in self.cast() {
                    *x -= rhs.clone();
                }
                self
            }
        }

        impl<AF, Rhs> core::ops::SubAssign<Rhs> for $name<AF>
        where
            Rhs: AbstractField,
            Self: CastableExtension<Rhs>,
        {
            fn sub_assign(&mut self, rhs: Rhs) {
                *self = self.clone() - rhs;
            }
        }

        impl<AF, Rhs> core::ops::Mul<Rhs> for $name<AF>
        where
            Rhs: AbstractField,
            Self: CastableExtension<Rhs>,
        {
            type Output = Self;
            fn mul(mut self, rhs: Rhs) -> Self::Output {
                todo!()
            }
        }

        impl<AF, Rhs> core::ops::MulAssign<Rhs> for $name<AF>
        where
            Rhs: AbstractField,
            Self: CastableExtension<Rhs>,
        {
            fn mul_assign(&mut self, rhs: Rhs) {
                *self = self.clone() * rhs;
            }
        }

        impl<Rhs> core::ops::Div<Rhs> for $name<$bottom>
        where
            Self: CastableExtension<Rhs>,
        {
            type Output = Self;
            fn div(mut self, rhs: Rhs) -> Self::Output {
                self * rhs.inverse()
            }
        }

        impl<Rhs> core::ops::DivAssign<Rhs> for $name<$bottom>
        where
            Self: CastableExtension<Rhs>,
        {
            fn div_assign(&mut self, rhs: Rhs) {
                *self = self.clone() / rhs;
            }
        }

        impl<AF> core::iter::Sum for $name<AF> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                let zero = <Self as $crate::AbstractField>::zero();
                iter.fold(zero, |acc, x| acc + x)
            }
        }

        impl<AF> core::iter::Product for $name<AF> {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                let one = <Self as $crate::AbstractField>::one();
                iter.fold(one, |acc, x| acc * x)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ext_af {
    () => {
        fn zero() -> Self {
            Self($crate::field_to_array(AF::zero()))
        }
        fn one() -> Self {
            Self($crate::field_to_array(AF::one()))
        }
        fn two() -> Self {
            Self($crate::field_to_array(AF::two()))
        }
        fn neg_one() -> Self {
            Self($crate::field_to_array(AF::neg_one()))
        }
        fn from_f(x: Self::F) -> Self {
            Self(x.0.map(AF::from_f))
        }
        fn from_bool(x: bool) -> Self {
            AF::from_bool(x).into()
        }
        fn from_canonical_u8(x: u8) -> Self {
            AF::from_canonical_u8(x).into()
        }
        fn from_canonical_u16(x: u16) -> Self {
            AF::from_canonical_u16(x).into()
        }
        fn from_canonical_u32(x: u32) -> Self {
            AF::from_canonical_u32(x).into()
        }
        fn from_canonical_u64(x: u64) -> Self {
            AF::from_canonical_u64(x).into()
        }
        fn from_canonical_usize(x: usize) -> Self {
            AF::from_canonical_usize(x).into()
        }
        fn from_wrapped_u32(x: u32) -> Self {
            AF::from_wrapped_u32(x).into()
        }
        fn from_wrapped_u64(x: u64) -> Self {
            AF::from_wrapped_u64(x).into()
        }
    };
}

#[macro_export]
macro_rules! impl_ext_f {
    ($base:path, $d:literal) => {
        type Packing = Self;
        fn halve(&self) -> Self {
            Self(self.0.map(|x| x.halve()))
        }
        fn order() -> num_bigint::BigUint {
            <$base as $crate::Field>::order().pow($d as u32)
        }
    };
}

#[macro_export]
macro_rules! make_extension {
    ($vis:vis struct $name:ident ( [ $base:path ; $d:literal ])) => {
        #[derive(
            Copy, Clone, Eq, PartialEq, Hash, Debug, Default, serde::Serialize, serde::Deserialize,
        )]
        #[repr(transparent)]
        $vis struct $name<AF>([AF; $d]);

        impl<AF: $crate::AbstractField<F = $base>> From<AF> for $name<AF> {
            fn from(x: AF) -> Self {
                Self($crate::field_to_array(x))
            }
        }

        impl core::fmt::Display for $name<$base> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                use itertools::Itertools;
                if <Self as $crate::Field>::is_zero(self) {
                    write!(f, "0")
                } else {
                    let str = self
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| !<$base as $crate::Field>::is_zero(x))
                        .map(|(i, x)| match (i, <$base as $crate::Field>::is_one(x)) {
                            (0, _) => alloc::format!("{x}"),
                            (1, true) => "X".into(),
                            (1, false) => alloc::format!("{x} X"),
                            (_, true) => alloc::format!("X^{i}"),
                            (_, false) => alloc::format!("{x} X^{i}"),
                        })
                        .join(" + ");
                    write!(f, "{}", str)
                }
            }
        }

        impl $crate::Packable for $name<$base> {}

        impl<AF: $crate::AbstractField<F = $base>> core::ops::Neg for $name<AF> {
            type Output = Self;
            fn neg(self) -> Self {
                Self(self.0.map(AF::neg))
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::ops::Add for $name<AF> {
            type Output = Self;
            fn add(mut self, rhs: Self) -> Self {
                for (l, r) in self.0.iter_mut().zip(rhs.0) {
                    *l += r;
                }
                self
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::ops::AddAssign for $name<AF> {
            fn add_assign(&mut self, rhs: Self) {
                *self = self.clone() + rhs;
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::iter::Sum for $name<AF> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                let zero = <Self as $crate::AbstractField>::zero();
                iter.fold(zero, |acc, x| acc + x)
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::ops::Sub for $name<AF> {
            type Output = Self;
            fn sub(mut self, rhs: Self) -> Self {
                for (l, r) in self.0.iter_mut().zip(rhs.0) {
                    *l -= r;
                }
                self
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::ops::SubAssign for $name<AF> {
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.clone() - rhs;
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::ops::MulAssign for $name<AF> {
            fn mul_assign(&mut self, rhs: Self) {
                *self = self.clone() * rhs;
            }
        }

        impl<AF: $crate::AbstractField<F = $base>> core::iter::Product for $name<AF> {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                let one = <Self as $crate::AbstractField>::one();
                iter.fold(one, |acc, x| acc * x)
            }
        }

        impl core::ops::Div for $name<$base> {
            type Output = Self;
            fn div(mut self, rhs: Self) -> Self {
                self * <Self as $crate::Field>::inverse(&rhs)
            }
        }

        impl core::ops::DivAssign for $name<$base> {
            fn div_assign(&mut self, rhs: Self) {
                *self = self.clone() / rhs;
            }
        }
    };
}

/*
#[macro_export]
macro_rules! unsafe_impl_castable_ext_of {
    ($ext:path, $base:path, $d:literal) => {
        impl<AF: AbstractField<F = Mersenne31>> AbstractExtensionField<AF> for CM31<AF> {
    }
}
*/
