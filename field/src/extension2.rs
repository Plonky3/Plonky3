#[macro_export]
macro_rules! define_ext {
    ($vis:vis $name:ident([$base:ty; $d:literal])) => {
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

        impl $crate::Packable for $name<$base> {}

        impl core::fmt::Display for $name<$base> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                use itertools::Itertools;
                if self.is_zero() {
                    write!(f, "0")
                } else {
                    let str = self
                        .0
                        .iter()
                        .enumerate()
                        .filter(|(_, x)| !x.is_zero())
                        .map(|(i, x)| match (i, x.is_one()) {
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

        impl rand::prelude::Distribution<$name<$base>>
            for rand::distributions::Standard
        where
            rand::distributions::Standard: rand::prelude::Distribution<$base>,
        {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $name<$base> {
                let mut res = [<$base as $crate::AbstractField>::zero(); $d];
                for r in res.iter_mut() {
                    *r = rand::distributions::Standard.sample(rng);
                }
                $name(res)
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
        fn double(&self) -> Self {
            Self(core::array::from_fn(|i| self.0[i].double()))
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
