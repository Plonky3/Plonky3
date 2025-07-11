//! A collection of macros designed to implement simple operations
//! whose implementations are often boilerplate based off some other operation.

// To help with reading the macros, note that the ? operator indicates an optional argument.
// If it doesn't appear, all call of ? in the body of the macro disappear.
//
// Hence `impl_add_assign!(Mersenne31)` will produce:
//
// impl AddAssign for Mersenne31
// ...
//
// whereas `impl_add_assign!(MontyField31, (MontyParameters, MP))` produces:
//
// impl<MP: MontyParameters> AddAssign for MontyField31<MP>
// ...

/// Given a struct which implements `Add` implement `AddAssign<T>` for
/// any type `T` which implements `Into<Self>`.
///
/// `AddAssign` is implemented in a simple way by calling `add`
/// and assigning the result to `*self`.
#[macro_export]
macro_rules! impl_add_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl<$($param_name: $type_param,)? T: Into<Self>> AddAssign<T> for $type$(<$param_name>)? {
                #[inline]
                fn add_assign(&mut self, rhs: T) {
                    *self = *self + rhs.into();
                }
            }
        }
    };
}

/// Given a struct which implements `Add` implement `Sum`.
///
/// `Sum` is implemented by just doing a reduce on the iterator.
#[macro_export]
macro_rules! ring_sum {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Sum for $type$(<$param_name>)? {
                #[inline]
                fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                    iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
                }
            }
        }
    };
}

/// Given a struct which implements `Sub` implement `SubAssign<T>` for
/// any type `T` which implements `Into<Self>`.
///
/// `SubAssign` is implemented in a simple way by calling `sub`
/// and assigning the result to `*self`.
#[macro_export]
macro_rules! impl_sub_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl<$($param_name: $type_param,)? T: Into<Self>> SubAssign<T> for $type$(<$param_name>)? {
                #[inline]
                fn sub_assign(&mut self, rhs: T) {
                    *self = *self - rhs.into();
                }
            }
        }
    };
}

/// Given a struct which implements `Mul` implement `MulAssign<T>` for
/// any type `T` which implements `Into<Self>`.
///
/// `MulAssign` is implemented in a simple way by calling `mul`
/// and assigning the result to `*self`. Similarly `Product` is implemented
/// in the similarly simple way of just doing a reduce on the iterator.
#[macro_export]
macro_rules! impl_mul_methods {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl<$($param_name: $type_param,)? T: Into<Self>> MulAssign<T> for $type$(<$param_name>)? {
                #[inline]
                fn mul_assign(&mut self, rhs: T) {
                    *self = *self * rhs.into();
                }
            }

            impl$(<$param_name: $type_param>)? Product for $type$(<$param_name>)? {
                #[inline]
                fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                    iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
                }
            }
        }
    };
}

/// Given two structs `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Add<Field>` for `Alg` and `Add<Alg>` for `Field`.
///
/// All are implemented in the simplest way by using `From` to map the `Field` element
/// to an `Alg` element and then applying the native `add` methods on `Alg` elements.
#[macro_export]
macro_rules! impl_add_base_field {
    ($alg_type:ty, $field_type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Add<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                fn add(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self + Self::from(rhs)
                }
            }

            impl$(<$param_name: $type_param>)? Add<$alg_type$(<$param_name>)?> for $field_type$(<$param_name>)? {
                type Output = $alg_type$(<$param_name>)?;

                #[inline]
                fn add(self, rhs: $alg_type$(<$param_name>)?) -> Self::Output {
                    $alg_type::from(self) + rhs
                }
            }
        }
    };
}

/// Given two structs `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Sub<Field>` for `Alg` and `Sub<Alg>` for `Field`.
///
/// All are implemented in the simplest way by using `From` to map the `Field` element
/// to an `Alg` element and then applying the native `sub` methods on `Alg` elements.
#[macro_export]
macro_rules! impl_sub_base_field {
    ($alg_type:ty, $field_type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Sub<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                fn sub(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self - Self::from(rhs)
                }
            }

            impl$(<$param_name: $type_param>)? Sub<$alg_type$(<$param_name>)?> for $field_type$(<$param_name>)? {
                type Output = $alg_type$(<$param_name>)?;

                #[inline]
                fn sub(self, rhs: $alg_type$(<$param_name>)?) -> Self::Output {
                    $alg_type::from(self) - rhs
                }
            }
        }
    };
}

/// Given two structs `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Mul<Field>` for `Alg` and `Mul<Alg>` for `Field`.
///
/// All are implemented in the simplest way by using `From` to map the `Field` element
/// to an `Alg` element and then applying the native `mul` methods on `Alg` elements.
#[macro_export]
macro_rules! impl_mul_base_field {
    ($alg_type:ty, $field_type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Mul<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                fn mul(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self * Self::from(rhs)
                }
            }

            impl$(<$param_name: $type_param>)? Mul<$alg_type$(<$param_name>)?> for $field_type$(<$param_name>)? {
                type Output = $alg_type$(<$param_name>)?;

                #[inline]
                fn mul(self, rhs: $alg_type$(<$param_name>)?) -> Self::Output {
                    $alg_type::from(self) * rhs
                }
            }
        }
    };
}

/// Given two structs `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Div<Field>` and `DivAssign<Field>` for `Alg`.
///
/// Both are implemented in the simplest way by first applying the `.inverse()` map from
/// `Field` then using the `From` to map the inverse to an `Alg` element before
///  applying the native `mul` or `mul_assign` methods on `Alg` elements.
///
/// This can also be used with `Alg = Field` to implement `Div` and `DivAssign` for Field.
#[macro_export]
macro_rules! impl_div_methods {
    ($alg_type:ty, $field_type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Div<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                #[allow(clippy::suspicious_arithmetic_impl)]
                fn div(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self * Self::from(rhs.inverse())
                }
            }

            impl$(<$param_name: $type_param>)? DivAssign<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                #[inline]
                #[allow(clippy::suspicious_op_assign_impl)]
                fn div_assign(&mut self, rhs: $field_type$(<$param_name>)?) {
                    *self *= Self::from(rhs.inverse());
                }
            }
        }
    };
}

/// Given two structs `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Sum<Field> and Product<Field>` for `Alg`.
///
/// Both are implemented in the simplest way by simply computing the Sum/Product as
/// field elements before mapping to an `Alg` element using `From`.
#[macro_export]
macro_rules! impl_sum_prod_base_field {
    ($alg_type:ty, $field_type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Sum<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                #[inline]
                fn sum<I>(iter: I) -> Self
                where
                    I: Iterator<Item = $field_type$(<$param_name>)?>,
                {
                    iter.sum::<$field_type$(<$param_name>)?>().into()
                }
            }

            impl$(<$param_name: $type_param>)? Product<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                #[inline]
                fn product<I>(iter: I) -> Self
                where
                    I: Iterator<Item = $field_type$(<$param_name>)?>,
                {
                    iter.product::<$field_type$(<$param_name>)?>().into()
                }
            }
        }
    };
}

/// Given a struct `Alg` which is a wrapper over `[Field; N]` for some `N`,
/// implement `Distribution<Alg>` for `StandardUniform`.
///
/// As `Distribution<Field>` is implemented for `StandardUniform` we can
/// already generate random `[Field; N]` elements so we just need to wrap the
/// result in `Alg`'s name.
#[macro_export]
macro_rules! impl_rng {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Distribution<$type$(<$param_name>)?> for StandardUniform {
                #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $type$(<$param_name>)? {
                $type(rng.random())
                }
            }
        }
    };
}

/// Given `Field` and `Algebra` structs where `Algebra` is simply a wrapper around `[Field; N]`
/// implement `PackedValue` for `Algebra`.
///
/// # Safety
/// `Algebra` must be `repr(transparent)` and castable from to/from `[Field; N]`. Assuming this
/// holds, these types have the same alignment and size, so all our reference casts are safe.
#[macro_export]
macro_rules! impl_packed_value {
    ($alg_type:ty, $field_type:ty, $width:expr $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            unsafe impl$(<$param_name: $type_param>)? PackedValue for $alg_type$(<$param_name>)? {
                type Value = $field_type$(<$param_name>)?;

                const WIDTH: usize = $width;

                #[inline]
                fn from_slice(slice: &[Self::Value]) -> &Self {
                    assert_eq!(slice.len(), Self::WIDTH);
                    unsafe { &*slice.as_ptr().cast() }
                }

                #[inline]
                fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
                    assert_eq!(slice.len(), Self::WIDTH);
                    unsafe { &mut *slice.as_mut_ptr().cast() }
                }

                #[inline]
                fn as_slice(&self) -> &[Self::Value] {
                    &self.0
                }

                #[inline]
                fn as_slice_mut(&mut self) -> &mut [Self::Value] {
                    &mut self.0
                }

                #[inline]
                fn from_fn<F: FnMut(usize) -> Self::Value>(f: F) -> Self {
                    Self(core::array::from_fn(f))
                }
            }
        }
    };
}

pub use {
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
};
