//! A collection of macros designed to implement simple operations
//! whose implementations are often boilerplate based off some other operation.

// To help with reading the macros, note that the ? operator indicates an optional argument.
// If it doesn't appear, all call of ? in the body of the macro disappear.
//
// Hence `ring_add_assign!(Mersenne31)` will produce:
//
// impl AddAssign for Mersenne31
// ...
//
// whereas `ring_add_assign!(MontyField31, (MontyParameters, MP))` produces:
//
// impl<MP: MontyParameters> AddAssign for MontyField31<FP>
// ...

/// Given a struct which implements `Add` implement `AddAssign`.
///
/// `AddAssign` is implemented in the simplest way by calling `add`
/// and assigning the result to `*self`.
#[macro_export]
macro_rules! ring_add_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? AddAssign for $type$(<$param_name>)? {
                #[inline]
                fn add_assign(&mut self, rhs: Self) {
                    *self = *self + rhs;
                }
            }
        }
    };
}

/// Given a struct which implements `Sub` implement `SubAssign`.
///
/// `SubAssign` is implemented in the simplest way by simply calling `sub`
/// and assigning the result to `*self`.
#[macro_export]
macro_rules! ring_sub_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? SubAssign for $type$(<$param_name>)? {
                #[inline]
                fn sub_assign(&mut self, rhs: Self) {
                    *self = *self - rhs;
                }
            }
        }
    };
}

/// Given a struct which implements `Mul` implement `MulAssign`.
///
/// `MulAssign` is implemented in the simplest way by simply calling `mul`
/// and assigning the result to `*self`.
#[macro_export]
macro_rules! ring_mul_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? MulAssign for $type$(<$param_name>)? {
                #[inline]
                fn mul_assign(&mut self, rhs: Self) {
                    *self = *self * rhs;
                }
            }
        }
    };
}

/// Given a struct which implements `Mul` and `.inverse()` implement `Div, DivAssign`.
///
/// Both are implemented in the simplest way by inverting the right hand side and
/// using `mul` or `mul_assign`.
#[macro_export]
macro_rules! field_div_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Div for $type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                #[allow(clippy::suspicious_arithmetic_impl)]
                fn div(self, rhs: Self) -> Self {
                    self * rhs.inverse()
                }
            }

            impl$(<$param_name: $type_param>)? DivAssign for $type$(<$param_name>)? {
                #[inline]
                #[allow(clippy::suspicious_op_assign_impl)]
                fn div_assign(&mut self, rhs: Self) {
                    *self *= rhs.inverse();
                }
            }
        }
    };
}

/// Given a two struct `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Add<Field> and AddAssign<Field>` for `Alg` and `Add<Alg>` for `Field`.
///
/// All are implemented in the simplest way by using `From` to map the `Field` element
/// to an `Alg` element and then applying the native `add` or `add_assign` methods on `Alg` elements.
#[macro_export]
macro_rules! algebra_from_field_add {
    ($alg_type:ty, $field_type: ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Add<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                fn add(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self + Self::from(rhs)
                }
            }

            impl$(<$param_name: $type_param>)? AddAssign<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                #[inline]
                fn add_assign(&mut self, rhs: $field_type$(<$param_name>)?) {
                    *self += Self::from(rhs);
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

/// Given a two struct `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Sub<Field> and SubAssign<Field>` for `Alg` and `Sub<Alg>` for `Field`.
///
/// All are implemented in the simplest way by using `From` to map the `Field` element
/// to an `Alg` element and then applying the native `sub` or `sub_assign` methods on `Alg` elements.
#[macro_export]
macro_rules! algebra_from_field_sub {
    ($alg_type:ty, $field_type: ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Sub<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                fn sub(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self - Self::from(rhs)
                }
            }

            impl$(<$param_name: $type_param>)? SubAssign<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                #[inline]
                fn sub_assign(&mut self, rhs: $field_type$(<$param_name>)?) {
                    *self -= Self::from(rhs);
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

/// Given a two struct `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Mul<Field> and MulAssign<Field>` for `Alg` and `Mul<Alg>` for `Field`.
///
/// All are implemented in the simplest way by using `From` to map the `Field` element
/// to an `Alg` element and then applying the native `mul` or `mul_assign` methods on `Alg` elements.
#[macro_export]
macro_rules! algebra_from_field_mul {
    ($alg_type:ty, $field_type: ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? Mul<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                type Output = Self;

                #[inline]
                fn mul(self, rhs: $field_type$(<$param_name>)?) -> Self {
                    self * Self::from(rhs)
                }
            }

            impl$(<$param_name: $type_param>)? MulAssign<$field_type$(<$param_name>)?> for $alg_type$(<$param_name>)? {
                #[inline]
                fn mul_assign(&mut self, rhs: $field_type$(<$param_name>)?) {
                    *self *= Self::from(rhs);
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

/// Given a two struct `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Div<Field> and DivAssign<Field>` for `Alg`.
///
/// Both are implemented in the simplest way by first applying the `.inverse()` map from
/// `Field` then using the `From` to map the inverse to an `Alg` element before
///  applying the native `mul` or `mul_assign` methods on `Alg` elements.
#[macro_export]
macro_rules! algebra_from_field_div {
    ($alg_type:ty, $field_type: ty $(, ($type_param:ty, $param_name:ty))?) => {
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

/// Given a two struct `Alg` and `Field` where `Alg` implements `From<Field>`, implement
/// `Sum<Field> and Product<Field>` for `Alg`.
///
/// Both are implemented in the simplest way by simply computing the Sum/Product as
/// field elements before mapping to an `Alg` element using `From`.
#[macro_export]
macro_rules! algebra_from_field_sum_prod {
    ($alg_type:ty, $field_type: ty $(, ($type_param:ty, $param_name:ty))?) => {
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

pub use {
    algebra_from_field_add, algebra_from_field_div, algebra_from_field_mul, algebra_from_field_sub,
    algebra_from_field_sum_prod, field_div_assign, ring_add_assign, ring_mul_assign,
    ring_sub_assign,
};
