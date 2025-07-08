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

#[macro_export]
macro_rules! field_div_assign {
    ($type:ty $(, ($type_param:ty, $param_name:ty))?) => {
        paste::paste! {
            impl$(<$param_name: $type_param>)? DivAssign for $type$(<$param_name>)? {
                #[inline]
                fn div_assign(&mut self, rhs: Self) {
                    *self = *self / rhs;
                }
            }
        }
    };
}

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
