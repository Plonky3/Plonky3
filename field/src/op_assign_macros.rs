#[macro_export]
macro_rules! ring_add_assign {
    ($type:ty) => {
        impl AddAssign for $type {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
    };

    ($type:ty, $type_param:ty) => {
        paste::paste! {
            impl<T: $type_param> AddAssign for $type<T> {
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
    ($type:ty) => {
        impl SubAssign for $type {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
    };

    ($type:ty, $type_param:ty) => {
        paste::paste! {
            impl<T: $type_param> SubAssign for $type<T> {
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
    ($type:ty) => {
        impl MulAssign for $type {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
    };

    ($type:ty, $type_param:ty) => {
        paste::paste! {
            impl<T: $type_param> MulAssign for $type<T> {
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
    ($type:ty) => {
        impl DivAssign for $type {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    };

    ($type:ty, $type_param:ty) => {
        paste::paste! {
            impl<T: $type_param> DivAssign for $type<T> {
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
    ($alg_type:ty, $field_type: ty) => {
        paste::paste! {
            impl Add<$field_type> for $alg_type {
                type Output = $alg_type;

                #[inline]
                fn add(self, rhs: $field_type) -> $alg_type {
                    self + Self::from(rhs)
                }
            }

            impl AddAssign<$field_type> for $alg_type {
                #[inline]
                fn add_assign(&mut self, rhs: $field_type) {
                    *self += Self::from(rhs);
                }
            }

            impl Add<$alg_type> for $field_type {
                type Output = $alg_type;

                #[inline]
                fn add(self, rhs: $alg_type) -> $alg_type {
                    $alg_type::from(self) + rhs
                }
            }
        }
    };
}

#[macro_export]
macro_rules! algebra_from_field_sub {
    ($alg_type:ty, $field_type: ty) => {
        paste::paste! {
            impl Sub<$field_type> for $alg_type {
                type Output = $alg_type;

                #[inline]
                fn sub(self, rhs: $field_type) -> $alg_type {
                    self - Self::from(rhs)
                }
            }

            impl SubAssign<$field_type> for $alg_type {
                #[inline]
                fn sub_assign(&mut self, rhs: $field_type) {
                    *self -= Self::from(rhs);
                }
            }

            impl Sub<$alg_type> for $field_type {
                type Output = $alg_type;

                #[inline]
                fn sub(self, rhs: $alg_type) -> $alg_type {
                    $alg_type::from(self) - rhs
                }
            }
        }
    };
}

#[macro_export]
macro_rules! algebra_from_field_mul {
    ($alg_type:ty, $field_type: ty) => {
        paste::paste! {
            impl Mul<$field_type> for $alg_type {
                type Output = $alg_type;

                #[inline]
                fn mul(self, rhs: $field_type) -> $alg_type {
                    self * Self::from(rhs)
                }
            }

            impl MulAssign<$field_type> for $alg_type {
                #[inline]
                fn mul_assign(&mut self, rhs: $field_type) {
                    *self *= Self::from(rhs);
                }
            }

            impl Mul<$alg_type> for $field_type {
                type Output = $alg_type;

                #[inline]
                fn mul(self, rhs: $alg_type) -> $alg_type {
                    $alg_type::from(self) * rhs
                }
            }
        }
    };
}

#[macro_export]
macro_rules! algebra_from_field_div {
    ($alg_type:ty, $field_type: ty) => {
        paste::paste! {
            impl Div<$field_type> for $alg_type {
                type Output = $alg_type;

                #[inline]
                #[allow(clippy::suspicious_arithmetic_impl)]
                fn div(self, rhs: $field_type) -> $alg_type {
                    self * Self::from(rhs.inverse())
                }
            }

            impl DivAssign<$field_type> for $alg_type {
                #[inline]
                #[allow(clippy::suspicious_op_assign_impl)]
                fn div_assign(&mut self, rhs: $field_type) {
                    *self *= Self::from(rhs.inverse());
                }
            }
        }
    };
}

#[macro_export]
macro_rules! algebra_from_field_sum_prod {
    ($alg_type:ty, $field_type: ty) => {
        paste::paste! {
            impl Sum<$field_type> for $alg_type {
                #[inline]
                fn sum<I>(iter: I) -> $alg_type
                where
                    I: Iterator<Item = $field_type>,
                {
                    iter.sum::<$field_type>().into()
                }
            }

            impl Product<$field_type> for $alg_type {
                #[inline]
                fn product<I>(iter: I) -> $alg_type
                where
                    I: Iterator<Item = $field_type>,
                {
                    iter.product::<$field_type>().into()
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
