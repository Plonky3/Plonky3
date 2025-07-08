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

pub use {field_div_assign, ring_add_assign, ring_mul_assign, ring_sub_assign};
