#[macro_export]
macro_rules! make_div_assign {
    ( $T:ty ) => {
        impl core::ops::DivAssign for $T {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    };
}
