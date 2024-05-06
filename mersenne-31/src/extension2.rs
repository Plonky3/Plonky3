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

#[cfg(test)]
mod tests {
    use p3_field_testing::test_field;
    test_field!(crate::CM31<crate::Mersenne31>);
}
