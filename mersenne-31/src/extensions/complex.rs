use core::ops::Mul;

use crate::Mersenne31;
use p3_field::{
    binomial_extension, define_binomial_ext, define_ext, impl_ext_af, impl_ext_f, AbstractField,
    Field,
};

define_binomial_ext!(
    pub M31Complex,
    pub AbstractM31Complex<AF>,
    [Mersenne31; 2],
    complex,
    gen = Self::constant([12, 1]),
    order_d_subgroup = Self::constant([1, 0]).0,
);

impl<AF: AbstractField<F = Mersenne31>> AbstractM31Complex<AF> {
    fn constant(values: [u32; 2]) -> Self {
        Self(values.map(|x| AF::from_f(Mersenne31::new(x))))
    }
}

// Equivalent to CM31\[x\] over (x^2 - 2 - i) as the irreducible polynomial.
define_ext!(
    pub M31ToweredQuartic,
    pub AbstractM31ToweredQuartic<AF>,
    [Mersenne31; 4],
);

impl<AF: AbstractField<F = Mersenne31>> AbstractM31ToweredQuartic<AF> {
    fn constant(values: [u32; 4]) -> Self {
        Self(values.map(|x| AF::from_f(Mersenne31::new(x))))
    }
    fn to_complex_pair(self) -> [AbstractM31Complex<AF>; 2] {
        let [x0, x1, x2, x3] = self.0;
        [AbstractM31Complex([x0, x1]), AbstractM31Complex([x2, x3])]
    }
    fn from_complex_pair([c0, c1]: [AbstractM31Complex<AF>; 2]) -> Self {
        let [x0, x1] = c0.0;
        let [x2, x3] = c1.0;
        Self([x0, x1, x2, x3])
    }
}

impl<AF: AbstractField<F = Mersenne31>> Mul for AbstractM31ToweredQuartic<AF> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let [x0, x1] = binomial_extension::quadratic::mul(
            &self.to_complex_pair(),
            &rhs.to_complex_pair(),
            M31Complex::constant([2, 1]),
        );
        Self::from_complex_pair([x0, x1])
    }
}

impl<AF: AbstractField<F = Mersenne31>> AbstractField for AbstractM31ToweredQuartic<AF> {
    type F = M31ToweredQuartic;
    impl_ext_af!();
    fn generator() -> Self {
        Self::constant(todo!())
    }
}

impl Field for M31ToweredQuartic {
    impl_ext_f!(Mersenne31, 4);
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_field;
    test_field!(crate::M31Complex);
}
