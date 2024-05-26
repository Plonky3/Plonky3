use p3_field::{
    AbstractExtension, AbstractExtensionAlgebra, AbstractField, BinomialExtensionAlgebra,
    BinomialExtensionParams, Complex, ComplexExtendable, Extension, Field, HasBase,
};

use crate::Mersenne31;

const fn m31(x: u32) -> Mersenne31 {
    Mersenne31::new(x)
}
const fn m31s<const N: usize>(xs: [u32; N]) -> [Mersenne31; N] {
    let mut ys = [Mersenne31::new(0); N];
    let mut i = 0;
    while i < N {
        ys[i] = m31(xs[i]);
        i += 1;
    }
    ys
}

impl ComplexExtendable for Mersenne31 {
    const COMPLEX_GEN: [Self; 2] = m31s([12, 1]);
    const CIRCLE_TWO_ADICITY: usize = 31;
    fn circle_two_adic_generator(bits: usize) -> Complex<Self> {
        let base = Complex::new(m31(311_014_874), m31(1_584_694_829));
        base.exp_power_of_2(Self::CIRCLE_TWO_ADICITY - bits)
    }
}

#[cfg(test)]
mod test_m31_complex {
    use super::*;
    use p3_field::Complex;
    use p3_field_testing::{test_complex, test_field};
    test_field!(Complex<Mersenne31>);
    test_complex!(Mersenne31);
}

pub type Mersenne31Cubic = BinomialExtensionAlgebra<Mersenne31, 3, Mersenne31CubicParams>;

#[derive(Debug)]
pub struct Mersenne31CubicParams;

impl BinomialExtensionParams<Mersenne31, 3> for Mersenne31CubicParams {
    const W: Mersenne31 = m31(5);
    const ORDER_D_SUBGROUP: [Mersenne31; 3] = m31s([1, 1513477735, 634005911]);
    const GEN: [Mersenne31; 3] = m31s([10, 1, 0]);
}

#[cfg(test)]
mod test_m31_cubic {
    use super::*;
    use p3_field_testing::test_field;
    test_field!(Extension<Mersenne31Cubic>);
}

#[derive(Debug)]
pub struct Mersenne31Quintic;

impl HasBase for Mersenne31Quintic {
    type Base = Mersenne31;
}

impl AbstractExtensionAlgebra for Mersenne31Quintic {
    const D: usize = 5;
    type Repr<AF: AbstractField<F = Self::Base>> = [AF; 5];

    const GEN: Self::Repr<Self::Base> = m31s([4, 1, 0, 0, 0]);

    fn mul<AF: AbstractField<F = Self::Base>>(
        a: AbstractExtension<AF, Self>,
        b: AbstractExtension<AF, Self>,
    ) -> AbstractExtension<AF, Self> {
        let m = |i: usize, j: usize| a[i].clone() * b[j].clone();
        // prod = lo + hi * X^5
        //      = lo + hi * (X + 6)
        let lo = AbstractExtension::<AF, Self>([
            m(0, 0),
            m(1, 0) + m(0, 1),
            m(2, 0) + m(1, 1) + m(0, 2),
            m(3, 0) + m(2, 1) + m(1, 2) + m(0, 3),
            m(4, 0) + m(3, 1) + m(2, 2) + m(1, 3) + m(0, 4),
        ]);
        let hi = AbstractExtension::<AF, Self>([
            m(4, 1) + m(3, 2) + m(2, 3) + m(1, 4),
            m(4, 2) + m(3, 3) + m(2, 4),
            m(4, 3) + m(3, 4),
            m(4, 4),
            AF::zero(),
        ]);

        // shift hi right to get hi * X
        let mut hi_times_x = AbstractExtension::<AF, Self>::zero();
        hi_times_x.0[1..].clone_from_slice(&hi.0[..4]);

        // maybe better reduction possible for * 6 ?
        let w = AF::from_f(m31(6));
        lo + hi * w + hi_times_x
    }

    fn repeated_frobenius<AF: AbstractField<F = Self::Base>>(
        a: AbstractExtension<AF, Self>,
        count: usize,
    ) -> AbstractExtension<AF, Self> {
        a.exp_biguint(Mersenne31::order().pow((count % Self::D) as u32))
    }

    fn inverse(a: Extension<Self>) -> Extension<Self> {
        a.exp_biguint(Extension::<Self>::order() - 2u32)
    }
}

#[cfg(test)]
mod test_m31_quintic {
    use super::*;
    use p3_field::Extension;
    use p3_field_testing::test_field;
    test_field!(Extension<Mersenne31Quintic>);
}
