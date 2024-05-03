use core::ops::Mul;

use p3_field::{define_ext, impl_ext_af, impl_ext_f, AbstractField, Field};

use crate::Mersenne31;

// F[X]/(X^5 - X - 6)
define_ext!(pub M31x5([Mersenne31; 5]));

impl<AF: AbstractField<F = Mersenne31>> Mul for M31x5<AF> {
    type Output = Self;
    #[rustfmt::skip]
    fn mul(self, rhs: Self) -> Self::Output {
        let m = |i: usize, j: usize| self.0[i].clone() * rhs.0[j].clone();
        let w = |e: AF| e * AF::from_f(Mersenne31::from_canonical_u32(6));
        // todo: see the pattern? (add w part)
        Self([
        	m(0,0) + w(m(4,1) +   m(3,2) +   m(2,3) +   m(1,4)),
        	m(1,0) +   m(0,1) + w(m(4,2) +   m(3,3) +   m(2,4)) + m(1,4) + m(2,3) + m(3,2) + m(4,1),
        	m(2,0) +   m(1,1) +   m(0,2) + w(m(4,3) +   m(3,4))          + m(2,4) + m(3,3) + m(4,2),
        	m(3,0) +   m(2,1) +   m(1,2) +   m(0,3) + w(m(4,4))                   + m(3,4) + m(4,3),
        	m(4,0) +   m(3,1) +   m(2,2) +   m(1,3) +   m(0,4)                             + m(4,4),
        ])
    }
}

impl<AF: AbstractField<F = Mersenne31>> AbstractField for M31x5<AF> {
    impl_ext_af!();
    type F = M31x5<Mersenne31>;
    /// F.extension(x^5 - x - 6, 'u').multiplicative_generator()
    /// u + 4
    fn generator() -> Self {
        Self(
            [
                Mersenne31::new(4),
                Mersenne31::new(1),
                Mersenne31::zero(),
                Mersenne31::zero(),
                Mersenne31::zero(),
            ]
            .map(AF::from_f),
        )
    }
}

impl Field for M31x5<Mersenne31> {
    impl_ext_f!(Mersenne31, 5);
    fn try_inverse(&self) -> Option<Self> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_add_neg_sub_mul;

    use super::*;

    #[test]
    fn quintic() {
        test_add_neg_sub_mul::<M31x5<Mersenne31>>()
    }
}
