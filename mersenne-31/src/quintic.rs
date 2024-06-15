use core::ops::Mul;

use p3_field::{
    define_ext, field_to_array, impl_ext_af, impl_ext_f, AbstractExtensionField, AbstractField,
    Field,
};

use crate::Mersenne31;

// F[X]/(X^5 - X - 6)
define_ext!(pub M31x5([Mersenne31; 5]));

impl<AF: AbstractField<F = Mersenne31>> Mul for M31x5<AF> {
    type Output = Self;
    #[rustfmt::skip]
    fn mul(self, rhs: Self) -> Self::Output {
        let m = |i: usize, j: usize| self.0[i].clone() * rhs.0[j].clone();
        // prod = lo + hi * X^5
        //      = lo + hi * (X + 6)
        let lo = Self([
            m(0,0),
            m(1,0) + m(0,1),
            m(2,0) + m(1,1) + m(0,2),
            m(3,0) + m(2,1) + m(1,2) + m(0,3),
            m(4,0) + m(3,1) + m(2,2) + m(1,3) + m(0,4),
        ]);
        let hi = Self([
                     m(4,1) + m(3,2) + m(2,3) + m(1,4),
                              m(4,2) + m(3,3) + m(2,4),
                                       m(4,3) + m(3,4),
                                                m(4,4),
            AF::zero(),
        ]);

        // shift hi right to get hi * X
        let mut hi_times_x = Self::zero();
        hi_times_x.0[1..].clone_from_slice(&hi.0[..4]);

        // maybe better reduction possible for * 6 ?
        let w = AF::from_f(Mersenne31::from_canonical_u8(6));
        lo + hi * w + hi_times_x
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
}

impl<AF: AbstractField<F = Mersenne31>> AbstractExtensionField<AF> for M31x5<AF> {
    const D: usize = 5;

    fn from_base(b: AF) -> Self {
        Self(field_to_array(b))
    }

    fn from_base_slice(bs: &[AF]) -> Self {
        let mut me = Self::zero();
        for i in 0..5 {
            me.0[i] = bs[i].clone();
        }
        me
    }

    fn from_base_fn<F: FnMut(usize) -> AF>(f: F) -> Self {
        Self(core::array::from_fn(f))
    }

    fn as_base_slice(&self) -> &[AF] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_field;
    test_field!(crate::M31x5<crate::Mersenne31>);
}
