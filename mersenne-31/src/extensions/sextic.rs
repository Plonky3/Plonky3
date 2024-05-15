use p3_field::{define_binomial_ext, AbstractField};

use crate::Mersenne31;

define_binomial_ext!(
    pub M31Sextic,
    pub AbstractM31Sextic<AF>,
    [Mersenne31; 6],
    w = Mersenne31::new(5),
    // todo
    gen = Self::constant([1, 1, 1, 1, 1, 1]),
    order_d_subgroup = [],
);

impl<AF: AbstractField<F = Mersenne31>> AbstractM31Sextic<AF> {
    fn constant(values: [u32; 6]) -> Self {
        Self(values.map(|x| AF::from_f(Mersenne31::new(x))))
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_field;
    // test_field!(crate::M31Sextic);
}
