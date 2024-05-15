use p3_field::BinomialExtensionParams;

use crate::Mersenne31;

#[derive(Debug)]
pub struct Mersenne31Cubic;

impl BinomialExtensionParams<Mersenne31, 3> for Mersenne31Cubic {
    const W: Mersenne31 = Mersenne31::new(5);
    const ORDER_D_SUBGROUP: [Mersenne31; 3] = [
        Mersenne31::new(1),
        Mersenne31::new(1513477735),
        Mersenne31::new(634005911),
    ];
    const GEN: [Mersenne31; 3] = [Mersenne31::new(10), Mersenne31::new(1), Mersenne31::new(0)];
}

#[cfg(test)]
mod tests {
    use p3_field::{BinomialExtensionAlgebra, Extension};
    use p3_field_testing::{test_add_neg_sub_mul, test_field, test_inv_div};

    use super::*;

    type M31 = Mersenne31;
    type M31x3 = Extension<M31, BinomialExtensionAlgebra<M31, 3, Mersenne31Cubic>>;

    // test_field!(super::M31x3);
    #[test]
    fn test_cubic() {
        test_add_neg_sub_mul::<M31x3>();
        test_inv_div::<M31x3>();
    }
}
