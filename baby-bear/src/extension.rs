use p3_field::extension::BinomiallyExtendable;
use p3_field::AbstractField;

use crate::BabyBear;

impl BinomiallyExtendable<4> for BabyBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^4 - 11).is_irreducible()`.
    const W: Self = Self::new(11);

    // DTH_ROOT = W^((p - 1)/4)
    const DTH_ROOT: Self = Self::new(1728404513);

    fn ext_multiplicative_group_generator() -> [Self; 4] {
        [Self::new(8), Self::ONE, Self::ZERO, Self::ZERO]
    }
}

impl BinomiallyExtendable<5> for BabyBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^5 - 2).is_irreducible()`.
    const W: Self = Self::TWO;

    // DTH_ROOT = W^((p - 1)/5)
    const DTH_ROOT: Self = Self::new(815036133);

    fn ext_multiplicative_group_generator() -> [Self; 5] {
        [Self::new(8), Self::ONE, Self::ZERO, Self::ZERO, Self::ZERO]
    }
}

#[cfg(test)]
mod test_quatric_extension {

    use p3_field_testing::test_field;

    test_field!(p3_field::extension::binomial_extension::BinomialExtensionField<crate::BabyBear,4>);
}
#[cfg(test)]
mod test_quintic_extension {

    use p3_field_testing::test_field;

    test_field!(p3_field::extension::binomial_extension::BinomialExtensionField<crate::BabyBear,5>);
}
