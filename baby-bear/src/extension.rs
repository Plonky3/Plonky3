use p3_field::extension::{BinomiallyExtendable, HasFrobenuis};
use p3_field::AbstractField;

use crate::BabyBear;

impl BinomiallyExtendable<4> for BabyBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^4 - 11).is_irreducible()`.
    const W: Self = Self::new(11);

    fn ext_multiplicative_group_generator() -> [Self; 4] {
        [Self::new(8), Self::ONE, Self::ZERO, Self::ZERO]
    }
}
impl HasFrobenuis<4> for BabyBear {
    // DTH_ROOT = W^((p - 1)/4)
    const DTH_ROOT: Self = Self::new(1728404513);
}

impl BinomiallyExtendable<5> for BabyBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^5 - 2).is_irreducible()`.
    const W: Self = Self::new(2);

    fn ext_multiplicative_group_generator() -> [Self; 5] {
        [Self::new(8), Self::ONE, Self::ZERO, Self::ZERO, Self::ZERO]
    }
}
impl HasFrobenuis<5> for BabyBear {
    // DTH_ROOT = W^((p - 1)/5)
    const DTH_ROOT: Self = Self::new(815036133);
}

#[cfg(test)]
mod test_tesseractic_extension {

    use p3_field_testing::test_field;

    test_field!(p3_field::extension::tesseractic::TesseracticBef<crate::BabyBear>);
}
#[cfg(test)]
mod test_quintic_extension {

    use p3_field_testing::test_field;

    test_field!(p3_field::extension::quintic::QuinticBef<crate::BabyBear>);
}
