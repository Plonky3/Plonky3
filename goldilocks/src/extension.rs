use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBinomialExtension};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};

use crate::Goldilocks;

impl BinomiallyExtendable<2> for Goldilocks {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    const W: Self = Self::new(7);

    // DTH_ROOT = W^((p - 1)/2).
    const DTH_ROOT: Self = Self::new(18446744069414584320);

    const EXT_GENERATOR: [Self; 2] = [
        Self::new(18081566051660590251),
        Self::new(16121475356294670766),
    ];
}

impl HasTwoAdicBinomialExtension<2> for Goldilocks {
    const EXT_TWO_ADICITY: usize = 33;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        assert!(bits <= 33);

        if bits == 33 {
            [Self::ZERO, Self::new(15659105665374529263)]
        } else {
            [Self::two_adic_generator(bits), Self::ZERO]
        }
    }
}

#[cfg(test)]
mod test_quadratic_extension {

    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    test_field!(super::EF, &super::ZEROS, &super::ONES);

    test_two_adic_extension_field!(super::F, super::EF);
}
