use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::{AbstractField, TwoAdicField};

use crate::Goldilocks;

impl BinomiallyExtendable<2> for Goldilocks {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    fn w() -> Self {
        Self::new(7)
    }

    // DTH_ROOT = W^((p - 1)/2).
    fn dth_root() -> Self {
        Self::new(18_446_744_069_414_584_320)
    }

    fn ext_generator() -> [Self; 2] {
        [
            Self::new(18_081_566_051_660_590_251),
            Self::new(16_121_475_356_294_670_766),
        ]
    }
}

impl HasTwoAdicBionmialExtension<2> for Goldilocks {
    const EXT_TWO_ADICITY: usize = 33;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        assert!(bits <= 33);

        if bits == 33 {
            [Self::zero(), Self::new(15_659_105_665_374_529_263)]
        } else {
            [Self::two_adic_generator(bits), Self::zero()]
        }
    }
}

#[cfg(test)]
mod test_quadratic_extension {

    use p3_field::extension::BinomialExtensionField;
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;

    test_field!(super::EF);

    test_two_adic_extension_field!(super::F, super::EF);
}
