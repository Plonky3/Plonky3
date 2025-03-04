use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBinomialExtension};
use p3_field::{field_to_array, PrimeCharacteristicRing, TwoAdicField};

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

impl BinomiallyExtendable<5> for Goldilocks {
    // Verifiable via:
    //  ```sage
    //  # Define Fp
    //  p = 2**64 - 2**32 + 1
    //  F = GF(p)

    //  # Define Fp[z]
    //  R.<z> = PolynomialRing(F)

    //  # The polynomial x^5-3 is irreducible
    //  assert(R(z^5-3).is_irreducible())
    //  ```
    const W: Self = Self::new(3);

    // 5-th root = w^((p - 1)/5)
    const DTH_ROOT: Self = Self::new(1041288259238279555);

    // Generator of the extension field
    // Obtained by finding the smallest Hamming weight vector
    // with appropriate order, starting at [0,1,0,0,0]
    const EXT_GENERATOR: [Self; 5] = [Self::TWO, Self::ONE, Self::ZERO, Self::ZERO, Self::ZERO];
}

impl HasTwoAdicBinomialExtension<5> for Goldilocks {
    const EXT_TWO_ADICITY: usize = 32;

    fn ext_two_adic_generator(bits: usize) -> [Self; 5] {
        assert!(bits <= 32);

        field_to_array(Self::two_adic_generator(bits))
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

#[cfg(test)]
mod test_quintic_extension {

    use p3_field::extension::BinomialExtensionField;
    use p3_field_testing::{test_add_neg_sub_mul, test_inv_div, test_inverse, test_two_adic_extension_field};

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 5>;

    #[test]
    fn test_add_neg_sub_mul_w() {
        test_add_neg_sub_mul::<EF>();
    }

    #[test]
    fn test_inv_div_w() {
        test_inv_div::<EF>();
    }

    #[test]
    fn test_inverse_w() {
        test_inverse::<EF>();
    }

    test_two_adic_extension_field!(super::F, super::EF);
}
