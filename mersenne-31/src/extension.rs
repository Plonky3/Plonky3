use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::{field_to_array, AbstractField, TwoAdicField};

use crate::{Mersenne31, Mersenne31Complex};

impl BinomiallyExtendable<2> for Mersenne31Complex<Mersenne31> {
    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^2 - i - 2
    // assert f2.is_irreducible()
    // ```
    fn w() -> Self {
        Self::new(Mersenne31::new(2), Mersenne31::one())
    }

    // Verifiable in Sage with
    // ```sage
    // K2.<j> = K.extension(f2)
    //  g = j + 6
    // for f in factor(p^4 - 1):
    //   assert g^((p^4-1) // f) != 1
    // ```
    fn ext_multiplicative_group_generator() -> [Self; 2] {
        [Self::new_real(Mersenne31::new(6)), Self::one()]
    }

    // DTH_ROOT = W^((p^2 - 1)/2).
    fn dth_root() -> Self {
        Self::new_real(Mersenne31::new(2147483646))
    }
}

impl HasTwoAdicBionmialExtension<2> for Mersenne31Complex<Mersenne31> {
    const EXT_TWO_ADICITY: usize = 33;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        assert!(bits <= 33);
        if bits == 33 {
            [
                Self::zero(),
                Self::new(Mersenne31::new(1437746044), Mersenne31::new(946469285)),
            ]
        } else {
            [Self::two_adic_generator(bits), Self::zero()]
        }
    }
}

impl BinomiallyExtendable<3> for Mersenne31Complex<Mersenne31> {
    // Verifiable in Sage with
    // ```sage
    // p = 2**31 - 1  # Mersenne31
    // F = GF(p)  # The base field GF(p)
    // R.<x> = F[]  # The polynomial ring over F
    // K.<i> = F.extension(x^2 + 1)  # The complex extension field
    // R2.<y> = K[]
    // f2 = y^3 - 5*i
    // assert f2.is_irreducible()
    // ```
    fn w() -> Self {
        Self::new_imag(Mersenne31::new(5))
    }

    // DTH_ROOT = W^((p^2 - 1)/2).
    fn dth_root() -> Self {
        Self::new_real(Mersenne31::new(634005911))
    }

    // Verifiable in Sage with
    // ```sage
    // K2.<j> = K.extension(f2)
    //  g = j + 5
    // for f in factor(p^6 - 1):
    //   assert g^((p^6-1) // f) != 1
    // ```
    fn ext_multiplicative_group_generator() -> [Self; 3] {
        [
            Self::new_real(Mersenne31::new(5)),
            Self::new_real(Mersenne31::one()),
            Self::zero(),
        ]
    }
}

impl HasTwoAdicBionmialExtension<3> for Mersenne31Complex<Mersenne31> {
    const EXT_TWO_ADICITY: usize = 32;

    fn ext_two_adic_generator(bits: usize) -> [Self; 3] {
        field_to_array::<Self, 3>(Self::two_adic_generator(bits))
    }
}

#[cfg(test)]
mod test_cubic_extension {
    use p3_field::extension::BinomialExtensionField;
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::{Mersenne31, Mersenne31Complex};

    type F = Mersenne31Complex<Mersenne31>;
    type EF = BinomialExtensionField<F, 3>;

    test_field!(super::EF);

    test_two_adic_extension_field!(super::F, super::EF);
}

#[cfg(test)]
mod test_quadratic_extension {

    use p3_field::extension::BinomialExtensionField;
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::{Mersenne31, Mersenne31Complex};

    type F = Mersenne31Complex<Mersenne31>;
    type EF = BinomialExtensionField<F, 2>;

    test_field!(super::EF);

    test_two_adic_extension_field!(super::F, super::EF);
}
