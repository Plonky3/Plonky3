use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::{AbstractField, PrimeField32, TwoAdicField};

use crate::KoalaBear;

// As p - 1 = 2^31 - 2^24 = 127 * 2^24, the KoalaBear field only admits small binomial extensions of degree 2, 4, 8.

impl BinomiallyExtendable<2> for KoalaBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 3).is_irreducible()`.
    fn w() -> Self {
        Self::new(3)
    }

    // DTH_ROOT = W^((p - 1)/2) = -1
    fn dth_root() -> Self {
        Self::new(Self::ORDER_U32 - 1)
    }

    fn ext_generator() -> [Self; 2] {
        [Self::new(3), Self::one()]
    }
}

impl HasTwoAdicBionmialExtension<2> for KoalaBear {
    const EXT_TWO_ADICITY: usize = 25;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        assert!(bits <= 25);

        match bits {
            25 => [Self::zero(), Self::from_canonical_u32(1759267465)],
            _ => [Self::two_adic_generator(bits), Self::zero()],
        }
    }
}

// Might be worth writing this as a tower of extensions.

impl BinomiallyExtendable<4> for KoalaBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^4 - 3).is_irreducible()`.
    fn w() -> Self {
        Self::new(3)
    }

    // DTH_ROOT = W^((p - 1)/4)
    fn dth_root() -> Self {
        Self::new(2113994754)
    }

    fn ext_generator() -> [Self; 4] {
        [Self::new(2), Self::one(), Self::zero(), Self::zero()]
    }
}

impl HasTwoAdicBionmialExtension<4> for KoalaBear {
    const EXT_TWO_ADICITY: usize = 26;

    fn ext_two_adic_generator(bits: usize) -> [Self; 4] {
        assert!(bits <= 26);

        match bits {
            26 => [
                Self::zero(),
                Self::zero(),
                Self::zero(),
                Self::from_canonical_u32(777715144),
            ],
            25 => [
                Self::zero(),
                Self::zero(),
                Self::from_canonical_u32(1759267465),
                Self::zero(),
            ],
            _ => [
                Self::two_adic_generator(bits),
                Self::zero(),
                Self::zero(),
                Self::zero(),
            ],
        }
    }
}

#[cfg(test)]
mod test_quadratic_extension {
    use alloc::format;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::KoalaBear;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 2>;

    test_field!(super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::zero()), "0");
        assert_eq!(format!("{}", EF::one()), "1");
        assert_eq!(format!("{}", EF::two()), "2");

        assert_eq!(
            format!("{}", EF::from_base_slice(&[F::two(), F::one()])),
            "2 + X"
        );
    }
}

#[cfg(test)]
mod test_quartic_extension {
    use alloc::format;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::KoalaBear;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    test_field!(super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::zero()), "0");
        assert_eq!(format!("{}", EF::one()), "1");
        assert_eq!(format!("{}", EF::two()), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_base_slice(&[F::two(), F::one(), F::zero(), F::two()])
            ),
            "2 + X + 2 X^3"
        );
    }
}
