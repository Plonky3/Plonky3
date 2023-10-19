use p3_field::extension::{BinomiallyExtendable, HasTwoAdicBionmialExtension};
use p3_field::{field_to_array, AbstractField, TwoAdicField};

use crate::BabyBear;

impl BinomiallyExtendable<4> for BabyBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^4 - 11).is_irreducible()`.
    fn w() -> Self {
        Self::new(11)
    }

    // DTH_ROOT = W^((p - 1)/4)
    fn dth_root() -> Self {
        Self::new(1728404513)
    }

    fn ext_generator() -> [Self; 4] {
        [Self::new(8), Self::one(), Self::zero(), Self::zero()]
    }
}

impl HasTwoAdicBionmialExtension<4> for BabyBear {
    const EXT_TWO_ADICITY: usize = 29;

    fn ext_two_adic_generator(bits: usize) -> [Self; 4] {
        assert!(bits <= 29);

        match bits {
            29 => [
                Self::zero(),
                Self::zero(),
                Self::zero(),
                Self::from_canonical_u32(124907976),
            ],
            28 => [
                Self::zero(),
                Self::zero(),
                Self::from_canonical_u32(1996171314),
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

impl BinomiallyExtendable<5> for BabyBear {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^5 - 2).is_irreducible()`.
    fn w() -> Self {
        Self::new(2)
    }

    // DTH_ROOT = W^((p - 1)/5)
    fn dth_root() -> Self {
        Self::new(815036133)
    }

    fn ext_generator() -> [Self; 5] {
        [
            Self::new(8),
            Self::one(),
            Self::zero(),
            Self::zero(),
            Self::zero(),
        ]
    }
}

impl HasTwoAdicBionmialExtension<5> for BabyBear {
    const EXT_TWO_ADICITY: usize = 27;

    fn ext_two_adic_generator(bits: usize) -> [Self; 5] {
        field_to_array::<Self, 5>(Self::two_adic_generator(bits))
    }
}

#[cfg(test)]
mod test_quartic_extension {
    use alloc::format;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::BabyBear;

    type F = BabyBear;
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

#[cfg(test)]
mod test_quintic_extension {
    use p3_field::extension::BinomialExtensionField;
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::BabyBear;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 5>;

    test_field!(super::EF);
    test_two_adic_extension_field!(super::F, super::EF);
}
