#[cfg(test)]
mod test_quartic_extension {
    use alloc::format;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{FieldExtensionAlgebra, PrimeCharacteristicRing};
    use p3_field_testing::{test_field, test_two_adic_extension_field};

    use crate::BabyBear;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    test_field!(super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_base_slice(&[F::TWO, F::ONE, F::ZERO, F::TWO])
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
