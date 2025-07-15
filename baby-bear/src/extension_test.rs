#[cfg(test)]
mod test_quartic_extension {
    use alloc::format;

    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::BabyBear;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^4 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 8] {
        [
            (BigUint::from(2u8), 29),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(97u8), 1),
            (BigUint::from(12241u16), 1),
            (BigUint::from(32472031u32), 1),
            (BigUint::from(1706804017873u64), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
    test_extension_field!(super::F, super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_basis_coefficients_slice(&[F::TWO, F::ONE, F::ZERO, F::TWO]).unwrap()
            ),
            "2 + X + 2 X^3"
        );
    }

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}

#[cfg(test)]
mod test_quintic_extension {
    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::BabyBear;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 5>;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^5 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 6] {
        [
            (BigUint::from(2u8), 27),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 2),
            (BigUint::from(26321u16), 1),
            (BigUint::from(1081891u32), 1),
            (BigUint::from(115384818561587951104978331u128), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
    test_extension_field!(super::F, super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}

#[cfg(test)]
mod test_octic_extension {
    use alloc::format;

    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::BabyBear;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 8>;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of p^8 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 12] {
        [
            (BigUint::from(2u8), 30),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 1),
            (BigUint::from(17u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(97u8), 1),
            (BigUint::from(12241u16), 1),
            (BigUint::from(1666201u32), 1),
            (BigUint::from(32472031u32), 1),
            (BigUint::from(74565857u32), 1),
            (BigUint::from(1706804017873u64), 1),
            (BigUint::from(3889181823063218424889u128), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
    test_extension_field!(super::F, super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_basis_coefficients_slice(&[
                    F::TWO,
                    F::ONE,
                    F::ZERO,
                    F::TWO,
                    F::TWO,
                    F::ZERO,
                    F::ZERO,
                    F::ONE
                ])
                .unwrap()
            ),
            "2 + X + 2 X^3 + 2 X^4 + X^7"
        );
    }

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}
