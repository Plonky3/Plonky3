//! Generic field-testing suites: the per-level `test_binary_field!` base suite for every tower
//! level, `Gf2`'s own coverage, and the extension structure between all byte-aligned tower level
//! pairs.

/// `p3_field_testing::test_prime_field!` is not usable against [`Gf2`](p3_binary_field::Gf2):
/// its `generate_from_small_int_tests!` sub-test hard-codes literals up to `108` as canonical
/// representatives, which only holds for fields with `order() > 108`. `Gf2`'s canonical range
/// is `{0, 1}`, so every literal above `1` makes `from_canonical_checked` return `None`, which
/// the macro asserts is `Some`.
///
/// The passing subset is listed explicitly below: `generate_from_int_tests!` restricted to the
/// two values that are canonical for `Gf2` (`0` and `1`), across every integer type
/// `test_prime_field!` covers. `binary_field::gf2::tests` (in `p3-binary-field`'s own unit
/// tests) carries the rest of `Gf2`'s `QuotientMap` coverage, including large-value reduction,
/// for every one of these integer types.
mod gf2_prime_field {
    use p3_binary_field::Gf2;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::integers::QuotientMap;
    use p3_field_testing::generate_from_int_tests;

    #[test]
    fn test_canonical_zero_and_one_every_integer_type() {
        generate_from_int_tests!(Gf2, 0u8, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1u8, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0u16, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1u16, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0u32, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1u32, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0u64, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1u64, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0u128, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1u128, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0usize, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1usize, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0i8, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1i8, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0i16, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1i16, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0i32, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1i32, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0i64, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1i64, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0i128, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1i128, Gf2::ONE);
        generate_from_int_tests!(Gf2, 0isize, Gf2::ZERO);
        generate_from_int_tests!(Gf2, 1isize, Gf2::ONE);
    }
}

mod binary_field_2 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField2;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^2 - 1 = 3`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 1] {
        [(BigUint::from(3u32), 1)]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod binary_field_4 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField4;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^4 - 1 = 15`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 2] {
        [(BigUint::from(3u32), 1), (BigUint::from(5u32), 1)]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod binary_field_8 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField8;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^8 - 1 = 255`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 3] {
        [
            (BigUint::from(3u32), 1),
            (BigUint::from(5u32), 1),
            (BigUint::from(17u32), 1),
        ]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod binary_field_16 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField16;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^16 - 1 = 65535`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 4] {
        [
            (BigUint::from(3u32), 1),
            (BigUint::from(5u32), 1),
            (BigUint::from(17u32), 1),
            (BigUint::from(257u32), 1),
        ]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod binary_field_32 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField32;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^32 - 1 = 4294967295`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 5] {
        [
            (BigUint::from(3u32), 1),
            (BigUint::from(5u32), 1),
            (BigUint::from(17u32), 1),
            (BigUint::from(257u32), 1),
            (BigUint::from(65537u32), 1),
        ]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod binary_field_64 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField64;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^64 - 1`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 7] {
        [
            (BigUint::from(3u32), 1),
            (BigUint::from(5u32), 1),
            (BigUint::from(17u32), 1),
            (BigUint::from(257u32), 1),
            (BigUint::from(641u32), 1),
            (BigUint::from(65537u32), 1),
            (BigUint::from(6700417u32), 1),
        ]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod binary_field_128 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::BinaryField128;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^128 - 1`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 9] {
        [
            (BigUint::from(3u32), 1),
            (BigUint::from(5u32), 1),
            (BigUint::from(17u32), 1),
            (BigUint::from(257u32), 1),
            (BigUint::from(641u32), 1),
            (BigUint::from(65537u32), 1),
            (BigUint::from(274177u32), 1),
            (BigUint::from(6700417u32), 1),
            (BigUint::from(67280421310721u64), 1),
        ]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

/// The same field as the widest tower level, in the polynomial basis of the GHASH modulus.
///
/// None of its arithmetic is shared with the tower's, so it runs the whole base suite.
mod ghash_128 {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_binary_field;

    type F = p3_binary_field::Ghash128;

    const ZEROS: [F; 1] = [F::ZERO];
    const ONES: [F; 1] = [F::ONE];

    /// Prime factorization of `2^128 - 1`.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 9] {
        [
            (BigUint::from(3u32), 1),
            (BigUint::from(5u32), 1),
            (BigUint::from(17u32), 1),
            (BigUint::from(257u32), 1),
            (BigUint::from(641u32), 1),
            (BigUint::from(65537u32), 1),
            (BigUint::from(274177u32), 1),
            (BigUint::from(6700417u32), 1),
            (BigUint::from(67280421310721u64), 1),
        ]
    }

    test_binary_field!(
        super::F,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
}

mod gf2_16_over_gf2_8 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField8;
    type EF = p3_binary_field::BinaryField16;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_32_over_gf2_8 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField8;
    type EF = p3_binary_field::BinaryField32;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_64_over_gf2_8 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField8;
    type EF = p3_binary_field::BinaryField64;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_128_over_gf2_8 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField8;
    type EF = p3_binary_field::BinaryField128;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_32_over_gf2_16 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField16;
    type EF = p3_binary_field::BinaryField32;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_64_over_gf2_16 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField16;
    type EF = p3_binary_field::BinaryField64;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_128_over_gf2_16 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField16;
    type EF = p3_binary_field::BinaryField128;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_64_over_gf2_32 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField32;
    type EF = p3_binary_field::BinaryField64;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_128_over_gf2_32 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField32;
    type EF = p3_binary_field::BinaryField128;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}

mod gf2_128_over_gf2_64 {
    use p3_field_testing::{test_extension_field, test_frobenius};

    type F = p3_binary_field::BinaryField64;
    type EF = p3_binary_field::BinaryField128;

    test_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);
}
