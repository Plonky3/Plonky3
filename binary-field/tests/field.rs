//! Generic field-testing suites for the extension structure between byte-aligned tower levels.

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
