//! The generic characteristic-two ring suite over the bit-sliced packings of the two-element field.
//!
//! These were pinned only by hand-written tests of their own.
//!
//! The packed-field suite does not apply: they carry no scalar lanes to compare against.

use p3_field_testing::test_ring_with_eq_char2;

mod width_8 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x8,
        &[<::p3_binary_field::PackedGf2x8>::ZERO],
        &[<::p3_binary_field::PackedGf2x8>::ONE]
    );
}

mod width_16 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x16,
        &[<::p3_binary_field::PackedGf2x16>::ZERO],
        &[<::p3_binary_field::PackedGf2x16>::ONE]
    );
}

mod width_32 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x32,
        &[<::p3_binary_field::PackedGf2x32>::ZERO],
        &[<::p3_binary_field::PackedGf2x32>::ONE]
    );
}

mod width_64 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x64,
        &[<::p3_binary_field::PackedGf2x64>::ZERO],
        &[<::p3_binary_field::PackedGf2x64>::ONE]
    );
}

mod width_128 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x128,
        &[<::p3_binary_field::PackedGf2x128>::ZERO],
        &[<::p3_binary_field::PackedGf2x128>::ONE]
    );
}

mod width_256 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x256,
        &[<::p3_binary_field::PackedGf2x256>::ZERO],
        &[<::p3_binary_field::PackedGf2x256>::ONE]
    );
}

mod width_512 {
    crate::test_ring_with_eq_char2!(
        ::p3_binary_field::PackedGf2x512,
        &[<::p3_binary_field::PackedGf2x512>::ZERO],
        &[<::p3_binary_field::PackedGf2x512>::ONE]
    );
}
