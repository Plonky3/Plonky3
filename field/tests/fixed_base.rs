use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, FixedBase, PrimeCharacteristicRing};
use p3_field_testing::test_fixed_base;
use proptest::prelude::*;

type EF4 = BinomialExtensionField<BabyBear, 4>;

fn ef4(coefficients: [u32; 4]) -> EF4 {
    EF4::from_basis_coefficients_fn(|i| BabyBear::new(coefficients[i]))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn ef4_powers_match_square_and_multiply(
        base: [u32; 4],
        one_limb in prop::collection::vec(any::<[u64; 1]>(), 0..40),
        two_limbs in prop::collection::vec(any::<[u64; 2]>(), 0..40),
    ) {
        let base = ef4(base);
        // Lanes of the base field's packing, then the extension as its own packing.
        test_fixed_base::<BabyBear, EF4, 1>(base, &one_limb);
        test_fixed_base::<BabyBear, EF4, 2>(base, &two_limbs);
        test_fixed_base::<EF4, EF4, 2>(base, &two_limbs);
    }

    #[test]
    fn prime_field_powers_match_square_and_multiply(
        base: u32,
        exponents in prop::collection::vec(any::<[u64; 2]>(), 0..40),
    ) {
        test_fixed_base::<BabyBear, BabyBear, 2>(BabyBear::new(base), &exponents);
    }
}

#[test]
fn an_empty_range_holds_only_the_zero_exponent() {
    let table = FixedBase::new(EF4::GENERATOR, 0);
    assert_eq!(table.pow(&[0u64; 2]), EF4::ONE);
    let mut out = [EF4::ZERO; 3];
    table.pow_into::<BabyBear, 2>(&[[0; 2]; 3], &mut out);
    assert_eq!(out, [EF4::ONE; 3]);
}

#[test]
#[should_panic(expected = "exceeds")]
fn pow_rejects_an_exponent_past_the_bound() {
    let table = FixedBase::with_window(EF4::GENERATOR, 70, 8);
    let _ = table.pow(&[0, 1 << 6]);
}

#[test]
#[should_panic(expected = "exceeds")]
fn pow_into_rejects_an_exponent_past_the_bound() {
    let table = FixedBase::with_window(EF4::GENERATOR, 70, 8);
    // The offending exponent sits inside a full lockstep group.
    let mut exponents = vec![[1u64, 0]; 4096];
    exponents[17] = [0, 1 << 6];
    let mut out = vec![EF4::ZERO; exponents.len()];
    table.pow_into::<BabyBear, 2>(&exponents, &mut out);
}

#[test]
#[should_panic(expected = "one output per exponent")]
fn pow_into_rejects_mismatched_lengths() {
    let table = FixedBase::new(EF4::GENERATOR, 64);
    let mut out = [EF4::ZERO; 2];
    table.pow_into::<BabyBear, 1>(&[[1]; 3], &mut out);
}

#[test]
#[should_panic(expected = "window")]
fn a_zero_window_is_rejected() {
    let _ = FixedBase::with_window(EF4::GENERATOR, 64, 0);
}

#[test]
#[should_panic(expected = "window")]
fn an_oversized_window_is_rejected() {
    let _ = FixedBase::with_window(EF4::GENERATOR, 64, 17);
}
