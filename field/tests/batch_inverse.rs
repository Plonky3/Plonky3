use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{
    BasedVectorSpace, Field, PrimeCharacteristicRing, batch_multiplicative_inverse,
    batch_multiplicative_inverse_map,
};
use p3_goldilocks::Goldilocks;

#[test]
fn mapped_inverses_match_individual_inverses() {
    fn check<F: Field>() {
        struct Input(u32);

        for n in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1023, 1024, 1025, 1026, 1027, 2048, 2051,
        ] {
            let inputs = (0..n).map(|i| Input(i as u32)).collect::<Vec<_>>();
            let mut output = vec![F::ZERO; n];
            batch_multiplicative_inverse_map(
                &inputs,
                |input| F::from_u32(input.0 + 1),
                &mut output,
            );
            for (i, &inverse) in output.iter().enumerate() {
                assert_eq!(inverse, F::from_u32(i as u32 + 1).inverse(), "n={n}, i={i}");
            }
            let mapped = (1..=n).map(|i| F::from_u32(i as u32)).collect::<Vec<_>>();
            assert_eq!(batch_multiplicative_inverse(&mapped), output);
        }
    }

    check::<BabyBear>();
    check::<Goldilocks>();
    check::<BinomialExtensionField<BabyBear, 4>>();
}

#[test]
fn mapped_inverses_support_extension_valued_maps() {
    type EF = BinomialExtensionField<BabyBear, 4>;
    let z = EF::from_basis_coefficients_slice(&[
        BabyBear::ONE,
        BabyBear::TWO,
        BabyBear::ZERO,
        BabyBear::ONE,
    ])
    .unwrap();
    let inputs = (0..2051).map(BabyBear::from_u32).collect::<Vec<_>>();
    let mut output = EF::zero_vec(inputs.len());
    batch_multiplicative_inverse_map(&inputs, |&x| z - x, &mut output);
    for (&x, inverse) in inputs.iter().zip(output) {
        assert_eq!(inverse, (z - x).inverse());
    }
}

#[test]
fn mapped_inverse_does_not_evaluate_an_empty_input() {
    batch_multiplicative_inverse_map::<u32, BabyBear, _>(
        &[],
        |_| panic!("empty input must not be mapped"),
        &mut [],
    );
}

#[test]
#[should_panic(expected = "input and output lengths must match")]
fn mapped_inverse_rejects_a_short_output() {
    batch_multiplicative_inverse_map(&[BabyBear::ONE], |&x| x, &mut []);
}

#[test]
#[should_panic(expected = "input and output lengths must match")]
fn mapped_inverse_rejects_a_long_output() {
    batch_multiplicative_inverse_map(&[], |x: &BabyBear| *x, &mut [BabyBear::ONE]);
}

#[test]
fn mapped_inverse_rejects_zero_in_any_pass() {
    for (n, zero_index) in [
        (1, 0),
        (4, 3),
        (1027, 1024),
        (1027, 1026),
        (2051, 1025),
        (2051, 2048),
    ] {
        let mut inputs = vec![2u32; n];
        inputs[zero_index] = 1;
        let panic = std::panic::catch_unwind(|| {
            let mut output = BabyBear::zero_vec(n);
            batch_multiplicative_inverse_map(
                &inputs,
                |&x| BabyBear::from_u32(x) - BabyBear::ONE,
                &mut output,
            );
        })
        .expect_err("a mapped zero must panic");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap();
        assert!(message.contains("Tried to invert zero"), "{message}");
    }
}
