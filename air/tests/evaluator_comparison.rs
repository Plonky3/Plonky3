#![cfg(feature = "testing")]

use p3_air::testing::{
    AssertionKind, ConstraintMismatch, ConstraintValue, EvaluationInputs, PackedEvaluationInputs,
    PackedMismatch, RecordingAirBuilder, compare_packed, compare_scalar,
};
use p3_air::{AirBuilder, AirLayout, ExtensionBuilder, PermutationAirBuilder, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_mersenne_31::{Mersenne31, QM31};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

const fn layout() -> AirLayout {
    AirLayout {
        main_width: 2,
        preprocessed_width: 1,
        num_public_values: 1,
        num_periodic_columns: 1,
        permutation_width: 1,
        num_permutation_challenges: 1,
        num_permutation_values: 1,
    }
}
fn fixture<F: Field, EF: ExtensionField<F>>(seed: usize) -> EvaluationInputs<F, EF> {
    let mut n = seed;
    let mut e = seed + 100;
    EvaluationInputs::from_fn(
        layout(),
        vec![F::from_u32(17)],
        || {
            n += 1;
            -F::from_usize(n)
        },
        || {
            e += 1;
            EF::from_basis_coefficients_fn(|i| F::from_usize(e + i * 7))
        },
    )
}

// The reference asserts literal input leaves; the candidate independently selects each accessor.
// Dropping, broadcasting, or confusing any input changes this recording.
fn check_inputs<F: Field, EF: ExtensionField<F>>() {
    let lanes: Vec<_> = (0..<F as Field>::Packing::WIDTH)
        .map(|i| fixture::<F, EF>(i * 1000))
        .collect();
    let packed = EvaluationInputs::pack(&lanes);
    for (i, input) in lanes.iter().enumerate() {
        let reference = |b: &mut RecordingAirBuilder<'_, F, EF>| {
            for row in &input.main {
                for &v in row {
                    b.assert_zero(v);
                }
            }
            for row in &input.preprocessed {
                for &v in row {
                    b.assert_zero(v);
                }
            }
            for &v in &input.public_values {
                b.assert_zero(v);
            }
            for &v in &input.periodic_values {
                b.assert_zero(v);
            }
            for &v in &input.selectors {
                b.assert_zero(v);
            }
            for row in &input.permutation {
                for &v in row {
                    b.assert_zero_ext(v);
                }
            }
            for &v in &input.permutation_randomness {
                b.assert_zero_ext(v);
            }
            for &v in &input.permutation_values {
                b.assert_zero_ext(v);
            }
        };
        compare_scalar(input, reference, |b| all_inputs(b)).unwrap();
        assert_eq!(
            packed.lane(i).record(|b| all_inputs(b)),
            input.record(|b| all_inputs(b))
        );
    }
    compare_packed(&packed, |b| all_inputs(b), |b| all_inputs(b)).unwrap();
    eprintln!(
        "{} packing width: {}",
        core::any::type_name::<F>(),
        <F as Field>::Packing::WIDTH
    );
}
fn all_inputs<AB: PermutationAirBuilder>(b: &mut AB) {
    let main = b.main();
    for &v in main.current_slice().iter().chain(main.next_slice()) {
        b.assert_zero(v);
    }
    let prep = b.preprocessed().clone();
    for &v in prep.current_slice().iter().chain(prep.next_slice()) {
        b.assert_zero(v);
    }
    let public = b.public_values().to_vec();
    for v in public {
        b.assert_zero(v);
    }
    let periodic = b.periodic_values().to_vec();
    for v in periodic {
        b.assert_zero(v);
    }
    b.assert_zero(b.is_first_row());
    b.assert_zero(b.is_last_row());
    b.assert_zero(b.is_transition());
    let perm = b.permutation();
    for &v in perm.current_slice().iter().chain(perm.next_slice()) {
        b.assert_zero_ext(v);
    }
    let random = b.permutation_randomness().to_vec();
    for v in random {
        b.assert_zero_ext(v);
    }
    let values = b.permutation_values().to_vec();
    for v in values {
        b.assert_zero_ext(v);
    }
}
#[test]
fn every_input_survives_scalar_and_packed_evaluation() {
    check_inputs::<F, EF>();
    check_inputs::<Mersenne31, QM31>();
}
#[test]
fn subtraction_mutation_is_rejected_and_reassociation_passes() {
    let input = fixture::<F, EF>(3);
    let err = compare_scalar(
        &input,
        |b| {
            let m = b.main();
            b.assert_zero(m.current_slice()[0] - m.current_slice()[1]);
        },
        |b| {
            let m = b.main();
            b.assert_zero(m.current_slice()[1] - m.current_slice()[0]);
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ConstraintMismatch::Value { assertion: 0, .. }
    ));
    compare_scalar(
        &input,
        |b| {
            let m = b.main();
            let x = m.current_slice()[0];
            let y = m.current_slice()[1];
            b.assert_zero((x + y) + F::TWO);
        },
        |b| {
            let m = b.main();
            let x = m.current_slice()[0];
            let y = m.current_slice()[1];
            b.assert_zero(x + (y + F::TWO));
        },
    )
    .unwrap();
}
#[test]
fn true_extension_constants_are_compared() {
    let input = fixture::<F, EF>(0);
    let c = EF::from_basis_coefficients_fn(|i| F::from_usize(i + 7));
    assert_ne!(c, EF::from(F::from_usize(7)));
    assert!(matches!(
        compare_scalar(
            &input,
            |b| b.assert_zero_ext(c),
            |b| b.assert_zero_ext(c + EF::ONE)
        ),
        Err(ConstraintMismatch::Value { assertion: 0, .. })
    ));
    let lanes = vec![input; <F as Field>::Packing::WIDTH];
    let packed = EvaluationInputs::pack(&lanes);
    compare_packed(
        &packed,
        |b| b.assert_zero_ext(c),
        |b| {
            b.assert_zero_ext(<EF as ExtensionField<F>>::ExtensionPacking::from(c));
        },
    )
    .unwrap();
}
#[test]
fn routing_order_and_count_are_checked_before_values() {
    let input = fixture::<F, EF>(0);
    let reference = |b: &mut RecordingAirBuilder<'_, F, EF>| {
        b.assert_zero(F::ZERO);
        b.assert_zero_ext(EF::ZERO);
    };
    let err = compare_scalar(&input, reference, |b| {
        b.assert_zero_ext(EF::ZERO);
        b.assert_zero(F::ZERO);
    })
    .unwrap_err();
    assert_eq!(
        err,
        ConstraintMismatch::Kind {
            assertion: 0,
            expected: AssertionKind::Base,
            actual: AssertionKind::Extension
        }
    );
    let err = compare_scalar(&input, reference, |b| {
        b.assert_zero(F::ONE);
        b.assert_zero_ext(EF::ZERO);
        b.assert_zero(F::ZERO);
    })
    .unwrap_err();
    assert_eq!(
        err,
        ConstraintMismatch::Count {
            expected: 2,
            actual: 3
        }
    );
    let err = compare_scalar(&input, reference, |b| b.assert_zero(F::ONE)).unwrap_err();
    assert_eq!(
        err,
        ConstraintMismatch::Count {
            expected: 2,
            actual: 1
        }
    );
    let err = compare_scalar(&input, reference, |b| {
        b.assert_zero(F::ONE);
        b.assert_zero(F::ZERO);
    })
    .unwrap_err();
    assert!(matches!(err, ConstraintMismatch::Kind { assertion: 1, .. }));
}
#[test]
fn empty_streams_zero_widths_zeros_and_duplicates_are_preserved() {
    let input =
        EvaluationInputs::<F>::from_fn(AirLayout::default(), vec![], || F::ZERO, || F::ZERO);
    compare_scalar(&input, |_| {}, |_| {}).unwrap();
    let recording = input.record(|b| {
        assert!(b.main().current_slice().is_empty());
        assert!(b.preprocessed().next_slice().is_empty());
        assert!(b.permutation().current_slice().is_empty());
        b.assert_zeros([F::ZERO; 2]);
        b.assert_zero_ext(F::ZERO);
        b.when(F::ZERO).assert_zeros([F::ONE; 3]);
    });
    assert_eq!(
        recording.assertions(),
        &[
            ConstraintValue::Base(F::ZERO),
            ConstraintValue::Base(F::ZERO),
            ConstraintValue::Extension(F::ZERO),
            ConstraintValue::Base(F::ZERO),
            ConstraintValue::Base(F::ZERO),
            ConstraintValue::Base(F::ZERO),
        ]
    );
    compare_packed(
        &EvaluationInputs::pack(&vec![input; <F as Field>::Packing::WIDTH]),
        |_| {},
        |_| {},
    )
    .unwrap();
}
#[test]
fn one_packed_base_or_extension_lane_mutation_is_located() {
    let lanes: Vec<_> = (0..<F as Field>::Packing::WIDTH)
        .map(|i| fixture::<F, EF>(i * 100))
        .collect();
    let packed: PackedEvaluationInputs<F, EF> = EvaluationInputs::pack(&lanes);
    let lane = <F as Field>::Packing::WIDTH - 1;
    let err = compare_packed(
        &packed,
        |b| b.assert_zero(b.main().current_slice()[0]),
        |b| {
            let mut v = b.main().current_slice()[0];
            v.as_slice_mut()[lane] += F::ONE;
            b.assert_zero(v);
        },
    )
    .unwrap_err();
    assert_eq!(err.lane, lane);
    assert!(matches!(
        err.mismatch,
        ConstraintMismatch::Value { assertion: 0, .. }
    ));
    let err = compare_packed(
        &packed,
        |b| b.assert_zero_ext(b.permutation_randomness()[0]),
        |b| {
            let mut v = b.permutation_randomness()[0];
            v.add_assign_lane(lane, EF::ONE);
            b.assert_zero_ext(v);
        },
    )
    .unwrap_err();
    assert_eq!(err.lane, lane);
    assert!(matches!(
        err.mismatch,
        ConstraintMismatch::Value { assertion: 0, .. }
    ));
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    ))]
    const {
        assert!(
            <F as Field>::Packing::WIDTH > 1,
            "this target must exercise real SIMD lanes"
        );
    }
}
#[test]
fn packing_rejects_different_public_values() {
    let mut lanes = vec![fixture::<F, EF>(0); <F as Field>::Packing::WIDTH];
    if lanes.len() == 1 {
        return;
    }
    lanes[1].public_values[0] += F::ONE;
    assert!(std::panic::catch_unwind(|| EvaluationInputs::pack(&lanes)).is_err());
}
#[test]
#[should_panic(expected = "row widths differ")]
fn malformed_windows_are_rejected() {
    let mut input = fixture::<F, EF>(0);
    input.main[1].clear();
    input.record(|_| {});
}

#[test]
fn packing_and_sampling_reject_inconsistent_shapes() {
    let input = fixture::<F, EF>(0);
    assert!(
        std::panic::catch_unwind(|| EvaluationInputs::pack(&[] as &[EvaluationInputs<F, EF>]))
            .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| EvaluationInputs::<F>::from_fn(
            layout(),
            vec![],
            || F::ZERO,
            || F::ZERO
        ))
        .is_err()
    );
    let mut lanes = vec![input; <F as Field>::Packing::WIDTH];
    if lanes.len() > 1 {
        lanes[1].periodic_values.push(F::ONE);
        assert!(std::panic::catch_unwind(|| EvaluationInputs::pack(&lanes)).is_err());
    }
    for rows in [0, 1] {
        let mut input = fixture::<F, EF>(0);
        if rows == 0 {
            input.preprocessed[1].clear();
        } else {
            input.permutation[1].clear();
        }
        assert!(std::panic::catch_unwind(|| input.record(|_| {})).is_err());
    }
}

// The recorder accepts expression rings without imposing Field or PackedField.
// This private probe deliberately violates square's contract in one configuration.
mod hook_probe {
    use core::iter::{Product, Sum};
    use core::ops;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static SQUARE_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    struct Probe<const FAULTY: bool>(F);
    impl<const FAULTY: bool> From<F> for Probe<FAULTY> {
        fn from(value: F) -> Self {
            Self(value)
        }
    }
    macro_rules! binary {
        ($trait:ident, $method:ident, $assign:ident, $assign_method:ident, $op:tt) => {
            impl<const FAULTY: bool, T: Into<Self>> ops::$trait<T> for Probe<FAULTY> {
                type Output = Self;
                fn $method(self, rhs: T) -> Self { Self(self.0 $op rhs.into().0) }
            }
            impl<const FAULTY: bool, T: Into<Self>> ops::$assign<T> for Probe<FAULTY> {
                fn $assign_method(&mut self, rhs: T) { *self = *self $op rhs; }
            }
        };
    }
    binary!(Add, add, AddAssign, add_assign, +);
    binary!(Sub, sub, SubAssign, sub_assign, -);
    binary!(Mul, mul, MulAssign, mul_assign, *);
    impl<const FAULTY: bool> ops::Neg for Probe<FAULTY> {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    impl<const FAULTY: bool> Sum for Probe<FAULTY> {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::ZERO, |a, b| a + b)
        }
    }
    impl<const FAULTY: bool> Product for Probe<FAULTY> {
        fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::ONE, |a, b| a * b)
        }
    }
    impl<const FAULTY: bool> PrimeCharacteristicRing for Probe<FAULTY> {
        type PrimeSubfield = <F as PrimeCharacteristicRing>::PrimeSubfield;
        const ZERO: Self = Self(F::ZERO);
        const ONE: Self = Self(F::ONE);
        const TWO: Self = Self(F::TWO);
        const NEG_ONE: Self = Self(F::NEG_ONE);
        fn from_prime_subfield(value: Self::PrimeSubfield) -> Self {
            Self(value)
        }
        fn square(&self) -> Self {
            SQUARE_CALLS.fetch_add(1, Ordering::SeqCst);
            Self(self.0 * self.0 + if FAULTY { F::ONE } else { F::ZERO })
        }
    }
    impl<const FAULTY: bool> Algebra<F> for Probe<FAULTY> {}

    fn check<const FAULTY: bool>() {
        let input = EvaluationInputs::<F, F, Probe<FAULTY>, Probe<FAULTY>>::from_fn(
            AirLayout {
                main_width: 1,
                ..Default::default()
            },
            vec![],
            || Probe(F::from_u32(7)),
            || Probe(F::ZERO),
        );
        SQUARE_CALLS.store(0, Ordering::SeqCst);
        let reference = input.record(|b| {
            let x = b.main().current_slice()[0];
            b.assert_zero(x * x);
        });
        assert_eq!(SQUARE_CALLS.load(Ordering::SeqCst), 0);
        let candidate = input.record(|b| {
            let x = b.main().current_slice()[0];
            b.assert_zero(x.square());
        });
        // This witness observes the two callbacks' distinct dispatch. Numeric equality alone
        // cannot certify dispatch or independent implementation.
        assert_eq!(SQUARE_CALLS.load(Ordering::SeqCst), 1);
        if FAULTY {
            assert_eq!(
                reference.compare(&candidate),
                Err(ConstraintMismatch::Value {
                    assertion: 0,
                    expected: ConstraintValue::Base(Probe(F::from_u32(49))),
                    actual: ConstraintValue::Base(Probe(F::from_u32(50))),
                })
            );
        } else {
            reference.compare(&candidate).unwrap();
        }
    }
    #[test]
    fn independent_callbacks_expose_a_faulty_arithmetic_hook() {
        check::<false>();
        check::<true>();
    }
}

#[test]
fn same_kind_order_changes_are_rejected() {
    let input = fixture::<F, EF>(0);
    assert_eq!(
        compare_scalar(
            &input,
            |b| {
                b.assert_zero(F::ONE);
                b.assert_zero(F::TWO);
            },
            |b| {
                b.assert_zero(F::TWO);
                b.assert_zero(F::ONE);
            }
        ),
        Err(ConstraintMismatch::Value {
            assertion: 0,
            expected: ConstraintValue::Base(F::ONE),
            actual: ConstraintValue::Base(F::TWO)
        })
    );
}

#[test]
fn mixed_base_and_extension_arithmetic_matches_independent_reference() {
    let lanes: Vec<_> = (0..<F as Field>::Packing::WIDTH)
        .map(|i| fixture::<F, EF>(i * 31))
        .collect();
    let constant = EF::from_basis_coefficients_fn(|i| F::from_usize(i + 13));
    let reference = |b: &mut RecordingAirBuilder<'_, F, EF>| {
        let m = b.main();
        let x = m.current_slice()[0];
        let y = m.next_slice()[1];
        let z = b.permutation_randomness()[0];
        b.assert_zero_ext(z * x * y + constant);
    };
    fn candidate<AB: PermutationAirBuilder<F = F, EF = EF>>(b: &mut AB, constant: EF) {
        let m = b.main();
        let x: AB::Expr = m.current_slice()[0].into();
        let y: AB::Expr = m.next_slice()[1].into();
        let z: AB::ExprEF = b.permutation_randomness()[0].into();
        b.assert_zero_ext(z * AB::ExprEF::from(x * y) + AB::ExprEF::from(constant));
    }
    for input in &lanes {
        compare_scalar(input, reference, |b| candidate(b, constant)).unwrap();
    }
    compare_packed(&EvaluationInputs::pack(&lanes), reference, |b| {
        candidate(b, constant);
    })
    .unwrap();
}

#[test]
fn scalar_mismatch_propagates_as_an_error() {
    fn compare() -> Result<(), Box<dyn core::error::Error>> {
        compare_scalar(
            &fixture::<F, EF>(0),
            |b| b.assert_zero(F::ONE),
            |b| b.assert_zero(F::TWO),
        )?;
        Ok(())
    }
    let err = compare().unwrap_err();
    assert!(err.downcast_ref::<ConstraintMismatch<F, EF>>().is_some());
    assert_eq!(
        err.to_string(),
        "assertion 0 value differs: expected Base(1), actual Base(2)"
    );
    let count = ConstraintMismatch::<F, EF>::Count {
        expected: 3,
        actual: 2,
    };
    assert_eq!(
        count.to_string(),
        "assertion counts differ: expected 3, actual 2"
    );
    let kind = ConstraintMismatch::<F, EF>::Kind {
        assertion: 4,
        expected: AssertionKind::Base,
        actual: AssertionKind::Extension,
    };
    assert_eq!(
        kind.to_string(),
        "assertion 4 kind differs: expected Base, actual Extension"
    );
}

#[test]
fn packed_mismatch_propagates_as_an_error() {
    fn compare() -> Result<(), Box<dyn core::error::Error>> {
        let lanes = vec![fixture::<F, EF>(0); <F as Field>::Packing::WIDTH];
        compare_packed(
            &EvaluationInputs::pack(&lanes),
            |b| b.assert_zero(F::ONE),
            |b| b.assert_zero(<F as Field>::Packing::from(F::TWO)),
        )?;
        Ok(())
    }
    let err = compare().unwrap_err();
    assert!(err.downcast_ref::<PackedMismatch<F, EF>>().is_some());
    assert_eq!(
        err.to_string(),
        "SIMD lane 0: assertion 0 value differs: expected Base(1), actual Base(2)"
    );
}

#[test]
fn reusable_scalar_and_packed_recorders_preserve_interleaving() {
    fn evaluate<AB: PermutationAirBuilder>(b: &mut AB) {
        b.assert_zero(b.main().current_slice()[0]);
        b.assert_zero_ext(b.permutation().next_slice()[0]);
        b.assert_zero(b.is_last_row());
    }
    let input = fixture::<F, EF>(0);
    let expected = [
        ConstraintValue::Base(input.main[0][0]),
        ConstraintValue::Extension(input.permutation[1][0]),
        ConstraintValue::Base(input.selectors[1]),
    ];
    let mut recorder = RecordingAirBuilder::new(&input);
    assert!(recorder.recording().assertions().is_empty());
    for _ in 0..3 {
        evaluate(&mut recorder);
        assert_eq!(recorder.recording().assertions(), expected);
        recorder.reset();
        assert!(recorder.recording().assertions().is_empty());
    }
    recorder.assert_zero_ext(EF::ONE);
    assert_eq!(
        recorder.recording().assertions(),
        &[ConstraintValue::Extension(EF::ONE)]
    );
    recorder.reset();
    recorder.reset();
    evaluate(&mut recorder);
    assert_eq!(recorder.recording().assertions(), expected);

    let lanes: Vec<_> = (0..<F as Field>::Packing::WIDTH)
        .map(|i| fixture::<F, EF>(i * 13))
        .collect();
    let packed = EvaluationInputs::pack(&lanes);
    let mut recorder = RecordingAirBuilder::new(&packed);
    for _ in 0..3 {
        evaluate(&mut recorder);
        for (lane, input) in lanes.iter().enumerate() {
            let values = recorder.recording().assertions();
            assert!(
                matches!(values[0], ConstraintValue::Base(v) if PackedValue::extract(&v, lane) == input.main[0][0])
            );
            assert!(
                matches!(values[1], ConstraintValue::Extension(v) if PackedFieldExtension::<F, EF>::extract(&v, lane) == input.permutation[1][0])
            );
            assert!(
                matches!(values[2], ConstraintValue::Base(v) if PackedValue::extract(&v, lane) == input.selectors[1])
            );
        }
        recorder.reset();
        assert!(recorder.recording().assertions().is_empty());
    }
}

#[test]
fn recorder_construction_rejects_every_malformed_window() {
    for window in 0..3 {
        let mut input = fixture::<F, EF>(0);
        match window {
            0 => input.main[1].clear(),
            1 => input.preprocessed[1].clear(),
            _ => input.permutation[1].clear(),
        }
        assert!(std::panic::catch_unwind(|| RecordingAirBuilder::new(&input)).is_err());
    }
}
