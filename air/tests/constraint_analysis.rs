use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, ConstraintCounts, ConstraintDegrees,
    ConstraintLayoutBuilder, ExtensionBuilder, NamedAirBuilder, NamedExtensionBuilder,
    PermutationAirBuilder, SymbolicAirBuilder, WindowAccess, capture_constraint_layout,
    capture_symbolic_constraints, get_constraint_layout,
};
use p3_baby_bear::BabyBear as F;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

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

fn mixed_eval<AB: PermutationAirBuilder<F = F, EF = EF>>(builder: &mut AB) {
    let main = builder.main();
    let x: AB::Expr = main.current(0).unwrap().into();
    let next = main.next(1).unwrap();
    let prep = builder.preprocessed().next(0).unwrap();
    let public: AB::Expr = builder.public_values()[0].into();
    let periodic: AB::Expr = builder.periodic_values()[0].into();
    let perm: AB::ExprEF = builder.permutation().next(0).unwrap().into();
    let challenge: AB::ExprEF = builder.permutation_randomness()[0].into();
    let value: AB::ExprEF = builder.permutation_values()[0].clone().into();
    let coefficient = EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1));
    builder.assert_zero(x.clone() * prep + public);
    builder.assert_zero_ext(perm.clone() * challenge + value + coefficient);
    builder.when_transition().assert_zero(next - x.clone());
    builder.when_first_row().assert_zero(periodic.clone());
    builder
        .when_last_row()
        .assert_zero_ext(perm.clone() * periodic);
    builder.assert_zeros_with_filter(F::ZERO, [x.clone(), x]);
    builder.when(F::ZERO).assert_zeros_ext([perm.clone(), perm]);
    builder.assert_zero(F::ZERO);
    builder.assert_zero_ext(EF::ZERO);
}

#[test]
fn layout_matches_symbolic_with_interleaving_filtering_and_duplicates() {
    let symbolic = capture_symbolic_constraints::<F, EF>(layout(), mixed_eval);
    let cheap = capture_constraint_layout::<F, EF>(layout(), mixed_eval);
    assert_eq!(cheap.base_indices, [0, 2, 3, 5, 6, 9]);
    assert_eq!(cheap.ext_indices, [1, 4, 7, 8, 10]);
    assert_eq!(cheap.base_indices, symbolic.constraint_layout.base_indices);
    assert_eq!(cheap.ext_indices, symbolic.constraint_layout.ext_indices);
    assert_eq!(
        symbolic.constraint_counts(),
        ConstraintCounts {
            base: 6,
            extension: 5
        }
    );
}

#[test]
fn layout_inputs_are_zero_and_windows_preserve_width_including_empty() {
    for shape in [AirLayout::default(), layout()] {
        let builder = ConstraintLayoutBuilder::<F, EF>::new(shape);
        for window in [builder.main(), builder.preprocessed().clone()] {
            assert!(window.current_slice().iter().all(|v| *v == F::ZERO));
            assert_eq!(window.current_slice().len(), window.next_slice().len());
            assert_eq!(window.current(window.current_slice().len()), None);
        }
        assert_eq!(builder.main().current_slice().len(), shape.main_width);
        assert_eq!(
            builder.preprocessed().next_slice().len(),
            shape.preprocessed_width
        );
        assert_eq!(
            builder.permutation().current_slice(),
            vec![EF::ZERO; shape.permutation_width]
        );
        assert_eq!(
            builder.permutation().next_slice(),
            vec![EF::ZERO; shape.permutation_width]
        );
        assert_eq!(
            builder.public_values(),
            vec![F::ZERO; shape.num_public_values]
        );
        assert_eq!(
            builder.periodic_values(),
            vec![F::ZERO; shape.num_periodic_columns]
        );
        assert_eq!(
            builder.permutation_randomness(),
            vec![EF::ZERO; shape.num_permutation_challenges]
        );
        assert_eq!(
            builder.permutation_values(),
            vec![EF::ZERO; shape.num_permutation_values]
        );
        assert_eq!(builder.is_first_row(), F::ZERO);
        assert_eq!(builder.is_last_row(), F::ZERO);
        assert_eq!(builder.is_transition_window(2), F::ZERO);
        assert_eq!(builder.finish().total_constraints(), 0);
    }
}

#[test]
#[should_panic(expected = "exceeds supported arity")]
fn unsupported_window_is_rejected() {
    let _ = ConstraintLayoutBuilder::<F>::new(AirLayout::default()).is_transition_window(3);
}

#[test]
fn zero_column_capture_and_named_assertions_keep_identity_without_evaluating_labels() {
    let cheap = capture_constraint_layout::<F, EF>(AirLayout::default(), |builder| {
        assert!(builder.main().current_slice().is_empty());
        builder.assert_zero_named(F::ZERO, || panic!("label should not be evaluated"));
        builder.assert_zero_ext_named(EF::ZERO, "extension");
        builder.assert_zeros::<0, F>([]);
        builder.assert_zeros_ext::<0, EF>([]);
    });
    assert_eq!(cheap.base_indices, [0]);
    assert_eq!(cheap.ext_indices, [1]);
}

#[test]
fn split_degrees_account_for_periodic_columns_transition_and_coefficients() {
    let constraints = capture_symbolic_constraints::<F, EF>(layout(), |builder| {
        let x = builder.main().current(0).unwrap();
        let periodic = builder.periodic_values()[0];
        let permutation = builder.permutation().current(0).unwrap();
        let coefficient = EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1));
        builder.when_transition().assert_zero(x * periodic);
        builder
            .when_transition()
            .assert_zero_ext(permutation * periodic + coefficient);
    });
    let before = format!("{:?}", constraints);
    assert_eq!(
        constraints.constraint_counts(),
        ConstraintCounts {
            base: 1,
            extension: 1
        }
    );
    assert_eq!(
        constraints.max_constraint_degrees(),
        ConstraintDegrees {
            base: 2,
            extension: 2
        }
    );
    assert_eq!(
        constraints.max_constraint_degrees_with_transition(1),
        ConstraintDegrees {
            base: 3,
            extension: 3
        }
    );
    assert_eq!(
        constraints.max_constraint_degrees_with_transition(2),
        ConstraintDegrees {
            base: 4,
            extension: 4
        }
    );
    // At N=8 and period=2: trace degree 7, periodic degree 4, transition degree 1.
    assert_eq!(
        constraints.max_constraint_poly_degrees(8, &[2]),
        ConstraintDegrees {
            base: 12,
            extension: 12
        }
    );
    assert_eq!(
        constraints.max_constraint_poly_degrees(16, &[4]),
        ConstraintDegrees {
            base: 28,
            extension: 28
        }
    );
    assert_eq!(format!("{:?}", constraints), before);

    let coefficient = capture_symbolic_constraints::<F, EF>(layout(), |builder| {
        let x = builder.main().current(0).unwrap();
        let constant = EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1));
        builder.assert_zero(x * F::TWO);
        builder.assert_zero_ext(p3_air::SymbolicExpressionExt::from(x) * constant);
    });
    assert_eq!(
        coefficient.max_constraint_degrees(),
        ConstraintDegrees {
            base: 1,
            extension: 1
        }
    );
    assert_eq!(
        coefficient.max_constraint_poly_degrees(8, &[]),
        ConstraintDegrees {
            base: 7,
            extension: 7
        }
    );
}

#[test]
fn empty_streams_and_constant_constraints_have_zero_degrees() {
    for populated in [false, true] {
        let mut builder = SymbolicAirBuilder::<F, EF>::new(AirLayout::default());
        if populated {
            builder.assert_zero(F::TWO);
            builder.assert_zero_ext(EF::TWO);
        }
        let constraints = builder.finish();
        assert_eq!(
            constraints.constraint_counts(),
            ConstraintCounts {
                base: usize::from(populated),
                extension: usize::from(populated)
            }
        );
        assert_eq!(
            constraints.max_constraint_degrees(),
            ConstraintDegrees::default()
        );
        assert_eq!(
            constraints.max_constraint_degrees_with_transition(1),
            ConstraintDegrees::default()
        );
        assert_eq!(
            constraints.max_constraint_poly_degrees(8, &[]),
            ConstraintDegrees::default()
        );
    }
}

#[test]
fn split_degrees_keep_distinct_maxima_and_an_empty_stream() {
    let mut builder = SymbolicAirBuilder::<F, EF>::new(layout());
    let x = builder.main().current(0).unwrap();
    builder.assert_zero(x);
    let constraints = builder.finish();
    assert_eq!(
        constraints.max_constraint_degrees(),
        ConstraintDegrees {
            base: 1,
            extension: 0
        }
    );
    assert_eq!(
        constraints.max_constraint_poly_degrees(8, &[]),
        ConstraintDegrees {
            base: 7,
            extension: 0
        }
    );

    let mut builder = SymbolicAirBuilder::<F, EF>::new(layout());
    let permutation = builder.permutation().current(0).unwrap();
    builder.assert_zero_ext(permutation * permutation * permutation);
    let constraints = builder.finish();
    assert_eq!(
        constraints.max_constraint_degrees(),
        ConstraintDegrees {
            base: 0,
            extension: 3
        }
    );
    assert_eq!(
        constraints.max_constraint_poly_degrees(8, &[]),
        ConstraintDegrees {
            base: 0,
            extension: 21
        }
    );
}

#[test]
fn legacy_layout_helper_retains_builder_specialization() {
    struct Specialized;
    impl BaseAir<F> for Specialized {
        fn width(&self) -> usize {
            0
        }
    }
    impl Air<SymbolicAirBuilder<F>> for Specialized {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F>) {
            builder.assert_zeros([F::ZERO, F::ZERO]);
        }
    }
    impl Air<ConstraintLayoutBuilder<F>> for Specialized {
        fn eval(&self, builder: &mut ConstraintLayoutBuilder<F>) {
            builder.assert_zero_ext(F::ZERO);
        }
    }
    let legacy = get_constraint_layout::<F, F, _>(&Specialized, AirLayout::default());
    assert_eq!(legacy.base_indices, [0, 1]);
    assert!(legacy.ext_indices.is_empty());
    let concrete = capture_constraint_layout::<F, F>(AirLayout::default(), |builder| {
        Specialized.eval(builder);
    });
    assert!(concrete.base_indices.is_empty());
    assert_eq!(concrete.ext_indices, [0]);
}

#[test]
fn zero_width_windows_match_symbolic_capture() {
    fn eval<AB: PermutationAirBuilder<F = F, EF = EF>>(builder: &mut AB) {
        assert!(builder.preprocessed().current_slice().is_empty());
        assert!(builder.preprocessed().next_slice().is_empty());
        assert!(builder.main().current_slice().is_empty());
        assert!(builder.main().next_slice().is_empty());
        assert!(builder.permutation().current_slice().is_empty());
        assert!(builder.permutation().next_slice().is_empty());
        builder.assert_zero(F::ZERO);
        builder.assert_zero_ext(EF::ZERO);
    }
    let symbolic = capture_symbolic_constraints::<F, EF>(AirLayout::default(), eval);
    let cheap = capture_constraint_layout::<F, EF>(AirLayout::default(), eval);
    assert_eq!(cheap.base_indices, [0]);
    assert_eq!(cheap.ext_indices, [1]);
    assert_eq!(cheap.base_indices, symbolic.constraint_layout.base_indices);
    assert_eq!(cheap.ext_indices, symbolic.constraint_layout.ext_indices);
}

#[test]
fn layout_window_clones_share_storage() {
    let builder = ConstraintLayoutBuilder::<F, EF>::new(layout());
    let main = builder.main();
    let main_clone = builder.main();
    assert!(core::ptr::eq(
        main.current_slice(),
        main_clone.current_slice()
    ));
    let permutation = builder.permutation();
    let permutation_clone = builder.permutation();
    assert!(core::ptr::eq(
        permutation.next_slice(),
        permutation_clone.next_slice()
    ));
}

#[test]
fn seeded_assertion_sequences_match_symbolic_layouts() {
    struct AssertionSequence(Vec<u8>);

    impl BaseAir<F> for AssertionSequence {
        fn width(&self) -> usize {
            1
        }
    }

    impl<AB: PermutationAirBuilder<F = F, EF = EF>> Air<AB> for AssertionSequence {
        fn eval(&self, builder: &mut AB) {
            for &action in &self.0 {
                let zero = action & 2 != 0;
                if action & 1 == 0 {
                    let value: AB::Expr = if zero {
                        F::ZERO.into()
                    } else {
                        builder.main().current(0).unwrap().into()
                    };
                    match (action >> 2) & 3 {
                        0 => builder.assert_zero(value),
                        1 => builder.when(F::ZERO).assert_zero(value),
                        2 => builder.when(F::ONE).assert_zero(value),
                        _ => builder.when_transition().assert_zero(value),
                    }
                } else {
                    let value: AB::ExprEF = if zero {
                        EF::ZERO.into()
                    } else {
                        builder.permutation().current(0).unwrap().into()
                    };
                    match (action >> 2) & 3 {
                        0 => builder.assert_zero_ext(value),
                        1 => builder.when(F::ZERO).assert_zero_ext(value),
                        2 => builder.when(F::ONE).assert_zero_ext(value),
                        _ => builder.when_transition().assert_zero_ext(value),
                    }
                }
            }
        }
    }

    let shape = AirLayout {
        main_width: 1,
        permutation_width: 1,
        ..Default::default()
    };
    // Fixed seeds make randomized stream interleavings reproducible.
    for seed in 0..64_u64 {
        let mut state = seed;
        let air = AssertionSequence(
            (0..128)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 32) as u8 & 15
                })
                .collect(),
        );
        let symbolic = capture_symbolic_constraints::<F, EF>(shape, |builder| air.eval(builder));
        let symbolic = symbolic.constraint_layout;
        let cheap = capture_constraint_layout::<F, EF>(shape, |builder| air.eval(builder));
        assert_eq!(cheap.base_indices, symbolic.base_indices, "seed {seed}");
        assert_eq!(cheap.ext_indices, symbolic.ext_indices, "seed {seed}");
        assert_eq!(cheap.total_constraints(), air.0.len());
    }
}

#[test]
fn symbolic_windows_reject_malformed_row_storage() {
    use p3_matrix::dense::RowMajorMatrix;

    for (values, width) in [(vec![F::ZERO; 5], 2), (vec![F::ZERO], 0)] {
        // Construct directly so the window check sees malformed public storage.
        let mut window = RowMajorMatrix::new(vec![], width);
        window.values = values;
        assert!(std::panic::catch_unwind(|| window.current_slice()).is_err());
        assert!(std::panic::catch_unwind(|| window.next_slice()).is_err());
    }
}
