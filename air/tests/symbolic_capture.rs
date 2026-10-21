use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, BaseEntry, BaseLeaf, ExtEntry, ExtLeaf, ExtensionBuilder,
    PermutationAirBuilder, SymbolicAirBuilder, SymbolicConstraints, SymbolicExpression,
    SymbolicExpressionExt, WindowAccess, capture_symbolic_constraints,
    get_all_symbolic_constraints, get_constraint_layout, get_symbolic_constraints,
    get_symbolic_constraints_extension,
};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

#[test]
fn closure_capture_preserves_interleaved_zero_and_duplicate_assertions() {
    let inputs = vec![F::ZERO, F::TWO, F::TWO];
    let captured = capture_symbolic_constraints::<F, EF>(AirLayout::default(), move |builder| {
        for value in inputs {
            builder.assert_zero(value);
            builder.assert_zero_ext(value);
        }
        builder.assert_zero_ext(EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1)));
    });

    assert_eq!(captured.constraint_layout().base_indices, [0, 2, 4]);
    assert_eq!(captured.constraint_layout().ext_indices, [1, 3, 5, 6]);
    assert_eq!(captured.constraint_layout().total_constraints(), 7);
    assert_eq!(captured.base_constraints().len(), 3);
    assert_eq!(captured.extension_constraints().len(), 4);
    for (expression, expected) in captured
        .base_constraints()
        .iter()
        .zip([F::ZERO, F::TWO, F::TWO])
    {
        assert!(
            matches!(expression, SymbolicExpression::Leaf(BaseLeaf::Constant(c)) if *c == expected)
        );
    }
    for (expression, expected) in
        captured.extension_constraints()[..3]
            .iter()
            .zip([F::ZERO, F::TWO, F::TWO])
    {
        assert!(
            matches!(expression, SymbolicExpressionExt::Leaf(ExtLeaf::Base(SymbolicExpression::Leaf(BaseLeaf::Constant(c)))) if *c == expected)
        );
    }
    assert!(
        matches!(&captured.extension_constraints()[3], SymbolicExpressionExt::Leaf(ExtLeaf::ExtConstant(c)) if <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(c) == [F::ONE, F::TWO, F::from_u8(3), F::from_u8(4)])
    );
}

#[test]
fn empty_capture_has_no_constraints_or_indices() {
    let captured: SymbolicConstraints<F> =
        capture_symbolic_constraints(AirLayout::default(), |_| {});
    assert!(captured.base_constraints().is_empty());
    assert!(captured.extension_constraints().is_empty());
    assert!(captured.constraint_layout().base_indices.is_empty());
    assert!(captured.constraint_layout().ext_indices.is_empty());
}

#[test]
fn closure_capture_uses_explicit_input_layout() {
    let layout = AirLayout {
        main_width: 2,
        preprocessed_width: 3,
        num_public_values: 4,
        num_periodic_columns: 5,
        permutation_width: 6,
        num_permutation_challenges: 7,
        num_permutation_values: 8,
    };
    let captured = capture_symbolic_constraints::<F, EF>(layout, |builder| {
        builder.assert_zero(builder.main().next(1).unwrap());
        builder.assert_zero(builder.preprocessed().next(2).unwrap());
        builder.assert_zero(builder.public_values()[3]);
        builder.assert_zero(builder.periodic_values()[4]);
        builder.assert_zero_ext(builder.permutation().next(5).unwrap());
        builder.assert_zero_ext(builder.permutation_randomness()[6]);
        builder.assert_zero_ext(builder.permutation_values()[7]);
    });
    let expected_base = [
        (BaseEntry::Main { offset: 1 }, 1),
        (BaseEntry::Preprocessed { offset: 1 }, 2),
        (BaseEntry::Public, 3),
        (BaseEntry::Periodic, 4),
    ];
    assert_eq!(captured.base_constraints().len(), expected_base.len());
    for (expression, (entry, index)) in captured.base_constraints().iter().zip(expected_base) {
        assert!(
            matches!(expression, SymbolicExpression::Leaf(BaseLeaf::Variable(v)) if v.entry == entry && v.index == index)
        );
    }
    let expected_ext = [
        (ExtEntry::Permutation { offset: 1 }, 5),
        (ExtEntry::Challenge, 6),
        (ExtEntry::PermutationValue, 7),
    ];
    assert_eq!(captured.extension_constraints().len(), expected_ext.len());
    for (expression, (entry, index)) in captured.extension_constraints().iter().zip(expected_ext) {
        assert!(
            matches!(expression, SymbolicExpressionExt::Leaf(ExtLeaf::ExtVariable(v)) if v.entry == entry && v.index == index)
        );
    }
}

struct CaptureAir;

impl BaseAir<F> for CaptureAir {
    fn width(&self) -> usize {
        1
    }
}

impl<E: p3_field::ExtensionField<F>> Air<SymbolicAirBuilder<F, E>> for CaptureAir
where
    SymbolicExpressionExt<F, E>: p3_field::Algebra<E>,
{
    fn eval(&self, builder: &mut SymbolicAirBuilder<F, E>) {
        let x = builder.main().current(0).unwrap();
        builder.assert_zero(x * x);
        builder.assert_zero_ext(x);
        builder.assert_zero(x * x);
        builder.assert_zero_ext(F::ZERO);
    }
}

#[test]
fn finish_matches_existing_getters_and_collection_helpers() {
    let layout = AirLayout::from_air(&CaptureAir);
    let mut builder = SymbolicAirBuilder::<F, EF>::new(layout);
    CaptureAir.eval(&mut builder);
    let old_base = serde_json::to_value(builder.base_constraints()).unwrap();
    let old_ext = serde_json::to_value(builder.extension_constraints()).unwrap();
    let old_layout = builder.constraint_layout();
    let captured = builder.finish();
    assert_eq!(
        serde_json::to_value(captured.base_constraints()).unwrap(),
        old_base
    );
    assert_eq!(
        serde_json::to_value(captured.extension_constraints()).unwrap(),
        old_ext
    );
    assert_eq!(
        captured.constraint_layout().base_indices,
        old_layout.base_indices
    );
    assert_eq!(
        captured.constraint_layout().ext_indices,
        old_layout.ext_indices
    );
    assert_eq!(captured.constraint_layout().base_indices, [0, 2]);
    assert_eq!(captured.constraint_layout().ext_indices, [1, 3]);
    let prover_layout = get_constraint_layout::<F, F, _>(&CaptureAir, layout);
    assert_eq!(
        prover_layout.base_indices,
        captured.constraint_layout().base_indices
    );
    assert_eq!(
        prover_layout.ext_indices,
        captured.constraint_layout().ext_indices
    );

    let base_ptr = captured.base_constraints().as_ptr();
    let ext_ptr = captured.extension_constraints().as_ptr();
    let (owned_base, owned_ext, owned_layout) = captured.into_parts();
    assert_eq!(owned_base.as_ptr(), base_ptr);
    assert_eq!(owned_ext.as_ptr(), ext_ptr);
    assert_eq!(owned_layout.base_indices, [0, 2]);
    assert_eq!(owned_layout.ext_indices, [1, 3]);

    let (base, ext) = get_all_symbolic_constraints::<F, EF, _>(&CaptureAir, layout);
    assert_eq!(serde_json::to_value(base).unwrap(), old_base);
    assert_eq!(serde_json::to_value(ext).unwrap(), old_ext);
    assert_eq!(
        serde_json::to_value(get_symbolic_constraints(&CaptureAir, layout)).unwrap(),
        old_base
    );
    assert_eq!(
        serde_json::to_value(get_symbolic_constraints_extension::<F, EF, _>(
            &CaptureAir,
            layout
        ))
        .unwrap(),
        old_ext
    );
}
