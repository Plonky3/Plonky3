use core::cell::Cell;

use p3_air::{
    AirBuilder, AirLayout, AirLayoutInput, AirLayoutMismatch, BaseAir, ExtensionBuilder,
    PermutationAirBuilder, WindowAccess,
};
use p3_baby_bear::BabyBear as F;
use p3_field::PrimeCharacteristicRing;

// Unlike RowWindow, this permits malformed windows with independently sized rows.
#[derive(Clone)]
struct TestWindow {
    current: Vec<F>,
    next: Vec<F>,
}

impl TestWindow {
    fn new(width: usize) -> Self {
        Self {
            current: vec![F::ZERO; width],
            next: vec![F::ZERO; width],
        }
    }
}

impl WindowAccess<F> for TestWindow {
    fn current_slice(&self) -> &[F] {
        &self.current
    }
    fn next_slice(&self) -> &[F] {
        &self.next
    }
}

struct TestBuilder<const PERMUTATION: bool> {
    main: TestWindow,
    preprocessed: TestWindow,
    public: Vec<F>,
    periodic: Vec<F>,
    permutation: TestWindow,
    randomness: Vec<F>,
    values: Vec<F>,
    main_calls: Cell<usize>,
    permutation_calls: Cell<usize>,
}

impl<const P: bool> TestBuilder<P> {
    fn new(layout: AirLayout) -> Self {
        Self {
            main: TestWindow::new(layout.main_width),
            preprocessed: TestWindow::new(layout.preprocessed_width),
            public: vec![F::ZERO; layout.num_public_values],
            periodic: vec![F::ZERO; layout.num_periodic_columns],
            permutation: TestWindow::new(layout.permutation_width),
            randomness: vec![F::ZERO; layout.num_permutation_challenges],
            values: vec![F::ZERO; layout.num_permutation_values],
            main_calls: Cell::new(0),
            permutation_calls: Cell::new(0),
        }
    }
}

impl<const P: bool> AirBuilder for TestBuilder<P> {
    type F = F;
    type Expr = F;
    type Var = F;
    type MainWindow = TestWindow;
    type PreprocessedWindow = TestWindow;
    type PublicVar = F;
    type PeriodicVar = F;

    fn main(&self) -> TestWindow {
        self.main_calls.set(self.main_calls.get() + 1);
        self.main.clone()
    }
    fn preprocessed(&self) -> &TestWindow {
        &self.preprocessed
    }
    fn public_values(&self) -> &[F] {
        &self.public
    }
    fn periodic_values(&self) -> &[F] {
        &self.periodic
    }
    fn is_first_row(&self) -> F {
        panic!("validation must not evaluate selectors")
    }
    fn is_last_row(&self) -> F {
        panic!("validation must not evaluate selectors")
    }
    fn is_transition(&self) -> F {
        panic!("validation must not evaluate selectors")
    }
    fn assert_zero<I: Into<F>>(&mut self, _: I) {
        panic!("validation must not evaluate constraints")
    }
}

impl ExtensionBuilder for TestBuilder<true> {
    type EF = F;
    type ExprEF = F;
    type VarEF = F;

    fn assert_zero_ext<I: Into<F>>(&mut self, _: I) {
        panic!("validation must not evaluate constraints")
    }
}

impl PermutationAirBuilder for TestBuilder<true> {
    type MP = TestWindow;
    type RandomVar = F;
    type PermutationVar = F;

    fn permutation(&self) -> TestWindow {
        self.permutation_calls.set(self.permutation_calls.get() + 1);
        self.permutation.clone()
    }
    fn permutation_randomness(&self) -> &[F] {
        &self.randomness
    }
    fn permutation_values(&self) -> &[F] {
        &self.values
    }
}

const fn layout() -> AirLayout {
    AirLayout {
        main_width: 3,
        preprocessed_width: 2,
        num_public_values: 4,
        num_periodic_columns: 5,
        permutation_width: 6,
        num_permutation_challenges: 7,
        num_permutation_values: 8,
    }
}

#[test]
fn accepts_nonzero_shapes_and_acquires_each_owned_window_once() {
    let layout = layout();
    let base = TestBuilder::<false>::new(layout);
    assert_eq!(layout.validate_against_builder(&base), Ok(()));
    assert_eq!(base.main_calls.get(), 1);

    let full = TestBuilder::<true>::new(layout);
    assert_eq!(layout.validate_against_permutation_builder(&full), Ok(()));
    assert_eq!(full.main_calls.get(), 1);
    assert_eq!(full.permutation_calls.get(), 1);
}

#[test]
fn accepts_all_zero_dimensions() {
    let layout = AirLayout::default();
    assert_eq!(
        layout.validate_against_builder(&TestBuilder::<false>::new(layout)),
        Ok(())
    );
    assert_eq!(
        layout.validate_against_permutation_builder(&TestBuilder::<true>::new(layout)),
        Ok(())
    );
}

#[test]
fn reports_each_base_accessor_mismatch_and_full_validation_checks_base_inputs() {
    use AirLayoutInput::*;
    // Every case starts with a valid builder and shortens only the named accessor.
    for (input, expected, actual) in [
        (MainCurrent, 3, 2),
        (MainNext, 3, 2),
        (PreprocessedCurrent, 2, 1),
        (PreprocessedNext, 2, 1),
        (PublicValues, 4, 3),
        (PeriodicValues, 5, 4),
    ] {
        let layout = layout();
        let mut base = TestBuilder::<false>::new(layout);
        let mut full = TestBuilder::<true>::new(layout);
        fn shorten<const P: bool>(builder: &mut TestBuilder<P>, input: AirLayoutInput) {
            let slice = match input {
                MainCurrent => &mut builder.main.current,
                MainNext => &mut builder.main.next,
                PreprocessedCurrent => &mut builder.preprocessed.current,
                PreprocessedNext => &mut builder.preprocessed.next,
                PublicValues => &mut builder.public,
                PeriodicValues => &mut builder.periodic,
                _ => unreachable!(),
            };
            slice.pop();
        }
        shorten(&mut base, input);
        shorten(&mut full, input);
        let error = AirLayoutMismatch {
            input,
            expected,
            actual,
        };
        assert_eq!(layout.validate_against_builder(&base), Err(error));
        assert_eq!(
            layout.validate_against_permutation_builder(&full),
            Err(error)
        );
    }
}

#[test]
fn reports_each_permutation_accessor_mismatch() {
    use AirLayoutInput::*;
    for (input, expected, actual) in [
        (PermutationCurrent, 6, 7),
        (PermutationNext, 6, 7),
        (PermutationRandomness, 7, 8),
        (PermutationValues, 8, 9),
    ] {
        let layout = layout();
        let mut builder = TestBuilder::<true>::new(layout);
        let slice = match input {
            PermutationCurrent => &mut builder.permutation.current,
            PermutationNext => &mut builder.permutation.next,
            PermutationRandomness => &mut builder.randomness,
            PermutationValues => &mut builder.values,
            _ => unreachable!(),
        };
        slice.push(F::ZERO);
        assert_eq!(layout.validate_against_builder(&builder), Ok(()));
        assert_eq!(
            layout.validate_against_permutation_builder(&builder),
            Err(AirLayoutMismatch {
                input,
                expected,
                actual
            })
        );
    }
}

#[test]
fn explicit_preprocessed_width_is_authoritative_without_materializing_air_data() {
    struct MetadataOnly;
    impl BaseAir<F> for MetadataOnly {
        fn width(&self) -> usize {
            3
        }
        fn preprocessed_width(&self) -> usize {
            9
        }
        fn num_public_values(&self) -> usize {
            4
        }
        fn num_periodic_columns(&self) -> usize {
            5
        }
        fn preprocessed_trace(&self) -> Option<p3_matrix::dense::RowMajorMatrix<F>> {
            panic!("must not materialize preprocessed trace")
        }
        fn periodic_columns(&self) -> std::borrow::Cow<'_, [Vec<F>]> {
            panic!("must not materialize periodic columns")
        }
    }
    let layout = layout();
    layout.validate_against_air(&MetadataOnly);
    assert_eq!(
        layout.validate_against_builder(&TestBuilder::<false>::new(layout)),
        Ok(())
    );
}

#[test]
fn mismatch_is_a_standard_error_with_accessor_and_lengths_in_display() {
    let error = AirLayoutMismatch {
        input: AirLayoutInput::MainNext,
        expected: 3,
        actual: 1,
    };
    let error: &dyn core::error::Error = &error;
    let message = error.to_string();
    assert!(message.contains("main().next_slice()"));
    assert!(message.contains("expected 3"));
    assert!(message.contains("got 1"));
}
