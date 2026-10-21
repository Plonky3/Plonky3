//! Opt-in comparison of AIR evaluators on concrete assignments.
//!
//! Supply an independently implemented reference and arbitrary field assignments, including
//! assignments that do not satisfy the AIR. Comparing the candidate to itself, or sampling only
//! valid witnesses, can hide errors. Agreement on samples is not a polynomial identity proof.
//! The caller owns sampling and seeding. Values are compared, so harmless reassociation is allowed;
//! swapping same-kind assertions that happen to evaluate equally cannot be detected.
//!
//! ```
//! use p3_air::{AirBuilder, AirLayout, WindowAccess};
//! use p3_air::testing::{EvaluationInputs, compare_scalar};
//! use p3_baby_bear::BabyBear as F;
//! use p3_field::PrimeCharacteristicRing;
//!
//! let inputs = EvaluationInputs::<F>::from_fn(
//!     AirLayout { main_width: 1, ..Default::default() },
//!     vec![],
//!     || F::from_u32(7), // Arbitrary assignments, not necessarily a valid witness.
//!     || F::from_u32(11),
//! );
//! compare_scalar(&inputs,
//!     |b| { let x = b.main().current_slice()[0]; b.assert_zero(x * x + x); },
//!     |b| { let x = b.main().current_slice()[0]; b.assert_zero(x * (x + F::ONE)); },
//! ).unwrap();
//! ```

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue};

use crate::{
    AirBuilder, AirLayout, ExtensionBuilder, PassthroughNamedAirBuilder, PermutationAirBuilder,
    RowWindow,
};

/// Owned assignments for one evaluation, with shared scalar public values.
///
/// `B` and `E` are the base and extension expression rings: normally `F` and `EF`, or their
/// canonical packings. All non-public inputs may vary independently across packed lanes.
/// Row arrays are ordered `[current, next]`; selectors are `[first, last, transition]`.
#[derive(Clone, Debug)]
pub struct EvaluationInputs<F, EF = F, B = F, E = EF> {
    /// Main trace rows.
    pub main: [Vec<B>; 2],
    /// Preprocessed trace rows.
    pub preprocessed: [Vec<B>; 2],
    /// Permutation trace rows.
    pub permutation: [Vec<E>; 2],
    /// Public values, identical across SIMD lanes.
    pub public_values: Vec<F>,
    /// Periodic values at the evaluation point.
    pub periodic_values: Vec<B>,
    /// Permutation challenges.
    pub permutation_randomness: Vec<E>,
    /// Expected permutation values.
    pub permutation_values: Vec<E>,
    /// First-row, last-row, and transition selectors; arbitrary field values are allowed.
    pub selectors: [B; 3],
    extension: PhantomData<EF>,
}

/// Inputs using the fields' canonical native packing types.
pub type PackedEvaluationInputs<F, EF = F> =
    EvaluationInputs<F, EF, <F as Field>::Packing, <EF as ExtensionField<F>>::ExtensionPacking>;

/// A recorder using the fields' canonical native packing types.
pub type PackedRecordingAirBuilder<'a, F, EF = F> = RecordingAirBuilder<
    'a,
    F,
    EF,
    <F as Field>::Packing,
    <EF as ExtensionField<F>>::ExtensionPacking,
>;

impl<F, EF, B, E> EvaluationInputs<F, EF, B, E> {
    /// Sample each input using caller-owned generators.
    ///
    /// Base samples fill main rows, preprocessed rows, periodic values, then selectors.
    /// Extension samples fill permutation rows, challenges, then permutation values.
    /// Within each trace, current rows precede next rows and columns increase in order.
    ///
    /// # Panics
    /// Panics if the public-value count differs from `layout.num_public_values`.
    pub fn from_fn(
        layout: AirLayout,
        public_values: Vec<F>,
        mut base: impl FnMut() -> B,
        mut extension: impl FnMut() -> E,
    ) -> Self {
        assert_eq!(
            public_values.len(),
            layout.num_public_values,
            "public-value count differs"
        );
        Self {
            main: core::array::from_fn(|_| (0..layout.main_width).map(|_| base()).collect()),
            preprocessed: core::array::from_fn(|_| {
                (0..layout.preprocessed_width).map(|_| base()).collect()
            }),
            permutation: core::array::from_fn(|_| {
                (0..layout.permutation_width).map(|_| extension()).collect()
            }),
            public_values,
            periodic_values: (0..layout.num_periodic_columns).map(|_| base()).collect(),
            permutation_randomness: (0..layout.num_permutation_challenges)
                .map(|_| extension())
                .collect(),
            permutation_values: (0..layout.num_permutation_values)
                .map(|_| extension())
                .collect(),
            selectors: core::array::from_fn(|_| base()),
            extension: PhantomData,
        }
    }

    fn shape(&self) -> [usize; 7] {
        assert_eq!(
            self.main[0].len(),
            self.main[1].len(),
            "main row widths differ"
        );
        assert_eq!(
            self.preprocessed[0].len(),
            self.preprocessed[1].len(),
            "preprocessed row widths differ"
        );
        assert_eq!(
            self.permutation[0].len(),
            self.permutation[1].len(),
            "permutation row widths differ"
        );
        [
            self.main[0].len(),
            self.preprocessed[0].len(),
            self.permutation[0].len(),
            self.public_values.len(),
            self.periodic_values.len(),
            self.permutation_randomness.len(),
            self.permutation_values.len(),
        ]
    }
}

impl<F, EF, B, E> EvaluationInputs<F, EF, B, E>
where
    F: Field,
    EF: ExtensionField<F>,
    B: Algebra<F> + Copy + Send + Sync,
    E: Algebra<B> + Algebra<EF> + Copy + Send + Sync,
{
    /// Evaluate a callback, retaining every assertion in global order.
    ///
    /// # Panics
    /// Panics if any current/next row widths differ, including in release builds.
    pub fn record(
        &self,
        evaluate: impl FnOnce(&mut RecordingAirBuilder<'_, F, EF, B, E>),
    ) -> ConstraintEvaluations<B, E> {
        self.shape();
        let mut builder = RecordingAirBuilder {
            inputs: self,
            preprocessed: RowWindow::from_two_rows(&self.preprocessed[0], &self.preprocessed[1]),
            recording: ConstraintEvaluations {
                assertions: Vec::new(),
            },
        };
        evaluate(&mut builder);
        builder.recording
    }
}

impl<F: Field, EF: ExtensionField<F>> EvaluationInputs<F, EF> {
    /// Pack exactly one scalar fixture per native SIMD lane.
    ///
    /// # Panics
    /// Panics on the wrong lane count, malformed or different shapes, or different public values.
    pub fn pack(lanes: &[Self]) -> PackedEvaluationInputs<F, EF> {
        assert_eq!(
            lanes.len(),
            F::Packing::WIDTH,
            "lane count differs from packing width"
        );
        let first = &lanes[0];
        let shape = first.shape();
        for input in lanes {
            assert_eq!(input.shape(), shape, "input shapes differ");
            assert_eq!(
                input.public_values, first.public_values,
                "public values differ"
            );
        }
        let base = |get: fn(&Self) -> &[F]| {
            (0..get(first).len())
                .map(|i| F::Packing::from_fn(|l| get(&lanes[l])[i]))
                .collect()
        };
        let ext = |get: fn(&Self) -> &[EF]| {
            (0..get(first).len())
                .map(|i| EF::ExtensionPacking::from_ext_fn(|l| get(&lanes[l])[i]))
                .collect()
        };
        EvaluationInputs {
            main: [base(|s| &s.main[0]), base(|s| &s.main[1])],
            preprocessed: [base(|s| &s.preprocessed[0]), base(|s| &s.preprocessed[1])],
            permutation: [ext(|s| &s.permutation[0]), ext(|s| &s.permutation[1])],
            public_values: first.public_values.clone(),
            periodic_values: base(|s| &s.periodic_values),
            permutation_randomness: ext(|s| &s.permutation_randomness),
            permutation_values: ext(|s| &s.permutation_values),
            selectors: core::array::from_fn(|i| F::Packing::from_fn(|l| lanes[l].selectors[i])),
            extension: PhantomData,
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> PackedEvaluationInputs<F, EF> {
    /// Extract one scalar fixture, copying the shared public values unchanged.
    ///
    /// # Panics
    /// Panics if `lane` is outside the native packing width or any row pair is malformed.
    pub fn lane(&self, lane: usize) -> EvaluationInputs<F, EF> {
        assert!(lane < F::Packing::WIDTH, "lane outside packing width");
        self.shape();
        let base = |v: &[F::Packing]| v.iter().map(|v| PackedValue::extract(v, lane)).collect();
        let ext = |v: &[EF::ExtensionPacking]| v.iter().map(|v| v.extract(lane)).collect();
        EvaluationInputs {
            main: core::array::from_fn(|i| base(&self.main[i])),
            preprocessed: core::array::from_fn(|i| base(&self.preprocessed[i])),
            permutation: core::array::from_fn(|i| ext(&self.permutation[i])),
            public_values: self.public_values.clone(),
            periodic_values: base(&self.periodic_values),
            permutation_randomness: ext(&self.permutation_randomness),
            permutation_values: ext(&self.permutation_values),
            selectors: self.selectors.map(|v| PackedValue::extract(&v, lane)),
            extension: PhantomData,
        }
    }
}

/// The assertion channel, kept distinct even when the fields are identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssertionKind {
    /// A base expression assertion.
    Base,
    /// An extension expression assertion.
    Extension,
}

/// One concrete assertion value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintValue<B, E> {
    /// Value passed to `assert_zero`.
    Base(B),
    /// Value passed to `assert_zero_ext`.
    Extension(E),
}
impl<B, E> ConstraintValue<B, E> {
    const fn kind(&self) -> AssertionKind {
        match self {
            Self::Base(_) => AssertionKind::Base,
            Self::Extension(_) => AssertionKind::Extension,
        }
    }
}

/// An immutable sequence retaining count, channels, order, zeros, and duplicates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintEvaluations<B, E> {
    assertions: Vec<ConstraintValue<B, E>>,
}
impl<B, E> ConstraintEvaluations<B, E> {
    /// All assertions in global emission order. The slice index is the global assertion index.
    pub fn assertions(&self) -> &[ConstraintValue<B, E>] {
        &self.assertions
    }
}
impl<B: Copy + PartialEq, E: Copy + PartialEq> ConstraintEvaluations<B, E> {
    /// Compare count, then the complete channel sequence, then values.
    ///
    /// Structural differences take priority even when an earlier value differs.
    pub fn compare(&self, actual: &Self) -> Result<(), ConstraintMismatch<B, E>> {
        if self.assertions.len() != actual.assertions.len() {
            return Err(ConstraintMismatch::Count {
                expected: self.assertions.len(),
                actual: actual.assertions.len(),
            });
        }
        for (assertion, (expected, actual)) in
            self.assertions.iter().zip(&actual.assertions).enumerate()
        {
            if expected.kind() != actual.kind() {
                return Err(ConstraintMismatch::Kind {
                    assertion,
                    expected: expected.kind(),
                    actual: actual.kind(),
                });
            }
        }
        for (assertion, (&expected, &actual)) in
            self.assertions.iter().zip(&actual.assertions).enumerate()
        {
            if expected != actual {
                return Err(ConstraintMismatch::Value {
                    assertion,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// First difference, with structural checks performed before any value checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstraintMismatch<B, E> {
    /// Total assertion counts differ.
    Count {
        /// Reference count.
        expected: usize,
        /// Candidate count.
        actual: usize,
    },
    /// The base/extension routing differs at a global assertion index.
    Kind {
        /// Global assertion index.
        assertion: usize,
        /// Reference channel.
        expected: AssertionKind,
        /// Candidate channel.
        actual: AssertionKind,
    },
    /// Concrete values differ at a global assertion index.
    Value {
        /// Global assertion index.
        assertion: usize,
        /// Reference value.
        expected: ConstraintValue<B, E>,
        /// Candidate value.
        actual: ConstraintValue<B, E>,
    },
}

/// A packed comparison failure, located at the first disagreeing SIMD lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedMismatch<F, EF> {
    /// SIMD lane checked against its scalar reference.
    pub lane: usize,
    /// Difference within that lane, with a global assertion index when counts agree.
    pub mismatch: ConstraintMismatch<F, EF>,
}

/// Compare independently supplied scalar evaluators on identical inputs.
pub fn compare_scalar<F: Field, EF: ExtensionField<F>>(
    inputs: &EvaluationInputs<F, EF>,
    reference: impl FnOnce(&mut RecordingAirBuilder<'_, F, EF>),
    candidate: impl FnOnce(&mut RecordingAirBuilder<'_, F, EF>),
) -> Result<(), ConstraintMismatch<F, EF>> {
    inputs.record(reference).compare(&inputs.record(candidate))
}

/// Compare one packed candidate evaluation to an independent scalar reference on every lane.
///
/// The reference is invoked once per lane. It must evaluate the same constraints each time;
/// callback state is the caller's responsibility. Native packing width may be one on some targets.
pub fn compare_packed<F: Field, EF: ExtensionField<F>>(
    inputs: &PackedEvaluationInputs<F, EF>,
    mut reference: impl FnMut(&mut RecordingAirBuilder<'_, F, EF>),
    candidate: impl FnOnce(&mut PackedRecordingAirBuilder<'_, F, EF>),
) -> Result<(), PackedMismatch<F, EF>> {
    let actual = inputs.record(candidate);
    for lane in 0..F::Packing::WIDTH {
        let expected = inputs.lane(lane).record(&mut reference);
        let actual = ConstraintEvaluations {
            assertions: actual
                .assertions
                .iter()
                .map(|v| match v {
                    ConstraintValue::Base(v) => {
                        ConstraintValue::Base(PackedValue::extract(v, lane))
                    }
                    ConstraintValue::Extension(v) => ConstraintValue::Extension(v.extract(lane)),
                })
                .collect(),
        };
        expected
            .compare(&actual)
            .map_err(|mismatch| PackedMismatch { lane, mismatch })?;
    }
    Ok(())
}

/// A concrete builder recording all assertions without testing whether they vanish.
///
/// The expression bounds deliberately allow test rings as well as fields and packed fields.
/// Named assertions are recorded without evaluating their labels.
pub struct RecordingAirBuilder<'a, F, EF = F, B = F, E = EF> {
    inputs: &'a EvaluationInputs<F, EF, B, E>,
    preprocessed: RowWindow<'a, B>,
    recording: ConstraintEvaluations<B, E>,
}
impl<'a, F, EF, B, E> AirBuilder for RecordingAirBuilder<'a, F, EF, B, E>
where
    F: Field,
    B: Algebra<F> + Copy + Send + Sync,
{
    type F = F;
    type Expr = B;
    type Var = B;
    type MainWindow = RowWindow<'a, B>;
    type PreprocessedWindow = RowWindow<'a, B>;
    type PublicVar = F;
    type PeriodicVar = B;
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(&self.inputs.main[0], &self.inputs.main[1])
    }
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }
    fn is_first_row(&self) -> B {
        self.inputs.selectors[0]
    }
    fn is_last_row(&self) -> B {
        self.inputs.selectors[1]
    }
    fn is_transition(&self) -> B {
        self.inputs.selectors[2]
    }
    fn public_values(&self) -> &[F] {
        &self.inputs.public_values
    }
    fn periodic_values(&self) -> &[B] {
        &self.inputs.periodic_values
    }
    fn assert_zero<I: Into<B>>(&mut self, x: I) {
        self.recording
            .assertions
            .push(ConstraintValue::Base(x.into()));
    }
}
impl<F, EF, B, E> ExtensionBuilder for RecordingAirBuilder<'_, F, EF, B, E>
where
    F: Field,
    EF: ExtensionField<F>,
    B: Algebra<F> + Copy + Send + Sync,
    E: Algebra<B> + Algebra<EF> + Copy + Send + Sync,
{
    type EF = EF;
    type ExprEF = E;
    type VarEF = E;
    fn assert_zero_ext<I: Into<E>>(&mut self, x: I) {
        self.recording
            .assertions
            .push(ConstraintValue::Extension(x.into()));
    }
}
impl<'a, F, EF, B, E> PermutationAirBuilder for RecordingAirBuilder<'a, F, EF, B, E>
where
    F: Field,
    EF: ExtensionField<F>,
    B: Algebra<F> + Copy + Send + Sync,
    E: Algebra<B> + Algebra<EF> + Copy + Send + Sync,
{
    type MP = RowWindow<'a, E>;
    type RandomVar = E;
    type PermutationVar = E;
    fn permutation(&self) -> Self::MP {
        RowWindow::from_two_rows(&self.inputs.permutation[0], &self.inputs.permutation[1])
    }
    fn permutation_randomness(&self) -> &[E] {
        &self.inputs.permutation_randomness
    }
    fn permutation_values(&self) -> &[E] {
        &self.inputs.permutation_values
    }
}
impl<F, EF, B, E> PassthroughNamedAirBuilder for RecordingAirBuilder<'_, F, EF, B, E>
where
    F: Field,
    B: Algebra<F> + Copy + Send + Sync,
{
}
