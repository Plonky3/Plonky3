use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use super::{AirLayout, ConstraintLayout};
use crate::{
    AirBuilder, ExtensionBuilder, PassthroughNamedAirBuilder, PermutationAirBuilder, WindowAccess,
};

/// Owned two-row input window for [`ConstraintLayoutBuilder`].
///
/// Both rows remain accessible when the width is zero.
#[derive(Clone, Debug)]
pub struct ConstraintLayoutWindow<T> {
    values: Option<Arc<[T]>>,
}

impl<T: Clone> ConstraintLayoutWindow<T> {
    fn new(width: usize, zero: T) -> Self {
        Self {
            values: (width != 0).then(|| core::iter::repeat_n(zero, 2 * width).collect()),
        }
    }
}

impl<T> WindowAccess<T> for ConstraintLayoutWindow<T> {
    fn current_slice(&self) -> &[T] {
        let values = self.values.as_deref().unwrap_or_default();
        &values[..values.len() / 2]
    }

    fn next_slice(&self) -> &[T] {
        let values = self.values.as_deref().unwrap_or_default();
        &values[values.len() / 2..]
    }
}

/// Collects assertion ordering using concrete field arithmetic, without expression trees.
///
/// All inputs and selectors are zero; base expressions use `F`, while extension expressions use `EF`.
/// Every assertion is recorded, including duplicates and assertions under zero filters.
/// Constraint values are discarded.
///
/// This is useful for evaluators generic over their builder.
/// An AIR may specialize its implementation for a builder type or branch on concrete values, so this collector need not reproduce evaluation by [`super::SymbolicAirBuilder`].
/// [`super::get_constraint_layout`] continues to use symbolic evaluation.
#[derive(Debug)]
pub struct ConstraintLayoutBuilder<F: Field, EF: ExtensionField<F> = F> {
    main: ConstraintLayoutWindow<F>,
    preprocessed: ConstraintLayoutWindow<F>,
    permutation: ConstraintLayoutWindow<EF>,
    public_values: Vec<F>,
    periodic_values: Vec<F>,
    permutation_randomness: Vec<EF>,
    permutation_values: Vec<EF>,
    constraint_layout: ConstraintLayout,
}

impl<F: Field, EF: ExtensionField<F>> ConstraintLayoutBuilder<F, EF> {
    /// Allocate zero-valued inputs with the supplied widths and counts.
    pub fn new(layout: AirLayout) -> Self {
        Self {
            main: ConstraintLayoutWindow::new(layout.main_width, F::ZERO),
            preprocessed: ConstraintLayoutWindow::new(layout.preprocessed_width, F::ZERO),
            permutation: ConstraintLayoutWindow::new(layout.permutation_width, EF::ZERO),
            public_values: vec![F::ZERO; layout.num_public_values],
            periodic_values: vec![F::ZERO; layout.num_periodic_columns],
            permutation_randomness: vec![EF::ZERO; layout.num_permutation_challenges],
            permutation_values: vec![EF::ZERO; layout.num_permutation_values],
            constraint_layout: ConstraintLayout::default(),
        }
    }

    /// Consume the builder and return global indices for each assertion stream.
    pub fn finish(self) -> ConstraintLayout {
        self.constraint_layout
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for ConstraintLayoutBuilder<F, EF> {
    type F = F;
    type Expr = F;
    type Var = F;
    type MainWindow = ConstraintLayoutWindow<F>;
    type PreprocessedWindow = ConstraintLayoutWindow<F>;
    type PublicVar = F;
    type PeriodicVar = F;

    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> F {
        F::ZERO
    }

    fn is_last_row(&self) -> F {
        F::ZERO
    }

    fn is_transition(&self) -> F {
        F::ZERO
    }

    fn assert_zero<I: Into<F>>(&mut self, _x: I) {
        let index = self.constraint_layout.total_constraints();
        self.constraint_layout.base_indices.push(index);
    }

    fn public_values(&self) -> &[F] {
        &self.public_values
    }

    fn periodic_values(&self) -> &[F] {
        &self.periodic_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for ConstraintLayoutBuilder<F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    fn assert_zero_ext<I: Into<EF>>(&mut self, _x: I) {
        let index = self.constraint_layout.total_constraints();
        self.constraint_layout.ext_indices.push(index);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for ConstraintLayoutBuilder<F, EF> {
    type MP = ConstraintLayoutWindow<EF>;
    type RandomVar = EF;
    type PermutationVar = EF;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[EF] {
        &self.permutation_randomness
    }

    fn permutation_values(&self) -> &[EF] {
        &self.permutation_values
    }
}

impl<F: Field, EF: ExtensionField<F>> PassthroughNamedAirBuilder
    for ConstraintLayoutBuilder<F, EF>
{
}

/// Evaluate a closure once to collect assertion ordering without symbolic capture.
///
/// The closure need not implement [`crate::Air`] or [`crate::BaseAir`].
/// The caller supplies the shape of every input it reads.
/// See [`ConstraintLayoutBuilder`] for the zero-valued evaluation semantics.
///
/// ```
/// use p3_air::{AirBuilder, AirLayout, WindowAccess, capture_constraint_layout};
/// use p3_baby_bear::BabyBear;
///
/// let layout = capture_constraint_layout::<BabyBear, BabyBear>(
///     AirLayout { main_width: 1, ..Default::default() },
///     |builder| {
///         let main = builder.main();
///         builder.assert_bool(main.current(0).unwrap());
///         builder.when_first_row().assert_eq(main.next(0).unwrap(), main.current(0).unwrap());
///     },
/// );
/// assert_eq!(layout.base_indices, [0, 1]);
/// assert!(layout.ext_indices.is_empty());
/// ```
pub fn capture_constraint_layout<F: Field, EF: ExtensionField<F>>(
    layout: AirLayout,
    eval: impl FnOnce(&mut ConstraintLayoutBuilder<F, EF>),
) -> ConstraintLayout {
    let mut builder = ConstraintLayoutBuilder::new(layout);
    eval(&mut builder);
    builder.finish()
}
