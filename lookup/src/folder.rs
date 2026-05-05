use alloc::vec::Vec;

use p3_air::{AirBuilder, ExtensionBuilder, PermutationAirBuilder, RowWindow};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;
use p3_uni_stark::{
    PackedChallenge, PackedVal, ProverConstraintFolder, StarkGenericConfig, Val,
    VerifierConstraintFolder,
};

use crate::builder::InteractionBuilder;

pub struct ProverConstraintFolderWithLookups<'a, SC: StarkGenericConfig> {
    pub inner: ProverConstraintFolder<'a, SC>,
    pub permutation: RowMajorMatrixView<'a, PackedChallenge<SC>>,
    pub permutation_challenges: &'a [PackedChallenge<SC>],
    pub permutation_values: &'a [PackedChallenge<SC>],
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolderWithLookups<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type PreprocessedWindow = RowWindow<'a, PackedVal<SC>>;
    type MainWindow = RowWindow<'a, PackedVal<SC>>;
    type PublicVar = Val<SC>;
    type PeriodicVar = PackedVal<SC>;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_view(&self.inner.main)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        self.inner.preprocessed()
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.inner.is_transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x);
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.inner.assert_zeros(array);
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for ProverConstraintFolderWithLookups<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.inner.assert_zero_ext(x);
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder
    for ProverConstraintFolderWithLookups<'a, SC>
{
    type RandomVar = PackedChallenge<SC>;
    type MP = RowWindow<'a, PackedChallenge<SC>>;

    type PermutationVar = PackedChallenge<SC>;

    fn permutation(&self) -> Self::MP {
        RowWindow::from_view(&self.permutation)
    }

    fn permutation_randomness(&self) -> &[PackedChallenge<SC>] {
        self.permutation_challenges
    }

    fn permutation_values(&self) -> &[PackedChallenge<SC>] {
        self.permutation_values
    }
}

pub struct VerifierConstraintFolderWithLookups<'a, SC: StarkGenericConfig> {
    pub inner: VerifierConstraintFolder<'a, SC>,
    pub permutation: ViewPair<'a, SC::Challenge>,
    pub permutation_challenges: &'a [SC::Challenge],
    pub permutation_values: &'a [SC::Challenge],
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolderWithLookups<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type PublicVar = Val<SC>;
    type PreprocessedWindow = RowWindow<'a, SC::Challenge>;
    type MainWindow = RowWindow<'a, SC::Challenge>;
    type PeriodicVar = SC::Challenge;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.inner.main.top.values, self.inner.main.bottom.values)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        self.inner.preprocessed()
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.inner.is_transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x);
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.inner.assert_zeros(array);
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for VerifierConstraintFolderWithLookups<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.inner.accumulator *= self.inner.alpha;
        self.inner.accumulator += x.into();
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder
    for VerifierConstraintFolderWithLookups<'a, SC>
{
    type RandomVar = SC::Challenge;
    type MP = RowWindow<'a, SC::Challenge>;

    type PermutationVar = SC::Challenge;

    fn permutation(&self) -> Self::MP {
        RowWindow::from_two_rows(self.permutation.top.values, self.permutation.bottom.values)
    }

    fn permutation_randomness(&self) -> &[SC::Challenge] {
        self.permutation_challenges
    }

    fn permutation_values(&self) -> &[SC::Challenge] {
        self.permutation_values
    }
}

impl<SC: StarkGenericConfig> InteractionBuilder for ProverConstraintFolderWithLookups<'_, SC> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        _count: impl Into<Self::Expr>,
        _count_weight: u32,
    ) {
        // Drain the iterator so side effects in a wrapping adapter still fire,
        // preserving the semantics of a real recording builder.
        fields.into_iter().for_each(drop);
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Self::Expr)>,
    ) {
        // Same rationale as the global path: keep iterator consumption observable.
        tuples.into_iter().for_each(drop);
    }
}

impl<SC: StarkGenericConfig> InteractionBuilder for VerifierConstraintFolderWithLookups<'_, SC> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        _count: impl Into<Self::Expr>,
        _count_weight: u32,
    ) {
        // Drain the iterator so side effects in a wrapping adapter still fire.
        fields.into_iter().for_each(drop);
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Self::Expr)>,
    ) {
        // Same rationale as the global path.
        tuples.into_iter().for_each(drop);
    }
}
