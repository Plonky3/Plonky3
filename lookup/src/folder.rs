use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;
use p3_uni_stark::{
    PackedChallenge, PackedVal, ProverConstraintFolder, StarkGenericConfig, Val,
    VerifierConstraintFolder,
};

pub struct ProverConstraintFolderWithLookups<'a, SC: StarkGenericConfig> {
    pub inner: ProverConstraintFolder<'a, SC>,
    pub permutation: RowMajorMatrixView<'a, PackedChallenge<SC>>,
    pub permutation_challenges: &'a [PackedChallenge<SC>],
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolderWithLookups<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;

    fn main(&self) -> Self::M {
        self.inner.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row
    }

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.inner.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x);
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.inner.assert_zeros(array);
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues
    for ProverConstraintFolderWithLookups<'_, SC>
{
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.inner.public_values
    }
}

impl<SC: StarkGenericConfig> PairBuilder for ProverConstraintFolderWithLookups<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.inner
            .preprocessed
            .unwrap_or_else(|| panic!("Missing preprocessed columns"))
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
        let alpha_power = self.inner.alpha_powers[self.inner.constraint_index];
        self.inner.accumulator += <PackedChallenge<SC>>::from(alpha_power) * x.into();
        self.inner.constraint_index += 1;
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder
    for ProverConstraintFolderWithLookups<'a, SC>
{
    type MP = RowMajorMatrixView<'a, PackedChallenge<SC>>;

    type RandomVar = PackedChallenge<SC>;
    fn permutation(&self) -> RowMajorMatrixView<'a, PackedChallenge<SC>> {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[PackedChallenge<SC>] {
        self.permutation_challenges
    }
}

pub struct VerifierConstraintFolderWithLookups<'a, SC: StarkGenericConfig> {
    pub inner: VerifierConstraintFolder<'a, SC>,
    pub permutation: ViewPair<'a, SC::Challenge>,
    pub permutation_challenges: &'a [SC::Challenge],
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolderWithLookups<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;

    fn main(&self) -> Self::M {
        self.inner.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row
    }

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.inner.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x);
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.inner.assert_zeros(array);
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues
    for VerifierConstraintFolderWithLookups<'_, SC>
{
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.inner.public_values
    }
}

impl<SC: StarkGenericConfig> PairBuilder for VerifierConstraintFolderWithLookups<'_, SC> {
    fn preprocessed(&self) -> Self::M {
        self.inner
            .preprocessed
            .unwrap_or_else(|| panic!("Missing preprocessed columns"))
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
    type MP = ViewPair<'a, SC::Challenge>;

    type RandomVar = SC::Challenge;

    fn permutation(&self) -> ViewPair<'a, SC::Challenge> {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[SC::Challenge] {
        self.permutation_challenges
    }
}
