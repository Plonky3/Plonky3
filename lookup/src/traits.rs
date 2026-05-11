use p3_air::{AirBuilder, ExtensionBuilder, PermutationAirBuilder, RowWindow};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::ViewPair;
use p3_uni_stark::{StarkGenericConfig, Val};

pub use crate::types::{Kind, Lookup, LookupData, LookupError};

/// A trait for lookup argument.
pub trait LookupGadget {
    /// Generates the permutation matrix for the lookup argument.
    fn generate_permutation<SC: StarkGenericConfig>(
        &self,
        main: &RowMajorMatrix<Val<SC>>,
        preprocessed: &Option<RowMajorMatrix<Val<SC>>>,
        public_values: &[Val<SC>],
        lookups: &[Lookup<Val<SC>>],
        lookup_data: &mut [LookupData<SC::Challenge>],
        permutation_challenges: &[SC::Challenge],
    ) -> RowMajorMatrix<SC::Challenge>;

    /// Evaluates the final cumulated value over all AIRs involved in the interaction,
    /// and checks that it is equal to the expected final value.
    ///
    /// For example, in LogUp:
    /// - it sums all expected cumulated values provided by each AIR within one interaction,
    /// - checks that the sum is equal to 0.
    fn verify_global_final_value<EF: Field>(
        &self,
        all_expected_cumulated: &[EF],
    ) -> Result<(), LookupError>;

    /// Computes the polynomial degree of a lookup transition constraint.
    fn constraint_degree<F: Field>(&self, context: &Lookup<F>) -> usize;
}

/// A builder to generate the lookup traces, given the main trace, public values and permutation challenges.
pub struct LookupTraceBuilder<'a, SC: StarkGenericConfig> {
    main: ViewPair<'a, Val<SC>>,
    preprocessed: RowWindow<'a, Val<SC>>,
    public_values: &'a [Val<SC>],
    permutation_challenges: &'a [SC::Challenge],
    height: usize,
    row: usize,
}

impl<'a, SC: StarkGenericConfig> LookupTraceBuilder<'a, SC> {
    pub fn new(
        main: ViewPair<'a, Val<SC>>,
        preprocessed: ViewPair<'a, Val<SC>>,
        public_values: &'a [Val<SC>],
        permutation_challenges: &'a [SC::Challenge],
        height: usize,
        row: usize,
    ) -> Self {
        Self {
            main,
            preprocessed: RowWindow::from_two_rows(
                preprocessed.top.values,
                preprocessed.bottom.values,
            ),
            public_values,
            permutation_challenges,
            height,
            row,
        }
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for LookupTraceBuilder<'a, SC> {
    type F = Val<SC>;
    type Expr = Val<SC>;
    type Var = Val<SC>;
    type PreprocessedWindow = RowWindow<'a, Val<SC>>;
    type MainWindow = RowWindow<'a, Val<SC>>;
    type PublicVar = Val<SC>;
    type PeriodicVar = Val<SC>;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.main.top.values, self.main.bottom.values)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        Self::F::from_bool(self.row == 0)
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        Self::F::from_bool(self.row + 1 == self.height)
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        Self::F::from_bool(self.row + 1 < self.height)
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert!(x.into() == Self::F::ZERO);
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for item in array {
            assert!(item.into() == Self::F::ZERO);
        }
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for LookupTraceBuilder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        assert!(x.into() == SC::Challenge::ZERO);
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder for LookupTraceBuilder<'a, SC> {
    type MP = RowWindow<'a, SC::Challenge>;
    type RandomVar = SC::Challenge;

    type PermutationVar = SC::Challenge;

    fn permutation(&self) -> Self::MP {
        panic!("we should not be accessing the permutation matrix while building it");
    }

    fn permutation_randomness(&self) -> &[SC::Challenge] {
        self.permutation_challenges
    }

    fn permutation_values(&self) -> &[SC::Challenge] {
        &[]
    }
}
