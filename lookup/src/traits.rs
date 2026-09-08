use p3_air::{AirBuilder, ExtensionBuilder, PermutationAirBuilder, RowWindow};
use p3_field::{ExtensionField, Field};
use p3_matrix::stack::ViewPair;

pub use crate::types::{Kind, Lookup, LookupError, LookupTerminal};

/// A builder to generate the lookup traces, given the main trace, public values and permutation challenges.
pub struct LookupTraceBuilder<'a, F: Field, EF: ExtensionField<F>> {
    main: ViewPair<'a, F>,
    preprocessed: RowWindow<'a, F>,
    public_values: &'a [F],
    permutation_challenges: &'a [EF],
    height: usize,
    row: usize,
}

impl<'a, F: Field, EF: ExtensionField<F>> LookupTraceBuilder<'a, F, EF> {
    pub fn new(
        main: ViewPair<'a, F>,
        preprocessed: ViewPair<'a, F>,
        public_values: &'a [F],
        permutation_challenges: &'a [EF],
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

impl<'a, F: Field, EF: ExtensionField<F>> AirBuilder for LookupTraceBuilder<'a, F, EF> {
    type F = F;
    type Expr = F;
    type Var = F;
    type PreprocessedWindow = RowWindow<'a, F>;
    type MainWindow = RowWindow<'a, F>;
    type PublicVar = F;
    type PeriodicVar = F;

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
    fn is_transition(&self) -> Self::Expr {
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

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for LookupTraceBuilder<'_, F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        assert!(x.into() == EF::ZERO);
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PermutationAirBuilder for LookupTraceBuilder<'a, F, EF> {
    type MP = RowWindow<'a, EF>;
    type RandomVar = EF;

    type PermutationVar = EF;

    fn permutation(&self) -> Self::MP {
        panic!("we should not be accessing the permutation matrix while building it");
    }

    fn permutation_randomness(&self) -> &[EF] {
        self.permutation_challenges
    }

    fn permutation_values(&self) -> &[EF] {
        &[]
    }
}
