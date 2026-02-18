use alloc::vec::Vec;

use p3_air::lookup::LookupEvaluator;
/// Public re-exports of lookup types.
pub use p3_air::lookup::{Direction, Kind, Lookup, LookupData, LookupError, LookupInput};
use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, Entry, ExtensionBuilder, PermutationAirBuilder,
    SymbolicExpression,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::ViewPair;
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::warn;

/// Converts `LookupData<F>` to `LookupData<SymbolicExpression<F>>`.
pub fn lookup_data_to_expr<F: Clone>(
    lookup_data: &[LookupData<F>],
) -> Vec<LookupData<SymbolicExpression<F>>> {
    lookup_data
        .iter()
        .map(|data| {
            let expected = SymbolicExpression::Constant(data.expected_cumulated.clone());
            LookupData {
                name: data.name.clone(),
                aux_idx: data.aux_idx,
                expected_cumulated: expected,
            }
        })
        .collect()
}

/// A trait for lookup argument.
pub trait LookupGadget: LookupEvaluator {
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
    fn constraint_degree<F: Field>(&self, context: Lookup<F>) -> usize;
}

/// A builder to generate the lookup traces, given the main trace, public values and permutation challenges.
pub struct LookupTraceBuilder<'a, SC: StarkGenericConfig> {
    main: ViewPair<'a, Val<SC>>,
    preprocessed: Option<ViewPair<'a, Val<SC>>>,
    public_values: &'a [Val<SC>],
    permutation_challenges: &'a [SC::Challenge],
    height: usize,
    row: usize,
}

impl<'a, SC: StarkGenericConfig> LookupTraceBuilder<'a, SC> {
    pub const fn new(
        main: ViewPair<'a, Val<SC>>,
        preprocessed: Option<ViewPair<'a, Val<SC>>>,
        public_values: &'a [Val<SC>],
        permutation_challenges: &'a [SC::Challenge],
        height: usize,
        row: usize,
    ) -> Self {
        Self {
            main,
            preprocessed,
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
    type M = ViewPair<'a, Val<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
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
        if size == 2 {
            Self::F::from_bool(self.row + 1 < self.height)
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
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
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for LookupTraceBuilder<'_, SC> {
    type PublicVar = Val<SC>;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
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
    type MP = RowMajorMatrixView<'a, SC::Challenge>;

    type RandomVar = SC::Challenge;
    fn permutation(&self) -> RowMajorMatrixView<'a, SC::Challenge> {
        panic!("we should not be accessing the permutation matrix while building it");
    }

    fn permutation_randomness(&self) -> &[SC::Challenge] {
        self.permutation_challenges
    }
}

/// Evaluates a symbolic expression in the context of an AIR builder.
///
/// Converts `SymbolicExpression<F>` to the builder's expression type `AB::Expr`.
pub fn symbolic_to_expr<AB>(builder: &AB, expr: &SymbolicExpression<AB::F>) -> AB::Expr
where
    AB: AirBuilderWithPublicValues + PermutationAirBuilder,
{
    match expr {
        SymbolicExpression::Variable(v) => match v.entry {
            Entry::Main { offset } => match offset {
                0 => builder.main().row_slice(0).unwrap()[v.index].clone().into(),
                1 => builder.main().row_slice(1).unwrap()[v.index].clone().into(),
                _ => panic!("Cannot have expressions involving more than two rows."),
            },
            Entry::Public => builder.public_values()[v.index].into(),
            Entry::Preprocessed { offset } => match offset {
                0 => builder
                    .preprocessed()
                    .expect("Missing preprocessed columns")
                    .row_slice(0)
                    .unwrap()[v.index]
                    .clone()
                    .into(),
                1 => builder
                    .preprocessed()
                    .expect("Missing preprocessed columns")
                    .row_slice(1)
                    .unwrap()[v.index]
                    .clone()
                    .into(),
                _ => panic!("Cannot have expressions involving more than two rows."),
            },
            _ => unimplemented!("Entry type {:?} not supported in interactions", v.entry),
        },
        SymbolicExpression::IsFirstRow => {
            warn!("IsFirstRow is not normalized");
            builder.is_first_row()
        }
        SymbolicExpression::IsLastRow => {
            warn!("IsLastRow is not normalized");
            builder.is_last_row()
        }
        SymbolicExpression::IsTransition => {
            warn!("IsTransition is not normalized");
            builder.is_transition_window(2)
        }
        SymbolicExpression::Constant(c) => AB::Expr::from(*c),
        SymbolicExpression::Add { x, y, .. } => {
            symbolic_to_expr(builder, x) + symbolic_to_expr(builder, y)
        }
        SymbolicExpression::Sub { x, y, .. } => {
            symbolic_to_expr(builder, x) - symbolic_to_expr(builder, y)
        }
        SymbolicExpression::Neg { x, .. } => -symbolic_to_expr(builder, x),
        SymbolicExpression::Mul { x, y, .. } => {
            symbolic_to_expr(builder, x) * symbolic_to_expr(builder, y)
        }
    }
}
