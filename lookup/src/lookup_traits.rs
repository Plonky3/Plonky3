use alloc::vec::Vec;

use p3_air::lookup::LookupEvaluator;
/// Public re-exports of lookup types.
pub use p3_air::lookup::{Direction, Kind, Lookup, LookupData, LookupError, LookupInput};
use p3_air::{
    AirBuilder, Entry, ExtensionBuilder, PermutationAirBuilder, RowWindow, SymbolicExpression,
    WindowAccess,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
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
    fn constraint_degree<F: Field>(&self, context: &Lookup<F>) -> usize;
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
    type PublicVar = Val<SC>;
    type M = RowWindow<'a, Val<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        RowWindow::new(self.main.top.values, self.main.bottom.values)
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
            .map(|p| RowWindow::new(p.top.values, p.bottom.values))
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
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

    fn permutation(&self) -> Self::MP {
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
    AB: AirBuilder + PermutationAirBuilder,
{
    match expr {
        SymbolicExpression::Variable(v) => match v.entry {
            Entry::Main { offset } => {
                let main = builder.main();
                match offset {
                    0 => main.local(v.index).into(),
                    1 => main.next(v.index).into(),
                    _ => panic!("Cannot have expressions involving more than two rows."),
                }
            }
            Entry::Public => builder.public_values()[v.index].into(),
            Entry::Preprocessed { offset } => {
                let prep = builder
                    .preprocessed()
                    .expect("Missing preprocessed columns");
                match offset {
                    0 => prep.local(v.index).into(),
                    1 => prep.next(v.index).into(),
                    _ => panic!("Cannot have expressions involving more than two rows."),
                }
            }
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
            builder.is_transition()
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
