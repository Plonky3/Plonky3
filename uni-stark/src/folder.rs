use alloc::vec::Vec;

use p3_air::{AirBuilder, ExtensionBuilder, RowWindow};
use p3_field::{Algebra, BasedVectorSpace};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

/// Packed constraint folder for SIMD-optimized prover evaluation.
///
/// Uses packed types to evaluate constraints on multiple domain points simultaneously.
///
/// Collects constraints during `air.eval()` into separate base/ext vectors, then
/// combines them in [`Self::finalize_constraints`] using decomposed alpha powers and
/// `batched_linear_combination` for efficient SIMD accumulation.
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    /// The [`RowMajorMatrixView`] containing rows on which the constraint polynomial is evaluated.
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The preprocessed columns as a [`RowMajorMatrixView`].
    /// Zero-width when the AIR has no preprocessed trace.
    pub preprocessed: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// Pre-built window over the preprocessed columns.
    pub preprocessed_window: RowWindow<'a, PackedVal<SC>>,
    /// Periodic column values at the current row(s), one packed value per column.
    pub periodic_values: &'a [PackedVal<SC>],
    /// Public inputs to the [AIR](`p3_air::Air`) implementation.
    pub public_values: &'a [Val<SC>],
    /// Evaluations of the first-row selector polynomial.
    /// Non-zero only on the first trace row.
    pub is_first_row: PackedVal<SC>,
    /// Evaluations of the last-row selector polynomial.
    /// Non-zero only on the last trace row.
    pub is_last_row: PackedVal<SC>,
    /// Evaluations of the transition selector polynomial.
    /// Zero only on the last trace row.
    pub is_transition: PackedVal<SC>,
    /// Base-field alpha powers, reordered to match base constraint emission order.
    /// `base_alpha_powers[d][j]` = d-th basis coefficient of alpha power for j-th base constraint.
    pub base_alpha_powers: &'a [Vec<Val<SC>>],
    /// Extension-field alpha powers, reordered to match ext constraint emission order.
    pub ext_alpha_powers: &'a [SC::Challenge],
    /// Collected base-field constraints for this row
    pub base_constraints: Vec<PackedVal<SC>>,
    /// Collected extension-field constraints for this row
    pub ext_constraints: Vec<PackedChallenge<SC>>,
    /// Current constraint index being processed (debug-only bookkeeping)
    pub constraint_index: usize,
    /// Total number of constraints in the AIR (debug-only bookkeeping)
    pub constraint_count: usize,
}

/// Handles constraint verification for the verifier in a STARK system.
///
/// Similar to [`ProverConstraintFolder`] but operates on committed values rather than the full trace,
/// using a more efficient accumulation method for verification.
#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    /// Pair of consecutive rows from the committed polynomial evaluations as a [`ViewPair`].
    pub main: ViewPair<'a, SC::Challenge>,
    /// The preprocessed columns as a [`ViewPair`].
    /// Zero-width when the AIR has no preprocessed trace.
    pub preprocessed: ViewPair<'a, SC::Challenge>,
    /// Pre-built window over the preprocessed columns.
    pub preprocessed_window: RowWindow<'a, SC::Challenge>,
    /// Periodic column values at the opened point.
    pub periodic_values: &'a [SC::Challenge],
    /// Public values that are inputs to the computation
    pub public_values: &'a [Val<SC>],
    /// Evaluations of the first-row selector polynomial.
    /// Non-zero only on the first trace row.
    pub is_first_row: SC::Challenge,
    /// Evaluations of the last-row selector polynomial.
    /// Non-zero only on the last trace row.
    pub is_last_row: SC::Challenge,
    /// Evaluations of the transition selector polynomial.
    /// Zero only on the last trace row.
    pub is_transition: SC::Challenge,
    /// Single challenge value used for constraint combination
    pub alpha: SC::Challenge,
    /// Running accumulator for all constraints
    pub accumulator: SC::Challenge,
}

impl<SC: StarkGenericConfig> ProverConstraintFolder<'_, SC> {
    /// Combine all collected constraints with their pre-computed alpha powers.
    ///
    /// Base constraints use [`Algebra::batched_linear_combination`] per basis dimension,
    /// decomposing the extension-field multiply into D base-field SIMD dot products.
    /// Extension constraints use the same method with scalar EF coefficients.
    ///
    /// We keep base and extension constraints separate because the base constraints can
    /// stay in the base field and use packed SIMD arithmetic. Decomposing EF powers of
    /// `alpha` into base-field coordinates turns the base-field fold into a small number
    /// of packed dot-products, avoiding repeated cross-field promotions.
    #[inline]
    pub fn finalize_constraints(&self) -> PackedChallenge<SC> {
        debug_assert_eq!(self.constraint_index, self.constraint_count);

        let base = &self.base_constraints;
        let base_powers = self.base_alpha_powers;
        let acc = PackedChallenge::<SC>::from_basis_coefficients_fn(|d| {
            PackedVal::<SC>::batched_linear_combination(base, &base_powers[d])
        });
        acc + PackedChallenge::<SC>::batched_linear_combination(
            &self.ext_constraints,
            self.ext_alpha_powers,
        )
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type PreprocessedWindow = RowWindow<'a, PackedVal<SC>>;
    type MainWindow = RowWindow<'a, PackedVal<SC>>;
    type PublicVar = Val<SC>;
    type PeriodicVar = PackedVal<SC>;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        RowWindow::from_view(&self.main)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.is_transition
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.base_constraints.extend(expr_array);
        self.constraint_index += N;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for ProverConstraintFolder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.ext_constraints.push(x.into());
        self.constraint_index += 1;
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type PreprocessedWindow = RowWindow<'a, SC::Challenge>;
    type MainWindow = RowWindow<'a, SC::Challenge>;
    type PublicVar = Val<SC>;
    type PeriodicVar = SC::Challenge;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.main.top.values, self.main.bottom.values)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        assert!(size <= 2, "only two-row windows are supported, got {size}");
        self.is_transition
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}
