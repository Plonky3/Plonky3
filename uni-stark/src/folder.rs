use alloc::vec::Vec;

use p3_air::{AirBuilder, ExtensionBuilder, PeriodicAirBuilder, RowWindow};
use p3_field::{Algebra, BasedVectorSpace, PackedField, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

/// Batch size for constraint linear-combination chunks in [`finalize_constraints`].
const CONSTRAINT_BATCH: usize = 8;

/// Batched linear combination of packed extension field values with EF coefficients.
///
/// Extension-field analogue of [`PackedField::packed_linear_combination`]. Processes
/// `coeffs` and `values` in chunks of [`CONSTRAINT_BATCH`], then handles the remainder.
#[inline]
fn batched_ext_linear_combination<PE, EF>(coeffs: &[EF], values: &[PE]) -> PE
where
    EF: p3_field::Field,
    PE: PrimeCharacteristicRing + Algebra<EF> + Copy,
{
    debug_assert_eq!(coeffs.len(), values.len());

    let len = coeffs.len();
    let mut acc = PE::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        let batch: [PE; CONSTRAINT_BATCH] =
            core::array::from_fn(|i| values[start + i] * coeffs[start + i]);
        acc += PE::sum_array::<CONSTRAINT_BATCH>(&batch);
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Batched linear combination of packed base field values with F coefficients.
///
/// Wraps [`PackedField::packed_linear_combination`] with batched chunking
/// and remainder handling, mirroring [`batched_ext_linear_combination`].
#[inline]
fn batched_base_linear_combination<P: PackedField>(coeffs: &[P::Scalar], values: &[P]) -> P {
    debug_assert_eq!(coeffs.len(), values.len());

    let len = coeffs.len();
    let mut acc = P::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        acc += P::packed_linear_combination::<CONSTRAINT_BATCH>(
            &coeffs[start..start + CONSTRAINT_BATCH],
            &values[start..start + CONSTRAINT_BATCH],
        );
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Packed constraint folder for SIMD-optimized prover evaluation.
///
/// Uses packed types to evaluate constraints on multiple domain points simultaneously.
///
/// Collects constraints during `air.eval()` into separate base/ext vectors, then
/// combines them in [`Self::finalize_constraints`] using decomposed alpha powers and
/// `packed_linear_combination` for efficient SIMD accumulation.
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    /// The [`RowMajorMatrixView`] containing rows on which the constraint polynomial is evaluated.
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The preprocessed columns as a [`RowMajorMatrixView`].
    /// Zero-width when the AIR has no preprocessed trace.
    pub preprocessed: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// Pre-built window over the preprocessed columns.
    pub preprocessed_window: RowWindow<'a, PackedVal<SC>>,
    /// Periodic column values at the current packed row.
    pub periodic_values: &'a [PackedVal<SC>],
    /// Public inputs to the [AIR](`p3_air::Air`) implementation.
    pub public_values: &'a [Val<SC>],
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for rows where transition constraints should be applied
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
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: SC::Challenge,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: SC::Challenge,
    /// Evaluations of the Selector polynomial for rows where transition constraints should be applied
    pub is_transition: SC::Challenge,
    /// Single challenge value used for constraint combination
    pub alpha: SC::Challenge,
    /// Running accumulator for all constraints
    pub accumulator: SC::Challenge,
}

impl<SC: StarkGenericConfig> ProverConstraintFolder<'_, SC> {
    /// Combine all collected constraints with their pre-computed alpha powers.
    ///
    /// Base constraints use `batched_base_linear_combination` per basis dimension,
    /// decomposing the extension-field multiply into D base-field SIMD dot products.
    /// Extension constraints use `batched_ext_linear_combination` with scalar EF
    /// coefficients. Both process in chunks of `CONSTRAINT_BATCH`.
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
            batched_base_linear_combination(&base_powers[d], base)
        });
        acc + batched_ext_linear_combination(self.ext_alpha_powers, &self.ext_constraints)
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type PreprocessedWindow = RowWindow<'a, PackedVal<SC>>;
    type MainWindow = RowWindow<'a, PackedVal<SC>>;
    type PublicVar = Val<SC>;

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

impl<SC: StarkGenericConfig> PeriodicAirBuilder for ProverConstraintFolder<'_, SC> {
    type PeriodicVar = PackedVal<SC>;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type PreprocessedWindow = RowWindow<'a, SC::Challenge>;
    type MainWindow = RowWindow<'a, SC::Challenge>;
    type PublicVar = Val<SC>;

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
}

impl<SC: StarkGenericConfig> PeriodicAirBuilder for VerifierConstraintFolder<'_, SC> {
    type PeriodicVar = SC::Challenge;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}
