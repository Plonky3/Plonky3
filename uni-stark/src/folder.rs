use alloc::vec::Vec;

use p3_air::{AirBuilder, AirBuilderWithPublicValues};
use p3_field::BasedVectorSpace;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;

use crate::constraint_batch::{batched_base_linear_combination, batched_ext_linear_combination};
use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

/// Handles constraint accumulation for the prover in a STARK system.
///
/// Constraints are **collected** into separate base and extension buckets
/// during [`AirBuilder::assert_zero`] / [`AirBuilder::assert_zeros`] /
/// [`p3_air::ExtensionBuilder::assert_zero_ext`], then combined in a single
/// batched fold via [`ProverConstraintFolder::finalize_constraints`].
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    /// The [`RowMajorMatrixView`] containing rows on which the constraint polynomial is evaluated.
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// The preprocessed columns (if any) as a [`RowMajorMatrixView`].
    pub preprocessed: Option<RowMajorMatrixView<'a, PackedVal<SC>>>,
    /// Public inputs to the [AIR](`p3_air::Air`) implementation.
    pub public_values: &'a [Val<SC>],
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for rows where transition constraints should be applied
    pub is_transition: PackedVal<SC>,
    /// Challenge powers used for randomized constraint combination
    pub alpha_powers: &'a [SC::Challenge],
    /// Challenge powers decomposed into their base field component.
    pub decomposed_alpha_powers: &'a [Vec<Val<SC>>],
    /// Total number of constraints expected (used for debug assertions).
    pub constraint_count: usize,
    /// Collected base-field (packed) constraint expressions for this packed row.
    pub base_constraints: Vec<PackedVal<SC>>,
    /// Collected extension-field (packed) constraint expressions for this packed row.
    pub ext_constraints: Vec<PackedChallenge<SC>>,
    /// Current constraint index being processed
    pub constraint_index: usize,
}

/// Handles constraint verification for the verifier in a STARK system.
///
/// Similar to [`ProverConstraintFolder`] but operates on committed values rather than the full trace,
/// using a more efficient accumulation method for verification.
#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    /// Pair of consecutive rows from the committed polynomial evaluations as a [`ViewPair`].
    pub main: ViewPair<'a, SC::Challenge>,
    /// The preprocessed columns (if any) as a [`ViewPair`].
    pub preprocessed: Option<ViewPair<'a, SC::Challenge>>,
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

impl<'a, SC: StarkGenericConfig> ProverConstraintFolder<'a, SC> {
    /// Fold all collected constraints into a single packed extension field value.
    ///
    /// Base constraints are folded per extension basis dimension using
    /// [`batched_base_linear_combination`], and extension constraints are folded
    /// using [`batched_ext_linear_combination`].
    #[inline]
    pub fn finalize_constraints(self) -> PackedChallenge<SC> {
        debug_assert_eq!(
            self.constraint_index, self.constraint_count,
            "expected {} constraints, got {}",
            self.constraint_count, self.constraint_index,
        );

        let base_count = self.base_constraints.len();
        let ext_count = self.ext_constraints.len();

        // Base constraints: D independent base-field dot products, one per
        // extension basis coefficient. Each produces one packed base coefficient
        // of the folded packed challenge.
        let base_acc = PackedChallenge::<SC>::from_basis_coefficients_fn(|d| {
            batched_base_linear_combination(
                &self.decomposed_alpha_powers[d][..base_count],
                &self.base_constraints,
            )
        });

        // Extension constraints: single EF-coefficient dot product.
        let ext_acc = batched_ext_linear_combination(
            &self.alpha_powers[base_count..base_count + ext_count],
            &self.ext_constraints,
        );

        base_acc + ext_acc
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        self.base_constraints.extend(array.map(Into::into));
        self.constraint_index += N;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> VerifierConstraintFolder<'_, SC> {
    /// Returns the accumulated constraint value.
    #[inline]
    pub const fn finalize_constraints(self) -> SC::Challenge {
        self.accumulator
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for VerifierConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}
