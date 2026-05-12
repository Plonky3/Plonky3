use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::symbolic::SymbolicExpr;
use crate::symbolic::expression::BaseLeaf;
use crate::symbolic::expression_ext::SymbolicExpressionExt;
use crate::symbolic::variable::{BaseEntry, ExtEntry, SymbolicVariableExt};
use crate::{
    Air, AirBuilder, BaseAir, ExtensionBuilder, PermutationAirBuilder, SymbolicExpression,
    SymbolicVariable, WindowAccess,
};

/// Describes the shape of an AIR for symbolic constraint evaluation.
///
/// Bundles the various width/count parameters needed to construct a
/// [`SymbolicAirBuilder`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AirLayout {
    /// Width of [`AirBuilder::preprocessed`].
    pub preprocessed_width: usize,
    /// Width of [`AirBuilder::main`].
    pub main_width: usize,
    /// Length of [`AirBuilder::public_values`].
    pub num_public_values: usize,
    /// Width of [`PermutationAirBuilder::permutation`].
    pub permutation_width: usize,
    /// Length of [`PermutationAirBuilder::permutation_randomness`].
    pub num_permutation_challenges: usize,
    /// Length of [`PermutationAirBuilder::permutation_values`].
    pub num_permutation_values: usize,
    /// Length of [`AirBuilder::periodic_values`].
    pub num_periodic_columns: usize,
}

impl AirLayout {
    /// Derive layout from an AIR's metadata.
    pub fn from_air<F: Clone + Send + Sync>(air: &impl BaseAir<F>) -> Self {
        Self {
            preprocessed_width: air.preprocessed_width(),
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            num_periodic_columns: air.num_periodic_columns(),
            ..Default::default()
        }
    }
}

#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(air: &A, layout: AirLayout) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(air, layout)
}

#[instrument(
    name = "infer base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_max_constraint_degree_extension<F, EF, A>(air: &A, layout: AirLayout) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) = get_all_symbolic_constraints(air, layout);

    let base_degree = base_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);

    let extension_degree = extension_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);
    base_degree.max(extension_degree)
}

#[instrument(
    name = "evaluate base constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints<F, A>(air: &A, layout: AirLayout) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    builder.base_constraints()
}

#[instrument(
    name = "evaluate extension constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints_extension<F, EF, A>(
    air: &A,
    layout: AirLayout,
) -> Vec<SymbolicExpressionExt<F, EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    builder.extension_constraints()
}

#[instrument(
    name = "evaluate all constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_all_symbolic_constraints<F, EF, A>(
    air: &A,
    layout: AirLayout,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<SymbolicExpressionExt<F, EF>>,
)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    (builder.base_constraints(), builder.extension_constraints())
}

/// Symbolic AIR builder that records constraints.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F> = F> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    periodic: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    permutation: RowMajorMatrix<SymbolicVariableExt<F, EF>>,
    permutation_challenges: Vec<SymbolicVariableExt<F, EF>>,
    permutation_values: Vec<SymbolicVariableExt<F, EF>>,
    extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
    constraint_types: Vec<ConstraintType>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    pub fn new(layout: AirLayout) -> Self {
        let AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width,
            num_permutation_challenges,
            num_permutation_values,
            num_periodic_columns,
        } = layout;
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width).map(move |index| {
                    SymbolicVariable::new(BaseEntry::Preprocessed { offset }, index)
                })
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..main_width)
                    .map(move |index| SymbolicVariable::new(BaseEntry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(BaseEntry::Public, index))
            .collect();
        let periodic = (0..num_periodic_columns)
            .map(|index| SymbolicVariable::new(BaseEntry::Periodic, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width).map(move |index| {
                    SymbolicVariableExt::new(ExtEntry::Permutation { offset }, index)
                })
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| SymbolicVariableExt::new(ExtEntry::Challenge, index))
            .collect();
        let permutation_values = (0..num_permutation_values)
            .map(|index| SymbolicVariableExt::new(ExtEntry::PermutationValue, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            public_values,
            periodic,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            permutation_values,
            extension_constraints: vec![],
            constraint_types: vec![],
        }
    }

    /// Return the constraint layout mapping global indices to base/ext streams.
    pub fn constraint_layout(&self) -> ConstraintLayout {
        let mut base_indices = Vec::new();
        let mut ext_indices = Vec::new();
        for (idx, kind) in self.constraint_types.iter().enumerate() {
            match kind {
                ConstraintType::Base => base_indices.push(idx),
                ConstraintType::Ext => ext_indices.push(idx),
            }
        }
        ConstraintLayout {
            base_indices,
            ext_indices,
        }
    }

    pub fn extension_constraints(&self) -> Vec<SymbolicExpressionExt<F, EF>> {
        self.extension_constraints.clone()
    }

    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.base_constraints.clone()
    }
}

/// Implement `WindowAccess` for `RowMajorMatrix` treating it as a two-row window
/// (first row = current, second row = next).
///
/// # Panics
///
/// Panics if the matrix does not have exactly 2 rows.
impl<T: Clone + Send + Sync> WindowAccess<T> for RowMajorMatrix<T> {
    fn current_slice(&self) -> &[T] {
        assert_eq!(
            self.height(),
            2,
            "WindowAccess for RowMajorMatrix requires exactly 2 rows, got {}",
            self.height()
        );
        let values: &[T] = self.values.borrow();
        &values[..self.width]
    }

    fn next_slice(&self) -> &[T] {
        assert_eq!(
            self.height(),
            2,
            "WindowAccess for RowMajorMatrix requires exactly 2 rows, got {}",
            self.height()
        );
        let values: &[T] = self.values.borrow();
        &values[self.width..]
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type PreprocessedWindow = RowMajorMatrix<Self::Var>;
    type MainWindow = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<F>;
    type PeriodicVar = SymbolicVariable<F>;

    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpr::Leaf(BaseLeaf::IsFirstRow)
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpr::Leaf(BaseLeaf::IsLastRow)
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpr::Leaf(BaseLeaf::IsTransition)
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Base);
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type EF = EF;
    type ExprEF = SymbolicExpressionExt<F, EF>;
    type VarEF = SymbolicVariableExt<F, EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Ext);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = SymbolicVariableExt<F, EF>;

    type PermutationVar = SymbolicVariableExt<F, EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &self.permutation_values
    }
}

/// Tracks whether a constraint was emitted via `assert_zero` (base) or `assert_zero_ext` (ext).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConstraintType {
    /// Emitted via `assert_zero` from the base trace.
    Base,
    /// Emitted via `assert_zero_ext` from the extension trace.
    Ext,
}

/// Maps between global constraint indices and the separated base/ext streams.
///
/// When alpha powers are pre-computed in global order `[α^{N−1}, …, α⁰]`,
/// the layout tells us which powers correspond to base-field constraints (for
/// `batched_linear_combination`) and which to extension-field constraints.
#[derive(Debug, Default)]
pub struct ConstraintLayout {
    /// Global indices of base-field constraints, in emission order.
    pub base_indices: Vec<usize>,
    /// Global indices of extension-field constraints, in emission order.
    pub ext_indices: Vec<usize>,
}

impl ConstraintLayout {
    /// Total number of constraints (base + extension).
    pub const fn total_constraints(&self) -> usize {
        self.base_indices.len() + self.ext_indices.len()
    }

    /// Decompose `α` into reordered powers for base and extension constraints.
    ///
    /// Returns `(base_alpha_powers, ext_alpha_powers)` where:
    /// - `base_alpha_powers[d][j]` = d-th basis coefficient of the alpha power for
    ///   the j-th base constraint (transposed + reordered for `batched_linear_combination`)
    /// - `ext_alpha_powers[j]` = full EF alpha power for the j-th extension constraint
    ///
    /// Constraints are emitted in one global order and folded into a single random
    /// linear combination using powers of `α`:
    ///
    /// `C_fold(X) = Σ_{i=0..K−1} α^{K−1−i} · Cᵢ(X)`.
    ///
    /// We use descending powers because the verifier evaluates the fold at a single
    /// point via Horner (streaming): `acc = acc·α + Cᵢ`.
    ///
    /// The prover accumulates base-field constraints with packed (SIMD) arithmetic for
    /// throughput, while extension constraints must stay in the extension field. This
    /// method splits the precomputed powers accordingly, and also transposes EF powers
    /// into their base-field coordinates so the base-field path can use
    /// `batched_linear_combination` without repeated cross-field conversions.
    pub fn decompose_alpha<F: Field, EF: ExtensionField<F>>(
        &self,
        alpha: EF,
    ) -> (Vec<Vec<F>>, Vec<EF>) {
        let total = self.total_constraints();

        // alpha_powers[i] = α^{total − 1 − i}, so constraint i gets
        // weight α^{total − 1 − i} in the linear combination.
        let mut alpha_powers = alpha.powers().collect_n(total);
        alpha_powers.reverse();

        // Base: transpose EF -> [F; D] and reorder by base_indices in one pass
        let base_alpha_powers = (0..EF::DIMENSION)
            .map(|d| {
                self.base_indices
                    .iter()
                    .map(|&idx| alpha_powers[idx].as_basis_coefficients_slice()[d])
                    .collect()
            })
            .collect();

        // Ext: pick full EF powers by ext_indices
        let ext_alpha_powers = self
            .ext_indices
            .iter()
            .map(|&idx| alpha_powers[idx])
            .collect();

        (base_alpha_powers, ext_alpha_powers)
    }
}

/// Evaluate the AIR symbolically and return the constraint layout.
///
/// This runs `air.eval()` on a [`SymbolicAirBuilder`] to discover which constraints
/// are base-field vs extension-field, and their global ordering. The layout is used
/// by the prover to reorder decomposed alpha powers for efficient accumulation.
///
/// Most builder dimensions are derived from the AIR trait methods. `num_public_values`
/// is passed explicitly because `BaseAirWithPublicValues::num_public_values` defaults
/// to 0 and many AIRs do not override it.
#[instrument(name = "compute constraint layout", skip_all, level = "debug")]
pub fn get_constraint_layout<F, EF, A>(air: &A, layout: AirLayout) -> ConstraintLayout
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
    SymbolicExpression<EF>: Algebra<SymbolicExpression<F>>,
{
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    builder.constraint_layout()
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    use super::*;
    use crate::BaseAir;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[derive(Debug)]
    struct MockAir {
        constraints: Vec<SymbolicVariable<F>>,
        width: usize,
    }

    impl BaseAir<F> for MockAir {
        fn width(&self) -> usize {
            self.width
        }
        fn num_periodic_columns(&self) -> usize {
            self.periodic_columns().len()
        }
    }

    impl Air<SymbolicAirBuilder<F>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F>) {
            for constraint in &self.constraints {
                builder.assert_zero(*constraint);
            }
        }
    }

    const fn layout(
        preprocessed_width: usize,
        main_width: usize,
        num_public_values: usize,
        num_periodic_columns: usize,
    ) -> AirLayout {
        AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width: 0,
            num_permutation_challenges: 0,
            num_permutation_values: 0,
            num_periodic_columns,
        }
    }

    const fn layout_with_perm(
        preprocessed_width: usize,
        main_width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_permutation_challenges: usize,
        num_periodic_columns: usize,
    ) -> AirLayout {
        AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width,
            num_permutation_challenges,
            num_permutation_values: 0,
            num_periodic_columns,
        }
    }

    #[test]
    fn test_get_max_constraint_degree_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let l = layout(3, air.width, air.num_public_values(), 0);
        let max_degree = get_max_constraint_degree(&air, l);
        assert_eq!(
            max_degree, 0,
            "No constraints should result in a degree of 0"
        );
    }

    #[test]
    fn test_get_max_constraint_degree_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0),
                SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1),
                SymbolicVariable::new(BaseEntry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let l = layout(3, air.width, air.num_public_values(), 0);
        let max_degree = get_max_constraint_degree(&air, l);
        assert_eq!(max_degree, 1, "Max constraint degree should be 1");
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1 = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0);
        let c2 = SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1);

        let air = MockAir {
            constraints: vec![c1, c2],
            width: 4,
        };

        let l = layout(3, air.width, air.num_public_values(), 0);
        let constraints = get_symbolic_constraints(&air, l);

        assert_eq!(constraints.len(), 2, "Should return exactly 2 constraints");

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Leaf(BaseLeaf::Variable(v)) if v.index == c1.index && v.entry == c1.entry)),
            "Expected constraint {c1:?} was not found"
        );

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Leaf(BaseLeaf::Variable(v)) if v.index == c2.index && v.entry == c2.entry)),
            "Expected constraint {c2:?} was not found"
        );
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        let builder = SymbolicAirBuilder::<F>::new(layout(2, 4, 3, 0));

        let expected_main = [
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 1),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 2),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 3),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 0),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 1),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 2),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 3),
        ];

        let builder_main = builder.main.values;

        assert_eq!(
            builder_main.len(),
            expected_main.len(),
            "Main matrix should have the expected length"
        );

        for (expected, actual) in expected_main.iter().zip(builder_main.iter()) {
            assert_eq!(expected.index, actual.index, "Index mismatch");
            assert_eq!(expected.entry, actual.entry, "Entry mismatch");
        }
    }

    #[test]
    fn test_symbolic_air_builder_is_first_last_row() {
        let builder = SymbolicAirBuilder::<F>::new(layout(2, 4, 3, 0));

        assert!(
            matches!(
                builder.is_first_row(),
                SymbolicExpression::Leaf(BaseLeaf::IsFirstRow)
            ),
            "First row condition did not match"
        );

        assert!(
            matches!(
                builder.is_last_row(),
                SymbolicExpression::Leaf(BaseLeaf::IsLastRow)
            ),
            "Last row condition did not match"
        );
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<F>::new(layout(2, 4, 3, 0));
        let expr = SymbolicExpression::Leaf(BaseLeaf::Constant(F::new(5)));
        builder.assert_zero(expr);

        let constraints = builder.base_constraints();
        assert_eq!(constraints.len(), 1, "One constraint should be recorded");

        assert!(
            constraints.iter().any(
                |x| matches!(x, SymbolicExpression::Leaf(BaseLeaf::Constant(val)) if *val == F::new(5))
            ),
            "Constraint should match the asserted one"
        );
    }

    #[test]
    fn test_is_transition_window_size_2() {
        // Window size 2 returns the transition selector.
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 2, 0, 0));
        let expr = builder.is_transition_window(2);
        assert!(matches!(
            expr,
            SymbolicExpression::Leaf(BaseLeaf::IsTransition)
        ));
    }

    #[test]
    #[should_panic(expected = "uni-stark only supports a window size of 2")]
    fn test_is_transition_window_size_3_panics() {
        // Window size 3 is not supported and should panic.
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 2, 0, 0));
        let _ = builder.is_transition_window(3);
    }

    #[test]
    fn test_main_returns_correct_dimensions() {
        // The main matrix has 2 rows (one per offset) and the given width.
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 3, 0, 0));
        let main = builder.main();

        // 2 rows times 3 columns gives 6 entries.
        assert_eq!(main.values.len(), 6);

        // First row has offset 0.
        assert_eq!(main.values[0].entry, BaseEntry::Main { offset: 0 });
        assert_eq!(main.values[0].index, 0);
        assert_eq!(main.values[2].index, 2);

        // Second row has offset 1.
        assert_eq!(main.values[3].entry, BaseEntry::Main { offset: 1 });
        assert_eq!(main.values[3].index, 0);
    }

    #[test]
    fn test_preprocessed_returns_correct_dimensions() {
        // The preprocessed matrix has 2 rows and the given preprocessed width.
        let builder = SymbolicAirBuilder::<F>::new(layout(2, 3, 0, 0));
        let prep = builder.preprocessed();

        // 2 rows times 2 columns gives 4 entries.
        assert_eq!(prep.values.len(), 4);
        assert_eq!(prep.values[0].entry, BaseEntry::Preprocessed { offset: 0 });
        assert_eq!(prep.values[0].index, 0);
        assert_eq!(prep.values[1].index, 1);
        assert_eq!(prep.values[2].entry, BaseEntry::Preprocessed { offset: 1 });
    }

    #[test]
    fn test_preprocessed_returns_none_when_width_is_zero() {
        // A builder with zero preprocessed columns should report no preprocessed trace.
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 3, 0, 0));
        assert_eq!(builder.preprocessed().width, 0);
    }

    #[test]
    fn test_public_values_correct_count_and_entries() {
        // All public value variables have the public entry kind.
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 2, 5, 0));
        let pv = builder.public_values();
        assert_eq!(pv.len(), 5);
        for (i, var) in pv.iter().enumerate() {
            assert_eq!(var.entry, BaseEntry::Public);
            assert_eq!(var.index, i);
        }
    }

    #[test]
    fn test_assert_zero_ext_records_constraint() {
        // Asserting an extension constraint records it in the builder.
        let mut builder = SymbolicAirBuilder::<F, EF>::new(layout_with_perm(0, 2, 0, 2, 1, 0));
        let expr = SymbolicExpressionExt::<F, EF>::from(F::new(7));
        builder.assert_zero_ext(expr);
        let ext_constraints = builder.extension_constraints();
        assert_eq!(ext_constraints.len(), 1);
    }

    #[test]
    fn test_assert_zeros_ext_batches_extension_constraints() {
        let mut builder = SymbolicAirBuilder::<F, EF>::new(layout_with_perm(0, 2, 0, 2, 1, 0));
        let a = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let b = SymbolicExpressionExt::<F, EF>::from(F::new(4));
        builder.assert_zeros_ext([a, b]);

        assert_eq!(builder.extension_constraints().len(), 2);
        assert_eq!(builder.constraint_layout().ext_indices.len(), 2);
    }

    #[test]
    fn test_extension_constraints_initially_empty() {
        // A fresh builder starts with no extension constraints.
        let builder = SymbolicAirBuilder::<F, EF>::new(layout(0, 2, 0, 0));
        assert!(builder.extension_constraints().is_empty());
    }

    #[test]
    fn test_permutation_returns_correct_dimensions() {
        // The permutation matrix has 2 rows and the given permutation width.
        let builder = SymbolicAirBuilder::<F, EF>::new(layout_with_perm(0, 2, 0, 3, 0, 0));
        let perm = builder.permutation();

        // 2 rows times 3 columns gives 6 entries.
        assert_eq!(perm.values.len(), 6);
        assert_eq!(perm.values[0].entry, ExtEntry::Permutation { offset: 0 });
        assert_eq!(perm.values[0].index, 0);
        assert_eq!(perm.values[3].entry, ExtEntry::Permutation { offset: 1 });
    }

    #[test]
    fn test_permutation_randomness_correct_count() {
        // All challenge variables have the challenge entry kind.
        let builder = SymbolicAirBuilder::<F, EF>::new(layout_with_perm(0, 2, 0, 2, 4, 0));
        let challenges = builder.permutation_randomness();
        assert_eq!(challenges.len(), 4);
        for (i, var) in challenges.iter().enumerate() {
            assert_eq!(var.entry, ExtEntry::Challenge);
            assert_eq!(var.index, i);
        }
    }

    #[derive(Debug)]
    struct ExtMockAir {
        width: usize,
    }

    impl BaseAir<F> for ExtMockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl Air<SymbolicAirBuilder<F, EF>> for ExtMockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F, EF>) {
            // Record one base constraint from the main trace.
            let main = builder.main();
            builder.assert_zero(main.values[0]);

            // Record one extension constraint from the permutation trace.
            let perm = builder.permutation();
            builder.assert_zero_ext(perm.values[0]);
        }
    }

    #[test]
    fn test_get_symbolic_constraints_extension() {
        // Only the extension constraint is returned.
        let air = ExtMockAir { width: 2 };
        let l = layout_with_perm(0, air.width, air.num_public_values(), 3, 1, 0);
        let ext_constraints = get_symbolic_constraints_extension::<F, EF, _>(&air, l);
        assert_eq!(ext_constraints.len(), 1);
    }

    #[test]
    fn test_get_all_symbolic_constraints() {
        // Both the base and extension constraint are returned.
        let air = ExtMockAir { width: 2 };
        let l = layout_with_perm(0, air.width, air.num_public_values(), 3, 1, 0);
        let (base, ext) = get_all_symbolic_constraints::<F, EF, _>(&air, l);
        assert_eq!(base.len(), 1);
        assert_eq!(ext.len(), 1);
    }

    #[test]
    fn test_get_max_constraint_degree_extension() {
        // The max degree covers both base and extension constraints.
        let air = ExtMockAir { width: 2 };
        let l = layout_with_perm(0, air.width, air.num_public_values(), 3, 1, 0);
        let max_deg = get_max_constraint_degree_extension::<F, EF, _>(&air, l);

        // Both constraints are single variables with degree 1.
        assert_eq!(max_deg, 1);
    }

    #[test]
    fn test_total_constraints_empty_layout() {
        // An empty layout has zero base and zero extension constraints.
        let layout = ConstraintLayout::default();

        // Total should be 0.
        assert_eq!(layout.total_constraints(), 0);
    }

    #[test]
    fn test_total_constraints_base_only() {
        // Layout with 3 base constraints and no extension constraints.
        let layout = ConstraintLayout {
            base_indices: vec![0, 1, 2],
            ext_indices: vec![],
        };

        // Only base constraints contribute: 3 + 0 = 3.
        assert_eq!(layout.total_constraints(), 3);
    }

    #[test]
    fn test_total_constraints_ext_only() {
        // Layout with no base constraints and 2 extension constraints.
        let layout = ConstraintLayout {
            base_indices: vec![],
            ext_indices: vec![0, 1],
        };

        // Only extension constraints contribute: 0 + 2 = 2.
        assert_eq!(layout.total_constraints(), 2);
    }

    #[test]
    fn test_total_constraints_mixed() {
        // Layout with both types interleaved.
        // Global order: base(0), ext(1), base(2), ext(3), base(4).
        let layout = ConstraintLayout {
            base_indices: vec![0, 2, 4],
            ext_indices: vec![1, 3],
        };

        // Total is the sum of both vectors: 3 + 2 = 5.
        assert_eq!(layout.total_constraints(), 5);
    }

    #[test]
    fn test_decompose_alpha_empty_layout() {
        // No constraints at all.
        let layout = ConstraintLayout::default();
        let alpha = EF::TWO;

        // Decompose with zero constraints.
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        // Base output has D columns (one per basis dimension), all empty.
        assert_eq!(base.len(), <EF as BasedVectorSpace<F>>::DIMENSION);
        for col in &base {
            assert!(col.is_empty());
        }

        // Extension output is empty too.
        assert!(ext.is_empty());
    }

    #[test]
    fn test_decompose_alpha_single_base_constraint() {
        // One base constraint at global index 0.
        let layout = ConstraintLayout {
            base_indices: vec![0],
            ext_indices: vec![],
        };
        let alpha = EF::TWO;

        // Only 1 constraint total.
        // Constraint 0 gets alpha^{1 - 1 - 0} = alpha^0 = 1.
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        // The expected power is EF::ONE (the identity element).
        let expected_coeffs = EF::ONE.as_basis_coefficients_slice();

        // Each basis dimension column has exactly one entry matching 1.
        for (d, col) in base.iter().enumerate() {
            assert_eq!(col.len(), 1);
            assert_eq!(col[0], expected_coeffs[d]);
        }

        // No extension constraints.
        assert!(ext.is_empty());
    }

    #[test]
    fn test_decompose_alpha_single_ext_constraint() {
        // One extension constraint at global index 0.
        let layout = ConstraintLayout {
            base_indices: vec![],
            ext_indices: vec![0],
        };
        let alpha = EF::TWO;

        // Only 1 constraint total.
        // Constraint 0 gets alpha^{1 - 1 - 0} = alpha^0 = 1.
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        // Base columns exist but are empty (no base constraints).
        for col in &base {
            assert!(col.is_empty());
        }

        // The single extension power should be 1.
        assert_eq!(ext, vec![EF::ONE]);
    }

    #[test]
    fn test_decompose_alpha_two_base_constraints() {
        // Two base constraints at global indices 0 and 1.
        let layout = ConstraintLayout {
            base_indices: vec![0, 1],
            ext_indices: vec![],
        };
        let alpha = EF::TWO;

        // 2 constraints total. Descending powers:
        //   constraint 0 -> alpha^{2 - 1 - 0} = alpha^1 = alpha
        //   constraint 1 -> alpha^{2 - 1 - 1} = alpha^0 = 1
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        // Expected power for each base constraint.
        let power_0 = alpha;
        let power_1 = EF::ONE;

        // Decompose each power into D base-field coefficients.
        let coeffs_0 = power_0.as_basis_coefficients_slice();
        let coeffs_1 = power_1.as_basis_coefficients_slice();

        // Each column has 2 entries in emission order (constraint 0 first, then 1).
        for (d, col) in base.iter().enumerate() {
            assert_eq!(col.len(), 2);
            assert_eq!(col[0], coeffs_0[d], "mismatch at d={d} for constraint 0");
            assert_eq!(col[1], coeffs_1[d], "mismatch at d={d} for constraint 1");
        }

        // No extension constraints.
        assert!(ext.is_empty());
    }

    #[test]
    fn test_decompose_alpha_interleaved_base_and_ext() {
        // Three constraints in global order: base(0), ext(1), base(2).
        let layout = ConstraintLayout {
            base_indices: vec![0, 2],
            ext_indices: vec![1],
        };
        let alpha = EF::TWO;

        // 3 constraints total. Descending powers:
        //   constraint 0 -> alpha^{3 - 1 - 0} = alpha^2
        //   constraint 1 -> alpha^{3 - 1 - 1} = alpha^1
        //   constraint 2 -> alpha^{3 - 1 - 2} = alpha^0
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        // Compute the expected powers.
        let alpha_sq = alpha * alpha;
        let power_base_0 = alpha_sq; // constraint 0 -> alpha^2
        let power_base_2 = EF::ONE; // constraint 2 -> alpha^0
        let power_ext_1 = alpha; // constraint 1 -> alpha^1

        // Base columns: constraint 0 first, then constraint 2.
        let coeffs_0 = power_base_0.as_basis_coefficients_slice();
        let coeffs_2 = power_base_2.as_basis_coefficients_slice();
        for (d, col) in base.iter().enumerate() {
            assert_eq!(col.len(), 2);
            assert_eq!(col[0], coeffs_0[d]);
            assert_eq!(col[1], coeffs_2[d]);
        }

        // Extension vector: only constraint 1.
        assert_eq!(ext, vec![power_ext_1]);
    }

    #[test]
    fn test_decompose_alpha_all_ext_constraints() {
        // Four extension constraints and no base constraints.
        let layout = ConstraintLayout {
            base_indices: vec![],
            ext_indices: vec![0, 1, 2, 3],
        };
        let alpha = EF::TWO;

        // 4 constraints total. Descending powers:
        //   constraint 0 -> alpha^3
        //   constraint 1 -> alpha^2
        //   constraint 2 -> alpha^1
        //   constraint 3 -> alpha^0
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        // Base columns exist but are all empty.
        for col in &base {
            assert!(col.is_empty());
        }

        // Extension powers in emission order (descending).
        let a2 = alpha * alpha;
        let a3 = a2 * alpha;
        assert_eq!(ext, vec![a3, a2, alpha, EF::ONE]);
    }

    #[test]
    fn test_decompose_alpha_dimension_count() {
        // Even with a single constraint, the base output has D columns.
        let layout = ConstraintLayout {
            base_indices: vec![0],
            ext_indices: vec![],
        };

        let (base, _) = layout.decompose_alpha::<F, EF>(EF::TWO);

        // D = 4 for BinomialExtensionField<BabyBear, 4>.
        assert_eq!(base.len(), <EF as BasedVectorSpace<F>>::DIMENSION);
    }

    #[test]
    fn test_decompose_alpha_identity_element() {
        // When alpha = 1, every power of alpha is 1.
        let layout = ConstraintLayout {
            base_indices: vec![0, 2],
            ext_indices: vec![1],
        };

        let (base, ext) = layout.decompose_alpha::<F, EF>(EF::ONE);

        // All base powers decompose to the coefficients of 1.
        let one_coeffs = EF::ONE.as_basis_coefficients_slice();
        for (d, col) in base.iter().enumerate() {
            for &val in col {
                assert_eq!(val, one_coeffs[d]);
            }
        }

        // All extension powers are 1.
        for &val in &ext {
            assert_eq!(val, EF::ONE);
        }
    }
}
