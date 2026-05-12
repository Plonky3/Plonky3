//! Symbolic builder that records constraints plus bus interactions.

use alloc::string::String;
use alloc::vec::Vec;

use p3_air::symbolic::{
    AirLayout, ConstraintLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
    SymbolicVariable, SymbolicVariableExt,
};
use p3_air::{AirBuilder, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

use crate::builder::{InteractionBuilder, SymbolicInteraction, SymbolicLocalInteraction};

/// Symbolic builder that captures constraints and bus interactions side by side.
#[derive(Debug)]
pub struct InteractionSymbolicBuilder<F: Field, EF: ExtensionField<F> = F> {
    /// Wrapped constraint-only builder. All non-interaction methods forward here.
    inner: SymbolicAirBuilder<F, EF>,
    /// Cross-AIR messages pushed so far, in emission order.
    global_interactions: Vec<SymbolicInteraction<F>>,
    /// Intra-AIR lookups pushed so far, in emission order.
    local_interactions: Vec<SymbolicLocalInteraction<F>>,
}

impl<F: Field, EF: ExtensionField<F>> InteractionSymbolicBuilder<F, EF> {
    /// Create an empty builder from a column / challenge layout.
    pub fn new(layout: AirLayout) -> Self {
        Self {
            inner: SymbolicAirBuilder::new(layout),
            global_interactions: Vec::new(),
            local_interactions: Vec::new(),
        }
    }

    /// Cross-AIR interactions recorded so far, in the order they were pushed.
    pub fn global_interactions(&self) -> &[SymbolicInteraction<F>] {
        &self.global_interactions
    }

    /// Intra-AIR interactions recorded so far, in the order they were pushed.
    pub fn local_interactions(&self) -> &[SymbolicLocalInteraction<F>] {
        &self.local_interactions
    }

    /// Symbolic base-field constraints captured by the inner builder.
    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.inner.base_constraints()
    }

    /// Symbolic extension-field constraints captured by the inner builder.
    pub fn extension_constraints(&self) -> Vec<SymbolicExpressionExt<F, EF>>
    where
        SymbolicExpressionExt<F, EF>: Algebra<EF>,
    {
        self.inner.extension_constraints()
    }

    /// Mapping from global constraint indices into separated base / extension streams.
    pub fn constraint_layout(&self) -> ConstraintLayout {
        self.inner.constraint_layout()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for InteractionSymbolicBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type PreprocessedWindow = RowMajorMatrix<Self::Var>;
    type MainWindow = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<F>;
    type PeriodicVar = SymbolicVariable<F>;

    fn main(&self) -> Self::MainWindow {
        self.inner.main()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        self.inner.preprocessed()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(x);
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values()
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for InteractionSymbolicBuilder<F, EF>
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
        self.inner.assert_zero_ext(x);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for InteractionSymbolicBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = RowMajorMatrix<Self::VarEF>;
    type RandomVar = SymbolicVariableExt<F, EF>;
    type PermutationVar = SymbolicVariableExt<F, EF>;

    fn permutation(&self) -> Self::MP {
        self.inner.permutation()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.inner.permutation_randomness()
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        self.inner.permutation_values()
    }
}

impl<F: Field, EF: ExtensionField<F>> InteractionBuilder for InteractionSymbolicBuilder<F, EF> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        count: impl Into<Self::Expr>,
        count_weight: u32,
    ) {
        // Materialize lazy inputs into owned expressions and append one record.
        //
        // - Collected eagerly so the record outlives the caller's iterator.
        // - Push order is preserved: it drives auxiliary-column assignment.
        self.global_interactions.push(SymbolicInteraction {
            bus_name: String::from(bus_name),
            fields: fields.into_iter().map(Into::into).collect(),
            count: count.into(),
            count_weight,
        });
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Self::Expr)>,
    ) {
        // One call → one grouped record, not N separate ones.
        //
        // - A local lookup folds into a single running sum.
        // - All `(fields, count)` pairs must stay paired for the folder.
        self.local_interactions.push(SymbolicLocalInteraction {
            tuples: tuples.into_iter().collect(),
        });
    }

    fn num_global_interactions(&self) -> usize {
        self.global_interactions.len()
    }

    fn num_local_interactions(&self) -> usize {
        self.local_interactions.len()
    }
}
