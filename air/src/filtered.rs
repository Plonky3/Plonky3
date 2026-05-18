use p3_field::Dup;

use crate::builder::{AirBuilder, AirBuilderWithContext, ExtensionBuilder, PermutationAirBuilder};

/// A wrapper around an [`AirBuilder`] that enforces constraints only when a specified condition is met.
///
/// This struct allows selectively applying constraints to certain rows or under certain conditions in the AIR,
/// without modifying the underlying logic. All constraints asserted through this filtered builder will be
/// multiplied by the given `condition`, effectively disabling them when `condition` evaluates to zero.
#[derive(Debug)]
pub struct FilteredAirBuilder<'a, AB: AirBuilder> {
    /// Reference to the underlying inner [`AirBuilder`] where constraints are ultimately recorded.
    pub inner: &'a mut AB,

    /// Condition expression that controls when the constraints are enforced.
    ///
    /// If `condition` evaluates to zero, constraints asserted through this builder have no effect.
    pub(crate) condition: AB::Expr,
}

impl<AB: AirBuilder> FilteredAirBuilder<'_, AB> {
    pub fn condition(&self) -> AB::Expr {
        self.condition.dup()
    }
}

impl<AB: AirBuilder> AirBuilder for FilteredAirBuilder<'_, AB> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type PreprocessedWindow = AB::PreprocessedWindow;
    type MainWindow = AB::MainWindow;
    type PublicVar = AB::PublicVar;
    type PeriodicVar = AB::PeriodicVar;

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

    fn is_transition(&self) -> Self::Expr {
        self.inner.is_transition()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(self.condition() * x.into());
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values()
    }
}

impl<AB: ExtensionBuilder> ExtensionBuilder for FilteredAirBuilder<'_, AB> {
    type EF = AB::EF;
    type ExprEF = AB::ExprEF;
    type VarEF = AB::VarEF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let ext_x: Self::ExprEF = x.into();
        let condition: AB::Expr = self.condition();

        self.inner.assert_zero_ext(ext_x * condition);
    }
}

impl<AB: PermutationAirBuilder> PermutationAirBuilder for FilteredAirBuilder<'_, AB> {
    type MP = AB::MP;
    type RandomVar = AB::RandomVar;
    type PermutationVar = AB::PermutationVar;

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

impl<AB: AirBuilderWithContext> AirBuilderWithContext for FilteredAirBuilder<'_, AB> {
    type EvalContext = AB::EvalContext;

    fn eval_context(&self) -> &Self::EvalContext {
        self.inner.eval_context()
    }
}
