//! Symbolic capability boundary for the reference current-row AIR protocol.

use alloc::vec::Vec;

use p3_air::{
    AirBuilder, AirLayout, ExtensionBuilder, SymbolicAirBuilder, SymbolicExpression,
    SymbolicExpressionExt, SymbolicVariable, SymbolicVariableExt,
};
use p3_matrix::dense::RowMajorMatrix;

use super::Val;

/// Symbolic builder accepted by [`super::ReferenceZkAir::compile`].
///
/// Ordinary [`AirBuilder`] methods record expressions for validation by the compiler.
/// [`ExtensionBuilder`] records explicit extension constraints so compilation can reject them, including constraints equal to zero.
///
/// This type deliberately does not implement permutation, runtime context or interaction builder traits.
/// Its symbolic implementation is private, with no dereference or inner-builder escape.
/// Generic current-row AIRs bounded only by `AirBuilder<F = Val>` remain usable.
#[derive(Debug)]
pub struct CurrentRowAirBuilder {
    inner: SymbolicAirBuilder<Val>,
}

impl CurrentRowAirBuilder {
    pub(super) fn new(layout: AirLayout) -> Self {
        Self {
            inner: SymbolicAirBuilder::new(layout),
        }
    }

    pub(super) fn base_constraints(&self) -> Vec<SymbolicExpression<Val>> {
        self.inner.base_constraints()
    }

    pub(super) fn has_extension_constraints(&self) -> bool {
        !self.inner.extension_constraints().is_empty()
    }
}

impl AirBuilder for CurrentRowAirBuilder {
    type F = Val;
    type Expr = SymbolicExpression<Val>;
    type Var = SymbolicVariable<Val>;
    type PreprocessedWindow = RowMajorMatrix<Self::Var>;
    type MainWindow = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<Val>;
    type PeriodicVar = SymbolicVariable<Val>;

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

impl ExtensionBuilder for CurrentRowAirBuilder {
    type EF = Val;
    type ExprEF = SymbolicExpressionExt<Val, Val>;
    type VarEF = SymbolicVariableExt<Val, Val>;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, x: I) {
        self.inner.assert_zero_ext(x);
    }
}
