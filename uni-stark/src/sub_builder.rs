//! Helpers for reusing an `AirBuilder` on a restricted set of trace columns.
//!
//! The uni-STARK builders often need to enforce constraints that refer to only a slice of the main
//! trace.  `SubMatrixRowSlices` offers a cheap view over a subset of columns, and `SubAirBuilder`
//! wires that view into any `AirBuilder` implementation so a sub-air can be evaluated independently
//! without cloning trace data.

// Code from SP1 with minor modifications:
// https://github.com/succinctlabs/sp1/blob/main/crates/stark/src/air/sub_builder.rs

use alloc::vec::Vec;
use core::ops::{Deref, Range};

use p3_air::{AirBuilder, BaseAir};
use p3_matrix::Matrix;

/// A light-weight view over a contiguous subset of the columns of `inner`.
///
/// The rows are shared with the original matrix, but iteration stops at the configured
/// `column_range`, allowing downstream code to treat the projection as a stand-alone submatrix
/// of the larger original one.
pub struct SubMatrixRowSlices<M: Matrix<T>, T: Send + Sync + Clone> {
    inner: M,
    column_range: Range<usize>,
    _phantom: core::marker::PhantomData<T>,
}

impl<M: Matrix<T>, T: Send + Sync + Clone> SubMatrixRowSlices<M, T> {
    /// Creates a new [`SubMatrixRowSlices`].
    #[must_use]
    pub const fn new(inner: M, column_range: Range<usize>) -> Self {
        Self {
            inner,
            column_range,
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Implement `Matrix` for `SubMatrixRowSlices`.
impl<M: Matrix<T>, T: Send + Sync + Clone> Matrix<T> for SubMatrixRowSlices<M, T> {
    #[inline]
    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        self.inner.row(r).map(|row| {
            row.into_iter()
                .take(self.column_range.end)
                .skip(self.column_range.start)
        })
    }

    #[inline]
    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [T]>> {
        self.row(r)?.into_iter().collect::<Vec<_>>().into()
    }

    #[inline]
    fn width(&self) -> usize {
        self.column_range.len()
    }

    #[inline]
    fn height(&self) -> usize {
        self.inner.height()
    }
}

/// Evaluates a sub-AIR against a restricted slice of the parent trace.
///
/// This is useful whenever a standalone component AIR is embedded in a larger system but only owns
/// a few columns. `SubAirBuilder` reuses the parent builder for bookkeeping so witness generation
/// and constraint enforcement stay in sync.
pub struct SubAirBuilder<'a, AB: AirBuilder, SubAir: BaseAir<AB::F>, T> {
    inner: &'a mut AB,
    column_range: Range<usize>,
    _phantom: core::marker::PhantomData<(SubAir, T)>,
}

impl<'a, AB: AirBuilder, SubAir: BaseAir<AB::F>, T> SubAirBuilder<'a, AB, SubAir, T> {
    /// Creates a new [`SubAirBuilder`] that only exposes the specified `column_range` to the evaluated sub-air.
    #[must_use]
    pub const fn new(inner: &'a mut AB, column_range: Range<usize>) -> Self {
        Self {
            inner,
            column_range,
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Implements `AirBuilder` for `SubAirBuilder`.
impl<AB: AirBuilder, SubAir: BaseAir<AB::F>, F> AirBuilder for SubAirBuilder<'_, AB, SubAir, F> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = SubMatrixRowSlices<AB::M, Self::Var>;

    fn main(&self) -> Self::M {
        let matrix = self.inner.main();

        SubMatrixRowSlices::new(matrix, self.column_range.clone())
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
        self.inner.assert_zero(x.into());
    }
}
