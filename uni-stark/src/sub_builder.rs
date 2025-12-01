//! Helpers for reusing an [`AirBuilder`] on a restricted set of trace columns.
//!
//! The uni-STARK builders often need to enforce constraints that refer to only a slice of the main
//! trace. [`HorizontallyTruncated`] offers a cheap view over a subset of columns, and
//! [`SubAirBuilder`] wires that view into any [`AirBuilder`] implementation so a sub-air can be
//! evaluated independently without cloning trace data.

// Code inpsired from SP1 with additional modifications:
// https://github.com/succinctlabs/sp1/blob/main/crates/stark/src/air/sub_builder.rs

use core::ops::Range;

use p3_air::{AirBuilder, BaseAir};
use p3_matrix::horizontally_truncated::HorizontallyTruncated;

/// Evaluates a sub-AIR against a restricted slice of the parent trace.
///
/// This is useful whenever a standalone component AIR is embedded in a larger system but only owns
/// a few columns. `SubAirBuilder` reuses the parent builder for bookkeeping so witness generation
/// and constraint enforcement stay in sync.
pub struct SubAirBuilder<'a, AB: AirBuilder, SubAir: BaseAir<AB::F>, T> {
    /// Mutable reference to the parent builder.
    inner: &'a mut AB,

    /// Column range (in the parent trace) that the sub-AIR is allowed to see.
    column_range: Range<usize>,

    /// Marker for the sub-AIR and witness type.
    _phantom: core::marker::PhantomData<(SubAir, T)>,
}

impl<'a, AB: AirBuilder, SubAir: BaseAir<AB::F>, T> SubAirBuilder<'a, AB, SubAir, T> {
    /// Create a new [`SubAirBuilder`] exposing only `column_range` to the sub-AIR.
    ///
    /// The range must lie entirely inside the parent trace width.
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
    type M = HorizontallyTruncated<Self::Var, AB::M>;

    fn main(&self) -> Self::M {
        let matrix = self.inner.main();

        HorizontallyTruncated::new_with_range(matrix, self.column_range.clone())
            .expect("sub-air column range exceeds parent width")
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
