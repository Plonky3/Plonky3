//! Helpers for restricting a builder to a subset of trace columns.
//!
//! The uni-STARK builders often need to enforce constraints that refer to only a slice of the main
//! trace. [`SubSliced`] offers a cheap view over a subset of columns, and [`SubAirBuilder`] wires
//! that view into any [`AirBuilder`] implementation so a sub-air can be evaluated independently
//! without copying trace data.

// Code inpsired from SP1 with additional modifications:
// https://github.com/succinctlabs/sp1/blob/main/crates/stark/src/air/sub_builder.rs

use core::marker::PhantomData;
use core::ops::Range;

use p3_air::{AirBuilder, BaseAir, WindowAccess};

/// A column-restricted view over a trace window.
///
/// Wraps an inner window and exposes only the columns within
/// the given range. Lets a sub-AIR see a contiguous subset
/// of the parent trace without copying data.
pub struct SubSliced<W, T> {
    window: W,
    range: Range<usize>,
    _marker: PhantomData<T>,
}

impl<W: WindowAccess<T>, T> WindowAccess<T> for SubSliced<W, T> {
    #[inline]
    fn local(&self) -> &[T] {
        &self.window.local()[self.range.clone()]
    }

    #[inline]
    fn next(&self) -> &[T] {
        &self.window.next()[self.range.clone()]
    }
}

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
    type M = SubSliced<AB::M, AB::Var>;
    type PublicVar = AB::PublicVar;

    fn main(&self) -> Self::M {
        SubSliced {
            window: self.inner.main(),
            range: self.column_range.clone(),
            _marker: PhantomData,
        }
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
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
        self.inner.assert_zero(x.into());
    }
}
