use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use super::{
    AirLayout, ConstraintLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
    constraint_degree_from_poly_degree,
};

/// Owned symbolic constraints with a mapping back to global assertion order.
///
/// Both streams retain emission order, including duplicate and zero constraints.
/// [`Self::constraint_layout`] records how the two streams were interleaved.
/// Its base and extension index counts always match the corresponding stream lengths.
///
/// The streams cannot be mutated independently of their layout:
///
/// ```compile_fail,E0616
/// use p3_air::SymbolicConstraints;
/// use p3_baby_bear::BabyBear;
///
/// fn clear_constraints(mut captured: SymbolicConstraints<BabyBear>) {
///     captured.base_constraints.clear();
/// }
/// ```
#[derive(Debug)]
pub struct SymbolicConstraints<F: Field, EF: ExtensionField<F> = F> {
    /// Constraints emitted through [`crate::AirBuilder::assert_zero`].
    pub(super) base_constraints: Vec<SymbolicExpression<F>>,
    /// Constraints emitted through [`crate::ExtensionBuilder::assert_zero_ext`].
    pub(super) extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
    /// Global assertion indices for the base and extension streams.
    pub(super) constraint_layout: ConstraintLayout,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicConstraints<F, EF> {
    /// Construct a result after the caller has validated stream lengths and index order.
    pub(crate) const fn from_parts(
        base_constraints: Vec<SymbolicExpression<F>>,
        extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
        constraint_layout: ConstraintLayout,
    ) -> Self {
        Self {
            base_constraints,
            extension_constraints,
            constraint_layout,
        }
    }

    /// Constraints emitted through [`crate::AirBuilder::assert_zero`], in emission order.
    pub fn base_constraints(&self) -> &[SymbolicExpression<F>] {
        &self.base_constraints
    }

    /// Constraints emitted through [`crate::ExtensionBuilder::assert_zero_ext`], in emission order.
    pub fn extension_constraints(&self) -> &[SymbolicExpressionExt<F, EF>] {
        &self.extension_constraints
    }

    /// Global assertion indices for the base and extension streams.
    pub const fn constraint_layout(&self) -> &ConstraintLayout {
        &self.constraint_layout
    }

    /// Consume the result, returning the base constraints, extension constraints, and layout.
    ///
    /// The expression vectors and layout are moved without cloning.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<SymbolicExpression<F>>,
        Vec<SymbolicExpressionExt<F, EF>>,
        ConstraintLayout,
    ) {
        (
            self.base_constraints,
            self.extension_constraints,
            self.constraint_layout,
        )
    }
}

/// Number of assertions in each emission stream.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstraintCounts {
    /// Number of base-field assertions, including zero and duplicate assertions.
    pub base: usize,
    /// Number of extension-field assertions, including zero and duplicate assertions.
    pub extension: usize,
}

/// Conservative maximum degrees for the two assertion streams.
///
/// The method returning this value determines the degree units.
/// An empty stream has degree zero.
/// Bounds do not account for algebraic cancellation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstraintDegrees {
    /// Maximum degree of a base-field assertion.
    pub base: usize,
    /// Maximum degree of an extension-field assertion.
    pub extension: usize,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicConstraints<F, EF> {
    /// Count each stream without traversing expressions.
    pub const fn constraint_counts(&self) -> ConstraintCounts {
        ConstraintCounts {
            base: self.base_constraints.len(),
            extension: self.extension_constraints.len(),
        }
    }

    /// Maximum cached degree multiples, with transition selectors weighted zero.
    ///
    /// Trace, periodic, and boundary-selector leaves have weight one; constants, public values, and challenges have weight zero.
    /// These conservative multiples are trace-size independent, rather than raw polynomial degree bounds.
    pub fn max_constraint_degrees(&self) -> ConstraintDegrees {
        self.max_constraint_degrees_with_transition(0)
    }

    /// Maximum degree multiples with a caller-selected transition-selector weight.
    ///
    /// Uses [`super::SymbolicExpr::degree_multiple_with_transition`] for each root.
    /// A zero weight reuses cached degrees; other weights traverse expressions.
    pub fn max_constraint_degrees_with_transition(
        &self,
        transition_degree: usize,
    ) -> ConstraintDegrees {
        max_constraint_degrees(
            &self.base_constraints,
            &self.extension_constraints,
            transition_degree,
        )
    }

    /// Maximum raw polynomial degree bounds for a trace and its periodic columns.
    ///
    /// Uses [`super::SymbolicExpr::poly_degree`]: trace and boundary-selector leaves have degree `trace_len - 1` (saturating), transition selectors have degree one, and periodic columns use their supplied periods.
    /// Missing periodic entries use the conservative full-trace period, as in the expression API.
    ///
    /// These are polynomial bounds before vanishing-polynomial division, rather than protocol quotient sizing.
    /// Callers choose their domain-specific sizing policy, such as [`super::constraint_degree_from_poly_degree`].
    pub fn max_constraint_poly_degrees(
        &self,
        trace_len: usize,
        periodic_periods: &[usize],
    ) -> ConstraintDegrees {
        max_constraint_poly_degrees(
            &self.base_constraints,
            &self.extension_constraints,
            trace_len,
            periodic_periods,
        )
    }

    /// Maximum constraint degree, in multiples of the degree-`(trace_len - 1)` trace polynomials.
    ///
    /// This is the degree a two-adic quotient is sized from.
    ///
    /// # Arguments
    ///
    /// - `trace_len`: the trace height the degrees are measured against.
    /// - `periodic_periods`: the period of each periodic column, empty when the AIR has none.
    pub fn max_constraint_degree(&self, trace_len: usize, periodic_periods: &[usize]) -> usize {
        // Without periodic columns, start from the cached degree multiple.
        //
        // It weighs the transition selector 0, which is exact for one transition factor.
        // A product of two or more transition factors outgrows it:
        //
        //     T^2 a b:   degree multiple 2,   polynomial degree 2 (n - 1) + 2 = 2n
        //
        // So also convert the trace-size-aware degree, and keep the larger of the two.
        let poly = self.max_constraint_poly_degrees(trace_len, periodic_periods);
        let poly = constraint_degree_from_poly_degree(poly.base.max(poly.extension), trace_len);
        if periodic_periods.is_empty() {
            let cached = self.max_constraint_degrees();
            // Why: the maximum never lowers the cached degree, so no existing AIR loses chunks.
            return cached.base.max(cached.extension).max(poly);
        }

        // Periodic columns can have degree below `trace_len - 1`.
        // The cached multiple would overstate them, so the trace-size-aware degree alone decides.
        poly
    }
}

// Borrowing the streams lets degree inference reuse summaries without collecting a layout.
pub(super) fn max_constraint_degrees<F: Field, EF: ExtensionField<F>>(
    base: &[SymbolicExpression<F>],
    extension: &[SymbolicExpressionExt<F, EF>],
    transition_degree: usize,
) -> ConstraintDegrees {
    ConstraintDegrees {
        base: max_degree(base, |c| {
            c.degree_multiple_with_transition(transition_degree)
        }),
        extension: max_degree(extension, |c| {
            c.degree_multiple_with_transition(transition_degree)
        }),
    }
}

pub(super) fn max_constraint_poly_degrees<F: Field, EF: ExtensionField<F>>(
    base: &[SymbolicExpression<F>],
    extension: &[SymbolicExpressionExt<F, EF>],
    trace_len: usize,
    periodic_periods: &[usize],
) -> ConstraintDegrees {
    ConstraintDegrees {
        base: max_degree(base, |c| c.poly_degree(trace_len, periodic_periods)),
        extension: max_degree(extension, |c| c.poly_degree(trace_len, periodic_periods)),
    }
}

fn max_degree<T>(constraints: &[T], degree: impl Fn(&T) -> usize) -> usize {
    constraints.iter().map(degree).max().unwrap_or(0)
}

/// Evaluate a closure once and take ownership of its symbolic constraints.
///
/// The closure does not need to implement [`crate::Air`] or [`crate::BaseAir`].
/// The caller must supply an [`AirLayout`] matching every input the closure reads.
/// No AIR metadata is available to validate the layout automatically.
/// When calling an existing AIR, use [`AirLayout::validate_against_air`] if validation is needed.
///
/// ```
/// use p3_air::{AirBuilder, AirLayout, WindowAccess, capture_symbolic_constraints};
/// use p3_baby_bear::BabyBear;
///
/// let constraints = capture_symbolic_constraints::<BabyBear, BabyBear>(
///     AirLayout { main_width: 1, ..Default::default() },
///     |builder| {
///         let main = builder.main();
///         let value = main.current(0).unwrap();
///         builder.assert_bool(value);
///         builder.assert_eq(main.next(0).unwrap(), value);
///     },
/// );
/// assert_eq!(constraints.base_constraints().len(), 2);
/// assert_eq!(constraints.base_constraints()[0].degree_multiple(), 2);
/// assert_eq!(constraints.constraint_layout().base_indices, [0, 1]);
/// assert!(constraints.extension_constraints().is_empty());
/// ```
#[must_use]
pub fn capture_symbolic_constraints<F: Field, EF: ExtensionField<F>>(
    layout: AirLayout,
    eval: impl FnOnce(&mut SymbolicAirBuilder<F, EF>),
) -> SymbolicConstraints<F, EF> {
    let mut builder = SymbolicAirBuilder::new(layout);
    eval(&mut builder);
    builder.finish()
}
