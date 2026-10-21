use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use super::{
    AirLayout, ConstraintLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
};

/// Owned symbolic constraints with a mapping back to global assertion order.
///
/// Both streams retain emission order, including duplicate and zero constraints.
/// [`Self::constraint_layout`] records how the two streams were interleaved.
#[derive(Debug)]
pub struct SymbolicConstraints<F: Field, EF: ExtensionField<F> = F> {
    /// Constraints emitted through [`crate::AirBuilder::assert_zero`].
    pub base_constraints: Vec<SymbolicExpression<F>>,
    /// Constraints emitted through [`crate::ExtensionBuilder::assert_zero_ext`].
    pub extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
    /// Global assertion indices for the base and extension streams.
    pub constraint_layout: ConstraintLayout,
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
/// The method returning this value determines the degree units. An empty stream
/// has degree zero. Bounds do not account for algebraic cancellation.
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
    /// Trace, periodic, and boundary-selector leaves have weight one; constants,
    /// public values, and challenges have weight zero. These conservative multiples
    /// are trace-size independent, rather than raw polynomial degree bounds.
    pub fn max_constraint_degrees(&self) -> ConstraintDegrees {
        ConstraintDegrees {
            base: self
                .base_constraints
                .iter()
                .map(|c| c.degree_multiple())
                .max()
                .unwrap_or(0),
            extension: self
                .extension_constraints
                .iter()
                .map(|c| c.degree_multiple())
                .max()
                .unwrap_or(0),
        }
    }

    /// Maximum degree multiples with a caller-selected transition-selector weight.
    ///
    /// Uses [`super::SymbolicExpr::degree_multiple_with_transition`] for each root.
    /// A zero weight reuses cached degrees; other weights traverse expressions.
    pub fn max_constraint_degrees_with_transition(
        &self,
        transition_degree: usize,
    ) -> ConstraintDegrees {
        ConstraintDegrees {
            base: self
                .base_constraints
                .iter()
                .map(|c| c.degree_multiple_with_transition(transition_degree))
                .max()
                .unwrap_or(0),
            extension: self
                .extension_constraints
                .iter()
                .map(|c| c.degree_multiple_with_transition(transition_degree))
                .max()
                .unwrap_or(0),
        }
    }

    /// Maximum raw polynomial degree bounds for a trace and its periodic columns.
    ///
    /// Uses [`super::SymbolicExpr::poly_degree`]: trace and boundary-selector leaves
    /// have degree `trace_len - 1` (saturating), transition selectors have degree one,
    /// and periodic columns use their supplied periods. Missing periodic entries
    /// use the conservative full-trace period, as in the expression API.
    ///
    /// These are polynomial bounds before vanishing-polynomial division, rather
    /// than protocol quotient sizing. Callers choose their domain-specific sizing
    /// policy, such as [`super::constraint_degree_from_poly_degree`].
    pub fn max_constraint_poly_degrees(
        &self,
        trace_len: usize,
        periodic_periods: &[usize],
    ) -> ConstraintDegrees {
        ConstraintDegrees {
            base: self
                .base_constraints
                .iter()
                .map(|c| c.poly_degree(trace_len, periodic_periods))
                .max()
                .unwrap_or(0),
            extension: self
                .extension_constraints
                .iter()
                .map(|c| c.poly_degree(trace_len, periodic_periods))
                .max()
                .unwrap_or(0),
        }
    }
}

/// Evaluate a closure once and take ownership of its symbolic constraints.
///
/// The closure does not need to implement [`crate::Air`] or [`crate::BaseAir`].
/// The caller must supply an [`AirLayout`] matching every input the closure reads;
/// no AIR metadata is available to validate the layout automatically. When calling
/// an existing AIR, use [`AirLayout::validate_against_air`] if validation is needed.
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
/// assert_eq!(constraints.base_constraints.len(), 2);
/// assert_eq!(constraints.base_constraints[0].degree_multiple(), 2);
/// assert_eq!(constraints.constraint_layout.base_indices, [0, 1]);
/// assert!(constraints.extension_constraints.is_empty());
/// ```
pub fn capture_symbolic_constraints<F: Field, EF: ExtensionField<F>>(
    layout: AirLayout,
    eval: impl FnOnce(&mut SymbolicAirBuilder<F, EF>),
) -> SymbolicConstraints<F, EF> {
    let mut builder = SymbolicAirBuilder::new(layout);
    eval(&mut builder);
    builder.finish()
}
