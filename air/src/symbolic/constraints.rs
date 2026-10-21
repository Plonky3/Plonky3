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
