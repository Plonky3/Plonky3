//! Direction-preserving AIR declarations for the binary-native bus.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use p3_air::symbolic::{
    AirLayout, ConstraintLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
};
use p3_air::{Air, AirBuilder, DebugConstraintBuilder, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

use crate::BusDirection;

/// Row activation carried by one bus declaration.
#[derive(Clone, Debug)]
pub enum BusActivation<E> {
    /// Every row contributes one tuple.
    Always,
    /// One expression selects whether the row contributes its tuple.
    Boolean(E),
}

/// Capability for recording a binary-native bus declaration.
///
/// Implementations receive only declarations whose activation constraint was emitted.
#[doc(hidden)]
pub trait BusInteractionRecorder: AirBuilder {
    /// Store one checked declaration.
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        token: RecordToken,
        bus_name: &str,
        direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        activation: BusActivation<Self::Expr>,
    );
}

/// Proof that the public declaration path emitted its activation constraint.
#[doc(hidden)]
pub struct RecordToken(());

/// AIR interface for binary-native bus declarations.
///
/// Direction is metadata rather than the sign of a field expression.
/// It therefore remains meaningful in characteristic two.
///
/// A conditional declaration adds a constraint of twice the selector's degree.
/// Backend row selectors are not valid activations unless they are independently Boolean.
///
/// Filtering a declaration is intentionally unsupported.
/// A filtered builder would constrain an activation only on filtered rows while recording it globally.
///
/// ```compile_fail
/// use p3_air::AirBuilder;
/// use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder};
///
/// fn filtered_declaration<AB: BusInteractionBuilder>(builder: &mut AB) {
///     let condition = builder.is_first_row();
///     builder.when(condition).push_bus_interaction(
///         "bus",
///         BusDirection::Push,
///         core::iter::empty::<AB::Expr>(),
///         BusActivation::Always,
///     );
/// }
/// ```
pub trait BusInteractionBuilder: BusInteractionRecorder {
    /// Declare one tuple contribution on a named bus.
    ///
    /// A conditional activation is constrained to zero or one before it is recorded.
    ///
    /// # Arguments
    ///
    /// - `bus_name`: channel shared by every matching declaration.
    /// - `direction`: side of the multiset equality receiving the tuple.
    /// - `fields`: tuple expressions in slot order.
    /// - `activation`: whether every row or only selected rows contribute.
    fn push_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus_name: &str,
        direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        activation: BusActivation<Self::Expr>,
    ) {
        if let BusActivation::Boolean(selector) = &activation {
            self.assert_zero(selector.clone().bool_check());
        }

        self.record_bus_interaction(RecordToken(()), bus_name, direction, fields, activation);
    }
}

impl<T: BusInteractionRecorder> BusInteractionBuilder for T {}

/// One symbolic tuple contribution emitted by an AIR.
#[derive(Clone, Debug)]
pub struct SymbolicBusInteraction<F: Field> {
    /// Channel shared by matching contributions.
    pub bus_name: String,
    /// Side of the multiset equality receiving the tuple.
    pub direction: BusDirection,
    /// Tuple expressions in slot order.
    pub fields: Vec<SymbolicExpression<F>>,
    /// Row activation expression, when the contribution is conditional.
    pub activation: BusActivation<SymbolicExpression<F>>,
}

/// Symbolic AIR builder that retains binary-native bus declarations.
#[derive(Debug)]
pub struct BusSymbolicBuilder<F: Field, EF: ExtensionField<F> = F> {
    /// Constraint recorder supplying symbolic trace variables.
    inner: SymbolicAirBuilder<F, EF>,
    /// Shape the symbolic trace variables were allocated against.
    layout: AirLayout,
    /// Bus declarations in AIR emission order.
    interactions: Vec<SymbolicBusInteraction<F>>,
}

impl<F: Field, EF: ExtensionField<F>> BusSymbolicBuilder<F, EF> {
    /// Create an empty symbolic builder for one AIR layout.
    #[must_use]
    pub fn new(layout: AirLayout) -> Self {
        // Keep ordinary constraints and bus metadata in one symbolic evaluation.
        Self {
            inner: SymbolicAirBuilder::new(layout),
            layout,
            interactions: Vec::new(),
        }
    }

    /// Evaluate one AIR symbolically and retain its bus declarations.
    ///
    /// # Panics
    ///
    /// Panics when the supplied layout disagrees with the AIR's declared widths.
    #[must_use]
    pub fn from_air<A>(air: &A, layout: AirLayout) -> Self
    where
        A: Air<Self>,
    {
        // A mismatched layout would assign expressions to the wrong columns.
        layout.validate_against_air(air);

        // Run the AIR once so constraints and declarations share one expression graph.
        let mut builder = Self::new(layout);
        air.eval(&mut builder);
        builder
    }

    /// Shape the symbolic trace variables were allocated against.
    #[must_use]
    pub const fn layout(&self) -> AirLayout {
        self.layout
    }

    /// Symbolic declarations in AIR emission order.
    #[must_use]
    pub fn interactions(&self) -> &[SymbolicBusInteraction<F>] {
        &self.interactions
    }

    /// Symbolic base-field constraints emitted in the same AIR evaluation.
    #[must_use]
    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.inner.base_constraints()
    }

    /// Symbolic extension-field constraints emitted in the same AIR evaluation.
    #[must_use]
    pub fn extension_constraints(&self) -> Vec<SymbolicExpressionExt<F, EF>> {
        self.inner.extension_constraints()
    }

    /// Global emission order of base-field and extension-field constraints.
    #[must_use]
    pub fn constraint_layout(&self) -> ConstraintLayout {
        self.inner.constraint_layout()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for BusSymbolicBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = <SymbolicAirBuilder<F, EF> as AirBuilder>::Var;
    type PreprocessedWindow = <SymbolicAirBuilder<F, EF> as AirBuilder>::PreprocessedWindow;
    type MainWindow = <SymbolicAirBuilder<F, EF> as AirBuilder>::MainWindow;
    type PublicVar = <SymbolicAirBuilder<F, EF> as AirBuilder>::PublicVar;
    type PeriodicVar = <SymbolicAirBuilder<F, EF> as AirBuilder>::PeriodicVar;

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

    fn assert_zero<I: Into<Self::Expr>>(&mut self, value: I) {
        self.inner.assert_zero(value);
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values()
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for BusSymbolicBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type EF = EF;
    type ExprEF = SymbolicExpressionExt<F, EF>;
    type VarEF = <SymbolicAirBuilder<F, EF> as ExtensionBuilder>::VarEF;

    fn assert_zero_ext<I: Into<Self::ExprEF>>(&mut self, value: I) {
        self.inner.assert_zero_ext(value);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for BusSymbolicBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = <SymbolicAirBuilder<F, EF> as PermutationAirBuilder>::MP;
    type RandomVar = <SymbolicAirBuilder<F, EF> as PermutationAirBuilder>::RandomVar;
    type PermutationVar = <SymbolicAirBuilder<F, EF> as PermutationAirBuilder>::PermutationVar;

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

impl<F: Field, EF: ExtensionField<F>> BusInteractionRecorder for BusSymbolicBuilder<F, EF> {
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        _token: RecordToken,
        bus_name: &str,
        direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        activation: BusActivation<Self::Expr>,
    ) {
        // Keep direction outside field arithmetic so negation cannot erase it.
        self.interactions.push(SymbolicBusInteraction {
            bus_name: bus_name.to_string(),
            direction,
            fields: fields.into_iter().map(Into::into).collect(),
            activation,
        });
    }
}

impl<F: Field, EF: ExtensionField<F>> BusInteractionRecorder for DebugConstraintBuilder<'_, F, EF> {
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        _token: RecordToken,
        _bus_name: &str,
        _direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        _activation: BusActivation<Self::Expr>,
    ) {
        // Resolve every field conversion while leaving concrete debugging to AIR constraints.
        fields.into_iter().for_each(|field| {
            let _ = field.into();
        });
    }
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;
    use alloc::vec;

    use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExpr};
    use p3_air::{Air, BaseAir, WindowAccess, check_constraints};
    use p3_binary_field::BinaryField128;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    /// Two-column AIR used to inspect symbolic bus declarations.
    struct DirectionAir {
        /// Direction of the second declaration.
        second: BusDirection,
        /// Whether the second declaration carries a Boolean selector.
        conditional: bool,
    }

    impl BaseAir<BinaryField128> for DirectionAir {
        fn width(&self) -> usize {
            // One column carries payloads and one carries row activation.
            2
        }
    }

    impl<AB> Air<AB> for DirectionAir
    where
        AB: BusInteractionBuilder<F = BinaryField128>,
    {
        fn eval(&self, builder: &mut AB) {
            // Keep the payload and selector expressions independent.
            let value: AB::Expr = builder.main().current_slice()[0].into();
            let selector: AB::Expr = builder.main().current_slice()[1].into();

            // Emit the first copy on the push side.
            builder.push_bus_interaction(
                "dispatch",
                BusDirection::Push,
                [value.clone()],
                BusActivation::Always,
            );

            let activation = if self.conditional {
                BusActivation::Boolean(selector)
            } else {
                BusActivation::Always
            };

            // Emit an identical expression with independently recorded direction metadata.
            builder.push_bus_interaction("dispatch", self.second, [value], activation);
        }
    }

    /// Build the symbolic declaration profile for one direction choice.
    fn profile(second: BusDirection, conditional: bool) -> BusSymbolicBuilder<BinaryField128> {
        // The fixture has independent payload and selector columns.
        let air = DirectionAir {
            second,
            conditional,
        };
        BusSymbolicBuilder::from_air(&air, AirLayout::from_air(&air))
    }

    #[test]
    fn identical_pushes_do_not_cancel_in_characteristic_two() {
        // Fixture state: two identical payload expressions are both pushes.
        let profile = profile(BusDirection::Push, false);

        // Structural records remain two entries even though `-x == x` in this field.
        assert_eq!(profile.interactions().len(), 2);
        assert_eq!(
            profile
                .interactions()
                .iter()
                .map(|interaction| interaction.direction)
                .collect::<Vec<_>>(),
            vec![BusDirection::Push, BusDirection::Push],
        );
    }

    #[test]
    fn push_and_pull_remain_distinct_in_the_symbolic_profile() {
        // Fixture state: the payload expressions are identical and only direction differs.
        let profile = profile(BusDirection::Pull, true);
        let interactions = profile.interactions();

        // The profile keeps the two multiset sides distinct before any field evaluation.
        assert_eq!(interactions[0].direction, BusDirection::Push);
        assert_eq!(interactions[1].direction, BusDirection::Pull);

        // Both records still point at the same arbitrary main-column expression.
        for interaction in interactions {
            assert!(matches!(
                interaction.fields.as_slice(),
                [SymbolicExpr::Leaf(BaseLeaf::Variable(variable))]
                    if variable.entry == BaseEntry::Main { offset: 0 }
                        && variable.index == 0
            ));
        }

        // Conditional activation retains its independent selector expression.
        assert!(matches!(
            interactions[1].activation,
            BusActivation::Boolean(SymbolicExpr::Leaf(BaseLeaf::Variable(variable)))
                if variable.entry == BaseEntry::Main { offset: 0 }
                    && variable.index == 1
        ));
    }

    #[test]
    fn boolean_activation_adds_its_own_constraint() {
        // One conditional declaration contributes the polynomial s * (s - 1).
        let conditional = profile(BusDirection::Pull, true);
        assert_eq!(conditional.base_constraints().len(), 1);
        let constraint = &conditional.base_constraints()[0];

        // Both Boolean values satisfy the constraint.
        assert_eq!(
            evaluate(constraint, [BinaryField128::ZERO, BinaryField128::ZERO]),
            BinaryField128::ZERO
        );
        assert_eq!(
            evaluate(constraint, [BinaryField128::ZERO, BinaryField128::ONE]),
            BinaryField128::ZERO
        );

        // A non-Boolean field element does not satisfy the constraint.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0xB055_B001);
        let non_boolean = loop {
            let candidate = rng.random::<BinaryField128>();
            if candidate != BinaryField128::ZERO && candidate != BinaryField128::ONE {
                break candidate;
            }
        };
        assert_ne!(
            evaluate(constraint, [BinaryField128::ZERO, non_boolean]),
            BinaryField128::ZERO
        );

        // Unconditional declarations add no constraint to the AIR.
        let unconditional = profile(BusDirection::Pull, false);
        assert!(unconditional.base_constraints().is_empty());
    }

    /// Builds a concrete four-row trace with one payload and one selector column.
    fn selector_trace(selectors: [BinaryField128; 4]) -> RowMajorMatrix<BinaryField128> {
        let values = selectors
            .into_iter()
            .enumerate()
            .flat_map(|(row, selector)| [BinaryField128::from_usize(row), selector])
            .collect();
        RowMajorMatrix::new(values, 2)
    }

    #[test]
    fn debug_builder_accepts_boolean_bus_activations() {
        // Every selector is zero or one.
        let trace = selector_trace([
            BinaryField128::ZERO,
            BinaryField128::ONE,
            BinaryField128::ONE,
            BinaryField128::ZERO,
        ]);

        // The automatic Booleanity constraint vanishes on every row.
        let air = DirectionAir {
            second: BusDirection::Pull,
            conditional: true,
        };
        check_constraints(&air, &trace, &[]);
    }

    #[test]
    #[should_panic]
    fn debug_builder_rejects_non_boolean_bus_activations() {
        // The tower generator is distinct from both Boolean values.
        let trace = selector_trace([
            BinaryField128::ZERO,
            BinaryField128::ONE,
            BinaryField128::GENERATOR,
            BinaryField128::ZERO,
        ]);

        // Concrete constraint checking must reject the malformed selector row.
        let air = DirectionAir {
            second: BusDirection::Pull,
            conditional: true,
        };
        check_constraints(&air, &trace, &[]);
    }

    /// Evaluate the symbolic arithmetic used by the selector fixture.
    fn evaluate(
        expression: &SymbolicExpression<BinaryField128>,
        current: [BinaryField128; 2],
    ) -> BinaryField128 {
        match expression {
            SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => match variable.entry {
                BaseEntry::Main { offset: 0 } => current[variable.index],
                _ => panic!("the selector fixture uses only current main columns"),
            },
            SymbolicExpr::Leaf(BaseLeaf::Constant(value)) => *value,
            SymbolicExpr::Leaf(_) => panic!("the selector fixture uses no row selectors"),
            SymbolicExpr::Add { x, y, .. } => evaluate(x, current) + evaluate(y, current),
            SymbolicExpr::Sub { x, y, .. } => evaluate(x, current) - evaluate(y, current),
            SymbolicExpr::Neg { x, .. } => -evaluate(x, current),
            SymbolicExpr::Mul { x, y, .. } => evaluate(x, current) * evaluate(y, current),
        }
    }

    /// AIR covering compound expressions and mixed constraint kinds.
    struct RichAir;

    impl BaseAir<BinaryField128> for RichAir {
        fn width(&self) -> usize {
            // Payload and activation occupy separate trace columns.
            2
        }

        fn num_public_values(&self) -> usize {
            // One public value participates in the declared tuple.
            1
        }

        fn num_periodic_columns(&self) -> usize {
            // One periodic value participates in the declared tuple.
            1
        }

        fn periodic_columns(&self) -> Cow<'_, [Vec<BinaryField128>]> {
            // A two-row public cycle is enough to allocate the symbolic entry.
            Cow::Owned(vec![vec![BinaryField128::ZERO, BinaryField128::ONE]])
        }
    }

    impl<AB> Air<AB> for RichAir
    where
        AB: BusInteractionBuilder<F = BinaryField128> + ExtensionBuilder<EF = BinaryField128>,
        AB::ExprEF: From<BinaryField128>,
    {
        fn eval(&self, builder: &mut AB) {
            // Emit a base constraint before the extension constraint.
            let value: AB::Expr = builder.main().current_slice()[0].into();
            builder.assert_zero(value.clone() - value.clone());
            builder.assert_zero_ext(BinaryField128::ZERO);

            // Mix current, next, public, and periodic expressions in one tuple field.
            let selector: AB::Expr = builder.main().current_slice()[1].into();
            let next: AB::Expr = builder.main().next_slice()[0].into();
            let public: AB::Expr = builder.public_values()[0].into();
            let periodic: AB::Expr = builder.periodic_values()[0].into();
            let compound = value * selector.clone() + next + public + periodic;
            builder.push_bus_interaction(
                "rich",
                BusDirection::Push,
                [compound],
                BusActivation::Boolean(selector),
            );
        }
    }

    #[test]
    fn compound_fields_preserve_global_constraint_order() {
        // The AIR emits base, extension, then automatic Booleanity constraints.
        let air = RichAir;
        let profile = BusSymbolicBuilder::from_air(&air, AirLayout::from_air(&air));
        assert_eq!(profile.interactions().len(), 1);
        let entries = variable_entries(&profile.interactions()[0].fields[0]);
        assert!(entries.contains(&BaseEntry::Main { offset: 1 }));
        assert!(entries.contains(&BaseEntry::Public));
        assert!(entries.contains(&BaseEntry::Periodic));
        assert_eq!(profile.base_constraints().len(), 2);
        assert_eq!(profile.extension_constraints().len(), 1);

        // Global positions retain the interleaving required by alpha decomposition.
        let layout = profile.constraint_layout();
        assert_eq!(layout.base_indices, vec![0, 2]);
        assert_eq!(layout.ext_indices, vec![1]);
    }

    /// Collects every trace or public entry referenced by one symbolic expression.
    fn variable_entries(expression: &SymbolicExpression<BinaryField128>) -> Vec<BaseEntry> {
        let mut entries = Vec::new();
        let mut pending = vec![expression];
        while let Some(expression) = pending.pop() {
            match expression {
                SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => entries.push(variable.entry),
                SymbolicExpr::Leaf(_) => {}
                SymbolicExpr::Add { x, y, .. }
                | SymbolicExpr::Sub { x, y, .. }
                | SymbolicExpr::Mul { x, y, .. } => {
                    pending.push(x);
                    pending.push(y);
                }
                SymbolicExpr::Neg { x, .. } => pending.push(x),
            }
        }
        entries
    }
}
