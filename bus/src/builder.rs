//! Direction-preserving AIR declarations for the binary-native bus.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt};
use p3_air::{Air, AirBuilder, BaseAir, ExtensionBuilder, PermutationAirBuilder};
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

/// Opt-in AIR interface for binary-native bus declarations.
///
/// Direction is metadata rather than the sign of a field expression.
/// It therefore remains meaningful in characteristic two.
pub trait BusInteractionBuilder: AirBuilder {
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

        self.record_bus_interaction_unchecked(bus_name, direction, fields, activation);
    }

    /// Record a declaration whose activation constraints are already enforced.
    ///
    /// This is the implementation hook beneath the checked public declaration path.
    #[doc(hidden)]
    fn record_bus_interaction_unchecked<E: Into<Self::Expr>>(
        &mut self,
        bus_name: &str,
        direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        activation: BusActivation<Self::Expr>,
    );
}

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
        A: BaseAir<F> + Air<Self>,
    {
        // A mismatched layout would assign expressions to the wrong columns.
        layout.validate_against_air(air);

        // Run the AIR once so constraints and declarations share one expression graph.
        let mut builder = Self::new(layout);
        air.eval(&mut builder);
        builder
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

impl<F: Field, EF: ExtensionField<F>> BusInteractionBuilder for BusSymbolicBuilder<F, EF> {
    fn record_bus_interaction_unchecked<E: Into<Self::Expr>>(
        &mut self,
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExpr};
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_binary_field::BinaryField128;

    use super::*;

    /// One-column AIR used to inspect symbolic bus declarations.
    struct DirectionAir {
        /// Direction of the second declaration.
        second: BusDirection,
        /// Whether the second declaration carries a Boolean selector.
        conditional: bool,
    }

    impl BaseAir<BinaryField128> for DirectionAir {
        fn width(&self) -> usize {
            // One trace column supplies both identical payload expressions.
            1
        }
    }

    impl<AB> Air<AB> for DirectionAir
    where
        AB: BusInteractionBuilder<F = BinaryField128>,
    {
        fn eval(&self, builder: &mut AB) {
            // Read one arbitrary AIR expression rather than a materialized column index.
            let value: AB::Expr = builder.main().current_slice()[0].into();

            // Emit the first copy on the push side.
            builder.push_bus_interaction(
                "dispatch",
                BusDirection::Push,
                [value.clone()],
                BusActivation::Always,
            );

            let activation = if self.conditional {
                BusActivation::Boolean(value.clone())
            } else {
                BusActivation::Always
            };

            // Emit an identical expression with independently recorded direction metadata.
            builder.push_bus_interaction("dispatch", self.second, [value], activation);
        }
    }

    /// Build the symbolic declaration profile for one direction choice.
    fn profile(second: BusDirection, conditional: bool) -> BusSymbolicBuilder<BinaryField128> {
        // The fixture has one main column and no auxiliary inputs.
        let air = DirectionAir {
            second,
            conditional,
        };
        BusSymbolicBuilder::from_air(&air, AirLayout::from_air(&air))
    }

    #[test]
    fn identical_pushes_do_not_cancel_in_characteristic_two() {
        // Fixture state: two identical payload expressions are both pushes.
        let profile = profile(BusDirection::Push, true);

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

        // Conditional activation retains its trace expression for later reconstruction.
        assert!(matches!(
            interactions[1].activation,
            BusActivation::Boolean(SymbolicExpr::Leaf(BaseLeaf::Variable(variable)))
                if variable.entry == BaseEntry::Main { offset: 0 }
                    && variable.index == 0
        ));
    }

    #[test]
    fn boolean_activation_adds_its_own_constraint() {
        // One conditional declaration contributes one Booleanity constraint.
        let conditional = profile(BusDirection::Pull, true);
        assert_eq!(conditional.base_constraints().len(), 1);

        // Unconditional declarations add no constraint to the AIR.
        let unconditional = profile(BusDirection::Pull, false);
        assert!(unconditional.base_constraints().is_empty());
    }
}
