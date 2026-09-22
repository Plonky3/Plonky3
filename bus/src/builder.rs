//! Direction-preserving AIR declarations for the binary-native bus.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use p3_air::symbolic::{
    AirLayout, BaseEntry, BaseLeaf, ConstraintLayout, SymbolicAirBuilder, SymbolicExpr,
    SymbolicExpression, SymbolicExpressionExt,
};
use p3_air::{Air, AirBuilder, DebugConstraintBuilder, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_lookup::{
    Count, IndexedLookupBuilder, InteractionBuilder, InteractionSymbolicBuilder, TraceWindow,
};

use crate::{BusDirection, BusName};

/// End of a table at which a boundary declaration contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BusBoundary {
    /// Row zero.
    First,
    /// The last row of the table.
    Last,
}

impl BusBoundary {
    /// Both ends in stable first-then-last order.
    pub const ALL: [Self; 2] = [Self::First, Self::Last];

    /// Whether one row of a table of this height sits at this end.
    #[must_use]
    pub const fn contains_row(self, row: usize, height: usize) -> bool {
        match self {
            Self::First => row == 0,
            Self::Last => row + 1 == height,
        }
    }
}

/// Row activation carried by one bus declaration.
#[derive(Clone, Debug)]
pub enum BusActivation<E> {
    /// Every row contributes one tuple.
    Always,
    /// Only the row at one end of the table contributes its tuple.
    ///
    /// An initial state pushed once, and a final state pulled once, are the two uses.
    ///
    /// The indicator is supplied by the backend, and over the Boolean hypercube it is zero or one on every row, so the declaration owes no Booleanity constraint.
    ///
    /// Emitting no constraint has a sharp edge.
    ///
    /// A table carrying nothing else reaches the zerocheck with no constraint family, and setup asserts rather than returning an error.
    ///
    /// The same declaration under a caller-supplied selector keeps its Booleanity check and is accepted, so such a table needs a local constraint of its own.
    ///
    /// The block still spans the whole table, and its other rows contribute the product identity.
    ///
    /// The crate README says why, and what a height-one block would cost instead.
    Boundary(BusBoundary),
    /// One expression selects whether the row contributes its tuple.
    ///
    /// Soundness needs that expression to be zero or one on every row.
    ///
    /// The public declaration path emits that constraint.
    ///
    /// A declaration assembled by hand carries no such constraint, so its author owes one.
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
        bus: BusName<'_>,
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
/// use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName};
///
/// fn filtered_declaration<AB: BusInteractionBuilder>(builder: &mut AB) {
///     let condition = builder.is_first_row();
///     builder.when(condition).push_bus_interaction(
///         BusName::new("bus"),
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
    /// That Booleanity check is the only thing a declaration leaves the batched zerocheck.
    ///
    /// An AIR declaring nothing else, and no conditional activation either, is refused.
    ///
    /// # Arguments
    ///
    /// - `bus`: channel shared by every matching declaration.
    /// - `direction`: side of the multiset equality receiving the tuple.
    /// - `fields`: tuple expressions in slot order.
    /// - `activation`: whether every row, one boundary row, or only selected rows contribute.
    fn push_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        bus: BusName<'_>,
        direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        activation: BusActivation<Self::Expr>,
    ) {
        // A backend boundary indicator is Boolean on every row of the hypercube already.
        if let BusActivation::Boolean(selector) = &activation {
            self.assert_zero(selector.clone().bool_check());
        }

        self.record_bus_interaction(RecordToken(()), bus, direction, fields, activation);
    }
}

impl<T: BusInteractionRecorder> BusInteractionBuilder for T {}

/// One symbolic tuple contribution emitted by an AIR.
#[derive(Clone, Debug)]
pub struct SymbolicBusInteraction<F: Field> {
    /// Channel shared by matching contributions.
    ///
    /// The declaration path writes a checked name here.
    ///
    /// The plan rechecks every name it is given, so a profile assembled by hand is caught before the transcript.
    pub bus_name: String,
    /// Side of the multiset equality receiving the tuple.
    pub direction: BusDirection,
    /// Tuple expressions in slot order.
    pub fields: Vec<SymbolicExpression<F>>,
    /// Row activation expression, when the contribution is conditional.
    pub activation: BusActivation<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicBusInteraction<F> {
    /// Channel this declaration contributes to.
    ///
    /// # Errors
    ///
    /// Returns an error when the retained name is outside the alphabet, which the declaration path never leaves.
    pub const fn bus(&self) -> Result<BusName<'_>, crate::BusNameError> {
        BusName::try_new(self.bus_name.as_str())
    }

    /// Sorted current-row main and preprocessed columns this declaration reads.
    ///
    /// Only these columns have to be opened and folded for the declaration to be resolved.
    #[must_use]
    pub fn referenced_columns(&self) -> (Vec<usize>, Vec<usize>) {
        let mut main = alloc::collections::BTreeSet::new();
        let mut preprocessed = alloc::collections::BTreeSet::new();
        let mut seen = alloc::collections::BTreeSet::new();
        let mut pending = self
            .fields
            .iter()
            .chain(match &self.activation {
                // A boundary indicator is supplied by the backend rather than read from a column.
                BusActivation::Always | BusActivation::Boundary(_) => None,
                BusActivation::Boolean(selector) => Some(selector),
            })
            .collect::<Vec<_>>();

        // Arithmetic nodes share their operands, so each distinct node is visited once.
        while let Some(expression) = pending.pop() {
            if !seen.insert(core::ptr::from_ref(expression)) {
                continue;
            }
            match expression {
                SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => match variable.entry {
                    BaseEntry::Main { offset: 0 } => {
                        main.insert(variable.index);
                    }
                    BaseEntry::Preprocessed { offset: 0 } => {
                        preprocessed.insert(variable.index);
                    }
                    _ => {}
                },
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
        (
            main.into_iter().collect(),
            preprocessed.into_iter().collect(),
        )
    }

    /// Degree of this interaction's selected factor under a transition-degree scale.
    #[must_use]
    pub fn factor_degree_multiple_with_transition(&self, multiple: usize) -> usize {
        let payload = self
            .fields
            .iter()
            .map(|expression| expression.degree_multiple_with_transition(multiple))
            .max()
            .unwrap_or(0);
        match &self.activation {
            BusActivation::Always => payload,
            // Both boundary indicators carry the degree multiple of a single trace variable.
            BusActivation::Boundary(_) => payload + 1,
            BusActivation::Boolean(selector) => {
                payload + selector.degree_multiple_with_transition(multiple)
            }
        }
    }
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
        bus: BusName<'_>,
        direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        activation: BusActivation<Self::Expr>,
    ) {
        // Keep direction outside field arithmetic so negation cannot erase it.
        self.interactions.push(SymbolicBusInteraction {
            bus_name: bus.as_str().to_string(),
            direction,
            fields: fields.into_iter().map(Into::into).collect(),
            activation,
        });
    }
}

impl<F: Field, EF: ExtensionField<F>> InteractionBuilder for BusSymbolicBuilder<F, EF> {
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        fields: impl IntoIterator<Item = E>,
        _count: impl Into<Count<Self::Expr>>,
    ) {
        // This dedicated pass records binary-bus declarations; the lookup pass records these.
        fields.into_iter().for_each(drop);
    }

    fn push_local_interaction(
        &mut self,
        tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
        // Drain caller-owned iterators while leaving lookup metadata to its dedicated pass.
        tuples.into_iter().for_each(drop);
    }
}

impl<F: Field, EF: ExtensionField<F>> IndexedLookupBuilder for BusSymbolicBuilder<F, EF> {
    fn push_indexed_read(
        &mut self,
        _table: &str,
        _position: usize,
        payload: impl IntoIterator<Item = usize>,
    ) {
        // Indexed declarations are recorded by their own symbolic builder.
        payload.into_iter().for_each(drop);
    }

    fn push_indexed_table(
        &mut self,
        _name: &str,
        _window: TraceWindow,
        columns: impl IntoIterator<Item = usize>,
    ) {
        // Indexed declarations are recorded by their own symbolic builder.
        columns.into_iter().for_each(drop);
    }

    fn num_indexed_reads(&self) -> usize {
        0
    }

    fn num_indexed_tables(&self) -> usize {
        0
    }
}

impl<F: Field, EF: ExtensionField<F>> BusInteractionRecorder for InteractionSymbolicBuilder<F, EF> {
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        _token: RecordToken,
        _bus: BusName<'_>,
        _direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        _activation: BusActivation<Self::Expr>,
    ) {
        // This dedicated pass records lookup declarations; the bus pass records these.
        fields.into_iter().for_each(|field| {
            let _ = field.into();
        });
    }
}

impl<F: Field, EF: ExtensionField<F>> BusInteractionRecorder for DebugConstraintBuilder<'_, F, EF> {
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        _token: RecordToken,
        _bus: BusName<'_>,
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
    use alloc::{format, vec};

    use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExpr};
    use p3_air::{Air, BaseAir, WindowAccess, check_constraints};
    use p3_binary_field::BinaryField128;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    /// The one place this fixture names its channel.
    const DISPATCH: BusName<'static> = BusName::new("dispatch");

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
                DISPATCH,
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
            builder.push_bus_interaction(DISPATCH, self.second, [value], activation);
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
    fn dedicated_symbolic_passes_record_only_their_protocol() {
        struct MixedAir;

        impl BaseAir<BinaryField128> for MixedAir {
            fn width(&self) -> usize {
                2
            }
        }

        impl<AB> Air<AB> for MixedAir
        where
            AB: BusInteractionBuilder<F = BinaryField128> + InteractionBuilder,
        {
            fn eval(&self, builder: &mut AB) {
                let value: AB::Expr = builder.main().current_slice()[0].into();
                let selector: AB::Expr = builder.main().current_slice()[1].into();
                builder.assert_zero(value.clone() - value.clone());
                builder.push_interaction("legacy", [value.clone()], 1);
                builder.push_bus_interaction(
                    BusName::new("binary"),
                    BusDirection::Push,
                    [value],
                    BusActivation::Boolean(selector),
                );
            }
        }

        let layout = AirLayout::from_air(&MixedAir);
        let bus = BusSymbolicBuilder::<BinaryField128>::from_air(&MixedAir, layout);
        let lookup = InteractionSymbolicBuilder::<BinaryField128>::from_air(&MixedAir, layout);

        // Each dedicated pass keeps only its own protocol metadata.
        assert_eq!(bus.interactions().len(), 1);
        assert_eq!(lookup.global_interactions().len(), 1);

        // Both passes retain the same ordinary constraints, including bus selector Booleanity.
        assert_eq!(
            format!("{:?}", bus.base_constraints()),
            format!("{:?}", lookup.base_constraints())
        );
        assert_eq!(
            format!("{:?}", bus.constraint_layout()),
            format!("{:?}", lookup.constraint_layout())
        );
        assert_eq!(bus.base_constraints().len(), 2);
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
                BusName::new("rich"),
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

#[cfg(test)]
mod boundary_tests {
    use alloc::vec;

    use p3_air::symbolic::AirLayout;
    use p3_air::{Air, BaseAir, WindowAccess, check_constraints};
    use p3_binary_field::BinaryField128;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    /// The one place this fixture names its channel.
    const STATE: BusName<'static> = BusName::new("state");

    /// One column carrying a state word, flushed once at each end of the table.
    ///
    /// The initial state enters the bus once and the final state leaves it once.
    ///
    /// The rows in between say nothing about either.
    struct BoundaryAir {
        /// Whether the two ends are declared as boundaries or as caller selectors.
        first_class: bool,
    }

    impl BaseAir<BinaryField128> for BoundaryAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl<AB> Air<AB> for BoundaryAir
    where
        AB: BusInteractionBuilder<F = BinaryField128>,
    {
        fn eval(&self, builder: &mut AB) {
            let value: AB::Expr = builder.main().current_slice()[0].into();
            let (first, last) = if self.first_class {
                (
                    BusActivation::Boundary(BusBoundary::First),
                    BusActivation::Boundary(BusBoundary::Last),
                )
            } else {
                // The same statement written with the backend selectors as caller expressions.
                (
                    BusActivation::Boolean(builder.is_first_row()),
                    BusActivation::Boolean(builder.is_last_row()),
                )
            };
            builder.push_bus_interaction(STATE, BusDirection::Push, [value.clone()], first);
            builder.push_bus_interaction(STATE, BusDirection::Pull, [value], last);
        }
    }

    fn profile(first_class: bool) -> BusSymbolicBuilder<BinaryField128> {
        let air = BoundaryAir { first_class };
        BusSymbolicBuilder::from_air(&air, AirLayout::from_air(&air))
    }

    #[test]
    fn a_boundary_flush_leaves_the_zerocheck_untouched() {
        // A backend row indicator is Boolean on every row of the hypercube by construction.
        assert!(profile(true).base_constraints().is_empty());

        // Spelling the same selector as a caller expression buys two degree-two constraints.
        assert_eq!(profile(false).base_constraints().len(), 2);
    }

    #[test]
    fn a_boundary_flush_costs_what_the_selector_form_costs() {
        // Both forms weight the payload by one linear selector, so the factor degree agrees.
        for (boundary, selector) in profile(true)
            .interactions()
            .iter()
            .zip(profile(false).interactions())
        {
            assert_eq!(
                boundary.factor_degree_multiple_with_transition(1),
                selector.factor_degree_multiple_with_transition(1),
            );
            assert_eq!(boundary.factor_degree_multiple_with_transition(1), 2);
        }

        // A boundary declaration reads no extra column, because its indicator is not a column.
        let boundaries = profile(true);
        let selectors = profile(false);
        assert_eq!(
            boundaries.interactions()[0].referenced_columns(),
            (vec![0], vec![])
        );
        assert_eq!(
            boundaries.interactions()[0].referenced_columns(),
            selectors.interactions()[0].referenced_columns(),
        );
    }

    #[test]
    fn a_boundary_flush_imposes_nothing_on_the_trace() {
        // Any four-row trace satisfies an AIR whose only declarations are boundaries.
        let trace = RowMajorMatrix::new(
            (0..4).map(BinaryField128::from_usize).collect::<Vec<_>>(),
            1,
        );
        check_constraints(&BoundaryAir { first_class: true }, &trace, &[]);
    }

    #[test]
    fn the_boundary_end_agrees_with_the_row_it_names() {
        // The predicate is the same one the replay and the prover row loop evaluate.
        for height in [1usize, 2, 8] {
            for row in 0..height {
                assert_eq!(BusBoundary::First.contains_row(row, height), row == 0,);
                assert_eq!(
                    BusBoundary::Last.contains_row(row, height),
                    row + 1 == height,
                );
            }
        }

        // A one-row table has both ends on its single row.
        assert!(
            BusBoundary::ALL
                .into_iter()
                .all(|end| end.contains_row(0, 1))
        );
    }
}
