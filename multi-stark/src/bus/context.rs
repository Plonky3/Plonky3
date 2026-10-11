//! Binding binary-native bus terminal claims to committed AIR columns.
//!
//! [`BusContext`] derives every block and the columns it reads from AIR metadata.
//! It materializes the ProductGKR leaves for proving.
//! It evaluates the bus shares at the shared sumcheck point for verification.
//! Those evaluations read each AIR's single opening at that point.
//! A periodic column is never opened; the verifier evaluates it at that point instead.

use alloc::vec::Vec;

use p3_air::symbolic::{AirLayout, BaseEntry, BaseLeaf, SymbolicExpr};
use p3_air::{Air, BaseAir};
use p3_bus::{
    BusActivation, BusBlock, BusBlockOwner, BusBoundary, BusDirection, BusEvaluation, BusPlan,
    BusPlanInput, BusSymbolicBuilder, SymbolicBusInteraction,
};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{ColumnView, Table};

use super::error::BusBindingError;
use crate::selectors::{periodic_num_variables, periodic_table};

/// Source columns used by one declaration, retaining their original AIR indices.
struct DeclarationSources {
    main: Vec<usize>,
    preprocessed: Vec<usize>,
    periodic: Vec<usize>,
}

/// Cheap row selectors; more general expressions keep the full factor evaluator.
enum RowSelector<'a, F: Field> {
    General,
    Boundary(BusBoundary),
    Column(ColumnView<'a, F>),
}

impl<F: Field> RowSelector<'_, F> {
    fn is_inactive(&self, row: usize, height: usize) -> bool {
        match self {
            Self::General => false,
            Self::Boundary(boundary) => !boundary.contains_row(row, height),
            Self::Column(column) => column.value(row).is_zero(),
        }
    }
}

/// Verifier-derived bus declarations and their checked physical layout.
pub(crate) struct BusContext<F: Field, EF: ExtensionField<F>> {
    /// One symbolic declaration profile per AIR instance.
    profiles: Vec<BusSymbolicBuilder<F, EF>>,
    /// Exact sources per AIR and declaration, before they are unioned for composition.
    declaration_sources: Vec<Vec<DeclarationSources>>,
    /// Shared mixed-height layout derived from the profiles.
    plan: BusPlan,
    /// Sorted main columns each AIR's declarations read, in AIR order.
    main_columns: Vec<Vec<usize>>,
    /// Sorted preprocessed columns each AIR's declarations read, in AIR order.
    preprocessed_columns: Vec<Vec<usize>>,
    /// Sorted periodic columns each AIR's declarations read, in AIR order.
    periodic_columns: Vec<Vec<usize>>,
    /// Periodic columns each AIR declares, in AIR order.
    periodic_widths: Vec<usize>,
}

impl<F, EF> BusContext<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Derive the complete bus statement from AIR metadata and public trace heights.
    pub(crate) fn build<A>(airs: &[&A], heights: &[usize]) -> Result<Option<Self>, BusBindingError>
    where
        A: Air<BusSymbolicBuilder<F, EF>>,
    {
        if airs.len() != heights.len() {
            return Err(BusBindingError::InstanceCountMismatch {
                airs: airs.len(),
                heights: heights.len(),
            });
        }

        // Each AIR is evaluated once under the dedicated bus recorder.
        let profiles = airs
            .iter()
            .map(|air| BusSymbolicBuilder::from_air(*air, AirLayout::from_air(*air)))
            .collect::<Vec<_>>();
        let inputs = profiles
            .iter()
            .zip(heights)
            .map(|(profile, &log_height)| BusPlanInput {
                log_height,
                interactions: profile.interactions(),
            })
            .collect::<Vec<_>>();
        let Some(plan) = BusPlan::build(&inputs)? else {
            return Ok(None);
        };

        // Keep each declaration's sources for leaf construction. Composition still uses
        // their per-AIR union, so all terms share each interpolated source column.
        let declaration_sources = profiles
            .iter()
            .map(|profile| {
                profile
                    .interactions()
                    .iter()
                    .map(|interaction| {
                        let (main, preprocessed) = interaction.referenced_columns();
                        DeclarationSources {
                            main,
                            preprocessed,
                            periodic: interaction.referenced_periodic_columns(),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (main_columns, preprocessed_columns) = declaration_sources
            .iter()
            .map(|declarations| {
                let mut main = alloc::collections::BTreeSet::new();
                let mut preprocessed = alloc::collections::BTreeSet::new();
                for sources in declarations {
                    main.extend(sources.main.iter().copied());
                    preprocessed.extend(sources.preprocessed.iter().copied());
                }
                (
                    main.into_iter().collect::<Vec<_>>(),
                    preprocessed.into_iter().collect::<Vec<_>>(),
                )
            })
            .collect::<(Vec<_>, Vec<_>)>();

        // Periodic columns are never opened; the verifier evaluates them at the shared point.
        let periodic_columns = declaration_sources
            .iter()
            .map(|declarations| {
                declarations
                    .iter()
                    .flat_map(|sources| sources.periodic.iter().copied())
                    .collect::<alloc::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        let periodic_widths = airs.iter().map(|air| air.num_periodic_columns()).collect();
        Ok(Some(Self {
            profiles,
            declaration_sources,
            plan,
            main_columns,
            preprocessed_columns,
            periodic_columns,
            periodic_widths,
        }))
    }

    /// Main columns one AIR's declarations read, in ascending order.
    pub(crate) fn main_columns(&self, air: usize) -> &[usize] {
        &self.main_columns[air]
    }

    /// Preprocessed columns one AIR's declarations read, in ascending order.
    pub(crate) fn preprocessed_columns(&self, air: usize) -> &[usize] {
        &self.preprocessed_columns[air]
    }

    /// Periodic columns one AIR's declarations read, in ascending order.
    pub(crate) fn periodic_columns(&self, air: usize) -> &[usize] {
        &self.periodic_columns[air]
    }

    /// Periodic tables at full trace height, for the AIRs whose declarations read one.
    ///
    /// An AIR whose declarations read none gets `None`, so its columns are never materialized.
    pub(crate) fn periodic_tables<A: BaseAir<F>>(
        &self,
        airs: &[&A],
        num_variables: &[usize],
    ) -> Vec<Option<Table<F>>> {
        airs.iter()
            .zip(num_variables)
            .zip(&self.periodic_columns)
            .map(|((air, &num_vars), read)| {
                if read.is_empty() {
                    None
                } else {
                    periodic_table(*air, num_vars)
                }
            })
            .collect()
    }

    /// Period vectors of every periodic column, for the AIRs whose declarations read one.
    ///
    /// An AIR whose declarations read none gets `None`, so its columns are never fetched.
    ///
    /// # Panics
    ///
    /// Panics when a declaration does not fit its trace height, as [`Self::periodic_tables`] does.
    pub(crate) fn period_vectors<A: BaseAir<F>>(
        &self,
        airs: &[&A],
        num_variables: &[usize],
    ) -> Vec<Option<Vec<Vec<F>>>> {
        airs.iter()
            .zip(num_variables)
            .zip(&self.periodic_columns)
            .map(|((air, &num_vars), read)| {
                if read.is_empty() {
                    return None;
                }
                let columns = air.periodic_columns();
                if columns.is_empty() {
                    return None;
                }
                // Reject a declaration the trace cannot hold, matching the verifier's own check.
                let periods = columns.iter().map(Vec::len).collect::<Vec<_>>();
                periodic_num_variables(air.num_periodic_columns(), &periods, num_vars)
                    .expect("periodic column declaration must fit the trace height");
                Some(columns.into_owned())
            })
            .collect()
    }

    /// Reject committed tables that cannot resolve the declarations their AIR owns.
    ///
    /// One dry evaluation per declaration keeps a caller mistake out of the prover's row loop.
    pub(crate) fn check_tables(
        &self,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        public_values: &[&[F]],
    ) -> Result<(), BusBindingError> {
        // Zero weights still walk every expression, which is what resolves each leaf.
        let weights = EF::zero_vec(self.plan.fingerprint_width());
        for block in BusDirection::ALL
            .into_iter()
            .flat_map(|direction| self.plan.blocks(direction))
        {
            let air = block.owner.air;
            let main = F::zero_vec(tables[air].num_polys());
            let fixed = F::zero_vec(preprocessed[air].map_or(0, Table::num_polys));
            let periodic = F::zero_vec(self.periodic_widths[air]);
            self.plan
                .compile_factor(block.bus, self.interaction(block.owner), &weights, EF::ZERO)?
                .evaluate(
                    &mut Vec::new(),
                    BusEvaluation {
                        main: &main,
                        preprocessed: &fixed,
                        public: public_values[air],
                        periodic: &periodic,
                        is_first_row: F::ZERO,
                        is_last_row: F::ZERO,
                        is_transition: F::ZERO,
                    },
                )?;
        }
        Ok(())
    }

    /// Checked layout read by the product tree, the shared sumcheck and the security report.
    pub(crate) const fn plan(&self) -> &BusPlan {
        // One shared plan keeps physical leaf order identical across every phase.
        &self.plan
    }

    /// Symbolic declaration owned by one planned block.
    pub(crate) fn interaction(&self, owner: BusBlockOwner) -> &SymbolicBusInteraction<F> {
        // Stable owner coordinates avoid searching by mutable physical block order.
        &self.profiles[owner.air].interactions()[owner.declaration]
    }

    /// Every retained declaration in AIR order.
    pub(crate) fn interactions(&self) -> impl Iterator<Item = &SymbolicBusInteraction<F>> {
        // AIR order makes degree derivation independent of the optimized leaf layout.
        self.profiles
            .iter()
            .flat_map(BusSymbolicBuilder::interactions)
    }

    /// Per-variable degree of the equality-weighted bus composition.
    pub(crate) fn composition_degree(&self) -> usize {
        self.interactions()
            .map(|interaction| interaction.factor_degree_multiple_with_transition(1))
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1)
    }

    /// Variables in the tallest participating AIR table.
    pub(crate) fn max_num_variables(&self) -> usize {
        BusDirection::ALL
            .into_iter()
            .flat_map(|direction| self.plan.blocks(direction))
            .map(|block| block.log_height)
            .max()
            .expect("a bus context contains at least one declaration")
    }

    /// Evaluate the formal bus composition at the terminal sumcheck point.
    ///
    /// The point spans the shared cube, which may be wider than any bus table.
    /// Each share reads its rows off the point's suffix.
    /// Every coordinate ahead of that suffix joins its all-one-vertex selector.
    ///
    /// `periodic` holds each AIR's periodic columns evaluated at that same suffix.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn terminal_composition(
        &self,
        output: &p3_bus::BusReductionOutput<EF>,
        direction_challenge: EF,
        point: &Point<EF>,
        main: &[&[EF]],
        preprocessed: &[&[EF]],
        periodic: &[&[EF]],
        public_values: &[&[F]],
    ) -> Result<EF, BusBindingError> {
        // The terminal point must at least address the tallest participating table.
        if point.num_variables() < self.max_num_variables() {
            return Err(BusBindingError::CompositionPointDimension {
                expected: self.max_num_variables(),
                actual: point.num_variables(),
            });
        }
        let expected_product_dimension = self.plan.product_shape().log_height();
        if output.product.point.len() != expected_product_dimension {
            return Err(BusBindingError::ProductPointDimension {
                expected: expected_product_dimension,
                actual: output.product.point.len(),
            });
        }
        let weights = output.challenges.fingerprint_weights();
        let mut terminal = EF::ZERO;

        for direction in [BusDirection::Push, BusDirection::Pull] {
            let direction_weight = match direction {
                BusDirection::Push => EF::ONE,
                BusDirection::Pull => direction_challenge,
            };
            for share in self.plan.terminal_shares(direction) {
                let air = share.owner.air;
                let row_point = &point.as_slice()[point.num_variables() - share.row_variables..];
                let unused = &point.as_slice()[..point.num_variables() - share.row_variables];
                let fixed_all_one_selector = unused.iter().copied().product::<EF>();
                let block_weight = share.prefix_weight(&output.product.point).ok_or(
                    BusBindingError::ProductPointDimension {
                        expected: expected_product_dimension,
                        actual: output.product.point.len(),
                    },
                )?;
                // Truncating to the shorter point would return a well-formed but wrong weight.
                let row_suffix = &output.product.point[share.prefix_variables..];
                if row_suffix.len() != row_point.len() {
                    return Err(BusBindingError::ProductPointDimension {
                        expected: share.prefix_variables + share.row_variables,
                        actual: output.product.point.len(),
                    });
                }
                let row_weight = Point::eval_eq(row_suffix, row_point);
                let boundary = crate::selectors::BoundaryEvals::at(row_point);
                let factor = self.plan.evaluate_factor(
                    share.bus,
                    self.interaction(share.owner),
                    BusEvaluation {
                        main: main[air],
                        preprocessed: preprocessed[air],
                        public: public_values[air],
                        periodic: periodic[air],
                        is_first_row: boundary.first,
                        is_last_row: boundary.last,
                        is_transition: boundary.transition,
                    },
                    &weights,
                    output.challenges.offset,
                )?;
                terminal += direction_weight
                    * block_weight
                    * fixed_all_one_selector
                    * row_weight
                    * (factor - EF::ONE);
            }
        }
        Ok(terminal)
    }

    /// Materialize both direction-specific product prefixes from committed tables.
    ///
    /// `periodic` holds the tables [`Self::periodic_tables`] returns.
    pub(crate) fn materialize(
        &self,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        periodic: &[Option<Table<F>>],
        public_values: &[&[F]],
        challenges: &p3_bus::BusChallenges<EF>,
    ) -> [Vec<EF>; 2] {
        // Fingerprint weights are shared by every row and every declaration.
        let weights = challenges.fingerprint_weights();
        BusDirection::ALL.map(|direction| {
            // Blocks are independent, and concatenating them restores the planned leaf order.
            self.plan
                .blocks(direction)
                .iter()
                .map(|block| {
                    self.materialize_block(
                        block,
                        tables,
                        preprocessed,
                        periodic,
                        public_values,
                        &weights,
                        challenges.offset,
                    )
                })
                .collect::<Vec<_>>()
                .concat()
        })
    }

    /// Materialize one aligned block from the exact tables committed by this proof.
    #[allow(clippy::too_many_arguments)]
    fn materialize_block(
        &self,
        block: &BusBlock,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        periodic: &[Option<Table<F>>],
        public_values: &[&[F]],
        weights: &[EF],
        offset: EF,
    ) -> Vec<EF> {
        let air = block.owner.air;
        let interaction = &self.profiles[air].interactions()[block.owner.declaration];

        // Slot placement and the named-domain contribution are settled once for the block.
        let factor = self
            .plan
            .compile_factor(block.bus, interaction, weights, offset)
            .expect("a checked bus plan compiles against its own declarations");
        // Representation-independent views read a packed Boolean table without expanding it.
        // Only the columns this declaration names are read, so an unread one is never decoded.
        let sources = &self.declaration_sources[air][block.owner.declaration];
        let main_indices = &sources.main;
        let main_columns = main_indices
            .iter()
            .map(|&column| tables[air].column(column))
            .collect::<Vec<_>>();
        let fixed_indices = &sources.preprocessed;
        let fixed_columns = preprocessed[air]
            .iter()
            .flat_map(|table| fixed_indices.iter().map(|&column| table.column(column)))
            .collect::<Vec<_>>();
        let periodic_indices = &sources.periodic;
        let periodic_columns = periodic[air]
            .iter()
            .flat_map(|table| periodic_indices.iter().map(|&column| table.column(column)))
            .collect::<Vec<_>>();
        let main_width = tables[air].num_polys();
        let fixed_width = preprocessed[air].map_or(0, Table::num_polys);
        let periodic_width = self.periodic_widths[air];
        let public = public_values[air];
        let height = 1usize << block.log_height;

        // Resolve this after compiling the factor and its source views. Only exact zero
        // suppresses a row; a non-Boolean nonzero value keeps its original interpolation.
        let selector = match &interaction.activation {
            BusActivation::Boundary(boundary) => RowSelector::Boundary(*boundary),
            BusActivation::Boolean(SymbolicExpr::Leaf(BaseLeaf::Variable(variable))) => {
                let column = match variable.entry {
                    BaseEntry::Main { offset: 0 } => Some(tables[air].column(variable.index)),
                    BaseEntry::Preprocessed { offset: 0 } => {
                        preprocessed[air].map(|table| table.column(variable.index))
                    }
                    BaseEntry::Periodic => periodic[air]
                        .as_ref()
                        .map(|table| table.column(variable.index)),
                    _ => None,
                };
                column.map_or(RowSelector::General, RowSelector::Column)
            }
            _ => RowSelector::General,
        };

        // Boolean rows stay in the base field, so every tuple term is a cheap mixed product.
        (0..height)
            .into_par_iter()
            .map_init(
                || {
                    (
                        F::zero_vec(main_width),
                        F::zero_vec(fixed_width),
                        F::zero_vec(periodic_width),
                        Vec::new(),
                    )
                },
                |(main, fixed, periodic, scratch), row| {
                    if selector.is_inactive(row, height) {
                        return EF::ONE;
                    }
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, column) in main_indices.iter().zip(&main_columns) {
                        main[index] = column.value(row);
                    }
                    for (&index, column) in fixed_indices.iter().zip(&fixed_columns) {
                        fixed[index] = column.value(row);
                    }
                    for (&index, column) in periodic_indices.iter().zip(&periodic_columns) {
                        periodic[index] = column.value(row);
                    }
                    factor
                        .evaluate(
                            scratch,
                            BusEvaluation {
                                main,
                                preprocessed: fixed,
                                public,
                                periodic,
                                is_first_row: F::from_bool(row == 0),
                                is_last_row: F::from_bool(row + 1 == height),
                                is_transition: F::from_bool(row + 1 < height),
                            },
                        )
                        // Table widths are checked against every AIR before this loop starts.
                        .expect("a checked bus plan resolves against its committed table")
                },
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_bus::{
        BusActivation, BusDirection, BusEvaluationError, BusInteractionBuilder, BusName,
        BusSymbolicBuilder,
    };
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_sumcheck::layout::Table;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{BusBindingError, BusContext};

    struct EmptyAir;

    impl BaseAir<BabyBear> for EmptyAir {
        fn width(&self) -> usize {
            0
        }
    }

    impl Air<BusSymbolicBuilder<BabyBear>> for EmptyAir {
        fn eval(&self, _builder: &mut BusSymbolicBuilder<BabyBear>) {}
    }

    struct TwoColumnAir;

    impl BaseAir<BabyBear> for TwoColumnAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB: BusInteractionBuilder<F = BabyBear>> Air<AB> for TwoColumnAir {
        fn eval(&self, builder: &mut AB) {
            let value: AB::Expr = builder.main().current_slice()[1].into();
            builder.push_bus_interaction(
                BusName::new("memory"),
                BusDirection::Push,
                [value],
                BusActivation::Always,
            );
        }
    }

    /// One main column, and a declaration reading a periodic column of four rows.
    struct PeriodicAir;

    impl BaseAir<BabyBear> for PeriodicAir {
        fn width(&self) -> usize {
            1
        }

        fn num_periodic_columns(&self) -> usize {
            1
        }

        fn periodic_columns(&self) -> Cow<'_, [Vec<BabyBear>]> {
            Cow::Owned(vec![(0..4).map(BabyBear::from_u64).collect()])
        }
    }

    impl<AB: BusInteractionBuilder<F = BabyBear>> Air<AB> for PeriodicAir {
        fn eval(&self, builder: &mut AB) {
            let value: AB::Expr = builder.periodic_values()[0].into();
            builder.push_bus_interaction(
                BusName::new("image"),
                BusDirection::Push,
                [value],
                BusActivation::Always,
            );
        }
    }

    #[test]
    fn rejects_a_table_narrower_than_the_air_that_owns_it() {
        let context = BusContext::<BabyBear, BabyBear>::build(&[&TwoColumnAir], &[1])
            .unwrap()
            .unwrap();
        // One committed column cannot resolve a declaration reading the second one.
        let narrow = Table::new(RowMajorMatrix::new(vec![BabyBear::ZERO; 2], 2));

        assert_eq!(
            context.check_tables(&[&narrow], &[None], &[&[]]),
            Err(BusBindingError::Evaluation(
                BusEvaluationError::MainColumn { column: 1 }
            ))
        );

        // The declared width resolves the same declaration without an error.
        let wide = Table::new(RowMajorMatrix::new(vec![BabyBear::ZERO; 4], 2));
        assert_eq!(context.check_tables(&[&wide], &[None], &[&[]]), Ok(()));
    }

    #[test]
    fn only_the_columns_a_declaration_reads_are_lifted() {
        // This AIR has two main columns and its declaration reads the second one.
        let context = BusContext::<BabyBear, BabyBear>::build(&[&TwoColumnAir], &[1])
            .unwrap()
            .unwrap();

        assert_eq!(context.main_columns(0), &[1]);
        assert!(context.preprocessed_columns(0).is_empty());
    }

    #[test]
    #[should_panic(expected = "periodic column declaration must fit the trace height")]
    fn rejects_a_period_longer_than_the_trace() {
        // The prover reads row `r` of a periodic column as entry `r mod period`, through a mask.
        // That holds only for a period dividing the height, so two rows cannot carry four.
        let context = BusContext::<BabyBear, BabyBear>::build(&[&PeriodicAir], &[1])
            .unwrap()
            .unwrap();

        context.period_vectors(&[&PeriodicAir], &[1]);
    }

    #[test]
    fn rejects_mismatched_instance_counts() {
        let Err(error) = BusContext::<BabyBear, BabyBear>::build(&[&EmptyAir], &[]) else {
            panic!("AIR and height counts must agree");
        };

        assert_eq!(
            error,
            BusBindingError::InstanceCountMismatch {
                airs: 1,
                heights: 0,
            }
        );
    }

    struct SparseSourcesAir {
        detailed: bool,
        single_row: bool,
    }

    impl<F: Field> BaseAir<F> for SparseSourcesAir {
        fn width(&self) -> usize {
            if self.detailed { 8 } else { 4 }
        }

        fn preprocessed_width(&self) -> usize {
            if self.detailed { 5 } else { 0 }
        }

        fn num_public_values(&self) -> usize {
            if self.detailed { 2 } else { 0 }
        }

        fn num_periodic_columns(&self) -> usize {
            if self.detailed { 3 } else { 0 }
        }

        fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
            if self.detailed {
                let mut columns = vec![
                    vec![F::ZERO, F::ONE],
                    vec![F::ONE, F::ONE, F::ZERO, F::ZERO],
                    vec![F::ZERO, F::ONE, F::ONE, F::ZERO],
                ];
                if self.single_row {
                    for column in &mut columns {
                        column.truncate(1);
                    }
                }
                Cow::Owned(columns)
            } else {
                Cow::Borrowed(&[])
            }
        }
    }

    impl<AB> Air<AB> for SparseSourcesAir
    where
        AB: BusInteractionBuilder,
        AB::F: Field,
        AB::Expr: From<AB::PeriodicVar>,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder
                .main()
                .current_slice()
                .iter()
                .copied()
                .map(Into::into)
                .collect::<Vec<AB::Expr>>();
            if !self.detailed {
                builder.push_bus_interaction(
                    BusName::new("plain"),
                    BusDirection::Pull,
                    [main[2].clone()],
                    BusActivation::Always,
                );
                return;
            }
            let fixed = builder
                .preprocessed()
                .current_slice()
                .iter()
                .copied()
                .map(Into::into)
                .collect::<Vec<AB::Expr>>();
            let periodic = builder
                .periodic_values()
                .iter()
                .copied()
                .map(AB::Expr::from)
                .collect::<Vec<_>>();
            let public = builder
                .public_values()
                .iter()
                .copied()
                .map(Into::into)
                .collect::<Vec<AB::Expr>>();
            // Sparse payload indices and columns read only by a nonlinear selector.
            builder.push_bus_interaction(
                BusName::new("nonlinear"),
                BusDirection::Push,
                [main[1].clone() * main[7].clone()
                    + fixed[3].clone()
                    + periodic[2].clone()
                    + public[1].clone()],
                BusActivation::Boolean(main[5].clone() * fixed[0].clone()),
            );
            // A different declaration needs an activation-only periodic column.
            builder.push_bus_interaction(
                BusName::new("periodic"),
                BusDirection::Pull,
                [main[4].clone() + fixed[4].clone()],
                BusActivation::Boolean(periodic[1].clone()),
            );
            for (name, selector) in [
                ("main_selected", main[5].clone()),
                ("fixed_selected", fixed[0].clone()),
            ] {
                builder.push_bus_interaction(
                    BusName::new(name),
                    BusDirection::Push,
                    [main[1].clone() * main[7].clone() + fixed[3].clone()],
                    BusActivation::Boolean(selector),
                );
            }
            builder.push_bus_interaction(
                BusName::new("boundary"),
                BusDirection::Push,
                [main[2].clone() + periodic[0].clone()],
                BusActivation::Boundary(p3_bus::BusBoundary::First),
            );
            // This declaration reads no committed or periodic column.
            builder.push_bus_interaction(
                BusName::new("public"),
                BusDirection::Pull,
                [public[0].clone()],
                BusActivation::Boundary(p3_bus::BusBoundary::Last),
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum SourceMode {
        DenseBits,
        PackedBits,
        Native,
    }

    fn source_table<F: Field>(
        width: usize,
        log_height: usize,
        mode: SourceMode,
        rng: &mut SmallRng,
    ) -> Table<F>
    where
        StandardUniform: Distribution<F>,
    {
        let height = 1usize << log_height;
        if matches!(mode, SourceMode::Native) {
            // Include zero, one, and non-Boolean activations in the same table.
            let nonboolean = core::iter::repeat_with(|| rng.random::<F>())
                .find(|value| !value.is_zero() && !value.is_one())
                .unwrap();
            return Table::new(RowMajorMatrix::new(
                (0..width * height)
                    .map(|i| match i % 3 {
                        0 => F::ZERO,
                        1 => F::ONE,
                        _ => nonboolean,
                    })
                    .collect(),
                height,
            ));
        }
        let bits = (0..width * height)
            .map(|_| rng.random::<bool>())
            .collect::<Vec<_>>();
        if matches!(mode, SourceMode::PackedBits) {
            let mut words = vec![0; width * height.div_ceil(64)];
            for column in 0..width {
                for row in 0..height {
                    if bits[column * height + row] {
                        words[(row / 64) * width + column] |= 1u64 << (row % 64);
                    }
                }
            }
            Table::from_packed_bits(RowMajorMatrix::new(words, width), log_height)
        } else {
            Table::new(RowMajorMatrix::new(
                bits.into_iter().map(F::from_bool).collect(),
                height,
            ))
        }
    }

    fn check_materialized_sources<F: Field>(heights: [usize; 2])
    where
        StandardUniform: Distribution<F>,
    {
        let airs = [
            &SparseSourcesAir {
                detailed: true,
                single_row: heights[0] == 0,
            },
            &SparseSourcesAir {
                detailed: false,
                single_row: heights[1] == 0,
            },
        ];
        let context = BusContext::<F, F>::build(&airs, &heights).unwrap().unwrap();
        let periodic = context.periodic_tables(&airs, &heights);
        let mut rng = SmallRng::seed_from_u64(0x50_75_ce);
        let public = [rng.random::<F>(), rng.random::<F>()];
        let public_values = [public.as_slice(), &[]];
        let challenges = p3_bus::BusChallenges {
            fingerprint: (0..p3_util::log2_strict_usize(context.plan().fingerprint_width()))
                .map(|_| rng.random::<F>())
                .collect(),
            offset: rng.random::<F>(),
        };
        let weights = challenges.fingerprint_weights();
        for mode in [
            SourceMode::DenseBits,
            SourceMode::PackedBits,
            SourceMode::Native,
        ] {
            let tables = [
                source_table(8, heights[0], mode, &mut rng),
                source_table(4, heights[1], mode, &mut rng),
            ];
            let fixed_mode = match mode {
                SourceMode::DenseBits => SourceMode::PackedBits,
                SourceMode::PackedBits => SourceMode::DenseBits,
                SourceMode::Native => SourceMode::Native,
            };
            let fixed = source_table(5, heights[0], fixed_mode, &mut rng);
            let table_refs = [&tables[0], &tables[1]];
            let fixed_refs = [Some(&fixed), None];
            assert_eq!(
                context.check_tables(&table_refs, &fixed_refs, &public_values),
                Ok(())
            );
            let actual = context.materialize(
                &table_refs,
                &fixed_refs,
                &periodic,
                &public_values,
                &challenges,
            );
            for (direction, actual) in BusDirection::ALL.into_iter().zip(actual) {
                let mut expected = Vec::new();
                for block in context.plan().blocks(direction) {
                    let air = block.owner.air;
                    let height = 1usize << block.log_height;
                    for row in 0..height {
                        // Resolve every original AIR column, independently of dependency lists.
                        let main = tables[air]
                            .columns()
                            .map(|c| c.value(row))
                            .collect::<Vec<_>>();
                        let fixed = fixed_refs[air].map_or_else(Vec::new, |table| {
                            table.columns().map(|c| c.value(row)).collect()
                        });
                        let periodic = periodic[air].as_ref().map_or_else(Vec::new, |table| {
                            table.columns().map(|c| c.value(row)).collect()
                        });
                        expected.push(
                            context
                                .plan()
                                .evaluate_factor(
                                    block.bus,
                                    context.interaction(block.owner),
                                    p3_bus::BusEvaluation {
                                        main: &main,
                                        preprocessed: &fixed,
                                        public: public_values[air],
                                        periodic: &periodic,
                                        is_first_row: F::from_bool(row == 0),
                                        is_last_row: F::from_bool(row + 1 == height),
                                        is_transition: F::from_bool(row + 1 < height),
                                    },
                                    &weights,
                                    challenges.offset,
                                )
                                .unwrap(),
                        );
                    }
                }
                assert_eq!(
                    actual, expected,
                    "mode={mode:?}, heights={heights:?}, direction={direction:?}"
                );
            }
        }
    }

    #[test]
    fn materialized_leaves_match_full_rows_for_disjoint_declarations() {
        for heights in [[4, 2], [0, 0]] {
            check_materialized_sources::<p3_binary_field::BinaryField128>(heights);
            check_materialized_sources::<p3_binary_field::Ghash128>(heights);
            check_materialized_sources::<BabyBear>(heights);
        }
    }

    #[test]
    fn inactive_selectors_do_not_hide_missing_payload_columns() {
        struct SelectedAir;

        impl BaseAir<BabyBear> for SelectedAir {
            fn width(&self) -> usize {
                2
            }
        }

        impl<AB: BusInteractionBuilder<F = BabyBear>> Air<AB> for SelectedAir {
            fn eval(&self, builder: &mut AB) {
                let main = builder.main();
                let row = main.current_slice();
                builder.push_bus_interaction(
                    BusName::new("selected"),
                    BusDirection::Push,
                    [row[1]],
                    BusActivation::Boolean(row[0].into()),
                );
            }
        }

        let context = BusContext::<BabyBear, BabyBear>::build(&[&SelectedAir], &[1])
            .unwrap()
            .unwrap();
        let narrow = Table::new(RowMajorMatrix::new(vec![BabyBear::ZERO; 2], 2));
        assert_eq!(
            context.check_tables(&[&narrow], &[None], &[&[]]),
            Err(BusBindingError::Evaluation(
                BusEvaluationError::MainColumn { column: 1 }
            ))
        );
    }
}
