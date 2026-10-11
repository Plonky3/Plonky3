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
    BusActivation, BusAffineExpression, BusBlock, BusBlockOwner, BusBoundary, BusDirection,
    BusEvaluation, BusFactorPlan, BusPlan, BusPlanInput, BusSymbolicBuilder,
    SymbolicBusInteraction,
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

/// An affine expression bound to the original, validated column views.
struct AffineRow<'a, F: Field, EF: Field> {
    constant: EF,
    columns: Vec<(ColumnView<'a, F>, EF)>,
    first_row: EF,
    last_row: EF,
    transition: EF,
}

impl<'a, F: Field, EF: ExtensionField<F>> AffineRow<'a, F, EF> {
    fn bind(
        expression: BusAffineExpression<EF>,
        sources: [(&[usize], &[ColumnView<'a, F>]); 3],
    ) -> Option<Self> {
        let mut columns = Vec::with_capacity(
            expression.main.len() + expression.preprocessed.len() + expression.periodic.len(),
        );
        for (coefficients, (indices, views)) in [
            expression.main,
            expression.preprocessed,
            expression.periodic,
        ]
        .into_iter()
        .zip(sources)
        {
            for (index, coefficient) in coefficients {
                let position = indices.binary_search(&index).ok()?;
                columns.push((*views.get(position)?, coefficient));
            }
        }
        Some(Self {
            constant: expression.constant,
            columns,
            first_row: expression.first_row,
            last_row: expression.last_row,
            transition: expression.transition,
        })
    }

    fn evaluate(&self, row: usize, height: usize) -> EF {
        let mut value = self.constant;
        for &(column, coefficient) in &self.columns {
            value += match column {
                ColumnView::Dense(values) => coefficient * values[row],
                ColumnView::Boolean { words, column, len } => {
                    assert!(row < len, "column row out of bounds");
                    let bit = (words.values[(row / 64) * words.width + column] >> (row % 64)) & 1;
                    [EF::ZERO, coefficient][bit as usize]
                }
            };
        }
        if row == 0 {
            value += self.first_row;
        }
        if row + 1 == height {
            value += self.last_row;
        }
        if row + 1 < height {
            value += self.transition;
        }
        value
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

    /// Prepare a Boolean affine evaluator only when columns and live rows provide reuse.
    #[inline(never)]
    fn try_affine_leaves(
        factor: &BusFactorPlan<'_, F, EF>,
        public: &[F],
        sources: [(&[usize], &[ColumnView<'_, F>]); 3],
        selector: &RowSelector<'_, F>,
        height: usize,
    ) -> Option<Vec<EF>> {
        if height < 8 || matches!(selector, RowSelector::Boundary(_)) {
            return None;
        }
        let columns = || sources.iter().flat_map(|(_, columns)| columns.iter());
        if columns()
            .filter(|column| matches!(column, ColumnView::Boolean { .. }))
            .take(8)
            .count()
            < 8
        {
            return None;
        }
        // Inspect all original views before consulting zero-pruned coefficients.
        if sources
            .iter()
            .any(|(indices, columns)| indices.len() != columns.len())
            || columns().any(|column| column.len() != height)
        {
            return None;
        }
        if let RowSelector::Column(column) = selector
            && column
                .values()
                .filter(|value| !value.is_zero())
                .take(8)
                .count()
                < 8
        {
            return None;
        }
        let affine = factor.affine(public).ok().flatten()?;
        let difference = AffineRow::bind(affine.factor_minus_one, sources)?;
        let activation = match affine.activation {
            Some(expression) => Some(AffineRow::bind(expression, sources)?),
            None => None,
        };
        Some(
            (0..height)
                .into_par_iter()
                .map(|row| {
                    if selector.is_inactive(row, height) {
                        return EF::ONE;
                    }
                    let difference = difference.evaluate(row, height);
                    EF::ONE
                        + activation.as_ref().map_or(difference, |activation| {
                            difference * activation.evaluate(row, height)
                        })
                })
                .collect(),
        )
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

        // Dense tables cannot use the Boolean coefficient lookup. Check table storage
        // before scanning the declaration's columns or preparing affine coefficients.
        let has_packed_sources = tables[air].packed_bits().is_some()
            || preprocessed[air].is_some_and(|table| table.packed_bits().is_some())
            || periodic[air]
                .as_ref()
                .is_some_and(|table| table.packed_bits().is_some());
        if has_packed_sources
            && let Some(leaves) = Self::try_affine_leaves(
                &factor,
                public,
                [
                    (main_indices.as_slice(), main_columns.as_slice()),
                    (fixed_indices.as_slice(), fixed_columns.as_slice()),
                    (periodic_indices.as_slice(), periodic_columns.as_slice()),
                ],
                &selector,
                height,
            )
        {
            return leaves;
        }

        let initialize = || {
            (
                F::zero_vec(main_width),
                F::zero_vec(fixed_width),
                F::zero_vec(periodic_width),
                Vec::new(),
            )
        };
        let evaluate = |(main, fixed, periodic, scratch): &mut (Vec<F>, Vec<F>, Vec<F>, Vec<F>),
                        row| {
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
        };

        // A boundary block has one live leaf. Fill the padding directly and
        // initialize row scratch only for the row whose payload is evaluated.
        if height > 1
            && let RowSelector::Boundary(boundary) = &selector
        {
            let row = match boundary {
                BusBoundary::First => 0,
                BusBoundary::Last => height - 1,
            };
            let mut leaves = alloc::vec![EF::ONE; height];
            leaves[row] = evaluate(&mut initialize(), row);
            return leaves;
        }

        // Boolean rows stay in the base field, so every tuple term is a cheap mixed product.
        (0..height)
            .into_par_iter()
            .map_init(initialize, evaluate)
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

    use super::{BusBindingError, BusContext, RowSelector};

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

    mod affine_rows {
        extern crate std;

        use alloc::borrow::Cow;
        use alloc::vec;
        use alloc::vec::Vec;

        use p3_air::{Air, BaseAir, WindowAccess};
        use p3_bus::{
            BusActivation, BusBoundary, BusChallenges, BusDirection, BusEvaluation,
            BusInteractionBuilder, BusName,
        };
        use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
        use p3_matrix::dense::RowMajorMatrix;
        use p3_sumcheck::layout::Table;

        use super::BusContext;

        struct AffineAir {
            single_row: bool,
            nonlinear: bool,
            selector: u8,
        }
        impl<F: Field> BaseAir<F> for AffineAir {
            fn width(&self) -> usize {
                10
            }
            fn preprocessed_width(&self) -> usize {
                2
            }
            fn num_public_values(&self) -> usize {
                2
            }
            fn num_periodic_columns(&self) -> usize {
                1
            }
            fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
                Cow::Owned(vec![if self.single_row {
                    vec![F::ONE]
                } else {
                    vec![F::ZERO, F::ONE]
                }])
            }
        }
        impl<AB> Air<AB> for AffineAir
        where
            AB: BusInteractionBuilder,
            AB::F: Field,
            AB::Expr: From<AB::PeriodicVar>,
        {
            fn eval(&self, b: &mut AB) {
                let main = b.main();
                let row = main.current_slice();
                let prep = b.preprocessed().clone();
                let fixed = prep.current_slice();
                let public: Vec<AB::Expr> =
                    b.public_values().iter().copied().map(Into::into).collect();
                let periodic: AB::Expr = b.periodic_values()[0].into();
                let mut fields: Vec<AB::Expr> = (0..9)
                    .map(|i| row[i] * AB::F::interpolation_node(i + 2))
                    .collect();
                fields.push(row[0] * public[0].clone() + fixed[0] + periodic.clone());
                fields.push(if self.nonlinear {
                    row[0] * row[1] * public[1].clone()
                } else {
                    row[2] - row[3]
                });
                fields.push(
                    b.is_first_row()
                        + b.is_last_row() * AB::F::interpolation_node(2)
                        + b.is_transition() * AB::F::interpolation_node(3),
                );
                let activation = match self.selector {
                    0 => BusActivation::Always,
                    1 => BusActivation::Boundary(BusBoundary::First),
                    2 => BusActivation::Boundary(BusBoundary::Last),
                    3 => BusActivation::Boolean(fixed[1].into()),
                    _ => BusActivation::Boolean(fixed[1] + periodic),
                };
                b.push_bus_interaction(
                    BusName::new("affine-edge"),
                    BusDirection::Push,
                    fields,
                    activation,
                );
            }
        }
        fn tables<F: Field>(log: usize) -> (Table<F>, Table<F>) {
            let height = 1usize << log;
            let mut words = vec![0u64; 10 * height.div_ceil(64)];
            for row in 0..height {
                for col in 0..10 {
                    let x = (row as u64).wrapping_mul(0x9e3779b97f4a7c15)
                        ^ (col as u64 + 17).wrapping_mul(0xbf58476d1ce4e5b9);
                    words[(row / 64) * 10 + col] |= ((x ^ (x >> 29) ^ (x >> 47)) & 1) << (row % 64);
                }
            }
            let main = Table::from_packed_bits(RowMajorMatrix::new(words, 10), log);
            let fixed = Table::new(RowMajorMatrix::new(
                (0..2)
                    .flat_map(|col| {
                        (0..height).map(move |row| {
                            if col == 0 {
                                F::interpolation_node(3 + row % 7)
                            } else {
                                [F::ZERO, F::ONE, F::interpolation_node(2)][row % 3]
                            }
                        })
                    })
                    .collect(),
                height,
            ));
            (main, fixed)
        }
        fn challenges<F: Field, EF: ExtensionField<F>>(
            width: usize,
            mode: usize,
        ) -> BusChallenges<EF> {
            let value = |j: usize| match mode {
                0 => EF::ZERO,
                1 => EF::ONE,
                _ => <EF as BasedVectorSpace<F>>::from_basis_coefficients_fn(|i| {
                    F::interpolation_node(17 + j * 3 + i)
                }),
            };
            BusChallenges {
                fingerprint: (0..width.ilog2() as usize).map(value).collect(),
                offset: value(7),
            }
        }
        fn check_fields<F: Field, EF: ExtensionField<F>>() {
            for log in [0, 3, 6] {
                for nonlinear in [false, true] {
                    for selector in 0..5 {
                        let air = AffineAir {
                            single_row: log == 0,
                            nonlinear,
                            selector,
                        };
                        let context = BusContext::<F, EF>::build(&[&air], &[log])
                            .unwrap()
                            .unwrap();
                        let periodic = context.periodic_tables(&[&air], &[log]);
                        let (main, fixed) = tables::<F>(log);
                        let height = 1usize << log;
                        for public_mode in 0..3 {
                            for challenge_mode in 0..3 {
                                let public =
                                    [F::interpolation_node(2), F::interpolation_node(public_mode)];
                                let ch = challenges::<F, EF>(
                                    context.plan().fingerprint_width(),
                                    challenge_mode,
                                );
                                let actual = context.materialize(
                                    &[&main],
                                    &[Some(&fixed)],
                                    &periodic,
                                    &[&public],
                                    &ch,
                                );
                                // Expand every original column independently of coefficient/source pruning.
                                let weights = ch.fingerprint_weights();
                                let block = &context.plan().blocks(BusDirection::Push)[0];
                                let factor = context
                                    .plan()
                                    .compile_factor(
                                        block.bus,
                                        context.interaction(block.owner),
                                        &weights,
                                        ch.offset,
                                    )
                                    .unwrap();
                                assert_eq!(actual[0].len(), height);
                                assert!(actual[1].is_empty());
                                for (row, &leaf) in actual[0].iter().enumerate() {
                                    let main_row: Vec<F> =
                                        main.columns().map(|c| c.value(row)).collect();
                                    let fixed_row: Vec<F> =
                                        fixed.columns().map(|c| c.value(row)).collect();
                                    let periodic_row: Vec<F> = periodic[0]
                                        .as_ref()
                                        .unwrap()
                                        .columns()
                                        .map(|c| c.value(row))
                                        .collect();
                                    let reference = factor
                                        .evaluate(
                                            &mut Vec::new(),
                                            BusEvaluation {
                                                main: &main_row,
                                                preprocessed: &fixed_row,
                                                public: &public,
                                                periodic: &periodic_row,
                                                is_first_row: F::from_bool(row == 0),
                                                is_last_row: F::from_bool(row + 1 == height),
                                                is_transition: F::from_bool(row + 1 < height),
                                            },
                                        )
                                        .unwrap();
                                    assert_eq!(
                                        leaf, reference,
                                        "log={log}, row={row}, nonlinear={nonlinear}, selector={selector}, public={public_mode}, challenge={challenge_mode}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        #[test]
        fn affine_rows_cover_distinct_source_and_challenge_fields() {
            check_fields::<p3_binary_field::BinaryField32, p3_binary_field::BinaryField128>();
            check_fields::<p3_binary_field::Poly64, p3_binary_field::Poly192>();
            check_fields::<
                p3_baby_bear::BabyBear,
                p3_field::extension::BinomialExtensionField<p3_baby_bear::BabyBear, 4>,
            >();
        }
        #[test]
        fn zero_coefficients_do_not_hide_incomplete_sources() {
            use p3_binary_field::BinaryField128 as F;

            use self::std::panic::{AssertUnwindSafe, catch_unwind};
            let air = AffineAir {
                single_row: false,
                nonlinear: false,
                selector: 0,
            };
            let context = BusContext::<F, F>::build(&[&air], &[6]).unwrap().unwrap();
            let periodic = context.periodic_tables(&[&air], &[6]);
            let (main, fixed) = tables::<F>(6);
            let (short_main, short_fixed) = tables::<F>(3);
            let public = [F::ZERO; 2];
            // Zero weights prune affine coefficients; the original graph still reads its sources.
            let ch = challenges::<F, F>(context.plan().fingerprint_width(), 0);
            for (main, fixed) in [
                (&main, None),
                (&short_main, Some(&fixed)),
                (&main, Some(&short_fixed)),
            ] {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        context.materialize(&[main], &[fixed], &periodic, &[&public], &ch)
                    }))
                    .is_err()
                );
            }
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    context.materialize(&[&main], &[Some(&fixed)], &periodic, &[&[]], &ch)
                }))
                .is_err()
            );
        }
    }

    mod affine_routing {
        use alloc::string::ToString;
        use alloc::vec;
        use alloc::vec::Vec;

        use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpression, SymbolicVariable};
        use p3_binary_field::{BinaryField32 as F, BinaryField128 as EF};
        use p3_bus::{
            BusActivation, BusBoundary, BusDirection, BusPlan, BusPlanInput, SymbolicBusInteraction,
        };
        use p3_field::{Field, PrimeCharacteristicRing};
        use p3_matrix::dense::RowMajorMatrix;
        use p3_sumcheck::layout::{ColumnView, Table};

        use super::{BusContext, RowSelector};

        #[test]
        fn packed_affine_route_requires_reused_columns_and_live_rows() {
            let main_indices: Vec<_> = (0..8).collect();
            let fixed_indices = [0];
            let packed = Table::<F>::from_packed_bits(RowMajorMatrix::new(vec![0x55; 8], 8), 3);
            let packed_views: Vec<_> = packed.columns().collect();
            let dense = [F::ZERO; 8];
            let mut narrow_views = packed_views.clone();
            narrow_views[7] = ColumnView::Dense(&dense);
            let variable = |entry, index| {
                SymbolicExpression::<F>::from(SymbolicVariable::<F>::new(entry, index))
            };
            let mut fields = (0..8)
                .map(|i| variable(BaseEntry::Main { offset: 0 }, i))
                .collect::<Vec<_>>();
            fields.extend(
                [
                    BaseLeaf::IsFirstRow,
                    BaseLeaf::IsLastRow,
                    BaseLeaf::IsTransition,
                ]
                .map(SymbolicExpression::Leaf),
            );
            for boundary in [false, true] {
                let interactions = [SymbolicBusInteraction {
                    bus_name: "route".to_string(),
                    direction: BusDirection::Push,
                    fields: fields.clone(),
                    activation: if boundary {
                        BusActivation::Boundary(BusBoundary::First)
                    } else {
                        BusActivation::Boolean(variable(BaseEntry::Preprocessed { offset: 0 }, 0))
                    },
                }];
                let plan = BusPlan::build(&[BusPlanInput {
                    log_height: 3,
                    interactions: &interactions,
                }])
                .unwrap()
                .unwrap();
                let weights = (0..plan.fingerprint_width())
                    .map(|i| EF::interpolation_node(17 + i))
                    .collect::<Vec<_>>();
                let factor = plan
                    .compile_factor(0, &interactions[0], &weights, EF::interpolation_node(31))
                    .unwrap();
                for active in [7, 8] {
                    let mut selectors = [F::ONE; 8];
                    if active == 7 {
                        selectors[7] = F::ZERO;
                    }
                    let fixed = [ColumnView::Dense(&selectors)];
                    let selector = if boundary {
                        RowSelector::Boundary(BusBoundary::First)
                    } else {
                        RowSelector::Column(fixed[0])
                    };
                    for (views, packed_count) in [(&packed_views, 8), (&narrow_views, 7)] {
                        let result = BusContext::<F, EF>::try_affine_leaves(
                            &factor,
                            &[],
                            [(&main_indices, views), (&fixed_indices, &fixed), (&[], &[])],
                            &selector,
                            8,
                        );
                        assert_eq!(
                            result.is_some(),
                            !boundary && active == 8 && packed_count == 8,
                            "boundary={boundary}, active={active}, packed={packed_count}"
                        );
                    }
                }
            }
        }
    }
}
