//! Binding binary-native bus terminal claims to committed AIR columns.
//!
//! [`BusContext`] derives every block and opening from AIR metadata. It materializes
//! the ProductGKR leaves for proving and reconstructs the formal composition from
//! commitment-bound column evaluations for verification.

use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::AirLayout;
use p3_bus::{
    BusBlock, BusBlockOwner, BusDirection, BusEvaluation, BusPlan, BusPlanInput,
    BusSymbolicBuilder, SymbolicBusInteraction,
};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;

use super::error::BusBindingError;
use super::math::equality_evaluation;

/// Verifier-derived bus declarations and their checked physical layout.
pub(crate) struct BusContext<F: Field, EF: ExtensionField<F>> {
    /// One symbolic declaration profile per AIR instance.
    profiles: Vec<BusSymbolicBuilder<F, EF>>,
    /// Shared mixed-height layout derived from the profiles.
    plan: BusPlan,
    /// Sorted main columns each AIR's declarations read, in AIR order.
    main_columns: Vec<Vec<usize>>,
    /// Sorted preprocessed columns each AIR's declarations read, in AIR order.
    preprocessed_columns: Vec<Vec<usize>>,
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

        // Only the columns a declaration reads have to be opened, lifted and folded.
        let (main_columns, preprocessed_columns) = profiles
            .iter()
            .map(|profile| {
                let mut main = alloc::collections::BTreeSet::new();
                let mut preprocessed = alloc::collections::BTreeSet::new();
                for interaction in profile.interactions() {
                    let (fields, fixed) = interaction.referenced_columns();
                    main.extend(fields);
                    preprocessed.extend(fixed);
                }
                (
                    main.into_iter().collect::<Vec<_>>(),
                    preprocessed.into_iter().collect::<Vec<_>>(),
                )
            })
            .collect::<(Vec<_>, Vec<_>)>();
        Ok(Some(Self {
            profiles,
            plan,
            main_columns,
            preprocessed_columns,
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
            self.plan
                .compile_factor(block.bus, self.interaction(block.owner), &weights, EF::ZERO)?
                .evaluate(
                    &mut Vec::new(),
                    BusEvaluation {
                        main: &main,
                        preprocessed: &fixed,
                        public: public_values[air],
                        is_first_row: F::ZERO,
                        is_last_row: F::ZERO,
                        is_transition: F::ZERO,
                    },
                )?;
        }
        Ok(())
    }

    /// Checked layout used by both the transcript and opening schedule.
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

    /// Whether one AIR owns any bus declaration.
    pub(crate) fn contains_air(&self, air: usize) -> bool {
        // Only participating tables need a second prescribed-point opening.
        !self.profiles[air].interactions().is_empty()
    }

    /// Evaluate the formal bus composition at the terminal sumcheck point.
    pub(crate) fn terminal_composition(
        &self,
        output: &p3_bus::BusReductionOutput<EF>,
        direction_challenge: EF,
        point: &Point<EF>,
        main: &[&[EF]],
        preprocessed: &[&[EF]],
        public_values: &[&[F]],
    ) -> Result<EF, BusBindingError> {
        // The terminal point belongs to the tallest participating table.
        if point.num_variables() != self.max_num_variables() {
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
                let row_weight =
                    equality_evaluation(&output.product.point[share.prefix_variables..], row_point)
                        .ok_or(BusBindingError::ProductPointDimension {
                            expected: share.prefix_variables + share.row_variables,
                            actual: output.product.point.len(),
                        })?;
                let boundary = crate::selectors::BoundaryEvals::at(row_point);
                let factor = self.plan.evaluate_factor(
                    share.bus,
                    self.interaction(share.owner),
                    BusEvaluation {
                        main: main[air],
                        preprocessed: preprocessed[air],
                        public: public_values[air],
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
    pub(crate) fn materialize(
        &self,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
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
    fn materialize_block(
        &self,
        block: &BusBlock,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
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
        let main_columns = tables[air].columns().collect::<Vec<_>>();
        let fixed_columns = preprocessed[air]
            .iter()
            .flat_map(|table| table.columns())
            .collect::<Vec<_>>();
        let public = public_values[air];
        let height = 1usize << block.log_height;

        // Boolean rows stay in the base field, so every tuple term is a cheap mixed product.
        (0..height)
            .into_par_iter()
            .map_init(
                || {
                    (
                        F::zero_vec(main_columns.len()),
                        F::zero_vec(fixed_columns.len()),
                        Vec::new(),
                    )
                },
                |(main, fixed, scratch), row| {
                    for (value, column) in main.iter_mut().zip(&main_columns) {
                        *value = column.value(row);
                    }
                    for (value, column) in fixed.iter_mut().zip(&fixed_columns) {
                        *value = column.value(row);
                    }
                    factor
                        .evaluate(
                            scratch,
                            BusEvaluation {
                                main,
                                preprocessed: fixed,
                                public,
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
    use alloc::vec;

    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_bus::{
        BusActivation, BusDirection, BusEvaluationError, BusInteractionBuilder, BusSymbolicBuilder,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_sumcheck::layout::Table;

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
                "memory",
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
    fn only_the_columns_a_declaration_reads_are_scheduled() {
        // This AIR has two main columns and its declaration reads the second one.
        let context = BusContext::<BabyBear, BabyBear>::build(&[&TwoColumnAir], &[1])
            .unwrap()
            .unwrap();

        assert_eq!(context.main_columns(0), &[1]);
        assert!(context.preprocessed_columns(0).is_empty());
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
}
