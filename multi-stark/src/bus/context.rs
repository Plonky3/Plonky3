//! Binding binary-native bus terminal claims to committed AIR columns.
//!
//! [`BusContext`] derives every block and opening from AIR metadata. It materializes
//! the ProductGKR leaves for proving and reconstructs the formal composition from
//! commitment-bound column evaluations for verification.

use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::AirLayout;
use p3_bus::{
    BusBlockOwner, BusDirection, BusEvaluation, BusPlan, BusPlanInput, BusSymbolicBuilder,
    SymbolicBusInteraction,
};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;

use super::error::BusBindingError;
use super::math::{equality_at_vertex, equality_evaluation};

/// Verifier-derived bus declarations and their checked physical layout.
pub(crate) struct BusContext<F: Field, EF: ExtensionField<F>> {
    /// One symbolic declaration profile per AIR instance.
    profiles: Vec<BusSymbolicBuilder<F, EF>>,
    /// Shared mixed-height layout derived from the profiles.
    plan: BusPlan,
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
        Ok(Some(Self { profiles, plan }))
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
                let block_weight = equality_at_vertex(
                    &output.product.point[..share.prefix_variables],
                    share.prefix_index,
                );
                let row_weight =
                    equality_evaluation(&output.product.point[share.prefix_variables..], row_point);
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
        // Column views are shared by every declaration emitted by the same AIR.
        let columns = tables
            .iter()
            .zip(preprocessed)
            .map(|(main, preprocessed)| {
                (
                    main.iter_polys().collect::<Vec<_>>(),
                    preprocessed
                        .iter()
                        .flat_map(|table| table.iter_polys())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        BusDirection::ALL.map(|direction| {
            let expected = self
                .plan
                .security_geometry()
                .non_padding_leaf_count(direction);
            let mut leaves = Vec::with_capacity(expected);
            let mut scratch = columns
                .iter()
                .map(|(main, preprocessed)| {
                    (EF::zero_vec(main.len()), EF::zero_vec(preprocessed.len()))
                })
                .collect::<Vec<_>>();
            for block in self.plan.blocks(direction) {
                let air = block.owner.air;
                let interaction = &self.profiles[air].interactions()[block.owner.declaration];
                let height = 1usize << block.log_height;
                let (main_columns, prep_columns) = &columns[air];
                let (main, prep) = &mut scratch[air];

                // Resolve every row from the exact tables committed by this proof.
                for row in 0..height {
                    for (value, column) in main.iter_mut().zip(main_columns) {
                        *value = column[row].into();
                    }
                    for (value, column) in prep.iter_mut().zip(prep_columns) {
                        *value = column[row].into();
                    }
                    let factor = self
                        .plan
                        .evaluate_factor(
                            block.bus,
                            interaction,
                            BusEvaluation {
                                main,
                                preprocessed: prep,
                                public: public_values[air],
                                is_first_row: EF::from_bool(row == 0),
                                is_last_row: EF::from_bool(row + 1 == height),
                                is_transition: EF::from_bool(row + 1 < height),
                            },
                            &weights,
                            challenges.offset,
                        )
                        .expect("a checked bus plan resolves against its committed table");
                    leaves.push(factor);
                }
            }
            leaves
        })
    }
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, BaseAir};
    use p3_baby_bear::BabyBear;
    use p3_bus::BusSymbolicBuilder;

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
