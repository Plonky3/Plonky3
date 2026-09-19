//! Binding binary-native bus terminal claims to committed AIR columns.
//!
//! [`BusContext`] derives every block and opening from AIR metadata. It materializes
//! the ProductGKR leaves for proving and reconstructs the formal composition from
//! commitment-bound column evaluations for verification.

use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::AirLayout;
use p3_bus::{
    BusBlockOwner, BusDirection, BusEvaluation, BusEvaluationError, BusPlan, BusPlanError,
    BusPlanInput, BusSymbolicBuilder, SymbolicBusInteraction,
};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use thiserror::Error;

/// Verifier-derived bus declarations and their checked physical layout.
pub(crate) struct BusContext<F: Field, EF: ExtensionField<F>> {
    /// One symbolic declaration profile per AIR instance.
    profiles: Vec<BusSymbolicBuilder<F, EF>>,
    /// Shared mixed-height layout derived from the profiles.
    plan: BusPlan,
}

/// Failure to plan, evaluate, or authenticate a bus statement.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BusBindingError {
    /// Symbolic declarations do not define a supported statement.
    #[error(transparent)]
    Plan(#[from] BusPlanError),
    /// A symbolic declaration cannot be evaluated from its supplied values.
    #[error(transparent)]
    Evaluation(#[from] BusEvaluationError),
    /// ProductGKR returned a different number of terminal values than planned.
    #[error("binary-bus ProductGKR returned {actual} values, expected {expected}")]
    ProductValueCount {
        /// Count fixed by the push-and-pull statement.
        expected: usize,
        /// Count returned by the reduction.
        actual: usize,
    },
    /// ProductGKR returned a terminal point of the wrong dimension.
    #[error("binary-bus ProductGKR point has dimension {actual}, expected {expected}")]
    ProductPointDimension {
        /// Dimension fixed by the public product-tree shape.
        expected: usize,
        /// Dimension returned by the reduction.
        actual: usize,
    },
    /// The composition sumcheck returned a terminal point of the wrong dimension.
    #[error("binary-bus composition point has dimension {actual}, expected {expected}")]
    CompositionPointDimension {
        /// Dimension fixed by the tallest participating table.
        expected: usize,
        /// Dimension returned by the sumcheck.
        actual: usize,
    },
    /// The composition proof starts from a claim other than the ProductGKR terminal identity.
    #[error("binary-bus composition initial claim disagrees with ProductGKR")]
    InitialClaimMismatch,
    /// The composition sumcheck terminal claim differs from committed-column evaluation.
    #[error("binary-bus composition terminal claim is not authenticated")]
    TerminalMismatch,
}

impl<F, EF> BusContext<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Derive the complete bus statement from AIR metadata and public trace heights.
    pub(crate) fn build<A>(airs: &[&A], heights: &[usize]) -> Result<Option<Self>, BusPlanError>
    where
        A: Air<BusSymbolicBuilder<F, EF>>,
    {
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

    /// Variables in the tallest participating AIR table.
    pub(crate) fn max_num_variables(&self) -> usize {
        [BusDirection::Push, BusDirection::Pull]
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

    /// Initial claim after randomly batching the push and pull terminal identities.
    pub(crate) fn composition_claim(
        &self,
        output: &p3_bus::BusReductionOutput<EF>,
        direction_challenge: EF,
    ) -> Result<EF, BusBindingError> {
        // ProductGKR is planned for exactly two trees in push-then-pull order.
        if output.product.values.len() != 2 {
            return Err(BusBindingError::ProductValueCount {
                expected: 2,
                actual: output.product.values.len(),
            });
        }
        Ok((output.product.values[0] - EF::ONE)
            + direction_challenge * (output.product.values[1] - EF::ONE))
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
        [BusDirection::Push, BusDirection::Pull].map(|direction| {
            let expected =
                self.plan.security_geometry().non_padding_leaf_counts()[direction_index(direction)];
            let mut leaves = Vec::with_capacity(expected);
            for block in self.plan.blocks(direction) {
                let air = block.owner.air;
                let interaction = &self.profiles[air].interactions()[block.owner.declaration];
                let table = &tables[air];
                let preprocessed = preprocessed[air].as_ref();
                let height = 1usize << block.log_height;
                let main_columns = table.iter_polys().collect::<Vec<_>>();
                let prep_columns = preprocessed
                    .into_iter()
                    .flat_map(|table| table.iter_polys())
                    .collect::<Vec<_>>();
                let mut main = EF::zero_vec(main_columns.len());
                let mut prep = EF::zero_vec(prep_columns.len());

                // Resolve every row from the exact tables committed by this proof.
                for row in 0..height {
                    for (value, column) in main.iter_mut().zip(&main_columns) {
                        *value = column[row].into();
                    }
                    for (value, column) in prep.iter_mut().zip(&prep_columns) {
                        *value = column[row].into();
                    }
                    let factor = self
                        .plan
                        .evaluate_factor(
                            block.bus,
                            interaction,
                            BusEvaluation {
                                main: &main,
                                preprocessed: &prep,
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

const fn direction_index(direction: BusDirection) -> usize {
    // Array order is fixed as push then pull throughout planning and proving.
    match direction {
        BusDirection::Push => 0,
        BusDirection::Pull => 1,
    }
}

fn equality_at_vertex<T: Field>(point: &[T], vertex: usize) -> T {
    // Evaluate the multilinear equality selector for the block's Boolean prefix.
    point
        .iter()
        .enumerate()
        .map(|(coordinate, &challenge)| {
            let bit = (vertex >> (point.len() - 1 - coordinate)) & 1;
            if bit == 0 {
                T::ONE - challenge
            } else {
                challenge
            }
        })
        .product()
}

fn equality_evaluation<T: Field>(left: &[T], right: &[T]) -> T {
    // Multiply the one-coordinate equality extensions in tensor-product order.
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left * right + (T::ONE - left) * (T::ONE - right))
        .product()
}
