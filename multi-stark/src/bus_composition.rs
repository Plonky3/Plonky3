//! Sumcheck binding product-GKR leaf claims to committed trace polynomials.
//!
//! For each direction, ProductGKR leaves a claim `L(q)`. This module proves
//! `L(q) - 1` equals the weighted sum of the rowwise bus factors minus one.
//! Short tables are lifted with `eq(prefix, 1^k)`, whose Boolean-cube sum is one.
//!
//! The terminal expression is checked only after its source columns open from the PCS.

use alloc::vec::Vec;

use p3_bus::{
    BusActivation, BusDirection, BusEvaluation, BusPlan, BusReductionOutput, SymbolicBusInteraction,
};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;
use p3_sumcheck::layout::Table;

use crate::bus::BusContext;

/// Prover state for the mixed-height bus composition polynomial.
pub(crate) struct BusCompositionProver<F: Field, EF: ExtensionField<F>> {
    /// Checked tuple and block layout used by every composition evaluation.
    plan: BusPlan,
    /// Tuple equality coefficients sampled after commitment.
    fingerprint_weights: Vec<EF>,
    /// Random tuple-fingerprint shift.
    offset: EF,
    /// Planned block compositions in physical bus order.
    blocks: Vec<BlockState<F, EF>>,
    /// Number of global variables already bound.
    round: usize,
}

/// One block's formal composition and folded source multilinears.
struct BlockState<F: Field, EF: ExtensionField<F>> {
    /// Named bus selecting the tuple-domain slots.
    bus: usize,
    /// Symbolic factor expression retained from the AIR.
    interaction: SymbolicBusInteraction<F>,
    /// Main trace column polynomials.
    main: Vec<Poly<EF>>,
    /// Preprocessed trace column polynomials.
    preprocessed: Vec<Poly<EF>>,
    /// First-row, last-row, and transition selector polynomials.
    selectors: [Poly<EF>; 3],
    /// Equality polynomial anchored at the ProductGKR row point.
    equality: Poly<EF>,
    /// Fixed coefficient from direction batching and ProductGKR block selection.
    coefficient: EF,
    /// Number of global prefix variables absent from this AIR table.
    unused_prefix: usize,
    /// Evaluation of the fixed all-one-vertex selector on bound prefix coordinates.
    prefix_evaluation: EF,
    /// Cube sum of the unweighted row composition, used before this block activates.
    row_claim: EF,
    /// Public inputs read by this block's symbolic expressions.
    public_values: Vec<F>,
}

impl<F, EF> BusCompositionProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the formal polynomial whose cube sum must equal the ProductGKR claims.
    pub(crate) fn new(
        context: &BusContext<F, EF>,
        output: &BusReductionOutput<EF>,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        public_values: &[&[F]],
        direction_challenge: EF,
    ) -> Self {
        let num_variables = context.max_num_variables();
        let weights = output.challenges.fingerprint_weights();
        let mut blocks = Vec::new();

        for direction in [BusDirection::Push, BusDirection::Pull] {
            let direction_weight = match direction {
                BusDirection::Push => EF::ONE,
                BusDirection::Pull => direction_challenge,
            };
            for share in context.plan().terminal_shares(direction) {
                let air = share.owner.air;
                let interaction = context.interaction(share.owner).clone();
                let row_point = &output.product.point[share.prefix_variables..];
                let block_weight = equality_at_vertex(
                    &output.product.point[..share.prefix_variables],
                    share.prefix_index,
                );
                let coefficient = direction_weight * block_weight;
                let main = tables[air]
                    .iter_polys()
                    .map(|column| Poly::new(column.iter().copied().map(Into::into).collect()))
                    .collect::<Vec<_>>();
                let prep = preprocessed[air]
                    .iter()
                    .flat_map(|table| table.iter_polys())
                    .map(|column| Poly::new(column.iter().copied().map(Into::into).collect()))
                    .collect::<Vec<_>>();
                let height = 1usize << share.row_variables;
                let selectors = [
                    Poly::new((0..height).map(|row| EF::from_bool(row == 0)).collect()),
                    Poly::new(
                        (0..height)
                            .map(|row| EF::from_bool(row + 1 == height))
                            .collect(),
                    ),
                    Poly::new(
                        (0..height)
                            .map(|row| EF::from_bool(row + 1 < height))
                            .collect(),
                    ),
                ];
                let equality = Poly::new(equality_weights(row_point));
                let mut main_values = EF::zero_vec(main.len());
                let mut prep_values = EF::zero_vec(prep.len());
                let mut row_claim = EF::ZERO;
                for row in 0..height {
                    for (value, column) in main_values.iter_mut().zip(&main) {
                        *value = column.as_slice()[row];
                    }
                    for (value, column) in prep_values.iter_mut().zip(&prep) {
                        *value = column.as_slice()[row];
                    }
                    let factor = context
                        .plan()
                        .evaluate_factor(
                            share.bus,
                            &interaction,
                            BusEvaluation {
                                main: &main_values,
                                preprocessed: &prep_values,
                                public: public_values[air],
                                is_first_row: selectors[0].as_slice()[row],
                                is_last_row: selectors[1].as_slice()[row],
                                is_transition: selectors[2].as_slice()[row],
                            },
                            &weights,
                            output.challenges.offset,
                        )
                        .expect("a planned expression resolves against its owning table");
                    row_claim += equality.as_slice()[row] * (factor - EF::ONE);
                }
                blocks.push(BlockState {
                    bus: share.bus,
                    interaction,
                    main,
                    preprocessed: prep,
                    selectors,
                    equality,
                    coefficient,
                    unused_prefix: num_variables - share.row_variables,
                    prefix_evaluation: EF::ONE,
                    row_claim,
                    public_values: public_values[air].to_vec(),
                });
            }
        }

        Self {
            plan: context.plan().clone(),
            fingerprint_weights: weights,
            offset: output.challenges.offset,
            blocks,
            round: 0,
        }
    }

    /// Per-variable degree of the batched formal composition.
    pub(crate) fn degree(context: &BusContext<F, EF>) -> usize {
        // The row equality polynomial adds one degree to every factor expression.
        context
            .interactions()
            .map(factor_degree)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1)
    }

    fn evaluate_block(
        plan: &BusPlan,
        block: &BlockState<F, EF>,
        public: &[F],
        weights: &[EF],
        offset: EF,
        node: EF,
    ) -> EF {
        // Interpolate the current leading variable, then sum over the remaining Boolean cube.
        let half = block.equality.as_slice().len() / 2;
        let mut main = EF::zero_vec(block.main.len());
        let mut prep = EF::zero_vec(block.preprocessed.len());
        let mut sum = EF::ZERO;
        for row in 0..half {
            let interpolate = |poly: &Poly<EF>| {
                let values = poly.as_slice();
                values[row] + (values[row + half] - values[row]) * node
            };
            for (value, polynomial) in main.iter_mut().zip(&block.main) {
                *value = interpolate(polynomial);
            }
            for (value, polynomial) in prep.iter_mut().zip(&block.preprocessed) {
                *value = interpolate(polynomial);
            }
            let factor = plan
                .evaluate_factor(
                    block.bus,
                    &block.interaction,
                    BusEvaluation {
                        main: &main,
                        preprocessed: &prep,
                        public,
                        is_first_row: interpolate(&block.selectors[0]),
                        is_last_row: interpolate(&block.selectors[1]),
                        is_transition: interpolate(&block.selectors[2]),
                    },
                    weights,
                    offset,
                )
                .expect("a planned expression resolves against its folded table");
            sum += interpolate(&block.equality) * (factor - EF::ONE);
        }
        sum
    }
}

impl<F, EF> RoundProver<EF> for BusCompositionProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn fold(&mut self, challenge: EF) {
        // Dormant blocks evaluate one more coordinate of χ_k at the all-one vertex.
        for block in &mut self.blocks {
            if self.round < block.unused_prefix {
                block.prefix_evaluation *= challenge;
                continue;
            }
            for polynomial in block
                .main
                .iter_mut()
                .chain(&mut block.preprocessed)
                .chain(&mut block.selectors)
                .chain(core::iter::once(&mut block.equality))
            {
                polynomial.fix_prefix_var_mut(challenge);
            }
        }
        self.round += 1;
    }

    fn round_poly(&self) -> Vec<EF> {
        // Generic-degree encoding omits node one, which the verifier derives from the claim.
        let degree = self
            .blocks
            .iter()
            .map(|block| factor_degree(&block.interaction) + 1)
            .max()
            .unwrap_or(1)
            .max(1);
        (0..degree)
            .map(round_evaluation_node::<EF>)
            .map(|node| {
                self.blocks
                    .iter()
                    .map(|block| {
                        let body = if self.round < block.unused_prefix {
                            // χ_k contributes the current node; its remaining cube sum is one.
                            node * block.row_claim
                        } else {
                            // Active blocks evaluate their formal composition at this node.
                            Self::evaluate_block(
                                &self.plan,
                                block,
                                &block.public_values,
                                &self.fingerprint_weights,
                                self.offset,
                                node,
                            )
                        };
                        block.coefficient * block.prefix_evaluation * body
                    })
                    .sum()
            })
            .collect()
    }
}

fn round_evaluation_node<F: Field>(index: usize) -> F {
    // The compact wire carries h(0), h(2), ..., omitting h(1) for the sum constraint.
    F::interpolation_node(if index == 0 { 0 } else { index + 1 })
}

fn factor_degree<F: Field>(interaction: &SymbolicBusInteraction<F>) -> usize {
    // Fingerprinting is linear in payload expressions; selection multiplies by its activation.
    let payload = interaction
        .fields
        .iter()
        .map(|expression| expression.degree_multiple_with_transition(1))
        .max()
        .unwrap_or(0);
    match &interaction.activation {
        BusActivation::Always => payload,
        BusActivation::Boolean(selector) => payload + selector.degree_multiple_with_transition(1),
    }
}

fn equality_weights<F: Field>(point: &[F]) -> Vec<F> {
    // Expand lexicographically so table indices and Boolean vertices agree.
    let mut weights = alloc::vec![F::ONE];
    for &coordinate in point {
        let mut next = Vec::with_capacity(weights.len() * 2);
        for &weight in &weights {
            next.push(weight * (F::ONE - coordinate));
            next.push(weight * coordinate);
        }
        weights = next;
    }
    weights
}

fn equality_at_vertex<F: Field>(point: &[F], vertex: usize) -> F {
    point
        .iter()
        .enumerate()
        .map(|(coordinate, &challenge)| {
            let bit = (vertex >> (point.len() - 1 - coordinate)) & 1;
            if bit == 0 {
                F::ONE - challenge
            } else {
                challenge
            }
        })
        .product()
}

#[cfg(test)]
mod tests {
    use p3_air::symbolic::AirLayout;
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusSymbolicBuilder};
    use p3_field::PrimeCharacteristicRing;

    use super::{factor_degree, round_evaluation_node};

    struct TransitionBusAir;

    impl BaseAir<BabyBear> for TransitionBusAir {
        fn width(&self) -> usize {
            // One column supplies the payload used by the transition-weighted tuple.
            1
        }
    }

    impl<AB: BusInteractionBuilder<F = BabyBear>> Air<AB> for TransitionBusAir {
        fn eval(&self, builder: &mut AB) {
            // Multiplying by the transition selector exposes its round-degree contribution.
            let value: AB::Expr = builder.main().current_slice()[0].into();
            builder.push_bus_interaction(
                "transition",
                BusDirection::Push,
                [builder.is_transition() * value],
                BusActivation::Always,
            );
        }
    }

    #[test]
    fn transition_selectors_contribute_one_to_each_round_degree() {
        let profile: BusSymbolicBuilder<BabyBear> =
            BusSymbolicBuilder::from_air(&TransitionBusAir, AirLayout::from_air(&TransitionBusAir));

        assert_eq!(factor_degree(&profile.interactions()[0]), 2);
    }

    #[test]
    fn round_nodes_are_distinct_in_characteristic_two() {
        // Integer embedding would map node two back to zero in characteristic two.
        let nodes = (0..5)
            .map(round_evaluation_node::<BinaryField128>)
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(nodes[0], BinaryField128::ZERO);
        assert_ne!(nodes[1], BinaryField128::ZERO);
        for (index, node) in nodes.iter().enumerate() {
            assert!(!nodes[index + 1..].contains(node));
        }
    }
}
