//! Sumcheck binding product-GKR leaf claims to committed trace polynomials.
//!
//! For each direction, ProductGKR leaves a claim `L(q)`. This module proves
//! `L(q) - 1` equals the weighted sum of the rowwise bus factors minus one.
//! Short tables are lifted with `eq(prefix, 1^k)`, whose Boolean-cube sum is one.
//!
//! The terminal expression is checked only after its source columns open from the PCS.

use alloc::vec::Vec;

use p3_bus::{BusDirection, BusEvaluation, BusReductionOutput};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{RoundPolyInterpolator, RoundProver};
use p3_sumcheck::layout::Table;

use crate::bus::BusContext;
use crate::bus::math::{equality_at_vertex, equality_weights};

/// Prover state for the mixed-height bus composition polynomial.
pub(crate) struct BusCompositionProver<'a, F: Field, EF: ExtensionField<F>> {
    /// Checked declarations and physical layout used by every evaluation.
    context: &'a BusContext<F, EF>,
    /// Tuple equality coefficients sampled after commitment.
    fingerprint_weights: Vec<EF>,
    /// Random tuple-fingerprint shift.
    offset: EF,
    /// Folded source polynomials grouped once per AIR.
    airs: Vec<AirState<F, EF>>,
    /// Maximum round degree, derived once from the public plan.
    degree: usize,
    /// Number of global variables already bound.
    round: usize,
}

/// One bus term evaluated from an AIR's shared folded columns.
struct CompositionTerm<EF> {
    /// Coordinates of the symbolic declaration in the bus context.
    owner: p3_bus::BusBlockOwner,
    /// Named bus selecting the tuple-domain slots.
    bus: usize,
    /// Fixed coefficient from direction batching and ProductGKR block selection.
    coefficient: EF,
    /// Cube sum of the unweighted row composition before this AIR activates.
    row_claim: EF,
}

/// Folded source multilinears shared by every bus term owned by one AIR.
struct AirState<F: Field, EF: ExtensionField<F>> {
    /// Main trace column polynomials.
    main: Vec<Poly<EF>>,
    /// Preprocessed trace column polynomials.
    preprocessed: Vec<Poly<EF>>,
    /// First-row, last-row, and transition selector polynomials.
    selectors: [Poly<EF>; 3],
    /// Equality polynomial anchored at the ProductGKR row point.
    equality: Poly<EF>,
    /// Bus terms emitted by this AIR.
    terms: Vec<CompositionTerm<EF>>,
    /// Number of global prefix variables absent from this AIR table.
    unused_prefix: usize,
    /// Evaluation of the fixed all-one-vertex selector on bound prefix coordinates.
    prefix_evaluation: EF,
    /// Public inputs read by this block's symbolic expressions.
    public_values: Vec<F>,
}

impl<F, EF> AirState<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Compute every term's initial cube sum in one pass over the shared AIR columns.
    fn initialize_claims(
        &mut self,
        context: &BusContext<F, EF>,
        fingerprint_weights: &[EF],
        offset: EF,
    ) {
        let height = self.equality.as_slice().len();
        let mut main = EF::zero_vec(self.main.len());
        let mut preprocessed = EF::zero_vec(self.preprocessed.len());

        for row in 0..height {
            for (value, column) in main.iter_mut().zip(&self.main) {
                *value = column.as_slice()[row];
            }
            for (value, column) in preprocessed.iter_mut().zip(&self.preprocessed) {
                *value = column.as_slice()[row];
            }
            let evaluation = BusEvaluation {
                main: &main,
                preprocessed: &preprocessed,
                public: &self.public_values,
                is_first_row: self.selectors[0].as_slice()[row],
                is_last_row: self.selectors[1].as_slice()[row],
                is_transition: self.selectors[2].as_slice()[row],
            };
            let equality = self.equality.as_slice()[row];
            for term in &mut self.terms {
                let factor = context
                    .plan()
                    .evaluate_factor(
                        term.bus,
                        context.interaction(term.owner),
                        evaluation,
                        fingerprint_weights,
                        offset,
                    )
                    .expect("a planned expression resolves against its owning table");
                term.row_claim += equality * (factor - EF::ONE);
            }
        }
    }
}

impl<'a, F, EF> BusCompositionProver<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the formal polynomial whose cube sum must equal the ProductGKR claims.
    pub(crate) fn new(
        context: &'a BusContext<F, EF>,
        output: &BusReductionOutput<EF>,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        public_values: &[&[F]],
        direction_challenge: EF,
    ) -> Self {
        let num_variables = context.max_num_variables();
        let weights = output.challenges.fingerprint_weights();
        let mut airs = (0..tables.len()).map(|_| None).collect::<Vec<_>>();

        for direction in BusDirection::ALL {
            let direction_weight = match direction {
                BusDirection::Push => EF::ONE,
                BusDirection::Pull => direction_challenge,
            };
            for share in context.plan().terminal_shares(direction) {
                let air = share.owner.air;
                let row_point = &output.product.point[share.prefix_variables..];
                let block_weight = equality_at_vertex(
                    &output.product.point[..share.prefix_variables],
                    share.prefix_index,
                );
                let coefficient = direction_weight * block_weight;
                let state = airs[air].get_or_insert_with(|| {
                    let main = tables[air]
                        .iter_polys()
                        .map(|column| Poly::new(column.iter().copied().map(Into::into).collect()))
                        .collect::<Vec<_>>();
                    let preprocessed = preprocessed[air]
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
                    AirState {
                        main,
                        preprocessed,
                        selectors,
                        equality: Poly::new(equality_weights(row_point)),
                        terms: Vec::new(),
                        unused_prefix: num_variables - share.row_variables,
                        prefix_evaluation: EF::ONE,
                        public_values: public_values[air].to_vec(),
                    }
                });
                state.terms.push(CompositionTerm {
                    owner: share.owner,
                    bus: share.bus,
                    coefficient,
                    row_claim: EF::ZERO,
                });
            }
        }

        for air in airs.iter_mut().flatten() {
            air.initialize_claims(context, &weights, output.challenges.offset);
        }

        Self {
            context,
            fingerprint_weights: weights,
            offset: output.challenges.offset,
            airs: airs.into_iter().flatten().collect(),
            degree: context.composition_degree(),
            round: 0,
        }
    }

    fn evaluate_air(&self, air: &AirState<F, EF>, node: EF) -> EF {
        // Interpolate shared columns once, then evaluate every declaration owned by this AIR.
        let half = air.equality.as_slice().len() / 2;
        let mut main = EF::zero_vec(air.main.len());
        let mut prep = EF::zero_vec(air.preprocessed.len());
        let mut sum = EF::ZERO;
        for row in 0..half {
            let interpolate = |poly: &Poly<EF>| {
                let values = poly.as_slice();
                values[row] + (values[row + half] - values[row]) * node
            };
            for (value, polynomial) in main.iter_mut().zip(&air.main) {
                *value = interpolate(polynomial);
            }
            for (value, polynomial) in prep.iter_mut().zip(&air.preprocessed) {
                *value = interpolate(polynomial);
            }
            let evaluation = BusEvaluation {
                main: &main,
                preprocessed: &prep,
                public: &air.public_values,
                is_first_row: interpolate(&air.selectors[0]),
                is_last_row: interpolate(&air.selectors[1]),
                is_transition: interpolate(&air.selectors[2]),
            };
            let equality = interpolate(&air.equality);
            for term in &air.terms {
                let factor = self
                    .context
                    .plan()
                    .evaluate_factor(
                        term.bus,
                        self.context.interaction(term.owner),
                        evaluation,
                        &self.fingerprint_weights,
                        self.offset,
                    )
                    .expect("a planned expression resolves against its folded table");
                sum += term.coefficient * equality * (factor - EF::ONE);
            }
        }
        sum
    }
}

impl<F, EF> RoundProver<EF> for BusCompositionProver<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn fold(&mut self, challenge: EF) {
        // Dormant blocks evaluate one more coordinate of χ_k at the all-one vertex.
        for air in &mut self.airs {
            if self.round < air.unused_prefix {
                air.prefix_evaluation *= challenge;
                continue;
            }
            for polynomial in air
                .main
                .iter_mut()
                .chain(&mut air.preprocessed)
                .chain(&mut air.selectors)
                .chain(core::iter::once(&mut air.equality))
            {
                polynomial.fix_prefix_var_mut(challenge);
            }
        }
        self.round += 1;
    }

    fn round_poly(&self) -> Vec<EF> {
        // Generic-degree encoding omits node one, which the verifier derives from the claim.
        (0..self.degree)
            .map(RoundPolyInterpolator::<EF>::transmitted_node)
            .map(|node| {
                self.airs
                    .iter()
                    .map(|air| {
                        let body = if self.round < air.unused_prefix {
                            // χ_k contributes the current node; its remaining cube sum is one.
                            node * air
                                .terms
                                .iter()
                                .map(|term| term.coefficient * term.row_claim)
                                .sum::<EF>()
                        } else {
                            // Active terms share one interpolation of their AIR columns.
                            self.evaluate_air(air, node)
                        };
                        air.prefix_evaluation * body
                    })
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_air::symbolic::AirLayout;
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusSymbolicBuilder};
    use p3_field::PrimeCharacteristicRing;
    use p3_sumcheck::generic_degree::RoundPolyInterpolator;

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

        assert_eq!(
            profile.interactions()[0].factor_degree_multiple_with_transition(1),
            2
        );
    }

    #[test]
    fn round_nodes_are_distinct_in_characteristic_two() {
        // Integer embedding would map node two back to zero in characteristic two.
        let nodes = (0..5)
            .map(RoundPolyInterpolator::<BinaryField128>::transmitted_node)
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(nodes[0], BinaryField128::ZERO);
        assert_ne!(nodes[1], BinaryField128::ZERO);
        for (index, node) in nodes.iter().enumerate() {
            assert!(!nodes[index + 1..].contains(node));
        }
    }
}
