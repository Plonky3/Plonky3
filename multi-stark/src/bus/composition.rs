//! Bus family of the shared AIR sumcheck, binding product-GKR leaf claims to committed traces.
//!
//! For each direction, ProductGKR leaves a claim `L(q)`. This family proves
//! `L(q) - 1` equals the weighted sum of the rowwise bus factors minus one.
//! Short tables are lifted with `eq(prefix, 1^k)`, whose Boolean-cube sum is one.
//!
//! The family runs inside the zerocheck sumcheck, over the same cube and challenges.
//! Its terminal expression is checked against the same openings the AIR constraints read.
//!
//! Round zero lifts every source column into the challenge field before any folding.
//!
//! A degree-four extension therefore holds four times the trace for the whole reduction.
//!
//! The batched zerocheck avoids that by folding packed base-field rows in its first round.

use alloc::vec::Vec;

use p3_bus::{BusDirection, BusEvaluation, BusReductionOutput};
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{RoundPolyInterpolator, RoundProver};
use p3_sumcheck::layout::Table;

use crate::bus::BusContext;

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
    /// Polynomials of the main columns this AIR's declarations read.
    main: Vec<Poly<EF>>,
    /// Column index of each of those, and the declared width they are placed into.
    main_layout: (Vec<usize>, usize),
    /// Polynomials of the preprocessed columns this AIR's declarations read.
    preprocessed: Vec<Poly<EF>>,
    /// Column index of each of those, and the declared width they are placed into.
    preprocessed_layout: (Vec<usize>, usize),
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
    ///
    /// The result is read only while the global prefix is still ahead of this table.
    fn initialize_claims(
        &mut self,
        context: &BusContext<F, EF>,
        fingerprint_weights: &[EF],
        offset: EF,
    ) {
        let height = self.equality.as_slice().len();

        // Slot placement is settled once per term, outside the row loop below.
        let factors = self
            .terms
            .iter()
            .map(|term| {
                context
                    .plan()
                    .compile_factor(
                        term.bus,
                        context.interaction(term.owner),
                        fingerprint_weights,
                        offset,
                    )
                    .expect("a checked bus plan compiles against its own declarations")
            })
            .collect::<Vec<_>>();

        let main_polys = &self.main;
        let (main_indices, main_width) = (&self.main_layout.0, self.main_layout.1);
        let fixed_polys = &self.preprocessed;
        let (fixed_indices, fixed_width) =
            (&self.preprocessed_layout.0, self.preprocessed_layout.1);
        let selectors = &self.selectors;
        let equality = &self.equality;
        let public_values = &self.public_values;
        let claims = (0..height)
            .into_par_iter()
            .par_fold_reduce(
                || {
                    (
                        EF::zero_vec(factors.len()),
                        EF::zero_vec(main_width),
                        EF::zero_vec(fixed_width),
                        Vec::new(),
                    )
                },
                |(mut claims, mut main, mut preprocessed, mut scratch), row| {
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, column) in main_indices.iter().zip(main_polys) {
                        main[index] = column.as_slice()[row];
                    }
                    for (&index, column) in fixed_indices.iter().zip(fixed_polys) {
                        preprocessed[index] = column.as_slice()[row];
                    }
                    let evaluation = BusEvaluation {
                        main: &main,
                        preprocessed: &preprocessed,
                        public: public_values,
                        periodic: &[],
                        is_first_row: selectors[0].as_slice()[row],
                        is_last_row: selectors[1].as_slice()[row],
                        is_transition: selectors[2].as_slice()[row],
                    };
                    let weight = equality.as_slice()[row];
                    for (claim, factor) in claims.iter_mut().zip(&factors) {
                        let value = factor
                            .evaluate::<EF>(&mut scratch, evaluation)
                            .expect("a planned expression resolves against its owning table");
                        *claim += weight * (value - EF::ONE);
                    }
                    (claims, main, preprocessed, scratch)
                },
                |(mut left, main, preprocessed, scratch), (right, ..)| {
                    for (claim, partial) in left.iter_mut().zip(right) {
                        *claim += partial;
                    }
                    (left, main, preprocessed, scratch)
                },
            )
            .0;

        for (term, claim) in self.terms.iter_mut().zip(claims) {
            term.row_claim = claim;
        }
    }
}

impl<'a, F, EF> BusCompositionProver<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the formal polynomial whose cube sum must equal the ProductGKR claims.
    ///
    /// # Arguments
    ///
    /// - `num_variables`: width of the shared cube, at least the tallest bus table.
    pub(crate) fn new(
        context: &'a BusContext<F, EF>,
        output: &BusReductionOutput<EF>,
        tables: &[&Table<F>],
        preprocessed: &[Option<&Table<F>>],
        public_values: &[&[F]],
        direction_challenge: EF,
        num_variables: usize,
    ) -> Self {
        debug_assert!(num_variables >= context.max_num_variables());
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
                let block_weight = share
                    .prefix_weight(&output.product.point)
                    .expect("a planned share addresses its own product-tree point");
                let coefficient = direction_weight * block_weight;
                let state = airs[air].get_or_insert_with(|| {
                    // Column views read a packed Boolean table without expanding it first.
                    // Only the columns a declaration reads are lifted, and then folded.
                    let main_columns = context.main_columns(air);
                    let main = main_columns
                        .iter()
                        .map(|&column| {
                            Poly::new(
                                tables[air]
                                    .column(column)
                                    .values()
                                    .map(Into::into)
                                    .collect(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let fixed_columns = context.preprocessed_columns(air);
                    let fixed_width = preprocessed[air].map_or(0, Table::num_polys);
                    let preprocessed = preprocessed[air]
                        .iter()
                        .flat_map(|table| {
                            fixed_columns.iter().map(|&column| {
                                Poly::new(table.column(column).values().map(Into::into).collect())
                            })
                        })
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
                        main_layout: (main_columns.to_vec(), tables[air].num_polys()),
                        preprocessed,
                        preprocessed_layout: (fixed_columns.to_vec(), fixed_width),
                        selectors,
                        equality: Poly::new(Point::new(row_point).equality_weights_msb()),
                        terms: Vec::new(),
                        unused_prefix: num_variables - share.row_variables,
                        prefix_evaluation: EF::ONE,
                        public_values: public_values[air].to_vec(),
                    }
                });
                // Row geometry is captured from the first share and reused by every later one.
                debug_assert_eq!(
                    state.unused_prefix,
                    num_variables - share.row_variables,
                    "every block of one AIR shares its trace height"
                );
                state.terms.push(CompositionTerm {
                    owner: share.owner,
                    bus: share.bus,
                    coefficient,
                    row_claim: EF::ZERO,
                });
            }
        }

        // A table as tall as the statement never reads its own claim, so it never pays for one.
        for air in airs
            .iter_mut()
            .flatten()
            .filter(|air| air.unused_prefix > 0)
        {
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
        // Slot placement is settled once per term, outside the row loop below.
        let factors = air
            .terms
            .iter()
            .map(|term| {
                self.context
                    .plan()
                    .compile_factor(
                        term.bus,
                        self.context.interaction(term.owner),
                        &self.fingerprint_weights,
                        self.offset,
                    )
                    .expect("a checked bus plan compiles against its own declarations")
            })
            .collect::<Vec<_>>();

        // Interpolate shared columns once, then evaluate every declaration owned by this AIR.
        let half = air.equality.as_slice().len() / 2;
        (0..half)
            .into_par_iter()
            .map_init(
                || {
                    (
                        EF::zero_vec(air.main_layout.1),
                        EF::zero_vec(air.preprocessed_layout.1),
                        Vec::new(),
                    )
                },
                |(main, prep, scratch), row| {
                    let interpolate = |poly: &Poly<EF>| {
                        let values = poly.as_slice();
                        values[row] + (values[row + half] - values[row]) * node
                    };
                    // Unread columns keep their zero, which no planned expression names.
                    for (&index, polynomial) in air.main_layout.0.iter().zip(&air.main) {
                        main[index] = interpolate(polynomial);
                    }
                    for (&index, polynomial) in
                        air.preprocessed_layout.0.iter().zip(&air.preprocessed)
                    {
                        prep[index] = interpolate(polynomial);
                    }
                    let evaluation = BusEvaluation {
                        main,
                        preprocessed: prep,
                        public: &air.public_values,
                        periodic: &[],
                        is_first_row: interpolate(&air.selectors[0]),
                        is_last_row: interpolate(&air.selectors[1]),
                        is_transition: interpolate(&air.selectors[2]),
                    };
                    let equality = interpolate(&air.equality);
                    air.terms
                        .iter()
                        .zip(&factors)
                        .map(|(term, factor)| {
                            let value = factor
                                .evaluate::<EF>(scratch, evaluation)
                                .expect("a planned expression resolves against its folded table");
                            term.coefficient * equality * (value - EF::ONE)
                        })
                        .sum::<EF>()
                },
            )
            .sum()
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
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName, BusSymbolicBuilder};
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
                BusName::new("transition"),
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
