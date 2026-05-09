//! Provers for the stacked layout, specialised per residual-sumcheck binding mode.
//!
//! # Modules
//!
//! - Prefix prover: SIMD-packed first round.
//! - Suffix prover: SVO-accumulator preprocessing.

mod prefix;
mod suffix;

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::point::Point;
pub use prefix::PrefixProver;
pub use suffix::SuffixProver;

use crate::sumcheck::SumcheckData;
use crate::sumcheck::layout::{LayoutStrategy, Table, Witness};
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};

/// Stacked-sumcheck prover layout
pub trait Layout<F: TwoAdicField, EF: ExtensionField<F>>: Sized {
    /// Builds this layout from a committed witness.
    fn from_witness(witness: Witness<F>) -> Self;

    /// Builds a witness structure for this layout from source tables.
    fn new_witness(tables: Vec<Table<F>>, folding: usize) -> Witness<F>;

    /// Commits to the witness and returns the layout.
    ///
    /// # Arguments
    ///
    /// - `dft`                    — base-field DFT used to encode the codeword.
    /// - `mmcs`                   — Merkle commitment scheme over the base field.
    /// - `challenger`             — Fiat–Shamir transcript; absorbs the Merkle root.
    /// - `witness`                — stacked committed polynomial plus its tables.
    /// - `folding`                — folding factor consumed by the first WHIR round.
    /// - `starting_log_inv_rate`  — initial log-inverse rate of the RS code.
    fn commit<Dft, MT, Challenger>(
        dft: &Dft,
        mmcs: &MT,
        challenger: &mut Challenger,
        witness: Witness<F>,
        folding: usize,
        starting_log_inv_rate: usize,
    ) -> (Self, MT::Commitment, MT::ProverData<DenseMatrix<F>>)
    where
        Dft: TwoAdicSubgroupDft<F>,
        MT: Mmcs<F>,
        Challenger: CanObserve<MT::Commitment>;

    /// Returns the total number of concrete openings recorded so far.
    fn num_claims(&self) -> usize;

    /// Returns the verifier strategy required to replay this committed layout.
    fn strategy() -> LayoutStrategy;

    /// Returns the variable order.
    fn variable_order() -> VariableOrder {
        Self::strategy().variable_order
    }

    /// Returns the number of variables of first round
    fn folding(&self) -> usize;

    /// Returns the number of variables of the stacked polynomial.
    fn num_variables(&self) -> usize;

    /// Returns the number of variables of table `id`.
    fn num_variables_table(&self, id: usize) -> usize;

    /// Records opening claims for the selected columns of `table_idx`.
    fn eval<Ch>(&mut self, table_idx: usize, polys: &[usize], challenger: &mut Ch) -> Vec<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;

    /// Samples a virtual evaluation on the full stacked polynomial.
    fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;

    /// Processes initial rounds of sumcheck and returns the residual sumcheck prover.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the unpacked product polynomial.
    /// - Folding challenges sampled during preprocessing.
    fn into_sumcheck<Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SumcheckProver<F, EF>, Point<EF>)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>;
}

#[cfg(test)]
pub(super) mod test_utils {
    //! Shared fixtures and end-to-end roundtrip runners used by both mode tests.
    //!
    //! - Fixed two-table witness reused by every scenario.
    //! - Per-mode runners that drive prover and verifier side-by-side.
    //! - Proptest strategy for random opening schedules.

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_challenger::FieldChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::sumcheck::SumcheckData;
    use crate::sumcheck::layout::{Layout, LayoutStrategy, Table, TableShape, Verifier, Witness};
    use crate::sumcheck::strategy::VariableOrder;
    use crate::sumcheck::tests::*;

    /// Preprocessing rounds each end-to-end test consumes on both sides.
    pub(crate) const FOLDING: usize = 4;
    /// Proof-of-work difficulty for the intermediate sumcheck phase.
    pub(crate) const POW_BITS: usize = 10;
    /// STIR-style constraint points sampled for the intermediate round.
    pub(crate) const ROUND_EQ_POINTS: usize = 10;
    /// STIR-style selector points sampled for the intermediate round.
    pub(crate) const ROUND_SEL_POINTS: usize = 100;

    /// Opening schedule with columns in ascending order inside each claim.
    pub(crate) const ASCENDING_POLYS: &[(usize, &[usize])] =
        &[(0, &[0, 1]), (0, &[0]), (1, &[0, 1]), (1, &[1])];

    /// Opening schedule with non-ascending columns in the first claim.
    ///
    /// Targets the alpha / partial-eval misalignment that used to bite suffix mode.
    pub(crate) const NON_ASCENDING_POLYS: &[(usize, &[usize])] =
        &[(0, &[1, 0]), (0, &[0]), (1, &[0, 1]), (1, &[1])];

    /// Builds the fixed two-table witness shared by every roundtrip test.
    ///
    /// # Layout
    ///
    /// - Table 0: arity 9, two columns.
    /// - Table 1: arity 10, two columns.
    pub(crate) fn build_tables() -> Vec<Table<F>> {
        let mut rng = SmallRng::seed_from_u64(1);
        // Table at index 1 in the insertion order: arity 10, two columns.
        let a0 = Poly::<F>::rand(&mut rng, 10);
        let a1 = Poly::<F>::rand(&mut rng, 10);
        // Table at index 0 in the insertion order: arity 9, two columns.
        let b0 = Poly::<F>::rand(&mut rng, 9);
        let b1 = Poly::<F>::rand(&mut rng, 9);
        vec![Table::new(vec![b0, b1]), Table::new(vec![a0, a1])]
    }

    /// Returns the per-table shape used by the verifier side.
    pub(crate) fn table_shapes() -> Vec<TableShape> {
        vec![TableShape::new(9, 2), TableShape::new(10, 2)]
    }

    /// Proptest strategy: random opening schedules over the fixed witness.
    ///
    /// # Shape
    ///
    /// - 1..=6 calls in total.
    /// - Each call picks a random table index (0 or 1) and a non-empty
    ///   subset of its 2 columns, possibly in any order.
    pub(crate) fn arb_opening_schedule() -> impl Strategy<Value = Vec<(usize, Vec<usize>)>> {
        // One call: (table_idx in {0, 1}, random non-empty permutation of [0, 1]).
        let one_call =
            (0usize..2, prop::collection::vec(0usize..2, 1..=2)).prop_map(|(table_idx, polys)| {
                // Deduplicate while preserving first-seen order so the opening
                // list stays valid (columns open at most once per claim).
                let mut seen = [false; 2];
                let dedup: Vec<usize> = polys
                    .into_iter()
                    .filter(|&p| {
                        let first = !seen[p];
                        seen[p] = true;
                        first
                    })
                    .collect();
                (table_idx, dedup)
            });
        prop::collection::vec(one_call, 1..=6)
    }

    /// Runs the verifier side and the final batched-sum check.
    ///
    /// Mode-agnostic: the binding direction is passed in as a runtime tag used
    /// only to evaluate the batched-constraint polynomial. Every caller-provided
    /// value comes straight from the prover side.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_roundtrip(
        strategy: LayoutStrategy,
        shapes: &[TableShape],
        stacked_num_variables: usize,
        opening_claims: Vec<(usize, Vec<usize>, Vec<EF>)>,
        virtual_eval: EF,
        proof0: &SumcheckData<F, EF>,
        proof1: &SumcheckData<F, EF>,
        proof2: &SumcheckData<F, EF>,
        intermediate_evals: &[EF],
        expected_randomness: &Point<EF>,
        final_folded_value: EF,
    ) {
        // Fresh challenger: verifier must stay in lockstep with the prover transcript.
        let mut verifier_challenger = challenger();
        let mut verifier: Verifier<F, EF> = Verifier::new(shapes, strategy);

        // Re-sample the same opening points and record the claimed evaluations.
        // `add_claim` samples the point + absorbs the evals internally, mirroring
        // the prover's `eval`.
        for (table_idx, polys, evals) in opening_claims {
            verifier.add_claim(table_idx, &polys, &evals, &mut verifier_challenger);
        }

        // Re-sample the virtual claim too; mirrors the prover's `add_virtual_eval`.
        verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

        // Sample the batching challenge, build the initial constraint, seed the running sum.
        let alpha = verifier_challenger.sample_algebra_element();
        let initial_constraint = verifier.constraint(alpha);
        let mut sum = EF::ZERO;
        initial_constraint.combine_evals(&mut sum);
        assert_eq!(sum, verifier.sum(alpha));

        // Drive the three-proof transcript verification.
        let mut constraints = vec![initial_constraint];
        let mut verifier_challenge = Point::new(vec![]);
        verifier_challenge.extend(
            &proof0
                .verify_rounds(&mut verifier_challenger, &mut sum, 0)
                .unwrap(),
        );

        let intermediate_constraint = read_constraint(
            &mut verifier_challenger,
            intermediate_evals,
            stacked_num_variables - FOLDING,
            ROUND_EQ_POINTS,
            ROUND_SEL_POINTS,
        );
        intermediate_constraint.combine_evals(&mut sum);
        constraints.push(intermediate_constraint);
        verifier_challenge.extend(
            &proof1
                .verify_rounds(&mut verifier_challenger, &mut sum, POW_BITS)
                .unwrap(),
        );
        verifier_challenge.extend(
            &proof2
                .verify_rounds(&mut verifier_challenger, &mut sum, 0)
                .unwrap(),
        );

        // Final invariants:
        //     - Prover and verifier agreed on the same randomness.
        //     - Batched sum equals the final folded value times the batched weights.
        assert_eq!(expected_randomness, &verifier_challenge);
        let weights = strategy
            .variable_order
            .eval_constraints_poly(&constraints, &verifier_challenge);
        assert_eq!(sum, final_folded_value * weights);
    }

    /// Drives the intermediate + final sumcheck phases on the residual prover.
    ///
    /// Returns the two extra transcript chunks, the intermediate evals, and
    /// the final folded value. Extends `prover_randomness` with every
    /// challenge sampled during these two phases.
    pub(crate) fn drive_intermediate_and_final(
        prover: &mut crate::sumcheck::strategy::SumcheckProver<F, EF>,
        prover_challenger: &mut MyChallenger,
        prover_randomness: &mut Point<EF>,
        stacked_num_variables: usize,
    ) -> (SumcheckData<F, EF>, SumcheckData<F, EF>, Vec<EF>, EF) {
        // Intermediate phase: build a STIR-style constraint and absorb it.
        let mut intermediate_evals: Vec<EF> = Vec::new();
        let constraint = make_constraint_ext(
            prover_challenger,
            &mut intermediate_evals,
            prover.num_variables(),
            ROUND_EQ_POINTS,
            ROUND_SEL_POINTS,
            &prover.evals(),
        );

        let mut proof1 = SumcheckData::<F, EF>::default();
        prover_randomness.extend(&prover.compute_sumcheck_polynomials(
            &mut proof1,
            prover_challenger,
            FOLDING,
            POW_BITS,
            Some(constraint),
        ));
        let remaining_vars = stacked_num_variables - FOLDING - FOLDING;
        assert_eq!(proof1.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), remaining_vars);

        // Final phase: fold the remaining variables down to a constant.
        let mut proof2 = SumcheckData::<F, EF>::default();
        prover_randomness.extend(&prover.compute_sumcheck_polynomials(
            &mut proof2,
            prover_challenger,
            remaining_vars,
            0,
            None,
        ));
        assert_eq!(proof2.num_rounds(), remaining_vars);
        assert_eq!(prover.num_variables(), 0);

        let final_folded_value = prover.evals().as_constant().unwrap();
        (proof1, proof2, intermediate_evals, final_folded_value)
    }

    /// Replays an opening schedule against a prover, returning the
    /// `(table_idx, polys, evals)` triples the verifier will reconstruct.
    ///
    /// The prover's `eval` internally samples the point and absorbs the
    /// returned evaluations, keeping the challenger in lockstep with the
    /// verifier's `add_claim` replay on the other side.
    fn replay_schedule<F>(
        calls: &[(usize, &[usize])],
        mut step: F,
    ) -> Vec<(usize, Vec<usize>, Vec<EF>)>
    where
        F: FnMut(usize, &[usize]) -> Vec<EF>,
    {
        calls
            .iter()
            .map(|&(table_idx, polys)| (table_idx, polys.to_vec(), step(table_idx, polys)))
            .collect()
    }

    /// Runs the full mode-generic roundtrip with verifier metadata from the layout.
    pub(crate) fn run_roundtrip_test<L>(
        witness: Witness<F>,
        shapes: &[TableShape],
        calls: &[(usize, &[usize])],
    ) where
        L: Layout<F, EF>,
    {
        let mut prover_challenger = challenger();
        let stacked_num_variables = witness.num_variables();
        // Snapshot the stacked polynomial before the witness is consumed.
        let stacked_poly = witness.poly().clone();

        // Prover: build the selected layout, record openings, add a virtual claim.
        let mut prover_state = L::from_witness(witness);
        let strategy = L::strategy();
        let order = strategy.variable_order;
        let opening_claims = replay_schedule(calls, |t, polys| {
            prover_state.eval(t, polys, &mut prover_challenger)
        });
        let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

        // Preprocessing: consume FOLDING rounds, hand off the residual prover.
        let mut proof0 = SumcheckData::<F, EF>::default();
        let (mut prover, mut prover_randomness) =
            prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
        assert_eq!(proof0.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

        // Intermediate + final rounds (mode-agnostic once the residual prover exists).
        let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
            &mut prover,
            &mut prover_challenger,
            &mut prover_randomness,
            stacked_num_variables,
        );

        let final_eval = match order {
            VariableOrder::Prefix => stacked_poly.eval_base(&prover_randomness),
            VariableOrder::Suffix => stacked_poly.eval_base(&prover_randomness.reversed()),
        };
        assert_eq!(final_eval, final_folded_value);

        verify_roundtrip(
            strategy,
            shapes,
            stacked_num_variables,
            opening_claims,
            virtual_eval,
            &proof0,
            &proof1,
            &proof2,
            &intermediate_evals,
            &prover_randomness,
            final_folded_value,
        );
    }

    /// Minimum source-table arity used by the shape proptest.
    ///
    /// # Constraints
    ///
    /// - Must be at least the preprocessing depth (every table arity >= FOLDING).
    /// - Must be at least `log2(packing_width)` so prefix mode accepts it.
    ///   BabyBear packing widths on current targets peak at 16, giving `k_pack = 4`.
    const SHAPE_MIN_ARITY: usize = 4;

    /// Upper bound on per-table arity in the shape proptest.
    const SHAPE_MAX_ARITY: usize = 8;

    /// Upper bound on the number of columns per table.
    const SHAPE_MAX_COLUMNS: usize = 3;

    /// Upper bound on the number of tables per witness.
    const SHAPE_MAX_TABLES: usize = 3;

    /// Upper bound on the number of opening calls in the generated schedule.
    const SHAPE_MAX_CALLS: usize = 5;

    /// One `(arity, column_count)` pair per source table.
    pub(crate) type WitnessShape = Vec<(usize, usize)>;

    /// One `(table_index, column_subset)` pair per opening call.
    pub(crate) type OpeningSchedule = Vec<(usize, Vec<usize>)>;

    /// Builds a deterministic witness matching the given `(arity, column_count)` shape.
    ///
    /// # Arguments
    ///
    /// - `shape` — one `(arity, column_count)` pair per source table.
    pub(crate) fn tables_from_shape(shape: &[(usize, usize)]) -> Vec<Table<F>> {
        // Fixed seed: every proptest case gets reproducible polynomial evaluations.
        let mut rng = SmallRng::seed_from_u64(42);

        // One table per (arity, column_count) pair; each column is a random polynomial.
        shape
            .iter()
            .map(|&(arity, num_cols)| {
                let polys: Vec<Poly<F>> = (0..num_cols)
                    .map(|_| Poly::<F>::rand(&mut rng, arity))
                    .collect();
                Table::new(polys)
            })
            .collect()
    }

    /// Mirrors a `(arity, column_count)` shape onto the verifier-side table shapes.
    pub(crate) fn table_shapes_from(shape: &[(usize, usize)]) -> Vec<TableShape> {
        shape
            .iter()
            .map(|&(arity, num_cols)| TableShape::new(arity, num_cols))
            .collect()
    }

    /// Proptest strategy: random witness shape paired with a valid opening schedule.
    ///
    /// # Shape
    ///
    /// - 1..=`SHAPE_MAX_TABLES` source tables.
    /// - Each table: arity in `[SHAPE_MIN_ARITY, SHAPE_MAX_ARITY]`, column count
    ///   in `[1, SHAPE_MAX_COLUMNS]`.
    ///
    /// # Schedule
    ///
    /// - 1..=`SHAPE_MAX_CALLS` opening calls over the generated witness.
    /// - Every call picks an existing table index and a non-empty, de-duplicated
    ///   subset of that table's columns (columns may appear in any order).
    pub(crate) fn arb_witness_and_schedule()
    -> impl Strategy<Value = (WitnessShape, OpeningSchedule)> {
        // Step 1: pick the witness shape.
        //
        // Stacked arity must accommodate two phases of FOLDING rounds plus the
        // final fold-to-constant phase, so total size must exceed `2^(2 * FOLDING - 1)`.
        let shape = prop::collection::vec(
            (
                SHAPE_MIN_ARITY..=SHAPE_MAX_ARITY,
                1usize..=SHAPE_MAX_COLUMNS,
            ),
            1..=SHAPE_MAX_TABLES,
        )
        .prop_filter(
            "stacked polynomial must support two phases of FOLDING rounds",
            |shape| {
                let total: usize = shape.iter().map(|&(a, c)| (1usize << a) * c).sum();
                // log2_ceil(total) >= 2 * FOLDING.
                total > (1usize << (2 * FOLDING - 1))
            },
        );

        shape.prop_flat_map(|shape| {
            // Step 2: given the shape, pick a schedule that respects it.
            let num_tables = shape.len();
            let shape_for_calls = shape.clone();

            // One call: pick a table, then a non-empty unique column subset.
            let one_call = (0..num_tables).prop_flat_map(move |t_idx| {
                let num_cols = shape_for_calls[t_idx].1;
                // Draw a random column sequence of length 1..=num_cols, then deduplicate
                // while preserving first-seen order so the opening list stays valid.
                prop::collection::vec(0..num_cols, 1..=num_cols).prop_map(move |cols| {
                    let mut seen = vec![false; num_cols];
                    let dedup: Vec<usize> = cols
                        .into_iter()
                        .filter(|&c| {
                            let first = !seen[c];
                            seen[c] = true;
                            first
                        })
                        .collect();
                    (t_idx, dedup)
                })
            });

            // Step 3: bundle the schedule back with its originating shape.
            prop::collection::vec(one_call, 1..=SHAPE_MAX_CALLS)
                .prop_map(move |sched| (shape.clone(), sched))
        })
    }
}

#[cfg(test)]
mod tests {

    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::test_utils::{
        ASCENDING_POLYS, NON_ASCENDING_POLYS, arb_opening_schedule, arb_witness_and_schedule,
        table_shapes_from,
    };
    use super::{PrefixProver, SuffixProver};
    use crate::sumcheck::layout::Layout;
    use crate::sumcheck::layout::prover::test_utils::{
        FOLDING, build_tables, run_roundtrip_test, table_shapes, tables_from_shape,
    };
    use crate::sumcheck::tests::*;

    #[test]
    fn num_claims_counts_every_recorded_opening() {
        fn run_num_claims_test_with<L>(witness: crate::sumcheck::layout::Witness<F>)
        where
            L: Layout<F, EF>,
        {
            let mut prover = L::from_witness(witness);
            assert_eq!(prover.num_claims(), 0);

            let mut ch = challenger();
            prover.eval(0, &[0, 1], &mut ch);
            assert_eq!(prover.num_claims(), 2);

            prover.eval(1, &[0], &mut ch);
            assert_eq!(prover.num_claims(), 3);
        }

        run_num_claims_test_with::<SuffixProver<F, EF>>(SuffixProver::<F, EF>::new_witness(
            build_tables(),
            FOLDING,
        ));
        run_num_claims_test_with::<PrefixProver<F, EF>>(PrefixProver::<F, EF>::new_witness(
            build_tables(),
            FOLDING,
        ));
    }

    #[test]
    fn roundtrip_ascending_polys() {
        run_roundtrip_test::<PrefixProver<F, EF>>(
            PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            ASCENDING_POLYS,
        );

        run_roundtrip_test::<SuffixProver<F, EF>>(
            SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            ASCENDING_POLYS,
        );
    }

    #[test]
    fn roundtrip_non_ascending_polys() {
        run_roundtrip_test::<PrefixProver<F, EF>>(
            PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            NON_ASCENDING_POLYS,
        );

        run_roundtrip_test::<SuffixProver<F, EF>>(
            SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING),
            &table_shapes(),
            NON_ASCENDING_POLYS,
        );
    }

    fn run_shape_test<L>(shape: &[(usize, usize)], schedule: &[(usize, Vec<usize>)])
    where
        L: Layout<F, EF>,
    {
        let witness = L::new_witness(tables_from_shape(shape), FOLDING);
        let shapes = table_shapes_from(shape);
        let borrowed: Vec<(usize, &[usize])> = schedule
            .iter()
            .map(|(t, polys)| (*t, polys.as_slice()))
            .collect();
        run_roundtrip_test::<L>(witness, &shapes, &borrowed);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        // Invariant:
        //     Every valid opening schedule over the fixed two-table witness
        //     roundtrips through the protocol without prover/verifier divergence.
        //
        // Coverage: includes non-ascending column orders that previously exposed
        // alpha / partial-eval alignment bugs.
        #[test]
        fn roundtrip_proptest(schedule in arb_opening_schedule()) {
            let borrowed: Vec<(usize, &[usize])> = schedule
                .iter()
                .map(|(t, polys)| (*t, polys.as_slice()))
                .collect();

            run_roundtrip_test::<PrefixProver<F, EF>>(
                PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING),
                &table_shapes(),
                &borrowed,
            );
            run_roundtrip_test::<SuffixProver<F, EF>>(
                SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING),
                &table_shapes(),
                &borrowed,
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]

        // Invariant:
        //     Roundtrip agreement holds for valid generated witness shapes, not
        //     only the fixed two-table fixture.
        #[test]
        fn roundtrip_shape_proptest((shape, schedule) in arb_witness_and_schedule()) {
            run_shape_test::<PrefixProver<F, EF>>(&shape, &schedule);
            run_shape_test::<SuffixProver<F, EF>>(&shape, &schedule);
        }
    }
}
