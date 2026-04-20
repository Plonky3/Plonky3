//! Provers for the stacked layout, specialised per residual-sumcheck binding mode.
//!
//! # Modules
//!
//! - Prefix prover: SIMD-packed first round.
//! - Suffix prover: SVO-accumulator preprocessing.

mod prefix;
mod suffix;

pub use prefix::PrefixProver;
pub use suffix::SuffixProver;

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

    use super::{PrefixProver, SuffixProver};
    use crate::sumcheck::SumcheckData;
    use crate::sumcheck::layout::{Table, TableShape, Verifier, Witness};
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
    pub(crate) fn build_witness() -> Witness<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        // Table at index 1 in the insertion order: arity 10, two columns.
        let a0 = Poly::<F>::rand(&mut rng, 10);
        let a1 = Poly::<F>::rand(&mut rng, 10);
        // Table at index 0 in the insertion order: arity 9, two columns.
        let b0 = Poly::<F>::rand(&mut rng, 9);
        let b1 = Poly::<F>::rand(&mut rng, 9);
        Witness::new(
            vec![Table::new(vec![b0, b1]), Table::new(vec![a0, a1])],
            FOLDING,
        )
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
        order: VariableOrder,
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
        let mut verifier: Verifier<F, EF> = Verifier::new(shapes);

        // Re-sample the same opening points and record the claimed evaluations.
        for (table_idx, polys, evals) in opening_claims {
            let point = Point::expand_from_univariate(
                verifier_challenger.sample_algebra_element(),
                verifier.num_variables_table(table_idx),
            );
            verifier_challenger.observe_algebra_slice(&evals);
            verifier.add_claim(table_idx, point, &polys, &evals);
        }

        // Re-sample the virtual claim too.
        let virtual_point = Point::expand_from_univariate(
            verifier_challenger.sample_algebra_element(),
            stacked_num_variables,
        );
        verifier_challenger.observe_algebra_element(virtual_eval);
        verifier.add_virtual_eval(virtual_point, virtual_eval);

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
        let weights = order.eval_constraints_poly(&constraints, &verifier_challenge);
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
            &prover.poly(),
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

        let final_folded_value = prover.poly().as_constant().unwrap();
        (proof1, proof2, intermediate_evals, final_folded_value)
    }

    /// Replays an opening schedule against a prover, returning the
    /// `(table_idx, polys, evals)` triples the verifier will reconstruct.
    ///
    /// Mutates the challenger in lockstep with what the prover observes,
    /// so both sides stay in sync when the verifier replays the transcript.
    fn replay_schedule<F>(
        calls: &[(usize, &[usize])],
        challenger: &mut MyChallenger,
        mut step: F,
    ) -> Vec<(usize, Vec<usize>, Vec<EF>)>
    where
        F: FnMut(&mut MyChallenger, usize, &[usize]) -> Vec<EF>,
    {
        calls
            .iter()
            .map(|&(table_idx, polys)| {
                let evals = step(challenger, table_idx, polys);
                challenger.observe_algebra_slice(&evals);
                (table_idx, polys.to_vec(), evals)
            })
            .collect()
    }

    /// Runs the full prefix-mode roundtrip against a caller-supplied witness.
    pub(crate) fn run_prefix_roundtrip_with(
        witness: Witness<F>,
        shapes: &[TableShape],
        calls: &[(usize, &[usize])],
    ) {
        let mut prover_challenger = challenger();
        let stacked_num_variables = witness.num_variables();

        // Prover: build prefix mode, record openings, add a virtual claim.
        let mut prover_state: PrefixProver<F, EF> = witness.as_prefix_prover();
        let stacked_poly = prover_state.poly().clone();
        let opening_claims = replay_schedule(calls, &mut prover_challenger, |ch, t, polys| {
            let point = Point::expand_from_univariate(
                ch.sample_algebra_element(),
                prover_state.num_variables_table(t),
            );
            prover_state.eval(&point, t, polys)
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

        // Prefix mode binds variables in order: evaluate directly at the folded point.
        assert_eq!(
            stacked_poly.eval_base(&prover_randomness),
            final_folded_value,
        );

        verify_roundtrip(
            VariableOrder::Prefix,
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

    /// Runs the full suffix-mode roundtrip against a caller-supplied witness.
    pub(crate) fn run_suffix_roundtrip_with(
        witness: Witness<F>,
        shapes: &[TableShape],
        calls: &[(usize, &[usize])],
    ) {
        let mut prover_challenger = challenger();
        let stacked_num_variables = witness.num_variables();

        let mut prover_state: SuffixProver<F, EF> = witness.as_suffix_prover();
        let stacked_poly = prover_state.poly().clone();
        let opening_claims = replay_schedule(calls, &mut prover_challenger, |ch, t, polys| {
            let point = Point::expand_from_univariate(
                ch.sample_algebra_element(),
                prover_state.num_variables_table(t),
            );
            prover_state.eval(&point, t, polys)
        });
        let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

        let mut proof0 = SumcheckData::<F, EF>::default();
        let (mut prover, mut prover_randomness) =
            prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
        assert_eq!(proof0.num_rounds(), FOLDING);
        assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

        let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
            &mut prover,
            &mut prover_challenger,
            &mut prover_randomness,
            stacked_num_variables,
        );

        // Suffix mode binds variables in reverse: evaluate at the reversed folded point.
        assert_eq!(
            stacked_poly.eval_base(&prover_randomness.reversed()),
            final_folded_value,
        );

        verify_roundtrip(
            VariableOrder::Suffix,
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

    /// Thin shim: runs the prefix-mode roundtrip on the shared fixed-shape witness.
    pub(crate) fn run_prefix_roundtrip(calls: &[(usize, &[usize])]) {
        run_prefix_roundtrip_with(build_witness(), &table_shapes(), calls);
    }

    /// Thin shim: runs the suffix-mode roundtrip on the shared fixed-shape witness.
    pub(crate) fn run_suffix_roundtrip(calls: &[(usize, &[usize])]) {
        run_suffix_roundtrip_with(build_witness(), &table_shapes(), calls);
    }

    /// Minimum source-table arity used by the shape proptest.
    ///
    /// # Constraints
    ///
    /// - Must exceed the preprocessing depth (every table arity > FOLDING).
    /// - Must be at least `log2(packing_width)` so prefix mode accepts it.
    ///   BabyBear packing widths on current targets peak at 16, giving `k_pack = 4`.
    const SHAPE_MIN_ARITY: usize = 5;

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
    pub(crate) fn build_witness_from_shape(shape: &[(usize, usize)]) -> Witness<F> {
        // Fixed seed: every proptest case gets reproducible polynomial evaluations.
        let mut rng = SmallRng::seed_from_u64(42);

        // One table per (arity, column_count) pair; each column is a random polynomial.
        let tables: Vec<Table<F>> = shape
            .iter()
            .map(|&(arity, num_cols)| {
                let polys: Vec<Poly<F>> = (0..num_cols)
                    .map(|_| Poly::<F>::rand(&mut rng, arity))
                    .collect();
                Table::new(polys)
            })
            .collect();

        Witness::new(tables, FOLDING)
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
