use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use proptest::prelude::*;

use super::test_utils::{
    ASCENDING_POLYS, NON_ASCENDING_POLYS, POW_BITS, ROUND_EQ_POINTS, ROUND_SEL_POINTS,
    arb_opening_schedule, arb_witness_and_schedule, drive_intermediate_and_final,
    table_shapes_from,
};
use super::{PrefixProver, SuffixProver};
use crate::SumcheckData;
use crate::layout::prover::test_utils::{
    FOLDING, build_tables, run_roundtrip_test, table_shapes, tables_from_shape,
};
use crate::layout::{Layout, Verifier};
use crate::table::OpeningBatch;
use crate::tests::*;

#[test]
fn num_claims_counts_every_recorded_opening() {
    fn run_num_claims_test_with<L>(witness: crate::layout::Witness<F>)
    where
        L: Layout<F, EF>,
    {
        let mut prover = L::from_witness(witness);
        assert_eq!(prover.num_claims(), 0);

        let mut ch = challenger();
        prover.eval(0, &OpeningBatch::new(vec![0, 1], Vec::new()), &mut ch);
        assert_eq!(prover.num_claims(), 2);

        prover.eval(1, &OpeningBatch::new(vec![0], Vec::new()), &mut ch);
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
fn eval_current_preserves_order() {
    // Invariant: returned evals follow the requested column order, not a sorted order.
    fn run_eval_current_test_with<L>()
    where
        L: Layout<F, EF>,
    {
        // Two identical provers over the same tables and folding depth.
        let mut prover = L::from_witness(L::new_witness(build_tables(), FOLDING));
        let mut reversed = L::from_witness(L::new_witness(build_tables(), FOLDING));

        // Independent transcripts seeded identically, so draws match.
        let mut prover_ch = challenger();
        let mut reversed_ch = challenger();

        // Request columns [1, 0]: evals must come back in that exact order.
        let evals = prover.eval(
            0,
            &OpeningBatch::new(vec![1, 0], Vec::new()),
            &mut prover_ch,
        );
        // Request the same columns in swapped order [0, 1].
        let reversed_evals = reversed.eval(
            0,
            &OpeningBatch::new(vec![0, 1], Vec::new()),
            &mut reversed_ch,
        );

        // Swapping the request order swaps the eval order:
        //     [eval(col 1), eval(col 0)] == reverse([eval(col 0), eval(col 1)])
        assert_eq!(
            evals.to_vec(),
            reversed_evals
                .to_vec()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
        );
        // Both record the same number of claims regardless of order.
        assert_eq!(prover.num_claims(), reversed.num_claims());
    }

    // Exercise both binding orders.
    run_eval_current_test_with::<SuffixProver<F, EF>>();
    run_eval_current_test_with::<PrefixProver<F, EF>>();
}

#[test]
fn prefix_eval_accepts_next() {
    // Invariant: a batch with only a next opening (no current openings) is accepted.
    let mut prover = PrefixProver::<F, EF>::from_witness(PrefixProver::<F, EF>::new_witness(
        build_tables(),
        FOLDING,
    ));
    let mut ch = challenger();

    // Request: current = [], next = [col 0]  → one next opening, zero current.
    let evals = prover.eval(0, &OpeningBatch::new(Vec::new(), vec![0]), &mut ch);
    // One next opening yields one eval and records one claim.
    assert_eq!(evals.len(), 1);
    assert_eq!(prover.num_claims(), 1);
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

#[test]
fn suffix_roundtrip_mixed_eq_next_requests() {
    // Invariant: a full prove/verify roundtrip agrees when batches mix
    // current and next openings, under suffix-first variable binding.
    let witness = SuffixProver::<F, EF>::new_witness(build_tables(), FOLDING);
    let shapes = table_shapes();
    let stacked_num_variables = witness.num_variables();
    // Keep a copy of the stacked polynomial to cross-check the final fold.
    let stacked_poly = witness.poly().clone();
    let strategy = SuffixProver::<F, EF>::strategy();

    // Mixed schedule: each tuple is (table, current columns, next columns).
    //     table 0: current [0], next [0, 1]
    //     table 1: current [],  next [0]
    //     table 0: current [1], next [0]
    let schedule = [
        (0, vec![0], vec![0, 1]),
        (1, vec![], vec![0]),
        (0, vec![1], vec![0]),
    ];

    // Prover side: sample points and absorb evals through the transcript.
    let mut prover_challenger = challenger();
    let mut prover_state = SuffixProver::<F, EF>::from_witness(witness);
    let opening_claims = schedule
        .iter()
        .map(|(table_idx, current, next)| {
            // Pack current and next column requests into one batch.
            let batch = OpeningBatch::new(current.clone(), next.clone());
            let evals = prover_state.eval(*table_idx, &batch, &mut prover_challenger);
            // Retain request and evals so the verifier can replay identically.
            (*table_idx, batch, evals)
        })
        .collect::<Vec<_>>();
    // One virtual claim over the full stacked polynomial.
    let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

    // First sumcheck stage folds the SVO rounds and writes their proof.
    let mut proof0 = SumcheckData::<F, EF>::default();
    let (mut prover, mut prover_randomness) =
        prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
    // The first stage consumes exactly the folding-depth rounds.
    assert_eq!(proof0.num_rounds(), FOLDING);
    assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

    // Remaining stages drive the prover to a single folded value.
    let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
        &mut prover,
        &mut prover_challenger,
        &mut prover_randomness,
        stacked_num_variables,
    );

    // Suffix binding reverses the challenge order for the direct evaluation check.
    let final_eval = stacked_poly.eval_base(&prover_randomness.reversed());
    assert_eq!(final_eval, final_folded_value);

    // Verifier side: fresh transcript, mirror every prover absorption.
    let mut verifier_challenger = challenger();
    let mut verifier = Verifier::<F, EF>::new(&shapes, strategy);
    for (table_idx, batch, evals) in opening_claims {
        verifier
            .add_claim(table_idx, &batch, &evals, &mut verifier_challenger)
            .unwrap();
    }
    verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

    // Batching challenge and the initial constraint over all recorded claims.
    let alpha = verifier_challenger.sample_algebra_element();
    let initial_constraint = verifier.constraint(alpha);
    let mut sum = EF::ZERO;
    initial_constraint.combine_evals(&mut sum);
    // The constraint's combined value must equal the alpha-batched claim sum.
    assert_eq!(sum, verifier.sum(alpha));

    // Collect each stage's constraint and replay the challenges it folded.
    let mut constraints = vec![initial_constraint];
    let mut verifier_challenge = Point::new(vec![]);
    verifier_challenge.extend(
        &proof0
            .verify_rounds(&mut verifier_challenger, &mut sum, FOLDING, 0)
            .unwrap(),
    );

    // Rebuild the intermediate-stage constraint from its transcript reads.
    let intermediate_constraint = read_constraint(
        &mut verifier_challenger,
        &intermediate_evals,
        stacked_num_variables - FOLDING,
        ROUND_EQ_POINTS,
        ROUND_SEL_POINTS,
    );
    intermediate_constraint.combine_evals(&mut sum);
    constraints.push(intermediate_constraint);
    // The grinding stage carries proof-of-work bits; the final stage none.
    verifier_challenge.extend(
        &proof1
            .verify_rounds(&mut verifier_challenger, &mut sum, FOLDING, POW_BITS)
            .unwrap(),
    );
    verifier_challenge.extend(
        &proof2
            .verify_rounds(
                &mut verifier_challenger,
                &mut sum,
                stacked_num_variables - 2 * FOLDING,
                0,
            )
            .unwrap(),
    );

    // Both sides must have folded the identical challenge vector.
    assert_eq!(prover_randomness, verifier_challenge);
    // Final identity: running sum equals folded value times the constraint weights.
    let weights = strategy
        .variable_order
        .eval_constraints_poly(&constraints, &verifier_challenge);
    assert_eq!(sum, final_folded_value * weights);
}

#[test]
fn prefix_roundtrip_mixed_eq_next_requests() {
    // Invariant: same mixed-batch roundtrip as the suffix case, but under
    // prefix-first variable binding, so no challenge reversal is needed.
    let witness = PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING);
    let shapes = table_shapes();
    let stacked_num_variables = witness.num_variables();
    let stacked_poly = witness.poly().clone();
    let strategy = PrefixProver::<F, EF>::strategy();

    // Mixed schedule: each tuple is (table, current columns, next columns).
    //     table 0: current [0], next [1]
    //     table 1: current [],  next [0]
    //     table 0: current [1], next [0]
    let schedule = [
        (0, vec![0], vec![1]),
        (1, vec![], vec![0]),
        (0, vec![1], vec![0]),
    ];

    // Prover side: sample points and absorb evals through the transcript.
    let mut prover_challenger = challenger();
    let mut prover_state = PrefixProver::<F, EF>::from_witness(witness);
    let opening_claims = schedule
        .iter()
        .map(|(table_idx, current, next)| {
            // Pack current and next column requests into one batch.
            let batch = OpeningBatch::new(current.clone(), next.clone());
            let evals = prover_state.eval(*table_idx, &batch, &mut prover_challenger);
            (*table_idx, batch, evals)
        })
        .collect::<Vec<_>>();
    let virtual_eval = prover_state.add_virtual_eval(&mut prover_challenger);

    // First sumcheck stage folds the SVO rounds and writes their proof.
    let mut proof0 = SumcheckData::<F, EF>::default();
    let (mut prover, mut prover_randomness) =
        prover_state.into_sumcheck(&mut proof0, 0, &mut prover_challenger);
    assert_eq!(proof0.num_rounds(), FOLDING);
    assert_eq!(prover.num_variables(), stacked_num_variables - FOLDING);

    // Remaining stages drive the prover to a single folded value.
    let (proof1, proof2, intermediate_evals, final_folded_value) = drive_intermediate_and_final(
        &mut prover,
        &mut prover_challenger,
        &mut prover_randomness,
        stacked_num_variables,
    );

    // Prefix binding evaluates directly at the folded challenges, no reversal.
    let final_eval = stacked_poly.eval_base(&prover_randomness);
    assert_eq!(final_eval, final_folded_value);

    // Verifier side: fresh transcript, mirror every prover absorption.
    let mut verifier_challenger = challenger();
    let mut verifier = Verifier::<F, EF>::new(&shapes, strategy);
    for (table_idx, batch, evals) in opening_claims {
        verifier
            .add_claim(table_idx, &batch, &evals, &mut verifier_challenger)
            .unwrap();
    }
    verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

    // Batching challenge and the initial constraint over all recorded claims.
    let alpha = verifier_challenger.sample_algebra_element();
    let initial_constraint = verifier.constraint(alpha);
    let mut sum = EF::ZERO;
    initial_constraint.combine_evals(&mut sum);
    // The constraint's combined value must equal the alpha-batched claim sum.
    assert_eq!(sum, verifier.sum(alpha));

    // Collect each stage's constraint and replay the challenges it folded.
    let mut constraints = vec![initial_constraint];
    let mut verifier_challenge = Point::new(vec![]);
    verifier_challenge.extend(
        &proof0
            .verify_rounds(&mut verifier_challenger, &mut sum, FOLDING, 0)
            .unwrap(),
    );

    // Rebuild the intermediate-stage constraint from its transcript reads.
    let intermediate_constraint = read_constraint(
        &mut verifier_challenger,
        &intermediate_evals,
        stacked_num_variables - FOLDING,
        ROUND_EQ_POINTS,
        ROUND_SEL_POINTS,
    );
    intermediate_constraint.combine_evals(&mut sum);
    constraints.push(intermediate_constraint);
    // The grinding stage carries proof-of-work bits; the final stage none.
    verifier_challenge.extend(
        &proof1
            .verify_rounds(&mut verifier_challenger, &mut sum, FOLDING, POW_BITS)
            .unwrap(),
    );
    verifier_challenge.extend(
        &proof2
            .verify_rounds(
                &mut verifier_challenger,
                &mut sum,
                stacked_num_variables - 2 * FOLDING,
                0,
            )
            .unwrap(),
    );

    // Both sides must have folded the identical challenge vector.
    assert_eq!(prover_randomness, verifier_challenge);
    // Final identity: running sum equals folded value times the constraint weights.
    let weights = strategy
        .variable_order
        .eval_constraints_poly(&constraints, &verifier_challenge);
    assert_eq!(sum, final_folded_value * weights);
}

#[test]
fn prefix_next_is_slot_local_not_full_stacked_next() {
    // Invariant: the repeat-last successor view must be taken per-column
    // inside its own slot, never across the interleaved stacked layout.
    //
    // Why: stacking interleaves columns, so the index "one past" in the
    // full layout lands on a neighbouring column, not the next row of one
    // column. The two successor evaluations must therefore differ.

    // Two 2-variable columns with distinct values.
    let col0 = Poly::new(vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
    ]);
    let col1 = Poly::new(vec![
        F::from_u64(13),
        F::from_u64(17),
        F::from_u64(19),
        F::from_u64(23),
    ]);

    // Interleave the two columns into one 3-variable stacked polynomial.
    //     stacked[2*i]   = col0[i]   (even slots)
    //     stacked[2*i+1] = col1[i]   (odd slots)
    let mut stacked = F::zero_vec(8);
    for local_idx in 0..4 {
        stacked[local_idx << 1] = col0.as_slice()[local_idx];
        stacked[(local_idx << 1) | 1] = col1.as_slice()[local_idx];
    }
    let stacked = Poly::new(stacked);

    // Slot-local point over the two column variables.
    let local_point = Point::new(vec![EF::from_u64(29), EF::from_u64(31)]);
    // Selector coordinate 0 lifts that point into col0's even slots.
    let selector_point = Point::new(vec![EF::ZERO]);
    let mut full_point = local_point.clone();
    full_point.extend(&selector_point);

    // Successor inside col0 alone vs successor across the stacked layout.
    let slot_local_next = col0.eval_next_base(&local_point);
    let full_stacked_next = stacked.eval_next_base(&full_point);

    // They must disagree: the stacked successor crosses into col1.
    assert_ne!(slot_local_next, full_stacked_next);
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
