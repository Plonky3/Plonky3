#[path = "common/config.rs"]
#[allow(dead_code)]
mod config;

use config::{MyConfig, Val, make_config};
use p3_air::{Air, BaseAir, PermutationAirBuilder, WindowAccess};
use p3_batch_stark::{
    BatchVerificationError, ProverData, StarkInstance, VerificationError, prove_batch, verify_batch,
};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;

// One local lookup whose query count reads the transition selector.
//
//     columns: a (queried value), t (table value), m (table multiplicity)
//     query:   a, counted by is_transition, weight 1
//     table:   t, counted by -m
#[derive(Clone, Copy, Debug)]
struct SelectorCountAir;

impl<F> BaseAir<F> for SelectorCountAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for SelectorCountAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let (a, t, m) = (local[0], local[1], local[2]);
        let is_transition = builder.is_transition();
        builder.push_local_interaction(vec![
            (vec![a.into()], Count::bounded(is_transition, 1)),
            (vec![t.into()], Count::provided(-m.into())),
        ]);
    }
}

#[test]
fn selector_in_a_lookup_count_rejects_the_honest_proof() {
    // Fixture state: 8 rows.
    //
    //     row      a   t   m   transition (0/1)   transition (STARK)
    //     0        5   5   7   1                  1 - omega^7
    //     1 .. 6   5   0   0   1                  omega^i - omega^7
    //     7        5   0   0   0                  0
    //
    // Invariant: with 0/1 selectors the bus balances, 7 queries of 5 against one entry of multiplicity 7.
    // The debug checks therefore pass, and the prover commits fractions computed with 0/1 counts.
    // The STARK pins each fraction with its own selector values, which differ from 1 on rows 0 to 6.
    let config = make_config(1);
    let n = 8;
    let mut values = Val::zero_vec(3 * n);
    for row in 0..n {
        values[3 * row] = Val::from_u64(5);
    }
    values[1] = Val::from_u64(5);
    values[2] = Val::from_u64(7);
    let trace = RowMajorMatrix::new(values, 3);

    let air = SelectorCountAir;
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: vec![],
    }];
    let prover_data = ProverData::<MyConfig>::from_instances(&config, &instances).unwrap();
    let proof = prove_batch(&config, &instances, &prover_data).unwrap();

    // The honest proof fails the check at zeta on the only instance.
    let err = verify_batch(&config, &[air], &proof, &[vec![]], &prover_data.common).unwrap_err();
    assert!(
        matches!(
            err,
            BatchVerificationError::Verification(VerificationError::OodEvaluationMismatch {
                index: Some(0)
            })
        ),
        "expected OodEvaluationMismatch on instance 0, got {err:?}"
    );
}
