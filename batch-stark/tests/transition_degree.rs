// The shared test configuration; this file uses only the default one.
#[allow(dead_code)]
#[path = "common/config.rs"]
mod config;

use config::{Val, make_config};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

// Asserts a^2 = b on transition rows, under one or two transition selectors.
#[derive(Clone, Copy)]
struct SquareAir {
    nested: bool,
}

impl<F> BaseAir<F> for SquareAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilder> Air<AB> for SquareAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let a = main.current(0).unwrap();
        let b = main.current(1).unwrap();
        let mut outer = builder.when_transition();
        if self.nested {
            outer.when_transition().assert_eq(a * a, b);
        } else {
            outer.assert_eq(a * a, b);
        }
    }
}

// Rows (i + 1, (i + 1)^2): every row satisfies a^2 = b.
fn trace(rows: usize) -> RowMajorMatrix<Val> {
    let values = (1..=rows as u32)
        .flat_map(|i| {
            let a = Val::from_u32(i);
            [a, a.square()]
        })
        .collect();
    RowMajorMatrix::new(values, 2)
}

fn prove_and_verify(air: SquareAir) {
    let config = make_config(7);
    let trace = trace(8);
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: vec![],
    }];
    let prover_data = ProverData::from_instances(&config, &instances).unwrap();
    let proof = prove_batch(&config, &instances, &prover_data).unwrap();
    verify_batch(&config, &[air], &proof, &[vec![]], &prover_data.common).unwrap();
}

#[test]
fn one_transition_factor_verifies() {
    // Fixture state: 8 rows, constraint T(X) * (a(X)^2 - b(X)), degree 1 + 14 = 15.
    prove_and_verify(SquareAir { nested: false });
}

#[test]
fn two_transition_factors_verify() {
    // Fixture state: 8 rows, constraint T(X)^2 * (a(X)^2 - b(X)).
    //
    //     deg C = 2 + 2 * 7 = 16,   deg Q = 16 - 8 = 8
    //     degree multiple 2   ->  1 chunk of 8 coefficients, too small for deg Q = 8
    //     constraint degree 3 ->  2 chunks of 8 coefficients
    //
    // Invariant: an honest batch proof of a satisfied trace verifies.
    //
    // Mutation: size the quotient by the degree multiple alone.
    //
    // Verification then fails with an out-of-domain evaluation mismatch.
    prove_and_verify(SquareAir { nested: true });
}
