//! Small AIR proofs with explicit, inexpensive demonstration parameters.
//! Run with `cargo run -p p3-configs --features baby-bear,koala-bear --example prove_prime_fields`.
//! These parameters are for demonstration only; neither configuration is hiding.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_configs::{baby_bear, koala_bear, uni_stark};
use p3_field::PrimeCharacteristicRing;
use p3_fri::FriParameters;
use p3_matrix::dense::RowMajorMatrix;

struct FibonacciAir;

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        let public = builder.public_values();
        let [a, b, output] = [public[0], public[1], public[2]];
        builder.when_first_row().assert_eq(local[0], a);
        builder.when_first_row().assert_eq(local[1], b);
        builder.when_transition().assert_eq(next[0], local[1]);
        builder
            .when_transition()
            .assert_eq(next[1], local[0] + local[1]);
        builder.when_last_row().assert_eq(local[1], output);
    }
}

const fn demonstration_parameters() -> FriParameters<()> {
    FriParameters {
        log_blowup: 2,
        log_final_poly_len: 1,
        max_log_arity: 2,
        num_queries: 4,
        batch_proof_of_work_bits: 1,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: (),
    }
}

// Instantiate the same AIR and proof flow for each concrete supported field.
macro_rules! prime_example {
    ($name:ident, $field:ident) => {
        fn $name() {
            use $field::{Config, Val};

            // Eight rows, ending at (13, 21). Public expectations are independent
            // of the trace-generation logic.
            let mut values = Vec::new();
            let (mut a, mut b) = (Val::ZERO, Val::ONE);
            for _ in 0..8 {
                values.extend([a, b]);
                (a, b) = (b, a + b);
            }
            let trace = RowMajorMatrix::new(values, 2);
            let public = [Val::ZERO, Val::ONE, Val::from_u32(21)];
            let config = $field::new(demonstration_parameters(), 0).with_ood_proof_of_work_bits(1);
            let proof = uni_stark::prove(&config, &FibonacciAir, trace, &public);
            let bytes = postcard::to_allocvec(&proof).unwrap();
            let proof: uni_stark::Proof<Config> = postcard::from_bytes(&bytes).unwrap();
            // Check the actual proof honors the chosen query count and early stop.
            assert_eq!(proof.opening_proof.input_openings[0].opened_values.len(), 4);
            assert_eq!(proof.opening_proof.final_poly.len(), 2);

            // Construct verifier state independently. uni-stark clones the initial
            // challenger inside this config for each prove/verify invocation.
            let verifier =
                $field::new(demonstration_parameters(), 0).with_ood_proof_of_work_bits(1);
            uni_stark::verify(&verifier, &FibonacciAir, &proof, &public).unwrap();
            for index in 0..public.len() {
                let mut tampered = public;
                tampered[index] += Val::ONE;
                assert!(uni_stark::verify(&verifier, &FibonacciAir, &proof, &tampered).is_err());
            }
            println!(
                "{}: verified 8 rows, {} proof bytes",
                stringify!($field),
                bytes.len()
            );
        }
    };
}

prime_example!(prove_baby_bear, baby_bear);
prime_example!(prove_koala_bear, koala_bear);

fn main() {
    prove_baby_bear();
    prove_koala_bear();
}

#[cfg(test)]
mod tests {
    #[test]
    fn baby_bear_round_trip_rejects_changed_public_values() {
        super::prove_baby_bear();
    }

    #[test]
    fn koala_bear_round_trip_rejects_changed_public_values() {
        super::prove_koala_bear();
    }
}
