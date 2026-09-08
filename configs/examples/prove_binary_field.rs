//! A complete AIR proof over GF(2^128), with the additive-domain binary PCS.
//!
//! Run with `cargo run --release -p p3-configs --features binary --example prove_binary_field`.
//! The PCS is binding but not hiding; this example does not provide zero knowledge.

use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::TowerLevel;
use p3_configs::binary::{self, BinaryPcsConfig, BinaryPcsParams, Challenger, Config, Val as F};
use p3_configs::multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_sumcheck::layout::Table;

fn config(log_height: usize) -> Config {
    // Two trace columns add one variable to the stacked polynomial.
    // This is an example PCS target, not an end-to-end security claim.
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 100,
    };
    Config::new(BinaryPcsConfig::try_new(log_height + 1, params).unwrap())
}

fn challenger() -> Challenger {
    binary::challenger(b"p3-multi-stark-binary-recurrence-v1")
}

/// A nonlinear recurrence: (a, b) -> (b, a * b + a).
///
/// Addition is XOR and multiplication is tower-field multiplication. Nonlinear
/// constraints exercise interpolation beyond the two prime-subfield elements.
struct RecurrenceAir;

impl<F> BaseAir<F> for RecurrenceAir {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for RecurrenceAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        let public = builder.public_values();
        let (a, b, output) = (public[0], public[1], public[2]);

        builder.when_first_row().assert_eq(local[0], a);
        builder.when_first_row().assert_eq(local[1], b);
        builder.when_transition().assert_eq(next[0], local[1]);
        builder
            .when_transition()
            .assert_eq(next[1], local[0] * local[1] + local[0]);
        builder.when_last_row().assert_eq(local[1], output);
    }
}

fn trace(log_height: usize) -> (Table<F>, [F; 3]) {
    // Integer constructors map into GF(2), so use tower representations for seeds.
    let initial = [
        F::from_repr(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210),
        F::from_repr(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef),
    ];
    let [mut a, mut b] = initial;
    let mut values = Vec::with_capacity(2 << log_height);
    for _ in 0..1 << log_height {
        values.extend([a, b]);
        (a, b) = (b, a * b + a);
    }
    let output = values[values.len() - 1];
    let table = Table::new(RowMajorMatrix::new(values, 2).transpose());
    (table, [initial[0], initial[1], output])
}

fn main() {
    let log_height = 8;
    let config = config(log_height);
    let (table, public) = trace(log_height);
    let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger());

    let start = Instant::now();
    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &RecurrenceAir,
            table,
            &pk,
            &public,
        )]),
        0,
        &mut challenger(),
    );
    let proving_time = start.elapsed();
    let bytes = postcard::to_allocvec(&proof).unwrap();
    let proof: MultiStarkProof<Config> = postcard::from_bytes(&bytes).unwrap();

    let start = Instant::now();
    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &RecurrenceAir,
            &vk,
            log_height,
            &public,
        )]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("binary AIR proof must verify");
    println!(
        "Verified {} rows over GF(2^128) using BinaryPcs: {} bytes, prove {:?}, verify {:?}",
        1 << log_height,
        bytes.len(),
        proving_time,
        start.elapsed(),
    );
}

#[cfg(test)]
mod tests {
    use p3_binary_pcs::{BinaryPcsError, BinaryPcsProof};
    use p3_configs::binary::Mmcs;
    use p3_configs::multi_stark::config::{MultiStarkConfig, PcsError};
    use p3_configs::multi_stark::{VerificationError, VerifyingKey};
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    struct Fixture {
        config: Config,
        vk: VerifyingKey<Config>,
        proof: MultiStarkProof<Config>,
        public: [F; 3],
        log_height: usize,
        pow_bits: usize,
    }

    impl Fixture {
        fn new(log_height: usize, pow_bits: usize) -> Self {
            let config = config(log_height);
            let (table, public) = trace(log_height);
            let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger());
            let proof = prove(
                &config,
                ProverInstances::new(vec![ProverInstance::new(
                    &RecurrenceAir,
                    table,
                    &pk,
                    &public,
                )]),
                pow_bits,
                &mut challenger(),
            );
            // Exercise the wire boundary before checking either honest or tampered proofs.
            let bytes = postcard::to_allocvec(&proof).unwrap();
            let proof = postcard::from_bytes(&bytes).unwrap();
            Self {
                config,
                vk,
                proof,
                public,
                log_height,
                pow_bits,
            }
        }

        fn verify(&self) -> Result<(), VerificationError<PcsError<Config>>> {
            verify(
                &self.config,
                VerifierInstances::new(vec![VerifierInstance::new(
                    &RecurrenceAir,
                    &self.vk,
                    self.log_height,
                    &self.public,
                )]),
                &self.proof,
                self.pow_bits,
                &mut challenger(),
            )
        }
    }

    #[test]
    fn binary_air_proof_round_trips() {
        for log_height in [1, 2, 5] {
            for pow_bits in [0, 2] {
                Fixture::new(log_height, pow_bits).verify().unwrap();
            }
        }
    }

    #[test]
    fn rejects_changed_public_values() {
        for index in 0..3 {
            let mut fixture = Fixture::new(3, 0);
            fixture.public[index] += F::ONE;
            assert!(fixture.verify().is_err());
        }
    }

    #[test]
    fn rejects_changed_transcript_domain() {
        let fixture = Fixture::new(3, 0);
        let result = verify(
            &fixture.config,
            VerifierInstances::new(vec![VerifierInstance::new(
                &RecurrenceAir,
                &fixture.vk,
                fixture.log_height,
                &fixture.public,
            )]),
            &fixture.proof,
            fixture.pow_bits,
            &mut binary::challenger(b"different-application-v1"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejects_changed_sumcheck_polynomial() {
        let mut fixture = Fixture::new(3, 0);
        fixture.proof.sumcheck.round_polys[0][0] += F::ONE;
        assert!(fixture.verify().is_err());
    }

    #[test]
    fn rejects_changed_binary_pcs_codeword() {
        let mut fixture = Fixture::new(3, 0);
        fixture.proof.opening.final_codeword.as_mut_slice()[1] += F::ONE;
        assert!(matches!(
            fixture.verify(),
            Err(VerificationError::Opening(BinaryPcsError::FinalCheck))
        ));
    }

    #[test]
    fn rejects_an_invalid_trace() {
        let log_height = 3;
        let config = config(log_height);
        let (table, public) = trace(log_height);
        let mut columns = table.iter_polys().flatten().copied().collect::<Vec<_>>();
        // Corrupt an interior row while preserving all public boundary values.
        columns[2] += F::ONE;
        let table = Table::new(RowMajorMatrix::new(columns, 1 << log_height));
        let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger());
        let proof = prove(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &RecurrenceAir,
                table,
                &pk,
                &public,
            )]),
            0,
            &mut challenger(),
        );
        let fixture = Fixture {
            config,
            vk,
            proof,
            public,
            log_height,
            pow_bits: 0,
        };
        assert!(fixture.verify().is_err());
    }

    #[test]
    fn cloned_commitment_data_opens_independently() {
        use p3_commit::MultilinearPcs;
        use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableSpec};

        let log_height = 3;
        let config = config(log_height);
        let (table, _) = trace(log_height);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            table.shape(),
            vec![OpeningBatch::new(vec![0, 1], vec![0, 1])],
        )]);
        let expected = table.clone();
        let (commitment, data) = config
            .pcs()
            .commit(config.build_witness(vec![table]), &mut challenger());
        for seed in [b"first opening".as_slice(), b"second opening".as_slice()] {
            use p3_challenger::CanObserve;

            let mut prover = binary::challenger(seed);
            prover.observe(commitment.clone());
            let cloned = data.clone();
            for (actual, expected) in cloned.table(0).iter_polys().zip(expected.iter_polys()) {
                assert_eq!(actual, expected);
            }
            let proof: BinaryPcsProof<Mmcs> =
                config.pcs().open(cloned, protocol.clone(), &mut prover);
            let mut verifier = binary::challenger(seed);
            config
                .pcs()
                .verify(&commitment, &proof, &mut verifier, protocol.clone())
                .unwrap();
        }
    }

    #[test]
    fn mixed_height_binary_air_batch_round_trips() {
        // Two columns at heights 8 and 2 stack into 32 padded entries (arity 5).
        let config = config(4);
        let (tall, tall_public) = trace(3);
        let (short, short_public) = trace(1);
        let (pk, vk) = setup(
            &config,
            &[&RecurrenceAir, &RecurrenceAir],
            &mut challenger(),
        );
        let proof = prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(&RecurrenceAir, short, &pk, &short_public),
                ProverInstance::new(&RecurrenceAir, tall, &pk, &tall_public),
            ]),
            0,
            &mut challenger(),
        );
        verify(
            &config,
            VerifierInstances::new(vec![
                VerifierInstance::new(&RecurrenceAir, &vk, 1, &short_public),
                VerifierInstance::new(&RecurrenceAir, &vk, 3, &tall_public),
            ]),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();
    }

    // One preprocessed column against two main columns exercises distinct PCS arities.
    struct PreprocessedAir;

    impl BaseAir<F> for PreprocessedAir {
        fn width(&self) -> usize {
            2
        }
        fn num_public_values(&self) -> usize {
            3
        }
        fn preprocessed_width(&self) -> usize {
            1
        }
        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            let (table, _) = trace(3);
            let column = table.iter_polys().next().unwrap().to_vec();
            Some(RowMajorMatrix::new(column, 1))
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for PreprocessedAir {
        fn eval(&self, builder: &mut AB) {
            RecurrenceAir.eval(builder);
            let main = builder.main();
            let preprocessed = builder.preprocessed();
            let local = preprocessed.current_slice()[0];
            let next = preprocessed.next_slice()[0];
            builder.assert_eq(main.current_slice()[0], local);
            builder
                .when_transition()
                .assert_eq(main.next_slice()[0], next);
        }
    }

    #[test]
    fn binary_preprocessed_key_is_reusable_and_openings_are_checked() {
        use p3_challenger::CanObserve;

        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 100,
        };
        let config = config(3).with_preprocessed(BinaryPcsConfig::try_new(3, params).unwrap());
        let (pk, vk) = setup(&config, &[&PreprocessedAir], &mut challenger());
        for seed in [2, 3] {
            let fresh = || {
                let mut ch = challenger();
                ch.observe(F::from_repr(seed));
                ch
            };
            let (table, public) = trace(3);
            let mut proof = prove(
                &config,
                ProverInstances::new(vec![ProverInstance::new(
                    &PreprocessedAir,
                    table,
                    &pk,
                    &public,
                )]),
                0,
                &mut fresh(),
            );
            let check = |proof: &MultiStarkProof<Config>| {
                verify(
                    &config,
                    VerifierInstances::new(vec![VerifierInstance::new(
                        &PreprocessedAir,
                        &vk,
                        3,
                        &public,
                    )]),
                    proof,
                    0,
                    &mut fresh(),
                )
            };
            check(&proof).unwrap();
            proof
                .preprocessed_opening
                .as_mut()
                .unwrap()
                .final_codeword
                .as_mut_slice()[1] += F::ONE;
            assert!(check(&proof).is_err());
        }
    }
}
