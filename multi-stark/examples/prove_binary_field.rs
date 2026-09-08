//! A complete AIR proof over GF(2^128), with the additive-domain binary PCS.
//!
//! Run with `cargo run --release -p p3-multi-stark --example prove_binary_field`.
//! The PCS is binding but not hiding; this example does not provide zero knowledge.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BinaryPcsProverData, GroupedCodewordMmcs,
};
use p3_challenger::HashChallenger;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

type F = BinaryField128;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MerkleMmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Mmcs = GroupedCodewordMmcs<MerkleMmcs>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

struct Config {
    pcs: BinaryPcs<Mmcs>,
}

impl MultiStarkConfig for Config {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BinaryPcs<Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn min_num_variables(&self) -> usize {
        // The binary PCS folds at least one variable and does not pad individual tables.
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        SuffixProver::<F, F>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<Mmcs>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

fn config(log_height: usize) -> Config {
    // Two trace columns add one variable to the stacked polynomial.
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 100,
    };
    // Commit after up to three variable folds, with one coset per leaf.
    // The final batch and its leaves shrink to the number of remaining variables.
    let pcs_config = BinaryPcsConfig::try_new(log_height + 1, params)
        .unwrap()
        .try_with_folding(3.min(log_height + 1))
        .unwrap();
    let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let mmcs = Mmcs::for_folding(merkle, &pcs_config);
    Config {
        pcs: BinaryPcs::new(pcs_config, mmcs),
    }
}

fn challenger() -> Challenger {
    Challenger::from_hasher(
        b"p3-multi-stark-binary-recurrence-v1".to_vec(),
        Keccak256Hash,
    )
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
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    let log_height = 18;
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
        0,
        &mut challenger(),
    );
    let bytes = postcard::to_allocvec(&proof).unwrap();
    let proof: MultiStarkProof<Config> = postcard::from_bytes(&bytes).unwrap();

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
        "Verified {} rows over GF(2^128) using BinaryPcs: {} bytes",
        1 << log_height,
        bytes.len(),
    );
}

#[cfg(test)]
mod tests {
    use p3_binary_pcs::{BinaryPcsError, BinaryPcsProof};
    use p3_field::PrimeCharacteristicRing;
    use p3_multi_stark::config::PcsError;
    use p3_multi_stark::{VerificationError, VerifyingKey};

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
            .pcs
            .commit(config.build_witness(vec![table]), &mut challenger());
        for seed in [b"first opening".as_slice(), b"second opening".as_slice()] {
            use p3_challenger::CanObserve;

            let mut prover = Challenger::from_hasher(seed.to_vec(), Keccak256Hash);
            prover.observe(commitment.clone());
            let cloned = data.clone();
            for (actual, expected) in cloned.table(0).iter_polys().zip(expected.iter_polys()) {
                assert_eq!(actual, expected);
            }
            let proof: BinaryPcsProof<Mmcs> =
                config.pcs.open(cloned, protocol.clone(), &mut prover);
            let mut verifier = Challenger::from_hasher(seed.to_vec(), Keccak256Hash);
            config
                .pcs
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

    /// Uses the same PCS arity for equally wide main and preprocessed tables.
    struct PreprocessedConfig(Config);

    impl MultiStarkConfig for PreprocessedConfig {
        type Val = F;
        type Challenge = F;
        type Challenger = Challenger;
        type Pcs = BinaryPcs<Mmcs>;

        fn pcs(&self) -> &Self::Pcs {
            self.0.pcs()
        }
        fn preprocessed_pcs(&self) -> &Self::Pcs {
            self.0.pcs()
        }
        fn min_num_variables(&self) -> usize {
            self.0.min_num_variables()
        }
        fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
            self.0.build_witness(tables)
        }
        fn committed_table<'a>(
            &self,
            data: &'a BinaryPcsProverData<Mmcs>,
            index: usize,
        ) -> &'a Table<F> {
            self.0.committed_table(data, index)
        }
    }

    struct PreprocessedAir;

    impl BaseAir<F> for PreprocessedAir {
        fn width(&self) -> usize {
            2
        }
        fn num_public_values(&self) -> usize {
            3
        }
        fn preprocessed_width(&self) -> usize {
            2
        }
        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            let (table, _) = trace(3);
            let columns = table.iter_polys().flatten().copied().collect();
            Some(RowMajorMatrix::new(columns, 8).transpose())
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for PreprocessedAir {
        fn eval(&self, builder: &mut AB) {
            RecurrenceAir.eval(builder);
            let main = builder.main();
            let preprocessed = builder.preprocessed();
            let local = [
                preprocessed.current_slice()[0],
                preprocessed.current_slice()[1],
            ];
            let next = [preprocessed.next_slice()[0], preprocessed.next_slice()[1]];
            for column in 0..2 {
                builder.assert_eq(main.current_slice()[column], local[column]);
                builder
                    .when_transition()
                    .assert_eq(main.next_slice()[column], next[column]);
            }
        }
    }

    #[test]
    fn binary_preprocessed_key_is_reusable_and_openings_are_checked() {
        use p3_challenger::CanObserve;

        let config = PreprocessedConfig(config(3));
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
            let check = |proof: &MultiStarkProof<PreprocessedConfig>| {
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
