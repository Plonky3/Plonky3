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
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances,
    prove_with_security, setup, verify_with_security,
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
    pcs: BinaryPcs<F, F, Mmcs, Mmcs>,
}

impl MultiStarkConfig for Config {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BinaryPcs<F, F, Mmcs, Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // Keccak-256 is shared by the transcript and Merkle tree.
        Some(128)
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
        prover_data: &'a BinaryPcsProverData<F, F, Mmcs>,
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
    let pcs_config = BinaryPcsConfig::try_new::<F, F>(log_height + 1, params)
        .unwrap()
        .try_with_folding(3.min(log_height + 1))
        .unwrap();
    let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let mmcs = Mmcs::for_folding(merkle, &pcs_config);
    Config {
        pcs: BinaryPcs::new(pcs_config, mmcs.clone(), mmcs).unwrap(),
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
/// Addition is XOR and multiplication is tower-field multiplication.
///
/// Nonlinear constraints exercise interpolation past the two prime-subfield elements.
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
    let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger()).unwrap();

    let proof = prove_with_security(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &RecurrenceAir,
            table,
            &pk,
            &public,
        )]),
        0,
        100,
        &mut challenger(),
    )
    .expect("binary AIR proof must meet the 100-bit composed target");
    let bytes = postcard::to_allocvec(&proof).unwrap();
    let proof: MultiStarkProof<Config> = postcard::from_bytes(&bytes).unwrap();

    verify_with_security(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &RecurrenceAir,
            &vk,
            log_height,
            &public,
        )]),
        &proof,
        0,
        100,
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
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder, BusName};
    use p3_field::PrimeCharacteristicRing;
    use p3_multi_stark::config::PcsError;
    use p3_multi_stark::zerocheck::ZerocheckError;
    use p3_multi_stark::{SecurityError, VerificationError, VerifyingKey, prove, verify};

    use super::*;

    /// Binary-field AIR that contributes selected nonlinear payloads to one bus side.
    struct BinaryBusAir {
        /// Multiset side receiving this table's active rows.
        direction: BusDirection,
    }

    impl BaseAir<F> for BinaryBusAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB> Air<AB> for BinaryBusAir
    where
        AB: BusInteractionBuilder<F = F>,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();
            let value: AB::Expr = row[0].into();
            let selector: AB::Expr = row[1].into();
            builder.push_bus_interaction(
                BusName::new("binary-selected-square"),
                self.direction,
                [value.clone() * value],
                BusActivation::Boolean(selector),
            );
        }
    }

    /// Either the recurrence or one end of the binary bus, so both fit one batch.
    enum LiftedAir {
        /// The nonlinear recurrence, which declares no bus.
        Recurrence,
        /// One end of the selected-square bus.
        Bus(BinaryBusAir),
    }

    impl BaseAir<F> for LiftedAir {
        fn width(&self) -> usize {
            match self {
                Self::Recurrence => BaseAir::<F>::width(&RecurrenceAir),
                Self::Bus(air) => air.width(),
            }
        }

        fn num_public_values(&self) -> usize {
            match self {
                Self::Recurrence => BaseAir::<F>::num_public_values(&RecurrenceAir),
                Self::Bus(air) => air.num_public_values(),
            }
        }
    }

    impl<AB> Air<AB> for LiftedAir
    where
        AB: BusInteractionBuilder<F = F>,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::Recurrence => RecurrenceAir.eval(builder),
                Self::Bus(air) => air.eval(builder),
            }
        }
    }

    /// Builds one two-column binary bus table in trace-row order.
    ///
    /// Even rows below `live` are selected, so tables of any height can carry the same multiset.
    fn binary_bus_table(log_height: usize, live: usize) -> Table<F> {
        let mut rows = Vec::with_capacity(2 << log_height);
        for row in 0usize..1usize << log_height {
            rows.extend([
                F::from_repr((row + 2) as u128),
                F::from_bool(row < live && row.is_multiple_of(2)),
            ]);
        }
        Table::new(RowMajorMatrix::new(rows, 2).transpose())
    }

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
            let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger()).unwrap();
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
            )
            .unwrap();
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
    fn the_report_charges_every_binary_bus_draw() {
        // The bus contributes two draws this composition makes for itself.
        //
        // Each lands after the commitment and before the opening names a candidate.
        //
        // Nothing else in the suite builds a report that contains them.
        //
        // So a bus draw could be dropped, or misattributed, with every other test green.
        //
        // This commitment decodes uniquely, so its candidate set holds one member.
        //
        // The charge over it is therefore the identity.
        //
        // A bus draw that went uncharged would therefore read the same here.
        //
        // What this does catch is a bus draw that never reached the report at all.
        //
        // It also catches one attributed to a commitment, which shows as a component.
        //
        // An uncharged bus draw is caught by the list-decoding bus test in the WHIR suite.
        //
        // Pushing a term past the builder, or onto a closed report, does not compile.
        //
        // The builder and the report keep their lists private, and the commitment name is a closed set.
        let log_height = 3;
        let config = config(log_height + 1);
        let push = BinaryBusAir {
            direction: BusDirection::Push,
        };
        let pull = BinaryBusAir {
            direction: BusDirection::Pull,
        };
        let (_, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();
        let report = p3_multi_stark::security_report(
            &config,
            &VerifierInstances::new(vec![
                VerifierInstance::new(&push, &vk, log_height, &[]),
                VerifierInstance::new(&pull, &vk, log_height, &[]),
            ]),
        )
        .unwrap();
        assert!(report.unassessed_components().is_empty());

        for label in ["binary-bus", "binary-bus-batching"] {
            let term = report
                .terms()
                .iter()
                .find(|term| term.label == label)
                .unwrap_or_else(|| panic!("the report drops the {label} draw"));

            // A draw this composition makes belongs to no commitment.
            //
            // A component name here would mean the term took the opening route.
            //
            // It would then have settled at full strength instead of being charged.
            assert_eq!(
                term.component, None,
                "{label} is attributed to a commitment"
            );

            // Every bus draw is a real bound, and none exceeds the field's own width.
            assert!(term.bits.bits().is_finite() && term.bits.bits() > 0.0);
            assert!(term.bits.bits() <= 128.0);
        }

        // The batching scalar is one fresh draw that must avoid two roots.
        //
        //     128 field bits - log2(2)  ->  127 bits
        let batching = report
            .terms()
            .iter()
            .find(|term| term.label == "binary-bus-batching")
            .unwrap();
        assert_eq!(batching.bits.bits(), 127.0);

        // The bus is part of the composed bound the statement is graded against.
        report.require_security(100).unwrap();
    }

    #[test]
    fn binary_bus_proof_round_trips() {
        let log_height = 3;
        // Four stacked columns need two variables above the trace height.
        let config = config(log_height + 1);
        let push = BinaryBusAir {
            direction: BusDirection::Push,
        };
        let pull = BinaryBusAir {
            direction: BusDirection::Pull,
        };
        let (pk, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();
        let proof = prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(
                    &push,
                    binary_bus_table(log_height, 1 << log_height),
                    &pk,
                    &[],
                ),
                ProverInstance::new(
                    &pull,
                    binary_bus_table(log_height, 1 << log_height),
                    &pk,
                    &[],
                ),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();

        verify(
            &config,
            VerifierInstances::new(vec![
                VerifierInstance::new(&push, &vk, log_height, &[]),
                VerifierInstance::new(&pull, &vk, log_height, &[]),
            ]),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();
    }

    #[test]
    fn lifted_binary_bus_shares_round_trip() {
        // Each share shorter than the cube is lifted by the all-one-vertex selector.
        //
        //     lift(x) = prod over the unused coordinates of x_k
        //
        // A constant lift would multiply the share by 2^k, which is zero in characteristic two.
        let push = BinaryBusAir {
            direction: BusDirection::Push,
        };
        let pull = BinaryBusAir {
            direction: BusDirection::Pull,
        };

        // Both sides select rows 0 and 2, so they carry the same multiset at any height.
        let live = 4;

        // Push at 3 and pull at 2: the cube has 3 variables and only the pull share is lifted.
        // The 24 stacked cells pad to 2^5, which config(4) commits.
        let config_short = config(4);
        let (pk, vk) = setup(&config_short, &[&push, &pull], &mut challenger()).unwrap();
        let proof = prove(
            &config_short,
            ProverInstances::new(vec![
                ProverInstance::new(&push, binary_bus_table(3, live), &pk, &[]),
                ProverInstance::new(&pull, binary_bus_table(2, live), &pk, &[]),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();
        verify(
            &config_short,
            VerifierInstances::new(vec![
                VerifierInstance::new(&push, &vk, 3, &[]),
                VerifierInstance::new(&pull, &vk, 2, &[]),
            ]),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();

        // The same bus under a recurrence at 5: the cube has 5 variables and both shares are lifted.
        // The 88 stacked cells pad to 2^7, which config(6) commits.
        let config_tall = config(6);
        let (table, public) = trace(5);
        let recurrence = LiftedAir::Recurrence;
        let push = LiftedAir::Bus(push);
        let pull = LiftedAir::Bus(pull);
        let airs = [&recurrence, &push, &pull];
        let (pk, vk) = setup(&config_tall, &airs, &mut challenger()).unwrap();
        let proof = prove(
            &config_tall,
            ProverInstances::new(vec![
                ProverInstance::new(airs[0], table, &pk, &public),
                ProverInstance::new(airs[1], binary_bus_table(3, live), &pk, &[]),
                ProverInstance::new(airs[2], binary_bus_table(2, live), &pk, &[]),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();
        verify(
            &config_tall,
            VerifierInstances::new(vec![
                VerifierInstance::new(airs[0], &vk, 5, &public),
                VerifierInstance::new(airs[1], &vk, 3, &[]),
                VerifierInstance::new(airs[2], &vk, 2, &[]),
            ]),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();
    }

    #[test]
    fn security_certifies_binary_pcs_and_rejects_an_excessive_target() {
        for log_height in [1, 4, 18] {
            let config = config(log_height);
            let (_, vk) = setup(&config, &[&RecurrenceAir], &mut challenger()).unwrap();
            let public = [F::ZERO; 3];
            let instances = VerifierInstances::new(vec![VerifierInstance::new(
                &RecurrenceAir,
                &vk,
                log_height,
                &public,
            )]);
            let report = p3_multi_stark::security_report(&config, &instances).unwrap();
            assert!(report.unassessed_components().is_empty());
            report.require_security(100).unwrap();

            // The union bound lies between the target that passes and the one that does not.
            let Err(SecurityError::InsufficientSecurity {
                requested,
                available,
            }) = report.require_security(128)
            else {
                panic!("a 128-bit target must be refused for lack of bits");
            };
            assert_eq!(requested, 128);
            assert!((100.0..128.0).contains(&available), "{available}");
        }
        let config = config(4);
        let (table, public) = trace(4);
        let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger()).unwrap();
        let proof = prove_with_security(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &RecurrenceAir,
                table,
                &pk,
                &public,
            )]),
            0,
            100,
            &mut challenger(),
        )
        .unwrap();
        verify_with_security(
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(&RecurrenceAir, &vk, 4, &public)]),
            &proof,
            0,
            100,
            &mut challenger(),
        )
        .unwrap();
    }

    #[test]
    fn rejects_changed_public_values() {
        for index in 0..3 {
            let mut fixture = Fixture::new(3, 0);
            fixture.public[index] += F::ONE;

            // Public values are bound before any challenge, so every later one moves with them.
            assert!(matches!(
                fixture.verify(),
                Err(VerificationError::Opening(BinaryPcsError::FinalCheck))
            ));
        }
    }

    #[test]
    fn rejects_changed_sumcheck_polynomial() {
        let mut fixture = Fixture::new(3, 0);
        fixture.proof.sumcheck.round_polys[0][0] += F::ONE;

        // A changed round message moves the challenge, and so the point the opening answers.
        assert!(matches!(
            fixture.verify(),
            Err(VerificationError::Opening(BinaryPcsError::FinalCheck))
        ));
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
        let (pk, vk) = setup(&config, &[&RecurrenceAir], &mut challenger()).unwrap();
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
        )
        .unwrap();
        let fixture = Fixture {
            config,
            vk,
            proof,
            public,
            log_height,
            pow_bits: 0,
        };

        // The corrupted row leaves the constraint nonzero at the bound point.
        assert!(matches!(
            fixture.verify(),
            Err(VerificationError::Zerocheck(
                ZerocheckError::FinalSumMismatch
            ))
        ));
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
            .commit(config.build_witness(vec![table]), &mut challenger())
            .unwrap();
        for seed in [b"first opening".as_slice(), b"second opening".as_slice()] {
            let mut prover = Challenger::from_hasher(seed.to_vec(), Keccak256Hash);
            config.pcs.observe_commitment(&commitment, &mut prover);
            let cloned = data.clone();
            for (actual, expected) in cloned.table(0).iter_polys().zip(expected.iter_polys()) {
                assert_eq!(actual, expected);
            }
            let proof: BinaryPcsProof<F, F, Mmcs, Mmcs> = config
                .pcs
                .open(cloned, protocol.clone(), &mut prover)
                .unwrap();
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
        )
        .unwrap();
        let proof = prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(&RecurrenceAir, short, &pk, &short_public),
                ProverInstance::new(&RecurrenceAir, tall, &pk, &tall_public),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();
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
        type Pcs = BinaryPcs<F, F, Mmcs, Mmcs>;

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
            data: &'a BinaryPcsProverData<F, F, Mmcs>,
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
        let (pk, vk) = setup(&config, &[&PreprocessedAir], &mut challenger()).unwrap();
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
            )
            .unwrap();
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

            // The tampered codeword fails the scheme's own final check, not a later one.
            assert!(matches!(
                check(&proof),
                Err(VerificationError::Opening(BinaryPcsError::FinalCheck))
            ));
        }
    }
}
