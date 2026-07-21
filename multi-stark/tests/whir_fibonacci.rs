//! End-to-end multilinear AIR SNARK over WHIR: commit, zerocheck, open, verify.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::zerocheck::ZerocheckError;
use p3_multi_stark::{
    ProverInstance, ProverInstances, VerificationError, VerifierInstance, VerifierInstances, prove,
    setup, verify,
};
use p3_sumcheck::OpeningBatch;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use p3_whir::{
    DomainSeparator, FoldingFactor, ProtocolParameters, SecurityAssumption,
    VerifierError as WhirVerifierError, WhirConfig, WhirProver,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type L = PrefixProver<F, EF>;
type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

/// First-round folding factor; also the per-table padding floor.
const FOLDING: usize = 2;

/// A WHIR-backed multilinear AIR configuration over BabyBear.
struct WhirConfigForTest {
    /// The WHIR commitment scheme, fixed to one stacked-table arity.
    pcs: TestPcs,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn min_num_variables(&self) -> usize {
        // The witness pads each table to the first-round folding factor.
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        L::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a p3_whir::WhirProverData<F, EF, MyMmcs, L>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Fixed permutation so prover and verifier transcripts match exactly.
fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

/// Per-round log-inverse rates for a stacked polynomial.
fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let folding_schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .expect("valid folding schedule");
    let num_rounds = folding_schedule.len() - 1;
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for &folding in folding_schedule.iter().take(num_rounds) {
        rate += folding - 1;
        rates.push(rate);
    }
    rates
}

/// Build a configuration sized for a trace shape.
///
/// The committed polynomial stacks every trace column.
/// The WHIR configuration must match that stacked arity.
fn config_for(log_height: usize, width: usize) -> WhirConfigForTest {
    let stacked_num_variables = log_height + log2_ceil_usize(width);
    config_for_stacked(stacked_num_variables)
}

/// Build a configuration sized for a batch of same-shape trace tables.
fn batch_config_for(log_height: usize, width: usize, num_tables: usize) -> WhirConfigForTest {
    let stacked_num_variables = log2_ceil_usize(num_tables * width * (1 << log_height));
    config_for_stacked(stacked_num_variables)
}

/// Build a configuration sized for an already-stacked polynomial arity.
fn config_for_stacked(stacked_num_variables: usize) -> WhirConfigForTest {
    let folding_factor = FoldingFactor::Constant(FOLDING);

    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: default_round_log_inv_rates(stacked_num_variables, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let whir_config = WhirConfig::new(stacked_num_variables, params).unwrap();
    WhirConfigForTest {
        pcs: TestPcs::new(whir_config, MyDft::default(), mmcs),
    }
}

/// A challenger seeded with the same domain separator on both proof and verify sides.
fn challenger(config: &WhirConfigForTest) -> MyChallenger {
    let mut challenger = MyChallenger::new(perm());
    let mut ds = DomainSeparator::new(vec![]);
    config.pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut challenger);
    challenger
}

const NUM_COLS: usize = 2;

/// Fibonacci AIR.
///
/// - The first row equals the first two public values.
/// - Each transition advances the Fibonacci recurrence.
/// - The final row equals the output public value.
struct FibAir;

struct FibRow<T> {
    left: T,
    right: T,
}

impl<T> Borrow<FibRow<T>> for [T] {
    fn borrow(&self) -> &FibRow<T> {
        // Safety: two fields of type T in declaration order match the layout of [T; 2].
        debug_assert_eq!(self.len(), NUM_COLS);
        let ptr = self.as_ptr() as *const FibRow<T>;
        unsafe { &*ptr }
    }
}

impl<X> BaseAir<X> for FibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for FibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);

        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        let mut first = builder.when_first_row();
        first.assert_eq(local.left, a);
        first.assert_eq(local.right, b);

        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

/// Build a Fibonacci trace seeded with zero and one.
fn fib_trace(n: usize) -> RowMajorMatrix<F> {
    fib_trace_with(n, F::ZERO, F::ONE)
}

/// Build a Fibonacci trace from arbitrary first-row values.
fn fib_trace_with(n: usize, mut left: F, mut right: F) -> RowMajorMatrix<F> {
    let mut values = Vec::with_capacity(NUM_COLS * n);
    for _ in 0..n {
        values.push(left);
        values.push(right);
        let next_left = right;
        let next_right = left + right;
        left = next_left;
        right = next_right;
    }
    RowMajorMatrix::new(values, NUM_COLS)
}

/// Public inputs for the first row and final output.
fn fib_public_values(n: usize) -> [F; 3] {
    let trace = fib_trace(n);
    fib_public_values_for_trace(&trace)
}

/// Public inputs matching an already-built Fibonacci trace.
fn fib_public_values_for_trace(trace: &RowMajorMatrix<F>) -> [F; 3] {
    let n = trace.values.len() / NUM_COLS;
    let last = trace.values[(n - 1) * NUM_COLS + 1];
    [trace.values[0], trace.values[1], last]
}

#[test]
fn prove_verify_fibonacci_roundtrips() {
    // A satisfying trace must prove and verify end to end through WHIR.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibAir];

    // Fibonacci has no preprocessed trace, so setup yields empty keys.
    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(&config),
    );

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, log_height, &pis)]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest Fibonacci proof must verify");
}

#[test]
fn prove_verify_batched_fibonacci_roundtrips() {
    // Two same-height traces share one main commitment, one zerocheck, and one main opening.
    let n = 256;
    let log_height = log2_strict_usize(n);
    let air = FibAir;
    let trace_a = fib_trace(n);
    let trace_b = fib_trace_with(n, F::ONE, F::ONE);
    let pis_a = fib_public_values_for_trace(&trace_a);
    let pis_b = fib_public_values_for_trace(&trace_b);
    let config = batch_config_for(log_height, NUM_COLS, 2);
    let airs = [&air, &air];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(&config),
    );

    assert!(proof.preprocessed_opening.is_none());

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_height, &pis_a),
            VerifierInstance::new(&air, &vk, log_height, &pis_b),
        ]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest batched Fibonacci proof must verify");
}

#[test]
fn prove_verify_mixed_height_fibonacci_roundtrips() {
    // Invariant: two traces of different heights batch into one commitment and
    //   one zerocheck, and the honest proof verifies.
    //
    // Fixture state:
    //
    //     trace a: height 256 -> 8 variables
    //     trace b: height 128 -> 7 variables
    //     common bound point: 8 coordinates
    //
    // Each trace opens at the suffix of the common point matching its height.
    // The height-7 trace drops the leading coordinate before opening.
    // Equal-height batches never drop a coordinate, so this path is otherwise unexercised.
    let air = FibAir;
    let n_a = 256;
    let n_b = 128;
    let log_a = log2_strict_usize(n_a);
    let log_b = log2_strict_usize(n_b);
    let trace_a = fib_trace(n_a);
    let trace_b = fib_trace(n_b);
    let pis_a = fib_public_values_for_trace(&trace_a);
    let pis_b = fib_public_values_for_trace(&trace_b);

    // Size the config for the stacked cell count the layout planner computes.
    // That count is the summed cells across both tables, rounded up to a power of two.
    let cells = NUM_COLS * n_a + NUM_COLS * n_b;
    let config = config_for_stacked(log2_ceil_usize(cells));
    let airs = [&air, &air];

    // One setup, then one proof binding both traces under a shared commitment.
    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(&config),
    );

    // Both instances verify against the shared proof, each at its own height.
    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_a, &pis_a),
            VerifierInstance::new(&air, &vk, log_b, &pis_b),
        ]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest mixed-height batched proof must verify");
}

#[test]
fn verify_rejects_violated_constraint_in_shorter_table() {
    // Invariant: a broken constraint in the shorter trace of a mixed-height
    //   batch is rejected, so the shorter table's suffix opening is checked.
    //
    // Fixture state:
    //
    //     trace a: height 256, honest
    //     trace b: height 128, one transition broken
    let air = FibAir;
    let n_a = 256;
    let n_b = 128;
    let log_a = log2_strict_usize(n_a);
    let log_b = log2_strict_usize(n_b);
    let trace_a = fib_trace(n_a);
    let mut trace_b = fib_trace(n_b);
    // Mutation: shift row 2 of the shorter trace, breaking its transition.
    trace_b.values[2 * NUM_COLS] += F::ONE;
    let pis_a = fib_public_values_for_trace(&trace_a);
    let pis_b = fib_public_values_for_trace(&trace_b);

    let cells = NUM_COLS * n_a + NUM_COLS * n_b;
    let config = config_for_stacked(log2_ceil_usize(cells));
    let airs = [&air, &air];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(&config),
    );

    let err = verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_a, &pis_a),
            VerifierInstance::new(&air, &vk, log_b, &pis_b),
        ]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected zerocheck final-sum mismatch, got {err:?}"
    );
}

#[test]
fn verify_rejects_tampered_opening() {
    // Fixture state: the proof carries commitment-bound trace openings.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let mut proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(&config),
    );

    // Mutation: shift the first claimed current-row value by one field element.
    let batch = &proof.opening.evals[0];
    let mut current = batch.current().to_vec();
    current[0] += EF::ONE;
    proof.opening.evals[0] = OpeningBatch::new(current, batch.next().to_vec());

    // Expected rejection: the tampered value is no longer bound to the commitment.
    // Why: the verifier samples query positions from the absorbed value.
    //   the round verifies as one pruned multiproof
    //   -> failure reports a batched placeholder position, not a per-query index.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, log_height, &pis)]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    match err {
        VerificationError::Opening(WhirVerifierError::MerkleProofInvalid { position, reason }) => {
            assert_eq!(position, 0);
            assert_eq!(reason, "Base field Merkle multiproof verification failed");
        }
        other => panic!("expected a Merkle opening rejection, got {other:?}"),
    }
}

#[test]
fn verify_rejects_violated_constraint() {
    // Fixture state: a satisfying trace obeys every Fibonacci transition.
    let n = 256;
    let mut trace = fib_trace(n);
    // Mutation: shift one row value used by the transition constraints.
    trace.values[2 * NUM_COLS] += F::ONE;
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(&config),
    );

    // Expected rejection: the zerocheck closes on a nonzero constraint value.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, log_height, &pis)]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected zerocheck final-sum mismatch, got {err:?}"
    );
}

#[test]
fn verify_rejects_tampered_public_values() {
    // Fixture state: the public output equals the final trace row.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(&config),
    );

    // Mutation: shift the claimed output by one field element.
    let mut wrong = pis;
    wrong[2] += F::ONE;
    // Expected rejection: the wrong public value desyncs the transcript.
    // Why: the verifier derives a different opening point than the prover used.
    //   the round verifies as one pruned multiproof
    //   -> failure reports a batched placeholder position, not a per-query index.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &FibAir, &vk, log_height, &wrong,
        )]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .unwrap_err();
    match err {
        VerificationError::Opening(WhirVerifierError::MerkleProofInvalid { position, reason }) => {
            assert_eq!(position, 0);
            assert_eq!(reason, "Base field Merkle multiproof verification failed");
        }
        other => panic!("expected a Merkle opening rejection, got {other:?}"),
    }
}
