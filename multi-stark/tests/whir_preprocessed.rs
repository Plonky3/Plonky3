//! End-to-end multilinear AIR SNARK with a preprocessed trace, over WHIR.
//!
//! The preprocessed trace is committed once by [`setup`] and reused across proofs.
//! The main trace and the preprocessed trace live in two independent WHIR commitments,
//! opened at the single point the zerocheck binds.

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

/// First-round folding factor, also the per-table padding floor.
const FOLDING: usize = 2;

/// Main trace column count.
const MAIN_WIDTH: usize = 2;

/// Preprocessed trace column count.
const PREPROCESSED_WIDTH: usize = 1;

/// A WHIR-backed configuration with a separate scheme per committed table.
///
/// The main trace and the preprocessed trace stack different column counts.
/// Their stacked polynomials therefore have different arities.
/// Each needs its own WHIR configuration.
struct WhirConfigForTest {
    /// Scheme sized for the main stacked trace.
    pcs: TestPcs,
    /// Scheme sized for the preprocessed stacked trace.
    preprocessed_pcs: TestPcs,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn preprocessed_pcs(&self) -> &TestPcs {
        &self.preprocessed_pcs
    }

    fn min_num_variables(&self) -> usize {
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

/// Build a WHIR scheme sized for a stacked polynomial of a given column count.
fn pcs_for(log_height: usize, width: usize) -> TestPcs {
    let stacked_num_variables = log_height + log2_ceil_usize(width);
    pcs_for_stacked(stacked_num_variables)
}

/// Build a WHIR scheme sized for an already-stacked polynomial arity.
fn pcs_for_stacked(stacked_num_variables: usize) -> TestPcs {
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
    TestPcs::new(whir_config, MyDft::default(), mmcs)
}

/// Build a configuration sized for the trace shape, one scheme per committed table.
fn config_for(log_height: usize) -> WhirConfigForTest {
    WhirConfigForTest {
        pcs: pcs_for(log_height, MAIN_WIDTH),
        preprocessed_pcs: pcs_for(log_height, PREPROCESSED_WIDTH),
    }
}

/// Build a configuration whose main PCS commits several same-shape trace tables.
fn batch_config_for(log_height: usize, num_tables: usize) -> WhirConfigForTest {
    let main_stacked_num_variables = log2_ceil_usize(num_tables * MAIN_WIDTH * (1 << log_height));
    let preprocessed_stacked_num_variables =
        log2_ceil_usize(num_tables * PREPROCESSED_WIDTH * (1 << log_height));
    WhirConfigForTest {
        pcs: pcs_for_stacked(main_stacked_num_variables),
        preprocessed_pcs: pcs_for_stacked(preprocessed_stacked_num_variables),
    }
}

/// A challenger seeded with the same domain separator on both proof and verify sides.
///
/// Both schemes fold identically, so one domain separator covers the shared transcript.
fn challenger(config: &WhirConfigForTest) -> MyChallenger {
    let mut challenger = MyChallenger::new(perm());
    let mut ds = DomainSeparator::new(vec![]);
    config.pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut challenger);
    challenger
}

/// AIR pairing a two-column main trace with a fixed one-column preprocessed trace.
///
/// The preprocessed column holds fixed values `c_i = 3 + 2 * i`.
/// The main trace is `a_i = c_i` and `b_i = 2 * a_i`, so every constraint vanishes:
///
/// - all rows: `a == fixed`      (main col 0 tracks the preprocessed column)
/// - all rows: `b == a + a`      (main col 1 is twice col 0)
/// - transition: `a.next == fixed.next`  (reads the preprocessed next row)
struct PreprocessedAir {
    /// Trace height that the preprocessed column is generated to match.
    height: usize,
}

struct MainRow<T> {
    a: T,
    b: T,
}

impl<T> Borrow<MainRow<T>> for [T] {
    fn borrow(&self) -> &MainRow<T> {
        // Safety: two fields of type T in declaration order match the layout of [T; 2].
        debug_assert_eq!(self.len(), MAIN_WIDTH);
        let ptr = self.as_ptr() as *const MainRow<T>;
        unsafe { &*ptr }
    }
}

impl BaseAir<F> for PreprocessedAir {
    fn width(&self) -> usize {
        MAIN_WIDTH
    }
    fn preprocessed_width(&self) -> usize {
        PREPROCESSED_WIDTH
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        // The fixed column holds `3 + 2 * i` at row i.
        Some(RowMajorMatrix::new(
            fixed_column(self.height),
            PREPROCESSED_WIDTH,
        ))
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for PreprocessedAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local: &MainRow<AB::Var> = main.current_slice().borrow();
        let next: &MainRow<AB::Var> = main.next_slice().borrow();

        let preprocessed = builder.preprocessed();
        let fixed = preprocessed.current_slice()[0];
        let fixed_next = preprocessed.next_slice()[0];

        // Main column 0 equals the fixed preprocessed column.
        builder.assert_eq(local.a, fixed);
        // Main column 1 is twice column 0.
        builder.assert_eq(local.b, local.a + local.a);
        // The preprocessed column advances in step with column 0.
        builder.when_transition().assert_eq(next.a, fixed_next);
    }
}

/// The fixed preprocessed column for a height-`n` trace.
fn fixed_column(n: usize) -> Vec<F> {
    (0..n).map(|i| F::from_u64(3 + 2 * i as u64)).collect()
}

/// The satisfying main trace: `a = fixed`, `b = 2 * a`.
fn main_trace(fixed: &[F]) -> RowMajorMatrix<F> {
    let mut values = Vec::with_capacity(MAIN_WIDTH * fixed.len());
    for &c in fixed {
        values.push(c);
        values.push(c + c);
    }
    RowMajorMatrix::new(values, MAIN_WIDTH)
}

#[test]
fn prove_verify_preprocessed_roundtrips() {
    // A satisfying trace with a preprocessed column must prove and verify end to end.
    let n = 256;
    let fixed = fixed_column(n);
    let air = PreprocessedAir { height: n };
    let trace = main_trace(&fixed);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height);
    let airs = [&air];

    // Commit the preprocessed trace once, so both keys carry its commitment.
    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );
    // The preprocessed opening is present, matching the AIR's declared trace.
    assert!(proof.preprocessed_opening.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest preprocessed proof must verify");
}

#[test]
fn prove_verify_batched_preprocessed_roundtrips() {
    // Main traces share one commitment; each reused preprocessed key still opens separately.
    let n = 256;
    let log_height = log2_strict_usize(n);
    let fixed = fixed_column(n);
    let air = PreprocessedAir { height: n };
    let trace = main_trace(&fixed);
    let config = batch_config_for(log_height, 2);
    let airs = [&air, &air];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace.transpose()), &pk, &[]),
            ProverInstance::new(&air, Table::new(trace.transpose()), &pk, &[]),
        ]),
        0,
        &mut challenger(&config),
    );

    assert!(proof.preprocessed_opening.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_height, &[]),
            VerifierInstance::new(&air, &vk, log_height, &[]),
        ]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest batched preprocessed proof must verify");
}

#[test]
fn prove_verify_mixed_height_preprocessed_roundtrips() {
    // Invariant: two AIRs of different heights, each with a preprocessed trace,
    //   batch into one main commitment and one preprocessed commitment, and the
    //   honest proof verifies.
    //
    // Fixture state:
    //
    //     air a: height 256 -> 8 variables, 1 preprocessed column
    //     air b: height 128 -> 7 variables, 1 preprocessed column
    //
    // The main and preprocessed openings of the height-7 AIR both drop the
    //   leading coordinate before opening.
    let n_a = 256;
    let n_b = 128;
    let log_a = log2_strict_usize(n_a);
    let log_b = log2_strict_usize(n_b);
    let air_a = PreprocessedAir { height: n_a };
    let air_b = PreprocessedAir { height: n_b };
    let trace_a = main_trace(&fixed_column(n_a));
    let trace_b = main_trace(&fixed_column(n_b));

    // Size each scheme for the stacked cell count the layout planner computes.
    let main_cells = MAIN_WIDTH * n_a + MAIN_WIDTH * n_b;
    let preprocessed_cells = PREPROCESSED_WIDTH * n_a + PREPROCESSED_WIDTH * n_b;
    let config = WhirConfigForTest {
        pcs: pcs_for_stacked(log2_ceil_usize(main_cells)),
        preprocessed_pcs: pcs_for_stacked(log2_ceil_usize(preprocessed_cells)),
    };
    let airs = [&air_a, &air_b];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air_a, Table::new(trace_a.transpose()), &pk, &[]),
            ProverInstance::new(&air_b, Table::new(trace_b.transpose()), &pk, &[]),
        ]),
        0,
        &mut challenger(&config),
    );

    assert!(proof.preprocessed_opening.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air_a, &vk, log_a, &[]),
            VerifierInstance::new(&air_b, &vk, log_b, &[]),
        ]),
        &proof,
        0,
        &mut challenger(&config),
    )
    .expect("honest mixed-height batched preprocessed proof must verify");
}

#[test]
fn setup_is_reusable_across_proofs() {
    // One setup commits the preprocessed trace, then two independent proofs reuse it.
    let n = 256;
    let fixed = fixed_column(n);
    let air = PreprocessedAir { height: n };
    let trace = main_trace(&fixed);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height);
    let airs = [&air];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    // Each proof clones the committed preprocessed data and opens it at its own point.
    for _ in 0..2 {
        let proof = prove(
            &config,
            ProverInstances::new(vec![ProverInstance::new(
                &air,
                Table::new(trace.transpose()),
                &pk,
                &[],
            )]),
            0,
            &mut challenger(&config),
        );
        verify(
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
            &proof,
            0,
            &mut challenger(&config),
        )
        .expect("each proof reusing the preprocessed key must verify");
    }
}

#[test]
fn verify_rejects_violated_main_constraint() {
    // Fixture state: a satisfying trace obeys `b == 2 * a`.
    let n = 256;
    let fixed = fixed_column(n);
    let air = PreprocessedAir { height: n };
    let mut trace = main_trace(&fixed);
    // Mutation: break column 1 of the first row so `b != 2 * a`.
    trace.values[1] += F::ONE;
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height);
    let airs = [&air];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );

    // Expected rejection: the zerocheck closes on a nonzero constraint value.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
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
fn verify_rejects_tampered_preprocessed_opening() {
    // Fixture state: the proof carries commitment-bound preprocessed openings.
    let n = 256;
    let fixed = fixed_column(n);
    let air = PreprocessedAir { height: n };
    let trace = main_trace(&fixed);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height);
    let airs = [&air];

    let (pk, vk) = setup(&config, &airs, &mut challenger(&config));

    let mut proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(&config),
    );

    // Mutation: shift the first preprocessed current-row value by one field element.
    let opening = proof.preprocessed_opening.as_mut().unwrap();
    let batch = &opening.evals[0];
    let mut current = batch.current().to_vec();
    current[0] += EF::ONE;
    opening.evals[0] = OpeningBatch::new(current, batch.next().to_vec());

    // Expected rejection: the tampered value is no longer bound to the preprocessed commitment.
    // Why: the verifier samples query positions from the absorbed value.
    //   the round verifies as one pruned multiproof
    //   -> failure reports a batched placeholder position, not a per-query index.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
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
        other => panic!("expected a preprocessed Merkle opening rejection, got {other:?}"),
    }
}
