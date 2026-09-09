//! End-to-end multilinear AIR SNARK over WHIR: commit, zerocheck, open, verify.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::zerocheck::ZerocheckError;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerificationError, VerifierInstance,
    VerifierInstances, prove, setup, verify,
};
use p3_sumcheck::OpeningBatch;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use p3_whir::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, VerifierError as WhirVerifierError,
    WhirConfig, WhirProver,
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
    collision_bits: Option<usize>,
}

impl MultiStarkConfig for WhirConfigForTest {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        self.collision_bits
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
        collision_bits: None,
    }
}

/// A fresh challenger.
///
/// The scheme seeds its own transcript when it opens.
fn challenger() -> MyChallenger {
    MyChallenger::new(perm())
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

/// A local multiset equality declaring two tuples per row.
///
/// Its two blocks make the lookup reduction point one coordinate longer than the trace point.
/// That exercises the leading sumcheck rounds that run before any AIR stage activates.
struct LocalPermutationLookupAir;

impl BaseAir<F> for LocalPermutationLookupAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB> Air<AB> for LocalPermutationLookupAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();

        // Column 0 requests a value, column 1 provides one, both with multiplicity one.
        // The bus balances exactly when the two columns are permutations of each other.
        builder.push_local_interaction([
            (vec![local[0].into()], Count::bounded(AB::Expr::ONE, 1)),
            (vec![local[1].into()], Count::provided(-AB::Expr::ONE)),
        ]);
    }
}

/// Build a trace whose second column is the reverse of its first.
///
/// ```text
///     row   : 0      1      ...  n-1
///     col 0 : 0      1      ...  n-1
///     col 1 : n-1    n-2    ...  0
/// ```
fn permutation_trace(n: usize) -> RowMajorMatrix<F> {
    // Reversal is a permutation, so every requested value is provided exactly once.
    let values = (0..n)
        .flat_map(|row| [F::from_usize(row), F::from_usize(n - 1 - row)])
        .collect();
    RowMajorMatrix::new(values, 2)
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
fn security_rejects_missing_collision_evidence() {
    let config = config_for(4, NUM_COLS);
    let air = FibAir;
    let (_, vk) = setup(&config, &[&air], &mut challenger());
    let public = fib_public_values(16);
    let instances = VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &public)]);
    let report = p3_multi_stark::security_report(&config, &instances).unwrap();
    assert_eq!(report.security_bits(), None);
    assert!(report.require_security(1).is_err());
}

#[test]
fn security_rejects_verifier_height_overflow_without_panicking() {
    let config = config_for(4, NUM_COLS);
    let air = FibAir;
    let (_, vk) = setup(&config, &[&air], &mut challenger());
    let public = fib_public_values(16);
    let instances = VerifierInstances::new(vec![VerifierInstance::new(
        &air,
        &vk,
        usize::BITS as usize,
        &public,
    )]);
    assert!(p3_multi_stark::security_report(&config, &instances).is_err());
}

#[test]
fn security_checked_whir_roundtrip_and_target_rejection() {
    use p3_challenger::CanSample;
    use p3_multi_stark::{
        SecurityError, prove_with_security, security_report, verify_with_security,
    };

    let mut config = config_for(4, NUM_COLS);
    config.collision_bits = Some(100);
    let air = FibAir;
    let (pk, vk) = setup(&config, &[&air], &mut challenger());
    let public = fib_public_values(16);
    let verifier_instances =
        || VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &public)]);
    let prover_instances = || {
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(fib_trace(16).transpose()),
            &pk,
            &public,
        )])
    };
    let report = security_report(&config, &verifier_instances()).unwrap();
    let bits = report.security_bits().unwrap();
    // CapacityBound at stacked arity 5 and rate 1/2 has L = 2560.
    // Four degree-three sumcheck rounds cost 12/q for each candidate trace.
    let sumcheck_bits = report
        .terms()
        .iter()
        .find(|term| term.label == "constraint-sumcheck")
        .unwrap()
        .bits
        .bits();
    assert!((sumcheck_bits - (123.0 - 12f64.log2() - 2560f64.log2())).abs() < 1e-10);
    let pcs_bits = report
        .terms()
        .iter()
        .find(|term| term.label == "main-pcs")
        .unwrap()
        .bits
        .bits();
    assert!(
        bits >= 20.0 && bits <= pcs_bits,
        "composed {bits}, PCS {pcs_bits}"
    );
    assert!(
        report
            .terms()
            .iter()
            .any(|term| term.label == "constraint-sumcheck")
    );

    let mut rejected = challenger();
    assert!(matches!(
        prove_with_security(&config, prover_instances(), 0, 100, &mut rejected),
        Err(SecurityError::InsufficientSecurity { .. })
    ));
    let after_rejection: F = rejected.sample();
    let untouched: F = challenger().sample();
    assert_eq!(after_rejection, untouched);

    let proof = prove_with_security(&config, prover_instances(), 0, 20, &mut challenger()).unwrap();
    verify_with_security(
        &config,
        verifier_instances(),
        &proof,
        0,
        20,
        &mut challenger(),
    )
    .unwrap();
    let mut rejected = challenger();
    assert!(matches!(
        verify_with_security(&config, verifier_instances(), &proof, 0, 100, &mut rejected),
        Err(VerificationError::Security(
            SecurityError::InsufficientSecurity { .. }
        ))
    ));
    let after_rejection: F = rejected.sample();
    assert_eq!(after_rejection, untouched);
}

#[test]
fn security_checked_lookup_accounts_for_every_reduction() {
    use p3_multi_stark::{prove_with_security, security_report, verify_with_security};

    let mut config = config_for(4, 2);
    config.collision_bits = Some(100);
    let air = LocalPermutationLookupAir;
    let (pk, vk) = setup(&config, &[&air], &mut challenger());
    let verifier_instances =
        || VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &[])]);
    let report = security_report(&config, &verifier_instances()).unwrap();
    for (label, roots) in [
        ("logup-fingerprint", 96f64),
        ("fractional-gkr", 40f64),
        ("lookup-opening-link", 1f64),
        ("lookup-air-link", 1f64),
    ] {
        let term = report
            .terms()
            .iter()
            .find(|term| term.label == label)
            .unwrap_or_else(|| panic!("missing {label}"));
        assert!((term.bits.bits() - (123.0 - roots.log2() - 2560f64.log2())).abs() < 1e-10);
    }
    let proof = prove_with_security(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(permutation_trace(16).transpose()),
            &pk,
            &[],
        )]),
        0,
        20,
        &mut challenger(),
    )
    .unwrap();
    verify_with_security(
        &config,
        verifier_instances(),
        &proof,
        0,
        20,
        &mut challenger(),
    )
    .unwrap();
}

#[test]
fn security_rejects_degree_underhints_in_all_build_profiles() {
    struct UnderhintAir;
    impl<X> BaseAir<X> for UnderhintAir {
        fn width(&self) -> usize {
            1
        }
        fn max_constraint_degree(&self) -> Option<usize> {
            Some(1)
        }
    }
    impl<AB: AirBuilder> Air<AB> for UnderhintAir {
        fn eval(&self, builder: &mut AB) {
            let x = builder.main().current_slice()[0];
            builder.assert_zero(x * x);
        }
    }
    let config = config_for(4, 1);
    let air = UnderhintAir;
    let (_, vk) = setup(&config, &[&air], &mut challenger());
    let instances = VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &[])]);
    assert!(matches!(
        p3_multi_stark::security_report(&config, &instances),
        Err(p3_multi_stark::SecurityError::InvalidShape(_))
    ));
}

#[test]
fn security_small_base_challenges_cannot_claim_a_large_target() {
    type BaseLayout = PrefixProver<F, F>;
    type BasePcs = WhirProver<F, F, MyDft, MyMmcs, MyChallenger, BaseLayout>;
    struct BaseConfig(BasePcs);
    impl MultiStarkConfig for BaseConfig {
        type Val = F;
        type Challenge = F;
        type Challenger = MyChallenger;
        type Pcs = BasePcs;
        fn pcs(&self) -> &BasePcs {
            &self.0
        }
        fn collision_resistance_bits(&self) -> Option<usize> {
            Some(100)
        }
        fn min_num_variables(&self) -> usize {
            FOLDING
        }
        fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
            BaseLayout::new_witness(tables, FOLDING)
        }
        fn committed_table<'a>(
            &self,
            data: &'a p3_whir::WhirProverData<F, F, MyMmcs, BaseLayout>,
            index: usize,
        ) -> &'a Table<F> {
            data.table(index)
        }
    }
    let folding = FoldingFactor::Constant(FOLDING);
    let whir = WhirConfig::new(
        5,
        ProtocolParameters {
            security_level: 16,
            pow_bits: 0,
            round_log_inv_rates: default_round_log_inv_rates(5, &folding),
            folding_factor: folding,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        },
    )
    .unwrap();
    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let config = BaseConfig(BasePcs::new(whir, MyDft::default(), mmcs));
    let air = FibAir;
    let (_, vk) = setup(&config, &[&air], &mut challenger());
    let public = fib_public_values(16);
    let instances = VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &public)]);
    let report = p3_multi_stark::security_report(&config, &instances).unwrap();
    assert!(report.security_bits().unwrap() < 31.0);
    assert!(matches!(
        report.require_security(100),
        Err(p3_multi_stark::SecurityError::InsufficientSecurity { .. })
    ));
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
    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    );

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, log_height, &pis)]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("honest Fibonacci proof must verify");
}

#[test]
fn prove_verify_lookup_roundtrips_through_pcs() {
    // Fixture state: one 64-row AIR declaring two tuples per row.
    //
    //     lookup leaves : 2 * 64 = 128 -> a 7-variable reduction point
    //     trace point   :                 6 variables
    //     -----> one leading block-selector round before the AIR stage activates
    let n = 64;
    let log_height = log2_strict_usize(n);
    let air = LocalPermutationLookupAir;
    let trace = permutation_trace(n);
    let config = config_for(log_height, air.width());
    let (pk, vk) = setup(&config, &[&air], &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &[],
        )]),
        0,
        &mut challenger(),
    );
    // A lookup-declaring AIR must produce a reduction section in the proof.
    assert!(proof.lookup.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("honest lookup proof must verify through the trace PCS opening");
}

#[test]
fn prove_verify_mixed_height_lookups_roundtrip_through_pcs() {
    // Invariant: each trace is opened at its own suffix of the one shared bound point,
    // even though the reduction point is longer than either trace.
    //
    //     bound point : | block selectors | 64-row suffix |
    //                                     | 32-row suffix |
    let air = LocalPermutationLookupAir;
    let height_a = 64;
    let height_b = 32;
    let log_height_a = log2_strict_usize(height_a);
    let log_height_b = log2_strict_usize(height_b);
    let trace_a = permutation_trace(height_a);
    let trace_b = permutation_trace(height_b);
    let stacked_num_variables = log2_ceil_usize(2 * height_a + 2 * height_b);
    let config = config_for_stacked(stacked_num_variables);
    let (pk, vk) = setup(&config, &[&air, &air], &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &[]),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &[]),
        ]),
        0,
        &mut challenger(),
    );

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_height_a, &[]),
            VerifierInstance::new(&air, &vk, log_height_b, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("mixed-height lookup proof must open each trace at its own PCS suffix");
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

    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(),
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
        &mut challenger(),
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
    let mut config = config_for_stacked(log2_ceil_usize(cells));
    config.collision_bits = Some(100);
    let airs = [&air, &air];

    // One setup, then one proof binding both traces under a shared commitment.
    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let proof = p3_multi_stark::prove_with_security(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        20,
        &mut challenger(),
    )
    .unwrap();

    // Both instances verify against the shared proof, each at its own height.
    p3_multi_stark::verify_with_security(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_a, &pis_a),
            VerifierInstance::new(&air, &vk, log_b, &pis_b),
        ]),
        &proof,
        0,
        20,
        &mut challenger(),
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

    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(),
    );

    let err = verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&air, &vk, log_a, &pis_a),
            VerifierInstance::new(&air, &vk, log_b, &pis_b),
        ]),
        &proof,
        0,
        &mut challenger(),
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

    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let mut proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
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
        &mut challenger(),
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

    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    );

    // Expected rejection: the zerocheck closes on a nonzero constraint value.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, log_height, &pis)]),
        &proof,
        0,
        &mut challenger(),
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

    let (pk, vk) = setup(&config, &airs, &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
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
        &mut challenger(),
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

const WHIR_FIXTURE: &str = "tests/fixtures/multi_stark_whir_v0_8_0.postcard";

/// A fixed Fibonacci instance shared by the WHIR compat-fixture generator and checker.
fn whir_compat_case() -> (WhirConfigForTest, RowMajorMatrix<F>, [F; 3], usize) {
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    (config, trace, pis, log_height)
}

fn write_fixture(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let full_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(full_path, bytes)
}

fn read_fixture(path: &str) -> std::io::Result<Vec<u8>> {
    let full_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    std::fs::read(full_path)
}

#[test]
fn verify_whir_compat_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (config, _, pis, log_height) = whir_compat_case();
    let airs = [&FibAir];
    let (_, vk) = setup(&config, &airs, &mut challenger());

    let proof_bytes = read_fixture(WHIR_FIXTURE).expect(
        "Missing fixture. Run: cargo test -p p3-multi-stark --test whir_fibonacci -- --ignored",
    );
    let proof: MultiStarkProof<WhirConfigForTest> = postcard::from_bytes(&proof_bytes)?;

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, log_height, &pis)]),
        &proof,
        0,
        &mut challenger(),
    )?;
    Ok(())
}

#[test]
#[ignore]
fn generate_whir_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Regen: cargo test -p p3-multi-stark --test whir_fibonacci -- --ignored
    let (config, trace, pis, _) = whir_compat_case();
    let airs = [&FibAir];
    let (pk, _) = setup(&config, &airs, &mut challenger());

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    );

    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(WHIR_FIXTURE, &bytes)?;
    Ok(())
}
