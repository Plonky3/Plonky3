//! End-to-end multilinear AIR SNARK over WHIR: commit, zerocheck, open, verify.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, BoundaryEnd, BoundaryPublic, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_lookup::{Count, IndexedLookupBuilder, InteractionBuilder, TraceWindow};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::lookup::LookupError;
use p3_multi_stark::zerocheck::ZerocheckError;
use p3_multi_stark::{
    BoundaryIoError, MultiStarkProof, ProverInstance, ProverInstances, SecurityError,
    VerificationError, VerifierInstance, VerifierInstances, prove, setup, verify,
};
use p3_sumcheck::OpeningBatch;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};
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

/// Fibonacci AIR that binds its public inputs by position instead of by constraint.
///
/// Only the transition recurrence is asserted here.
/// The folder pins each listed cell to its public value in place of a boundary constraint.
struct FibIoAir;

/// The cells the AIR above binds by position.
///
/// ```text
///     column 0, first row -> public value 0
///     column 1, first row -> public value 1
///     column 1, last  row -> public value 2
/// ```
const FIB_IO_CELLS: [BoundaryPublic; 3] = [
    BoundaryPublic::new(0, BoundaryEnd::First, 0),
    BoundaryPublic::new(1, BoundaryEnd::First, 1),
    BoundaryPublic::new(1, BoundaryEnd::Last, 2),
];

impl<X> BaseAir<X> for FibIoAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &FIB_IO_CELLS
    }
}

impl<AB: AirBuilder> Air<AB> for FibIoAir {
    fn eval(&self, builder: &mut AB) {
        // Read the current row and the row after it.
        let main = builder.main();
        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        // Advance the recurrence, and assert nothing at either end.
        //
        //     next.left  = right
        //     next.right = left + right
        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);
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
    let (_, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
    let public = fib_public_values(16);
    let instances = VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &public)]);
    let report = p3_multi_stark::security_report(&config, &instances).unwrap();
    assert_eq!(report.security_bits(), None);

    // The named component is the one this configuration supplies no evidence for.
    assert!(matches!(
        report.require_security(1),
        Err(SecurityError::UnassessedComponent(
            "commitment-and-transcript-collision"
        ))
    ));
}

#[test]
fn security_rejects_verifier_height_overflow_without_panicking() {
    let config = config_for(4, NUM_COLS);
    let air = FibAir;
    let (_, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
    let public = fib_public_values(16);
    let instances = VerifierInstances::new(vec![VerifierInstance::new(
        &air,
        &vk,
        usize::BITS as usize,
        &public,
    )]);
    // The declared arity exceeds what a trace height can represent.
    assert!(matches!(
        p3_multi_stark::security_report(&config, &instances),
        Err(SecurityError::InvalidShape(
            "trace arity is outside the supported range"
        ))
    ));
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
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
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
        .find(|term| term.label == "whir-opening")
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
        Err(p3_multi_stark::ProvingError::Security(
            SecurityError::InsufficientSecurity { .. }
        ))
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
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
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
    let (_, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
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
    let (_, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
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
fn opening_budget_failure_preserves_challenger() {
    use p3_challenger::CanSample;
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let whir_config = WhirConfig::new(
        17,
        ProtocolParameters {
            security_level: 100,
            pow_bits: 16,
            round_log_inv_rates: default_round_log_inv_rates(17, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        },
    )
    .unwrap();
    let config = WhirConfigForTest {
        pcs: TestPcs::new(
            whir_config,
            MyDft::default(),
            MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0),
        ),
        collision_bits: config_for(16, NUM_COLS).collision_bits,
    };
    let air = FibAir;
    let (pk, _) = setup(&config, &[&air], &mut challenger()).unwrap();
    let trace = fib_trace(1 << 16);
    let pis = fib_public_values(1 << 16);
    let mut transcript = challenger();
    let before: F = transcript.clone().sample();
    let result = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(p3_multi_stark::ProvingError::Pcs {
            phase: "main opening",
            source: p3_whir::WhirConfigError::InitialClaimsBelowTarget {
                num_claims: 6,
                security_level: 100,
                ..
            },
        })
    ));
    assert_eq!(CanSample::<F>::sample(&mut transcript), before);
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
    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    )
    .unwrap();

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
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

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
    )
    .unwrap();
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
    let (pk, vk) = setup(&config, &[&air, &air], &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &[]),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &[]),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap();

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

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap();

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
    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&air, Table::new(trace_a.transpose()), &pk, &pis_a),
            ProverInstance::new(&air, Table::new(trace_b.transpose()), &pk, &pis_b),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap();

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

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    )
    .unwrap();

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

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    )
    .unwrap();

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

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    )
    .unwrap();

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

#[test]
fn verify_rejects_tampered_main_commitment() {
    // Fixture state: the proof carries the commitment the prover's commit phase produced.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    )
    .unwrap();

    // Mutation: shift the first word of the committed Merkle root.
    let mut roots = proof.commitment.roots().to_vec();
    roots[0][0] += F::ONE;
    proof.commitment = MerkleCap::new(roots);

    // Expected rejection: the commitment is absorbed inside the main-commitment bracket.
    // Why: the verifier derives different query positions than the prover answered.
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
    // The variant and the placeholder position are the assertion.
    //
    // The wording belongs to the commitment scheme, so it is not pinned here.
    assert!(
        matches!(
            err,
            VerificationError::Opening(WhirVerifierError::MerkleProofInvalid { position: 0, .. })
        ),
        "expected a Merkle opening rejection, got {err:?}"
    );
}

/// An honest 256-row Fibonacci proof, with everything the verifier needs to check it.
///
/// This AIR declares no preprocessed column, emits no interaction and reads no table by index.
/// All three optional sections of the proof are therefore absent.
fn honest_fibonacci() -> (
    WhirConfigForTest,
    p3_multi_stark::VerifyingKey<WhirConfigForTest>,
    usize,
    [F; 3],
    MultiStarkProof<WhirConfigForTest>,
) {
    // 256 rows of the two-column Fibonacci recurrence, with its three public values.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);

    // The key commits to nothing, since the AIR declares no preprocessed column.
    let (pk, vk) = setup(&config, &[&FibAir], &mut challenger()).unwrap();
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
    )
    .unwrap();
    (config, vk, log_height, pis, proof)
}

/// An honest 64-row proof of the local-permutation lookup AIR.
///
/// Its interactions emit one tuple per row, which forces a lookup section into the proof.
fn honest_lookup() -> (
    WhirConfigForTest,
    p3_multi_stark::VerifyingKey<WhirConfigForTest>,
    usize,
    MultiStarkProof<WhirConfigForTest>,
) {
    // 64 rows whose sent and received tuples cancel, so the fractional sum is zero.
    let n = 64;
    let log_height = log2_strict_usize(n);
    let air = LocalPermutationLookupAir;
    let trace = permutation_trace(n);
    let config = config_for(log_height, air.width());
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
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
    )
    .unwrap();

    // The emitted tuples are what put the section there, so it must be present.
    assert!(proof.lookup.is_some());
    (config, vk, log_height, proof)
}

/// An honest proof of the squares batch, where one table is read by index.
///
/// The declared read forces an indexed reduction into the proof.
fn honest_indexed() -> (
    WhirConfigForTest,
    p3_multi_stark::VerifyingKey<WhirConfigForTest>,
    (usize, usize),
    MultiStarkProof<WhirConfigForTest>,
) {
    // The table holds the squares 0, 1, 4, ..., one per row.
    // The reader walks it forwards then backwards, so it reads every entry twice.
    let table_rows = ((1 << FOLDING) * PackedF::WIDTH / 4).max(1 << FOLDING);
    let reader_rows = 2 * table_rows;
    let table_log = log2_strict_usize(table_rows);
    let reader_log = log2_strict_usize(reader_rows);
    let squares = RowMajorMatrix::new((0..table_rows).map(|v| F::from_usize(v * v)).collect(), 1);
    let named = (0..table_rows)
        .chain((0..table_rows).rev())
        .collect::<Vec<_>>();

    // Each reader row carries the index it names and the value it claims to find there.
    let reads = RowMajorMatrix::new(
        named
            .iter()
            .flat_map(|&v| [F::from_usize(v), F::from_usize(v * v)])
            .collect(),
        2,
    );

    // Both tables share one committed column stack, so the config covers their total height.
    let stacked_num_variables = log2_ceil_usize(2 * reader_rows + table_rows);
    let config = config_for_stacked(stacked_num_variables);
    let (pk, vk) = setup(
        &config,
        &[&SquaresBatch::Reader, &SquaresBatch::Table],
        &mut challenger(),
    )
    .unwrap();
    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(
                &SquaresBatch::Reader,
                Table::new(reads.transpose()),
                &pk,
                &[],
            ),
            ProverInstance::new(
                &SquaresBatch::Table,
                Table::new(squares.transpose()),
                &pk,
                &[],
            ),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap();

    // The declared read is what puts the section there, so it must be present.
    assert!(proof.indexed.is_some());
    (config, vk, (reader_log, table_log), proof)
}

/// Replays verification of a Fibonacci proof against the single AIR that produced it.
fn verify_fibonacci(
    config: &WhirConfigForTest,
    vk: &p3_multi_stark::VerifyingKey<WhirConfigForTest>,
    log_height: usize,
    pis: &[F; 3],
    proof: &MultiStarkProof<WhirConfigForTest>,
) -> Result<(), VerificationError<p3_multi_stark::config::PcsError<WhirConfigForTest>>> {
    verify(
        config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, vk, log_height, pis)]),
        proof,
        0,
        &mut challenger(),
    )
}

// Invariant: a proof carries an optional section exactly when the AIR set declares the feature.
//
//     preprocessed columns -> the preprocessed opening
//     interactions         -> the lookup section
//     indexed reads        -> the indexed reduction
//
// Each declaration is read off the AIRs alone, never off the proof.
// A disagreement is therefore refused before anything inside the section is touched.

#[test]
fn verify_rejects_an_unexpected_preprocessed_opening() {
    // Fixture state: the AIR declares no preprocessed column, so the key commits to none.
    //
    //     key   -> None
    //     proof -> None
    let (config, vk, log_height, pis, mut proof) = honest_fibonacci();
    assert!(proof.preprocessed_opening.is_none());

    // Mutation: reuse the main opening as a preprocessed one the key never asked for.
    //
    //     key   -> None
    //     proof -> Some(a well-formed opening)
    proof.preprocessed_opening = Some(proof.opening.clone());

    // Expected rejection: the key and the proof disagree on whether the section exists.
    // Why: the check reads presence only, never the section's contents.
    //   no commitment was absorbed for this opening to be bound to
    //   -> any value at all is refused on the same ground.
    let err = verify_fibonacci(&config, &vk, log_height, &pis, &proof).unwrap_err();
    assert!(
        matches!(err, VerificationError::UnexpectedPreprocessedOpening),
        "expected UnexpectedPreprocessedOpening, got {err:?}"
    );
}

#[test]
fn verify_rejects_a_lookup_proof_for_an_air_declaring_no_interaction() {
    // Fixture state: the Fibonacci AIR emits no tuple, so no lookup argument runs.
    //
    //     AIRs  -> no tuple emitted
    //     proof -> None
    let (config, vk, log_height, pis, mut proof) = honest_fibonacci();
    assert!(proof.lookup.is_none());

    // A genuine lookup section, proved for an AIR whose interactions really do emit tuples.
    let (_, _, _, lookup_proof) = honest_lookup();

    // Mutation: graft that section onto the proof of an AIR that emits none.
    //
    //     AIRs  -> no tuple emitted
    //     proof -> Some(a well-formed fractional-GKR proof)
    proof.lookup = lookup_proof.lookup;

    // Expected rejection: the layout rebuilt from the AIRs describes no lookup at all.
    // Why: the layout is derived from the AIRs, so the proof cannot claim one into being.
    //   a section with no layout to measure it against can never be checked
    //   -> it is refused rather than silently ignored.
    let err = verify_fibonacci(&config, &vk, log_height, &pis, &proof).unwrap_err();
    assert!(
        matches!(err, VerificationError::Lookup(LookupError::UnexpectedProof)),
        "expected Lookup(UnexpectedProof), got {err:?}"
    );
}

#[test]
fn verify_rejects_an_air_declaring_an_interaction_with_no_lookup_proof() {
    // Fixture state: the AIR's interactions force a lookup section into the proof.
    //
    //     AIRs  -> one tuple emitted per row
    //     proof -> Some(...)
    let (config, vk, log_height, mut proof) = honest_lookup();

    // Mutation: drop the section those interactions require.
    //
    //     AIRs  -> one tuple emitted per row
    //     proof -> None
    proof.lookup = None;

    // Expected rejection: the rebuilt layout describes a lookup with nothing to check.
    // Why: skipping the argument instead would hide the real fault.
    //   the zerocheck still expects the link claim the reduction would have produced
    //   -> the failure would surface later as a link mismatch, naming the wrong culprit.
    let air = LocalPermutationLookupAir;
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &[])]),
        &proof,
        0,
        &mut challenger(),
    )
    .unwrap_err();
    assert!(
        matches!(err, VerificationError::Lookup(LookupError::MissingProof)),
        "expected Lookup(MissingProof), got {err:?}"
    );
}

#[test]
fn verify_rejects_an_indexed_proof_for_an_air_declaring_no_indexed_read() {
    // Fixture state: the Fibonacci AIR reads no table by index.
    //
    //     AIRs  -> no read declared
    //     proof -> None
    let (config, vk, log_height, pis, mut proof) = honest_fibonacci();
    assert!(proof.indexed.is_none());

    // A genuine indexed section, proved for a batch where one AIR does read a table.
    let (_, _, _, indexed_proof) = honest_indexed();

    // Mutation: graft that section onto the proof of an AIR that reads nothing.
    //
    //     AIRs  -> no read declared
    //     proof -> Some(a well-formed reduction)
    proof.indexed = indexed_proof.indexed;

    // Expected rejection: the AIRs describe no indexed bracket for the section to fill.
    // Why: the transcript replays an indexed step only when a read is declared.
    //   nothing would absorb the grafted section, so it would bind to no challenge
    //   -> the proof is refused before the reduction is read.
    let err = verify_fibonacci(&config, &vk, log_height, &pis, &proof).unwrap_err();
    assert!(
        matches!(err, VerificationError::UnexpectedIndexedReduction),
        "expected UnexpectedIndexedReduction, got {err:?}"
    );
}

#[test]
fn verify_rejects_an_air_declaring_an_indexed_read_with_no_indexed_proof() {
    // Fixture state: the reader AIR looks every value up in the table AIR by index.
    //
    //     AIRs  -> one read declared
    //     proof -> Some(...)
    let (config, vk, (reader_log, table_log), mut proof) = honest_indexed();

    // Mutation: drop the section that declared read requires.
    //
    //     AIRs  -> one read declared
    //     proof -> None
    proof.indexed = None;

    // Expected rejection: one variant reports either direction of the disagreement.
    // Why: a declared read with no reduction to close it leaves the batch incomplete.
    //   the opening stage would go on to demand claims that were never produced
    //   -> the proof is refused while the transcript can still be released cleanly.
    let err = verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&SquaresBatch::Reader, &vk, reader_log, &[]),
            VerifierInstance::new(&SquaresBatch::Table, &vk, table_log, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .unwrap_err();
    assert!(
        matches!(err, VerificationError::UnexpectedIndexedReduction),
        "expected UnexpectedIndexedReduction, got {err:?}"
    );
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
    let (_, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    let (pk, _) = setup(&config, &airs, &mut challenger()).unwrap();

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
    )
    .unwrap();

    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(WHIR_FIXTURE, &bytes)?;
    Ok(())
}

#[test]
fn prove_verify_fibonacci_boundary_io_roundtrips() {
    // Invariant: a satisfying trace binding its public inputs by position round-trips.
    //
    // The AIR asserts no boundary constraint of its own.
    // Every seed and output cell is bound by an injected pin instead.
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibIoAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibIoAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &FibIoAir, &vk, log_height, &pis,
        )]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("honest boundary-IO Fibonacci proof must verify");
}

#[test]
fn verify_rejects_wrong_claimed_output_boundary_io() {
    // Invariant: an honest trace is rejected when the claimed output is not the one it ends on.
    //
    // Both sides share the wrong claim, so the transcript stays in step.
    // Only the pin can reject, which is what this checks end to end.
    //
    // Fixture state: a valid length-256 trace, output claim shifted by one.
    let n = 256;
    let trace = fib_trace(n);
    let mut pis = fib_public_values(n);
    // Mutation: claim an output one off from the trace's true final value.
    pis[2] += F::ONE;
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibIoAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    // Both sides share the wrong public value, keeping the transcript in sync.
    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibIoAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();

    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &FibIoAir, &vk, log_height, &pis,
        )]),
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

/// A cell naming a column one past the last real column.
///
/// Prover and verifier each validate the AIR they are handed.
/// The keys carry no AIR, so nothing else ties the two declarations together.
const OUT_OF_RANGE_CELLS: [BoundaryPublic; 1] =
    [BoundaryPublic::new(NUM_COLS, BoundaryEnd::Last, 2)];

/// The same Fibonacci recurrence, carrying the out-of-range declaration above.
struct FibIoAirBadColumn;

impl<X> BaseAir<X> for FibIoAirBadColumn {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &OUT_OF_RANGE_CELLS
    }
}

impl<AB: AirBuilder> Air<AB> for FibIoAirBadColumn {
    fn eval(&self, builder: &mut AB) {
        // Identical constraints, with only the declaration differing.
        FibIoAir.eval(builder);
    }
}

#[test]
fn security_rejects_an_invalid_boundary_io_declaration() {
    // Invariant: the report is fail-closed on a statement `verify` would reject.
    //
    // Mutation: name a column one past the last real one.
    //
    //     a level reported here would describe a statement nothing accepts
    let config = config_for(4, NUM_COLS);
    let public = fib_public_values(16);
    let (_, vk) = setup(&config, &[&FibIoAir], &mut challenger()).unwrap();
    let instances = VerifierInstances::new(vec![VerifierInstance::new(
        &FibIoAirBadColumn,
        &vk,
        4,
        &public,
    )]);

    assert!(matches!(
        p3_multi_stark::security_report(&config, &instances),
        Err(p3_multi_stark::SecurityError::InvalidShape(_))
    ));
}

#[test]
fn verify_rejects_an_invalid_boundary_io_declaration() {
    // Invariant: a malformed declaration is reported, not indexed past an end.
    //
    // Fixture state: an honest proof replayed against an AIR with a bad declaration.
    //
    //     columns present : 0, 1
    //     column named    : 2
    //                       → rejected before the transcript is touched
    let n = 256;
    let trace = fib_trace(n);
    let pis = fib_public_values(n);
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, NUM_COLS);
    let airs = [&FibIoAir];

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibIoAir,
            Table::new(trace.transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();

    // The keys carry no AIR of their own.
    // The verifier can therefore be handed a different one than the prover used.
    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(
            &FibIoAirBadColumn,
            &vk,
            log_height,
            &pis,
        )]),
        &proof,
        0,
        &mut challenger(),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::BoundaryIo {
                instance: 0,
                error: BoundaryIoError::ColumnOutOfRange {
                    column: NUM_COLS,
                    width: NUM_COLS
                }
            }
        ),
        "expected an out-of-range boundary-IO column, got {err:?}"
    );
}

/// Enum AIR carrying one instance of each way to bind a public input.
///
/// A wrapper forwards [`BaseAir`] by hand, one method at a time.
/// Forwarding `width` but not `public_boundary_io` would leave every listed cell unbound,
/// and an empty list is valid, so nothing would report it.
enum MixedFibAir {
    /// Public inputs asserted by the AIR's own boundary constraints.
    Constrained(FibAir),
    /// Public inputs listed for the backend to pin.
    BoundaryIo(FibIoAir),
}

impl<X> BaseAir<X> for MixedFibAir {
    fn width(&self) -> usize {
        match self {
            Self::Constrained(air) => <FibAir as BaseAir<X>>::width(air),
            Self::BoundaryIo(air) => <FibIoAir as BaseAir<X>>::width(air),
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Constrained(air) => <FibAir as BaseAir<X>>::num_public_values(air),
            Self::BoundaryIo(air) => <FibIoAir as BaseAir<X>>::num_public_values(air),
        }
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        match self {
            Self::Constrained(air) => <FibAir as BaseAir<X>>::public_boundary_io(air),
            Self::BoundaryIo(air) => <FibIoAir as BaseAir<X>>::public_boundary_io(air),
        }
    }
}

impl<AB: AirBuilder> Air<AB> for MixedFibAir {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Constrained(air) => air.eval(builder),
            Self::BoundaryIo(air) => air.eval(builder),
        }
    }
}

/// Prove and verify one batch holding both bindings, under the given public values.
fn prove_verify_mixed_binding_batch(
    n: usize,
    pis_constrained: &[F],
    pis_boundary_io: &[F],
) -> Result<(), VerificationError<p3_whir::VerifierError>> {
    let log_height = log2_strict_usize(n);
    let constrained = MixedFibAir::Constrained(FibAir);
    let boundary_io = MixedFibAir::BoundaryIo(FibIoAir);
    let config = batch_config_for(log_height, NUM_COLS, 2);
    let airs = [&constrained, &boundary_io];

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let trace = fib_trace(n);
    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(
                &constrained,
                Table::new(trace.transpose()),
                &pk,
                pis_constrained,
            ),
            ProverInstance::new(
                &boundary_io,
                Table::new(trace.transpose()),
                &pk,
                pis_boundary_io,
            ),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap();

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&constrained, &vk, log_height, pis_constrained),
            VerifierInstance::new(&boundary_io, &vk, log_height, pis_boundary_io),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
}

#[test]
fn prove_verify_mixed_binding_batch_roundtrips() {
    // Invariant: one batch may hold an AIR that lists cells next to one that does not.
    //
    //     instance 0: boundary constraints, no listed cell
    //     instance 1: no boundary constraint, three listed cells
    //
    // Both run through the same commitment, zerocheck and opening.
    let n = 256;
    let pis = fib_public_values(n);

    prove_verify_mixed_binding_batch(n, &pis, &pis)
        .expect("honest mixed-binding batch must verify");
}

#[test]
fn verify_rejects_wrong_claimed_output_in_a_mixed_binding_batch() {
    // Invariant: the wrapper forwards the declaration, so the listed instance stays bound.
    //
    // Mutation: shift the listed instance's output claim only.
    //
    //     instance 0: untouched, its boundary constraints still hold
    //     instance 1: pin on the last row fails by one
    //
    // A wrapper that dropped `public_boundary_io` would accept this batch.
    let n = 256;
    let pis = fib_public_values(n);
    let mut wrong = pis;
    wrong[2] += F::ONE;

    let err = prove_verify_mixed_binding_batch(n, &pis, &wrong).unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected a zerocheck rejection, got {err:?}"
    );
}

#[test]
fn prove_verify_mixed_height_fibonacci_boundary_io_roundtrips() {
    // Invariant: instances of different heights each pin their own cells.
    //
    // Fixture state:
    //
    //     trace a: height 256 -> 8 variables
    //     trace b: height 128 -> 7 variables
    //
    // Each instance's selectors are drawn at its own suffix of the common point.
    let air = FibIoAir;
    let n_a = 256;
    let n_b = 128;
    let log_a = log2_strict_usize(n_a);
    let log_b = log2_strict_usize(n_b);
    let trace_a = fib_trace(n_a);
    let trace_b = fib_trace(n_b);
    let pis_a = fib_public_values_for_trace(&trace_a);
    let pis_b = fib_public_values_for_trace(&trace_b);

    // Size the config for the stacked cell count the layout planner computes.
    let cells = NUM_COLS * n_a + NUM_COLS * n_b;
    let mut config = config_for_stacked(log2_ceil_usize(cells));
    config.collision_bits = Some(100);
    let airs = [&air, &air];

    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

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
    .expect("honest mixed-height boundary-IO proof must verify");
}

/// The cells the permutation lookup AIR below binds by position.
///
/// ```text
///     column 0, last  row -> public value 0
///     column 1, first row -> public value 1
/// ```
const PERMUTATION_IO_CELLS: [BoundaryPublic; 2] = [
    BoundaryPublic::new(0, BoundaryEnd::Last, 0),
    BoundaryPublic::new(1, BoundaryEnd::First, 1),
];

/// The local permutation lookup, with two of its cells bound by position.
///
/// It asserts no constraint of its own:
///
/// ```text
///     lookup family   : the two-tuple permutation
///     ordinary family : the injected pins, and nothing else
/// ```
struct PermutationIoLookupAir;

impl BaseAir<F> for PermutationIoLookupAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn num_public_values(&self) -> usize {
        2
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &PERMUTATION_IO_CELLS
    }
}

impl<AB> Air<AB> for PermutationIoLookupAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Identical lookups, with only the declaration setting the two apart.
        LocalPermutationLookupAir.eval(builder);
    }
}

/// Public values an honest reversed trace of `n` rows carries at the declared cells.
///
/// ```text
///     column 0, last  row : n - 1
///     column 1, first row : n - 1
/// ```
fn permutation_io_public_values(n: usize) -> [F; 2] {
    [F::from_usize(n - 1), F::from_usize(n - 1)]
}

/// Prove the permutation lookup AIR with boundary IO, then verify under the same public values.
fn prove_verify_permutation_io(
    n: usize,
    pis: &[F; 2],
) -> Result<(), VerificationError<p3_whir::VerifierError>> {
    let log_height = log2_strict_usize(n);
    let air = PermutationIoLookupAir;
    let config = config_for(log_height, air.width());
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(permutation_trace(n).transpose()),
            &pk,
            pis,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();
    assert!(proof.lookup.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, pis)]),
        &proof,
        0,
        &mut challenger(),
    )
}

#[test]
fn prove_verify_lookup_boundary_io_roundtrips_through_pcs() {
    // Invariant: an AIR with lookups and boundary IO folds the pins in both folders.
    //
    //     pin degree 2 > lookup degree 1
    //     -> nodes 0, 1 : lookup-aware folder
    //     -> node  2    : ordinary folder
    //     -> the two must batch the same ordinary family
    let n = 64;
    prove_verify_permutation_io(n, &permutation_io_public_values(n))
        .expect("honest lookup proof with boundary IO must verify");
}

#[test]
fn verify_rejects_wrong_claimed_output_lookup_boundary_io() {
    // Invariant: a lookup AIR's boundary cells are bound even though the lookup alone passes.
    //
    // Mutation: claim column 0 ends one past the value the reversed trace ends on.
    //
    //     lookup : still a permutation  -> accepted on its own
    //     pin    : (n - 1) - n != 0     -> the zerocheck rejects
    let n = 64;
    let mut pis = permutation_io_public_values(n);
    pis[0] += F::ONE;

    let err = prove_verify_permutation_io(n, &pis).unwrap_err();
    assert!(
        matches!(
            err,
            VerificationError::Zerocheck(ZerocheckError::FinalSumMismatch)
        ),
        "expected a zerocheck final-sum mismatch, got {err:?}"
    );
}

#[test]
fn security_counts_boundary_io_pins_as_constraints() {
    // Invariant: the security report sees the constraints the folder batches, pins included.
    //
    //     constraint AIR  : 2 first-row + 2 transition + 1 last-row = 5 constraints, degree 2
    //     boundary-IO AIR : 2 transition + 3 pins                    = 5 constraints, degree 2
    //                       → identical batching and sumcheck terms
    let config = config_for(4, NUM_COLS);
    let public = fib_public_values(16);
    let term_bits = |report: &p3_multi_stark::MultiStarkSecurityReport, label: &str| {
        report
            .terms()
            .iter()
            .find(|term| term.label == label)
            .unwrap_or_else(|| panic!("missing {label}"))
            .bits
            .bits()
    };

    let (_, vk) = setup(&config, &[&FibAir], &mut challenger()).unwrap();
    let constraint = p3_multi_stark::security_report(
        &config,
        &VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, 4, &public)]),
    )
    .unwrap();

    let (_, vk) = setup(&config, &[&FibIoAir], &mut challenger()).unwrap();
    let boundary_io = p3_multi_stark::security_report(
        &config,
        &VerifierInstances::new(vec![VerifierInstance::new(&FibIoAir, &vk, 4, &public)]),
    )
    .unwrap();

    for label in ["constraint-batching", "constraint-sumcheck"] {
        assert_eq!(
            term_bits(&boundary_io, label),
            term_bits(&constraint, label),
            "{label}"
        );
    }
}

/// The one cell the constraint-free AIR below binds by position.
const OUTPUT_ONLY_CELLS: [BoundaryPublic; 1] = [BoundaryPublic::new(0, BoundaryEnd::Last, 0)];

/// AIR that asserts nothing and binds its only public value by position.
///
/// Its ordinary constraint family is the single injected pin.
struct OutputOnlyIoAir;

impl<X> BaseAir<X> for OutputOnlyIoAir {
    fn width(&self) -> usize {
        1
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn num_public_values(&self) -> usize {
        1
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &OUTPUT_ONLY_CELLS
    }
}

impl<AB: AirBuilder> Air<AB> for OutputOnlyIoAir {
    fn eval(&self, _builder: &mut AB) {
        // Empty: the public value is bound by position, not by a constraint.
    }
}

/// The same AIR shape, with one constant constraint of its own beside the listed cell.
///
/// A constant family carries no round polynomial, so on its own it is not a valid
/// statement. The pin supplies the degree the round polynomials are sized against.
struct ConstantFamilyIoAir;

impl<X> BaseAir<X> for ConstantFamilyIoAir {
    fn width(&self) -> usize {
        1
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn num_public_values(&self) -> usize {
        1
    }

    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &OUTPUT_ONLY_CELLS
    }
}

impl<AB: AirBuilder> Air<AB> for ConstantFamilyIoAir {
    fn eval(&self, builder: &mut AB) {
        // Degree zero, and satisfied on every row.
        builder.assert_zero(AB::Expr::ZERO);
    }
}

#[test]
fn security_checked_roundtrip_for_a_constant_family_lifted_by_a_pin() {
    // Invariant: `get_air_degrees` and `security_report` agree on what is a valid statement.
    //
    // A constant own family scores degree zero and has no round polynomial.
    // A listed cell injects a degree-two pin, which is what the rounds are sized for.
    //
    //     without the cell : rejected by both, the family has no degree
    //     with    the cell : accepted by both, the pin supplies it
    //
    // Scoring it in one place and rejecting it in the other would make an AIR that proves
    // and verifies fail every security-checked entry point.
    let n = 256;
    let log_height = log2_strict_usize(n);
    let mut config = config_for(log_height, 1);
    config.collision_bits = Some(100);
    let air = ConstantFamilyIoAir;
    let trace = RowMajorMatrix::new((0..n).map(F::from_usize).collect(), 1);
    let public = [F::from_usize(n - 1)];
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

    // The report must produce a level rather than reject the shape.
    p3_multi_stark::security_report(
        &config,
        &VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &public)]),
    )
    .expect("a constant family lifted by a pin is a valid shape");

    let proof = p3_multi_stark::prove_with_security(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &public,
        )]),
        0,
        20,
        &mut challenger(),
    )
    .expect("a constant family lifted by a pin must pass the security assessment");

    p3_multi_stark::verify_with_security(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &public)]),
        &proof,
        0,
        20,
        &mut challenger(),
    )
    .expect("honest proof must verify");
}

#[test]
fn verify_rejects_a_wrong_claim_against_a_constant_family_lifted_by_a_pin() {
    // Mutation: claim a last row the trace does not carry.
    //
    // The constant family says nothing about it, so only the pin can reject.
    let n = 256;
    let log_height = log2_strict_usize(n);
    let config = config_for(log_height, 1);
    let air = ConstantFamilyIoAir;
    let trace = RowMajorMatrix::new((0..n).map(F::from_usize).collect(), 1);
    let public = [F::from_usize(n)];
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &public,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();

    let err = verify(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &public)]),
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
        "expected a zerocheck final-sum mismatch, got {err:?}"
    );
}

#[test]
fn security_checked_roundtrip_for_an_air_bound_only_by_boundary_io() {
    // Invariant: a pin counts as a constraint, so an AIR asserting nothing else is a valid statement.
    //
    // Fixture state: one column counting up, its last row the public output.
    //
    //     rows          : [0, 1, ..., 255]
    //     public values : [255]
    //
    // A single column stacks to the trace arity alone.
    // The PCS opening folds a packed prefix, which needs arity >= folding + log2(SIMD width):
    //
    //     arity 8 >= 2 + 4    -> covers every packing width up to 16
    let n = 256;
    let log_height = log2_strict_usize(n);
    let mut config = config_for(log_height, 1);
    config.collision_bits = Some(100);
    let air = OutputOnlyIoAir;
    let trace = RowMajorMatrix::new((0..n).map(F::from_usize).collect(), 1);
    let public = [F::from_usize(n - 1)];
    let (pk, vk) = setup(&config, &[&air], &mut challenger()).unwrap();

    let proof = p3_multi_stark::prove_with_security(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &air,
            Table::new(trace.transpose()),
            &pk,
            &public,
        )]),
        0,
        20,
        &mut challenger(),
    )
    .expect("a pin-only AIR must pass the security assessment");

    p3_multi_stark::verify_with_security(
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, log_height, &public)]),
        &proof,
        0,
        20,
        &mut challenger(),
    )
    .expect("honest pin-only proof must verify");
}

/// The two AIRs of the indexed-lookup batch, so one batch can hold both.
enum SquaresBatch {
    /// Provides the table out of its main trace.
    Table,
    /// Reads the table, naming an entry per row.
    Reader,
}

impl BaseAir<F> for SquaresBatch {
    fn width(&self) -> usize {
        match self {
            Self::Table => 1,
            Self::Reader => 2,
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB> Air<AB> for SquaresBatch
where
    AB: AirBuilder<F = F> + IndexedLookupBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Every AIR owes the zerocheck a constraint, and both traces open at zero.
        //
        //     table  : entry 0 squares to 0
        //     reader : the first row names entry 0
        let main = builder.main();
        let local = main.current_slice();
        builder.when_first_row().assert_zero(local[0]);

        match self {
            // The table's single column carries one value per entry.
            Self::Table => builder.push_indexed_table("squares", TraceWindow::Main, [0]),
            // Column 0 names the entry, column 1 holds what that row pulled.
            Self::Reader => builder.push_indexed_read("squares", 0, [1]),
        }
    }
}

#[test]
fn prove_verify_indexed_lookup_roundtrips_through_pcs() {
    // Fixture state: a four-entry table of squares, read by an eight-row reader.
    //
    //     table  : entry  0 1 2 3   ->  value  0 1 4 9
    //     reader : names  0 1 2 3 3 2 1 0
    //              holds  0 1 4 9 9 4 1 0
    //
    // Every entry is read exactly twice.
    //
    // A logarithmic-derivative lookup reads counts modulo the characteristic.
    //
    // Modulo two it could not tell two reads from none.
    //
    // That is the failure this reduction exists to avoid.
    //
    // The table grows with the target's packing, because the stacked commitment needs a
    // full packed element per prefix variable below the padding floor.
    //
    //     scalar   4 entries, 8 reader rows
    //     avx2     8 entries, 16 reader rows
    //     avx512  16 entries, 32 reader rows
    let table_rows = ((1 << FOLDING) * PackedF::WIDTH / 4).max(1 << FOLDING);
    let reader_rows = 2 * table_rows;
    let table_log = log2_strict_usize(table_rows);
    let reader_log = log2_strict_usize(reader_rows);

    let squares = RowMajorMatrix::new((0..table_rows).map(|v| F::from_usize(v * v)).collect(), 1);

    // Each entry is named once on the way up and once on the way down.
    //
    //     names  0 1 .. n-1 n-1 .. 1 0
    let named = (0..table_rows)
        .chain((0..table_rows).rev())
        .collect::<Vec<_>>();

    // A position column holds the entry under the reduction's embedding.
    //
    // Over a prime field that embedding is the entry itself.
    let reads = RowMajorMatrix::new(
        named
            .iter()
            .flat_map(|&v| [F::from_usize(v), F::from_usize(v * v)])
            .collect(),
        2,
    );

    let reader = SquaresBatch::Reader;
    let table = SquaresBatch::Table;
    // Both traces are stacked into one committed polynomial, two reader columns beside
    // the table's one.
    let stacked_num_variables = log2_ceil_usize(2 * reader_rows + table_rows);
    let config = config_for_stacked(stacked_num_variables);
    let (pk, vk) = setup(&config, &[&reader, &table], &mut challenger()).unwrap();

    let proof = prove(
        &config,
        ProverInstances::new(vec![
            ProverInstance::new(&reader, Table::new(reads.transpose()), &pk, &[]),
            ProverInstance::new(&table, Table::new(squares.transpose()), &pk, &[]),
        ]),
        0,
        &mut challenger(),
    )
    .unwrap();

    // An AIR declaring an indexed read must produce a reduction section in the proof.
    assert!(proof.indexed.is_some());

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&reader, &vk, reader_log, &[]),
            VerifierInstance::new(&table, &vk, table_log, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect("an honest indexed lookup must verify through the trace PCS opening");

    // The same statement, priced.
    //
    // Every soundness term the reduction adds has to reach the report, or a batch reading a
    // table would be charged as if it read nothing.
    let instances = VerifierInstances::new(vec![
        VerifierInstance::new(&reader, &vk, reader_log, &[]),
        VerifierInstance::new(&table, &vk, table_log, &[]),
    ]);
    let report = p3_multi_stark::security_report(&config, &instances).unwrap();
    let charged = report
        .terms()
        .iter()
        .map(|term| term.label)
        .filter(|label| label.starts_with("logup-star-"))
        .collect::<Vec<_>>();

    // One per challenge the reduction draws, and one per point it closes at.
    //
    // Batching is priced by how much there is to batch, and this batch has one reader
    // pulling one column, so neither batching term costs anything here.
    assert_eq!(
        charged,
        [
            "logup-star-entry-challenge",
            "logup-star-claim-point",
            "logup-star-fractional-gkr",
            "logup-star-product-sumcheck",
        ]
    );

    // Each one costs something, so none of them is a placeholder that prices nothing.
    assert!(
        report
            .terms()
            .iter()
            .filter(|term| term.label.starts_with("logup-star-"))
            .all(|term| term.bits.bits() > 0.0)
    );
}

#[test]
fn a_batch_declaring_no_read_is_charged_for_no_reduction() {
    // The counterpart to the round trip above, through the same entry point, on a batch
    // whose AIR declares no indexed read.
    //
    // Nothing the reduction would charge may appear, or every batch would pay for it.
    let config = config_for(4, NUM_COLS);
    let air = FibAir;
    let (_, vk) = setup(&config, &[&air], &mut challenger()).unwrap();
    let public = fib_public_values(16);

    let instances = VerifierInstances::new(vec![VerifierInstance::new(&air, &vk, 4, &public)]);
    let report = p3_multi_stark::security_report(&config, &instances).unwrap();

    assert!(
        !report
            .terms()
            .iter()
            .any(|term| term.label.starts_with("logup-star-"))
    );
}

#[test]
fn a_reader_pulling_the_wrong_value_is_rejected() {
    // Invariant: a row's pulled value has to be what the table holds at the entry it names.
    //
    // Fixture state: the same squares table and eight-row reader.
    //
    // Mutation: row 5 names entry 2 but holds 5 instead of 4.
    //
    //     honest : names 2 -> holds 4
    //     forged : names 2 -> holds 5
    //
    // The position column is untouched, so the pushforward is the honest one.
    //
    // What breaks is the claim tying the table's columns to it.
    let table_rows = 4usize;
    let reader_rows = 8usize;
    let table_log = log2_strict_usize(table_rows);
    let reader_log = log2_strict_usize(reader_rows);

    let squares = RowMajorMatrix::new((0..table_rows).map(|v| F::from_usize(v * v)).collect(), 1);

    let named = [0usize, 1, 2, 3, 3, 2, 1, 0];
    let mut values = named
        .iter()
        .flat_map(|&v| [F::from_usize(v), F::from_usize(v * v)])
        .collect::<Vec<_>>();
    values[11] = F::from_usize(5);
    let reads = RowMajorMatrix::new(values, 2);

    let reader = SquaresBatch::Reader;
    let table = SquaresBatch::Table;
    let stacked_num_variables = log2_ceil_usize(2 * reader_rows + table_rows);
    let config = config_for_stacked(stacked_num_variables);
    let (pk, vk) = setup(&config, &[&reader, &table], &mut challenger()).unwrap();

    // This pins the prover's refusal, not the verifier's.
    //
    // A wrong pulled value fails the prover's own claim check before a proof exists.
    let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(&reader, Table::new(reads.transpose()), &pk, &[]),
                ProverInstance::new(&table, Table::new(squares.transpose()), &pk, &[]),
            ]),
            0,
            &mut challenger(),
        )
    }));

    let Ok(Ok(proof)) = proved else {
        // Rejected while proving, which is the honest outcome for a false statement.
        return;
    };

    // Should a proof come out anyway, no verifier may take it.
    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&reader, &vk, reader_log, &[]),
            VerifierInstance::new(&table, &vk, table_log, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect_err("a reader pulling a value the table does not hold must be rejected");
}

#[test]
fn a_row_naming_an_entry_its_table_does_not_have_is_rejected() {
    // Invariant: a row may only name an entry of the table it reads.
    //
    // Fixture state: the four-entry squares table, read by an eight-row AIR.
    //
    // Mutation: row 7 names entry 5, one past the table.
    //
    //     honest : names 0 1 2 3 3 2 1 0
    //     forged : names 0 1 2 3 3 2 1 5
    //
    // Entry 5 has no embedding in this table, so no scatter can place that row.
    //
    // Row 7 rather than row 0, which the AIR pins to zero.
    //
    // Entry 5 panics while the witness is built either way, but a proof naming it on row 0
    // would fail the zerocheck close on its own.
    //
    // The verifier would then reject it whether or not the position comparison exists.
    let table_rows = 4usize;
    let reader_rows = 8usize;
    let reader_log = log2_strict_usize(reader_rows);
    let table_log = log2_strict_usize(table_rows);

    let squares = RowMajorMatrix::new((0..table_rows).map(|v| F::from_usize(v * v)).collect(), 1);

    let named = [0usize, 1, 2, 3, 3, 2, 1, 5];
    let reads = RowMajorMatrix::new(
        named
            .iter()
            .flat_map(|&v| [F::from_usize(v), F::from_usize(v * v)])
            .collect(),
        2,
    );

    let reader = SquaresBatch::Reader;
    let table = SquaresBatch::Table;
    let stacked_num_variables = log2_ceil_usize(2 * reader_rows + table_rows);
    let config = config_for_stacked(stacked_num_variables);
    let (pk, vk) = setup(&config, &[&reader, &table], &mut challenger()).unwrap();

    let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(&reader, Table::new(reads.transpose()), &pk, &[]),
                ProverInstance::new(&table, Table::new(squares.transpose()), &pk, &[]),
            ]),
            0,
            &mut challenger(),
        )
    }));

    let Ok(Ok(proof)) = proved else {
        // Rejected while proving, which is where an unmatched entry surfaces.
        return;
    };

    verify(
        &config,
        VerifierInstances::new(vec![
            VerifierInstance::new(&reader, &vk, reader_log, &[]),
            VerifierInstance::new(&table, &vk, table_log, &[]),
        ]),
        &proof,
        0,
        &mut challenger(),
    )
    .expect_err("a row naming an entry outside its table must be rejected");
}
