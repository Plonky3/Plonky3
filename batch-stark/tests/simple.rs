use core::borrow::Borrow;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::{CommonData, StarkInstance, VerificationError, prove_batch, verify_batch};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs, create_test_fri_params};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::StarkConfig;
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// --- Simple Fibonacci AIR and trace ---

#[derive(Debug, Clone, Copy)]
struct FibonacciAir {
    /// log2 of the trace height; used to size preprocessed columns.
    log_height: usize,
    /// Index to tamper with in preprocessed trace (None = no tampering).
    tamper_index: Option<usize>,
}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let a0 = pis[0];
        let b0 = pis[1];
        let x = pis[2];

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibRow<AB::Var> = (*local).borrow();
        let next: &FibRow<AB::Var> = (*next).borrow();

        let mut wf = builder.when_first_row();
        wf.assert_eq(local.left.clone(), a0);
        wf.assert_eq(local.right.clone(), b0);

        let mut wt = builder.when_transition();
        wt.assert_eq(local.right.clone(), next.left.clone());
        wt.assert_eq(local.left.clone() + local.right.clone(), next.right.clone());

        builder.when_last_row().assert_eq(local.right.clone(), x);
    }
}

#[derive(Clone, Copy, Debug)]
struct FibRow<F> {
    left: F,
    right: F,
}
impl<F> Borrow<FibRow<F>> for [F] {
    fn borrow(&self) -> &FibRow<F> {
        debug_assert_eq!(self.len(), 2);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibRow<F>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

fn fib_trace<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());
    let mut trace = RowMajorMatrix::new(F::zero_vec(n * 2), 2);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibRow<F>>() };
    assert!(prefix.is_empty() && suffix.is_empty());
    rows[0] = FibRow {
        left: F::from_u64(a),
        right: F::from_u64(b),
    };
    for i in 1..n {
        rows[i] = FibRow {
            left: rows[i - 1].right,
            right: rows[i - 1].left + rows[i - 1].right,
        };
    }
    trace
}

fn fib_n(n: usize) -> u64 {
    fib_n_from(0, 1, n)
}

fn fib_n_from(a0: u64, b0: u64, n: usize) -> u64 {
    let mut a = a0;
    let mut b = b0;
    for _ in 0..n {
        let t = a + b;
        a = b;
        b = t;
    }
    a
}

// --- Simple multiplication AIR and trace ---

#[derive(Debug, Clone, Copy)]
struct MulAir {
    reps: usize,
}
impl Default for MulAir {
    fn default() -> Self {
        Self { reps: 2 }
    }
}
impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        self.reps * 3
    }
}
impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();
        for i in 0..self.reps {
            let s = i * 3;
            let a = local[s].clone();
            let b = local[s + 1].clone();
            let c = local[s + 2].clone();
            builder.assert_eq(a.clone() * b.clone(), c);
            builder
                .when_transition()
                .assert_eq(a + AB::Expr::ONE, next[s].clone());
        }
    }
}

fn mul_trace<F: Field>(rows: usize, reps: usize) -> RowMajorMatrix<F> {
    assert!(rows.is_power_of_two());
    let w = reps * 3;
    let mut v = F::zero_vec(rows * w);
    // Keep a simple constant b and c = a*b
    for i in 0..rows {
        for rep in 0..reps {
            let idx = i * w + rep * 3;
            v[idx] = F::from_u64(i as u64);
            v[idx + 1] = F::from_u64(3);
            v[idx + 2] = v[idx] * v[idx + 1];
        }
    }
    RowMajorMatrix::new(v, w)
}

// --- Preprocessed multiplication AIR and trace ---

#[derive(Debug, Clone, Copy)]
struct PreprocessedMulAir {
    /// log2 of the trace height; used to size preprocessed columns.
    log_height: usize,
    /// Multiplier to use in constraint (2 for correct, 3 for incorrect test).
    multiplier: u64,
}

impl<F: Field> BaseAir<F> for PreprocessedMulAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let n = 1 << self.log_height;
        let mut m = RowMajorMatrix::new(F::zero_vec(n), 1);
        for (i, v) in m.values.iter_mut().enumerate().take(n) {
            *v = F::from_u64(i as u64);
        }
        Some(m)
    }
}

impl<AB> Air<AB> for PreprocessedMulAir
where
    AB: AirBuilder + PairBuilder,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let local_main = main.row_slice(0).expect("Matrix is empty?");
        let local_prep = preprocessed.row_slice(0).expect("Preprocessed is empty?");

        // Enforce: main[0] = multiplier * preprocessed[0]
        builder.assert_eq(
            local_main[0].clone(),
            local_prep[0].clone() * AB::Expr::from_u64(self.multiplier),
        );
    }
}

fn preprocessed_mul_trace<F: Field>(rows: usize, multiplier: u64) -> RowMajorMatrix<F> {
    assert!(rows.is_power_of_two());
    let mut v = F::zero_vec(rows);
    // main[0] = multiplier * preprocessed[0], where preprocessed[0] = row_index
    for (i, val) in v.iter_mut().enumerate() {
        *val = F::from_u64(i as u64 * multiplier);
    }
    RowMajorMatrix::new(v, 1)
}

// --- Config types ---

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type HidingValMmcs = MerkleTreeHidingMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    SmallRng,
    8,
    4,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type HidingChallengeMmcs = ExtensionMmcs<Val, Challenge, HidingValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type HidingPcs = HidingFriPcs<Val, Dft, HidingValMmcs, HidingChallengeMmcs, SmallRng>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
type MyHidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;

fn make_config(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

fn make_config_zk(seed: u64) -> MyHidingConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = HidingValMmcs::new(hash, compress, rng.clone());
    let challenge_mmcs = HidingChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = HidingPcs::new(dft, val_mmcs, fri_params, 4, rng);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

// Heterogeneous enum wrapper for batching
#[derive(Clone, Copy)]
enum DemoAir {
    Fib(FibonacciAir),
    Mul(MulAir),
    PreprocessedMul(PreprocessedMulAir),
}
impl<F: PrimeField64> BaseAir<F> for DemoAir {
    fn width(&self) -> usize {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::width(a),
            Self::Mul(a) => <MulAir as BaseAir<F>>::width(a),
            Self::PreprocessedMul(a) => <PreprocessedMulAir as BaseAir<F>>::width(a),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::Fib(a) => {
                let n = 1 << a.log_height;
                let mut m = RowMajorMatrix::new(F::zero_vec(n), 1);
                for (i, v) in m.values.iter_mut().enumerate().take(n) {
                    *v = F::from_u64(i as u64);
                }
                if let Some(idx) = a.tamper_index
                    && idx < n
                {
                    m.values[idx] += F::ONE;
                }
                Some(m)
            }
            Self::Mul(_) => None,
            Self::PreprocessedMul(a) => <PreprocessedMulAir as BaseAir<F>>::preprocessed_trace(a),
        }
    }
}
impl<AB> Air<AB> for DemoAir
where
    AB: AirBuilderWithPublicValues + PairBuilder,
    AB::F: PrimeField64,
{
    fn eval(&self, b: &mut AB) {
        match self {
            Self::Fib(a) => a.eval(b),
            Self::Mul(a) => a.eval(b),
            Self::PreprocessedMul(a) => a.eval(b),
        }
    }
}

// --- Test Helper Functions ---

/// Creates a Fibonacci instance with specified log height.
fn create_fib_instance(log_height: usize) -> (DemoAir, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let air = DemoAir::Fib(FibonacciAir {
        log_height,
        tamper_index: None,
    });
    let trace = fib_trace::<Val>(0, 1, n);
    let pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    (air, trace, pis)
}

/// Creates a multiplication instance with specified configuration.
fn create_mul_instance(log_height: usize, reps: usize) -> (DemoAir, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let mul = MulAir { reps };
    let air = DemoAir::Mul(mul);
    let trace = mul_trace::<Val>(n, reps);
    let pis = vec![];
    (air, trace, pis)
}

/// Creates a preprocessed multiplication instance with specified configuration.
fn create_preprocessed_mul_instance(
    log_height: usize,
    multiplier: u64,
) -> (DemoAir, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let air = DemoAir::PreprocessedMul(PreprocessedMulAir {
        log_height,
        multiplier,
    });
    let trace = preprocessed_mul_trace::<Val>(n, multiplier);
    let pis = vec![];
    (air, trace, pis)
}

#[test]
fn test_two_instances() -> Result<(), impl Debug> {
    let config = make_config(1337);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, 2 reps

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common)
}

#[test]
fn test_two_instances_zk() -> Result<(), impl Debug> {
    let config = make_config_zk(1337);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, 2 reps

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common)
}

#[test]
fn test_three_instances_mixed_sizes() -> Result<(), impl Debug> {
    let config = make_config(2025);

    let (air_fib16, fib16_trace, fib16_pis) = create_fib_instance(4); // 16 rows
    let (air_mul8, mul8_trace, mul8_pis) = create_mul_instance(3, 2); // 8 rows
    let (air_fib8, fib8_trace, fib8_pis) = create_fib_instance(3); // 8 rows

    let instances = vec![
        StarkInstance {
            air: &air_fib16,
            trace: fib16_trace,
            public_values: fib16_pis.clone(),
        },
        StarkInstance {
            air: &air_mul8,
            trace: mul8_trace,
            public_values: mul8_pis.clone(),
        },
        StarkInstance {
            air: &air_fib8,
            trace: fib8_trace,
            public_values: fib8_pis.clone(),
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);
    let airs = vec![air_fib16, air_mul8, air_fib8];
    let pvs = vec![fib16_pis, mul8_pis, fib8_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common)
}

#[test]
fn test_invalid_public_values_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(7);

    let (air_fib, trace, fib_pis) = create_fib_instance(4); // 16 rows
    let correct_x = fib_n(16);

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace,
        public_values: fib_pis,
    }];
    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    // Wrong public value at verify => should reject
    let airs = vec![air_fib];
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_batch(&config, &airs, &proof, &wrong_pvs, &common);
    assert!(res.is_err(), "Should reject wrong public values");
    Ok::<_, Box<dyn std::error::Error>>(())
}

#[test]
fn test_different_widths() -> Result<(), impl Debug> {
    let config = make_config(4242);

    // Mul with reps=2 (width=6) and reps=3 (width=9)
    let (air_mul2, mul2_trace, mul2_pis) = create_mul_instance(3, 2); // 8 rows, width=6
    let (air_fib, fib_trace, fib_pis) = create_fib_instance(3); // 8 rows, width=2
    let (air_mul3, mul3_trace, mul3_pis) = create_mul_instance(4, 3); // 16 rows, width=9

    let instances = vec![
        StarkInstance {
            air: &air_mul2,
            trace: mul2_trace,
            public_values: mul2_pis.clone(),
        },
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul3,
            trace: mul3_trace,
            public_values: mul3_pis.clone(),
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul2_pis, fib_pis, mul3_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common)
}

#[test]
fn test_preprocessed_tampered_fails() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(9999);

    // Single Fibonacci instance with 8 rows and preprocessed index column.
    let (air, trace, fib_pis) = create_fib_instance(3);
    let instances = vec![StarkInstance {
        air: &air,
        trace,
        public_values: fib_pis.clone(),
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    // First, sanity-check that verification succeeds with matching preprocessed data.
    let airs = vec![air];
    let ok_res = verify_batch(&config, &airs, &proof, from_ref(&fib_pis), &common);
    assert!(
        ok_res.is_ok(),
        "Expected verification to succeed with matching preprocessed data"
    );

    // Now tamper with the preprocessed trace by modifying the tamper_index in the AIR
    // used to derive the preprocessed commitment for verification.
    // The proof was generated with the original AIR, but we verify with a tampered AIR
    // that would produce different preprocessed columns.
    let air_tampered = DemoAir::Fib(FibonacciAir {
        log_height: 3,
        tamper_index: Some(2),
    });
    // Create CommonData with tampered AIR to test verification failure
    // Use the proof's degree_bits (which are log_degrees since ZK is not supported)
    let degree_bits = proof.degree_bits.clone();
    let airs_tampered = vec![air_tampered];
    let verify_common_tampered =
        CommonData::from_airs_and_degrees(&config, &airs_tampered, &degree_bits);

    let res = verify_batch(
        &config,
        &airs_tampered,
        &proof,
        &[fib_pis],
        &verify_common_tampered,
    );
    assert!(
        res.is_err(),
        "Verification should fail with tampered preprocessed columns"
    );
    Ok(())
}

#[test]
fn test_preprocessed_reuse_common_multi_proofs() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(2026);

    // Single Fibonacci instance with preprocessed index column, 8 rows.
    let log_height = 3;
    let n = 1 << log_height;
    let air = DemoAir::Fib(FibonacciAir {
        log_height,
        tamper_index: None,
    });

    // First proof: standard Fibonacci trace starting from (0, 1).
    let trace1 = fib_trace::<Val>(0, 1, n);
    let fib_pis1 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    let instances1 = vec![StarkInstance {
        air: &air,
        trace: trace1,
        public_values: fib_pis1.clone(),
    }];
    let common = CommonData::from_instances(&config, &instances1);
    let proof1 = prove_batch(&config, instances1, &common);

    // Verify the first proof.
    let airs = vec![air];
    let res1 = verify_batch(&config, &airs, &proof1, from_ref(&fib_pis1), &common);
    assert!(res1.is_ok(), "First verification should succeed");

    // Second proof: DIFFERENT initial values (2, 3) - demonstrates CommonData is truly reusable
    // across different traces with the same AIR and degree.
    let trace2 = fib_trace::<Val>(2, 3, n);
    let fib_pis2 = vec![
        Val::from_u64(2),
        Val::from_u64(3),
        Val::from_u64(fib_n_from(2, 3, n)),
    ];
    let instances2 = vec![StarkInstance {
        air: &air,
        trace: trace2,
        public_values: fib_pis2.clone(),
    }];
    let proof2 = prove_batch(&config, instances2, &common);

    let res2 = verify_batch(&config, &airs, &proof2, &[fib_pis2], &common);
    assert!(
        res2.is_ok(),
        "Second verification should succeed with different trace values"
    );

    Ok(())
}

#[test]
fn test_single_instance() -> Result<(), impl Debug> {
    // Single-instance sanity test: prove and verify a single Fibonacci instance.
    let config = make_config(9999);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(5); // 32 rows

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: fib_trace,
        public_values: fib_pis.clone(),
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);
    let airs = vec![air_fib];
    verify_batch(&config, &airs, &proof, &[fib_pis], &common)
}

#[test]
fn test_mixed_preprocessed() -> Result<(), impl Debug> {
    let config = make_config(8888);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows, has preprocessed
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, no preprocessed

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);
    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common)
}

#[test]
fn test_invalid_trace_width_rejected() {
    // This test verifies that the verifier rejects proofs with incorrect trace width.
    use p3_batch_stark::proof::{BatchCommitments, BatchOpenedValues};
    use p3_uni_stark::OpenedValues;

    let config = make_config(55555);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: fib_trace,
        public_values: fib_pis.clone(),
    }];

    // Generate a valid proof
    let common = CommonData::from_instances(&config, &instances);
    let valid_proof = prove_batch(&config, instances, &common);

    // Tamper with the proof: change trace_local to have wrong width
    let mut tampered_proof = p3_batch_stark::proof::BatchProof {
        commitments: BatchCommitments {
            main: valid_proof.commitments.main,
            quotient_chunks: valid_proof.commitments.quotient_chunks,
        },
        opened_values: BatchOpenedValues {
            instances: vec![OpenedValues {
                trace_local: vec![valid_proof.opened_values.instances[0].trace_local[0]], // Wrong width: 1 instead of 2
                trace_next: valid_proof.opened_values.instances[0].trace_next.clone(),
                preprocessed_local: None,
                preprocessed_next: None,
                quotient_chunks: valid_proof.opened_values.instances[0]
                    .quotient_chunks
                    .clone(),
                random: None,
            }],
        },
        opening_proof: valid_proof.opening_proof.clone(),
        degree_bits: valid_proof.degree_bits.clone(),
    };

    // Verification should fail due to width mismatch
    let airs = vec![air_fib];
    let res = verify_batch(&config, &airs, &tampered_proof, from_ref(&fib_pis), &common);
    assert!(
        res.is_err(),
        "Verifier should reject trace with wrong width"
    );

    // Also test wrong trace_next width
    tampered_proof.opened_values.instances[0].trace_local =
        valid_proof.opened_values.instances[0].trace_local.clone();
    tampered_proof.opened_values.instances[0].trace_next =
        vec![valid_proof.opened_values.instances[0].trace_next[0]]; // Wrong width

    let res = verify_batch(&config, &airs, &tampered_proof, from_ref(&fib_pis), &common);
    assert!(
        res.is_err(),
        "Verifier should reject trace_next with wrong width"
    );
}

#[test]
fn test_reorder_instances_rejected() {
    // Reordering AIRs at verify should break transcript binding and be rejected.
    let config = make_config(123);

    let (air_a, tr_a, pv_a) = create_fib_instance(4);
    let (air_b, tr_b, pv_b) = create_mul_instance(4, 2);

    let instances = vec![
        StarkInstance {
            air: &air_a,
            trace: tr_a,
            public_values: pv_a.clone(),
        },
        StarkInstance {
            air: &air_b,
            trace: tr_b,
            public_values: pv_b.clone(),
        },
    ];

    // DemoAir::Fib has preprocessed columns, so compute degrees for swapped verification
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    // Swap order at verify -> should fail (create new CommonData with swapped AIRs)
    let airs_swapped = vec![air_b, air_a];
    let common_swapped = CommonData::from_airs_and_degrees(&config, &airs_swapped, &log_degrees);
    let res = verify_batch(
        &config,
        &airs_swapped,
        &proof,
        &[pv_b, pv_a],
        &common_swapped,
    );
    assert!(res.is_err(), "Verifier should reject reordered instances");
}

#[test]
fn test_quotient_chunk_element_len_rejected() {
    // Truncating an element from a quotient chunk should be rejected by shape checks.
    use core::slice::from_ref;

    let config = make_config(321);
    let (air, tr, pv) = create_fib_instance(4);

    let instances = vec![StarkInstance {
        air: &air,
        trace: tr,
        public_values: pv.clone(),
    }];
    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    let mut tampered = proof;
    tampered.opened_values.instances[0].quotient_chunks[0].pop();

    let airs = vec![air];
    let res = verify_batch(&config, &airs, &tampered, from_ref(&pv), &common);
    assert!(
        res.is_err(),
        "Verifier should reject truncated quotient chunk element"
    );
}

#[test]
fn test_circle_stark_batch() -> Result<(), impl Debug> {
    // Test batch-stark with Circle PCS (non-two-adic field)
    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 40,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    // Create two Fibonacci instances with different sizes.
    // Here we don't use preprocessed columns (Circle PCS + plain FibonacciAir).
    let air_fib1 = FibonacciAir {
        log_height: 0,
        tamper_index: None,
    };
    let air_fib2 = FibonacciAir {
        log_height: 0,
        tamper_index: None,
    };

    let fib_pis1 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(8))]; // F_8 = 21
    let fib_pis2 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(4))]; // F_4 = 3

    let trace1 = fib_trace::<Val>(0, 1, 8);
    let trace2 = fib_trace::<Val>(0, 1, 4);

    let instances = vec![
        StarkInstance {
            air: &air_fib1,
            trace: trace1,
            public_values: fib_pis1.clone(),
        },
        StarkInstance {
            air: &air_fib2,
            trace: trace2,
            public_values: fib_pis2.clone(),
        },
    ];

    // Generate batch-proof
    // Plain FibonacciAir doesn't have preprocessed columns
    let airs = vec![air_fib1, air_fib2];
    let common = CommonData::empty(airs.len());
    let proof = prove_batch(&config, instances, &common);

    // Verify batch-proof
    let public_values = vec![fib_pis1, fib_pis2];
    verify_batch(&config, &airs, &proof, &public_values, &common)
        .map_err(|e| format!("Verification failed: {:?}", e))
}

#[test]
fn test_preprocessed_constraint_positive() -> Result<(), impl Debug> {
    // Test that preprocessed columns are correctly used in constraints
    // Enforces: main[0] = 2 * preprocessed[0]
    let config = make_config(8888);

    let (air, trace, pis) = create_preprocessed_mul_instance(4, 2); // 16 rows, multiplier=2

    let instances = vec![StarkInstance {
        air: &air,
        trace,
        public_values: pis.clone(),
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);
    let airs = vec![air];
    verify_batch(&config, &airs, &proof, &[pis], &common)
}

#[test]
fn test_preprocessed_constraint_negative() -> Result<(), Box<dyn std::error::Error>> {
    // Test that incorrect preprocessed constraints are caught via OOD evaluation mismatch
    // Proof is generated with multiplier=2, but verification uses multiplier=3
    let config = make_config(9999);

    // Generate proof with multiplier=2
    let (air_prove, trace, pis) = create_preprocessed_mul_instance(4, 2); // 16 rows, multiplier=2

    let instances = vec![StarkInstance {
        air: &air_prove,
        trace,
        public_values: pis.clone(),
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    // Verify with wrong multiplier=3 (should fail)
    let air_verify = DemoAir::PreprocessedMul(PreprocessedMulAir {
        log_height: 4,
        multiplier: 3, // Wrong multiplier!
    });
    let airs = vec![air_verify];
    let degree_bits = proof.degree_bits.clone();
    let verify_common = CommonData::from_airs_and_degrees(&config, &airs, &degree_bits);

    let res = verify_batch(&config, &airs, &proof, &[pis], &verify_common);
    let err = res.expect_err(
        "Verification should fail when preprocessed constraint multiplier doesn't match",
    );
    match err {
        VerificationError::OodEvaluationMismatch { .. } => (),
        _ => panic!("unexpected error: {err:?}"),
    }
    Ok(())
}

#[test]
fn test_mixed_preprocessed_constraints() -> Result<(), impl Debug> {
    // Test batching PreprocessedMulAir (uses pp in constraints) with MulAir and FibonacciAir
    // This exercises matrix_to_instance routing and point-scheduling with heterogeneous instances
    // while preprocessed values actually affect constraints.
    let config = make_config(1111);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows, has pp but doesn't use in constraints
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, no pp
    let (air_pp_mul, pp_mul_trace, pp_mul_pis) = create_preprocessed_mul_instance(4, 2); // 16 rows, uses pp in constraints

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
        },
        StarkInstance {
            air: &air_pp_mul,
            trace: pp_mul_trace,
            public_values: pp_mul_pis.clone(),
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch(&config, instances, &common);

    let airs = vec![air_fib, air_mul, air_pp_mul];
    let pvs = vec![fib_pis, mul_pis, pp_mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common)
}
