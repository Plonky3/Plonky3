use core::borrow::Borrow;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::{StarkInstance, prove_batch, verify_batch};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs, create_test_fri_params};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::StarkConfig;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// --- Simple Fibonacci AIR and trace ---

#[derive(Debug, Clone, Copy)]
struct FibonacciAir;

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
    let mut a = 0u64;
    let mut b = 1u64;
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
    step: u64,
}
impl Default for MulAir {
    fn default() -> Self {
        Self { reps: 2, step: 1 }
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
                .assert_eq(a + AB::Expr::from_u64(self.step), next[s].clone());
        }
    }
}

fn mul_trace<F: Field>(rows: usize, reps: usize, _step: u64) -> RowMajorMatrix<F> {
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

// --- Config types ---

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

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

// Heterogeneous enum wrapper for batching
#[derive(Clone, Copy)]
enum DemoAir {
    Fib(FibonacciAir),
    Mul(MulAir),
}
impl<F> BaseAir<F> for DemoAir {
    fn width(&self) -> usize {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::width(a),
            Self::Mul(a) => <MulAir as BaseAir<F>>::width(a),
        }
    }
}
impl<AB: AirBuilderWithPublicValues> Air<AB> for DemoAir {
    fn eval(&self, b: &mut AB) {
        match self {
            Self::Fib(a) => a.eval(b),
            Self::Mul(a) => a.eval(b),
        }
    }
}

// --- Test Helper Functions ---

/// Creates a Fibonacci instance with specified log height.
fn create_fib_instance(log_height: usize) -> (DemoAir, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let air = DemoAir::Fib(FibonacciAir);
    let trace = fib_trace::<Val>(0, 1, n);
    let pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    (air, trace, pis)
}

/// Creates a multiplication instance with specified configuration.
fn create_mul_instance(
    log_height: usize,
    reps: usize,
    step: u64,
) -> (DemoAir, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let mul = MulAir { reps, step };
    let air = DemoAir::Mul(mul);
    let trace = mul_trace::<Val>(n, reps, step);
    let pis = vec![];
    (air, trace, pis)
}

#[test]
fn test_two_instances() -> Result<(), impl Debug> {
    let config = make_config(1337);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2, 1); // 16 rows, 2 reps

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

    let proof = prove_batch(&config, instances);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch(&config, &airs, &proof, &pvs)
}

#[test]
fn test_three_instances_mixed_sizes() -> Result<(), impl Debug> {
    let config = make_config(2025);

    let (air_fib16, fib16_trace, fib16_pis) = create_fib_instance(4); // 16 rows
    let (air_mul8, mul8_trace, mul8_pis) = create_mul_instance(3, 2, 1); // 8 rows
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

    let proof = prove_batch(&config, instances);
    let airs = vec![air_fib16, air_mul8, air_fib8];
    let pvs = vec![fib16_pis, mul8_pis, fib8_pis];
    verify_batch(&config, &airs, &proof, &pvs)
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
    let proof = prove_batch(&config, instances);

    // Wrong public value at verify => should reject
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_batch(&config, &[air_fib], &proof, &wrong_pvs);
    assert!(res.is_err(), "Should reject wrong public values");
    Ok::<_, Box<dyn std::error::Error>>(())
}

#[test]
fn test_different_widths() -> Result<(), impl Debug> {
    let config = make_config(4242);

    // Mul with reps=2 (width=6) and reps=3 (width=9)
    let (air_mul2, mul2_trace, mul2_pis) = create_mul_instance(3, 2, 1); // 8 rows, width=6
    let (air_fib, fib_trace, fib_pis) = create_fib_instance(3); // 8 rows, width=2
    let (air_mul3, mul3_trace, mul3_pis) = create_mul_instance(4, 3, 1); // 16 rows, width=9

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

    let proof = prove_batch(&config, instances);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul2_pis, fib_pis, mul3_pis];
    verify_batch(&config, &airs, &proof, &pvs)
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

    let proof = prove_batch(&config, instances);
    verify_batch(&config, &[air_fib], &proof, &[fib_pis])
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
    let valid_proof = prove_batch(&config, instances);

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
    let res = verify_batch(&config, &[air_fib], &tampered_proof, from_ref(&fib_pis));
    assert!(
        res.is_err(),
        "Verifier should reject trace with wrong width"
    );

    // Also test wrong trace_next width
    tampered_proof.opened_values.instances[0].trace_local =
        valid_proof.opened_values.instances[0].trace_local.clone();
    tampered_proof.opened_values.instances[0].trace_next =
        vec![valid_proof.opened_values.instances[0].trace_next[0]]; // Wrong width

    let res = verify_batch(&config, &[air_fib], &tampered_proof, from_ref(&fib_pis));
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
    let (air_b, tr_b, pv_b) = create_mul_instance(4, 2, 1);

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

    let proof = prove_batch(&config, instances);

    // Swap order at verify -> should fail
    let res = verify_batch(&config, &[air_b, air_a], &proof, &[pv_b, pv_a]);
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
    let proof = prove_batch(&config, instances);

    let mut tampered = proof;
    tampered.opened_values.instances[0].quotient_chunks[0].pop();

    let res = verify_batch(&config, &[air], &tampered, from_ref(&pv));
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

    // Create two Fibonacci instances with different sizes
    let air_fib1 = FibonacciAir;
    let air_fib2 = FibonacciAir;

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
    let proof = prove_batch(&config, instances);

    // Verify batch-proof
    let airs = vec![air_fib1, air_fib2];
    let public_values = vec![fib_pis1, fib_pis2];
    verify_batch(&config, &airs, &proof, &public_values)
        .map_err(|e| format!("Verification failed: {:?}", e))
}
