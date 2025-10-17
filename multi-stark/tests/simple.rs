use core::borrow::Borrow;
use core::fmt::Debug;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_test_fri_params};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::{StarkInstance, prove_multi, verify_multi};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
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
            DemoAir::Fib(a) => <FibonacciAir as BaseAir<F>>::width(a),
            DemoAir::Mul(a) => <MulAir as BaseAir<F>>::width(a),
        }
    }
}
impl<AB: AirBuilderWithPublicValues> Air<AB> for DemoAir {
    fn eval(&self, b: &mut AB) {
        match self {
            DemoAir::Fib(a) => a.eval(b),
            DemoAir::Mul(a) => a.eval(b),
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

    let proof = prove_multi(&config, instances);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_multi(&config, &airs, &proof, &pvs)
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

    let proof = prove_multi(&config, instances);
    let airs = vec![air_fib16, air_mul8, air_fib8];
    let pvs = vec![fib16_pis, mul8_pis, fib8_pis];
    verify_multi(&config, &airs, &proof, &pvs)
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
    let proof = prove_multi(&config, instances);

    // Wrong public value at verify => should reject
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_multi(&config, &[air_fib], &proof, &wrong_pvs);
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

    let proof = prove_multi(&config, instances);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul2_pis, fib_pis, mul3_pis];
    verify_multi(&config, &airs, &proof, &pvs)
}

#[test]
fn test_quotient_size_not_multiple_of_width() -> Result<(), impl Debug> {
    // This test exercises the padding fix for the case where quotient_size % WIDTH != 0.
    // We use a moderately sized trace that, when combined with quotient degree,
    // may produce quotient domains not evenly divisible by the packed WIDTH (typically 16 for BabyBear).
    // The key is ensuring the parallel loop in quotient_values doesn't panic on out-of-bounds
    // access when slicing the selector vectors.
    let config = make_config(9999);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(5); // 32 rows

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: fib_trace,
        public_values: fib_pis.clone(),
    }];

    // This should not panic due to out-of-bounds access in the packed slicing
    // (the bug would have caused a panic when quotient_size % WIDTH != 0)
    let proof = prove_multi(&config, instances);
    verify_multi(&config, &[air_fib], &proof, &[fib_pis])
}

#[test]
fn test_invalid_trace_width_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // This test verifies that the verifier rejects proofs with incorrect trace width.
    use p3_multi_stark::proof::{InstanceOpenedValues, MultiCommitments, MultiOpenedValues};

    let config = make_config(55555);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: fib_trace,
        public_values: fib_pis.clone(),
    }];

    // Generate a valid proof
    let valid_proof = prove_multi(&config, instances);

    // Tamper with the proof: change trace_local to have wrong width
    let mut tampered_proof = p3_multi_stark::proof::MultiProof {
        commitments: MultiCommitments {
            main: valid_proof.commitments.main,
            quotient_chunks: valid_proof.commitments.quotient_chunks,
        },
        opened_values: MultiOpenedValues {
            instances: vec![InstanceOpenedValues {
                trace_local: vec![valid_proof.opened_values.instances[0].trace_local[0]], // Wrong width: 1 instead of 2
                trace_next: valid_proof.opened_values.instances[0].trace_next.clone(),
                quotient_chunks: valid_proof.opened_values.instances[0]
                    .quotient_chunks
                    .clone(),
            }],
        },
        opening_proof: valid_proof.opening_proof.clone(),
        degree_bits: valid_proof.degree_bits.clone(),
    };

    // Verification should fail due to width mismatch
    let res = verify_multi(&config, &[air_fib], &tampered_proof, &[fib_pis.clone()]);
    assert!(
        res.is_err(),
        "Verifier should reject trace with wrong width"
    );

    // Also test wrong trace_next width
    tampered_proof.opened_values.instances[0].trace_local =
        valid_proof.opened_values.instances[0].trace_local.clone();
    tampered_proof.opened_values.instances[0].trace_next =
        vec![valid_proof.opened_values.instances[0].trace_next[0]]; // Wrong width

    let res = verify_multi(&config, &[air_fib], &tampered_proof, &[fib_pis]);
    assert!(
        res.is_err(),
        "Verifier should reject trace_next with wrong width"
    );

    Ok::<_, Box<dyn std::error::Error>>(())
}
