use core::borrow::Borrow;

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

#[test]
fn simple_two_instance_proof_verifies() {
    // Build non-ZK PCS config
    let config = make_config(1337);

    // Prepare two instances
    let fib = FibonacciAir;
    let mul = MulAir::default();
    let air_fib = DemoAir::Fib(fib);
    let air_mul = DemoAir::Mul(mul);

    let n_fib = 1 << 4; // 16
    let fib_trace = fib_trace::<Val>(0, 1, n_fib);
    let fib_pis = vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(fib_n(n_fib)),
    ];

    let n_mul = 1 << 4; // 16
    let mul_trace = mul_trace::<Val>(n_mul, mul.reps, mul.step);
    let mul_pis: Vec<Val> = vec![];

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
    verify_multi(&config, &airs, &proof, &pvs).expect("verification failed");
}

#[test]
fn three_instances_mixed_lengths_verifies() {
    let config = make_config(2025);

    let fib = FibonacciAir;
    let mul = MulAir::default();
    let air_fib16 = DemoAir::Fib(fib);
    let air_fib8 = DemoAir::Fib(fib);
    let air_mul8 = DemoAir::Mul(mul);

    let n_fib16 = 1 << 4;
    let n_fib8 = 1 << 3;
    let fib16_trace = fib_trace::<Val>(0, 1, n_fib16);
    let fib8_trace = fib_trace::<Val>(0, 1, n_fib8);
    let fib16_pis = vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(fib_n(n_fib16)),
    ];
    let fib8_pis = vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(fib_n(n_fib8)),
    ];

    let mul8_trace = mul_trace::<Val>(1 << 3, mul.reps, mul.step);
    let mul8_pis: Vec<Val> = vec![];

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
    verify_multi(&config, &airs, &proof, &pvs).expect("verification failed");
}

#[test]
fn invalid_public_values_rejected() {
    let config = make_config(7);

    // Single Fibonacci instance
    let fib = FibonacciAir;
    let air_fib = DemoAir::Fib(fib);
    let n_fib = 1 << 4; // 16
    let fib_trace = fib_trace::<Val>(0, 1, n_fib);
    let correct_x = fib_n(n_fib);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(correct_x)];
    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: fib_trace,
        public_values: fib_pis.clone(),
    }];
    let proof = prove_multi(&config, instances);

    // Wrong public value at verify => reject.
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_multi(&config, &vec![air_fib], &proof, &wrong_pvs);
    assert!(res.is_err());
}

#[test]
fn different_widths_verifies() {
    let config = make_config(4242);

    let fib = FibonacciAir;
    let air_fib = DemoAir::Fib(fib);

    // Mul with reps=2 (width=6) and reps=3 (width=9)
    let mul2 = MulAir { reps: 2, step: 1 };
    let mul3 = MulAir { reps: 3, step: 1 };
    let air_mul2 = DemoAir::Mul(mul2);
    let air_mul3 = DemoAir::Mul(mul3);

    let n_fib = 1 << 3; // 8
    let fib_trace = fib_trace::<Val>(0, 1, n_fib);
    let fib_pis = vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(fib_n(n_fib)),
    ];

    let mul2_trace = mul_trace::<Val>(1 << 3, mul2.reps, mul2.step);
    let mul3_trace = mul_trace::<Val>(1 << 4, mul3.reps, mul3.step);
    let mul_pis: Vec<Val> = vec![];

    let instances = vec![
        StarkInstance {
            air: &air_mul2,
            trace: mul2_trace,
            public_values: mul_pis.clone(),
        },
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul3,
            trace: mul3_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let proof = prove_multi(&config, instances);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul_pis.clone(), fib_pis, mul_pis];
    verify_multi(&config, &airs, &proof, &pvs).expect("verification failed");
}
