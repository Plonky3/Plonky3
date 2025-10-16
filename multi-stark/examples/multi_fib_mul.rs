use core::borrow::Borrow;

use itertools::Itertools;
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
use p3_merkle_tree::{MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use p3_multi_stark::{MultiProof, StarkInstance, prove_multi, verify_multi};
use p3_uni_stark::StarkConfig;
use rand::rngs::SmallRng;
use rand::SeedableRng;

// Simple Fibonacci AIR: two columns [a, b], next = [b, a+b]. Public values: [a0, b0, x_final]
#[derive(Debug, Clone, Copy)]
struct FibonacciAir;

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize { 2 }
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
struct FibRow<F> { left: F, right: F }
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
    rows[0] = FibRow { left: F::from_u64(a), right: F::from_u64(b) };
    for i in 1..n { rows[i] = FibRow { left: rows[i-1].right, right: rows[i-1].left + rows[i-1].right }; }
    trace
}

// Simple multiplication AIR: columns triplets (a, b, c) with c = a*b and optional transition a' = a + k.
#[derive(Debug, Clone, Copy)]
struct MulAir { reps: usize, step: u64 }

impl Default for MulAir { fn default() -> Self { Self { reps: 2, step: 1 } } }

impl<F> BaseAir<F> for MulAir { fn width(&self) -> usize { self.reps * 3 } }

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();
        for i in 0..self.reps {
            let s = i*3;
            let a = local[s].clone();
            let b = local[s+1].clone();
            let c = local[s+2].clone();
            builder.assert_eq(a.clone() * b.clone(), c);
            builder.when_transition().assert_eq(a + AB::Expr::from_u64(self.step), next[s].clone());
        }
    }
}

fn mul_trace<F: Field>(rows: usize, reps: usize, step: u64) -> RowMajorMatrix<F> {
    assert!(rows.is_power_of_two());
    let w = reps * 3;
    let mut v = F::zero_vec(rows * w);
    for (i, (a,b,c)) in v.iter_mut().tuples().enumerate() { let r = i / reps; *a = F::from_u64(r as u64); *b = F::from_u64(3); *c = *a * *b; }
    RowMajorMatrix::new(v, w)
}

fn fib_n(n: usize) -> u64 { let mut a=0u64; let mut b=1u64; for _ in 0..n { let t=a+b; a=b; b=t; } a }

fn main() {
    // Configure a simple non-ZK BabyBear FRI PCS.
    type Val = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
    type Challenge = BinomialExtensionField<Val, 4>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Dft = Radix2DitParallel<Val>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    // Build PCS
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);

    // Prepare two AIRs and traces of different lengths
    let fib = FibonacciAir;
    let n_fib = 1 << 4; // 16
    let a0 = 0u64; let b0 = 1u64; let x = fib_n(n_fib);
    let fib_trace = fib_trace::<Val>(a0, b0, n_fib);
    let fib_pis = vec![Val::from_u64(a0), Val::from_u64(b0), Val::from_u64(x)];

    let mul = MulAir::default();
    let n_mul = 1 << 5; // 32
    let mul_trace = mul_trace::<Val>(n_mul, mul.reps, mul.step);
    let mul_pis: Vec<Val> = vec![]; // no public inputs

    // Build an enum absraction so we can mix AIR types in one batch.
    #[derive(Clone, Copy)] enum DemoAir { Fib(FibonacciAir), Mul(MulAir) }
    impl<F> BaseAir<F> for DemoAir { fn width(&self) -> usize { match self { DemoAir::Fib(a) => <FibonacciAir as BaseAir<F>>::width(a), DemoAir::Mul(a) => <MulAir as BaseAir<F>>::width(a) } } }
    impl<AB: AirBuilderWithPublicValues> Air<AB> for DemoAir { fn eval(&self, b: &mut AB) { match self { DemoAir::Fib(a) => a.eval(b), DemoAir::Mul(a) => a.eval(b) } } }

    // Prove multi
    let air_fib = DemoAir::Fib(fib);
    let air_mul = DemoAir::Mul(mul);
    let instances = vec![
        StarkInstance { air: &air_fib, trace: fib_trace.clone(), public_values: fib_pis.clone() },
        StarkInstance { air: &air_mul, trace: mul_trace.clone(), public_values: mul_pis.clone() },
    ];
    let proof: MultiProof<_> = prove_multi(&config, instances);

    // Verify multi
    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_multi(&config, &airs, &proof, &pvs).expect("verification failed");

    println!("Multi-STARK example: success (fib n={}, mul rows={})", n_fib, n_mul);
}
