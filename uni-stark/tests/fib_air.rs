use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, PublicRow, StarkConfig};
use rand::thread_rng;

/// For testing the public values feature

pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let public_values = builder.public_values();
        // Only one row of public inputs
        let pis = public_values.row_slice(0);

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let local: &FibonacciRow<AB::Var> = main.row_slice(0).borrow();
        let next: &FibonacciRow<AB::Var> = main.row_slice(1).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left, a);
        when_first_row.assert_eq(local.right, b);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.right, next.left);

        // b' <- a + b
        when_transition.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace =
        RowMajorMatrix::new(vec![F::zero(); n * NUM_FIBONACCI_COLS], NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow::new(F::from_canonical_u64(a), F::from_canonical_u64(b));

    for i in 1..n {
        rows[i].left = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}

const NUM_FIBONACCI_COLS: usize = 2;

pub struct FibonacciRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> FibonacciRow<F> {
    fn new(left: F, right: F) -> FibonacciRow<F> {
        FibonacciRow { left, right }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

type Val = BabyBear;
type Perm = Poseidon2<Val, DiffusionMatrixBabybear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = FieldMerkleTreeMmcs<<Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16>;
type Dft = Radix2DitParallel;
type MyFriConfig = TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>;
type Pcs = TwoAdicFriPcs<MyFriConfig>;
type MyConfig = StarkConfig<Val, Challenge, Pcs, Challenger>;

#[test]
fn test_public_value() {
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 28,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(fri_config, dft, val_mmcs);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm.clone());
    // Public inputs are just one row
    let pis = PublicRow(vec![
        BabyBear::from_canonical_u64(0),
        BabyBear::from_canonical_u64(1),
        BabyBear::from_canonical_u64(21),
    ]);
    let proof = prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
    let mut challenger = Challenger::new(perm);
    verify(&config, &FibonacciAir {}, &mut challenger, &proof, &pis).expect("verification failed");
}

#[test]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 28,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let pcs = Pcs::new(fri_config, dft, val_mmcs);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm.clone());
    let pis = PublicRow(vec![
        BabyBear::from_canonical_u64(0),
        BabyBear::from_canonical_u64(1),
        BabyBear::from_canonical_u64(123_123), // incorrect result
    ]);
    let proof = prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
    let mut challenger = Challenger::new(perm.clone());
    verify(&config, &FibonacciAir {}, &mut challenger, &proof, &pis).expect("verification failed");
}
