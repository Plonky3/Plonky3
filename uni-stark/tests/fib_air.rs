use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{HidingFriPcs, TwoAdicFriPcs, create_test_fri_config};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher32To64, TruncatedPermutation,
};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::{StdRng, ThreadRng};
use rand::{SeedableRng, rng};

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

        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

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

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * NUM_FIBONACCI_COLS), NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow::new(F::from_u64(a), F::from_u64(b));

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
    const fn new(left: F, right: F) -> Self {
        Self { left, right }
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
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

/// n-th Fibonacci number expected to be x
fn test_public_value_impl(n: usize, x: u64) {
    let perm = Perm::new_from_rng_128(&mut rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_config = create_test_fri_config(challenge_mmcs);
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm.clone());
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];
    let proof = prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
    let mut challenger = Challenger::new(perm);
    verify(&config, &FibonacciAir {}, &mut challenger, &proof, &pis).expect("verification failed");
}

#[test]
fn test_zk() {
    type ByteHash = Keccak256Hash;
    let byte_hash = ByteHash {};

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    let u64_hash = U64Hash::new(KeccakF {});

    type FieldHash = SerializingHasher32To64<U64Hash>;
    let field_hash = FieldHash::new(u64_hash);

    type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
    let compress = MyCompress::new(u64_hash);

    type ValHidingMmcs = MerkleTreeHidingMmcs<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        FieldHash,
        MyCompress,
        ThreadRng,
        4,
        4,
    >;
    let val_mmcs = ValHidingMmcs::new(field_hash, compress, rng());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    type ChallengeHidingMmcs = ExtensionMmcs<Val, Challenge, ValHidingMmcs>;

    let n = 1 << 3;
    let x = 21;

    let challenge_mmcs = ChallengeHidingMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_config = create_test_fri_config(challenge_mmcs);
    type HidingPcs = HidingFriPcs<Val, Dft, ValHidingMmcs, ChallengeHidingMmcs, StdRng>;
    type MyHidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;
    let pcs = HidingPcs::new(dft, val_mmcs, fri_config, 4, StdRng::from_os_rng());
    let config = MyHidingConfig::new(pcs);
    let mut challenger = Challenger::from_hasher(vec![], byte_hash);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];
    let proof = prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
    let mut challenger = Challenger::from_hasher(vec![], byte_hash);
    verify(&config, &FibonacciAir {}, &mut challenger, &proof, &pis).expect("verification failed");
}

#[test]
fn test_one_row_trace() {
    test_public_value_impl(1, 1);
}

#[test]
fn test_public_value() {
    test_public_value_impl(1 << 3, 21);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
    let perm = Perm::new_from_rng_128(&mut rng());
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_config = create_test_fri_config(challenge_mmcs);
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm);
    let pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u32(123_123), // incorrect result
    ];
    prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
}
