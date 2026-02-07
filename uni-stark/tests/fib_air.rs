use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs, create_test_fri_params};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

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

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.left.clone(), a);
        when_first_row.assert_eq(local.right.clone(), b);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.right.clone(), next.left.clone());

        // b' <- a + b
        when_transition.assert_eq(local.left.clone() + local.right.clone(), next.right.clone());

        builder.when_last_row().assert_eq(local.right.clone(), x);
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

const TWO_ADIC_FIXTURE: &str = "tests/fixtures/uni_stark_two_adic_v1.postcard";
const CIRCLE_FIXTURE: &str = "tests/fixtures/uni_stark_circle_v1.postcard";

fn make_two_adic_config(log_final_poly_len: usize) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len,
        max_log_arity: 1,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

fn two_adic_compat_case() -> (MyConfig, FibonacciAir, Vec<Val>, RowMajorMatrix<Val>) {
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let config = make_two_adic_config(2);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(21)];
    (config, FibonacciAir {}, pis, trace)
}

type CircleVal = Mersenne31;
type CircleChallenge = BinomialExtensionField<CircleVal, 3>;
type CircleByteHash = Keccak256Hash;
type CircleFieldHash = SerializingHasher<CircleByteHash>;
type CircleCompress = CompressionFunctionFromHasher<CircleByteHash, 2, 32>;
type CircleValMmcs = MerkleTreeMmcs<CircleVal, u8, CircleFieldHash, CircleCompress, 32>;
type CircleChallengeMmcs = ExtensionMmcs<CircleVal, CircleChallenge, CircleValMmcs>;
type CircleChallenger = SerializingChallenger32<CircleVal, HashChallenger<u8, CircleByteHash, 32>>;
type CirclePcsType = CirclePcs<CircleVal, CircleValMmcs, CircleChallengeMmcs>;
type CircleConfig = StarkConfig<CirclePcsType, CircleChallenge, CircleChallenger>;

fn make_circle_config() -> CircleConfig {
    let byte_hash = CircleByteHash {};
    let field_hash = CircleFieldHash::new(byte_hash);
    let compress = CircleCompress::new(byte_hash);
    let val_mmcs = CircleValMmcs::new(field_hash, compress);
    let challenge_mmcs = CircleChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = CirclePcsType {
        mmcs: val_mmcs,
        fri_params,
        _phantom: core::marker::PhantomData,
    };
    let challenger = CircleChallenger::from_hasher(vec![], byte_hash);
    CircleConfig::new(pcs, challenger)
}

fn circle_compat_case() -> (
    CircleConfig,
    FibonacciAir,
    Vec<CircleVal>,
    RowMajorMatrix<CircleVal>,
) {
    let config = make_circle_config();
    let trace = generate_trace_rows::<CircleVal>(0, 1, 1 << 3);
    let pis = vec![
        CircleVal::from_u64(0),
        CircleVal::from_u64(1),
        CircleVal::from_u64(21),
    ];
    (config, FibonacciAir {}, pis, trace)
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

/// n-th Fibonacci number expected to be x
fn test_public_value_impl(n: usize, x: u64, log_final_poly_len: usize) {
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let config = make_two_adic_config(log_final_poly_len);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);
    verify(&config, &FibonacciAir {}, &proof, &pis).expect("verification failed");
}

#[test]
fn test_zk() {
    type ByteHash = Keccak256Hash;
    let byte_hash = ByteHash {};

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    let u64_hash = U64Hash::new(KeccakF {});

    type FieldHash = SerializingHasher<U64Hash>;
    let field_hash = FieldHash::new(u64_hash);

    type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
    let compress = MyCompress::new(u64_hash);

    type ValHidingMmcs = MerkleTreeHidingMmcs<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        FieldHash,
        MyCompress,
        SmallRng,
        4,
        4,
    >;

    let rng = SmallRng::seed_from_u64(1);
    let val_mmcs = ValHidingMmcs::new(field_hash, compress, rng);

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    type ChallengeHidingMmcs = ExtensionMmcs<Val, Challenge, ValHidingMmcs>;

    let n = 1 << 3;
    let x = 21;

    let challenge_mmcs = ChallengeHidingMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    type HidingPcs = HidingFriPcs<Val, Dft, ValHidingMmcs, ChallengeHidingMmcs, SmallRng>;
    type MyHidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;
    let pcs = HidingPcs::new(dft, val_mmcs, fri_params, 4, SmallRng::seed_from_u64(1));
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let config = MyHidingConfig::new(pcs, challenger);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];
    let proof = prove(&config, &FibonacciAir {}, trace, &pis);
    verify(&config, &FibonacciAir {}, &proof, &pis).expect("verification failed");
}

#[test]
fn test_one_row_trace() {
    // Need to set log_final_poly_len to ensure log_min_height > params.log_final_poly_len + params.log_blowup
    test_public_value_impl(1, 1, 0);
}

#[test]
fn test_public_value() {
    test_public_value_impl(1 << 3, 21, 2);
}

#[test]
fn verify_two_adic_compat_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (config, air, pis, _) = two_adic_compat_case();
    let proof_bytes = read_fixture(TWO_ADIC_FIXTURE)
        .expect("Missing fixture. Run: cargo test -p p3-uni-stark --test fib_air -- --ignored");
    let proof: p3_uni_stark::Proof<MyConfig> = postcard::from_bytes(&proof_bytes)?;
    verify(&config, &air, &proof, &pis)?;
    Ok(())
}

#[test]
fn verify_circle_compat_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (config, air, pis, _) = circle_compat_case();
    let proof_bytes = read_fixture(CIRCLE_FIXTURE)
        .expect("Missing fixture. Run: cargo test -p p3-uni-stark --test fib_air -- --ignored");
    let proof: p3_uni_stark::Proof<CircleConfig> = postcard::from_bytes(&proof_bytes)?;
    verify(&config, &air, &proof, &pis)?;
    Ok(())
}

#[test]
#[ignore]
fn generate_two_adic_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Regen: cargo test -p p3-uni-stark --test fib_air -- --ignored
    let (config, air, pis, trace) = two_adic_compat_case();
    let proof = prove(&config, &air, trace, &pis);
    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(TWO_ADIC_FIXTURE, &bytes)?;
    Ok(())
}

#[test]
#[ignore]
fn generate_circle_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Regen: cargo test -p p3-uni-stark --test fib_air -- --ignored
    let (config, air, pis, trace) = circle_compat_case();
    let proof = prove(&config, &air, trace, &pis);
    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(CIRCLE_FIXTURE, &bytes)?;
    Ok(())
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 1);
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);
    let pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u32(123_123), // incorrect result
    ];
    prove(&config, &FibonacciAir {}, trace, &pis);
}
