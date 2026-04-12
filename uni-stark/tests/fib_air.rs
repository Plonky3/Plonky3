use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{InvalidProofShapeError, StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// For testing the public values feature
pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }

    fn num_public_values(&self) -> usize {
        3
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        // All constraints are guarded by is_first_row / is_transition / is_last_row
        // (degree 1) applied to degree-1 expressions (trace vars minus public values),
        // giving a max constraint degree of 2.
        Some(2)
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let local: &FibonacciRow<AB::Var> = main.current_slice().borrow();
        let next: &FibonacciRow<AB::Var> = main.next_slice().borrow();

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
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
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
    let val_mmcs = ValMmcs::new(hash, compress, 0);
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

type ZkByteHash = Keccak256Hash;
type ZkU64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type ZkFieldHash = SerializingHasher<ZkU64Hash>;
type ZkCompress = CompressionFunctionFromHasher<ZkU64Hash, 2, 4>;
type ZkValHidingMmcs = MerkleTreeHidingMmcs<
    [Val; p3_keccak::VECTOR_LEN],
    [u64; p3_keccak::VECTOR_LEN],
    ZkFieldHash,
    ZkCompress,
    SmallRng,
    2,
    4,
    4,
>;
type ZkChallenger = SerializingChallenger32<Val, HashChallenger<u8, ZkByteHash, 32>>;
type ZkChallengeHidingMmcs = ExtensionMmcs<Val, Challenge, ZkValHidingMmcs>;
type ZkHidingPcs = HidingFriPcs<Val, Dft, ZkValHidingMmcs, ZkChallengeHidingMmcs, SmallRng>;
type ZkConfig = StarkConfig<ZkHidingPcs, Challenge, ZkChallenger>;

fn make_zk_config() -> ZkConfig {
    let byte_hash = ZkByteHash {};
    let u64_hash = ZkU64Hash::new(KeccakF {});
    let field_hash = ZkFieldHash::new(u64_hash);
    let compress = ZkCompress::new(u64_hash);
    let val_mmcs = ZkValHidingMmcs::new(field_hash, compress, 0, SmallRng::seed_from_u64(1));
    let challenge_mmcs = ZkChallengeHidingMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = ZkHidingPcs::new(dft, val_mmcs, fri_params, 4, SmallRng::seed_from_u64(1));
    let challenger = ZkChallenger::from_hasher(vec![], byte_hash);
    ZkConfig::new(pcs, challenger)
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
type CircleValMmcs = MerkleTreeMmcs<CircleVal, u8, CircleFieldHash, CircleCompress, 2, 32>;
type CircleChallengeMmcs = ExtensionMmcs<CircleVal, CircleChallenge, CircleValMmcs>;
type CircleChallenger = SerializingChallenger32<CircleVal, HashChallenger<u8, CircleByteHash, 32>>;
type CirclePcsType = CirclePcs<CircleVal, CircleValMmcs, CircleChallengeMmcs>;
type CircleConfig = StarkConfig<CirclePcsType, CircleChallenge, CircleChallenger>;

fn make_circle_config() -> CircleConfig {
    let byte_hash = CircleByteHash {};
    let field_hash = CircleFieldHash::new(byte_hash);
    let compress = CircleCompress::new(byte_hash);
    let val_mmcs = CircleValMmcs::new(field_hash, compress, 0);
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
    let n = 1 << 3;
    let x = 21;

    let trace = generate_trace_rows::<Val>(0, 1, n);
    let config = make_zk_config();
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
fn test_short_public_values_rejected() {
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let config = make_two_adic_config(2);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(21)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);
    let short_pis = vec![BabyBear::ZERO, BabyBear::ONE];
    let err = verify(&config, &FibonacciAir {}, &proof, &short_pis)
        .expect_err("verification should reject short public values");
    match err {
        p3_uni_stark::VerificationError::InvalidProofShape(
            InvalidProofShapeError::PublicValuesLengthMismatch { expected, got },
        ) => {
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
}

#[test]
fn test_degree_bits_too_large_rejected() {
    // The uni-stark verifier builds an evaluation domain via `1 << degree_bits`.
    // A malicious proof can set degree_bits >= usize::BITS (e.g. 64 on a 64-bit
    // platform), causing a shift overflow. The verifier must reject this with a
    // structured error before any domain construction.

    // Generate a valid 2^3 = 8-row Fibonacci trace: fib(0) = 0, fib(1) = 1.
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);

    // Non-ZK config with log_final_poly_len = 2 (FRI stops at degree 4).
    // The overflow check is independent of the ZK setting.
    let config = make_two_adic_config(2);

    // Public inputs: [fib(0), fib(1), fib(7)] = [0, 1, 21].
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(21)];

    // Produce a legitimate proof, then tamper with the degree_bits field.
    let mut proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Mutation: set degree_bits to exactly the bit width of usize, the
    // smallest value that overflows:
    //
    //     degree_bits = 64  (on 64-bit)
    //     1_usize << 64  →  shift overflow  →  must be caught
    proof.degree_bits = usize::BITS as usize;

    // Verification must fail deterministically, not panic.
    let err = verify(&config, &FibonacciAir {}, &proof, &pis)
        .expect_err("verification should reject oversized degree_bits");

    // Verify the error carries the correct diagnostic fields.
    // Unlike batch-stark, uni-stark has a single AIR so `air` is None.
    //
    //   - air: None              — uni-stark doesn't index by AIR
    //   - maximum: BITS - 1      — largest safe exponent (63 on 64-bit)
    //   - got: BITS              — the tampered value we injected (64)
    match err {
        p3_uni_stark::VerificationError::InvalidProofShape(
            InvalidProofShapeError::DegreeBitsTooLarge { air, maximum, got },
        ) => {
            assert_eq!(air, None);
            assert_eq!(maximum, usize::BITS as usize - 1);
            assert_eq!(got, usize::BITS as usize);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
}

#[test]
fn test_degree_bits_too_small_for_zk_rejected() {
    // In ZK mode the prover extends the trace by one bit (is_zk = 1), so the
    // verifier computes `base_degree_bits = degree_bits - is_zk`. If a
    // malicious proof sets degree_bits = 0 while is_zk = 1, that subtraction
    // underflows. The verifier must reject this before any arithmetic.

    // Generate a valid 2^3 = 8-row Fibonacci trace.
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);

    // ZK-enabled config — is_zk = 1, meaning degree_bits must be >= 1.
    let config = make_zk_config();

    // Public inputs: [fib(0), fib(1), fib(7)] = [0, 1, 21].
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(21)];

    // Produce a legitimate ZK proof, then tamper with degree_bits.
    let mut proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Mutation: set degree_bits to 0, below the ZK minimum.
    //
    //     is_zk = 1
    //     degree_bits = 0
    //     base_degree_bits = 0 - 1  →  underflow  →  must be caught
    proof.degree_bits = 0;

    // Verification must fail with a structured error.
    let err = verify(&config, &FibonacciAir {}, &proof, &pis)
        .expect_err("verification should reject too-small degree_bits in zk mode");

    // Verify the error carries the correct diagnostic fields:
    //
    //   - air: None       — uni-stark doesn't index by AIR
    //   - minimum: 1      — is_zk, the smallest acceptable degree_bits
    //   - got: 0          — the tampered value we injected
    match err {
        p3_uni_stark::VerificationError::InvalidProofShape(
            InvalidProofShapeError::DegreeBitsTooSmall { air, minimum, got },
        ) => {
            assert_eq!(air, None);
            assert_eq!(minimum, 1);
            assert_eq!(got, 0);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
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
#[should_panic(expected = "constraints not satisfied on row")]
fn test_incorrect_public_value() {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 1);
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
