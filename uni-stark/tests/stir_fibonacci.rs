//! End-to-end STIR-backed uni-stark proving: commit, prove, and verify a real AIR through
//! `TwoAdicStirPcs` via `p3_uni_stark::prove`/`verify`.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_stir::{SecurityAssumption, StirParameters, TwoAdicStirPcs};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{InvalidProofShapeError, StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// For testing the public values feature.
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
type Pcs = TwoAdicStirPcs<Val, Dft, ValMmcs, ChallengeMmcs, Challenge, Challenger>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

const STIR_FIXTURE: &str = "tests/fixtures/uni_stark_stir_v1.postcard";

/// `TwoAdicStirPcs::commit` requires every committed matrix to have at least
/// `2^log_starting_folding_factor` rows. `make_config` below sets
/// `log_starting_folding_factor = 2`, so `4` is the smallest trace height it accepts.
const MIN_TRACE_HEIGHT: usize = 1 << 2;

fn make_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let stir_params = StirParameters {
        log_blowup: 1,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type: SecurityAssumption::CapacityBound,
        security_level: 100,
        max_pow_bits: 20,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, stir_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

fn compat_case() -> (MyConfig, FibonacciAir, Vec<Val>, RowMajorMatrix<Val>) {
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let config = make_config();
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(21)];
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

/// n-th Fibonacci number expected to be x.
fn test_public_value_impl(n: usize, x: u64) {
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let config = make_config();
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace, &pis);
    verify(&config, &FibonacciAir {}, &proof, &pis).expect("verification failed");
}

#[test]
fn test_smallest_trace() {
    // rows: (0,1) (1,1) (1,2) (2,3) -> fib(4) = 3.
    test_public_value_impl(MIN_TRACE_HEIGHT, 3);
}

#[test]
fn test_public_value() {
    test_public_value_impl(1 << 3, 21);
}

#[test]
fn test_short_public_values_rejected() {
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let config = make_config();
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
fn verify_stir_compat_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (config, air, pis, _) = compat_case();
    let proof_bytes = read_fixture(STIR_FIXTURE).expect(
        "Missing fixture. Run: cargo test -p p3-uni-stark --test stir_fibonacci -- --ignored",
    );
    let proof: p3_uni_stark::Proof<MyConfig> = postcard::from_bytes(&proof_bytes)?;
    verify(&config, &air, &proof, &pis)?;
    Ok(())
}

#[test]
#[ignore]
fn generate_stir_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Regen: cargo test -p p3-uni-stark --test stir_fibonacci -- --ignored
    let (config, air, pis, trace) = compat_case();
    let proof = prove(&config, &air, trace, &pis);
    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(STIR_FIXTURE, &bytes)?;
    Ok(())
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "constraints not satisfied on row")]
fn test_incorrect_public_value() {
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let config = make_config();
    let pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u32(123_123), // incorrect result
    ];
    prove(&config, &FibonacciAir {}, trace, &pis);
}
