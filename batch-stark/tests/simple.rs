use core::borrow::Borrow;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_air::{Air, AirBuilder, BaseAir, PermutationAirBuilder, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::proof::{BatchProof, OpenedValuesWithLookups};
use p3_batch_stark::{
    CommonData, ProverData, StarkGenericConfig, StarkInstance, VerificationError, prove_batch,
    verify_batch,
};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_lookup::InteractionBuilder;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{InvalidProofShapeError, StarkConfig};
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

const TWO_ADIC_FIXTURE: &str = "tests/fixtures/batch_stark_two_adic_v1.postcard";
const CIRCLE_FIXTURE: &str = "tests/fixtures/batch_stark_circle_v1.postcard";

// --- Simple Fibonacci AIR and trace ---

#[derive(Debug, Clone, Copy)]
struct FibonacciAir {
    /// log2 of the trace height; used to size preprocessed columns.
    log_height: usize,
    /// Index to tamper with in preprocessed trace (None = no tampering).
    tamper_index: Option<usize>,
}

impl<F: Field> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        3
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let n = 1 << self.log_height;
        let mut m = RowMajorMatrix::new(F::zero_vec(n), 1);
        for (i, v) in m.values.iter_mut().enumerate().take(n) {
            *v = F::from_u64(i as u64);
        }
        if let Some(idx) = self.tamper_index
            && idx < n
        {
            m.values[idx] += F::ONE;
        }
        Some(m)
    }

    fn preprocessed_width(&self) -> usize {
        1
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let a0 = pis[0];
        let b0 = pis[1];
        let x = pis[2];

        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        let mut wf = builder.when_first_row();
        wf.assert_eq(local.left, a0);
        wf.assert_eq(local.right, b0);

        let mut wt = builder.when_transition();
        wt.assert_eq(local.right, next.left);
        wt.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
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
// The AIR has 3 * `reps` columns:
// - for each rep, 3 columns: `a`, `b`, `c` where we enforce `a * b = c`
// - an extra column at the end which is a permutation of the first `a` column (used for local lookups in `MulAirLookups`)

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
        self.reps * 3 + 1
    }
}
impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        for i in 0..self.reps {
            let s = i * 3;
            let a = local[s];
            let b = local[s + 1];
            let c = local[s + 2];
            builder.assert_eq(a * b, c);

            builder.when_transition().assert_eq(b, next[s]);
            builder.when_transition().assert_eq(a + b, next[s + 1]);
        }
    }
}

#[derive(Clone)]
struct PeriodicAir<F> {
    periodic: Vec<Vec<F>>,
}

impl<F: Field + PrimeCharacteristicRing> PeriodicAir<F> {
    fn new() -> Self {
        Self {
            periodic: vec![
                vec![
                    F::from_u64(1),
                    F::from_u64(2),
                    F::from_u64(3),
                    F::from_u64(4),
                ],
                vec![F::from_u64(10), F::from_u64(20)],
            ],
        }
    }

    fn valid_trace(&self, rows: usize) -> RowMajorMatrix<F> {
        let mut values = F::zero_vec(rows * 2);
        for (i, row) in values.chunks_exact_mut(2).enumerate() {
            row[0] = self.periodic[0][i % self.periodic[0].len()];
            row[1] = self.periodic[1][i % self.periodic[1].len()];
        }
        RowMajorMatrix::new(values, 2)
    }
}

impl<F: Field> BaseAir<F> for PeriodicAir<F> {
    fn width(&self) -> usize {
        2
    }

    fn num_periodic_columns(&self) -> usize {
        self.periodic.len()
    }

    fn periodic_columns(&self) -> Vec<Vec<F>> {
        self.periodic.clone()
    }
}

impl<AB: AirBuilder> Air<AB> for PeriodicAir<AB::F>
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let p0 = builder.periodic_values()[0].into();
        let p1 = builder.periodic_values()[1].into();
        builder.assert_eq(local[0], p0);
        builder.assert_eq(local[1], p1);
    }
}

// --- MulAirLookups structure for local and global lookups ---
// This AIR is a `MulAir` that can register global lookups with `FibAirLookups`, as well as local lookups with a lookup column. Its inputs are the Fibonacci values.
// - when `is_local` is true, this AIR creates local lookups between its first column and its last column.
//   The latter corresponds to a permutation of the first column:
//     - it receives the first column with multiplicity 1
//     - it sends the last column (permuted values) with multiplicity 1
// - when `is_global` is true, this AIR creates global lookups between its inputs and `FibAirLookups` AIR's inputs:
//     - For each `rep`, it sends its first two columns (a,b) to the global lookup with name `global_names[rep]` and multiplicity 1
// - `num_lookups` tracks the number of registered lookups. It is 0 when the structure is created,
//   and increments every time a new lookup is registered.
#[derive(Clone, Default)]
struct MulAirLookups {
    air: MulAir,
    is_local: bool,
    is_global: bool,
    global_names: Vec<String>,
}

impl MulAirLookups {
    const fn new(air: MulAir, is_local: bool, is_global: bool, global_names: Vec<String>) -> Self {
        Self {
            air,
            is_local,
            is_global,
            global_names,
        }
    }
}

impl<F> BaseAir<F> for MulAirLookups {
    fn width(&self) -> usize {
        <MulAir as BaseAir<F>>::width(&self.air)
    }
}

impl<AB> Air<AB> for MulAirLookups
where
    AB::Var: Debug,
    AB: AirBuilder + PermutationAirBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);

        // Cross-AIR and intra-AIR interactions.
        let main = builder.main();
        let local = main.current_slice();
        let last_idx = <Self as BaseAir<AB::F>>::width(self) - 1;
        let lut = local[last_idx]; // Extra column: permutation of 'a'

        for rep in 0..self.air.reps {
            let base_idx = rep * 3;
            let a = local[base_idx];
            let b = local[base_idx + 1];

            // Local lookup: 'a' against the permuted column.
            if self.is_local {
                builder.push_local_interaction(vec![
                    (vec![a.into()], AB::Expr::ONE),    // query (receive)
                    (vec![lut.into()], -AB::Expr::ONE), // table (send, negated)
                ]);
            }

            // Global lookup: send (a, b) to FibAirLookups.
            if self.is_global {
                builder.push_interaction(
                    &self.global_names[rep],
                    [a.into(), b.into()],
                    -AB::Expr::ONE, // Send = negative count
                    1,
                );
            }
        }
    }
}

fn mul_trace<F: Field>(rows: usize, reps: usize) -> RowMajorMatrix<F> {
    assert!(rows.is_power_of_two());
    // The extra column corresponds to a permutation of the first column.
    let w = reps * 3 + 1;
    let mut v = F::zero_vec(rows * w);
    let last_idx = w - 1;

    for rep in 0..reps {
        let mut a = F::ZERO;
        let mut b = F::ONE;
        for i in 0..rows {
            let idx = i * w + rep * 3;
            v[idx] = a;
            v[idx + 1] = b;
            v[idx + 2] = v[idx] * v[idx + 1];
            if i != rows - 1 {
                v[i * w + last_idx] = b;
            }
            let tmp = a + b;
            a = b;
            b = tmp;
        }
    }
    RowMajorMatrix::new(v, w)
}

// --- FibAirLookups structure for global lookups ---
// This AIR is a `FibonacciAir` that can register global lookups with `MulAir` AIRs.
// - when `is_global` is true, this AIR creates global lookups between its inputs and MulAir AIR's inputs:
//     - it receives its two columns (left,right) from the global lookup with name `name_and_mult.0`
//       and multiplicity `name_and_mult.1`. The default for `name_and_mult` is ("MulFib", 2).
// - `num_lookups` tracks the number of registered lookups. It is 0 when the structure is created,
//    and increments every time a new lookup is registered.
// - `name_and_mult` is used when `is_global` is true. If provided, it specifies the name and multiplicity of the global lookup.
//   If not provided and `is_global` is true, a default name "MulFib" and multiplicity 2 is used.
#[derive(Debug, Clone)]
struct FibAirLookups {
    air: FibonacciAir,
    is_global: bool,
    name_and_mult: Option<(String, u64)>,
}

impl Default for FibAirLookups {
    fn default() -> Self {
        Self {
            air: FibonacciAir {
                log_height: 3,
                tamper_index: None,
            },
            is_global: false,
            name_and_mult: None,
        }
    }
}

impl FibAirLookups {
    const fn new(air: FibonacciAir, is_global: bool, name_and_mult: Option<(String, u64)>) -> Self {
        Self {
            air,
            is_global,
            name_and_mult,
        }
    }
}

impl<F: Field> BaseAir<F> for FibAirLookups {
    fn width(&self) -> usize {
        <FibonacciAir as BaseAir<F>>::width(&self.air)
    }

    fn num_public_values(&self) -> usize {
        <FibonacciAir as BaseAir<F>>::num_public_values(&self.air)
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.air.preprocessed_trace()
    }

    fn preprocessed_width(&self) -> usize {
        <FibonacciAir as BaseAir<F>>::preprocessed_width(&self.air)
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for FibAirLookups {
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);

        // Global interaction — receive (left, right) from MulAirLookups.
        if self.is_global {
            let main = builder.main();
            let left = main.current(0).unwrap();
            let right = main.current(1).unwrap();

            let (name, multiplicity) = match &self.name_and_mult {
                Some((n, m)) => (n.clone(), *m),
                None => ("MulFib".to_string(), 2),
            };

            // Receive = positive count (counterpart to MulAirLookups' negative send).
            builder.push_interaction(
                &name,
                [left.into(), right.into()],
                AB::Expr::from_u64(multiplicity),
                1,
            );
        }
    }
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

    fn preprocessed_width(&self) -> usize {
        1
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB> Air<AB> for PreprocessedMulAir
where
    AB: AirBuilder,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_main = main.current_slice();

        // Copy the preprocessed value so the immutable borrow on `builder`
        // is released before the mutable `assert_eq` call.
        let prep_val = builder.preprocessed().current(0).unwrap();

        // Enforce: main[0] = multiplier * preprocessed[0]
        builder.assert_eq(
            local_main[0],
            prep_val * AB::Expr::from_u64(self.multiplier),
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
type PermWide = Poseidon2BabyBear<32>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyCompressWide = TruncatedPermutation<PermWide, 4, 8, 32>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type ValMmcsWide =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompressWide, 4, 8>;
type HidingValMmcs = MerkleTreeHidingMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    SmallRng,
    2,
    8,
    4,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type ChallengeMmcsWide = ExtensionMmcs<Val, Challenge, ValMmcsWide>;
type HidingChallengeMmcs = ExtensionMmcs<Val, Challenge, HidingValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyPcsWide = TwoAdicFriPcs<Val, Dft, ValMmcsWide, ChallengeMmcsWide>;
type HidingPcs = HidingFriPcs<Val, Dft, HidingValMmcs, HidingChallengeMmcs, SmallRng>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;
type MyConfigWide = StarkConfig<MyPcsWide, Challenge, Challenger>;
type MyHidingConfig = StarkConfig<HidingPcs, Challenge, Challenger>;

fn make_config(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Minimal FRI shape so a tiny trace still completes `prove_batch` / `verify_batch`.
fn make_config_allow_tiny_trace(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

/// Same as make_config, but with a different arity.
fn make_config_wide(seed: u64) -> MyConfigWide {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let perm_wide = PermWide::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompressWide::new(perm_wide);
    let val_mmcs = ValMmcsWide::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcsWide::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = MyPcsWide::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

fn make_two_adic_compat_config(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 1);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters {
        log_blowup: 2,
        log_final_poly_len: 2,
        max_log_arity: 1,
        num_queries: 2,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
}

fn make_config_zk(seed: u64) -> MyHidingConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = HidingValMmcs::new(hash, compress, 2, rng.clone());
    let challenge_mmcs = HidingChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = FriParameters::new_testing(challenge_mmcs, 2);
    let pcs = HidingPcs::new(dft, val_mmcs, fri_params, 4, rng);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
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
    let val_mmcs = CircleValMmcs::new(field_hash, compress, 3);
    let challenge_mmcs = CircleChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 8,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = CirclePcsType {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = CircleChallenger::from_hasher(vec![], byte_hash);
    CircleConfig::new(pcs, challenger)
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

    fn num_public_values(&self) -> usize {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::num_public_values(a),
            Self::Mul(a) => <MulAir as BaseAir<F>>::num_public_values(a),
            Self::PreprocessedMul(a) => <PreprocessedMulAir as BaseAir<F>>::num_public_values(a),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::preprocessed_trace(a),
            Self::Mul(_) => None,
            Self::PreprocessedMul(a) => <PreprocessedMulAir as BaseAir<F>>::preprocessed_trace(a),
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::preprocessed_width(a),
            Self::Mul(a) => <MulAir as BaseAir<F>>::preprocessed_width(a),
            Self::PreprocessedMul(a) => <PreprocessedMulAir as BaseAir<F>>::preprocessed_width(a),
        }
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::preprocessed_next_row_columns(a),
            Self::Mul(a) => <MulAir as BaseAir<F>>::preprocessed_next_row_columns(a),
            Self::PreprocessedMul(a) => {
                <PreprocessedMulAir as BaseAir<F>>::preprocessed_next_row_columns(a)
            }
        }
    }
}

// Heterogeneous enum wrapper for lookup-enabled AIRs
// `FibLookups` receives its inputs from `MulAirLookups` AIRs
// (see `FibAirLookups` and `MulAirLookups` definitions for more details)
#[derive(Clone)]
enum DemoAirWithLookups {
    FibLookups(FibAirLookups),
    MulLookups(MulAirLookups),
}

impl<F: Field> BaseAir<F> for DemoAirWithLookups {
    fn width(&self) -> usize {
        match self {
            Self::FibLookups(a) => <FibAirLookups as BaseAir<F>>::width(a),
            Self::MulLookups(a) => <MulAirLookups as BaseAir<F>>::width(a),
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::FibLookups(a) => <FibAirLookups as BaseAir<F>>::num_public_values(a),
            Self::MulLookups(a) => <MulAirLookups as BaseAir<F>>::num_public_values(a),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::FibLookups(a) => <FibAirLookups as BaseAir<F>>::preprocessed_trace(a),
            Self::MulLookups(a) => <MulAirLookups as BaseAir<F>>::preprocessed_trace(a),
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::FibLookups(a) => <FibAirLookups as BaseAir<F>>::preprocessed_width(a),
            Self::MulLookups(a) => <MulAirLookups as BaseAir<F>>::preprocessed_width(a),
        }
    }
}

impl<AB: PermutationAirBuilder + InteractionBuilder> Air<AB> for DemoAirWithLookups
where
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::FibLookups(a) => <FibAirLookups as Air<AB>>::eval(a, builder),
            Self::MulLookups(a) => <MulAirLookups as Air<AB>>::eval(a, builder),
        }
    }
}

impl<AB: PermutationAirBuilder> Air<AB> for DemoAir
where
    AB::Var: Debug,
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
            trace: &fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: &mul_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

#[test]
fn test_periodic_air() -> Result<(), impl Debug> {
    let config = make_config(42);
    let air = PeriodicAir::<Val>::new();
    let trace = air.valid_trace(1 << 6);
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: vec![],
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    verify_batch(&config, &[air], &proof, &[vec![]], common)
}

#[test]
fn test_periodic_air_zk() -> Result<(), impl Debug> {
    let config = make_config_zk(1234);
    let air = PeriodicAir::<Val>::new();
    let trace = air.valid_trace(1 << 6);
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: vec![],
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    verify_batch(&config, &[air], &proof, &[vec![]], common)
}

#[test]
fn test_two_instances_zk() -> Result<(), impl Debug> {
    let config = make_config_zk(1337);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, 2 reps

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: &fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: &mul_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
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
            trace: &fib16_trace,
            public_values: fib16_pis.clone(),
        },
        StarkInstance {
            air: &air_mul8,
            trace: &mul8_trace,
            public_values: mul8_pis.clone(),
        },
        StarkInstance {
            air: &air_fib8,
            trace: &fib8_trace,
            public_values: fib8_pis.clone(),
        },
    ];

    let prover_data: ProverData<MyConfig> = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    let airs = vec![air_fib16, air_mul8, air_fib8];
    let pvs = vec![fib16_pis, mul8_pis, fib8_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

#[test]
fn test_invalid_public_values_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(7);

    let (air_fib, trace, fib_pis) = create_fib_instance(4); // 16 rows
    let correct_x = fib_n(16);

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: &trace,
        public_values: fib_pis,
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);

    // Wrong public value at verify => should reject
    let airs = vec![air_fib];
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_batch(&config, &airs, &proof, &wrong_pvs, common);
    assert!(res.is_err(), "Should reject wrong public values");
    Ok::<_, Box<dyn std::error::Error>>(())
}

#[test]
fn test_short_public_values_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(7);

    let (air_fib, trace, fib_pis) = create_fib_instance(4);
    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: &trace,
        public_values: fib_pis,
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);

    let airs = vec![air_fib];
    let short_pvs = vec![vec![Val::from_u64(0), Val::from_u64(1)]];
    let err = verify_batch(&config, &airs, &proof, &short_pvs, common)
        .expect_err("Should reject short public values");
    match err {
        VerificationError::InvalidProofShape(
            InvalidProofShapeError::PublicValuesLengthMismatch { expected, got },
        ) => {
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
    Ok::<_, Box<dyn std::error::Error>>(())
}

#[test]
fn test_degree_bits_too_large_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // The verifier constructs evaluation domains via `1 << degree_bits`.
    // A malicious proof can set degree_bits >= usize::BITS (e.g. 64 on
    // a 64-bit platform), which would panic on the left shift. The guard
    // must reject this before any domain construction happens.

    // Non-ZK config — the overflow check is independent of the ZK setting.
    let config = make_config(7);

    // Build a valid Fibonacci proof with a 2^4 = 16-row trace.
    let (air_fib, trace, fib_pis) = create_fib_instance(4);
    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: &trace,
        public_values: fib_pis.clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let mut proof = prove_batch(&config, &instances, &prover_data);

    // Mutation: overwrite the first AIR's degree_bits to exactly the bit
    // width of usize, which is the smallest value that overflows.
    //
    //     degree_bits = 64  (on 64-bit)
    //     1_usize << 64  →  shift overflow  →  must be caught
    proof.degree_bits[0] = usize::BITS as usize;

    let airs = vec![air_fib];

    // Verification must fail with a structured error, not a panic.
    let err = verify_batch(&config, &airs, &proof, &[fib_pis], common)
        .expect_err("Should reject oversized degree_bits");

    // Verify the error carries the correct diagnostic fields:
    //   - air: Some(0)          — first (and only) AIR instance
    //   - maximum: BITS - 1     — largest safe exponent (63 on 64-bit)
    //   - got: BITS             — the tampered value we injected (64)
    match err {
        VerificationError::InvalidProofShape(InvalidProofShapeError::DegreeBitsTooLarge {
            air,
            maximum,
            got,
        }) => {
            assert_eq!(air, Some(0));
            assert_eq!(maximum, usize::BITS as usize - 1);
            assert_eq!(got, usize::BITS as usize);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
    Ok::<_, Box<dyn std::error::Error>>(())
}

#[test]
fn test_degree_bits_too_small_for_zk_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // In ZK mode the prover extends the trace by one bit (is_zk = 1), so
    // the verifier computes `base_degree_bits = degree_bits - is_zk`.
    // If degree_bits < is_zk that subtraction underflows. The guard must
    // reject this with a clear minimum-threshold error.

    // ZK-enabled config — is_zk = 1, meaning degree_bits must be >= 1.
    let config = make_config_zk(1337);

    // Build a valid Fibonacci proof with a 2^4 = 16-row trace.
    let (air_fib, trace, fib_pis) = create_fib_instance(4);
    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: &trace,
        public_values: fib_pis.clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let mut proof = prove_batch(&config, &instances, &prover_data);

    // Mutation: set degree_bits to 0, which is below the ZK minimum.
    //
    //     is_zk = 1
    //     degree_bits = 0
    //     base_degree_bits = 0 - 1  →  underflow  →  must be caught
    proof.degree_bits[0] = 0;

    let airs = vec![air_fib];

    // Verification must fail before any domain arithmetic.
    let err = verify_batch(&config, &airs, &proof, &[fib_pis], common)
        .expect_err("Should reject too-small degree_bits in zk mode");

    // Verify the error carries the correct diagnostic fields:
    //   - air: Some(0)   — first (and only) AIR instance
    //   - minimum: 1     — is_zk, the smallest acceptable degree_bits
    //   - got: 0         — the tampered value we injected
    match err {
        VerificationError::InvalidProofShape(InvalidProofShapeError::DegreeBitsTooSmall {
            air,
            minimum,
            got,
        }) => {
            assert_eq!(air, Some(0));
            assert_eq!(minimum, 1);
            assert_eq!(got, 0);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
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
            trace: &mul2_trace,
            public_values: mul2_pis.clone(),
        },
        StarkInstance {
            air: &air_fib,
            trace: &fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul3,
            trace: &mul3_trace,
            public_values: mul3_pis.clone(),
        },
    ];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul2_pis, fib_pis, mul3_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

#[test]
fn test_preprocessed_tampered_fails() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(9999);

    // Single Fibonacci instance with 8 rows and preprocessed index column.
    let (air, trace, fib_pis) = create_fib_instance(3);
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: fib_pis.clone(),
    }];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);

    // First, sanity-check that verification succeeds with matching preprocessed data.
    let airs = vec![air];
    let ok_res = verify_batch(&config, &airs, &proof, from_ref(&fib_pis), common);
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
    let prover_data_tampered =
        ProverData::from_airs_and_degrees(&config, &airs_tampered, &degree_bits);
    let common_tampered = &prover_data_tampered.common;

    let res = verify_batch(&config, &airs_tampered, &proof, &[fib_pis], common_tampered);
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
        trace: &trace1,
        public_values: fib_pis1.clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances1);
    let common = &prover_data.common;
    let proof1 = prove_batch(&config, &instances1, &prover_data);

    // Verify the first proof.
    let airs = vec![air];
    let res1 = verify_batch(&config, &airs, &proof1, from_ref(&fib_pis1), common);
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
        air: &airs[0],
        trace: &trace2,
        public_values: fib_pis2.clone(),
    }];
    let proof2 = prove_batch(&config, &instances2, &prover_data);

    let res2 = verify_batch(&config, &airs, &proof2, &[fib_pis2], common);
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
        trace: &fib_trace,
        public_values: fib_pis.clone(),
    }];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    let airs = vec![air_fib];
    verify_batch(&config, &airs, &proof, &[fib_pis], common)
}

#[test]
fn test_quotient_domain_size_not_multiple_of_packed_field_width() -> Result<(), impl Debug> {
    let config = make_config_allow_tiny_trace(77_007);

    // 2 rows → log2(trace)=1, one quotient chunk → quotient domain size = 2^1 = 2.
    let (air_fib, fib_trace, fib_pis) = create_fib_instance(1);

    let instances = vec![StarkInstance {
        air: &air_fib,
        trace: &fib_trace,
        public_values: fib_pis.clone(),
    }];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    let airs = vec![air_fib];
    verify_batch(&config, &airs, &proof, &[fib_pis], common)
}

#[test]
fn test_mixed_preprocessed() -> Result<(), impl Debug> {
    let config = make_config(8888);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows, has preprocessed
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, no preprocessed

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: &fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: &mul_trace,
            public_values: mul_pis.clone(),
        },
    ];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;

    let proof = prove_batch(&config, &instances, &prover_data);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];

    verify_batch(&config, &airs, &proof, &pvs, common)
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
        trace: &fib_trace,
        public_values: fib_pis.clone(),
    }];

    // Generate a valid proof
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let valid_proof = prove_batch(&config, &instances, &prover_data);

    // Tamper with the proof: change trace_local to have wrong width
    let mut tampered_proof = p3_batch_stark::proof::BatchProof {
        commitments: BatchCommitments {
            main: valid_proof.commitments.main,
            quotient_chunks: valid_proof.commitments.quotient_chunks,
            permutation: valid_proof.commitments.permutation,
            random: valid_proof.commitments.random,
        },
        opened_values: BatchOpenedValues {
            instances: vec![OpenedValuesWithLookups {
                base_opened_values: OpenedValues {
                    trace_local: vec![
                        valid_proof.opened_values.instances[0]
                            .base_opened_values
                            .trace_local[0],
                    ], // Wrong width: 1 instead of 2
                    trace_next: valid_proof.opened_values.instances[0]
                        .base_opened_values
                        .trace_next
                        .clone(),
                    preprocessed_local: None,
                    preprocessed_next: None,
                    quotient_chunks: valid_proof.opened_values.instances[0]
                        .base_opened_values
                        .quotient_chunks
                        .clone(),
                    random: None,
                },
                permutation_local: valid_proof.opened_values.instances[0]
                    .permutation_local
                    .clone(),
                permutation_next: valid_proof.opened_values.instances[0]
                    .permutation_next
                    .clone(),
            }],
        },
        opening_proof: valid_proof.opening_proof.clone(),
        global_lookup_data: valid_proof.global_lookup_data.clone(),
        degree_bits: valid_proof.degree_bits.clone(),
    };

    // Verification should fail due to width mismatch
    let airs = vec![air_fib];
    let res = verify_batch(&config, &airs, &tampered_proof, from_ref(&fib_pis), common);
    assert!(
        res.is_err(),
        "Verifier should reject trace with wrong width"
    );

    // Also test wrong trace_next width
    tampered_proof.opened_values.instances[0]
        .base_opened_values
        .trace_local = valid_proof.opened_values.instances[0]
        .base_opened_values
        .trace_local
        .clone();
    tampered_proof.opened_values.instances[0]
        .base_opened_values
        .trace_next = Some(vec![
        valid_proof.opened_values.instances[0]
            .base_opened_values
            .trace_next
            .as_ref()
            .unwrap()[0],
    ]); // Wrong width

    let res = verify_batch(&config, &airs, &tampered_proof, from_ref(&fib_pis), common);
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
            trace: &tr_a,
            public_values: pv_a.clone(),
        },
        StarkInstance {
            air: &air_b,
            trace: &tr_b,
            public_values: pv_b.clone(),
        },
    ];

    // DemoAir::Fib has preprocessed columns, so compute degrees for swapped verification
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();

    let prover_data = ProverData::from_instances(&config, &instances);
    let proof = prove_batch(&config, &instances, &prover_data);

    // Swap order at verify -> should fail (create new CommonData with swapped AIRs)
    let airs_swapped = vec![air_b, air_a];
    let prover_data_swapped =
        ProverData::from_airs_and_degrees(&config, &airs_swapped, &log_degrees);
    let common_swapped = &prover_data_swapped.common;
    let res = verify_batch(
        &config,
        &airs_swapped,
        &proof,
        &[pv_b, pv_a],
        common_swapped,
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
        trace: &tr,
        public_values: pv.clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);

    let mut tampered = proof;
    tampered.opened_values.instances[0]
        .base_opened_values
        .quotient_chunks[0]
        .pop();

    let airs = vec![air];
    let res = verify_batch(&config, &airs, &tampered, from_ref(&pv), common);
    assert!(
        res.is_err(),
        "Verifier should reject truncated quotient chunk element"
    );
}

#[test]
fn test_circle_stark_batch() -> Result<(), impl Debug> {
    // Test batch-stark with Circle PCS (non-two-adic field)
    let config = make_circle_config();

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

    let fib_pis1 = vec![
        CircleVal::from_u64(0),
        CircleVal::from_u64(1),
        CircleVal::from_u64(fib_n(8)),
    ]; // F_8 = 21
    let fib_pis2 = vec![
        CircleVal::from_u64(0),
        CircleVal::from_u64(1),
        CircleVal::from_u64(fib_n(4)),
    ]; // F_4 = 3

    let trace1 = fib_trace::<CircleVal>(0, 1, 8);
    let trace2 = fib_trace::<CircleVal>(0, 1, 4);

    let airs = vec![air_fib1, air_fib2];

    let instances = vec![
        StarkInstance {
            air: &airs[0],
            trace: &trace1,
            public_values: fib_pis1.clone(),
        },
        StarkInstance {
            air: &airs[1],
            trace: &trace2,
            public_values: fib_pis2.clone(),
        },
    ];

    // Generate batch-proof
    // Plain FibonacciAir doesn't have preprocessed columns
    let prover_data = ProverData::empty(airs.len());
    let proof = prove_batch(&config, &instances, &prover_data);

    // Verify batch-proof
    let public_values = vec![fib_pis1, fib_pis2];
    let common = &prover_data.common;
    verify_batch(&config, &airs, &proof, &public_values, common)
        .map_err(|e| format!("Verification failed: {:?}", e))
}

type CompatCase<Config, V> = (
    Config,
    Vec<DemoAirWithLookups>,
    Vec<RowMajorMatrix<V>>,
    Vec<Vec<V>>,
    Vec<usize>,
);

fn two_adic_compat_case() -> CompatCase<MyConfig, Val> {
    let config = make_two_adic_compat_config(777);
    let reps = 2;
    let log_n = 5;
    let n = 1 << log_n;

    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    );

    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, None);

    let mul_trace = mul_trace::<Val>(n, reps);
    let fib_trace = fib_trace::<Val>(0, 1, n);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];

    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let is_zk = config.is_zk();
    let log_degrees: Vec<usize> = vec![mul_trace.height(), fib_trace.height()]
        .into_iter()
        .map(|height| log2_strict_usize(height) + is_zk)
        .collect();
    (
        config,
        vec![air1, air2],
        vec![mul_trace, fib_trace],
        vec![vec![], fib_pis],
        log_degrees,
    )
}

fn circle_compat_case() -> CompatCase<CircleConfig, CircleVal> {
    let config = make_circle_config();
    let reps = 2;
    let log_n = 3;
    let n = 1 << log_n;

    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    );

    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, None);

    let mul_trace = mul_trace::<CircleVal>(n, reps);
    let fib_trace = fib_trace::<CircleVal>(0, 1, n);
    let fib_pis = vec![
        CircleVal::from_u64(0),
        CircleVal::from_u64(1),
        CircleVal::from_u64(fib_n(n)),
    ];

    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let is_zk = config.is_zk();
    let log_degrees: Vec<usize> = vec![mul_trace.height(), fib_trace.height()]
        .into_iter()
        .map(|height| log2_strict_usize(height) + is_zk)
        .collect();
    (
        config,
        vec![air1, air2],
        vec![mul_trace, fib_trace],
        vec![vec![], fib_pis],
        log_degrees,
    )
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

#[test]
fn verify_two_adic_compat_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (config, airs, _traces, pvs, _log_degrees) = two_adic_compat_case();
    let proof_bytes = read_fixture(TWO_ADIC_FIXTURE)
        .expect("Missing fixture. Run: cargo test -p p3-batch-stark --test simple -- --ignored");
    let proof: BatchProof<MyConfig> = postcard::from_bytes(&proof_bytes)?;
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &proof.degree_bits);
    let common = &prover_data.common;
    verify_batch(&config, &airs, &proof, &pvs, common)?;
    Ok(())
}

#[test]
fn verify_circle_compat_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let (config, airs, _traces, pvs, _log_degrees) = circle_compat_case();
    let proof_bytes = read_fixture(CIRCLE_FIXTURE)
        .expect("Missing fixture. Run: cargo test -p p3-batch-stark --test simple -- --ignored");
    let proof: BatchProof<CircleConfig> = postcard::from_bytes(&proof_bytes)?;
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &proof.degree_bits);
    let common = &prover_data.common;
    verify_batch(&config, &airs, &proof, &pvs, common)?;
    Ok(())
}

#[test]
#[ignore]
fn generate_two_adic_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Regen: cargo test -p p3-batch-stark --test simple -- --ignored
    let (config, airs, traces, pvs, log_degrees) = two_adic_compat_case();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &log_degrees);
    let _common = &prover_data.common;
    let traces = [&traces[0], &traces[1]];
    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs);
    let proof = prove_batch(&config, &instances, &prover_data);
    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(TWO_ADIC_FIXTURE, &bytes)?;
    Ok(())
}

#[test]
#[ignore]
fn generate_circle_fixture() -> Result<(), Box<dyn std::error::Error>> {
    // Regen: cargo test -p p3-batch-stark --test simple -- --ignored
    let (config, airs, traces, pvs, log_degrees) = circle_compat_case();
    let prover_data = ProverData::from_airs_and_degrees(&config, &airs, &log_degrees);
    let _common = &prover_data.common;
    let traces = [&traces[0], &traces[1]];
    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs);
    let proof = prove_batch(&config, &instances, &prover_data);
    let bytes = postcard::to_allocvec(&proof)?;
    write_fixture(CIRCLE_FIXTURE, &bytes)?;
    Ok(())
}

#[test]
fn test_preprocessed_constraint_positive() -> Result<(), impl Debug> {
    // Test that preprocessed columns are correctly used in constraints
    // Enforces: main[0] = 2 * preprocessed[0]
    let config = make_config(8888);

    let (air, trace, pis) = create_preprocessed_mul_instance(4, 2); // 16 rows, multiplier=2

    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: pis.clone(),
    }];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);
    let airs = vec![air];
    verify_batch(&config, &airs, &proof, &[pis], common)
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
        trace: &trace,
        public_values: pis.clone(),
    }];

    let prover_data = ProverData::from_instances(&config, &instances);
    let proof = prove_batch(&config, &instances, &prover_data);

    // Verify with wrong multiplier=3 (should fail)
    let air_verify = DemoAir::PreprocessedMul(PreprocessedMulAir {
        log_height: 4,
        multiplier: 3, // Wrong multiplier!
    });
    let airs = vec![air_verify];
    let degree_bits = proof.degree_bits.clone();
    let prover_data_verify = ProverData::from_airs_and_degrees(&config, &airs, &degree_bits);
    let common_verify = &prover_data_verify.common;

    let res = verify_batch(&config, &airs, &proof, &[pis], common_verify);
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
            trace: &fib_trace,
            public_values: fib_pis.clone(),
        },
        StarkInstance {
            air: &air_mul,
            trace: &mul_trace,
            public_values: mul_pis.clone(),
        },
        StarkInstance {
            air: &air_pp_mul,
            trace: &pp_mul_trace,
            public_values: pp_mul_pis.clone(),
        },
    ];

    let prover_data = ProverData::from_instances(&config, &instances);
    let common = &prover_data.common;
    let proof = prove_batch(&config, &instances, &prover_data);

    let airs = vec![air_fib, air_mul, air_pp_mul];
    let pvs = vec![fib_pis, mul_pis, pp_mul_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

// Tests for local and global lookup handling in multi-stark.

/// Test with local lookups only using MulAirLookups
#[test]
fn test_batch_stark_one_instance_local_only() -> Result<(), impl Debug> {
    let config = make_config(2024);

    let reps = 1;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, vec![]); // local only

    let log_height = 3; // 8 rows
    let mul_trace = mul_trace::<Val>(1 << log_height, reps);

    let airs = [DemoAirWithLookups::MulLookups(mul_air_lookups)];

    // Get lookups from the lookup-enabled AIRs
    let prover_data = ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height]);
    let common = &prover_data.common;
    let traces = [&mul_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![]]);

    let proof = prove_batch(&config, &instances, &prover_data);

    let pvs = vec![vec![]];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

/// Test with local lookups only, which fail due to wrong permutation column.
/// The failure occurs in `check_constraints` during proof generation, since it fails the last local constraint (the final local sum is not zero).
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "constraints not satisfied on row 7")]
fn test_batch_stark_one_instance_local_fails() {
    let config = make_config(2024);

    let reps = 2;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, vec![]); // local only

    let log_height = 3; // 8 rows
    let mut mul_trace = mul_trace::<Val>(1 << log_height, reps);

    // Tamper with the permutation column to cause lookup failure.
    mul_trace.values[reps * 3] = Val::from_u64(9999);

    let airs = [DemoAirWithLookups::MulLookups(mul_air_lookups)];

    // Get lookups from the lookup-enabled AIRs
    let prover_data = ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height]);
    let _common = &prover_data.common;
    let traces = [&mul_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![]]);

    prove_batch(&config, &instances, &prover_data);
}

/// Test with local lookups only, which fail due to wrong permutation column.
/// The verification fails, since the last local constraint fails (the final local sum is not zero).
#[cfg(not(debug_assertions))]
#[test]
#[should_panic(expected = "OodEvaluationMismatch")]
fn test_batch_stark_one_instance_local_fails() {
    let config = make_config(2024);

    let reps = 2;
    let log_height = 3; // 8 rows
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, vec![]); // local only

    let mut mul_trace = mul_trace::<Val>(1 << log_height, reps);

    // Tamper with the permutation column to cause lookup failure.
    mul_trace.values[reps * 3] = Val::from_u64(9999);

    let airs = [DemoAirWithLookups::MulLookups(mul_air_lookups)];

    // Get lookups from the lookup-enabled AIRs
    let prover_data = ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height]);
    let common = &prover_data.common;
    let traces = [&mul_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![]]);

    let proof = prove_batch(&config, &instances, &prover_data);

    verify_batch(&config, &airs, &proof, &[vec![]], common).unwrap();
}

/// Test with local lookups only using MulAirLookups
#[test]
fn test_batch_stark_local_lookups_only() -> Result<(), impl Debug> {
    let config = make_config(2024);

    let log_height = 4; // 16 rows
    let height = 1 << log_height;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps: 2 };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, vec![]); // local only
    let fib_air_lookups = FibAirLookups::new(
        FibonacciAir {
            log_height,
            tamper_index: None,
        },
        false,
        None,
    ); // no lookups

    let mul_trace = mul_trace::<Val>(height, 2);
    let fib_trace = fib_trace::<Val>(0, 1, 16);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let airs = [air1, air2];

    // Get lookups from the lookup-enabled AIRs
    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height, log_height]);
    let common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![], fib_pis.clone()]);

    let proof = prove_batch(&config, &instances, &prover_data);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

/// Test with global lookups only using MulAirLookups and FibAirLookups
#[test]
fn test_batch_stark_global_lookups_only() -> Result<(), impl Debug> {
    let config = make_config(2025);

    let reps = 2;
    // Create instances with global lookups configuration
    let mul_air = MulAir { reps };
    // Both global lookups (for each rep) look into the same FibAir inputs, so they share the same name.
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    ); // global only

    let log_n = 3;
    let n = 1 << log_n;

    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, None); // global lookups

    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace = fib_trace::<Val>(0, 1, n);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    // Get lookups from the lookup-enabled AIRs
    let airs = [air1, air2];
    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_n, log_n]);
    let common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![], fib_pis.clone()]);

    let proof = prove_batch(&config, &instances, &prover_data);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

/// Test with both local and global lookups using MulAirLookups and FibAirLookups
#[test]
fn test_batch_stark_both_lookups() -> Result<(), impl Debug> {
    let config = make_config(2026);

    let reps = 2;
    // Create instances with both local and global lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        true,
        true,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    ); // both

    let log_height = 4;
    let height = 1 << log_height;

    let fibonacci_air = FibonacciAir {
        log_height,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, None); // global lookups

    let mul_trace = mul_trace::<Val>(height, 2);
    let fib_trace = fib_trace::<Val>(0, 1, height);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let airs = [air1, air2];
    // Get lookups from the lookup-enabled AIRs
    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height, log_height]);
    let common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![], fib_pis.clone()]);

    let proof = prove_batch(&config, &instances, &prover_data);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

/// Test with both local and global lookups using MulAirLookups and FibAirLookups, with ZK mode activated
#[test]
fn test_batch_stark_both_lookups_zk() -> Result<(), impl Debug> {
    let config = make_config_zk(2026);

    let reps = 2;
    // Create instances with both local and global lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        true,
        true,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    ); // both

    let log_height = 4;
    let height = 1 << log_height;

    let fibonacci_air = FibonacciAir {
        log_height,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, None); // global lookups

    let mul_trace = mul_trace::<Val>(height, 2);
    let fib_trace = fib_trace::<Val>(0, 1, height);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let airs = [air1, air2];
    // Get lookups from the lookup-enabled AIRs
    let prover_data = ProverData::<MyHidingConfig>::from_airs_and_degrees(
        &config,
        &airs,
        &[log_height + config.is_zk(), log_height + config.is_zk()],
    );
    let common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![], fib_pis.clone()]);

    let proof = prove_batch(&config, &instances, &prover_data);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, common)
}

#[cfg(not(debug_assertions))]
#[test]
#[should_panic(expected = "LookupError(\"GlobalCumulativeMismatch(None): MulFib2\")")]
fn test_batch_stark_failed_global_lookup() {
    test_batch_stark_failed_global_lookup_inner();
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(
    expected = "Lookup mismatch (global lookup 'MulFib2'): tuple [\"0\", \"1\"] has net multiplicity 2013265920. Locations: [Location { instance: 0, lookup: 1, row: 0 }]"
)]
fn test_batch_stark_failed_global_lookup() {
    test_batch_stark_failed_global_lookup_inner();
}

fn test_batch_stark_failed_global_lookup_inner() {
    let config = make_config(2025);

    let reps = 2;
    // Create instances with global lookups configuration
    let mul_air = MulAir { reps };
    // MulAir uses two different names for its reps, which will create two separate global lookups
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        vec!["MulFib1".to_string(), "MulFib2".to_string()], // Different names!
    );
    // This creates a mismatch: MulAir sends to "MulFib1" and "MulFib2"
    // but FibAir only receives from "MulFib1"
    let log_n = 3;
    let n = 1 << log_n;
    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, Some(("MulFib1".to_string(), 1)));

    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace = fib_trace::<Val>(0, 1, n);

    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    let traces = [&mul_trace, &fib_trace];
    let pvs = vec![vec![], fib_pis];
    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    // Get lookups from the lookup-enabled AIRs
    let airs = [air1, air2];
    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_n, log_n]);
    let common = &prover_data.common;

    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs);

    let proof = prove_batch(&config, &instances, &prover_data);

    // This should panic with GlobalCumulativeMismatch because:
    // - MulAir sends values to "MulFib1" and "MulFib2" lookups
    // - FibAir only receives from "MulFib" lookup
    // - The global cumulative sums won't match
    verify_batch(&config, &airs, &proof, &pvs, common).unwrap();
}

#[test]
fn test_batch_stark_rejects_truncated_global_lookup_data() {
    let config = make_config(2025);

    let reps = 2;
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    );

    let log_n = 3;
    let n = 1 << log_n;
    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, None);

    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace = fib_trace::<Val>(0, 1, n);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];

    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let airs = [air1, air2];
    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_n, log_n]);
    let common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace];
    let pvs = vec![vec![], fib_pis];

    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs);
    let mut proof = prove_batch(&config, &instances, &prover_data);

    proof.global_lookup_data[0].pop();

    let err = verify_batch(&config, &airs, &proof, &pvs, common)
        .expect_err("Verifier should reject truncated global lookup data");
    match err {
        VerificationError::InvalidProofShape(
            InvalidProofShapeError::GlobalLookupDataCountMismatch { air, expected, got },
        ) => {
            assert_eq!(air, 0);
            assert_eq!(expected, 2);
            assert_eq!(got, 1);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
}

/// Builds a 3-AIR batch proof with global lookups, suitable for metadata
/// tampering tests. Returns all state needed to mutate and re-verify.
///
/// The fixture has the following topology:
///
/// ```text
///     AIR 0 (MulAir)   — 2 global lookups: "MulFib1", "MulFib2"
///     AIR 1 (FibAir)   — 1 global lookup sending into "MulFib1"
///     AIR 2 (FibAir)   — 1 global lookup sending into "MulFib2"
/// ```
#[allow(clippy::type_complexity)]
fn make_global_lookup_proof() -> (
    MyConfig,
    [DemoAirWithLookups; 3],
    BatchProof<MyConfig>,
    Vec<Vec<Val>>,
    CommonData<MyConfig>,
) {
    let config = make_config(2025);

    // MulAir with 2 repetitions and two named global lookups.
    let reps = 2;
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        vec!["MulFib1".to_string(), "MulFib2".to_string()],
    );

    // Two FibAir instances, each sending into one of the MulAir lookups.
    let log_n = 3;
    let n = 1 << log_n;
    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups_1 =
        FibAirLookups::new(fibonacci_air, true, Some(("MulFib1".to_string(), 1)));
    let fib_air_lookups_2 =
        FibAirLookups::new(fibonacci_air, true, Some(("MulFib2".to_string(), 1)));

    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace_1 = fib_trace::<Val>(0, 1, n);
    let fib_trace_2 = fib_trace::<Val>(0, 1, n);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];

    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups_1);
    let air3 = DemoAirWithLookups::FibLookups(fib_air_lookups_2);

    let airs = [air1, air2, air3];
    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_n, log_n, log_n]);
    let _common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace_1, &fib_trace_2];
    let pvs = vec![vec![], fib_pis.clone(), fib_pis];

    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs);
    let proof = prove_batch(&config, &instances, &prover_data);
    (config, airs, proof, pvs, prover_data.common)
}

#[test]
fn test_batch_stark_rejects_tampered_global_lookup_metadata() {
    // Global lookup data carries a `name` field that identifies which lookup
    // interaction the data belongs to. The verifier must cross-check this
    // against the AIR's declared interactions. A malicious proof could rename
    // a lookup to mix cumulative values across unrelated interactions,
    // breaking soundness.

    let (config, airs, mut proof, pvs, common) = make_global_lookup_proof();

    // Fixture state: AIR 0 has two global lookups.
    //
    //     global_lookup_data[0][0].name = "MulFib1"  (from AIR declaration)
    //     global_lookup_data[0][1].name = "MulFib2"  (from AIR declaration)
    //
    // Mutation: rename the first lookup to "tampered".
    //
    //     proof says:   name = "tampered"
    //     AIR declares: name = "MulFib1"
    //     → mismatch → error on AIR 0, lookup 0
    proof.global_lookup_data[0][0].name = "tampered".to_string();

    let err = verify_batch(&config, &airs, &proof, &pvs, &common)
        .expect_err("Verifier should reject tampered global lookup metadata");

    // Verify all diagnostic fields:
    //   - air: 0                    — MulAir is the first instance
    //   - lookup: 0                 — first global lookup within that AIR
    //   - expected_name: "MulFib1"  — what the AIR declares
    //   - got_name: "tampered"      — what the proof supplied
    //   - expected_aux_column: 0       — column index from the AIR
    //   - got_aux_column: 0            — unchanged, only name was tampered
    match err {
        VerificationError::InvalidProofShape(
            InvalidProofShapeError::GlobalLookupDataMetadataMismatch {
                air,
                lookup,
                expected_name,
                got_name,
                expected_aux_column,
                got_aux_column,
            },
        ) => {
            assert_eq!(air, 0);
            assert_eq!(lookup, 0);
            assert_eq!(expected_name, "MulFib1");
            assert_eq!(got_name, "tampered");
            assert_eq!(expected_aux_column, 0);
            assert_eq!(got_aux_column, 0);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
}

#[test]
fn test_batch_stark_rejects_tampered_global_lookup_aux_idx() {
    // The `aux_idx` field in global lookup data identifies which auxiliary
    // (permutation) column holds the running sum for that interaction. A
    // malicious proof could point to a different column, causing the verifier
    // to check the wrong cumulative value. The guard validates aux_idx
    // against the AIR declaration.

    let (config, airs, mut proof, pvs, common) = make_global_lookup_proof();

    // Fixture state: AIR 0's first global lookup has aux_idx = 0.
    //
    // Mutation: set aux_idx to 42, a column that doesn't correspond to
    // this interaction.
    //
    //     proof says:   aux_idx = 42
    //     AIR declares: aux_idx = 0
    //     → mismatch → error on AIR 0, lookup 0
    proof.global_lookup_data[0][0].aux_column = 42;

    let err = verify_batch(&config, &airs, &proof, &pvs, &common)
        .expect_err("Verifier should reject tampered global lookup aux index");

    // Verify all diagnostic fields:
    //   - air: 0                    — MulAir is the first instance
    //   - lookup: 0                 — first global lookup within that AIR
    //   - expected_name: "MulFib1"  — unchanged, only aux_idx was tampered
    //   - got_name: "MulFib1"       — matches (name is correct)
    //   - expected_aux_column: 0       — what the AIR declares
    //   - got_aux_column: 42           — the tampered value
    match err {
        VerificationError::InvalidProofShape(
            InvalidProofShapeError::GlobalLookupDataMetadataMismatch {
                air,
                lookup,
                expected_name,
                got_name,
                expected_aux_column,
                got_aux_column,
            },
        ) => {
            assert_eq!(air, 0);
            assert_eq!(lookup, 0);
            assert_eq!(expected_name, "MulFib1");
            assert_eq!(got_name, "MulFib1");
            assert_eq!(expected_aux_column, 0);
            assert_eq!(got_aux_column, 42);
        }
        _ => panic!("unexpected error: {err:?}"),
    }
}

/// Test mixing instances with lookups and instances without lookups.
/// We have the following instances:
/// - MulAir with both local and global lookups (looking into two different FibAir instances for each rep)
/// - FibAir without lookups
/// - FibAir with global lookups (sends values for first rep of MulAir)
/// - MulAir without lookups
/// - FibAir with global lookups (sends values for second rep of MulAir)
/// - MulAir with local lookups only
macro_rules! run_batch_stark_mixed_lookups {
    ($config:expr, $ConfigTy:ty) => {{
        let config: $ConfigTy = $config;

        let reps = 2;

        // Create instances with different lookup configurations:
        let mul_air_with_lookups = MulAir { reps };
        // This AIR has two different global lookups (one for each rep) with two different names.
        // It also has two local lookups (one for each rep).
        let mul_air_lookups = MulAirLookups::new(
            mul_air_with_lookups,
            true,
            true,
            vec!["MulFib1".to_string(), "MulFib2".to_string()],
        );
        // This AIR has no lookups.
        let mul_air_no_lookups =
            MulAirLookups::new(mul_air_with_lookups, false, false, vec![]);
        // This AIR only has local lookups.
        let mul_air_local_lookups =
            MulAirLookups::new(mul_air_with_lookups, true, false, vec![]); // local lookups only

        let log_n1 = 4; // 16 rows
        let log_n2 = 3; // 8 rows
        let n1 = 1 << log_n1;
        let n2 = 1 << log_n2;

        let fib_air_lookups = FibonacciAir {
            log_height: log_n1,
            tamper_index: None,
        };

        let fib_air_no_lookups = FibonacciAir {
            log_height: log_n2,
            tamper_index: None,
        };

        // The mul air with global lookups looks into two different Fibonacci instances.
        // So we have to create two separate FibAir instances with a different global lookup name.
        let fib_air_with_lookups1 = FibAirLookups::new(
            fib_air_lookups,
            true,
            Some(("MulFib1".to_string(), 1)),
        ); // global lookups
        let fib_air_with_lookups2 = FibAirLookups::new(
            fib_air_lookups,
            true,
            Some(("MulFib2".to_string(), 1)),
        ); // global lookups
        let fib_air_no_lookups =
            FibAirLookups::new(fib_air_no_lookups, false, None); // global lookups

        // Generate traces. The airs with and without lookups have different heights.
        let mul_with_lookups_trace = mul_trace::<Val>(n1, reps);
        let fib_with_lookups_trace = fib_trace::<Val>(0, 1, n1);
        let mul_no_lookups_trace = mul_trace::<Val>(n2, reps);
        let fib_no_lookups_trace = fib_trace::<Val>(0, 1, n2);

        // Public values
        let fib_with_lookups_pis =
            vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];
        let fib_no_lookups_pis =
            vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(8))];

        // Create lookup-enabled AIRs
        let air_mul_with_lookups = DemoAirWithLookups::MulLookups(mul_air_lookups);
        let air_fib_with_lookups1 = DemoAirWithLookups::FibLookups(fib_air_with_lookups1);
        let air_fib_with_lookups2 = DemoAirWithLookups::FibLookups(fib_air_with_lookups2);
        let air_mul_with_local_lookups =
            DemoAirWithLookups::MulLookups(mul_air_local_lookups);

        // Create non-lookup AIRs
        let air_mul_no_lookups = DemoAirWithLookups::MulLookups(mul_air_no_lookups);
        let air_fib_no_lookups = DemoAirWithLookups::FibLookups(fib_air_no_lookups);

        let mut all_airs = vec![
            air_mul_with_lookups,
            air_fib_no_lookups,
            air_fib_with_lookups1,
            air_mul_no_lookups,
            air_fib_with_lookups2,
            air_mul_with_local_lookups,
        ];

        // Get all lookups
        let prover_data = ProverData::<$ConfigTy>::from_airs_and_degrees(
            &config,
            &mut all_airs,
            &[log_n1, log_n2, log_n1, log_n2, log_n1, log_n1],
        );
        let common = &prover_data.common;

        let traces = [&mul_with_lookups_trace, &fib_no_lookups_trace, &fib_with_lookups_trace, &mul_no_lookups_trace, &fib_with_lookups_trace, &mul_with_lookups_trace];

        // Get all public values
        let all_pvs = vec![
            vec![],                       // mul with lookups
            fib_no_lookups_pis,           // fib no lookups
            fib_with_lookups_pis.clone(), // fib with lookups
            vec![],                       // mul no lookups
            fib_with_lookups_pis,         // fib with lookups
            vec![],                       // mul with local lookups
        ];

        // Create instances - mixing lookup and non-lookup instances
        let instances = StarkInstance::new_multiple(&all_airs, &traces, &all_pvs);

        let proof = prove_batch(&config, &instances, &prover_data);

        // Verify with mixed AIRs
        verify_batch(&config, &all_airs, &proof, &all_pvs, common)
    }};
}

#[test]
fn test_batch_stark_mixed_lookups() -> Result<(), impl Debug> {
    run_batch_stark_mixed_lookups!(make_config(2027), MyConfig)
}

#[test]
fn test_batch_stark_mixed_lookups_wide() -> Result<(), impl Debug> {
    run_batch_stark_mixed_lookups!(make_config_wide(2027), MyConfigWide)
}

// Single table with local lookup involving the Lagrange selectors. Since the selectors are not normalized,
// we need to add multiplicity columns and multiply them by the selectors.
#[derive(Debug, Clone, Copy)]
struct SingleTableLocalLookupAir;

impl<F> BaseAir<F> for SingleTableLocalLookupAir {
    fn width(&self) -> usize {
        7
    }
}

impl<AB> Air<AB> for SingleTableLocalLookupAir
where
    AB::Var: Debug,
    AB: AirBuilder + PermutationAirBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();

        let sender1 = local[0];
        let sender2 = local[1];
        let sender3 = local[2];
        let table = local[3];
        let mul1 = local[4];
        let mul2 = local[5];
        let mul3 = local[6];

        let is_first = builder.is_first_row();
        let is_trans = builder.is_transition();
        let is_last = builder.is_last_row();

        // Three independent local lookups, each with a different selector.
        // Lagrange selectors are not normalized, so we multiply on both sides.
        builder.push_local_interaction(vec![
            (vec![sender1.into()], is_first.clone()),
            (vec![table.into()], -(is_first * mul1.into())),
        ]);

        builder.push_local_interaction(vec![
            (vec![sender2.into()], is_trans.clone()),
            (vec![table.into()], -(is_trans * mul2.into())),
        ]);

        builder.push_local_interaction(vec![
            (vec![sender3.into()], is_last.clone()),
            (vec![table.into()], -(is_last * mul3.into())),
        ]);
    }
}

// Trace generation function for single table with local lookup
fn single_table_local_lookup_trace<F: Field>(height: usize) -> RowMajorMatrix<F> {
    assert!(height.is_power_of_two());
    assert!(height >= 2); // Need at least some transition rows and last row

    let width = 7;
    let mut v = F::zero_vec(height * width); // 2 columns
    // Column 0: all rows: value height - 1
    // Column 1: all rows except last: 7 to 1, and last value is 11
    // Column 2: all rows are 11 except last row which is 0
    // Column 3: Lookup table column: values 7 to 0
    // Column 4: (mult1) 1 at row 0, 0 elsewhere
    // Column 5: (mult2) 1 everywhere except last row, which is 0
    // Column 6: (mult3) 1 at last row, 0 elsewhere
    for i in 0..height {
        // Sender columns:
        // Column 0
        v[i * width] = F::from_u64((height - 1) as u64);
        // Column 1
        v[i * width + 1] = if i < height - 1 {
            F::from_u64((height - i - 1) as u64)
        } else {
            F::from_u64(11) // Last row value
        };
        // Column 2
        if i != height - 1 {
            v[i * width + 2] = F::from_u64(11);
        }
        // Column 3: lookup table column
        v[i * width + 3] = F::from_u64((height - i - 1) as u64);
        // Multiplicity columns
        v[i * width + 4] = if i == 0 { F::ONE } else { F::ZERO }; // mult1: is_first_row
        v[i * width + 5] = if i < height - 1 { F::ONE } else { F::ZERO }; // mult2: is_transition
        v[i * width + 6] = if i == height - 1 { F::ONE } else { F::ZERO }; // mult3: is_last_row
    }

    RowMajorMatrix::new(v, width)
}

/// Test with a single table doing local lookup between its two columns.
/// The goal of this test is to check that the use of (non-normalized) Lagrange selectors does not cause issues.
#[test]
fn test_single_table_local_lookup() -> Result<(), impl Debug> {
    let config = make_config(2029);

    let log_height = 3;
    let height = 1 << log_height; // Single table with 8 rows

    let air = SingleTableLocalLookupAir;
    let airs = [air];

    // Get lookups from the lookup-enabled AIR
    let prover_data = ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height]);
    let common = &prover_data.common;

    // Generate trace
    let trace = single_table_local_lookup_trace::<Val>(height);

    let traces = [&trace];
    let pvs = vec![vec![]]; // No public values

    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs);

    let proof = prove_batch(&config, &instances, &prover_data);

    verify_batch(&config, &airs, &proof, &pvs, common)
}

#[test]
fn test_invalid_permutation_opening_len_rejected() {
    // Tampering with permutation_local length should return Err, not panic.
    let config = make_config(9999);

    let log_height = 4;
    let height = 1 << log_height;
    let mul_air = MulAir { reps: 2 };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, vec![]);
    let fib_air_lookups = FibAirLookups::new(
        FibonacciAir {
            log_height,
            tamper_index: None,
        },
        false,
        None,
    );

    let mul_trace = mul_trace::<Val>(height, 2);
    let fib_trace = fib_trace::<Val>(0, 1, 16);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);
    let airs = [air1, air2];

    let prover_data =
        ProverData::<MyConfig>::from_airs_and_degrees(&config, &airs, &[log_height, log_height]);
    let common = &prover_data.common;
    let traces = [&mul_trace, &fib_trace];

    let instances = StarkInstance::new_multiple(&airs, &traces, &[vec![], fib_pis.clone()]);
    let mut proof = prove_batch(&config, &instances, &prover_data);

    // Find the instance with non-empty permutation openings and truncate it.
    let inst = proof
        .opened_values
        .instances
        .iter_mut()
        .find(|i| !i.permutation_local.is_empty())
        .expect("should have an instance with permutation openings");
    inst.permutation_local.pop();

    let pvs = vec![vec![], fib_pis];
    let res = verify_batch(&config, &airs, &proof, &pvs, common);
    assert!(
        res.is_err(),
        "Verifier should reject permutation opening with wrong length"
    );
}
