use core::borrow::Borrow;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder, PermutationAirBuilder,
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::proof::OpenedValuesWithLookups;
use p3_batch_stark::prover::prove_batch_no_lookups;
use p3_batch_stark::verifier::verify_batch_no_lookups;
use p3_batch_stark::{CommonData, StarkInstance, VerificationError, prove_batch, verify_batch};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs, create_test_fri_params};
use p3_keccak::Keccak256Hash;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{AirLookupHandler, AirNoLookup, Direction, Kind, Lookup};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{StarkConfig, SymbolicAirBuilder, SymbolicExpression};
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

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
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir
where
    AB::F: Field,
{
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
                .assert_eq(b.clone(), next[s].clone());
            builder
                .when_transition()
                .assert_eq(a + b, next[s + 1].clone());
        }
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
    num_lookups: usize,
    global_names: Vec<String>,
}

impl MulAirLookups {
    const fn new(
        air: MulAir,
        is_local: bool,
        is_global: bool,
        num_lookups: usize,
        global_names: Vec<String>,
    ) -> Self {
        Self {
            air,
            is_local,
            is_global,
            num_lookups,
            global_names,
        }
    }
}

impl<F> BaseAir<F> for MulAirLookups {
    fn width(&self) -> usize {
        <MulAir as BaseAir<F>>::width(&self.air)
    }
}

impl<AB: AirBuilder> Air<AB> for MulAirLookups
where
    AB::Var: Debug,
{
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);
    }
}

impl<AB> AirLookupHandler<AB> for MulAirLookups
where
    AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    AB::Var: Debug,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        let mut lookups = Vec::new();
        self.num_lookups = 0;

        // Create symbolic air builder to access symbolic variables
        let symbolic_air_builder =
            SymbolicAirBuilder::<AB::F>::new(0, BaseAir::<AB::F>::width(self), 0, 0, 0);
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

        let last_idx = symbolic_air_builder.main().width() - 1;
        let lut = symbolic_main_local[last_idx]; //  Extra column that corresponds to a permutation of 'a'

        if self.is_global {
            assert!(self.global_names.len() == self.air.reps);
        }
        // We add lookups rep by rep, so that we have a mix of local and global lookups, rather than having all local first then all global.
        for rep in 0..self.air.reps {
            if self.is_local {
                let base_idx = rep * 3;
                let a = symbolic_main_local[base_idx]; // First input
                // Create lookup inputs for each multiplication input
                // We'll create a local lookup table with integers 0 to height
                let lookup_inputs = vec![
                    // Lookup for 'a' against a permuted column.
                    (
                        vec![a.into()],
                        SymbolicExpression::Constant(AB::F::ONE),
                        Direction::Receive,
                    ),
                    // Provide the range values (this would be done in the trace generation)
                    (
                        vec![lut.into()], // This represents the range values
                        SymbolicExpression::Constant(AB::F::ONE),
                        Direction::Send,
                    ),
                ];

                let local_lookup =
                    AirLookupHandler::<AB>::register_lookup(self, Kind::Local, &lookup_inputs);
                lookups.push(local_lookup);
            }

            // Global lookups: between MulAir inputs and FibAir inputs
            if self.is_global {
                let base_idx = rep * 3;
                let a = symbolic_main_local[base_idx]; // First input
                let b = symbolic_main_local[base_idx + 1]; // Second input

                // Global lookup between MulAir inputs and FibAir inputs
                let lookup_inputs = vec![(
                    vec![a.into(), b.into()],
                    SymbolicExpression::Constant(AB::F::ONE),
                    Direction::Send, // MulAir sends data to the global lookup
                )];

                let global_lookup = AirLookupHandler::<AB>::register_lookup(
                    self,
                    Kind::Global(self.global_names[rep].clone()),
                    &lookup_inputs,
                );
                lookups.push(global_lookup);
            }
        }

        lookups
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
    num_lookups: usize,
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
            num_lookups: 0,
            name_and_mult: None,
        }
    }
}

impl FibAirLookups {
    const fn new(
        air: FibonacciAir,
        is_global: bool,
        num_lookups: usize,
        name_and_mult: Option<(String, u64)>,
    ) -> Self {
        Self {
            air,
            is_global,
            num_lookups,
            name_and_mult,
        }
    }
}

impl<F: Field> BaseAir<F> for FibAirLookups {
    fn width(&self) -> usize {
        <FibonacciAir as BaseAir<F>>::width(&self.air)
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.air.preprocessed_trace()
    }
}

impl<AB: PermutationAirBuilder + AirBuilderWithPublicValues> Air<AB> for FibAirLookups {
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);
    }
}

impl<AB> AirLookupHandler<AB> for FibAirLookups
where
    AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        let mut lookups = Vec::new();
        self.num_lookups = 0;

        if self.is_global {
            // Create symbolic air builder to access symbolic variables
            let symbolic_air_builder =
                SymbolicAirBuilder::<AB::F>::new(0, BaseAir::<AB::F>::width(self), 3, 0, 0);
            let symbolic_main = symbolic_air_builder.main();
            let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

            // Global lookups: between FibAir inputs and MulAir inputs
            // FibAir has 2 columns: left and right
            let left = symbolic_main_local[0]; // left column
            let right = symbolic_main_local[1]; // right column

            let (name, multiplicity) = match &self.name_and_mult {
                Some((n, m)) => (n.clone(), *m),
                None => ("MulFib".to_string(), 2),
            };

            // Global lookup between FibAir inputs and MulAir inputs
            let lookup_inputs = vec![(
                vec![left.into(), right.into()],
                SymbolicExpression::Constant(AB::F::from_u64(multiplicity)),
                Direction::Receive, // FibAir receives data from the global lookup
            )];

            let global_lookup =
                AirLookupHandler::<AB>::register_lookup(self, Kind::Global(name), &lookup_inputs);
            lookups.push(global_lookup);
        }

        lookups
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
}

impl<AB> Air<AB> for PreprocessedMulAir
where
    AB: AirBuilder + PairBuilder,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let local_main = main.row_slice(0).expect("Matrix is empty?");
        let local_prep = preprocessed.row_slice(0).expect("Preprocessed is empty?");

        // Enforce: main[0] = multiplier * preprocessed[0]
        builder.assert_eq(
            local_main[0].clone(),
            local_prep[0].clone() * AB::Expr::from_u64(self.multiplier),
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
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<MyPcs, Challenge, Challenger>;

fn make_config(seed: u64) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(seed);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = MyPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    StarkConfig::new(pcs, challenger)
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

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::preprocessed_trace(a),
            Self::Mul(_) => None,
            Self::PreprocessedMul(a) => <PreprocessedMulAir as BaseAir<F>>::preprocessed_trace(a),
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

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        match self {
            Self::FibLookups(a) => <FibAirLookups as BaseAir<F>>::preprocessed_trace(a),
            Self::MulLookups(a) => <MulAirLookups as BaseAir<F>>::preprocessed_trace(a),
        }
    }
}

impl<AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues> Air<AB>
    for DemoAirWithLookups
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

impl<AB> AirLookupHandler<AB> for DemoAirWithLookups
where
    AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    AB::Var: Copy + Debug,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            Self::FibLookups(a) => <FibAirLookups as AirLookupHandler<AB>>::add_lookup_columns(a),
            Self::MulLookups(a) => <MulAirLookups as AirLookupHandler<AB>>::add_lookup_columns(a),
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        match self {
            Self::FibLookups(a) => <FibAirLookups as AirLookupHandler<AB>>::get_lookups(a),
            Self::MulLookups(a) => <MulAirLookups as AirLookupHandler<AB>>::get_lookups(a),
        }
    }
}

impl<AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues> Air<AB> for DemoAir
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

type DemoAirNoLookup = AirNoLookup<DemoAir>;

/// Creates a Fibonacci instance with specified log height.
fn create_fib_instance(log_height: usize) -> (DemoAirNoLookup, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let air = DemoAirNoLookup::new(DemoAir::Fib(FibonacciAir {
        log_height,
        tamper_index: None,
    }));
    let trace = fib_trace::<Val>(0, 1, n);
    let pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    (air, trace, pis)
}

/// Creates a multiplication instance with specified configuration.
fn create_mul_instance(
    log_height: usize,
    reps: usize,
) -> (DemoAirNoLookup, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let mul = MulAir { reps };
    let air = DemoAirNoLookup::new(DemoAir::Mul(mul));
    let trace = mul_trace::<Val>(n, reps);
    let pis = vec![];
    (air, trace, pis)
}

/// Creates a preprocessed multiplication instance with specified configuration.
fn create_preprocessed_mul_instance(
    log_height: usize,
    multiplier: u64,
) -> (DemoAirNoLookup, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let air = DemoAirNoLookup::new(DemoAir::PreprocessedMul(PreprocessedMulAir {
        log_height,
        multiplier,
    }));
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
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
            lookups: vec![],
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_batch_no_lookups(&config, &airs, &proof, &pvs, &common)
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
            trace: fib16_trace,
            public_values: fib16_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_mul8,
            trace: mul8_trace,
            public_values: mul8_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_fib8,
            trace: fib8_trace,
            public_values: fib8_pis.clone(),
            lookups: vec![],
        },
    ];

    let common: CommonData<MyConfig> = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);
    let airs = vec![air_fib16, air_mul8, air_fib8];
    let pvs = vec![fib16_pis, mul8_pis, fib8_pis];
    verify_batch_no_lookups(&config, &airs, &proof, &pvs, &common)
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
        lookups: vec![],
    }];
    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    // Wrong public value at verify => should reject
    let airs = vec![air_fib];
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_batch_no_lookups(&config, &airs, &proof, &wrong_pvs, &common);
    assert!(res.is_err(), "Should reject wrong public values");
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
            trace: mul2_trace,
            public_values: mul2_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_mul3,
            trace: mul3_trace,
            public_values: mul3_pis.clone(),
            lookups: vec![],
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul2_pis, fib_pis, mul3_pis];
    verify_batch_no_lookups(&config, &airs, &proof, &pvs, &common)
}

#[test]
fn test_preprocessed_tampered_fails() -> Result<(), Box<dyn std::error::Error>> {
    let config = make_config(9999);

    // Single Fibonacci instance with 8 rows and preprocessed index column.
    let (air, trace, fib_pis) = create_fib_instance(3);
    let instances = vec![StarkInstance {
        air: &air,
        trace,
        public_values: fib_pis.clone(),
        lookups: vec![],
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    // First, sanity-check that verification succeeds with matching preprocessed data.
    let airs = vec![air];
    let ok_res = verify_batch_no_lookups(&config, &airs, &proof, from_ref(&fib_pis), &common);
    assert!(
        ok_res.is_ok(),
        "Expected verification to succeed with matching preprocessed data"
    );

    // Now tamper with the preprocessed trace by modifying the tamper_index in the AIR
    // used to derive the preprocessed commitment for verification.
    // The proof was generated with the original AIR, but we verify with a tampered AIR
    // that would produce different preprocessed columns.
    let air_tampered = DemoAirNoLookup::new(DemoAir::Fib(FibonacciAir {
        log_height: 3,
        tamper_index: Some(2),
    }));
    // Create CommonData with tampered AIR to test verification failure
    // Use the proof's degree_bits (which are log_degrees since ZK is not supported)
    let degree_bits = proof.degree_bits.clone();
    let mut airs_tampered = vec![air_tampered];
    let verify_common_tampered =
        CommonData::from_airs_and_degrees(&config, &mut airs_tampered, &degree_bits);

    let res = verify_batch_no_lookups(
        &config,
        &airs_tampered,
        &proof,
        &[fib_pis],
        &verify_common_tampered,
    );
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
    let air = DemoAirNoLookup::new(DemoAir::Fib(FibonacciAir {
        log_height,
        tamper_index: None,
    }));

    // First proof: standard Fibonacci trace starting from (0, 1).
    let trace1 = fib_trace::<Val>(0, 1, n);
    let fib_pis1 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    let instances1 = vec![StarkInstance {
        air: &air,
        trace: trace1,
        public_values: fib_pis1.clone(),
        lookups: vec![],
    }];
    let common = CommonData::from_instances(&config, &instances1);
    let proof1 = prove_batch_no_lookups(&config, &instances1, &common);

    // Verify the first proof.
    let airs = vec![air];
    let res1 = verify_batch_no_lookups(&config, &airs, &proof1, from_ref(&fib_pis1), &common);
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
        trace: trace2,
        public_values: fib_pis2.clone(),
        lookups: vec![],
    }];
    let proof2 = prove_batch_no_lookups(&config, &instances2, &common);

    let res2 = verify_batch_no_lookups(&config, &airs, &proof2, &[fib_pis2], &common);
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
        trace: fib_trace,
        public_values: fib_pis.clone(),
        lookups: vec![],
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);
    let airs = vec![air_fib];
    verify_batch_no_lookups(&config, &airs, &proof, &[fib_pis], &common)
}

#[test]
fn test_mixed_preprocessed() -> Result<(), impl Debug> {
    let config = make_config(8888);

    let (air_fib, fib_trace, fib_pis) = create_fib_instance(4); // 16 rows, has preprocessed
    let (air_mul, mul_trace, mul_pis) = create_mul_instance(4, 2); // 16 rows, no preprocessed

    let instances = vec![
        StarkInstance {
            air: &air_fib,
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
            lookups: vec![],
        },
    ];

    let common = CommonData::from_instances(&config, &instances);

    let proof = prove_batch_no_lookups(&config, &instances, &common);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];

    verify_batch_no_lookups(&config, &airs, &proof, &pvs, &common)
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
        lookups: vec![],
    }];

    // Generate a valid proof
    let common = CommonData::from_instances(&config, &instances);
    let valid_proof = prove_batch_no_lookups(&config, &instances, &common);

    // Tamper with the proof: change trace_local to have wrong width
    let mut tampered_proof = p3_batch_stark::proof::BatchProof {
        commitments: BatchCommitments {
            main: valid_proof.commitments.main,
            quotient_chunks: valid_proof.commitments.quotient_chunks,
            permutation: valid_proof.commitments.permutation,
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
    let res = verify_batch_no_lookups(&config, &airs, &tampered_proof, from_ref(&fib_pis), &common);
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
        .trace_next = vec![
        valid_proof.opened_values.instances[0]
            .base_opened_values
            .trace_next[0],
    ]; // Wrong width

    let res = verify_batch_no_lookups(&config, &airs, &tampered_proof, from_ref(&fib_pis), &common);
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
            trace: tr_a,
            public_values: pv_a.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_b,
            trace: tr_b,
            public_values: pv_b.clone(),
            lookups: vec![],
        },
    ];

    // DemoAir::Fib has preprocessed columns, so compute degrees for swapped verification
    let degrees: Vec<usize> = instances.iter().map(|i| i.trace.height()).collect();
    let log_degrees: Vec<usize> = degrees.iter().copied().map(log2_strict_usize).collect();

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    // Swap order at verify -> should fail (create new CommonData with swapped AIRs)
    let mut airs_swapped = vec![air_b, air_a];
    let common_swapped =
        CommonData::from_airs_and_degrees(&config, &mut airs_swapped, &log_degrees);
    let res = verify_batch_no_lookups(
        &config,
        &airs_swapped,
        &proof,
        &[pv_b, pv_a],
        &common_swapped,
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
        trace: tr,
        public_values: pv.clone(),
        lookups: vec![],
    }];
    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    let mut tampered = proof;
    tampered.opened_values.instances[0]
        .base_opened_values
        .quotient_chunks[0]
        .pop();

    let airs = vec![air];
    let res = verify_batch_no_lookups(&config, &airs, &tampered, from_ref(&pv), &common);
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
        commit_proof_of_work_bits: 8,
        query_proof_of_work_bits: 8,
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

    // Create two Fibonacci instances with different sizes.
    // Here we don't use preprocessed columns (Circle PCS + plain FibonacciAir).
    let air_fib1 = AirNoLookup::new(FibonacciAir {
        log_height: 0,
        tamper_index: None,
    });
    let air_fib2 = AirNoLookup::new(FibonacciAir {
        log_height: 0,
        tamper_index: None,
    });

    let fib_pis1 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(8))]; // F_8 = 21
    let fib_pis2 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(4))]; // F_4 = 3

    let trace1 = fib_trace::<Val>(0, 1, 8);
    let trace2 = fib_trace::<Val>(0, 1, 4);

    let airs = vec![air_fib1, air_fib2];

    let instances = vec![
        StarkInstance {
            air: &airs[0],
            trace: trace1,
            public_values: fib_pis1.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &airs[1],
            trace: trace2,
            public_values: fib_pis2.clone(),
            lookups: vec![],
        },
    ];

    // Generate batch-proof
    // Plain FibonacciAir doesn't have preprocessed columns
    let common = CommonData::empty(airs.len());
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    // Verify batch-proof
    let public_values = vec![fib_pis1, fib_pis2];
    verify_batch_no_lookups(&config, &airs, &proof, &public_values, &common)
        .map_err(|e| format!("Verification failed: {:?}", e))
}

#[test]
fn test_preprocessed_constraint_positive() -> Result<(), impl Debug> {
    // Test that preprocessed columns are correctly used in constraints
    // Enforces: main[0] = 2 * preprocessed[0]
    let config = make_config(8888);

    let (air, trace, pis) = create_preprocessed_mul_instance(4, 2); // 16 rows, multiplier=2

    let instances = vec![StarkInstance {
        air: &air,
        trace,
        public_values: pis.clone(),
        lookups: vec![],
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);
    let airs = vec![air];
    verify_batch_no_lookups(&config, &airs, &proof, &[pis], &common)
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
        trace,
        public_values: pis.clone(),
        lookups: vec![],
    }];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    // Verify with wrong multiplier=3 (should fail)
    let air_verify = DemoAir::PreprocessedMul(PreprocessedMulAir {
        log_height: 4,
        multiplier: 3, // Wrong multiplier!
    });
    let mut airs = vec![AirNoLookup::new(air_verify)];
    let degree_bits = proof.degree_bits.clone();
    let verify_common = CommonData::from_airs_and_degrees(&config, &mut airs, &degree_bits);

    let res = verify_batch_no_lookups(&config, &airs, &proof, &[pis], &verify_common);
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
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_mul,
            trace: mul_trace,
            public_values: mul_pis.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_pp_mul,
            trace: pp_mul_trace,
            public_values: pp_mul_pis.clone(),
            lookups: vec![],
        },
    ];

    let common = CommonData::from_instances(&config, &instances);
    let proof = prove_batch_no_lookups(&config, &instances, &common);

    let airs = vec![air_fib, air_mul, air_pp_mul];
    let pvs = vec![fib_pis, mul_pis, pp_mul_pis];
    verify_batch_no_lookups(&config, &airs, &proof, &pvs, &common)
}

// Tests for local and global lookup handling in multi-stark.

/// Test with local lookups only using MulAirLookups
#[test]
fn test_batch_stark_one_instance_local_only() -> Result<(), impl Debug> {
    let config = make_config(2024);

    let reps = 1;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, 0, vec![]); // local only

    let log_height = 3; // 8 rows
    let mul_trace = mul_trace::<Val>(1 << log_height, reps);

    let mut airs = [DemoAirWithLookups::MulLookups(mul_air_lookups)];

    // Get lookups from the lookup-enabled AIRs
    let common_data =
        CommonData::<MyConfig>::from_airs_and_degrees(&config, &mut airs, &[log_height]);

    let instances = StarkInstance::new_multiple(&airs, &[mul_trace], &[vec![]], &common_data);

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    let pvs = vec![vec![]];
    verify_batch(&config, &airs, &proof, &pvs, &common_data, &lookup_gadget)
}

/// Test with local lookups only, which fail due to wrong permutation column.
/// The failure occurs in `check_constraints` during proof generation, since it fails the last local constraint (the final local sum is not zero).
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "constraints had nonzero value on row 7")]
fn test_batch_stark_one_instance_local_fails() {
    let config = make_config(2024);

    let reps = 2;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, 0, vec![]); // local only

    let log_height = 3; // 8 rows
    let mut mul_trace = mul_trace::<Val>(1 << log_height, reps);

    // Tamper with the permutation column to cause lookup failure.
    mul_trace.values[reps * 3] = Val::from_u64(9999);

    let mut airs = [DemoAirWithLookups::MulLookups(mul_air_lookups)];

    // Get lookups from the lookup-enabled AIRs
    let common_data =
        CommonData::<MyConfig>::from_airs_and_degrees(&config, &mut airs, &[log_height]);

    let instances = StarkInstance::new_multiple(&airs, &[mul_trace], &[vec![]], &common_data);

    let lookup_gadget = LogUpGadget::new();
    prove_batch(&config, &instances, &common_data, &lookup_gadget);
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
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, 0, vec![]); // local only

    let mut mul_trace = mul_trace::<Val>(1 << log_height, reps);

    // Tamper with the permutation column to cause lookup failure.
    mul_trace.values[reps * 3] = Val::from_u64(9999);

    let mut airs = [DemoAirWithLookups::MulLookups(mul_air_lookups)];

    // Get lookups from the lookup-enabled AIRs
    let common_data =
        CommonData::<MyConfig>::from_airs_and_degrees(&config, &mut airs, &[log_height]);

    let instances = StarkInstance::new_multiple(&airs, &[mul_trace], &[vec![]], &common_data);

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    verify_batch(
        &config,
        &airs,
        &proof,
        &[vec![]],
        &common_data,
        &lookup_gadget,
    )
    .unwrap();
}

/// Test with local lookups only using MulAirLookups
#[test]
fn test_batch_stark_local_lookups_only() -> Result<(), impl Debug> {
    let config = make_config(2024);

    let log_height = 4; // 16 rows
    let height = 1 << log_height;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps: 2 };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, 0, vec![]); // local only
    let fib_air_lookups = FibAirLookups::new(
        FibonacciAir {
            log_height,
            tamper_index: None,
        },
        false,
        0,
        None,
    ); // no lookups

    let mul_trace = mul_trace::<Val>(height, 2);
    let fib_trace = fib_trace::<Val>(0, 1, 16);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let mut airs = [air1, air2];

    // Get lookups from the lookup-enabled AIRs
    let common_data = CommonData::<MyConfig>::from_airs_and_degrees(
        &config,
        &mut airs,
        &[log_height, log_height],
    );

    let instances = StarkInstance::new_multiple(
        &airs,
        &[mul_trace, fib_trace],
        &[vec![], fib_pis.clone()],
        &common_data,
    );

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common_data, &lookup_gadget)
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
        0,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    ); // global only

    let log_n = 3;
    let n = 1 << log_n;

    let fibonacci_air = FibonacciAir {
        log_height: log_n,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, 0, None); // global lookups

    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace = fib_trace::<Val>(0, 1, n);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    // Get lookups from the lookup-enabled AIRs
    let mut airs = [air1, air2];
    let common_data =
        CommonData::<MyConfig>::from_airs_and_degrees(&config, &mut airs, &[log_n, log_n]);

    let instances = StarkInstance::new_multiple(
        &airs,
        &[mul_trace, fib_trace],
        &[vec![], fib_pis.clone()],
        &common_data,
    );

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common_data, &lookup_gadget)
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
        0,
        vec!["MulFib".to_string(), "MulFib".to_string()],
    ); // both

    let log_height = 4;
    let height = 1 << log_height;

    let fibonacci_air = FibonacciAir {
        log_height,
        tamper_index: None,
    };
    let fib_air_lookups = FibAirLookups::new(fibonacci_air, true, 0, None); // global lookups

    let mul_trace = mul_trace::<Val>(height, 2);
    let fib_trace = fib_trace::<Val>(0, 1, height);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let mut airs = [air1, air2];
    // Get lookups from the lookup-enabled AIRs
    let common_data = CommonData::<MyConfig>::from_airs_and_degrees(
        &config,
        &mut airs,
        &[log_height, log_height],
    );

    let instances = StarkInstance::new_multiple(
        &airs,
        &[mul_trace, fib_trace],
        &[vec![], fib_pis.clone()],
        &common_data,
    );

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    let pvs = vec![vec![], fib_pis];
    verify_batch(&config, &airs, &proof, &pvs, &common_data, &lookup_gadget)
}

#[test]
#[should_panic(expected = "LookupError(GlobalCumulativeMismatch(Some(\"MulFib2\"))")]
fn test_batch_stark_failed_global_lookup() {
    let config = make_config(2025);

    let reps = 2;
    // Create instances with global lookups configuration
    let mul_air = MulAir { reps };
    // MulAir uses two different names for its reps, which will create two separate global lookups
    let mul_air_lookups = MulAirLookups::new(
        mul_air,
        false,
        true,
        0,
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
    let fib_air_lookups =
        FibAirLookups::new(fibonacci_air, true, 0, Some(("MulFib1".to_string(), 1)));

    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace = fib_trace::<Val>(0, 1, n);

    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];
    let traces = vec![mul_trace, fib_trace];
    let pvs = vec![vec![], fib_pis];
    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    // Get lookups from the lookup-enabled AIRs
    let mut airs = [air1, air2];
    let common_data =
        CommonData::<MyConfig>::from_airs_and_degrees(&config, &mut airs, &[log_n, log_n]);

    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs, &common_data);

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    // This should panic with GlobalCumulativeMismatch because:
    // - MulAir sends values to "MulFib1" and "MulFib2" lookups
    // - FibAir only receives from "MulFib" lookup
    // - The global cumulative sums won't match
    verify_batch(&config, &airs, &proof, &pvs, &common_data, &lookup_gadget).unwrap();
}

/// Test mixing instances with lookups and instances without lookups.
/// We have the following instances:
/// - MulAir with both local and global lookups (looking into two different FibAir instances for each rep)
/// - FibAir without lookups
/// - FibAir with global lookups (sends values for first rep of MulAir)
/// - MulAir without lookups
/// - FibAir with global lookups (sends values for second rep of MulAir)
/// - MulAir with local lookups only
#[test]
fn test_batch_stark_mixed_lookups() -> Result<(), impl Debug> {
    let config = make_config(2027);

    let reps = 2;

    // Create instances with different lookup configurations:
    let mul_air_with_lookups = MulAir { reps };
    // This AIR has two different global lookups (one for each rep) with two different names. It also has two local lookups (one for each rep).
    let mul_air_lookups = MulAirLookups::new(
        mul_air_with_lookups,
        true,
        true,
        0,
        vec!["MulFib1".to_string(), "MulFib2".to_string()],
    );
    // This AIR has no lookups.
    let mul_air_no_lookups = MulAirLookups::new(mul_air_with_lookups, false, false, 0, vec![]);
    // This AIR only has local lookups.
    let mul_air_local_lookups = MulAirLookups::new(mul_air_with_lookups, true, false, 0, vec![]); // local lookups only

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
    let fib_air_with_lookups1 =
        FibAirLookups::new(fib_air_lookups, true, 0, Some(("MulFib1".to_string(), 1))); // global lookups
    let fib_air_with_lookups2 =
        FibAirLookups::new(fib_air_lookups, true, 0, Some(("MulFib2".to_string(), 1))); // global lookups
    let fib_air_no_lookups = FibAirLookups::new(fib_air_no_lookups, false, 0, None); // global lookups

    // Generate traces. The airs with and without lookups have different heights.
    let mul_with_lookups_trace = mul_trace::<Val>(n1, reps);
    let fib_with_lookups_trace = fib_trace::<Val>(0, 1, n1);
    let mul_no_lookups_trace = mul_trace::<Val>(n2, reps);
    let fib_no_lookups_trace = fib_trace::<Val>(0, 1, n2);

    // Public values
    let fib_with_lookups_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];
    let fib_no_lookups_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(8))];

    // Create lookup-enabled AIRs
    let air_mul_with_lookups = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air_fib_with_lookups1 = DemoAirWithLookups::FibLookups(fib_air_with_lookups1);
    let air_fib_with_lookups2 = DemoAirWithLookups::FibLookups(fib_air_with_lookups2);
    let air_mul_with_local_lookups = DemoAirWithLookups::MulLookups(mul_air_local_lookups);

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
    let common_data = CommonData::<MyConfig>::from_airs_and_degrees(
        &config,
        &mut all_airs,
        &[log_n1, log_n2, log_n1, log_n2, log_n1, log_n1],
    );

    let traces = vec![
        mul_with_lookups_trace.clone(),
        fib_no_lookups_trace,
        fib_with_lookups_trace.clone(),
        mul_no_lookups_trace,
        fib_with_lookups_trace,
        mul_with_lookups_trace,
    ];

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
    let instances = StarkInstance::new_multiple(&all_airs, &traces, &all_pvs, &common_data);

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    // Verify with mixed AIRs
    verify_batch(
        &config,
        &all_airs,
        &proof,
        &all_pvs,
        &common_data,
        &lookup_gadget,
    )
}

// Single table with local lookup involving the Lagrange selectors. Since the selectors are not normalized,
// we need to add multiplicity columns and multiply them by the selectors.
#[derive(Debug, Clone, Copy)]
struct SingleTableLocalLookupAir {
    num_lookups: usize,
}

impl SingleTableLocalLookupAir {
    const fn new() -> Self {
        Self { num_lookups: 0 }
    }
}

impl<F> BaseAir<F> for SingleTableLocalLookupAir {
    fn width(&self) -> usize {
        7 // 7 columns: 3 sender columns (1 for each selector type), lookup table, 3 multiplicty columns (1 for each selector type)
    }
}

impl<AB: AirBuilder> Air<AB> for SingleTableLocalLookupAir
where
    AB::Var: Debug,
{
    fn eval(&self, _builder: &mut AB) {
        // No additional constraints needed for this simple table
    }
}

impl<AB> AirLookupHandler<AB> for SingleTableLocalLookupAir
where
    AB: PermutationAirBuilder + PairBuilder + AirBuilderWithPublicValues,
    AB::Var: Debug,
    AB::F: From<Val>,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let new_idx = self.num_lookups;
        self.num_lookups += 1;
        vec![new_idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        let mut lookups = Vec::new();
        self.num_lookups = 0;

        // Create symbolic air builder to access symbolic variables
        let symbolic_air_builder =
            SymbolicAirBuilder::<AB::F>::new(0, BaseAir::<AB::F>::width(self), 0, 0, 0);
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

        let sender_col1 = symbolic_main_local[0]; // Column that sends values
        let sender_col2 = symbolic_main_local[1]; // Column that sends values
        let sender_col3 = symbolic_main_local[2]; // Column that sends values
        let lookup_table_col = symbolic_main_local[3]; // Column that receives lookups
        let mul1 = symbolic_main_local[4]; // Multiplicity column for first selector
        let mul2 = symbolic_main_local[5]; // Multiplicity column for second selector
        let mul3 = symbolic_main_local[6]; // Multiplicity column for third selector

        // Local lookup: sender column looks up into lookup table column
        // Sender: send is_transition * sender_col
        // Receiver: receive lookup_table_col with multiplicity 1
        let lookup_inputs1 = vec![
            // Sender: send values from sender column with `is_first_row` multiplicity
            (
                vec![sender_col1.into()],
                symbolic_air_builder.is_first_row(),
                Direction::Receive,
            ),
            // Receiver: receive values in lookup table column with multiplicity 1 * `is_first_row` multiplicity.
            // Note that we need to multiply by `is_first_row` here because the Lagrange selectors are not normalized.
            (
                vec![lookup_table_col.into()],
                symbolic_air_builder.is_first_row() * mul1,
                Direction::Send,
            ),
        ];

        let lookup_inputs2 = vec![
            // Sender: send values from sender column with `is_last_row` multiplicity
            (
                vec![sender_col2.into()],
                symbolic_air_builder.is_transition(),
                Direction::Receive,
            ),
            // Receiver: receive values in lookup table column with multiplicity 1 * `is_transition` multiplicity.
            // Note that we need to multiply by `is_transition` here because the Lagrange selectors are not normalized.
            (
                vec![lookup_table_col.into()],
                symbolic_air_builder.is_transition() * mul2,
                Direction::Send,
            ),
        ];

        let lookup_inputs3 = vec![
            // Sender: send values from sender column with `is_transition` multiplicity
            (
                vec![sender_col3.into()],
                symbolic_air_builder.is_last_row(),
                Direction::Receive,
            ),
            // Receiver: receive values in lookup table column with multiplicity 1 * `is_last_row` multiplicity.
            // Note that we need to multiply by `is_last_row` here because the Lagrange selectors are not normalized.
            (
                vec![lookup_table_col.into()],
                symbolic_air_builder.is_last_row() * mul3,
                Direction::Send,
            ),
        ];

        let all_lookup_inputs = vec![lookup_inputs1, lookup_inputs2, lookup_inputs3];

        for lookup_inputs in all_lookup_inputs {
            let local_lookup =
                AirLookupHandler::<AB>::register_lookup(self, Kind::Local, &lookup_inputs);
            lookups.push(local_lookup);
        }

        lookups
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
    // Column 5: (mult1): 1 everywhere except last row, which is 0
    // Column 6: (mult2): 1 at last row, 0 elsewhere
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
/// The goal of this test is to check that the use of (non-normalized) Lagrange selectors does not cause isssues.
#[test]
fn test_single_table_local_lookup() -> Result<(), impl Debug> {
    let config = make_config(2029);

    let log_height = 3;
    let height = 1 << log_height; // Single table with 8 rows

    // Create instance
    let air = SingleTableLocalLookupAir::new();

    let mut airs = [air];

    // Get lookups from the lookup-enabled AIR
    let common_data =
        CommonData::<MyConfig>::from_airs_and_degrees(&config, &mut airs, &[log_height]);

    // Generate trace
    let trace = single_table_local_lookup_trace::<Val>(height);

    let traces = vec![trace];
    let pvs = vec![vec![]]; // No public values

    let instances = StarkInstance::new_multiple(&airs, &traces, &pvs, &common_data);

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_batch(&config, &instances, &common_data, &lookup_gadget);

    verify_batch(&config, &airs, &proof, &pvs, &common_data, &lookup_gadget)
}
