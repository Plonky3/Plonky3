use core::borrow::Borrow;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder, PermutationAirBuilder,
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
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
use p3_multi_stark::StarkInstance;
use p3_multi_stark::common::common_data;
use p3_multi_stark::proof::OpenedValuesWithLookups;
use p3_multi_stark::prover::{prove_multi, prove_multi_no_lookups};
use p3_multi_stark::verifier::{verify_multi, verify_multi_no_lookups};
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::StarkConfig;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};
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

impl<AB: PermutationAirBuilder + AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
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
impl<AB: AirBuilder> Air<AB> for MulAir
where
    AB::Var: Debug,
{
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

#[derive(Clone, Copy)]
struct MulAirLookups {
    air: MulAir,
    is_local: bool,
    is_global: bool,
    num_lookups: usize,
}

impl Default for MulAirLookups {
    fn default() -> Self {
        Self {
            air: MulAir::default(),
            is_local: false,
            is_global: false,
            num_lookups: 0,
        }
    }
}

impl MulAirLookups {
    fn new(air: MulAir, is_local: bool, is_global: bool, num_lookups: usize) -> Self {
        Self {
            air,
            is_local,
            is_global,
            num_lookups,
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
        self.air.eval(builder)
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

        // Create symbolic air builder to access symbolic variables
        let symbolic_air_builder =
            SymbolicAirBuilder::<AB::F>::new(0, <Self as BaseAir<AB::F>>::width(self), 0, 0, 0);
        let symbolic_main = symbolic_air_builder.main();
        let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

        // Local lookups: one for each mul input with extra column for integers 0 to height
        if self.is_local {
            let last_idx = symbolic_air_builder.main().width() - 1;
            let lut = symbolic_main_local[last_idx]; //  Extra column that corresponds to a permutation of 'a'
            for rep in 0..self.air.reps {
                let base_idx = rep * 3;
                let a = symbolic_main_local[base_idx]; // First input
                // Create lookup inputs for each multiplication input
                // We'll create a local lookup table with integers 0 to height
                let lookup_inputs = vec![
                    // Lookup for 'a' against a range table (0 to height)
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

                let local_lookup = <Self as AirLookupHandler<AB>>::register_lookup(
                    self,
                    Kind::Local,
                    &lookup_inputs,
                );
                lookups.push(local_lookup);
            }
        }

        // Global lookups: between MulAir inputs and FibAir inputs
        if self.is_global {
            for rep in 0..self.air.reps {
                let base_idx = rep * 3;
                let a = symbolic_main_local[base_idx]; // First input
                let b = symbolic_main_local[base_idx + 1]; // Second input

                // Global lookup between MulAir inputs and FibAir inputs
                let lookup_inputs = vec![(
                    vec![a.into(), b.into()],
                    SymbolicExpression::Constant(AB::F::ONE),
                    Direction::Send, // MulAir sends data to the global lookup
                )];

                let global_lookup = <Self as AirLookupHandler<AB>>::register_lookup(
                    self,
                    Kind::Global("MulFib".to_string()),
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

    // Keep a simple constant b and c = a*b
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

#[derive(Debug, Clone, Copy)]
struct FibAirLookups {
    air: FibonacciAir,
    is_global: bool,
    num_lookups: usize,
    reps: usize,
}

impl Default for FibAirLookups {
    fn default() -> Self {
        Self {
            air: FibonacciAir,
            is_global: false,
            num_lookups: 0,
            reps: 0,
        }
    }
}

impl FibAirLookups {
    fn new(air: FibonacciAir, is_global: bool, num_lookups: usize, reps: Option<usize>) -> Self {
        Self {
            air,
            is_global,
            num_lookups,
            reps: reps.unwrap_or(0),
        }
    }
}

impl<F> BaseAir<F> for FibAirLookups {
    fn width(&self) -> usize {
        <FibonacciAir as BaseAir<F>>::width(&self.air)
    }
}

impl<AB: PermutationAirBuilder + AirBuilderWithPublicValues> Air<AB> for FibAirLookups {
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder)
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

        if self.is_global {
            // Create symbolic air builder to access symbolic variables
            let symbolic_air_builder =
                SymbolicAirBuilder::<AB::F>::new(0, <Self as BaseAir<AB::F>>::width(self), 3, 0, 0);
            let symbolic_main = symbolic_air_builder.main();
            let symbolic_main_local = symbolic_main.row_slice(0).unwrap();

            // Global lookups: between FibAir inputs and MulAir inputs
            // FibAir has 2 columns: left and right
            let left = symbolic_main_local[0]; // left column
            let right = symbolic_main_local[1]; // right column

            // Global lookup between FibAir inputs and MulAir inputs
            let lookup_inputs = vec![(
                vec![left.into(), right.into()],
                SymbolicExpression::Constant(AB::F::from_u64(self.reps as u64)),
                Direction::Receive, // FibAir receives data from the global lookup
            )];

            let global_lookup = <Self as AirLookupHandler<AB>>::register_lookup(
                self,
                Kind::Global("MulFib".to_string()),
                &lookup_inputs,
            );
            lookups.push(global_lookup);
        }

        lookups
    }
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
            Self::Fib(a) => <FibonacciAir as BaseAir<F>>::width(a),
            Self::Mul(a) => <MulAir as BaseAir<F>>::width(a),
        }
    }
}

// Heterogeneous enum wrapper for lookup-enabled AIRs
#[derive(Clone, Copy)]
enum DemoAirWithLookups {
    FibLookups(FibAirLookups),
    MulLookups(MulAirLookups),
}

impl<F> BaseAir<F> for DemoAirWithLookups {
    fn width(&self) -> usize {
        match self {
            DemoAirWithLookups::FibLookups(a) => <FibAirLookups as BaseAir<F>>::width(a),
            DemoAirWithLookups::MulLookups(a) => <MulAirLookups as BaseAir<F>>::width(a),
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
            DemoAirWithLookups::FibLookups(a) => <FibAirLookups as Air<AB>>::eval(a, builder),
            DemoAirWithLookups::MulLookups(a) => <MulAirLookups as Air<AB>>::eval(a, builder),
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
            DemoAirWithLookups::FibLookups(a) => {
                <FibAirLookups as AirLookupHandler<AB>>::add_lookup_columns(a)
            }
            DemoAirWithLookups::MulLookups(a) => {
                <MulAirLookups as AirLookupHandler<AB>>::add_lookup_columns(a)
            }
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>> {
        match self {
            DemoAirWithLookups::FibLookups(a) => {
                <FibAirLookups as AirLookupHandler<AB>>::get_lookups(a)
            }
            DemoAirWithLookups::MulLookups(a) => {
                <MulAirLookups as AirLookupHandler<AB>>::get_lookups(a)
            }
        }
    }
}

impl<AB: PermutationAirBuilder + AirBuilderWithPublicValues> Air<AB> for DemoAir
where
    AB::Var: Debug,
{
    fn eval(&self, b: &mut AB) {
        match self {
            Self::Fib(a) => a.eval(b),
            Self::Mul(a) => a.eval(b),
        }
    }
}

type DemoAirNoLookup = AirNoLookup<DemoAir>;

// Demo function to show MulAirLookups usage
#[allow(dead_code)]
fn demo_mul_air_lookups() {
    let mul_air = MulAir { reps: 2 };
    let mul_air_with_local_lookups = MulAirLookups::new(mul_air, true, false, 0);
    let mul_air_with_global_lookups = MulAirLookups::new(mul_air, false, true, 0);
    let mul_air_with_both_lookups = MulAirLookups::new(mul_air, true, true, 0);

    // These would be used in actual proving/verifying contexts
    let _ = <MulAirLookups as BaseAir<Val>>::width(&mul_air_with_local_lookups);
    let _ = <MulAirLookups as BaseAir<Val>>::width(&mul_air_with_global_lookups);
    let _ = <MulAirLookups as BaseAir<Val>>::width(&mul_air_with_both_lookups);
}

// --- Test Helper Functions ---

/// Creates a Fibonacci instance with specified log height.
fn create_fib_instance(log_height: usize) -> (DemoAirNoLookup, RowMajorMatrix<Val>, Vec<Val>) {
    let n = 1 << log_height;
    let air = DemoAirNoLookup::new(DemoAir::Fib(FibonacciAir));
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

    let proof = prove_multi_no_lookups(&config, instances);

    let airs = vec![air_fib, air_mul];
    let pvs = vec![fib_pis, mul_pis];
    verify_multi_no_lookups(&config, &airs, &proof, &pvs)
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

    let proof = prove_multi_no_lookups(&config, instances);
    let airs = vec![air_fib16, air_mul8, air_fib8];
    let pvs = vec![fib16_pis, mul8_pis, fib8_pis];
    verify_multi_no_lookups(&config, &airs, &proof, &pvs)
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
    let proof = prove_multi_no_lookups(&config, instances);

    // Wrong public value at verify => should reject
    let wrong_pvs = vec![vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::from_u64(correct_x + 1),
    ]];
    let res = verify_multi_no_lookups(&config, &[air_fib], &proof, &wrong_pvs);
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

    let proof = prove_multi_no_lookups(&config, instances);
    let airs = vec![air_mul2, air_fib, air_mul3];
    let pvs = vec![mul2_pis, fib_pis, mul3_pis];
    verify_multi_no_lookups(&config, &airs, &proof, &pvs)
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

    let proof = prove_multi_no_lookups(&config, instances);
    verify_multi_no_lookups(&config, &[air_fib], &proof, &[fib_pis])
}

#[test]
fn test_invalid_trace_width_rejected() {
    // This test verifies that the verifier rejects proofs with incorrect trace width.
    use p3_multi_stark::proof::{MultiCommitments, MultiOpenedValues};
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
    let valid_proof = prove_multi_no_lookups(&config, instances);

    // Tamper with the proof: change trace_local to have wrong width
    let mut tampered_proof = p3_multi_stark::proof::MultiProof {
        commitments: MultiCommitments {
            main: valid_proof.commitments.main,
            quotient_chunks: valid_proof.commitments.quotient_chunks,
            permutation: valid_proof.commitments.permutation.clone(),
        },
        opened_values: MultiOpenedValues {
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
    let res = verify_multi_no_lookups(
        &config,
        &[air_fib.clone()],
        &tampered_proof,
        from_ref(&fib_pis),
    );
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

    let res = verify_multi_no_lookups(&config, &[air_fib], &tampered_proof, from_ref(&fib_pis));
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

    let proof = prove_multi_no_lookups(&config, instances);

    // Swap order at verify -> should fail
    let res = verify_multi_no_lookups(
        &config,
        &[air_b.clone(), air_a.clone()],
        &proof,
        &[pv_b.clone(), pv_a.clone()],
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
    let proof = prove_multi_no_lookups(&config, instances);

    let mut tampered = proof;
    tampered.opened_values.instances[0]
        .base_opened_values
        .quotient_chunks[0]
        .pop();

    let res = verify_multi_no_lookups(&config, &[air], &tampered, from_ref(&pv));
    assert!(
        res.is_err(),
        "Verifier should reject truncated quotient chunk element"
    );
}

#[test]
fn test_circle_stark_multi() -> Result<(), impl Debug> {
    // Test multi-stark with Circle PCS (non-two-adic field)
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
        proof_of_work_bits: 8,
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

    // Create two Fibonacci instances with different sizes
    let air_fib1 = AirNoLookup::new(FibonacciAir);
    let air_fib2 = AirNoLookup::new(FibonacciAir);

    let fib_pis1 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(8))]; // F_8 = 21
    let fib_pis2 = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(4))]; // F_4 = 3

    let trace1 = fib_trace::<Val>(0, 1, 8);
    let trace2 = fib_trace::<Val>(0, 1, 4);

    let instances = vec![
        StarkInstance {
            air: &air_fib1,
            trace: trace1,
            public_values: fib_pis1.clone(),
            lookups: vec![],
        },
        StarkInstance {
            air: &air_fib2,
            trace: trace2,
            public_values: fib_pis2.clone(),
            lookups: vec![],
        },
    ];

    // Generate multi-proof
    let proof = prove_multi_no_lookups(&config, instances);

    // Verify multi-proof
    let airs = vec![air_fib1, air_fib2];
    let public_values = vec![fib_pis1, fib_pis2];
    verify_multi_no_lookups(&config, &airs, &proof, &public_values)
        .map_err(|e| format!("Verification failed: {:?}", e))
}

// Tests for local and global lookup handling in multi-stark.

/// Test with local lookups only using MulAirLookups
/// This test demonstrates how to use local lookups in multi-stark proofs
#[test]
fn test_multi_stark_one_instance_local_only() -> Result<(), impl Debug> {
    let config = make_config(2024);

    let reps = 1;
    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, 0); // local only

    let mul_trace = mul_trace::<Val>(8, reps);

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups).clone();

    let mut airs = [air1];

    // Get lookups from the lookup-enabled AIRs
    let all_airs_lookups = common_data::<MyConfig, _>(&mut airs).lookups;

    let mul_lookups = all_airs_lookups[0].clone();

    let instances = vec![StarkInstance {
        air: &air1,
        trace: mul_trace,
        public_values: vec![],
        lookups: mul_lookups,
    }];

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_multi(&config, instances, &lookup_gadget);

    let mut airs = vec![air1];
    let pvs = vec![vec![]];
    verify_multi(&config, &mut airs, &proof, &pvs, &lookup_gadget)
}

/// Test with local lookups only using MulAirLookups
/// This test demonstrates how to use local lookups in multi-stark proofs
#[test]
fn test_multi_stark_local_lookups_only() -> Result<(), impl Debug> {
    let config = make_config(2024);

    // Create MulAir instance with local lookups configuration
    let mul_air = MulAir { reps: 2 };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, false, 0); // local only
    let fib_air_lookups = FibAirLookups::new(FibonacciAir, false, 0, None); // no lookups

    let mul_trace = mul_trace::<Val>(16, 2);
    let fib_trace = fib_trace::<Val>(0, 1, 16);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups).clone();
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups).clone();

    let mut airs = [air1, air2];

    // Get lookups from the lookup-enabled AIRs
    let all_airs_lookups = common_data::<MyConfig, _>(&mut airs).lookups;

    let mul_lookups = all_airs_lookups[0].clone();
    let fib_lookups = all_airs_lookups[1].clone();

    let instances = vec![
        StarkInstance {
            air: &air1,
            trace: mul_trace,
            public_values: vec![],
            lookups: mul_lookups,
        },
        StarkInstance {
            air: &air2,
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: fib_lookups,
        },
    ];

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_multi(&config, instances, &lookup_gadget);

    let mut airs = vec![air1, air2];
    let pvs = vec![vec![], fib_pis];
    verify_multi(&config, &mut airs, &proof, &pvs, &lookup_gadget)
}

/// Test with global lookups only using MulAirLookups and FibAirLookups  
#[test]
fn test_multi_stark_global_lookups_only() -> Result<(), impl Debug> {
    let config = make_config(2025);

    let reps = 2;
    // Create instances with global lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, false, true, 0); // global only
    let fib_air_lookups = FibAirLookups::new(FibonacciAir, true, 0, Some(reps)); // global lookups

    let n = 8;
    let mul_trace = mul_trace::<Val>(n, 2);
    let fib_trace = fib_trace::<Val>(0, 1, n);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(n))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    // Get lookups from the lookup-enabled AIRs
    let mut airs = [air1, air2];
    let all_airs_lookups = common_data::<MyConfig, _>(&mut airs).lookups;

    let mul_lookups = all_airs_lookups[0].clone();
    let fib_lookups = all_airs_lookups[1].clone();

    let instances = vec![
        StarkInstance {
            air: &air1,
            trace: mul_trace,
            public_values: vec![],
            lookups: mul_lookups,
        },
        StarkInstance {
            air: &air2,
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: fib_lookups,
        },
    ];

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_multi(&config, instances, &lookup_gadget);

    let mut airs = vec![air1, air2];
    let pvs = vec![vec![], fib_pis];
    verify_multi(&config, &mut airs, &proof, &pvs, &lookup_gadget)
}

/// Test with both local and global lookups using MulAirLookups and FibAirLookups
#[test]
fn test_multi_stark_both_lookups() -> Result<(), impl Debug> {
    let config = make_config(2026);

    let reps = 2;
    // Create instances with both local and global lookups configuration
    let mul_air = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air, true, true, 0); // both
    let fib_air_lookups = FibAirLookups::new(FibonacciAir, true, 0, Some(reps)); // global lookups

    let mul_trace = mul_trace::<Val>(16, 2);
    let fib_trace = fib_trace::<Val>(0, 1, 16);
    let fib_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];

    // Use the enum wrapper for heterogeneous types
    let air1 = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air2 = DemoAirWithLookups::FibLookups(fib_air_lookups);

    let mut airs = [air1, air2];
    // Get lookups from the lookup-enabled AIRs
    let all_airs_lookups = common_data::<MyConfig, _>(&mut airs).lookups;

    let mul_lookups = all_airs_lookups[0].clone();
    let fib_lookups = all_airs_lookups[1].clone();

    let instances = vec![
        StarkInstance {
            air: &air1,
            trace: mul_trace,
            public_values: vec![],
            lookups: mul_lookups,
        },
        StarkInstance {
            air: &air2,
            trace: fib_trace,
            public_values: fib_pis.clone(),
            lookups: fib_lookups,
        },
    ];

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_multi(&config, instances, &lookup_gadget);

    let mut airs = vec![air1, air2];
    let pvs = vec![vec![], fib_pis];
    verify_multi(&config, &mut airs, &proof, &pvs, &lookup_gadget)
}

/// Test mixing instances with lookups and instances without lookups
/// This test demonstrates multi-stark with a heterogeneous mix of lookup-enabled and
/// non-lookup instances, showcasing real-world scenarios where some computations
/// require lookups while others don't.
#[test]
fn test_multi_stark_mixed_lookups() -> Result<(), impl Debug> {
    let config = make_config(2027);

    let reps = 2;

    // Create instances with different lookup configurations:
    // 1. MulAir with both local and global lookups
    let mul_air_with_lookups = MulAir { reps };
    let mul_air_lookups = MulAirLookups::new(mul_air_with_lookups, true, true, 0); // both lookups
    let mul_air_no_lookups = MulAirLookups::new(mul_air_with_lookups, false, false, 0); // no lookups
    let mul_air_local_lookups = MulAirLookups::new(mul_air_with_lookups, true, false, 0); // local lookups only

    // 2. FibAir with global lookups only
    let fib_air_with_lookups = FibAirLookups::new(FibonacciAir, true, 0, Some(reps)); // global lookups
    let fib_air_no_lookups = FibAirLookups::new(FibonacciAir, false, 0, Some(reps)); // global lookups

    // Generate traces
    let mul_with_lookups_trace = mul_trace::<Val>(16, reps);
    let fib_with_lookups_trace = fib_trace::<Val>(0, 1, 16);
    let mul_no_lookups_trace = mul_trace::<Val>(8, reps);
    let fib_no_lookups_trace = fib_trace::<Val>(0, 1, 8);

    // Public values
    let fib_with_lookups_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(16))];
    let fib_no_lookups_pis = vec![Val::from_u64(0), Val::from_u64(1), Val::from_u64(fib_n(8))];

    // Create lookup-enabled AIRs
    let air_mul_with_lookups = DemoAirWithLookups::MulLookups(mul_air_lookups);
    let air_fib_with_lookups = DemoAirWithLookups::FibLookups(fib_air_with_lookups);
    let air_mul_with_local_lookups = DemoAirWithLookups::MulLookups(mul_air_local_lookups);

    // Create non-lookup AIRs
    let air_mul_no_lookups = DemoAirWithLookups::MulLookups(mul_air_no_lookups);
    let air_fib_no_lookups = DemoAirWithLookups::FibLookups(fib_air_no_lookups);

    // Get lookups for lookup-enabled AIRs only
    let mut airs_with_lookups = [
        air_mul_with_lookups,
        air_fib_with_lookups,
        air_mul_with_local_lookups,
    ];
    let all_airs_lookups = common_data::<MyConfig, _>(&mut airs_with_lookups).lookups;

    let mul_lookups = all_airs_lookups[0].clone();
    let fib_lookups = all_airs_lookups[1].clone();
    let mul_local_lookups = all_airs_lookups[2].clone();

    let mut all_airs = vec![
        air_mul_with_lookups,
        air_fib_no_lookups,
        air_fib_with_lookups,
        air_mul_no_lookups,
        air_mul_with_local_lookups,
    ];

    let traces = vec![
        mul_with_lookups_trace.clone(),
        fib_no_lookups_trace,
        fib_with_lookups_trace,
        mul_no_lookups_trace,
        mul_with_lookups_trace,
    ];

    let all_pvs = vec![
        vec![],               // mul with lookups
        fib_no_lookups_pis,   // fib no lookups
        fib_with_lookups_pis, // fib with lookups
        vec![],               // mul no lookups
        vec![],               // mul with local lookups
    ];

    let lookups = vec![
        mul_lookups,
        vec![], // no lookups
        fib_lookups,
        vec![], // no lookups
        mul_local_lookups,
    ];

    // Create instances - mixing lookup and non-lookup instances

    let instances = StarkInstance::new_multiple(&all_airs, &traces, &all_pvs, &lookups);

    let lookup_gadget = LogUpGadget::new();
    let proof = prove_multi(&config, instances, &lookup_gadget);

    // Verify with mixed AIRs

    verify_multi(&config, &mut all_airs, &proof, &all_pvs, &lookup_gadget)
}
