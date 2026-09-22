//! The machine-facing contract end to end: declare, prove, seal, and refuse bad input.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, BoundaryEnd, BoundaryPublic, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_bus::{
    BusActivation, BusBoundary, BusDirection, BusInteractionBuilder, BusName, BusSymbolicBuilder,
};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_lookup::{Count, InteractionBuilder, InteractionSymbolicBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::{MultiStarkConfig, PcsError};
use p3_multi_stark::contract::{
    BODY_REVISION, ColumnCounts, DeclarationError, ENVELOPE_VERSION, EnvelopeError, HEADER_LEN,
    HeightRange, LocalConstraints, MachineDeclaration, SealedVerificationError, TableDeclaration,
};
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, SecurityError, VerificationError,
    VerifierInstance, VerifierInstances, prove, setup,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_ceil_usize;
use p3_whir::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver, WhirProverData,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type L = PrefixProver<F, EF>;
type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

/// First-round folding factor; also the per-table padding floor.
const FOLDING: usize = 2;

/// Columns of the table proved below.
const NUM_COLS: usize = 2;

/// Base-two logarithm of the height every proof here runs at.
const LOG_HEIGHT: usize = 8;

/// Budget the fixture was measured against, in bytes.
const PROOF_BUDGET: usize = 1 << 16;

/// Security level every verification below must reach.
const SECURITY_TARGET: usize = 20;

/// Collision resistance the primitives of this configuration supply.
const COLLISION_BITS: usize = 100;

const FIXTURE: &str = "tests/fixtures/backend_contract_v2.envelope";

/// Body revision the fixture on disk was written under.
const FIXTURE_REVISION: u16 = 2;

/// The fixture the revision before this one, kept so its refusal stays covered.
const RETIRED_FIXTURE: &str = "tests/fixtures/backend_contract_v1.envelope";

/// Body revision that retired fixture was written under.
const RETIRED_REVISION: u16 = 1;

/// Digest of the retired fixture, pinned for the same reason as the current one.
const RETIRED_DIGEST: [u8; 32] = [
    221, 124, 62, 217, 215, 200, 177, 190, 217, 133, 169, 60, 188, 91, 111, 34, 191, 182, 104, 149,
    158, 140, 125, 78, 86, 192, 161, 83, 7, 24, 161, 71,
];

/// Digest of the fixture bytes, pinned so a silent regeneration cannot pass.
const FIXTURE_DIGEST: [u8; 32] = [
    67, 120, 119, 33, 178, 5, 117, 117, 239, 112, 28, 25, 240, 230, 83, 11, 241, 141, 72, 44, 47,
    112, 118, 245, 54, 155, 7, 227, 20, 32, 152, 65,
];

/// A commitment scheme that binds the trace without hiding it.
struct BindingConfig {
    pcs: TestPcs,
}

impl MultiStarkConfig for BindingConfig {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(COLLISION_BITS)
    }

    fn min_num_variables(&self) -> usize {
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        L::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a WhirProverData<F, EF, MyMmcs, L>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

fn challenger() -> MyChallenger {
    MyChallenger::new(perm())
}

fn round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let schedule = folding_factor
        .compute_folding_schedule(num_variables)
        .expect("valid folding schedule");
    let mut rates = Vec::with_capacity(schedule.len() - 1);
    let mut rate = 1;
    for &folding in schedule.iter().take(schedule.len() - 1) {
        rate += folding - 1;
        rates.push(rate);
    }
    rates
}

fn config() -> BindingConfig {
    let stacked = LOG_HEIGHT + log2_ceil_usize(NUM_COLS);
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let mmcs = MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        round_log_inv_rates: round_log_inv_rates(stacked, &folding_factor),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    let whir = WhirConfig::new(stacked, params).unwrap();
    BindingConfig {
        pcs: TestPcs::new(whir, MyDft::default(), mmcs),
    }
}

struct FibAir;

struct FibRow<T> {
    left: T,
    right: T,
}

impl<T> Borrow<FibRow<T>> for [T] {
    fn borrow(&self) -> &FibRow<T> {
        // Safety: two fields of type T in declaration order match the layout of [T; 2].
        debug_assert_eq!(self.len(), NUM_COLS);
        let ptr = self.as_ptr() as *const FibRow<T>;
        unsafe { &*ptr }
    }
}

impl<X> BaseAir<X> for FibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for FibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);

        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        let mut first = builder.when_first_row();
        first.assert_eq(local.left, a);
        first.assert_eq(local.right, b);

        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

/// A table of three columns, so its declaration cannot pass for the one above.
struct WideAir;

impl<X> BaseAir<X> for WideAir {
    fn width(&self) -> usize {
        NUM_COLS + 1
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for WideAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let current = main.current(0).expect("the table has three columns");
        let next = main.next(0).expect("the table has three columns");
        builder.when_transition().assert_eq(current, next);
    }
}

/// The recurrence above with one sign flipped, and every count left alone.
struct SubtractingFibAir;

impl<X> BaseAir<X> for SubtractingFibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for SubtractingFibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (a, b, x) = (pis[0], pis[1], pis[2]);

        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        let mut first = builder.when_first_row();
        first.assert_eq(local.left, a);
        first.assert_eq(local.right, b);

        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.right - local.left, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

/// The recurrence above with its two first-row assertions dropped.
struct UnpinnedFibAir;

impl<X> BaseAir<X> for UnpinnedFibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for UnpinnedFibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let x = builder.public_values()[2];

        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let next: &FibRow<AB::Var> = main.next_slice().borrow();

        let mut trans = builder.when_transition();
        trans.assert_eq(local.right, next.left);
        trans.assert_eq(local.left + local.right, next.right);

        builder.when_last_row().assert_eq(local.right, x);
    }
}

/// The same table, with those two cells pinned by the backend instead.
struct PinnedFibAir;

const PINS: [BoundaryPublic; 2] = [
    BoundaryPublic::new(0, BoundaryEnd::First, 0),
    BoundaryPublic::new(1, BoundaryEnd::First, 1),
];

impl<X> BaseAir<X> for PinnedFibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &PINS
    }
}

impl<AB: AirBuilder> Air<AB> for PinnedFibAir {
    fn eval(&self, builder: &mut AB) {
        UnpinnedFibAir.eval(builder);
    }
}

/// The same two cells again, wired to the other public value each.
struct SwappedPinFibAir;

const SWAPPED_PINS: [BoundaryPublic; 2] = [
    BoundaryPublic::new(0, BoundaryEnd::First, 1),
    BoundaryPublic::new(1, BoundaryEnd::First, 0),
];

impl<X> BaseAir<X> for SwappedPinFibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &SWAPPED_PINS
    }
}

impl<AB: AirBuilder> Air<AB> for SwappedPinFibAir {
    fn eval(&self, builder: &mut AB) {
        UnpinnedFibAir.eval(builder);
    }
}

/// A table that moves one tuple across a bus, with every part of the route a parameter.
struct BusAir {
    channel: &'static str,
    direction: BusDirection,
    cubed: bool,
}

impl<X> BaseAir<X> for BusAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder + BusInteractionBuilder> Air<AB> for BusAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let value: AB::Expr = main.current(0).expect("two columns").into();
        let selector: AB::Expr = main.current(1).expect("two columns").into();
        let square = value.clone() * value.clone();
        let payload = if self.cubed { square * value } else { square };
        builder.push_bus_interaction(
            BusName::new(self.channel),
            self.direction,
            [payload],
            BusActivation::Boolean(selector),
        );
    }
}

/// A table whose only variable is the end it flushes at.
struct BoundaryAir(BusBoundary);

impl<X> BaseAir<X> for BoundaryAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB: AirBuilder + BusInteractionBuilder> Air<AB> for BoundaryAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let value: AB::Expr = main.current(0).expect("two columns").into();
        builder.push_bus_interaction(
            BusName::new("edge"),
            BusDirection::Push,
            [value],
            BusActivation::Boundary(self.0),
        );
    }
}

/// A table whose periodic values are a parameter, so two of them differ in nothing else.
struct PeriodicAir([F; 2]);

impl BaseAir<F> for PeriodicAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
    fn num_periodic_columns(&self) -> usize {
        1
    }
    fn periodic_columns(&self) -> std::borrow::Cow<'_, [Vec<F>]> {
        std::borrow::Cow::Owned(vec![self.0.to_vec()])
    }
}

impl<AB: AirBuilder<F = F>> Air<AB> for PeriodicAir {
    fn eval(&self, builder: &mut AB) {
        UnpinnedFibAir.eval(builder);
    }
}

/// The recurrence above, plus a lookup that really does move tuples.
struct LookupFibAir;

impl<X> BaseAir<X> for LookupFibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for LookupFibAir {
    fn eval(&self, builder: &mut AB) {
        FibAir.eval(builder);
        let main = builder.main();
        let local: &FibRow<AB::Var> = main.current_slice().borrow();
        let (left, right) = (local.left, local.right);
        builder.push_local_interaction([
            (vec![left.into()], Count::bounded(AB::Expr::ONE, 1)),
            (vec![right.into()], Count::provided(-AB::Expr::ONE)),
        ]);
    }
}

/// The recurrence above, plus a lookup with no tuple at all.
struct InertLookupFibAir;

impl<X> BaseAir<X> for InertLookupFibAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for InertLookupFibAir {
    fn eval(&self, builder: &mut AB) {
        FibAir.eval(builder);
        builder.push_local_interaction(core::iter::empty::<(Vec<AB::Expr>, Count<AB::Expr>)>());
    }
}

fn trace(n: usize) -> RowMajorMatrix<F> {
    let (mut left, mut right) = (F::ZERO, F::ONE);
    let mut values = Vec::with_capacity(NUM_COLS * n);
    for _ in 0..n {
        values.push(left);
        values.push(right);
        let next_left = right;
        right += left;
        left = next_left;
    }
    RowMajorMatrix::new(values, NUM_COLS)
}

fn public_values(n: usize) -> [F; 3] {
    let trace = trace(n);
    let last = trace.values[(n - 1) * NUM_COLS + 1];
    [trace.values[0], trace.values[1], last]
}

/// The declaration the machine publishes, derived from the constraint system itself.
fn declaration() -> MachineDeclaration<Keccak256Hash> {
    declaration_for(&FibAir)
}

/// The declaration one constraint system produces, at the heights every test here uses.
fn declaration_for<A>(air: &A) -> MachineDeclaration<Keccak256Hash>
where
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>> + Air<BusSymbolicBuilder<F, EF>>,
{
    let table =
        TableDeclaration::from_constraints::<F, EF, A>(air, HeightRange::new(FOLDING as u32, 20));
    MachineDeclaration::new(Keccak256Hash, vec![table], PROOF_BUDGET, SECURITY_TARGET).unwrap()
}

/// Prove the fixed instance.
fn proof() -> MultiStarkProof<BindingConfig> {
    let n = 1 << LOG_HEIGHT;
    let config = config();
    let pis = public_values(n);
    let airs = [&FibAir];
    let (pk, _) = setup(&config, &airs, &mut challenger()).unwrap();

    prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &FibAir,
            Table::new(trace(n).transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap()
}

/// Prove the fixed instance and frame it under the declaration.
fn sealed() -> Vec<u8> {
    let declaration = declaration();
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();
    declaration.seal(&run, &proof()).unwrap().into_bytes()
}

/// Verify a byte string against the fixed instance.
fn check(bytes: &[u8]) -> Result<(), SealedVerificationError<PcsError<BindingConfig>>> {
    let config = config();
    let pis = public_values(1 << LOG_HEIGHT);
    let airs = [&FibAir];
    let (_, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let declaration = declaration();
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();
    declaration.verify(
        &run,
        bytes,
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, LOG_HEIGHT, &pis)]),
        &mut challenger(),
    )
}

/// Verify a byte string and take the framing rejection out of the wrapper.
///
/// The wrapper cannot be compared, because the verifier's own error is not comparable.
fn framing_error(bytes: &[u8]) -> EnvelopeError {
    match check(bytes).unwrap_err() {
        SealedVerificationError::Envelope(error) => error,
        other => panic!("expected a framing rejection, got {other:?}"),
    }
}

/// The digest the fixture is pinned by.
fn digest(bytes: &[u8]) -> [u8; 32] {
    Keccak256Hash.hash_iter(bytes.iter().copied())
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

#[test]
fn declaration_reads_the_constraint_system() {
    // The declaration must describe the table without anyone restating it by hand.
    let table = TableDeclaration::from_constraints::<F, EF, FibAir>(
        &FibAir,
        HeightRange::exactly(LOG_HEIGHT as u32),
    );
    assert_eq!(
        table.columns(),
        ColumnCounts {
            committed: 2,
            preprocessed: 0,
            public: 3,
        }
    );
    // Five assertions: two on the first row, two on the transition, one on the last.
    assert_eq!(table.constraints().count, 5);
    assert!(table.constraints().degree >= 2);
    assert!(table.flushes().is_empty());
    assert!(!table.has_lookups());
}

#[test]
fn a_sealed_proof_verifies() {
    check(&sealed()).unwrap();
}

#[test]
fn the_security_target_is_part_of_the_statement() {
    // The same tables under a different target are a different statement.
    let stated = declaration();
    let stricter = MachineDeclaration::new(
        Keccak256Hash,
        stated.tables().to_vec(),
        PROOF_BUDGET,
        SECURITY_TARGET + 1,
    )
    .unwrap();
    assert_eq!(stated.tables(), stricter.tables());

    // A proof sealed under one target does not open under the other.
    let run = stricter.run(&[LOG_HEIGHT], 0).unwrap();
    let err = stricter.open::<BindingConfig>(&run, &sealed()).unwrap_err();
    assert_eq!(err, EnvelopeError::RunMismatch);
}

#[test]
fn a_target_the_statement_cannot_reach_is_refused() {
    // The declaration carries the target, so the caller cannot lower it.
    let ambitious = MachineDeclaration::new(
        Keccak256Hash,
        declaration().tables().to_vec(),
        PROOF_BUDGET,
        256,
    )
    .unwrap();
    let run = ambitious.run(&[LOG_HEIGHT], 0).unwrap();
    let bytes = ambitious.seal(&run, &proof()).unwrap().into_bytes();

    let config = config();
    let pis = public_values(1 << LOG_HEIGHT);
    let airs = [&FibAir];
    let (_, vk) = setup(&config, &airs, &mut challenger()).unwrap();
    let err = ambitious
        .verify(
            &run,
            &bytes,
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, LOG_HEIGHT, &pis)]),
            &mut challenger(),
        )
        .unwrap_err();
    match err {
        SealedVerificationError::Verification(VerificationError::Security(
            SecurityError::InsufficientSecurity { requested, .. },
        )) => assert_eq!(requested, 256),
        other => panic!("expected a security refusal, got {other:?}"),
    }
}

#[test]
fn a_height_range_reaching_a_single_row_is_refused() {
    // One row panics deep in the reduction, so the floor belongs here instead.
    let table =
        TableDeclaration::from_constraints::<F, EF, FibAir>(&FibAir, HeightRange::new(0, 20));
    assert_eq!(
        MachineDeclaration::new(Keccak256Hash, vec![table], PROOF_BUDGET, SECURITY_TARGET)
            .unwrap_err(),
        DeclarationError::HeightBelowFloor {
            table: 0,
            min: 0,
            floor: 1,
        }
    );
}

#[test]
fn a_statement_that_describes_another_table_is_refused() {
    // The declaration is sealed against, so the framing passes and the AIRs are what disagree.
    let config = config();
    let pis = public_values(1 << LOG_HEIGHT);
    let airs = [&FibAir];
    let (_, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let refusal = |declaration: MachineDeclaration<Keccak256Hash>| {
        let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();
        let bytes = declaration.seal(&run, &proof()).unwrap().into_bytes();
        declaration
            .verify(
                &run,
                &bytes,
                &config,
                VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, LOG_HEIGHT, &pis)]),
                &mut challenger(),
            )
            .unwrap_err()
    };

    // A table of a different width, and one that agrees on every count but asserts something else.
    for (air_case, expected) in [
        (refusal(declaration_for(&WideAir)), "the committed columns"),
        (
            refusal(declaration_for(&SubtractingFibAir)),
            "the constraints themselves",
        ),
    ] {
        match air_case {
            SealedVerificationError::AirDisagreement { table, what } => {
                assert_eq!((table, what), (0, expected));
            }
            other => panic!("expected a constraint-system disagreement, got {other:?}"),
        }
    }
}

#[test]
fn a_cell_the_backend_pins_reaches_the_declaration() {
    // A symbolic pass runs the evaluation alone, so a pinned cell has to be added back.
    let heights = HeightRange::new(FOLDING as u32, 20);
    let unpinned =
        TableDeclaration::from_constraints::<F, EF, UnpinnedFibAir>(&UnpinnedFibAir, heights);
    let pinned = TableDeclaration::from_constraints::<F, EF, PinnedFibAir>(&PinnedFibAir, heights);

    // Three written assertions either way, and two pins the backend injects on top.
    assert_eq!(
        unpinned.constraints(),
        LocalConstraints {
            count: 3,
            degree: 2,
        }
    );
    assert_eq!(
        pinned.constraints(),
        LocalConstraints {
            count: 5,
            degree: 2,
        }
    );
    assert_ne!(unpinned, pinned);

    // A run of one statement is therefore refused by the other.
    let one = MachineDeclaration::new(Keccak256Hash, vec![unpinned], PROOF_BUDGET, SECURITY_TARGET)
        .unwrap();
    let other = MachineDeclaration::new(Keccak256Hash, vec![pinned], PROOF_BUDGET, SECURITY_TARGET)
        .unwrap();
    let run = one.run(&[LOG_HEIGHT], 0).unwrap();
    assert_eq!(
        other.seal(&run, &proof()).unwrap_err(),
        EnvelopeError::Declaration(DeclarationError::ForeignRun)
    );
}

#[test]
fn which_public_value_a_pinned_cell_names_reaches_the_declaration() {
    // The same two cells, so a count on its own cannot tell these two tables apart.
    let heights = HeightRange::new(FOLDING as u32, 20);
    let pinned = TableDeclaration::from_constraints::<F, EF, PinnedFibAir>(&PinnedFibAir, heights);
    let swapped =
        TableDeclaration::from_constraints::<F, EF, SwappedPinFibAir>(&SwappedPinFibAir, heights);

    assert_eq!(pinned.constraints(), swapped.constraints());
    assert_ne!(pinned, swapped);
}

#[test]
fn the_end_a_boundary_flush_names_reaches_the_declaration() {
    // A boundary activation leaves nothing in the zerocheck, so the end it names is the
    // only thing separating these two.
    let heights = HeightRange::new(FOLDING as u32, 20);
    let at =
        |end| TableDeclaration::from_constraints::<F, EF, BoundaryAir>(&BoundaryAir(end), heights);
    let first = at(BusBoundary::First);
    let last = at(BusBoundary::Last);

    // Neither writes a constraint, so every counted dimension agrees.
    assert_eq!(
        first.constraints(),
        LocalConstraints {
            count: 0,
            degree: 0
        }
    );
    assert_eq!(first.constraints(), last.constraints());

    // Only the encoded route can tell them apart.
    assert_ne!(first, last);
}

#[test]
fn every_part_of_a_bus_route_reaches_the_declaration() {
    // One symbolic pass keeps the Booleanity check and throws the whole route away.
    let heights = HeightRange::new(FOLDING as u32, 20);
    let route = |channel, direction, cubed| {
        TableDeclaration::from_constraints::<F, EF, BusAir>(
            &BusAir {
                channel,
                direction,
                cubed,
            },
            heights,
        )
    };
    let push = route("squares", BusDirection::Push, false);
    let pull = route("squares", BusDirection::Pull, false);
    let elsewhere = route("memory", BusDirection::Push, false);
    let cubed = route("squares", BusDirection::Push, true);

    // Every one of them writes a single Booleanity check and nothing else.
    for other in [&pull, &elsewhere, &cubed] {
        assert_eq!(
            other.constraints(),
            LocalConstraints {
                count: 1,
                degree: 2,
            }
        );
        assert_eq!(push.constraints(), other.constraints());
        assert_ne!(&push, other);
    }

    // A batch that balances is therefore not the batch that does not.
    let digest = |tables| {
        MachineDeclaration::new(Keccak256Hash, tables, PROOF_BUDGET, SECURITY_TARGET)
            .unwrap()
            .statement_digest()
    };
    assert_ne!(
        digest(vec![push.clone(), pull]),
        digest(vec![push.clone(), push])
    );
}

#[test]
fn a_statement_declaring_a_bus_refuses_a_proof_without_one() {
    // The statement moves a tuple across a bus, and the proof carries no bus part.
    let declaration = declaration_for(&BusAir {
        channel: "squares",
        direction: BusDirection::Push,
        cubed: false,
    });
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();

    let bytes = declaration.seal(&run, &proof()).unwrap().into_bytes();
    let err = declaration.open::<BindingConfig>(&run, &bytes).unwrap_err();
    assert_eq!(
        err,
        EnvelopeError::SectionMismatch {
            section: "bus",
            present: false,
        }
    );
}

#[test]
fn the_values_of_a_periodic_table_reach_the_declaration() {
    // Two tables alike in every count, reading different fixed values.
    let heights = HeightRange::new(FOLDING as u32, 20);
    let low = PeriodicAir([F::ONE, F::TWO]);
    let high = PeriodicAir([F::from_u8(3), F::from_u8(4)]);
    let one = TableDeclaration::from_constraints::<F, EF, PeriodicAir>(&low, heights);
    let other = TableDeclaration::from_constraints::<F, EF, PeriodicAir>(&high, heights);

    assert_eq!(one.columns(), other.columns());
    assert_eq!(one.constraints(), other.constraints());
    assert_ne!(one, other);
}

#[test]
fn a_lookup_that_carries_no_tuple_is_not_a_lookup() {
    // The reduction drops it, so the statement must drop it too or refuse an honest proof.
    let table = TableDeclaration::from_constraints::<F, EF, InertLookupFibAir>(
        &InertLookupFibAir,
        HeightRange::new(FOLDING as u32, 20),
    );
    assert!(!table.has_lookups());

    let n = 1 << LOG_HEIGHT;
    let config = config();
    let pis = public_values(n);
    let airs = [&InertLookupFibAir];
    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();
    let honest = prove(
        &config,
        ProverInstances::new(vec![ProverInstance::new(
            &InertLookupFibAir,
            Table::new(trace(n).transpose()),
            &pk,
            &pis,
        )]),
        0,
        &mut challenger(),
    )
    .unwrap();

    let declaration = declaration_for(&InertLookupFibAir);
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();
    let bytes = declaration.seal(&run, &honest).unwrap().into_bytes();
    declaration
        .verify(
            &run,
            &bytes,
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(
                &InertLookupFibAir,
                &vk,
                LOG_HEIGHT,
                &pis,
            )]),
            &mut challenger(),
        )
        .unwrap();
}

#[test]
fn a_height_outside_the_declared_range_is_refused() {
    let declaration = declaration();
    // The table declares 2..=20, so one below the floor names the whole range back.
    assert_eq!(
        declaration.run(&[1], 0).unwrap_err(),
        DeclarationError::HeightNotDeclared {
            table: 0,
            found: 1,
            min: 2,
            max: 20,
        }
    );
    assert_eq!(
        declaration.run(&[LOG_HEIGHT, LOG_HEIGHT], 0).unwrap_err(),
        DeclarationError::HeightCountMismatch {
            expected: 1,
            found: 2,
        }
    );
}

#[test]
fn a_truncated_header_is_refused() {
    let bytes = sealed();
    for length in [0, 1, 47] {
        assert_eq!(
            framing_error(&bytes[..length]),
            EnvelopeError::HeaderTooShort { found: length }
        );
    }
}

#[test]
fn the_wrong_opening_bytes_are_refused() {
    let mut bytes = sealed();
    bytes[0] ^= 1;
    assert_eq!(framing_error(&bytes), EnvelopeError::BadMagic);
}

#[test]
fn an_unknown_revision_is_refused() {
    let mut bytes = sealed();
    bytes[8..10].copy_from_slice(&(ENVELOPE_VERSION + 1).to_le_bytes());
    // The two numbers differ, so a report that swaps them fails here.
    assert_eq!(
        framing_error(&bytes),
        EnvelopeError::EnvelopeVersion {
            found: ENVELOPE_VERSION + 1,
            expected: ENVELOPE_VERSION,
        }
    );

    let mut bytes = sealed();
    bytes[10..12].copy_from_slice(&(BODY_REVISION + 1).to_le_bytes());
    assert_eq!(
        framing_error(&bytes),
        EnvelopeError::BodyRevision {
            found: BODY_REVISION + 1,
            expected: BODY_REVISION,
        }
    );
}

#[test]
fn a_fingerprint_from_another_statement_is_refused() {
    let mut bytes = sealed();
    bytes[12] ^= 1;
    assert_eq!(framing_error(&bytes), EnvelopeError::RunMismatch);
}

#[test]
fn a_length_field_above_the_budget_is_refused() {
    // The length a proof declares can never make the reader look past the budget.
    let mut bytes = sealed();
    bytes[44..48].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        framing_error(&bytes),
        EnvelopeError::BodyAboveBudget {
            found: u32::MAX as usize,
            budget: PROOF_BUDGET,
        }
    );
}

#[test]
fn a_length_field_disagreeing_with_the_input_is_refused() {
    let bytes = sealed();
    let body = bytes.len() - HEADER_LEN;

    let mut short = bytes.clone();
    short[44..48].copy_from_slice(&((body + 1) as u32).to_le_bytes());
    assert_eq!(
        framing_error(&short),
        EnvelopeError::Truncated {
            declared: body + 1,
            available: body,
        }
    );

    let mut long = bytes;
    long[44..48].copy_from_slice(&((body - 1) as u32).to_le_bytes());
    assert_eq!(
        framing_error(&long),
        EnvelopeError::TrailingBytes { extra: 1 }
    );
}

#[test]
fn trailing_bytes_are_refused() {
    // The decoder itself would ignore them, so the framing is what rejects them.
    let mut bytes = sealed();
    bytes.push(0);
    assert_eq!(
        framing_error(&bytes),
        EnvelopeError::TrailingBytes { extra: 1 }
    );
}

#[test]
fn a_body_the_decoder_does_not_finish_is_refused() {
    // The decoder stops at the end of the proof and ignores whatever follows.
    //
    // The framing is what notices, so a padded body is rejected rather than accepted.
    let bytes = sealed();
    let body = bytes.len() - HEADER_LEN;

    let mut padded = bytes;
    padded.push(0);
    padded[44..48].copy_from_slice(&((body + 1) as u32).to_le_bytes());

    assert_eq!(
        framing_error(&padded),
        EnvelopeError::UnreadBodyBytes { remaining: 1 }
    );
}

#[test]
fn a_corrupted_body_is_refused_before_the_transcript() {
    // The body is replaced by a length field that asks for far more than follows.
    let bytes = sealed();
    let mut body = vec![0xff_u8; 8];
    body.extend_from_slice(&[0u8; 8]);

    let mut forged = bytes[..HEADER_LEN].to_vec();
    forged[44..48].copy_from_slice(&(body.len() as u32).to_le_bytes());
    forged.extend_from_slice(&body);

    assert_eq!(framing_error(&forged), EnvelopeError::Malformed);
}

#[test]
fn the_declared_heights_must_match_the_instances() {
    // The run fixes the heights, and an instance list that disagrees is refused.
    let bytes = sealed();
    let config = config();
    let pis = public_values(1 << LOG_HEIGHT);
    let airs = [&FibAir];
    let (_, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let declaration = declaration();
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();
    let err = declaration
        .verify(
            &run,
            &bytes,
            &config,
            VerifierInstances::new(vec![VerifierInstance::new(
                &FibAir,
                &vk,
                LOG_HEIGHT - 1,
                &pis,
            )]),
            &mut challenger(),
        )
        .unwrap_err();
    // The wrapper is not comparable, so the variant is matched and its payload compared.
    match err {
        SealedVerificationError::RunDisagreement { what } => {
            assert_eq!(what, "a table height");
        }
        other => panic!("expected a height disagreement, got {other:?}"),
    }
}

#[test]
fn a_statement_declaring_a_lookup_refuses_a_proof_without_one() {
    // The statement is read off a table that does look something up, and the proof is not.
    let declaration = declaration_for(&LookupFibAir);
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();

    let bytes = declaration.seal(&run, &proof()).unwrap().into_bytes();
    let err = declaration.open::<BindingConfig>(&run, &bytes).unwrap_err();
    assert_eq!(
        err,
        EnvelopeError::SectionMismatch {
            section: "lookup",
            present: false,
        }
    );
}

#[test]
fn verify_the_compatibility_fixture() {
    let bytes = std::fs::read(fixture_path(FIXTURE)).expect(
        "missing fixture; run: cargo test -p p3-multi-stark --test backend_contract -- --ignored",
    );
    // The bytes are pinned, so regenerating the file without saying so fails here.
    assert_eq!(
        digest(&bytes),
        FIXTURE_DIGEST,
        "the fixture changed; regenerate it, bump the body revision, and pin the new digest"
    );
    // The pinned revision is the one on disk, and the one this build speaks.
    assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), FIXTURE_REVISION);
    assert_eq!(FIXTURE_REVISION, BODY_REVISION);
    check(&bytes).unwrap();
}

#[test]
fn a_fixture_from_an_older_revision_is_refused() {
    // The real file from the revision before this one, kept rather than reconstructed.
    let bytes = std::fs::read(fixture_path(RETIRED_FIXTURE)).expect("missing retired fixture");
    assert_eq!(digest(&bytes), RETIRED_DIGEST);
    assert_eq!(u16::from_le_bytes([bytes[10], bytes[11]]), RETIRED_REVISION);
    assert_eq!(
        framing_error(&bytes),
        EnvelopeError::BodyRevision {
            found: RETIRED_REVISION,
            expected: BODY_REVISION,
        }
    );
}

#[test]
#[ignore]
fn generate_the_compatibility_fixture() {
    // Regenerate with: cargo test -p p3-multi-stark --test backend_contract -- --ignored
    let path = fixture_path(FIXTURE);
    let bytes = sealed();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, &bytes).unwrap();
    println!("pin this digest, under body revision {BODY_REVISION}:");
    println!("{:?}", digest(&bytes));
}
