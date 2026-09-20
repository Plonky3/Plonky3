//! The machine-facing contract end to end: declare, prove, seal, and refuse bad input.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::{MultiStarkConfig, PcsError};
use p3_multi_stark::contract::declaration::{ColumnCounts, DeclarationError, HeightRange};
use p3_multi_stark::contract::envelope::{
    BODY_REVISION, ENVELOPE_VERSION, EnvelopeError, HEADER_LEN, SealedVerificationError,
    verify_sealed,
};
use p3_multi_stark::contract::{BindingOnly, Hiding, MachineDeclaration, TableDeclaration};
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove,
    setup,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
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

const FIXTURE: &str = "tests/fixtures/backend_contract_v1.envelope";

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
fn declaration() -> MachineDeclaration<BindingOnly> {
    let table = TableDeclaration::from_constraints::<F, EF, FibAir>(
        &FibAir,
        HeightRange::new(FOLDING as u32, 20),
    );
    MachineDeclaration::new(vec![table], PROOF_BUDGET).unwrap()
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
    verify_sealed(
        &declaration,
        &run,
        bytes,
        &config,
        VerifierInstances::new(vec![VerifierInstance::new(&FibAir, &vk, LOG_HEIGHT, &pis)]),
        &mut challenger(),
    )
}

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)
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
fn the_promise_is_part_of_the_type() {
    // The same tables under a different promise are a different statement.
    let binding = declaration();
    let hiding =
        MachineDeclaration::<Hiding>::new(binding.tables().to_vec(), PROOF_BUDGET).unwrap();
    assert_eq!(binding.tables(), hiding.tables());

    // A proof sealed under one promise does not open under the other.
    let run = hiding.run(&[LOG_HEIGHT], 0).unwrap();
    let err = hiding.open::<BindingConfig>(&run, &sealed()).unwrap_err();
    assert!(matches!(err, EnvelopeError::RunMismatch));
}

#[test]
fn a_height_outside_the_declared_range_is_refused() {
    let declaration = declaration();
    assert!(matches!(
        declaration.run(&[1], 0),
        Err(DeclarationError::HeightNotDeclared { .. })
    ));
    assert!(matches!(
        declaration.run(&[LOG_HEIGHT, LOG_HEIGHT], 0),
        Err(DeclarationError::HeightCountMismatch { .. })
    ));
}

#[test]
fn a_truncated_header_is_refused() {
    let bytes = sealed();
    for length in [0, 1, HEADER_LEN - 1] {
        let err = check(&bytes[..length]).unwrap_err();
        assert!(matches!(
            err,
            SealedVerificationError::Envelope(EnvelopeError::HeaderTooShort { .. })
        ));
    }
}

#[test]
fn the_wrong_opening_bytes_are_refused() {
    let mut bytes = sealed();
    bytes[0] ^= 1;
    assert!(matches!(
        check(&bytes).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::BadMagic)
    ));
}

#[test]
fn an_unknown_revision_is_refused() {
    let mut bytes = sealed();
    bytes[8..10].copy_from_slice(&(ENVELOPE_VERSION + 1).to_le_bytes());
    assert!(matches!(
        check(&bytes).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::EnvelopeVersion { .. })
    ));

    let mut bytes = sealed();
    bytes[10..12].copy_from_slice(&(BODY_REVISION + 1).to_le_bytes());
    assert!(matches!(
        check(&bytes).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::BodyRevision { .. })
    ));
}

#[test]
fn a_fingerprint_from_another_statement_is_refused() {
    let mut bytes = sealed();
    bytes[12] ^= 1;
    assert!(matches!(
        check(&bytes).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::RunMismatch)
    ));
}

#[test]
fn a_length_field_above_the_budget_is_refused() {
    // The length a proof declares can never make the reader look past the budget.
    let mut bytes = sealed();
    bytes[44..48].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        check(&bytes).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::BodyAboveBudget { .. })
    ));
}

#[test]
fn a_length_field_disagreeing_with_the_input_is_refused() {
    let bytes = sealed();
    let body = bytes.len() - HEADER_LEN;

    let mut short = bytes.clone();
    short[44..48].copy_from_slice(&((body + 1) as u32).to_le_bytes());
    assert!(matches!(
        check(&short).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::Truncated { .. })
    ));

    let mut long = bytes;
    long[44..48].copy_from_slice(&((body - 1) as u32).to_le_bytes());
    assert!(matches!(
        check(&long).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::TrailingBytes { .. })
    ));
}

#[test]
fn trailing_bytes_are_refused() {
    // The decoder itself would ignore them, so the framing is what rejects them.
    let mut bytes = sealed();
    bytes.push(0);
    assert!(matches!(
        check(&bytes).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::TrailingBytes { extra: 1 })
    ));
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

    assert!(matches!(
        check(&padded).unwrap_err(),
        SealedVerificationError::Envelope(EnvelopeError::UnreadBodyBytes { remaining: 1 })
    ));
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

    let err = check(&forged).unwrap_err();
    assert!(matches!(
        err,
        SealedVerificationError::Envelope(EnvelopeError::Malformed)
    ));
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
    let err = verify_sealed(
        &declaration,
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
    assert!(matches!(
        err,
        SealedVerificationError::RunDisagreement { .. }
    ));
}

#[test]
fn a_statement_declaring_a_lookup_refuses_a_proof_without_one() {
    // The proof carries no lookup part, and the statement says one must be there.
    let table = TableDeclaration::from_constraints::<F, EF, FibAir>(
        &FibAir,
        HeightRange::new(FOLDING as u32, 20),
    )
    .with_local_lookups(1);
    let declaration = MachineDeclaration::<BindingOnly>::new(vec![table], PROOF_BUDGET).unwrap();
    let run = declaration.run(&[LOG_HEIGHT], 0).unwrap();

    let bytes = declaration.seal(&run, &proof()).unwrap().into_bytes();
    let err = declaration.open::<BindingConfig>(&run, &bytes).unwrap_err();
    assert!(matches!(err, EnvelopeError::SectionMismatch { .. }));
}

#[test]
fn verify_the_compatibility_fixture() {
    let bytes = std::fs::read(fixture_path()).expect(
        "missing fixture; run: cargo test -p p3-multi-stark --test backend_contract -- --ignored",
    );
    check(&bytes).unwrap();
}

#[test]
#[ignore]
fn generate_the_compatibility_fixture() {
    // Regenerate with: cargo test -p p3-multi-stark --test backend_contract -- --ignored
    let path = fixture_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, sealed()).unwrap();
}
