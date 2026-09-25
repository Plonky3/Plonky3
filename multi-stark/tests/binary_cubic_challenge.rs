//! A small binary machine proved with values in `GF(2^64)` and challenges in `GF(2^192)`.
//!
//! Three tables, every cross-table channel the binary backend has:
//!
//! ```text
//!     accesses   one memory access per row, in execution order
//!     ram        the same accesses sorted by cell then clock      read-write memory
//!     rom        a fixed table the writes read their value from   indexed lookup
//!
//!     accesses --ram-access--> ram       bus:     (write, cell bits, clock bits, value)
//!     accesses --rom---------> rom       lookup:  value = rom[position] on a write
//! ```
//!
//! The machine runs under two commitments:
//!
//! ```text
//!     dense    one GF(2^64) element per cell, any value
//!     packed   sixty-four cells per committed element, every cell a bit
//! ```
//!
//! Both reach 128 bits: the 32-byte hash is the only term that binds.

use core::fmt::Debug;
use core::iter;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::BooleanTraceCommitmentData;
use p3_binary_pcs::whir::{
    BinaryWhirDomain, BinaryWhirProfile, BooleanWhirData, BooleanWhirPcs, BooleanWhirProver,
    BooleanWhirTracePcs, recommended_cap_height,
};
use p3_bus::{
    BusActivation, BusDirection, BusInteractionBuilder, BusName, RamAccess, RamAir, RamBoundary,
    RamLayout, RamStatement, RamTrace,
};
use p3_challenger::{CanObserve, HashChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_lookup::{IndexedLookupBuilder, TraceWindow};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::{Commitment, MultiStarkConfig, PcsError, PcsProverError, ProverData};
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, prove,
    security_report, setup, verify,
};
use p3_security::ErrorBits;
use p3_sumcheck::PrescribedPointPcs;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, TableShape, Witness, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::{SecurityAssumption, WhirProver, WhirProverData};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = Poly64;
type EF = Poly192;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type Domain = BinaryWhirDomain<F>;
type DensePcs = WhirProver<EF, F, Domain, Mmcs, Challenger, PrefixProver<F, EF>>;
type PackedPcs = BooleanWhirTracePcs<F, EF, Domain, Mmcs, Challenger>;

/// Security every verification here must reach.
const SECURITY_BITS: usize = 128;

/// Per-term target of the proximity schedule, above the composed one so the union still clears it.
const TERM_BITS: usize = 140;

/// Collision resistance of a 32-byte Keccak digest.
const COLLISION_BITS: usize = 128;

/// First-round folding factor of the dense commitment, also its smallest table arity.
const FOLDING: usize = 2;

/// Base-two logarithm of the inverse code rate.
const LOG_INV_RATE: usize = 2;

/// Accesses the machine makes, which is the height of both memory tables.
const STEPS: usize = 8;

/// Bits in a cell number: four cells.
const ADDRESS_BITS: usize = 2;

/// Bits in a clock reading: eight steps.
const CLOCK_BITS: usize = 3;

/// Channel the accesses go out on.
const ACCESS: &str = "ram-access";

/// Name of the fixed table the writes read.
const ROM: &str = "rom";

/// Label the report gives the hash's collision cap.
const COLLISION_LABEL: &str = "commitment-and-transcript-collision";

/// The program: `(write, cell, rom entry)` per step, the clock being the step.
///
/// A read names entry zero, which holds zero, and the lookup then says nothing.
const PROGRAM: [(bool, u64, usize); STEPS] = [
    (true, 1, 1),  // mem[1] = rom[1]
    (false, 1, 0), // read mem[1]
    (true, 2, 2),  // mem[2] = rom[2]
    (false, 0, 0), // read mem[0], never written
    (true, 1, 6),  // mem[1] = rom[6]
    (false, 1, 0), // read mem[1]
    (false, 2, 0), // read mem[2]
    (false, 3, 0), // read mem[3], never written
];

/// Columns of one access row: the bus payload, then the lookup.
const ACCESS_WIDTH: usize = 1 + ADDRESS_BITS + CLOCK_BITS + 1 + 2;

/// Column of the value an access reads or writes.
const VALUE: usize = 1 + ADDRESS_BITS + CLOCK_BITS;

/// Column naming the rom entry.
const POSITION: usize = VALUE + 1;

/// Column holding what the lookup pulled.
const PULLED: usize = VALUE + 2;

/// The three tables of the machine.
enum Chip {
    /// One access per row, in execution order.
    Accesses,
    /// The accesses sorted by cell then clock.
    Ram(Box<RamAir>),
    /// The fixed table, one value per entry.
    Rom,
}

impl Chip {
    /// Every table, in the order proofs list them.
    fn all() -> Vec<Self> {
        let ram = RamAir::new(statement()).expect("the machine's memory statement is valid");
        vec![Self::Accesses, Self::Ram(Box::new(ram)), Self::Rom]
    }
}

/// Memory of one self-contained proof: starts empty, exports nothing.
fn statement() -> RamStatement {
    RamStatement {
        access_bus: ACCESS.into(),
        access_count: STEPS,
        address_bits: ADDRESS_BITS,
        timestamp_bits: CLOCK_BITS,
        value_width: 1,
        boundary: RamBoundary::SingleProof,
    }
}

impl BaseAir<F> for Chip {
    fn width(&self) -> usize {
        match self {
            Self::Accesses => ACCESS_WIDTH,
            Self::Ram(ram) => BaseAir::<F>::width(ram.as_ref()),
            Self::Rom => 1,
        }
    }
}

impl<AB> Air<AB> for Chip
where
    AB: BusInteractionBuilder<F = F> + IndexedLookupBuilder,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Accesses => eval_accesses(builder),
            Self::Ram(ram) => ram.as_ref().eval(builder),
            Self::Rom => eval_rom(builder),
        }
    }
}

/// Issue every access on the bus, and tie each written value to the rom.
///
/// The machine owes the memory a rising clock, which this test does not constrain.
fn eval_accesses<AB: BusInteractionBuilder + IndexedLookupBuilder>(builder: &mut AB) {
    let main = builder.main();
    let row = main.current_slice();
    let write = row[0];
    let payload = row[..=VALUE]
        .iter()
        .map(|&cell| cell.into())
        .collect::<Vec<AB::Expr>>();
    let (value, pulled) = (row[VALUE], row[PULLED]);

    // A write stores what the rom holds at the entry the row names.
    builder.assert_bool(write);
    builder.assert_zero(write * (value - pulled));

    builder.push_bus_interaction(
        BusName::new(ACCESS),
        BusDirection::Push,
        payload,
        BusActivation::Always,
    );
    builder.push_indexed_read(ROM, POSITION, [PULLED]);
}

/// Provide the rom, whose entry zero holds zero.
fn eval_rom<AB: IndexedLookupBuilder>(builder: &mut AB) {
    let value = builder.main().current_slice()[0];
    builder.when_first_row().assert_zero(value);
    builder.push_indexed_table(ROM, TraceWindow::Main, [0]);
}

/// The machine's tables, in the order [`Chip::all`] lists them.
struct Traces {
    /// One matrix per table, row-major.
    matrices: Vec<RowMajorMatrix<F>>,
}

impl Traces {
    /// Run the program over this rom.
    fn run(rom: &[F]) -> Self {
        let mut memory = [F::ZERO; 1 << ADDRESS_BITS];
        let mut accesses = Vec::with_capacity(STEPS);
        let mut rows = Vec::with_capacity(STEPS * ACCESS_WIDTH);
        for (clock, &(write, cell, entry)) in PROGRAM.iter().enumerate() {
            // A rom narrower than the program reuses its entries.
            let entry = entry % rom.len();
            if write {
                memory[cell as usize] = rom[entry];
            }
            let value = memory[cell as usize];
            accesses.push(if write {
                RamAccess::write(cell, clock as u64, [value])
            } else {
                RamAccess::read(cell, clock as u64, [value])
            });

            // Payload digits least significant first, as the memory reads them.
            rows.push(F::from_bool(write));
            rows.extend((0..ADDRESS_BITS).map(|bit| F::from_bool((cell >> bit) & 1 == 1)));
            rows.extend((0..CLOCK_BITS).map(|bit| F::from_bool((clock >> bit) & 1 == 1)));
            rows.push(value);

            // An entry's position is its bit pattern read as a field element.
            rows.push(F::new(entry as u64));
            rows.push(rom[entry]);
        }
        let ram = RamTrace::build(&statement(), &accesses).expect("the program is consistent");
        let ram_width = ram.width();
        Self {
            matrices: vec![
                RowMajorMatrix::new(rows, ACCESS_WIDTH),
                RowMajorMatrix::new(ram.into_values(), ram_width),
                RowMajorMatrix::new(rom.to_vec(), 1),
            ],
        }
    }

    /// Every table's shape.
    fn shapes(&self) -> Vec<TableShape> {
        self.matrices
            .iter()
            .map(|matrix| TableShape::new(matrix.height().ilog2() as usize, matrix.width()))
            .collect()
    }

    /// Every table, one column per trace column.
    fn tables(&self) -> Vec<Table<F>> {
        self.matrices
            .iter()
            .map(|matrix| Table::new(matrix.transpose()))
            .collect()
    }
}

/// A rom of `entries` random values, entry zero at zero.
fn random_rom(entries: usize, seed: u64) -> Vec<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    iter::once(F::ZERO)
        .chain((1..entries).map(|_| rng.random()))
        .collect()
}

/// A fresh transcript, the same on both sides.
const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A Merkle tree over the sixty-four-bit alphabet.
const fn mmcs(cap_height: usize) -> Mmcs {
    Mmcs::new(
        Hash::new(Keccak256Hash),
        Compress::new(Keccak256Hash),
        cap_height,
    )
}

/// The proximity regime both commitments assume: unique decoding, proven, no conjecture.
const fn profile(folding: usize) -> BinaryWhirProfile {
    BinaryWhirProfile::unique_decoding(TERM_BITS, LOG_INV_RATE, folding)
}

/// Every cell a `GF(2^64)` element, opened through WHIR in the unique-decoding regime.
struct DenseConfig {
    /// Scheme sized for the stacked tables.
    pcs: DensePcs,
}

impl DenseConfig {
    /// Wire the scheme for tables of these shapes.
    fn new(shapes: &[TableShape]) -> Self {
        let (arity, _) = plan_stacked_layout(shapes);
        let domain = Domain::default();
        let config = profile(FOLDING)
            .config::<EF, F, Challenger, _>(arity, &domain)
            .expect("the schedule fits the stacked arity");
        let cap_height = recommended_cap_height(&config);
        Self {
            pcs: DensePcs::new(config, domain, mmcs(cap_height)),
        }
    }
}

impl MultiStarkConfig for DenseConfig {
    type Val = F;
    type Challenge = EF;
    type Challenger = Challenger;
    type Pcs = DensePcs;

    fn pcs(&self) -> &DensePcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(COLLISION_BITS)
    }

    fn min_num_variables(&self) -> usize {
        FOLDING
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        PrefixProver::<F, EF>::new_witness(tables, FOLDING)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a WhirProverData<F, EF, Mmcs, PrefixProver<F, EF>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Every cell a bit, packed sixty-four to a committed element.
struct PackedConfig {
    /// Scheme sized for the stacked bits.
    pcs: PackedPcs,
}

impl PackedConfig {
    /// Wire the scheme for tables of these shapes.
    fn new(shapes: &[TableShape]) -> Self {
        let (arity, _) = plan_stacked_layout(shapes);
        let absorbed = BitRingSwitch::<F, EF>::ABSORBED;
        let domain = Domain::default();
        let config = profile(1)
            .config::<EF, F, Challenger, _>(arity - absorbed, &domain)
            .expect("the schedule fits the packed arity");
        let cap_height = recommended_cap_height(&config);
        let prover = BooleanWhirProver::new(config, domain, mmcs(cap_height));
        let bits = BooleanWhirPcs::new(prover, arity).expect("the schedule commits the packing");
        Self {
            pcs: PackedPcs::from_commitment(bits),
        }
    }
}

impl MultiStarkConfig for PackedConfig {
    type Val = F;
    type Challenge = EF;
    type Challenger = Challenger;
    type Pcs = PackedPcs;

    fn pcs(&self) -> &PackedPcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(COLLISION_BITS)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Vec<Table<F>> {
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BooleanTraceCommitmentData<F, BooleanWhirData<F, EF, Mmcs>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Prove the machine over these traces, verify it, and return the proof with its security.
fn prove_and_verify<C>(config: &C, chips: &[Chip], traces: &Traces) -> (MultiStarkProof<C>, f64)
where
    C: MultiStarkConfig<Val = F, Challenge = EF, Challenger = Challenger>,
    C::Pcs: PrescribedPointPcs<EF, Challenger>,
    Challenger: CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
    PcsError<C>: Debug,
    PcsProverError<C>: Debug,
{
    let airs = chips.iter().collect::<Vec<_>>();
    let (pk, vk) = setup(config, &airs, &mut challenger()).unwrap();
    let heights = traces
        .shapes()
        .iter()
        .map(TableShape::num_variables)
        .collect::<Vec<_>>();

    let proof = prove(
        config,
        ProverInstances::new(
            chips
                .iter()
                .zip(traces.tables())
                .map(|(chip, table)| ProverInstance::new(chip, table, &pk, &[]))
                .collect(),
        ),
        0,
        &mut challenger(),
    )
    .unwrap();

    let instances = || {
        VerifierInstances::new(
            chips
                .iter()
                .zip(&heights)
                .map(|(chip, &height)| VerifierInstance::new(chip, &vk, height, &[]))
                .collect(),
        )
    };
    verify(config, instances(), &proof, 0, &mut challenger()).unwrap();

    // The report names every term, the hash's cap among them.
    let report = security_report(config, &instances()).unwrap();
    let algebraic = report
        .terms()
        .iter()
        .filter(|term| term.label != COLLISION_LABEL)
        .map(|term| term.bits)
        .collect::<Vec<_>>();

    // Every algebraic term together clears the target on its own.
    //
    // The union with the cap then sits just under 128, where the hash alone would put it.
    let union = ErrorBits::sum(&algebraic).bits();
    assert!(
        union >= SECURITY_BITS as f64,
        "algebraic terms union to {union}"
    );
    let composed = report.security_bits().expect("every component is assessed");
    assert!(
        composed > SECURITY_BITS as f64 - 0.01,
        "composed {composed}"
    );
    (proof, union)
}

#[test]
fn the_machine_proves_with_dense_tables() {
    // Eight rom entries of arbitrary field values exercise multi-bit positions.
    let traces = Traces::run(&random_rom(8, 0xD15E));
    let chips = Chip::all();
    let config = DenseConfig::new(&traces.shapes());
    let (proof, union) = prove_and_verify(&config, &chips, &traces);

    // The lookup reached the proof, and the regime is the proven one.
    assert!(proof.indexed.is_some());
    assert!(union >= SECURITY_BITS as f64);
    assert_eq!(
        profile(FOLDING).assumption(),
        SecurityAssumption::UniqueDecoding
    );
}

#[test]
fn the_machine_proves_with_bit_packed_tables() {
    // A two-entry rom of bits keeps every cell a bit, positions included.
    //
    //     entry 0 -> 0, entry 1 -> 1, and a position is its entry's bit pattern
    let traces = Traces::run(&[F::ZERO, F::ONE]);
    let chips = Chip::all();
    let config = PackedConfig::new(&traces.shapes());
    let (proof, _) = prove_and_verify(&config, &chips, &traces);
    assert!(proof.indexed.is_some());
}

#[test]
fn a_moved_memory_value_is_refused() {
    // The memory holds one value the accesses never produced, so the bus cannot balance.
    let mut traces = Traces::run(&random_rom(4, 0xBAD));
    let chips = Chip::all();
    let config = DenseConfig::new(&traces.shapes());
    let airs = chips.iter().collect::<Vec<_>>();
    let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

    let value = RamLayout::new(&statement()).unwrap().value;
    let ram = &mut traces.matrices[1];
    let width = ram.width();
    ram.values[width + value] += F::ONE;

    let heights = traces
        .shapes()
        .iter()
        .map(TableShape::num_variables)
        .collect::<Vec<_>>();
    let proved = prove(
        &config,
        ProverInstances::new(
            chips
                .iter()
                .zip(traces.tables())
                .map(|(chip, table)| ProverInstance::new(chip, table, &pk, &[]))
                .collect(),
        ),
        0,
        &mut challenger(),
    );
    // A prover may refuse an unbalanced bus itself; one that proves anyway is refused.
    if let Ok(proof) = proved {
        let instances = VerifierInstances::new(
            chips
                .iter()
                .zip(&heights)
                .map(|(chip, &height)| VerifierInstance::new(chip, &vk, height, &[]))
                .collect(),
        );
        assert!(verify(&config, instances, &proof, 0, &mut challenger()).is_err());
    }
}
