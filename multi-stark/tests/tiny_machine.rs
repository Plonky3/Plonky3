//! A tiny machine proved in two chained segments, through the public contract alone.
//!
//! The machine is an accumulator with four memory cells and three instructions:
//!
//! ```text
//!     LOAD  a        acc <- mem[a]
//!     STORE a        mem[a] <- acc
//!     MACC  a, k     acc <- acc * mem[a] + k
//! ```
//!
//! Five tables of five different shapes prove one segment of its execution:
//!
//! ```text
//!     cpu       one step per row          state transitions, one memory access per step
//!     bytecode  one instruction per row   the program, fixed at setup, looked up by the cpu
//!     macc      two gadgets per row       out = a * b + c, looked up by the cpu
//!     ram       one access per row        mutable memory, sorted by cell then clock
//!     image     one cell per row          the memory a segment inherits and hands on
//! ```
//!
//! The channels between them:
//!
//! ```text
//!     cpu --bytecode--> bytecode        lookup: (pc, opcode, address, immediate)
//!     cpu --macc------> macc            lookup: (acc, value, immediate, next acc)
//!     cpu --ram-access-> ram            bus:    (write, address bits, clock bits, value)
//!     image --ram-incoming-> ram        bus:    the value each touched cell starts with
//!     ram --ram-outgoing--> image       bus:    the value each touched cell ends with
//! ```
//!
//! A segment's boundary is its public values: the cpu state and the full memory image.
//!
//! The backend only learns which public values those are, never what they mean.

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_bus::{
    BusActivation, BusDirection, BusInteractionBuilder, BusName, RamAccess, RamAir, RamBoundary,
    RamStatement, RamTrace,
};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_lookup::{Count, InteractionBuilder};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::{MultiStarkConfig, PcsError, PcsProverError};
use p3_multi_stark::contract::{
    ChainError, EnvelopeError, HeightRange, MachineDeclaration, PublicSlot,
    SealedVerificationError, SegmentClaim, SegmentInterface, TableCost, TableDeclaration, chain,
};
use p3_multi_stark::{
    ProverInstance, ProverInstances, ProvingError, ProvingKey, VerifierInstance, VerifierInstances,
    VerifyingKey, prove, setup,
};
use p3_sumcheck::layout::{Layout, PrefixProver, Table, TableShape, Witness, plan_stacked_layout};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
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

type Declaration = MachineDeclaration<Keccak256Hash>;
type Verdict<T> = Result<T, SealedVerificationError<PcsError<MachineConfig>>>;

/// First-round folding factor, which is also the smallest table arity the scheme accepts.
const FOLDING: usize = 2;

/// Security level every verification must reach.
const SECURITY_TARGET: usize = 20;

/// Collision resistance the primitives of this configuration supply.
const COLLISION_BITS: usize = 100;

/// Largest sealed body a segment may take, in bytes.
const PROOF_BUDGET: usize = 1 << 17;

/// Number of memory cells.
const CELLS: usize = 4;

/// Bits in a cell number.
const ADDRESS_BITS: usize = 2;

/// Bits in a clock reading: two segments of four steps read the clock from 0 to 7.
const CLOCK_BITS: usize = 3;

/// Steps one segment executes, which is the cpu and ram height.
const STEPS: usize = 4;

/// Gadget instances one macc row holds.
const LANES: usize = 2;

/// Rows of the macc table, so it offers eight gadget slots.
const GADGET_ROWS: usize = 4;

/// Columns of one gadget lane: a, b, c, out, multiplicity.
const LANE_WIDTH: usize = 5;

/// Base-two logarithm of each table's height, in table order.
const LOG_HEIGHTS: [usize; 5] = [2, 3, 2, 2, 2];

/// Opcode of `LOAD a`, as the bytecode stores it.
const LOAD: u32 = 0;
/// Opcode of `STORE a`.
const STORE: u32 = 1;
/// Opcode of `MACC a, k`.
const MACC: u32 = 2;

/// The program: `(opcode, address, immediate)`, one instruction per pc.
///
/// Every segment reads a cell before it writes it, as a continuing memory requires.
const PROGRAM: [(u32, u32, u32); 8] = [
    (LOAD, 0, 0),  // acc = 3
    (MACC, 1, 2),  // acc = 3 * 5 + 2 = 17
    (STORE, 1, 0), // mem[1] = 17
    (MACC, 2, 1),  // acc = 17 * 7 + 1 = 120
    (MACC, 1, 0),  // acc = 120 * 17 = 2040
    (STORE, 1, 0), // mem[1] = 2040
    (LOAD, 3, 0),  // acc = 11
    (MACC, 1, 4),  // acc = 11 * 2040 + 4 = 22444
];

/// Memory before the first segment runs.
const INITIAL_MEMORY: [u32; CELLS] = [3, 5, 7, 11];

/// Lookup channel from the cpu to the bytecode.
const BYTECODE: &str = "bytecode";
/// Lookup channel from the cpu to the gadget table.
const GADGET: &str = "macc";
/// Bus the cpu issues its memory accesses on.
const ACCESS: &str = "ram-access";
/// Bus carrying the memory a segment inherits.
const INCOMING: &str = "ram-incoming";
/// Bus carrying the memory a segment hands on.
const OUTGOING: &str = "ram-outgoing";

/// Position of the cpu among the tables.
const CPU_TABLE: usize = 0;
/// Position of the memory image among the tables.
const IMAGE_TABLE: usize = 4;

/// A commitment scheme for the main traces and one for the bytecode fixed at setup.
struct MachineConfig {
    /// Scheme sized for the stacked main traces.
    pcs: TestPcs,
    /// Scheme sized for the stacked preprocessed traces.
    preprocessed_pcs: TestPcs,
}

impl MultiStarkConfig for MachineConfig {
    type Val = F;
    type Challenge = EF;
    type Challenger = MyChallenger;
    type Pcs = TestPcs;

    fn pcs(&self) -> &TestPcs {
        &self.pcs
    }

    fn preprocessed_pcs(&self) -> &TestPcs {
        &self.preprocessed_pcs
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

/// Fixed permutation, so prover and verifier transcripts agree.
fn perm() -> Perm {
    let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
    Perm::new_from_rng_128(&mut rng)
}

/// A fresh transcript.
fn challenger() -> MyChallenger {
    MyChallenger::new(perm())
}

/// Per-round log-inverse rates for a stacked polynomial of this arity.
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

/// A scheme for the stacked polynomial that tables of these shapes lay out into.
fn pcs_for(shapes: &[TableShape]) -> TestPcs {
    let (stacked, _) = plan_stacked_layout(shapes);
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
    TestPcs::new(
        WhirConfig::new(stacked, params).unwrap(),
        MyDft::default(),
        mmcs,
    )
}

/// Both schemes, sized for the tables at the heights every segment runs at.
fn config(chips: &[Chip]) -> MachineConfig {
    let main: Vec<TableShape> = chips
        .iter()
        .zip(LOG_HEIGHTS)
        .map(|(chip, log_height)| TableShape::new(log_height, BaseAir::<F>::width(chip)))
        .collect();
    let preprocessed: Vec<TableShape> = chips
        .iter()
        .zip(LOG_HEIGHTS)
        .filter(|(chip, _)| BaseAir::<F>::preprocessed_width(*chip) > 0)
        .map(|(chip, log_height)| {
            TableShape::new(log_height, BaseAir::<F>::preprocessed_width(chip))
        })
        .collect();
    MachineConfig {
        pcs: pcs_for(&main),
        preprocessed_pcs: pcs_for(&preprocessed),
    }
}

/// One cpu row.
///
/// The clock and the cell number are also written as bits, because memory reads them that way.
#[repr(C)]
struct CpuRow<T> {
    /// Position of this step's instruction.
    pc: T,
    /// Clock reading this step's memory access is stamped with.
    clock: T,
    /// The clock, least significant bit first.
    clock_bits: [T; CLOCK_BITS],
    /// Set on a `LOAD`.
    is_load: T,
    /// Set on a `STORE`, which is also the memory write marker.
    is_store: T,
    /// Set on a `MACC`.
    is_macc: T,
    /// The cell this step accesses, least significant bit first.
    address_bits: [T; ADDRESS_BITS],
    /// The instruction's immediate.
    immediate: T,
    /// Accumulator before this step.
    acc: T,
    /// Accumulator after this step.
    next_acc: T,
    /// Value this step reads from or writes to memory.
    value: T,
}

const CPU_WIDTH: usize = size_of::<CpuRow<u8>>();

impl<T> Borrow<CpuRow<T>> for [T] {
    fn borrow(&self) -> &CpuRow<T> {
        debug_assert_eq!(self.len(), CPU_WIDTH);
        // Safety: `CpuRow<T>` is `repr(C)` and holds exactly `CPU_WIDTH` fields of type `T`.
        unsafe { &*self.as_ptr().cast::<CpuRow<T>>() }
    }
}

/// Public values of the cpu: the state it starts from, then the state it leaves.
///
/// ```text
///     0 pc   1 clock   2 acc        entry
///     3 pc   4 clock   5 acc        exit
/// ```
const CPU_PUBLIC: usize = 6;

/// One image row, carrying every cell from its own onward.
///
/// Row `i` holds cell `i + j` in window slot `j`, so row zero holds the whole image.
///
/// Pinning row zero to the public image then pins every row's slot zero to its own cell.
#[repr(C)]
struct ImageRow<T> {
    /// The cell this row stands for.
    address: T,
    /// That cell, least significant bit first.
    address_bits: [T; ADDRESS_BITS],
    /// Whether this segment accesses the cell at all.
    touched: T,
    /// Inherited values of this cell and the ones after it.
    incoming: [T; CELLS],
    /// Handed-on values of this cell and the ones after it.
    outgoing: [T; CELLS],
}

const IMAGE_WIDTH: usize = size_of::<ImageRow<u8>>();

impl<T> Borrow<ImageRow<T>> for [T] {
    fn borrow(&self) -> &ImageRow<T> {
        debug_assert_eq!(self.len(), IMAGE_WIDTH);
        // Safety: `ImageRow<T>` is `repr(C)` and holds exactly `IMAGE_WIDTH` fields of type `T`.
        unsafe { &*self.as_ptr().cast::<ImageRow<T>>() }
    }
}

/// Public values of the image: the memory inherited, then the memory handed on.
const IMAGE_PUBLIC: usize = 2 * CELLS;

/// Columns of the bytecode fixed at setup: pc, opcode, address, immediate.
const BYTECODE_WIDTH: usize = 4;

/// The five tables of the machine, as one type so one proof can hold them all.
enum Chip {
    /// One step per row.
    Cpu,
    /// One instruction per row, fixed at setup, with a lookup multiplicity per proof.
    Bytecode,
    /// `LANES` instances of `out = a * b + c` per row.
    Macc,
    /// Memory sorted by cell then clock.
    Ram(Box<RamAir>),
    /// The memory a segment inherits and hands on.
    Image,
}

impl Chip {
    /// Every table, in the order proofs list them.
    fn all() -> Vec<Self> {
        let ram = RamAir::new(ram_statement()).expect("the machine's memory statement is valid");
        vec![
            Self::Cpu,
            Self::Bytecode,
            Self::Macc,
            Self::Ram(Box::new(ram)),
            Self::Image,
        ]
    }
}

/// Memory of one segment: one access per step, inheriting and handing on an image.
fn ram_statement() -> RamStatement {
    RamStatement {
        access_bus: ACCESS.into(),
        access_count: STEPS,
        address_bits: ADDRESS_BITS,
        timestamp_bits: CLOCK_BITS,
        value_width: 1,
        boundary: RamBoundary::Segment {
            incoming: INCOMING.into(),
            outgoing: OUTGOING.into(),
        },
    }
}

impl BaseAir<F> for Chip {
    fn width(&self) -> usize {
        match self {
            Self::Cpu => CPU_WIDTH,
            Self::Bytecode => 1,
            Self::Macc => LANES * LANE_WIDTH,
            Self::Ram(ram) => BaseAir::<F>::width(ram.as_ref()),
            Self::Image => IMAGE_WIDTH,
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Cpu => CPU_PUBLIC,
            Self::Image => IMAGE_PUBLIC,
            Self::Bytecode | Self::Macc | Self::Ram(_) => 0,
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Bytecode => BYTECODE_WIDTH,
            _ => 0,
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let Self::Bytecode = self else {
            return None;
        };
        let values = PROGRAM
            .iter()
            .enumerate()
            .flat_map(|(pc, &(opcode, address, immediate))| {
                [pc as u32, opcode, address, immediate].map(F::from_u32)
            })
            .collect();
        Some(RowMajorMatrix::new(values, BYTECODE_WIDTH))
    }
}

impl<AB> Air<AB> for Chip
where
    AB: InteractionBuilder<F = F> + BusInteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Cpu => eval_cpu(builder),
            Self::Bytecode => eval_bytecode(builder),
            Self::Macc => eval_macc(builder),
            Self::Ram(ram) => ram.as_ref().eval(builder),
            Self::Image => eval_image(builder),
        }
    }
}

/// Little-endian recomposition of bit columns.
fn from_bits<AB: AirBuilder>(bits: &[AB::Var]) -> AB::Expr {
    bits.iter()
        .rev()
        .fold(AB::Expr::ZERO, |acc, &bit| acc.double() + bit.into())
}

/// One step: decode through the bytecode, compute, touch memory, advance the state.
fn eval_cpu<AB: InteractionBuilder + BusInteractionBuilder>(builder: &mut AB) {
    let main = builder.main();
    let local: &CpuRow<AB::Var> = main.current_slice().borrow();
    let next: &CpuRow<AB::Var> = main.next_slice().borrow();
    let public: Vec<AB::Expr> = builder
        .public_values()
        .iter()
        .map(|&value| value.into())
        .collect();

    // Every bit column holds a bit, and exactly one opcode flag is set.
    for &bit in local.clock_bits.iter().chain(&local.address_bits) {
        builder.assert_bool(bit);
    }
    for flag in [local.is_load, local.is_store, local.is_macc] {
        builder.assert_bool(flag);
    }
    builder.assert_one(local.is_load + local.is_store + local.is_macc);
    builder.assert_eq(local.clock, from_bits::<AB>(&local.clock_bits));

    // LOAD takes the value read; STORE writes the accumulator and keeps it.
    builder.assert_zero(local.is_load * (local.next_acc - local.value));
    builder.assert_zero(local.is_store * (local.next_acc - local.acc));
    builder.assert_zero(local.is_store * (local.value - local.acc));

    // The state enters at the first row and leaves one step past the last.
    let mut first = builder.when_first_row();
    first.assert_eq(local.pc, public[0].clone());
    first.assert_eq(local.clock, public[1].clone());
    first.assert_eq(local.acc, public[2].clone());
    let mut last = builder.when_last_row();
    last.assert_eq(local.pc + AB::Expr::ONE, public[3].clone());
    last.assert_eq(local.clock + AB::Expr::ONE, public[4].clone());
    last.assert_eq(local.next_acc, public[5].clone());

    // One step moves the pc and the clock by one and carries the accumulator.
    let mut transition = builder.when_transition();
    transition.assert_eq(next.pc, local.pc + AB::Expr::ONE);
    transition.assert_eq(next.clock, local.clock + AB::Expr::ONE);
    transition.assert_eq(next.acc, local.next_acc);

    // The instruction at this pc is the one the bytecode holds.
    let opcode = local.is_store.into() + local.is_macc.into().double();
    let address = from_bits::<AB>(&local.address_bits);
    builder.push_interaction(
        BYTECODE,
        [local.pc.into(), opcode, address, local.immediate.into()],
        Count::bounded(AB::Expr::ONE, 1),
    );

    // A MACC step hands its arithmetic to the gadget table.
    builder.push_interaction(
        GADGET,
        [local.acc, local.value, local.immediate, local.next_acc],
        Count::bounded(local.is_macc.into(), 1),
    );

    // Every step makes one memory access, stamped with this step's clock.
    let access = core::iter::once(local.is_store)
        .chain(local.address_bits)
        .chain(local.clock_bits)
        .chain(core::iter::once(local.value))
        .map(Into::into);
    builder.push_bus_interaction(
        BusName::new(ACCESS),
        BusDirection::Push,
        access,
        BusActivation::Always,
    );
}

/// Serve each instruction as often as this proof executed it.
fn eval_bytecode<AB: InteractionBuilder>(builder: &mut AB) {
    // The instruction is fixed at setup; the multiplicity is how often this proof ran it.
    let instruction: Vec<AB::Expr> = builder
        .preprocessed()
        .current_slice()
        .iter()
        .map(|&cell| cell.into())
        .collect();
    let multiplicity: AB::Expr = builder.main().current_slice()[0].into();
    builder.push_interaction(BYTECODE, instruction, Count::provided(-multiplicity));
}

/// `LANES` instances of one dense gadget, each served on the gadget channel.
fn eval_macc<AB: InteractionBuilder>(builder: &mut AB) {
    let main = builder.main();
    let row: Vec<AB::Var> = main.current_slice().to_vec();
    for &[a, b, c, out, multiplicity] in row.as_chunks::<LANE_WIDTH>().0 {
        // Each lane is one instance of the same dense gadget.
        builder.assert_eq(out, a * b + c);
        builder.push_interaction(
            GADGET,
            [a, b, c, out],
            Count::provided(-multiplicity.into()),
        );
    }
}

/// Bind the public memory image to the tuples memory opens and closes each cell with.
fn eval_image<AB: BusInteractionBuilder>(builder: &mut AB) {
    let main = builder.main();
    let local: &ImageRow<AB::Var> = main.current_slice().borrow();
    let next: &ImageRow<AB::Var> = main.next_slice().borrow();
    let public: Vec<AB::Expr> = builder
        .public_values()
        .iter()
        .map(|&value| value.into())
        .collect();

    // Row i names cell i, in both forms.
    for &bit in &local.address_bits {
        builder.assert_bool(bit);
    }
    builder.assert_eq(local.address, from_bits::<AB>(&local.address_bits));
    builder.when_first_row().assert_zero(local.address);
    builder
        .when_transition()
        .assert_eq(next.address, local.address + AB::Expr::ONE);

    // Row zero is the public image, and each row after it shifts the window by one cell.
    let mut first = builder.when_first_row();
    for cell in 0..CELLS {
        first.assert_eq(local.incoming[cell], public[cell].clone());
        first.assert_eq(local.outgoing[cell], public[CELLS + cell].clone());
    }
    let mut transition = builder.when_transition();
    for slot in 0..CELLS - 1 {
        transition.assert_eq(next.incoming[slot], local.incoming[slot + 1]);
        transition.assert_eq(next.outgoing[slot], local.outgoing[slot + 1]);
    }

    // A cell this segment never touches passes through unchanged.
    builder.assert_zero(
        (AB::Expr::ONE - local.touched.into()) * (local.outgoing[0] - local.incoming[0]),
    );

    // A touched cell is handed to memory as it came in, and taken back as it ended.
    let entry = |value: AB::Var| {
        local
            .address_bits
            .iter()
            .copied()
            .chain([value])
            .map(Into::into)
    };
    builder.push_bus_interaction(
        BusName::new(INCOMING),
        BusDirection::Push,
        entry(local.incoming[0]),
        BusActivation::Boolean(local.touched.into()),
    );
    builder.push_bus_interaction(
        BusName::new(OUTGOING),
        BusDirection::Pull,
        entry(local.outgoing[0]),
        BusActivation::Boolean(local.touched.into()),
    );
}

/// What one segment's execution leaves on either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Boundary {
    /// Next instruction to run.
    pc: u32,
    /// Next clock reading.
    clock: u32,
    /// Accumulator.
    acc: F,
    /// Every memory cell.
    memory: [F; CELLS],
}

/// Everything a prover needs for one segment.
struct SegmentWitness {
    /// Main traces, row-major, in table order.
    traces: Vec<RowMajorMatrix<F>>,
    /// Public values, in table order.
    public_values: Vec<Vec<F>>,
    /// The boundary the segment leaves.
    exit: Boundary,
}

/// The machine before its first step.
fn initial_boundary() -> Boundary {
    Boundary {
        pc: 0,
        clock: 0,
        acc: F::ZERO,
        memory: INITIAL_MEMORY.map(F::from_u32),
    }
}

/// The low `count` bits of a value, least significant first.
fn bits(value: u32, count: usize) -> impl Iterator<Item = F> {
    (0..count).map(move |bit| F::from_bool((value >> bit) & 1 == 1))
}

/// Run `STEPS` instructions from a boundary and write down every table.
fn execute(entry: Boundary) -> SegmentWitness {
    let mut state = entry;
    let mut cpu = Vec::with_capacity(STEPS * CPU_WIDTH);
    let mut accesses = Vec::with_capacity(STEPS);
    let mut gadgets = Vec::new();
    let mut multiplicities = vec![F::ZERO; PROGRAM.len()];
    let mut touched = [false; CELLS];

    for _ in 0..STEPS {
        let (opcode, address, immediate) = PROGRAM[state.pc as usize];
        let cell = address as usize;
        let (value, next_acc) = match opcode {
            LOAD => (state.memory[cell], state.memory[cell]),
            STORE => (state.acc, state.acc),
            MACC => {
                let value = state.memory[cell];
                let out = state.acc * value + F::from_u32(immediate);
                gadgets.push([state.acc, value, F::from_u32(immediate), out]);
                (value, out)
            }
            _ => unreachable!("the program holds three opcodes"),
        };

        cpu.push(F::from_u32(state.pc));
        cpu.push(F::from_u32(state.clock));
        cpu.extend(bits(state.clock, CLOCK_BITS));
        cpu.extend([LOAD, STORE, MACC].map(|flag| F::from_bool(opcode == flag)));
        cpu.extend(bits(address, ADDRESS_BITS));
        cpu.extend([F::from_u32(immediate), state.acc, next_acc, value]);

        let time = u64::from(state.clock);
        accesses.push(if opcode == STORE {
            RamAccess::write(u64::from(address), time, [value])
        } else {
            RamAccess::read(u64::from(address), time, [value])
        });
        if opcode == STORE {
            state.memory[cell] = value;
        }

        multiplicities[state.pc as usize] += F::ONE;
        touched[cell] = true;
        state.pc += 1;
        state.clock += 1;
        state.acc = next_acc;
    }

    // Gadget slots fill row by row, and unused ones stay zero with a zero multiplicity.
    let mut macc = vec![F::ZERO; GADGET_ROWS * LANES * LANE_WIDTH];
    for (slot, gadget) in gadgets.iter().enumerate() {
        let lane = &mut macc[slot * LANE_WIDTH..(slot + 1) * LANE_WIDTH];
        lane[..4].copy_from_slice(gadget);
        lane[4] = F::ONE;
    }

    let ram = RamTrace::build(&ram_statement(), &accesses).expect("the execution is consistent");
    let ram_width = ram.width();

    let mut image = Vec::with_capacity(CELLS * IMAGE_WIDTH);
    for (cell, &touched) in touched.iter().enumerate() {
        image.push(F::from_usize(cell));
        image.extend(bits(cell as u32, ADDRESS_BITS));
        image.push(F::from_bool(touched));
        for side in [entry.memory, state.memory] {
            image.extend((0..CELLS).map(|slot| side.get(cell + slot).copied().unwrap_or(F::ZERO)));
        }
    }

    let cpu_public = [
        F::from_u32(entry.pc),
        F::from_u32(entry.clock),
        entry.acc,
        F::from_u32(state.pc),
        F::from_u32(state.clock),
        state.acc,
    ];
    let image_public = entry.memory.into_iter().chain(state.memory).collect();

    SegmentWitness {
        traces: vec![
            RowMajorMatrix::new(cpu, CPU_WIDTH),
            RowMajorMatrix::new(multiplicities, 1),
            RowMajorMatrix::new(macc, LANES * LANE_WIDTH),
            RowMajorMatrix::new(ram.into_values(), ram_width),
            RowMajorMatrix::new(image, IMAGE_WIDTH),
        ],
        public_values: vec![cpu_public.to_vec(), vec![], vec![], vec![], image_public],
        exit: state,
    }
}

/// The segment boundary: cpu state and memory image, entry side then exit side.
fn interface() -> SegmentInterface {
    let side = |cpu: usize, image: usize| {
        (cpu..cpu + 3)
            .map(|index| PublicSlot::new(CPU_TABLE, index))
            .chain((image..image + CELLS).map(|index| PublicSlot::new(IMAGE_TABLE, index)))
            .collect()
    };
    SegmentInterface::new(side(0, 0), side(3, CELLS))
}

/// The statement every segment is proved under, read off the tables themselves.
fn declaration(chips: &[Chip]) -> Declaration {
    let tables = chips
        .iter()
        .zip(LOG_HEIGHTS)
        .map(|(chip, log_height)| {
            TableDeclaration::from_constraints::<F, EF, Chip>(
                chip,
                HeightRange::exactly(log_height as u32),
            )
        })
        .collect();
    Declaration::new(Keccak256Hash, tables, PROOF_BUDGET, SECURITY_TARGET)
        .unwrap()
        .with_segment(interface())
        .unwrap()
}

/// The machine, its keys, and its statement, built once per test.
struct Machine {
    /// The tables, in proof order.
    chips: Vec<Chip>,
    /// Commitment schemes.
    config: MachineConfig,
    /// Key holding the committed bytecode.
    proving_key: ProvingKey<MachineConfig>,
    /// Key holding the bytecode commitment.
    verifying_key: VerifyingKey<MachineConfig>,
    /// The public statement, segment boundary included.
    declaration: Declaration,
}

impl Machine {
    /// Commit the bytecode and read the statement off the tables.
    fn new() -> Self {
        let chips = Chip::all();
        let config = config(&chips);
        let airs: Vec<&Chip> = chips.iter().collect();
        let (proving_key, verifying_key) = setup(&config, &airs, &mut challenger()).unwrap();
        let declaration = declaration(&chips);
        Self {
            chips,
            config,
            proving_key,
            verifying_key,
            declaration,
        }
    }

    /// Prove one segment and seal it under the statement.
    fn prove(&self, witness: &SegmentWitness) -> Vec<u8> {
        self.try_prove(witness).unwrap()
    }

    /// Prove one segment, reporting a witness the prover refuses.
    fn try_prove(
        &self,
        witness: &SegmentWitness,
    ) -> Result<Vec<u8>, ProvingError<PcsProverError<MachineConfig>>> {
        let instances = self
            .chips
            .iter()
            .zip(&witness.traces)
            .zip(&witness.public_values)
            .map(|((chip, trace), public)| {
                ProverInstance::new(
                    chip,
                    Table::new(trace.clone().transpose()),
                    &self.proving_key,
                    public,
                )
            })
            .collect();
        let proof = prove(
            &self.config,
            ProverInstances::new(instances),
            0,
            &mut challenger(),
        )?;
        let run = self.declaration.run(&LOG_HEIGHTS, 0).unwrap();
        Ok(self.declaration.seal(&run, &proof).unwrap().into_bytes())
    }

    /// Whether a witness yields a proof that verifies against its own public values.
    fn accepts(&self, witness: &SegmentWitness) -> bool {
        self.try_prove(witness)
            .is_ok_and(|bytes: Vec<u8>| self.verify(&bytes, &witness.public_values).is_ok())
    }

    /// Verify one sealed segment against public values, and return its boundary claim.
    fn verify(&self, bytes: &[u8], public_values: &[Vec<F>]) -> Verdict<SegmentClaim> {
        let run = self.declaration.run(&LOG_HEIGHTS, 0).unwrap();
        let instances = self
            .chips
            .iter()
            .zip(LOG_HEIGHTS)
            .zip(public_values)
            .map(|((chip, log_height), public)| {
                VerifierInstance::new(chip, &self.verifying_key, log_height, public)
            })
            .collect();
        self.declaration.verify_segment(
            &run,
            bytes,
            &self.config,
            VerifierInstances::new(instances),
            &mut challenger(),
        )
    }
}

/// Both segments of the execution, the second starting where the first stopped.
fn segments() -> [SegmentWitness; 2] {
    let first = execute(initial_boundary());
    let second = execute(first.exit);
    [first, second]
}

#[test]
fn the_execution_is_the_one_the_program_describes() {
    // Pin the witness itself, so every proof below proves the intended run.
    let [first, second] = segments();
    let memory = |cells: [u32; CELLS]| cells.map(F::from_u32);
    assert_eq!(
        first.exit,
        Boundary {
            pc: 4,
            clock: 4,
            acc: F::from_u32(120),
            memory: memory([3, 17, 7, 11]),
        }
    );
    assert_eq!(
        second.exit,
        Boundary {
            pc: 8,
            clock: 8,
            acc: F::from_u32(22_444),
            memory: memory([3, 2040, 7, 11]),
        }
    );
}

#[test]
fn two_segments_prove_verify_and_chain() {
    let machine = Machine::new();
    let segments = segments();

    let claims: Vec<SegmentClaim> = segments
        .iter()
        .map(|segment| {
            let bytes = machine.prove(segment);
            machine.verify(&bytes, &segment.public_values).unwrap()
        })
        .collect();

    // The prover could have predicted each claim before proving.
    for (claim, segment) in claims.iter().zip(&segments) {
        let public: Vec<&[F]> = segment.public_values.iter().map(Vec::as_slice).collect();
        assert_eq!(*claim, machine.declaration.segment_claim(&public).unwrap());
    }

    // The two segments join into one execution from the initial boundary to the final one.
    let execution = chain(&claims).unwrap();
    assert_eq!(execution.segments, 2);
    assert_eq!(execution.entry, claims[0].entry());
    assert_eq!(execution.exit, claims[1].exit());

    // Out of order, the second segment does not start where nothing stopped.
    assert_eq!(
        chain(&[claims[1], claims[0]]).unwrap_err(),
        ChainError::Broken { segment: 1 }
    );
}

#[test]
fn a_segment_that_skips_ahead_does_not_chain() {
    // The second segment is re-executed from a memory the first never left behind.
    let machine = Machine::new();
    let [first, _] = segments();
    let forged = execute(Boundary {
        memory: INITIAL_MEMORY.map(F::from_u32),
        ..first.exit
    });

    let claims = [&first, &forged].map(|segment| {
        let bytes = machine.prove(segment);
        machine.verify(&bytes, &segment.public_values).unwrap()
    });

    // Each proof is valid on its own, and only the chain notices the gap.
    assert_eq!(
        chain(&claims).unwrap_err(),
        ChainError::Broken { segment: 1 }
    );
}

#[test]
fn the_proof_shape_is_fixed_by_the_statement() {
    let machine = Machine::new();
    let run = machine.declaration.run(&LOG_HEIGHTS, 0).unwrap();
    let report = machine.declaration.cost_report::<EF>(&run).unwrap();

    // Committed cells, channel slots, live rounds, opened bytes, scratch bound, per table.
    let cost = |log_height: u32, committed, preprocessed, flushes, peak| TableCost {
        log_height,
        committed_cells: committed,
        preprocessed_cells: preprocessed,
        bus_flushes: flushes,
        sumcheck_rounds: log_height as usize,
        opening_bytes: (committed + preprocessed) >> log_height << 4,
        peak_temporary_bytes: peak,
    };
    let ram_width = BaseAir::<F>::width(&machine.chips[3]);
    assert_eq!(
        report.tables(),
        &[
            // cpu: 14 columns; two lookup flushes and one bus declaration per row.
            cost(2, 14 * 4, 0, 3 * 4, (14 * 2 + 2 * 2 * 4 + 4) * 16),
            // bytecode: one multiplicity column over four fixed ones; one lookup flush.
            cost(3, 8, 4 * 8, 8, (5 * 4 + 2 * 8) * 16),
            // macc: two lanes of five columns; one lookup flush per lane.
            cost(2, 10 * 4, 0, 2 * 4, (10 * 2 + 2 * 2 * 4) * 16),
            // ram: one access pull and two image declarations per row.
            cost(2, ram_width * 4, 0, 3 * 4, (ram_width * 2 + 3 * 4) * 16),
            // image: two image declarations per row.
            cost(2, IMAGE_WIDTH * 4, 0, 2 * 4, (IMAGE_WIDTH * 2 + 2 * 4) * 16),
        ]
    );
    assert_eq!(report.total().sumcheck_rounds, 3);

    // Proving the same segment twice yields the same bytes.
    let [first, _] = segments();
    let bytes = machine.prove(&first);
    assert_eq!(bytes, machine.prove(&first));
    assert!(bytes.len() <= PROOF_BUDGET);
}

#[test]
fn a_tampered_boundary_is_refused() {
    let machine = Machine::new();
    let [first, _] = segments();
    let bytes = machine.prove(&first);

    // Each public value of the boundary, nudged by one, breaks verification.
    for (table, index) in [
        (CPU_TABLE, 0),
        (CPU_TABLE, 5),
        (IMAGE_TABLE, 1),
        (IMAGE_TABLE, 6),
    ] {
        let mut public = first.public_values.clone();
        public[table][index] += F::ONE;
        assert!(
            matches!(
                machine.verify(&bytes, &public),
                Err(SealedVerificationError::Verification(_))
            ),
            "public value {index} of table {table} was not bound"
        );
    }
}

#[test]
fn a_malformed_proof_is_refused() {
    let machine = Machine::new();
    let [first, second] = segments();
    let bytes = machine.prove(&first);

    // A flipped body byte fails the decoder or the transcript, never passes.
    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 1;
    assert!(machine.verify(&flipped, &first.public_values).is_err());

    // A truncated input names what is missing.
    assert!(matches!(
        machine.verify(&bytes[..bytes.len() - 1], &first.public_values),
        Err(SealedVerificationError::Envelope(
            EnvelopeError::Truncated { .. }
        ))
    ));

    // The first segment's proof does not prove the second segment's boundary.
    assert!(matches!(
        machine.verify(&bytes, &second.public_values),
        Err(SealedVerificationError::Verification(_))
    ));
}

/// Overwrite one cell of one side of a segment's image, in its public values and its trace.
///
/// The trace keeps the window shape, so only the claim itself is false.
fn misstate_image(witness: &mut SegmentWitness, outgoing: bool, cell: usize, value: F) {
    let side = usize::from(outgoing) * CELLS;
    witness.public_values[IMAGE_TABLE][side + cell] = value;
    // Cell `c` sits in window slot `c - r` of every row `r <= c`.
    let image = &mut witness.traces[IMAGE_TABLE];
    for row in 0..=cell {
        image.values[row * IMAGE_WIDTH + 4 + side + cell - row] = value;
    }
}

#[test]
fn an_image_that_misstates_memory_is_refused() {
    let machine = Machine::new();
    let [honest, _] = segments();
    assert!(machine.accepts(&honest));

    // Cell 0 is read in this segment, so claiming it held 4 unbalances the inherited image.
    let mut inherited = execute(initial_boundary());
    misstate_image(&mut inherited, false, 0, F::from_u32(4));
    assert!(!machine.accepts(&inherited));

    // Cell 1 is written, so claiming it ends at 18 unbalances the handed-on image.
    let mut handed_on = execute(initial_boundary());
    misstate_image(&mut handed_on, true, 1, F::from_u32(18));
    assert!(!machine.accepts(&handed_on));

    // Cell 3 is never touched, so it must leave as it came.
    let mut untouched = execute(initial_boundary());
    misstate_image(&mut untouched, true, 3, F::from_u32(12));
    assert!(!machine.accepts(&untouched));
}
