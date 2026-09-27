//! A small integer machine, proved end to end over GF(2^128) with the binary backend.
//!
//! It runs real 64-bit integer instructions, through public APIs only.
//!
//! Over GF(2), field addition is XOR, so every integer operation is a Boolean circuit:
//!
//! ```text
//!     ALU     add, addi, sub, slt, beq, bne    one ripple-carry adder, 63 carries
//!     MUL     mul (low 64 bits)                array multiplier, one ripple adder per bit of b
//!     LOAD    ld                               aligned 64-bit read of memory
//!     STORE   sd                               aligned 64-bit write of memory
//! ```
//!
//! Each instruction table is mixed: a bit region for its circuit, a dense region for its plumbing.
//!
//! ```text
//!     bits    op flags, write flag, a, b, circuit bits
//!     dense   clock, pc, program read, rd, rs1, rs2, imm, target, extras, three memory accesses
//! ```
//!
//! The tables meet on buses, so their rows come in no particular order:
//!
//! ```text
//!     state       (clock, pc)              each row pulls its state and pushes the next one
//!     program     read-only lookup by pc   fixed at setup, a preprocessed table
//!     registers   timestamped memory       32 cells, zero seed, x0 never written
//!     ram         timestamped memory       8 cells, seeded from a public image
//!     clock-low   range table              shared by both memories
//!     clock-high  range table              shared by both memories
//! ```
//!
//! Register and memory values cross the buses as word views of their bits.
//!
//! The io table starts the machine at pc 0 and stops it at the halt pc.
//!
//! It reads a0 after the last step and pins it to the one public value.

use std::sync::LazyLock;
use std::time::Instant;

use p3_air::symbolic::AirLayout;
use p3_air::utils::word_view;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess, check_all_constraints};
use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
use p3_binary_pcs::{
    BinaryPcsConfig, BinaryPcsParams, BinaryPcsProverData, BooleanTracePcs, GroupedCodewordMmcs,
    MixedTraceCommitment, MixedTraceData, MixedTracePcs, committed_shapes, coordinate_basis,
};
use p3_bus::{
    BusActivation, BusDebugInstance, BusDebugReport, BusDirection, BusInteractionBuilder, BusName,
    BusSymbolicBuilder, ClockGap, ClockRangeAir, PublicImage, RangeRead, ReadOnlyMemoryBus,
    ReadOnlyMemoryInteractionBuilder, TimestampedAccess, TimestampedBoundaryAir, TimestampedMemory,
    TimestampedMemoryInteractionBuilder, TimestampedSeed,
};
use p3_challenger::HashChallenger;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::{MultiStarkConfig, PcsError};
use p3_multi_stark::contract::HeightRange;
use p3_multi_stark::{
    MachineDeclaration, MultiStarkProof, ProverInstance, ProverInstances, ProvingKey,
    TableDeclaration, VerificationError, VerifierInstance, VerifierInstances, VerifyingKey,
    prove_with_security, setup, verify_with_security,
};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

type F = BinaryField128;
type C = BinaryField64;
type Memory = TimestampedMemory<C, F>;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MerkleMmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Mmcs = GroupedCodewordMmcs<MerkleMmcs>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type Pcs = MixedTracePcs<F, Mmcs, Mmcs>;
type Proof = MultiStarkProof<MachineConfig>;
type Declaration = MachineDeclaration<Keccak256Hash>;

/// Bits in a machine word.
const WORD: usize = 64;

/// Memory accesses one row makes, so the clock advances by `g^3` per step.
const SLOTS: u64 = 3;

/// Registers, one cell each.
const REGISTERS: usize = 32;

/// Base-two logarithm of the ram size in 64-bit cells.
const LOG_RAM_CELLS: usize = 3;

/// Base-two logarithm of the program table height.
const LOG_PROGRAM: usize = 4;

/// Program slot every padding row reads: an empty entry no real row can execute.
const PADDING_PC: usize = (1 << LOG_PROGRAM) - 1;

/// Fields of one program entry: pc, op, rd, rs1, rs2, imm, target, write.
const ENTRY_WIDTH: usize = 8;

/// Coordinates one committed element absorbs: `2^7 = 128`.
const ABSORBED: usize = 7;

/// Security target every proof is graded against.
const SECURITY_BITS: usize = 100;

/// Largest sealed proof the statement accepts, in bytes.
const PROOF_BUDGET: usize = 1 << 24;

/// Bus carrying the machine state `(clock, pc)`.
const STATE: &str = "state";

/// Register numbers the example program uses.
const ZERO: usize = 0;
const A0: usize = 10;
const T0: usize = 5;
const T1: usize = 6;
const T2: usize = 7;
const T3: usize = 28;
const T4: usize = 29;

/// Ram cell holding the public input, at byte address 8.
const INPUT_CELL: usize = 1;

/// The public input: the factor of every product.
const INPUT: u64 = 3;

/// Coordinate basis of GF(2^128): bit `k` of a cell is its coordinate on `e_k`.
static BASIS: LazyLock<Vec<F>> = LazyLock::new(coordinate_basis::<F>);

/// Operations, numbered by the basis element that names them in the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Slt,
    Beq,
    Bne,
    Mul,
    Ld,
    Sd,
}

impl Op {
    /// Index of the basis element naming this operation.
    const fn code(self) -> usize {
        self as usize
    }

    /// Table that executes this operation.
    const fn kind(self) -> Kind {
        match self {
            Self::Add | Self::Sub | Self::Slt | Self::Beq | Self::Bne => Kind::Alu,
            Self::Mul => Kind::Mul,
            Self::Ld => Kind::Load,
            Self::Sd => Kind::Store,
        }
    }

    /// Whether this operation writes `rd`.
    const fn writes(self) -> bool {
        !matches!(self, Self::Beq | Self::Bne | Self::Sd)
    }
}

/// One instruction.
///
/// An immediate form reads `rs2 = x0`, and a register form has `imm = 0`.
///
/// So the second operand is always `x[rs2] + imm`, and in GF(2) that sum is an XOR.
#[derive(Clone, Copy, Debug)]
struct Instr {
    op: Op,
    rd: usize,
    rs1: usize,
    rs2: usize,
    imm: u64,
    /// Branch target, read only by `beq` and `bne`.
    target: usize,
}

impl Instr {
    /// An instruction from its fields.
    const fn new(op: Op, rd: usize, rs1: usize, rs2: usize, imm: u64, target: usize) -> Self {
        Self {
            op,
            rd,
            rs1,
            rs2,
            imm,
            target,
        }
    }

    /// Whether this instruction writes a register: x0 is never written.
    const fn write(&self) -> bool {
        self.op.writes() && self.rd != ZERO
    }
}

/// `rd = imm`.
const fn li(rd: usize, imm: u64) -> Instr {
    Instr::new(Op::Add, rd, ZERO, ZERO, imm, 0)
}

/// `rd = rs1 + imm`.
const fn addi(rd: usize, rs1: usize, imm: u64) -> Instr {
    Instr::new(Op::Add, rd, rs1, ZERO, imm, 0)
}

/// `rd = rs1 + rs2`.
const fn add(rd: usize, rs1: usize, rs2: usize) -> Instr {
    Instr::new(Op::Add, rd, rs1, rs2, 0, 0)
}

/// `rd = rs1 - rs2`.
const fn sub(rd: usize, rs1: usize, rs2: usize) -> Instr {
    Instr::new(Op::Sub, rd, rs1, rs2, 0, 0)
}

/// `rd = (rs1 < rs2)`, signed.
const fn slt(rd: usize, rs1: usize, rs2: usize) -> Instr {
    Instr::new(Op::Slt, rd, rs1, rs2, 0, 0)
}

/// Jump to `target` when `rs1 = rs2`.
const fn beq(rs1: usize, rs2: usize, target: usize) -> Instr {
    Instr::new(Op::Beq, ZERO, rs1, rs2, 0, target)
}

/// `rd = rs1 * rs2 mod 2^64`.
const fn mul(rd: usize, rs1: usize, rs2: usize) -> Instr {
    Instr::new(Op::Mul, rd, rs1, rs2, 0, 0)
}

/// Jump to `target` when `rs1 != rs2`.
const fn bne(rs1: usize, rs2: usize, target: usize) -> Instr {
    Instr::new(Op::Bne, ZERO, rs1, rs2, 0, target)
}

/// `rd = ram[rs1 + imm]`, 64-bit aligned.
const fn ld(rd: usize, imm: u64, rs1: usize) -> Instr {
    Instr::new(Op::Ld, rd, rs1, ZERO, imm, 0)
}

/// `ram[rs1 + imm] = rs2`, 64-bit aligned.
const fn sd(rs2: usize, imm: u64, rs1: usize) -> Instr {
    Instr::new(Op::Sd, ZERO, rs1, rs2, imm, 0)
}

/// Sums `i * input` for `i = 1..=iterations`, where the input is read from ram.
const fn program(iterations: u64) -> [Instr; 10] {
    [
        li(T0, 0),              // 0: acc = 0
        li(T1, 1),              // 1: i = 1
        li(T2, iterations + 1), // 2: bound
        ld(T3, 8, ZERO),        // 3: t3 = ram[8], the public input
        mul(T4, T1, T3),        // 4: loop: t4 = i * t3
        add(T0, T0, T4),        // 5: acc += t4
        addi(T1, T1, 1),        // 6: i += 1
        bne(T1, T2, 4),         // 7: until i = bound
        sd(T0, 0, ZERO),        // 8: ram[0] = acc
        ld(A0, 0, ZERO),        // 9: a0 = ram[0], the output
    ]
}

/// The example: `sum_{i=1..4} i * 3 = 30`.
const EXAMPLE: [Instr; 10] = program(4);

/// The other ALU operations: a signed compare feeding both branch outcomes.
const ALU_CHECK: [Instr; 9] = [
    li(T0, 7),              // 0: t0 = 7
    li(T1, (-2i64) as u64), // 1: t1 = -2
    sub(T2, T0, T1),        // 2: t2 = 9
    slt(T3, T1, T0),        // 3: t3 = (-2 < 7) = 1
    beq(T3, ZERO, 7),       // 4: not taken
    slt(T4, T0, T1),        // 5: t4 = (7 < -2) = 0
    beq(T4, ZERO, 8),       // 6: taken
    li(T2, 0),              // 7: skipped
    add(A0, T2, T3),        // 8: a0 = 10
];

/// One table executing one family of operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Alu,
    Mul,
    Load,
    Store,
}

/// Dense plumbing columns, relative to the start of the dense region.
const CLOCK: usize = 0;
const PC: usize = 1;
const COUNT: usize = 2;
const COUNT_INVERSE: usize = 3;
const RD: usize = 4;
const RS1: usize = 5;
const RS2: usize = 6;
const IMM: usize = 7;
const TARGET: usize = 8;
const PLUMBING: usize = 9;

/// Columns of one access: the previous time, then the low and high range reads.
const ACCESS_WIDTH: usize = 7;

/// Program entry fields, in lookup order.
const ENTRY_PC: usize = 0;
const ENTRY_RD: usize = 2;
const ENTRY_WRITE: usize = 7;

impl Kind {
    /// The four instruction tables, in proof order.
    const ALL: [Self; 4] = [Self::Alu, Self::Mul, Self::Load, Self::Store];

    /// Code of the first operation this table executes.
    const fn first_op(self) -> usize {
        match self {
            Self::Alu => Op::Add.code(),
            Self::Mul => Op::Mul.code(),
            Self::Load => Op::Ld.code(),
            Self::Store => Op::Sd.code(),
        }
    }

    /// One flag per operation: exactly one is set on a real row, none on padding.
    const fn flags(self) -> usize {
        match self {
            Self::Alu => 5,
            _ => 1,
        }
    }

    /// Circuit bits past the two operands.
    const fn body_bits(self) -> usize {
        match self {
            // sum, carries, lt, eq, taken
            Self::Alu => WORD + (WORD - 1) + 3,
            // a_i * b_0, then one adder per later bit of b
            Self::Mul => WORD + mul_step(WORD),
            // address = rs1 + imm
            Self::Load | Self::Store => WORD + (WORD - 1),
        }
    }

    /// Extra dense columns: old and new values that are not bits.
    const fn extras(self) -> usize {
        match self {
            // rd_old, eq_inverse
            Self::Alu => 2,
            // rd_old
            Self::Mul => 1,
            // value, then rd_old or ram_old
            Self::Load | Self::Store => 2,
        }
    }

    /// Column of the write flag, after the op flags.
    const fn write(self) -> usize {
        self.flags()
    }

    /// First bit of the operand `a`.
    const fn a(self) -> usize {
        self.flags() + 1
    }

    /// First bit of the operand `b`.
    const fn b(self) -> usize {
        self.a() + WORD
    }

    /// First bit of the circuit.
    const fn body(self) -> usize {
        self.b() + WORD
    }

    /// Width of the bit region, which is also the first dense column.
    const fn bits(self) -> usize {
        self.body() + self.body_bits()
    }

    /// Column of extra dense value `index`.
    const fn extra(self, index: usize) -> usize {
        self.bits() + PLUMBING + index
    }

    /// First column of access `index`, which is also its slot.
    const fn access(self, index: usize) -> usize {
        self.bits() + PLUMBING + self.extras() + index * ACCESS_WIDTH
    }

    /// Columns of one row.
    const fn width(self) -> usize {
        self.access(3)
    }

    /// Name in a diagnosis.
    const fn name(self) -> &'static str {
        match self {
            Self::Alu => "alu",
            Self::Mul => "mul",
            Self::Load => "load",
            Self::Store => "store",
        }
    }
}

/// ALU circuit columns, relative to the body.
const ALU_SUM: usize = 0;
const ALU_CARRY: usize = WORD;
const ALU_LT: usize = 2 * WORD - 1;
const ALU_EQ: usize = ALU_LT + 1;
const ALU_TAKEN: usize = ALU_EQ + 1;

/// Bits the multiplier's steps `1..step` hold, relative to the end of `a_i * b_0`.
///
/// Step `j` adds `a * b_j` into bits `j..64`: `64 - j` sum bits and `63 - j` carries.
const fn mul_step(step: usize) -> usize {
    let mut offset = 0;
    let mut j = 1;
    while j < step {
        offset += 2 * (WORD - j) - 1;
        j += 1;
    }
    offset
}

/// Io columns: the active flag (the one bit), the end clock, the a0 read, its value.
const IO_ACTIVE: usize = 0;
const IO_END: usize = 1;
const IO_ACCESS: usize = 2;
const IO_VALUE: usize = IO_ACCESS + ACCESS_WIDTH;
const IO_WIDTH: usize = IO_VALUE + 1;

/// Table positions.
const IO: usize = 4;
const PROGRAM: usize = 5;
const RAM: usize = 7;

/// The register file: 32 cells of one word.
fn registers() -> Memory {
    Memory::new("registers", "clock-low", "clock-high", 1).unwrap()
}

/// The ram: shares its range tables with the registers.
fn ram() -> Memory {
    Memory::new("ram", "clock-low", "clock-high", 1).unwrap()
}

/// The program lookup.
fn program_bus() -> ReadOnlyMemoryBus<F> {
    ReadOnlyMemoryBus::new("program").unwrap()
}

/// The public input: cell 1 holds 3.
fn image() -> PublicImage<F> {
    PublicImage::new(&ram(), LOG_RAM_CELLS, vec![(INPUT_CELL, vec![word(INPUT)])]).unwrap()
}

/// A 64-bit word as the field element whose low coordinates are its bits.
fn word(x: u64) -> F {
    word_view(&bits(x), &BASIS[..WORD])
}

/// The bits of a word, least significant first.
fn bits(x: u64) -> Vec<F> {
    (0..WORD).map(|i| F::from_bool((x >> i) & 1 == 1)).collect()
}

/// Address of program slot `pc`: `G^pc`, so `pc + 1` is one multiplication.
fn pc_address(pc: usize) -> F {
    F::GENERATOR.exp_u64(pc as u64)
}

/// Address of register `r`.
fn register(r: usize) -> F {
    TimestampedBoundaryAir::<C, F>::cell_address(r)
}

/// The clock at exponent `exponent`: `g^exponent`.
fn time(exponent: u64) -> F {
    Memory::tick().exp_u64(exponent)
}

/// One program slot as the lookup serves it.
///
/// A slot past the program is empty: op zero, registers x0, write zero.
fn entry(program: &[Instr], pc: usize) -> [F; ENTRY_WIDTH] {
    let Some(instr) = program.get(pc) else {
        return [
            pc_address(pc),
            F::ZERO,
            register(ZERO),
            register(ZERO),
            register(ZERO),
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];
    };
    let target = if matches!(instr.op, Op::Beq | Op::Bne) {
        pc_address(instr.target)
    } else {
        F::ZERO
    };
    [
        pc_address(pc),
        BASIS[instr.op.code()],
        register(instr.rd),
        register(instr.rs1),
        register(instr.rs2),
        word(instr.imm),
        target,
        F::from_bool(instr.write()),
    ]
}

/// Constrains `sum = x + y + carry_in` by a ripple-carry chain.
///
/// ```text
///     sum_i     = x_i + y_i + c_i
///     c_{i+1}   = x_i*y_i + x_i*c_i + y_i*c_i        majority, in characteristic two
/// ```
///
/// The carry out of the top bit is dropped: the sum is taken mod `2^n`.
fn ripple<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    x: &[AB::Expr],
    y: &[AB::Expr],
    carry_in: AB::Expr,
    sum: &[AB::Var],
    carries: &[AB::Var],
) {
    let mut c = carry_in;
    for i in 0..sum.len() {
        builder.assert_eq(sum[i], x[i].clone() + y[i].clone() + c.clone());
        if i + 1 < sum.len() {
            let majority =
                x[i].clone() * y[i].clone() + x[i].clone() * c.clone() + y[i].clone() * c;
            builder.assert_eq(carries[i], majority);
            c = carries[i].into();
        }
    }
}

/// Witness of [`ripple`]: the sum bits and the carries into bits `1..n`.
fn ripple_bits(x: &[bool], y: &[bool], carry_in: bool) -> (Vec<bool>, Vec<bool>) {
    let mut c = carry_in;
    let mut sum = Vec::with_capacity(x.len());
    let mut carries = Vec::with_capacity(x.len());
    for (&x, &y) in x.iter().zip(y) {
        sum.push(x ^ y ^ c);
        c = (x & y) | (x & c) | (y & c);
        carries.push(c);
    }
    carries.pop();
    (sum, carries)
}

/// The bits of a word, least significant first, as booleans.
fn bools(x: u64) -> Vec<bool> {
    (0..WORD).map(|i| (x >> i) & 1 == 1).collect()
}

/// One instruction table.
#[derive(Clone, Debug)]
struct InstructionAir {
    kind: Kind,
    registers: Memory,
    ram: Memory,
    program: ReadOnlyMemoryBus<F>,
}

/// Starts the machine, stops it, and reads the output.
#[derive(Clone, Debug)]
struct IoAir {
    registers: Memory,
    /// The pc after the last instruction.
    halt: usize,
}

/// The program, fixed at setup and served by pc.
#[derive(Clone, Debug)]
struct ProgramAir {
    program: ReadOnlyMemoryBus<F>,
    entries: Vec<[F; ENTRY_WIDTH]>,
}

/// Every table of the machine, in proof order.
#[derive(Clone, Debug)]
enum Chip {
    Instruction(InstructionAir),
    Io(IoAir),
    Program(ProgramAir),
    Boundary(TimestampedBoundaryAir<C, F>),
    Range(ClockRangeAir<C, F>),
}

impl Chip {
    /// The nine tables running `program`.
    fn all(program: &[Instr]) -> Vec<Self> {
        let instruction = |kind| {
            Self::Instruction(InstructionAir {
                kind,
                registers: registers(),
                ram: ram(),
                program: program_bus(),
            })
        };
        let entries = (0..1 << LOG_PROGRAM).map(|pc| entry(program, pc)).collect();
        let mut chips: Vec<Self> = Kind::ALL.into_iter().map(instruction).collect();
        chips.extend([
            Self::Io(IoAir {
                registers: registers(),
                halt: program.len(),
            }),
            Self::Program(ProgramAir {
                program: program_bus(),
                entries,
            }),
            Self::Boundary(
                TimestampedBoundaryAir::new(registers(), TimestampedSeed::Zero).unwrap(),
            ),
            Self::Boundary(
                TimestampedBoundaryAir::new(ram(), TimestampedSeed::Public(image())).unwrap(),
            ),
            Self::Range(ClockRangeAir::new(registers())),
        ]);
        chips
    }

    /// Name in a diagnosis, given the table's position.
    const fn name(&self, index: usize) -> &'static str {
        match self {
            Self::Instruction(air) => air.kind.name(),
            Self::Io(_) => "io",
            Self::Program(_) => "program",
            Self::Boundary(_) if index == RAM => "ram-boundary",
            Self::Boundary(_) => "register-boundary",
            Self::Range(_) => "range",
        }
    }
}

impl BaseAir<F> for Chip {
    fn width(&self) -> usize {
        match self {
            Self::Instruction(air) => air.kind.width(),
            Self::Io(_) => IO_WIDTH,
            Self::Program(_) => 1,
            Self::Boundary(air) => air.width(),
            Self::Range(air) => BaseAir::<F>::width(air),
        }
    }

    fn boolean_columns(&self) -> usize {
        match self {
            Self::Instruction(air) => air.kind.bits(),
            Self::Io(_) => 1,
            _ => 0,
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Instruction(_) | Self::Program(_) => Vec::new(),
            Self::Io(_) => vec![IO_ACTIVE],
            Self::Boundary(air) => air.main_next_row_columns(),
            Self::Range(air) => BaseAir::<F>::main_next_row_columns(air),
        }
    }

    fn num_public_values(&self) -> usize {
        match self {
            Self::Io(_) => 1,
            _ => 0,
        }
    }

    fn preprocessed_width(&self) -> usize {
        match self {
            Self::Program(_) => ENTRY_WIDTH,
            _ => 0,
        }
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        match self {
            Self::Program(_) => vec![ENTRY_PC],
            _ => Vec::new(),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let Self::Program(air) = self else {
            return None;
        };
        Some(RowMajorMatrix::new(air.entries.concat(), ENTRY_WIDTH))
    }

    fn num_periodic_columns(&self) -> usize {
        match self {
            Self::Boundary(air) => air.num_periodic_columns(),
            _ => 0,
        }
    }

    fn periodic_columns(&self) -> std::borrow::Cow<'_, [Vec<F>]> {
        match self {
            Self::Boundary(air) => air.periodic_columns(),
            _ => std::borrow::Cow::Borrowed(&[]),
        }
    }

    fn periodic_periods(&self) -> Vec<usize> {
        match self {
            Self::Boundary(air) => air.periodic_periods(),
            _ => Vec::new(),
        }
    }

    fn periodic_evaluations<EF: ExtensionField<F>>(&self, point: &[EF]) -> Option<Vec<EF>> {
        match self {
            Self::Boundary(air) => air.periodic_evaluations(point),
            _ => None,
        }
    }
}

impl<AB: BusInteractionBuilder<F = F>> Air<AB> for Chip {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Instruction(air) => air.eval(builder),
            Self::Io(air) => air.eval(builder),
            Self::Program(air) => air.eval(builder),
            Self::Boundary(air) => air.eval(builder),
            Self::Range(air) => air.eval(builder),
        }
    }
}

/// One access `(address, old -> new)` declared from the seven columns at `at`.
fn access<AB: BusInteractionBuilder<F = F>>(
    builder: &mut AB,
    memory: &Memory,
    row: &[AB::Var],
    at: usize,
    (clock, slot): (AB::Expr, usize),
    (address, old, new): (AB::Expr, AB::Expr, AB::Expr),
) {
    let read = |offset: usize| RangeRead {
        value: row[at + offset].into(),
        count: row[at + offset + 1].into(),
        count_inverse: row[at + offset + 2].into(),
    };
    let access = TimestampedAccess {
        address,
        previous: row[at].into(),
        old: vec![old],
        new: vec![new],
        gap: ClockGap {
            low: read(1),
            high: read(4),
        },
    };
    builder.timestamped_access(memory, clock, slot, access);
}

/// The state step: `g^3`, one tick per access slot.
fn step() -> F {
    Memory::tick().exp_u64(SLOTS)
}

impl InstructionAir {
    /// Fetch, execute, touch memory, and advance the state.
    fn eval<AB: BusInteractionBuilder<F = F>>(&self, builder: &mut AB) {
        let kind = self.kind;
        let row: Vec<AB::Var> = builder.main().current_slice().to_vec();
        let e = |column: usize| -> AB::Expr { row[column].into() };
        let d = kind.bits();
        let flags = &row[..kind.flags()];
        let a = &row[kind.a()..kind.b()];
        let b = &row[kind.b()..kind.body()];
        let body = kind.body();
        let (clock, pc) = (e(d + CLOCK), e(d + PC));
        let write = e(kind.write());
        let word_of = |bits: &[AB::Var]| word_view::<AB::Expr, F, _>(bits, &BASIS[..bits.len()]);

        // The program names the op by a basis element, so the flags read it back one-hot.
        let op = word_view::<AB::Expr, F, _>(flags, &BASIS[kind.first_op()..][..kind.flags()]);
        // A one-hot flag vector sums to one; padding reads an empty entry and sums to zero.
        let real = flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag.into());

        // Padding sits at clock zero, where its accesses cancel among themselves.
        builder.assert_zero((AB::Expr::ONE + real.clone()) * clock.clone());

        // Fetch: the entry at pc, fields in lookup order.
        let fields = [
            op,
            e(d + RD),
            e(d + RS1),
            e(d + RS2),
            e(d + IMM),
            e(d + TARGET),
            write.clone(),
        ];
        builder.read_only_memory(
            &self.program,
            pc.clone(),
            e(d + COUNT),
            e(d + COUNT_INVERSE),
            fields,
        );

        // Execute: the circuit gives the next pc and the three accesses.
        let touch = |builder: &mut AB, memory: &Memory, k: usize, values| {
            access(
                builder,
                memory,
                &row,
                kind.access(k),
                (clock.clone(), k),
                values,
            );
        };
        let next_pc = match kind {
            Kind::Alu => {
                let [add, sub, slt, beq, bne] = [0, 1, 2, 3, 4].map(&e);
                let sum = &row[body + ALU_SUM..body + ALU_CARRY];
                let carries = &row[body + ALU_CARRY..body + ALU_LT];
                let [lt, eq, taken] = [ALU_LT, ALU_EQ, ALU_TAKEN].map(|i| e(body + i));
                let (rd_old, eq_inverse) = (e(kind.extra(0)), e(kind.extra(1)));

                // Subtraction and comparisons compute `a + !b + 1`.
                let invert = sub.clone() + slt.clone() + beq.clone() + bne.clone();
                let x: Vec<AB::Expr> = a.iter().map(|&bit| bit.into()).collect();
                let y: Vec<AB::Expr> = b.iter().map(|&bit| bit.into() + invert.clone()).collect();
                ripple(builder, &x, &y, invert, sum, carries);

                // Signed less-than: the sign of `a - b`, unless the signs of `a` and `b` differ.
                let (a63, b63, s63) = (e(kind.a() + 63), e(kind.b() + 63), e(body + 63));
                builder.assert_eq(lt.clone(), s63.clone() + (a63.clone() + b63) * (a63 + s63));

                // Zero test of the difference `D`: `D * inv = 1 - eq` and `D * eq = 0`.
                let difference = word_of(sum);
                builder.assert_eq(difference.clone() * eq_inverse, AB::Expr::ONE + eq.clone());
                builder.assert_zero(difference.clone() * eq.clone());
                builder.assert_eq(taken.clone(), beq * eq.clone() + bne * (AB::Expr::ONE + eq));

                // The result, written only when the entry says so.
                let result = (add + sub) * difference + slt * lt * BASIS[0];
                let new = write.clone() * result + (AB::Expr::ONE + write) * rd_old.clone();
                touch(
                    builder,
                    &self.registers,
                    0,
                    (e(d + RS1), word_of(a), word_of(a)),
                );
                let second = word_of(b) + e(d + IMM);
                touch(
                    builder,
                    &self.registers,
                    1,
                    (e(d + RS2), second.clone(), second),
                );
                touch(builder, &self.registers, 2, (e(d + RD), rd_old, new));

                // A taken branch jumps to the entry's target.
                let fallthrough = pc * pc_address(1);
                fallthrough.clone() + taken * (e(d + TARGET) + fallthrough)
            }
            Kind::Mul => {
                let rd_old = e(kind.extra(0));

                // Row zero of the product: `a_i * b_0`.
                let first = &row[body..body + WORD];
                for (i, &bit) in first.iter().enumerate() {
                    builder.assert_eq(bit, e(kind.a() + i) * e(kind.b()));
                }

                // Step `j` adds `a * b_j` into bits `j..64`; bit `j` is final after it.
                let mut acc: Vec<AB::Expr> = first[1..].iter().map(|&bit| bit.into()).collect();
                let mut out = vec![e(body)];
                for j in 1..WORD {
                    let at = body + WORD + mul_step(j);
                    let sum = &row[at..at + WORD - j];
                    let carries = &row[at + WORD - j..at + 2 * (WORD - j) - 1];
                    let y: Vec<AB::Expr> = (j..WORD)
                        .map(|i| e(kind.a() + i - j) * e(kind.b() + j))
                        .collect();
                    ripple(builder, &acc, &y, AB::Expr::ZERO, sum, carries);
                    out.push(sum[0].into());
                    acc = sum[1..].iter().map(|&bit| bit.into()).collect();
                }

                let product = word_view::<AB::Expr, F, _>(&out, &BASIS[..WORD]);
                let new = write.clone() * product + (AB::Expr::ONE + write) * rd_old.clone();
                touch(
                    builder,
                    &self.registers,
                    0,
                    (e(d + RS1), word_of(a), word_of(a)),
                );
                touch(
                    builder,
                    &self.registers,
                    1,
                    (e(d + RS2), word_of(b), word_of(b)),
                );
                touch(builder, &self.registers, 2, (e(d + RD), rd_old, new));
                pc * pc_address(1)
            }
            Kind::Load | Kind::Store => {
                let sum = &row[body..body + WORD];
                let carries = &row[body + WORD..body + 2 * WORD - 1];

                // The address is `rs1 + imm`, and `b` holds the immediate's bits.
                builder.assert_eq(word_of(b), e(d + IMM));
                let x: Vec<AB::Expr> = a.iter().map(|&bit| bit.into()).collect();
                let y: Vec<AB::Expr> = b.iter().map(|&bit| bit.into()).collect();
                ripple(builder, &x, &y, AB::Expr::ZERO, sum, carries);

                // Aligned, and inside the ram: only bits `3..3 + log cells` may be set.
                for (i, &bit) in sum.iter().enumerate() {
                    if !(3..3 + LOG_RAM_CELLS).contains(&i) {
                        builder.assert_zero(bit);
                    }
                }

                // Cell `c` sits at `G^c = prod_k (1 + c_k * (G^(2^k) - 1))`.
                let cell = (0..LOG_RAM_CELLS).fold(AB::Expr::ONE, |cell, k| {
                    let factor = F::GENERATOR.exp_power_of_2(k) + F::ONE;
                    cell * (AB::Expr::ONE + e(body + 3 + k) * factor)
                });

                let (value, old) = (e(kind.extra(0)), e(kind.extra(1)));
                touch(
                    builder,
                    &self.registers,
                    0,
                    (e(d + RS1), word_of(a), word_of(a)),
                );
                if kind == Kind::Load {
                    // Read the cell, then write it to `rd`.
                    let new = write.clone() * value.clone() + (AB::Expr::ONE + write) * old.clone();
                    touch(builder, &self.ram, 1, (cell, value.clone(), value));
                    touch(builder, &self.registers, 2, (e(d + RD), old, new));
                } else {
                    // Read `rs2`, then write it to the cell; padding leaves the cell as it is.
                    let new =
                        real.clone() * value.clone() + (AB::Expr::ONE + real.clone()) * old.clone();
                    touch(
                        builder,
                        &self.registers,
                        1,
                        (e(d + RS2), value.clone(), value),
                    );
                    touch(builder, &self.ram, 2, (cell, old, new));
                }
                pc * pc_address(1)
            }
        };

        // Advance the state: pull `(clock, pc)`, push `(clock * g^3, next pc)`.
        builder.push_bus_interaction(
            BusName::new(STATE),
            BusDirection::Pull,
            [clock.clone(), e(d + PC)],
            BusActivation::Boolean(real.clone()),
        );
        builder.push_bus_interaction(
            BusName::new(STATE),
            BusDirection::Push,
            [clock * step(), next_pc],
            BusActivation::Boolean(real),
        );
    }
}

impl IoAir {
    /// Push the start state, pull the halt state, and read the output.
    fn eval<AB: BusInteractionBuilder<F = F>>(&self, builder: &mut AB) {
        let (row, next_active) = {
            let main = builder.main();
            (main.current_slice().to_vec(), main.next_slice()[IO_ACTIVE])
        };
        let output: AB::Expr = builder.public_values()[0].into();
        let e = |column: usize| -> AB::Expr { row[column].into() };
        let (active, end, value) = (e(IO_ACTIVE), e(IO_END), e(IO_VALUE));

        // Row zero is the only active row.
        builder.when_first_row().assert_one(active.clone());
        builder.when_transition().assert_zero(next_active);
        builder.assert_zero((AB::Expr::ONE + active.clone()) * end.clone());

        // Start at pc 0 at clock `g^3`, and stop at the halt pc.
        builder.push_bus_interaction(
            BusName::new(STATE),
            BusDirection::Push,
            [AB::Expr::from(step()), AB::Expr::from(pc_address(0))],
            BusActivation::Boolean(active.clone()),
        );
        builder.push_bus_interaction(
            BusName::new(STATE),
            BusDirection::Pull,
            [end.clone(), AB::Expr::from(pc_address(self.halt))],
            BusActivation::Boolean(active),
        );

        // Read a0 after the last step: that value is the public output.
        access(
            builder,
            &self.registers,
            &row,
            IO_ACCESS,
            (end, 0),
            (AB::Expr::from(register(A0)), value.clone(), value.clone()),
        );
        builder.when_first_row().assert_eq(value, output);
    }
}

impl ProgramAir {
    /// Serve every entry as often as the run reads it.
    fn eval<AB: BusInteractionBuilder<F = F>>(&self, builder: &mut AB) {
        let (entry, next_pc) = {
            let fixed = builder.preprocessed();
            let entry: Vec<AB::Expr> = fixed
                .current_slice()
                .iter()
                .map(|&cell| cell.into())
                .collect();
            (entry, fixed.next_slice()[ENTRY_PC])
        };
        let final_count: AB::Expr = builder.main().current_slice()[0].into();
        let (pc, fields) = entry.split_first().unwrap();

        // Slot `i` sits at pc address `G^i`.
        // The column is fixed at setup; the zerocheck needs one constraint per table.
        builder.when_first_row().assert_one(pc.clone());
        builder
            .when_transition()
            .assert_eq(next_pc, pc.clone() * pc_address(1));

        // Each entry enters the lookup at count one and leaves at the count its reads reached.
        let seed = [pc.clone(), AB::Expr::ONE]
            .into_iter()
            .chain(fields.iter().cloned());
        builder.push_bus_interaction(
            self.program.name(),
            BusDirection::Push,
            seed,
            BusActivation::Always,
        );
        let close = [pc.clone(), final_count]
            .into_iter()
            .chain(fields.iter().cloned());
        builder.push_bus_interaction(
            self.program.name(),
            BusDirection::Pull,
            close,
            BusActivation::Always,
        );
    }
}

/// One memory access, as clock exponents and word values.
#[derive(Clone, Copy, Debug, Default)]
struct Access {
    /// Clock exponent of the last access to the cell; zero is the seed.
    prev: u64,
    old: u64,
}

/// One executed instruction.
#[derive(Clone, Debug)]
struct Step {
    pc: usize,
    /// Clock exponent of the row: slot `k` accesses at `clock + k`.
    clock: u64,
    /// The first operand: `x[rs1]`.
    a: u64,
    /// The second operand: `x[rs2] + imm`, or the immediate of a load or store.
    b: u64,
    /// The result the row claims, the loaded word, or the stored word.
    value: u64,
    accesses: [Access; 3],
}

/// What a forger changes in an otherwise honest run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Forgery {
    Honest,
    /// The second `add` reads t0 as it was before the first `add`.
    StaleRead,
    /// The final `ld` returns 31.
    LoadReturns31,
    /// A `mul` claims `2 * 3 = 7`.
    MulClaims7,
    /// The last `bne` jumps back although `t1 = 5`.
    BranchBack,
    /// The statement claims output 29.
    Output29,
}

/// The access history of one memory: `(time, value)` pairs per cell, the seed first.
struct Cells {
    history: Vec<Vec<(u64, u64)>>,
}

impl Cells {
    /// Every cell seeded at time zero, from `image` or with zero.
    fn new(cells: usize, image: &[(usize, u64)]) -> Self {
        let mut history = vec![vec![(0, 0)]; cells];
        for &(cell, value) in image {
            history[cell] = vec![(0, value)];
        }
        Self { history }
    }

    /// The last access time and value of `cell`.
    fn last(&self, cell: usize) -> (u64, u64) {
        *self.history[cell].last().unwrap()
    }

    /// An access at `time` claiming the cell held `old` since time `prev`.
    fn claim(
        &mut self,
        cell: usize,
        time: u64,
        (prev, old): (u64, u64),
        write: Option<u64>,
    ) -> Access {
        let new = write.unwrap_or(old);
        self.history[cell].push((time, new));
        Access { prev, old }
    }

    /// An honest access at `time`.
    fn touch(&mut self, cell: usize, time: u64, write: Option<u64>) -> Access {
        self.claim(cell, time, self.last(cell), write)
    }
}

/// An execution: its steps, the final a0 read, and the final state of both memories.
struct Run {
    steps: Vec<Step>,
    /// Clock exponent after the last step.
    end: u64,
    output: Access,
    registers: Cells,
    ram: Cells,
}

/// Runs `program` from pc 0 to its end, applying `forgery` on the way.
fn execute(program: &[Instr], forgery: Forgery) -> Run {
    let mut registers = Cells::new(REGISTERS, &[]);
    let mut ram = Cells::new(1 << LOG_RAM_CELLS, &[(INPUT_CELL, INPUT)]);
    let mut steps: Vec<Step> = Vec::new();
    let mut adds = 0;
    let mut pc = 0;
    while pc < program.len() {
        let instr = program[pc];
        let clock = SLOTS * (steps.len() as u64 + 1);
        let mut next = pc + 1;
        let step = match instr.op.kind() {
            Kind::Alu | Kind::Mul => {
                // The second `add` may read t0 from before the first `add`.
                adds += usize::from(pc == 5);
                let first = if forgery == Forgery::StaleRead && pc == 5 && adds == 2 {
                    let history = &registers.history[instr.rs1];
                    let stale = history[history.len() - 2];
                    registers.claim(instr.rs1, clock, stale, None)
                } else {
                    registers.touch(instr.rs1, clock, None)
                };
                let second = registers.touch(instr.rs2, clock + 1, None);
                let (a, b) = (first.old, second.old ^ instr.imm);
                let value = match instr.op {
                    Op::Add => a.wrapping_add(b),
                    Op::Sub => a.wrapping_sub(b),
                    Op::Slt => u64::from((a as i64) < (b as i64)),
                    Op::Mul if forgery == Forgery::MulClaims7 && (a, b) == (2, 3) => 7,
                    Op::Mul => a.wrapping_mul(b),
                    _ => 0,
                };
                let taken = match instr.op {
                    Op::Beq => a == b,
                    Op::Bne => a != b,
                    _ => false,
                };
                if taken {
                    next = instr.target;
                }
                let write = instr.write().then_some(value);
                let third = registers.touch(instr.rd, clock + 2, write);
                Step {
                    pc,
                    clock,
                    a,
                    b,
                    value,
                    accesses: [first, second, third],
                }
            }
            Kind::Load => {
                let base = registers.touch(instr.rs1, clock, None);
                let cell = cell_of(base.old.wrapping_add(instr.imm));
                let read = if forgery == Forgery::LoadReturns31 && pc == 9 {
                    let (prev, _) = ram.last(cell);
                    ram.claim(cell, clock + 1, (prev, 31), None)
                } else {
                    ram.touch(cell, clock + 1, None)
                };
                let write = instr.write().then_some(read.old);
                let third = registers.touch(instr.rd, clock + 2, write);
                Step {
                    pc,
                    clock,
                    a: base.old,
                    b: instr.imm,
                    value: read.old,
                    accesses: [base, read, third],
                }
            }
            Kind::Store => {
                let base = registers.touch(instr.rs1, clock, None);
                let data = registers.touch(instr.rs2, clock + 1, None);
                let cell = cell_of(base.old.wrapping_add(instr.imm));
                let written = ram.touch(cell, clock + 2, Some(data.old));
                Step {
                    pc,
                    clock,
                    a: base.old,
                    b: instr.imm,
                    value: data.old,
                    accesses: [base, data, written],
                }
            }
        };
        steps.push(step);
        pc = next;
    }
    let end = SLOTS * (steps.len() as u64 + 1);
    let output = registers.touch(A0, end, None);
    Run {
        steps,
        end,
        output,
        registers,
        ram,
    }
}

/// The ram cell of an aligned byte address.
fn cell_of(address: u64) -> usize {
    assert_eq!(address % 8, 0, "unaligned access");
    let cell = (address / 8) as usize;
    assert!(cell < 1 << LOG_RAM_CELLS, "access outside the ram");
    cell
}

/// Read counts of the program lookup and the two range tables.
struct Counts {
    program: Vec<u64>,
    range: [Vec<u64>; 2],
    /// Row of each range factor, low then high.
    index: [std::collections::HashMap<F, usize>; 2],
}

impl Counts {
    /// No reads yet.
    fn new() -> Self {
        let height = ClockRangeAir::<C, F>::HEIGHT;
        let index = |factor: fn(usize) -> F| (0..height).map(|row| (factor(row), row)).collect();
        Self {
            program: vec![0; 1 << LOG_PROGRAM],
            range: [vec![0; height], vec![0; height]],
            index: [
                index(ClockRangeAir::<C, F>::low),
                index(ClockRangeAir::<C, F>::high),
            ],
        }
    }

    /// The `k`-th read of an entry holds count `G^k` and its inverse.
    fn next(count: &mut u64) -> [F; 2] {
        let value = F::GENERATOR.exp_u64(*count);
        *count += 1;
        [value, value.inverse()]
    }

    /// The next read of program slot `pc`.
    fn program(&mut self, pc: usize) -> [F; 2] {
        Self::next(&mut self.program[pc])
    }

    /// The seven access columns proving the gap from `prev` to `now`, or padding at time zero.
    fn access(&mut self, gap: Option<(u64, u64)>) -> [F; ACCESS_WIDTH] {
        let (prev, (low, high)) = match gap {
            Some((prev, now)) => (time(prev), Memory::gap_factors(now - prev).unwrap()),
            None => (
                F::ZERO,
                (
                    ClockRangeAir::<C, F>::low(0),
                    ClockRangeAir::<C, F>::high(0),
                ),
            ),
        };
        let mut read = |side: usize, factor: F| {
            let row = self.index[side][&factor];
            let [count, inverse] = Self::next(&mut self.range[side][row]);
            [factor, count, inverse]
        };
        let [l0, l1, l2] = read(0, low);
        let [h0, h1, h2] = read(1, high);
        [prev, l0, l1, l2, h0, h1, h2]
    }
}

/// Writes the bits of `values` from column `at`.
fn put_bits(row: &mut [F], at: usize, values: &[bool]) {
    for (cell, &bit) in row[at..].iter_mut().zip(values) {
        *cell = F::from_bool(bit);
    }
}

/// One row of an instruction table, or a padding row for `None`.
fn instruction_row(
    kind: Kind,
    program: &[Instr],
    step: Option<&Step>,
    counts: &mut Counts,
    forgery: Forgery,
) -> Vec<F> {
    let mut row = vec![F::ZERO; kind.width()];
    let pc = step.map_or(PADDING_PC, |step| step.pc);
    let entry = entry(program, pc);
    let (a, b) = step.map_or((0, 0), |step| (step.a, step.b));
    let (a_bits, b_bits) = (bools(a), bools(b));
    let body = kind.body();

    // Flags, write, and the two operands.
    let op = program.get(pc).map(|instr| instr.op);
    if let Some(op) = op {
        row[op.code() - kind.first_op()] = F::ONE;
    }
    row[kind.write()] = entry[ENTRY_WRITE];
    put_bits(&mut row, kind.a(), &a_bits);
    put_bits(&mut row, kind.b(), &b_bits);

    // The circuit, and the extra dense values.
    let olds = step.map_or([0; 3], |step| step.accesses.map(|access| access.old));
    match kind {
        Kind::Alu => {
            let invert = matches!(op, Some(Op::Sub | Op::Slt | Op::Beq | Op::Bne));
            let y: Vec<bool> = b_bits.iter().map(|&bit| bit ^ invert).collect();
            let (sum, carries) = ripple_bits(&a_bits, &y, invert);
            let lt = sum[63] ^ ((a_bits[63] ^ b_bits[63]) & (a_bits[63] ^ sum[63]));
            let difference = sum
                .iter()
                .rev()
                .fold(0u64, |x, &bit| (x << 1) | u64::from(bit));
            let eq = difference == 0;
            let honest = (op == Some(Op::Beq) && eq) || (op == Some(Op::Bne) && !eq);
            // The forged branch jumps although the operands are equal.
            let forged = forgery == Forgery::BranchBack && op == Some(Op::Bne) && eq;
            put_bits(&mut row, body + ALU_SUM, &sum);
            put_bits(&mut row, body + ALU_CARRY, &carries);
            put_bits(&mut row, body + ALU_LT, &[lt, eq, honest || forged]);
            row[kind.extra(0)] = word(olds[2]);
            row[kind.extra(1)] = word(difference).try_inverse().unwrap_or(F::ZERO);
        }
        Kind::Mul => {
            let first: Vec<bool> = a_bits.iter().map(|&bit| bit & b_bits[0]).collect();
            put_bits(&mut row, body, &first);
            let mut acc = first[1..].to_vec();
            let mut out = vec![body];
            for j in 1..WORD {
                let at = body + WORD + mul_step(j);
                let y: Vec<bool> = (j..WORD).map(|i| a_bits[i - j] & b_bits[j]).collect();
                let (sum, carries) = ripple_bits(&acc, &y, false);
                put_bits(&mut row, at, &sum);
                put_bits(&mut row, at + WORD - j, &carries);
                out.push(at);
                acc = sum[1..].to_vec();
            }
            // The row's product bits say what it claims, which is `a * b` unless forged.
            let claimed = step.map_or(0, |step| step.value);
            for (i, &column) in out.iter().enumerate() {
                row[column] = F::from_bool((claimed >> i) & 1 == 1);
            }
            row[kind.extra(0)] = word(olds[2]);
        }
        Kind::Load | Kind::Store => {
            let (sum, carries) = ripple_bits(&a_bits, &b_bits, false);
            put_bits(&mut row, body, &sum);
            put_bits(&mut row, body + WORD, &carries);
            row[kind.extra(0)] = word(olds[1]);
            row[kind.extra(1)] = word(olds[2]);
        }
    }

    // Plumbing: the clock, the fetched entry, and its read count.
    let d = kind.bits();
    row[d + CLOCK] = step.map_or(F::ZERO, |step| time(step.clock));
    row[d + PC] = entry[ENTRY_PC];
    [row[d + COUNT], row[d + COUNT_INVERSE]] = counts.program(pc);
    row[d + RD..d + PLUMBING].copy_from_slice(&entry[ENTRY_RD..ENTRY_WRITE]);

    // Accesses: slot `k` of a real row is at `clock + k`.
    for k in 0..3 {
        let gap = step.map(|step| (step.accesses[k].prev, step.clock + k as u64));
        let at = kind.access(k);
        row[at..at + ACCESS_WIDTH].copy_from_slice(&counts.access(gap));
    }
    row
}

/// Base-two logarithm of the height a table of `rows` rows needs.
fn log_height(rows: usize) -> usize {
    rows.next_power_of_two().max(2).trailing_zeros() as usize
}

/// The traces and public values of one run, with its table heights.
struct Witness {
    traces: Vec<RowMajorMatrix<F>>,
    public: Vec<Vec<F>>,
    log_heights: Vec<usize>,
}

/// Runs `program` and lays the run out as the nine traces.
fn witness(program: &[Instr], forgery: Forgery) -> Witness {
    let run = execute(program, forgery);
    let mut counts = Counts::new();
    let mut traces = Vec::new();

    // Instruction tables: each step goes to the table of its op, then padding.
    for kind in Kind::ALL {
        let steps: Vec<&Step> = run
            .steps
            .iter()
            .filter(|step| program[step.pc].op.kind() == kind)
            .collect();
        let height = 1 << log_height(steps.len());
        let rows: Vec<F> = (0..height)
            .flat_map(|i| {
                instruction_row(kind, program, steps.get(i).copied(), &mut counts, forgery)
            })
            .collect();
        traces.push(RowMajorMatrix::new(rows, kind.width()));
    }

    // Io: row zero reads a0 after the last step; row one is padding.
    let mut io = vec![F::ONE, time(run.end)];
    io.extend(counts.access(Some((run.output.prev, run.end))));
    io.push(word(run.output.old));
    io.extend([F::ZERO; 2]);
    io.extend(counts.access(None));
    io.push(F::ZERO);
    traces.push(RowMajorMatrix::new(io, IO_WIDTH));

    // Program: the count each entry's reads reached.
    let program_counts = counts
        .program
        .iter()
        .map(|&reads| F::GENERATOR.exp_u64(reads))
        .collect();
    traces.push(RowMajorMatrix::new(program_counts, 1));

    // Boundaries: each cell closes at its last access with its final value.
    for cells in [&run.registers, &run.ram] {
        let rows = (0..cells.history.len())
            .flat_map(|cell| {
                let (last, value) = cells.last(cell);
                [register(cell), time(last), word(value)]
            })
            .collect();
        traces.push(RowMajorMatrix::new(rows, 3));
    }

    // Range: each factor closes at the count its reads reached.
    let range = (0..ClockRangeAir::<C, F>::HEIGHT)
        .flat_map(|row| {
            [
                ClockRangeAir::<C, F>::low(row),
                F::GENERATOR.exp_u64(counts.range[0][row]),
                ClockRangeAir::<C, F>::high(row),
                F::GENERATOR.exp_u64(counts.range[1][row]),
            ]
        })
        .collect();
    traces.push(RowMajorMatrix::new(range, 4));

    // The statement's one public value: the output, or a false claim.
    let output = if forgery == Forgery::Output29 {
        29
    } else {
        run.output.old
    };
    let mut public = vec![Vec::new(); traces.len()];
    public[IO] = vec![word(output)];
    let log_heights = traces
        .iter()
        .map(|trace| trace.height().trailing_zeros() as usize)
        .collect();
    Witness {
        traces,
        public,
        log_heights,
    }
}

/// Main and preprocessed commitments, both mixed.
struct MachineConfig {
    pcs: Pcs,
    preprocessed_pcs: Pcs,
}

impl MultiStarkConfig for MachineConfig {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = Pcs;

    fn pcs(&self) -> &Pcs {
        &self.pcs
    }

    fn preprocessed_pcs(&self) -> &Pcs {
        &self.preprocessed_pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // Keccak-256 is shared by the transcript and the Merkle tree.
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Vec<Table<F>> {
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a MixedTraceData<F, BinaryPcsProverData<F, F, Mmcs>>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// A mixed commitment for tables of these shapes, each with a leading bit region.
fn commitment(shapes: &[TableShape], bits: Vec<usize>) -> Pcs {
    let (arity, _) = plan_stacked_layout(&committed_shapes::<F>(shapes, &bits));
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: SECURITY_BITS,
    };
    let config = BinaryPcsConfig::try_new::<F, F>(arity - ABSORBED, params)
        .unwrap()
        .try_with_folding(3.min(arity - ABSORBED))
        .unwrap();
    let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let mmcs = Mmcs::for_folding(merkle, &config);
    let inner = BooleanTracePcs::new(config, mmcs.clone(), mmcs, arity).unwrap();
    MixedTraceCommitment::new(inner, bits)
}

/// A fresh transcript.
fn challenger() -> Challenger {
    Challenger::from_hasher(b"p3-multi-stark-integer-machine-v1".to_vec(), Keccak256Hash)
}

/// Why a witness fails: the tables whose constraints break, and the buses that do not balance.
#[derive(Debug, PartialEq, Eq)]
struct Diagnosis {
    tables: Vec<&'static str>,
    buses: Vec<String>,
}

/// The machine at fixed table heights: its tables, keys and statement.
struct Machine {
    chips: Vec<Chip>,
    log_heights: Vec<usize>,
    config: MachineConfig,
    proving_key: ProvingKey<MachineConfig>,
    verifying_key: VerifyingKey<MachineConfig>,
    declaration: Declaration,
}

impl Machine {
    /// Commits the program and reads the statement off the tables.
    fn new(program: &[Instr], log_heights: Vec<usize>) -> Self {
        let chips = Chip::all(program);
        let shapes: Vec<TableShape> = chips
            .iter()
            .zip(&log_heights)
            .map(|(chip, &log_height)| TableShape::new(log_height, chip.width()))
            .collect();
        let bits = chips.iter().map(BaseAir::<F>::boolean_columns).collect();
        let config = MachineConfig {
            pcs: commitment(&shapes, bits),
            preprocessed_pcs: commitment(&[TableShape::new(LOG_PROGRAM, ENTRY_WIDTH)], vec![0]),
        };
        let refs: Vec<&Chip> = chips.iter().collect();
        let (proving_key, verifying_key) = setup(&config, &refs, &mut challenger()).unwrap();
        let tables = chips
            .iter()
            .zip(&log_heights)
            .map(|(chip, &log_height)| {
                TableDeclaration::from_constraints::<F, F, Chip>(
                    chip,
                    HeightRange::exactly(log_height as u32),
                )
            })
            .collect();
        let declaration =
            Declaration::new(Keccak256Hash, tables, PROOF_BUDGET, SECURITY_BITS).unwrap();
        Self {
            chips,
            log_heights,
            config,
            proving_key,
            verifying_key,
            declaration,
        }
    }

    /// The example machine, at the heights its honest run needs.
    fn example() -> Self {
        Self::new(&EXAMPLE, witness(&EXAMPLE, Forgery::Honest).log_heights)
    }

    /// Proves one witness, or reports why the prover refuses it.
    fn prove(&self, witness: &Witness) -> Result<Proof, String> {
        let instances = self
            .chips
            .iter()
            .zip(&witness.traces)
            .zip(&witness.public)
            .map(|((chip, trace), public)| {
                ProverInstance::new(
                    chip,
                    Table::new(trace.transpose()),
                    &self.proving_key,
                    public,
                )
            })
            .collect();
        prove_with_security(
            &self.config,
            ProverInstances::new(instances),
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
        .map_err(|error| format!("{error:?}"))
    }

    /// Verifies a proof against the public output.
    fn verify(
        &self,
        proof: &Proof,
        public: &[Vec<F>],
    ) -> Result<(), VerificationError<PcsError<MachineConfig>>> {
        let instances = self
            .chips
            .iter()
            .zip(&self.log_heights)
            .zip(public)
            .map(|((chip, &log_height), public)| {
                VerifierInstance::new(chip, &self.verifying_key, log_height, public)
            })
            .collect();
        verify_with_security(
            &self.config,
            VerifierInstances::new(instances),
            proof,
            0,
            SECURITY_BITS,
            &mut challenger(),
        )
    }

    /// Why no proof of a witness verifies: the prover's refusal, or the verifier's.
    ///
    /// Returns `None` when a proof verifies against the witness's own public values.
    fn refusal(&self, witness: &Witness) -> Option<String> {
        match self.prove(witness) {
            Err(error) => Some(format!("prover: {error}")),
            Ok(proof) => self
                .verify(&proof, &witness.public)
                .err()
                .map(|error| format!("verifier: {error:?}")),
        }
    }

    /// Replays every constraint and every bus of a witness.
    fn diagnose(&self, witness: &Witness) -> Diagnosis {
        let tables = self
            .chips
            .iter()
            .enumerate()
            .filter(|&(index, chip)| {
                let report = check_all_constraints(
                    chip,
                    &witness.traces[index],
                    &witness.public[index],
                    None,
                );
                !report.failures.is_empty()
            })
            .map(|(index, chip)| chip.name(index))
            .collect();

        let profiles: Vec<BusSymbolicBuilder<F, F>> = self
            .chips
            .iter()
            .map(|chip| BusSymbolicBuilder::from_air(chip, AirLayout::from_air::<F>(chip)))
            .collect();
        let main: Vec<Table<F>> = witness
            .traces
            .iter()
            .map(|trace| Table::new(trace.transpose()))
            .collect();
        let preprocessed = Table::new(
            self.chips[PROGRAM]
                .preprocessed_trace()
                .unwrap()
                .transpose(),
        );

        // The replay reads the public image as the ram boundary's periodic column.
        let image = self.chips[RAM].periodic_columns();
        let periodic = Table::new(RowMajorMatrix::new(image[0].clone(), 1 << LOG_RAM_CELLS));
        let instances: Vec<BusDebugInstance<'_, F>> = (0..self.chips.len())
            .map(|index| {
                let fixed = (index == PROGRAM).then_some(&preprocessed);
                let instance = BusDebugInstance::new(
                    &main[index],
                    fixed,
                    &witness.public[index],
                    &profiles[index],
                )
                .unwrap();
                if index == RAM {
                    instance.with_periodic(&periodic)
                } else {
                    instance
                }
            })
            .collect();
        let buses = BusDebugReport::check(&instances)
            .unwrap()
            .buses
            .into_iter()
            .map(|bus| bus.bus_name)
            .collect();
        Diagnosis { tables, buses }
    }
}

/// An unbalanced bus: the honest prover cannot build the product argument.
const UNBALANCED: &str = "prover: BusArgument(UnbalancedProducts)";

/// A broken constraint: the proof is built, and its zerocheck does not close.
const VIOLATED: &str = "verifier: Zerocheck(FinalSumMismatch)";

/// A forgery the machine refuses for the reason named.
fn assert_refused(forgery: Forgery, tables: &[&'static str], buses: &[&str], refusal: &str) {
    let machine = Machine::example();
    let witness = witness(&EXAMPLE, forgery);

    // The forged run keeps the honest table heights.
    assert_eq!(witness.log_heights, machine.log_heights);

    // Exactly the named constraints break, and exactly the named buses do not balance.
    let expected = Diagnosis {
        tables: tables.to_vec(),
        buses: buses.iter().map(|&bus| bus.to_string()).collect(),
    };
    assert_eq!(machine.diagnose(&witness), expected);

    // The prover or the verifier refuses it, as named.
    assert_eq!(machine.refusal(&witness).as_deref(), Some(refusal));
}

#[test]
fn the_example_computes_thirty() {
    let run = execute(&EXAMPLE, Forgery::Honest);

    // Four setup steps, four loop iterations of four steps, one store and one load.
    assert_eq!(run.steps.len(), 22);

    // a0 holds `3 + 6 + 9 + 12`, and so does ram cell 0.
    assert_eq!(run.output.old, 30);
    assert_eq!(run.ram.last(0).1, 30);

    // x0 is read but never changes.
    assert!(
        run.registers.history[ZERO]
            .iter()
            .all(|&(_, value)| value == 0)
    );
}

#[test]
fn sub_slt_and_beq_prove() {
    let witness = witness(&ALU_CHECK, Forgery::Honest);
    let machine = Machine::new(&ALU_CHECK, witness.log_heights.clone());

    // `9 + 1`: the subtraction crosses zero, and one beq falls through while the other jumps.
    assert_eq!(execute(&ALU_CHECK, Forgery::Honest).output.old, 10);
    assert_eq!(execute(&ALU_CHECK, Forgery::Honest).steps.len(), 8);

    // Every constraint holds, every bus balances, and the proof verifies.
    let clean = Diagnosis {
        tables: Vec::new(),
        buses: Vec::new(),
    };
    assert_eq!(machine.diagnose(&witness), clean);
    let proof = machine.prove(&witness).unwrap();
    machine.verify(&proof, &witness.public).unwrap();
}

#[test]
fn the_honest_run_balances_and_proves_the_same_bytes_twice() {
    let machine = Machine::example();
    let witness = witness(&EXAMPLE, Forgery::Honest);

    // Every constraint holds and every bus balances.
    let clean = Diagnosis {
        tables: Vec::new(),
        buses: Vec::new(),
    };
    assert_eq!(machine.diagnose(&witness), clean);

    // The proof verifies against output 30.
    let proof = machine.prove(&witness).unwrap();
    assert_eq!(witness.public[IO], [word(30)]);
    machine.verify(&proof, &witness.public).unwrap();

    // Proving again gives the same bytes.
    let bytes = postcard::to_allocvec(&proof).unwrap();
    let again = postcard::to_allocvec(&machine.prove(&witness).unwrap()).unwrap();
    assert_eq!(bytes, again);
}

#[test]
fn a_stale_register_read_leaves_the_registers_unbalanced() {
    // The second add reads t0 = 0, left before the first add, so a0 ends at 27.
    assert_eq!(execute(&EXAMPLE, Forgery::StaleRead).output.old, 27);
    assert_refused(Forgery::StaleRead, &[], &["registers"], UNBALANCED);
}

#[test]
fn a_load_returning_31_leaves_the_ram_unbalanced() {
    // The store left 30 in cell 0, but the load claims 31.
    assert_eq!(execute(&EXAMPLE, Forgery::LoadReturns31).output.old, 31);
    assert_refused(Forgery::LoadReturns31, &[], &["ram"], UNBALANCED);
}

#[test]
fn a_product_of_seven_breaks_the_multiplier() {
    // The run is consistent with `2 * 3 = 7`, so only the circuit can refuse it.
    assert_eq!(execute(&EXAMPLE, Forgery::MulClaims7).output.old, 31);
    assert_refused(Forgery::MulClaims7, &["mul"], &[], VIOLATED);
}

#[test]
fn a_branch_taken_on_equal_operands_breaks_the_alu() {
    // The branch rule refuses the jump, and the jump's state has no row to pull it.
    assert_refused(Forgery::BranchBack, &["alu"], &[STATE], UNBALANCED);
}

#[test]
fn a_false_output_breaks_the_io_pin() {
    // The trace is honest; only the claimed output is wrong.
    assert_refused(Forgery::Output29, &["io"], &[], VIOLATED);
}

#[test]
fn the_cost_report_matches_the_proof() {
    let machine = Machine::example();
    let witness = witness(&EXAMPLE, Forgery::Honest);
    let proof = machine.prove(&witness).unwrap();
    let run = machine.declaration.run(&machine.log_heights, 0).unwrap();
    let report = machine.declaration.cost_report::<F>(&run).unwrap();

    // Every table commits its width times its height.
    for ((cost, chip), &log_height) in report
        .tables()
        .iter()
        .zip(&machine.chips)
        .zip(&machine.log_heights)
    {
        assert_eq!(cost.committed_cells, chip.width() << log_height);
        assert_eq!(
            cost.preprocessed_cells,
            chip.preprocessed_width() << log_height
        );
        assert_eq!(
            cost.opened_values,
            chip.width()
                + chip.main_next_row_columns().len()
                + chip.preprocessed_width()
                + chip.preprocessed_next_row_columns().len()
        );
    }

    // The zerocheck runs one round per variable the report counts.
    assert_eq!(
        report.total().sumcheck_rounds,
        proof.sumcheck.round_polys.len()
    );

    // The mixed commitment opens each batch widened to its table: one value per bit, 128 per dense cell.
    let widened =
        |chip: &Chip| chip.boolean_columns() + 128 * (chip.width() - chip.boolean_columns());
    let expected: usize = machine
        .chips
        .iter()
        .map(|chip| widened(chip) * (1 + usize::from(!chip.main_next_row_columns().is_empty())))
        .sum();
    assert_eq!(proof.opening.values.len(), expected);

    // The program opens its eight fixed dense columns on the current row and, for the pc walk, the next.
    let fixed = proof.preprocessed_opening.as_ref().unwrap();
    assert_eq!(fixed.values.len(), 2 * ENTRY_WIDTH * 128);

    // The sealed proof fits the declared budget.
    let sealed = machine.declaration.seal(&run, &proof).unwrap().into_bytes();
    assert!(sealed.len() <= PROOF_BUDGET);
}

#[test]
#[ignore = "benchmark: cargo test --release -p p3-multi-stark --features parallel --test integer_machine -- --ignored --nocapture"]
fn benchmark() {
    println!("iterations  cycles  prove_ms  cycles_per_s  proof_kib  verify_ms");
    for iterations in [4, 64, 1024, 4096] {
        let program = program(iterations);
        let witness = witness(&program, Forgery::Honest);
        let cycles = execute(&program, Forgery::Honest).steps.len();
        let machine = Machine::new(&program, witness.log_heights.clone());
        let start = Instant::now();
        let proof = machine.prove(&witness).unwrap();
        let prove = start.elapsed().as_secs_f64();
        let start = Instant::now();
        machine.verify(&proof, &witness.public).unwrap();
        let verify = start.elapsed().as_secs_f64();
        let size = postcard::to_allocvec(&proof).unwrap().len();
        println!(
            "{iterations:>10}  {cycles:>6}  {:>8.1}  {:>12.0}  {:>9.1}  {:>9.1}",
            prove * 1e3,
            cycles as f64 / prove,
            size as f64 / 1024.0,
            verify * 1e3
        );
    }
}
