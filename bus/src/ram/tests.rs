use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use std::panic::{AssertUnwindSafe, catch_unwind};

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir, WindowAccess, check_constraints};
use p3_baby_bear::BabyBear;
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_sumcheck::layout::Table;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoroshiro128Plus;

use super::*;
use crate::{
    BusActivation, BusArgumentError, BusChallenges, BusDebugInstance, BusDebugReport, BusDirection,
    BusEvaluation, BusInteractionBuilder, BusPlan, BusPlanInput, BusSymbolicBuilder,
    ReadOnlyMemoryBus, ReadOnlyMemoryInteractionBuilder, ReadOnlyMemoryPlan,
};

type F = BinaryField128;

/// Binary-native transcript, matching the one the read-only memory tests use.
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Channel the machine's chips issue their accesses on.
const ACCESS: &str = "ram-access";

/// Channel carrying the memory image a continuing proof inherits.
const INCOMING: &str = "ram-incoming";

/// Channel carrying the memory image a continuing proof hands on.
const OUTGOING: &str = "ram-outgoing";

/// Builds a self-contained memory statement.
fn single_proof(access_count: usize, address_bits: usize, value_width: usize) -> RamStatement {
    RamStatement {
        access_bus: ACCESS.to_string(),
        access_count,
        address_bits,
        timestamp_bits: access_count.trailing_zeros().max(1) as usize,
        value_width,
        boundary: RamBoundary::SingleProof,
    }
}

/// Builds a continuing memory statement over the same dimensions.
fn segment(access_count: usize, address_bits: usize, value_width: usize) -> RamStatement {
    RamStatement {
        boundary: RamBoundary::Segment {
            incoming: INCOMING.to_string(),
            outgoing: OUTGOING.to_string(),
        },
        ..single_proof(access_count, address_bits, value_width)
    }
}

/// Wraps a statement in its constraint set.
fn air(statement: RamStatement) -> RamAir {
    RamAir::new(statement).expect("the fixture statement is well formed")
}

/// One distinguishable field value, so a forged value never coincides with an honest one.
fn value(tag: u64) -> Vec<F> {
    vec![F::GENERATOR.exp_u64(tag)]
}

/// Whether the constraints accept a concrete trace.
///
/// The constraint checker panics on a violation, so this reports rather than asserts.
///
/// An attack test can then state its precondition before it states its verdict.
fn air_accepts(air: &RamAir, trace: &RamTrace<F>) -> bool {
    let main = RowMajorMatrix::new(trace.values().to_vec(), trace.width());
    catch_unwind(AssertUnwindSafe(|| check_constraints(air, &main, &[]))).is_ok()
}

/// Stand-in for the machine's chips, producing every access this memory consumes.
///
/// It reads the memory's own columns, so it always issues exactly what the trace holds.
///
/// That is what an attack test wants: any imbalance then comes from the memory's declarations.
struct MachineTable {
    /// Total trace width, shared with the memory under test.
    width: usize,
    /// Channel the accesses go out on.
    bus: String,
    /// Payload columns of one access.
    columns: Vec<usize>,
}

impl MachineTable {
    /// Issues whatever the memory's trace says the machine issued.
    fn mirroring(air: &RamAir) -> Self {
        Self {
            width: air.layout().width,
            bus: air.statement().access_bus.clone(),
            columns: air.layout().access_columns().collect(),
        }
    }
}

impl<F2> BaseAir<F2> for MachineTable {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: BusInteractionBuilder> Air<AB> for MachineTable {
    fn eval(&self, builder: &mut AB) {
        let fields = {
            let main = builder.main();
            let row = main.current_slice();
            self.columns
                .iter()
                .map(|&column| row[column].into())
                .collect::<Vec<AB::Expr>>()
        };
        builder.push_bus_interaction(&self.bus, BusDirection::Push, fields, BusActivation::Always);
    }
}

/// A committed memory image on a trace of its own, one entry per cell.
///
/// Unlike the machine table this does not read the memory's columns.
///
/// An image the memory cannot change is the only way to test what the image binds.
struct ImageTable {
    /// Channel this image sits on.
    bus: String,
    /// Side of the multiset the image takes.
    direction: BusDirection,
    /// Slots in one entry.
    width: usize,
}

impl<F2> BaseAir<F2> for ImageTable {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: BusInteractionBuilder> Air<AB> for ImageTable {
    fn eval(&self, builder: &mut AB) {
        let fields = {
            let main = builder.main();
            let row = main.current_slice();
            (0..self.width)
                .map(|column| row[column].into())
                .collect::<Vec<AB::Expr>>()
        };
        builder.push_bus_interaction(&self.bus, self.direction, fields, BusActivation::Always);
    }
}

/// Lays out one image as a trace: cell digits least significant first, then the value.
fn image_trace(statement: &RamStatement, entries: &[(u64, Vec<F>)]) -> RamTrace<F> {
    let width = statement.image_payload_width();
    let mut values = vec![F::ZERO; entries.len() * width];
    for (row, (address, value)) in entries.iter().enumerate() {
        let cells = &mut values[row * width..(row + 1) * width];
        for (bit, digit) in cells.iter_mut().take(statement.address_bits).enumerate() {
            *digit = F::from_bool((address >> bit) & 1 == 1);
        }
        cells[statement.address_bits..].clone_from_slice(value);
    }
    RamTrace::from_values(values, width)
}

/// Column-major view of a trace, as the bus replay reads it.
fn table(trace: &RamTrace<F>) -> Table<F> {
    let (height, width) = (trace.height(), trace.width());
    let mut columns = Vec::with_capacity(height * width);
    for column in 0..width {
        for row in 0..height {
            columns.push(trace.row(row)[column]);
        }
    }
    Table::new(RowMajorMatrix::new(columns, height))
}

/// One table in a replay: its profile and the trace it was filled from.
struct Replay<'a> {
    /// Symbolic declarations of the table.
    profile: BusSymbolicBuilder<F, F>,
    /// Concrete trace the declarations read.
    trace: &'a RamTrace<F>,
}

/// Replays the memory against a machine, plus any image tables the caller supplies.
///
/// The machine reads a trace of its own, which is what lets a test forge one and not the other.
fn report(
    air: &RamAir,
    trace: &RamTrace<F>,
    issued: &RamTrace<F>,
    images: &[Replay<'_>],
) -> BusDebugReport<F> {
    let memory = BusSymbolicBuilder::<F, F>::from_air(air, AirLayout::from_air::<F>(air));
    let machine = MachineTable::mirroring(air);
    let machine_profile =
        BusSymbolicBuilder::<F, F>::from_air(&machine, AirLayout::from_air::<F>(&machine));

    let memory_table = table(trace);
    let machine_table = table(issued);
    let image_tables = images
        .iter()
        .map(|replay| table(replay.trace))
        .collect::<Vec<_>>();
    let mut instances = vec![
        BusDebugInstance::new(&memory_table, None, &[], &memory)
            .expect("the fixture trace matches the memory profile"),
        BusDebugInstance::new(&machine_table, None, &[], &machine_profile)
            .expect("the fixture trace matches the machine profile"),
    ];
    for (replay, image) in images.iter().zip(&image_tables) {
        instances.push(
            BusDebugInstance::new(image, None, &[], &replay.profile)
                .expect("the fixture image matches its profile"),
        );
    }
    BusDebugReport::check(&instances).expect("the fixture declarations are well formed")
}

/// Whether the access channel balances against a machine that issued this very trace.
fn buses_balance(air: &RamAir, trace: &RamTrace<F>) -> bool {
    buses_balance_against(air, trace, trace)
}

/// Whether the access channel balances against a machine that issued something else.
fn buses_balance_against(air: &RamAir, trace: &RamTrace<F>, issued: &RamTrace<F>) -> bool {
    report(air, trace, issued, &[])
        .buses
        .iter()
        .all(|bus| bus.bus_name != ACCESS)
}

/// Names of the multisets that do not balance.
fn unbalanced(air: &RamAir, trace: &RamTrace<F>, images: &[Replay<'_>]) -> Vec<String> {
    let mut names = report(air, trace, trace, images)
        .buses
        .into_iter()
        .map(|bus| bus.bus_name)
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Builds a plan holding this memory and its machine, the way a real statement would.
fn plan(air: &RamAir) -> BusPlan {
    let memory = BusSymbolicBuilder::<F, F>::from_air(air, AirLayout::from_air::<F>(air));
    let machine = MachineTable::mirroring(air);
    let machine_profile =
        BusSymbolicBuilder::<F, F>::from_air(&machine, AirLayout::from_air::<F>(&machine));
    let log_height = air.statement().access_count.trailing_zeros() as usize;
    BusPlan::build(&[
        BusPlanInput {
            log_height,
            interactions: memory.interactions(),
        },
        BusPlanInput {
            log_height,
            interactions: machine_profile.interactions(),
        },
    ])
    .expect("the fixture uses a valid bus shape")
    .expect("the fixture declares at least one tuple")
}

/// Lays accesses out in a caller-chosen order, whether or not it is the sorted one.
///
/// The comparison witness is filled with wrapping arithmetic.
///
/// A descending step then produces exactly the witness an adversary would supply.
///
/// Every adder identity still holds, and only the refused carry out is left to reject the row.
fn lay_out(statement: &RamStatement, rows: &[RamAccess<F>]) -> RamTrace<F> {
    let layout = RamLayout::new(statement).expect("the fixture statement is well formed");
    let mut values = vec![F::ZERO; rows.len() * layout.width];

    for (index, access) in rows.iter().enumerate() {
        let same = index > 0 && rows[index - 1].address == access.address;
        let row = &mut values[index * layout.width..(index + 1) * layout.width];

        row[layout.operation] = F::from_bool(access.write);
        for bit in 0..layout.address_bits {
            row[layout.address + bit] = F::from_bool((access.address >> bit) & 1 == 1);
        }
        for bit in 0..layout.timestamp_bits {
            row[layout.timestamp + bit] = F::from_bool((access.time >> bit) & 1 == 1);
        }
        row[layout.value..layout.value + layout.value_width].clone_from_slice(&access.value);
        row[layout.same_address] = F::from_bool(same);

        if index == 0 {
            continue;
        }
        let earlier = &rows[index - 1];
        let (left, right, bits) = if same {
            (earlier.time, access.time, layout.timestamp_bits)
        } else {
            (earlier.address, access.address, layout.address_bits)
        };

        // A wrapping gap is the honest witness taken modulo the digit count.
        let modulus = 1u128 << bits;
        let delta = ((u128::from(right) + modulus - u128::from(left)) % modulus) as u64;
        let mut carry = 0u64;
        let mut nonzero = 0u64;
        for bit in 0..bits {
            let addend = (left >> bit) & 1;
            let difference = (delta >> bit) & 1;
            row[layout.compare_delta + bit] = F::from_bool(difference == 1);
            row[layout.compare_carry + bit] = F::from_bool(carry == 1);
            carry = (addend & difference) | (addend & carry) | (difference & carry);
            nonzero |= difference;
            row[layout.compare_nonzero + bit] = F::from_bool(nonzero == 1);
        }
        row[layout.compare_carry + bits] = F::from_bool(carry == 1);
    }

    if statement.boundary.is_segment() {
        for index in 0..rows.len() {
            let ends = index + 1 == rows.len() || rows[index + 1].address != rows[index].address;
            values[index * layout.width + layout.group_end] = F::from_bool(ends);
        }
    }

    RamTrace::from_values(values, layout.width)
}

/// Reads one little-endian digit family out of a trace row as an integer.
fn read_bits(trace: &RamTrace<F>, row: usize, offset: usize, bits: usize) -> u64 {
    read_bits_in(trace, row, offset, bits)
}

/// Reads one little-endian digit family out of a trace row over any field.
fn read_bits_in<K: Field>(trace: &RamTrace<K>, row: usize, offset: usize, bits: usize) -> u64 {
    (0..bits)
        .filter(|&bit| trace.row(row)[offset + bit] == K::ONE)
        .fold(0u64, |value, bit| value | 1 << bit)
}

#[test]
fn the_statement_refuses_every_shape_it_cannot_prove() {
    let base = single_proof(4, 3, 1);

    // A one-row trace has no transition, so nothing would compare a row against another.
    //
    // Proof systems downstream refuse a one-row table outright.
    for access_count in [0, 1] {
        assert_eq!(
            RamStatement {
                access_count,
                ..base.clone()
            }
            .validate(),
            Err(RamError::TooFewAccesses {
                access_count,
                minimum: MIN_RAM_ACCESS_COUNT,
            })
        );
    }

    // The plan's product tree aligns each block, so a block height is a power of two.
    assert_eq!(
        RamStatement {
            access_count: 3,
            ..base.clone()
        }
        .validate(),
        Err(RamError::NonPowerOfTwoAccessCount { access_count: 3 })
    );

    // Both digit counts become explicit columns compared by an adder.
    for address_bits in [0, MAX_RAM_BIT_WIDTH + 1] {
        assert_eq!(
            RamStatement {
                address_bits,
                ..base.clone()
            }
            .validate(),
            Err(RamError::AddressBits {
                address_bits,
                maximum: MAX_RAM_BIT_WIDTH
            })
        );
    }
    for timestamp_bits in [0, MAX_RAM_BIT_WIDTH + 1] {
        assert_eq!(
            RamStatement {
                timestamp_bits,
                ..base.clone()
            }
            .validate(),
            Err(RamError::TimestampBits {
                timestamp_bits,
                maximum: MAX_RAM_BIT_WIDTH
            })
        );
    }

    // An access with no value components carries no memory state.
    assert_eq!(
        RamStatement {
            value_width: 0,
            ..base.clone()
        }
        .validate(),
        Err(RamError::EmptyValue)
    );

    // Two roles on one channel would let a tuple of one cancel a tuple of the other.
    assert_eq!(
        RamStatement {
            boundary: RamBoundary::Segment {
                incoming: ACCESS.to_string(),
                outgoing: OUTGOING.to_string(),
            },
            ..base
        }
        .validate(),
        Err(RamError::DuplicateBus {
            name: ACCESS.to_string()
        })
    );
}

#[test]
fn every_column_family_occupies_its_own_range() {
    let statement = segment(8, 5, 2);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // Every family is claimed once, and the claimed ranges tile the trace exactly.
    let families = [
        (layout.operation, 1),
        (layout.address, 5),
        (layout.timestamp, 3),
        (layout.value, 2),
        (layout.same_address, 1),
        (layout.compare_delta, 5),
        (layout.compare_carry, 6),
        (layout.compare_nonzero, 5),
        (layout.group_end, 1),
    ];
    let mut claimed = vec![0usize; layout.width];
    for (start, count) in families {
        for times in claimed.iter_mut().skip(start).take(count) {
            *times += 1;
        }
    }
    assert!(claimed.iter().all(|&times| times == 1));

    // The shared comparison witness is sized for the wider of the two comparisons.
    assert_eq!(layout.compare_bits, 5);

    // A self-contained proof hands on no image, so it allocates no closing marker.
    let lean =
        RamLayout::new(&single_proof(8, 5, 2)).expect("the fixture statement is well formed");
    assert_eq!(lean.width, layout.width - 1);
}

#[test]
fn an_honest_memory_satisfies_every_constraint_and_balances_every_channel() {
    // Write, read back, write again at one cell, then touch a second cell.
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, 0, value(11)),
        RamAccess::read(5, 1, value(11)),
        RamAccess::write(5, 2, value(13)),
        RamAccess::read(2, 3, vec![F::ZERO]),
    ];
    let trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let air = air(statement);

    assert_eq!(trace.width(), BaseAir::<F>::width(&air));
    assert_eq!(trace.height(), 4);
    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));
}

#[test]
fn the_sorted_order_groups_cells_and_orders_readings_inside_a_run() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(6, 0, value(11)),
        RamAccess::write(1, 1, value(13)),
        RamAccess::read(6, 2, value(11)),
        RamAccess::read(1, 3, value(13)),
    ];
    let trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // Cell one's two accesses come first, in reading order, then cell six's two.
    let sorted = (0..4)
        .map(|row| {
            (
                read_bits(&trace, row, layout.address, layout.address_bits),
                read_bits(&trace, row, layout.timestamp, layout.timestamp_bits),
                trace.row(row)[layout.same_address],
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sorted,
        vec![
            (1, 1, F::ZERO),
            (1, 3, F::ONE),
            (6, 0, F::ZERO),
            (6, 2, F::ONE),
        ]
    );
    assert!(air_accepts(&air(statement), &trace));
}

#[test]
fn a_non_boolean_cell_digit_is_refused() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, 0, value(11)),
        RamAccess::read(5, 1, value(11)),
        RamAccess::write(3, 2, value(13)),
        RamAccess::read(3, 3, value(13)),
    ];
    let mut trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // There is one copy of each access, and the machine reads it, so the channel follows.
    for row in 0..4 {
        trace.row_mut(row)[layout.address] = F::GENERATOR;
    }

    // Precondition: the multiset claim still holds, so nothing but the digit check can reject.
    assert!(buses_balance(&air, &trace));

    // A digit holding something else empties out the comparison.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_reading_digit_outside_the_bits_can_satisfy_the_whole_adder() {
    // Every other identity in the comparison is an equation, and a prover can solve equations.
    //
    // Solved over the whole field instead of over the bits, they stop saying anything about order.
    //
    // The gap below stays inside the bits, so only the reading digits are out of place.
    let statement = RamStatement {
        timestamp_bits: 2,
        ..single_proof(2, 1, 1)
    };
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());

    // A write and then a read of one cell, laid out in that order.
    let honest = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 1, value(11)),
    ];
    let mut trace = lay_out(&statement, &honest);
    assert!(air_accepts(&air, &trace));

    // Readings that are not numbers, with a solved witness around them.
    let loose = F::GENERATOR;
    trace.row_mut(0)[layout.timestamp] = loose;
    trace.row_mut(0)[layout.timestamp + 1] = F::ZERO;
    trace.row_mut(1)[layout.timestamp] = loose + F::ONE;
    trace.row_mut(1)[layout.timestamp + 1] = loose;
    trace.row_mut(1)[layout.compare_delta] = F::ONE;
    trace.row_mut(1)[layout.compare_delta + 1] = F::ZERO;
    trace.row_mut(1)[layout.compare_carry] = F::ZERO;
    trace.row_mut(1)[layout.compare_carry + 1] = loose;
    trace.row_mut(1)[layout.compare_carry + 2] = F::ZERO;
    trace.row_mut(1)[layout.compare_nonzero] = F::ONE;
    trace.row_mut(1)[layout.compare_nonzero + 1] = F::ONE;

    // Precondition: the gap stays inside the bits, so a gap check would not object.
    for bit in 0..layout.compare_bits {
        let digit = trace.row(1)[layout.compare_delta + bit];
        assert!(digit == F::ZERO || digit == F::ONE);
    }

    // Precondition: every sum digit holds, which is a plain sum in characteristic two.
    for bit in 0..layout.timestamp_bits {
        let earlier = trace.row(0)[layout.timestamp + bit];
        let gap = trace.row(1)[layout.compare_delta + bit];
        let carry = trace.row(1)[layout.compare_carry + bit];
        assert_eq!(trace.row(1)[layout.timestamp + bit], earlier + gap + carry);
    }

    // Precondition: every carry is the majority of the three inputs above it.
    for bit in 0..layout.timestamp_bits {
        let earlier = trace.row(0)[layout.timestamp + bit];
        let gap = trace.row(1)[layout.compare_delta + bit];
        let carry = trace.row(1)[layout.compare_carry + bit];
        assert_eq!(
            trace.row(1)[layout.compare_carry + bit + 1],
            earlier * gap + earlier * carry + gap * carry,
        );
    }

    // Precondition: nothing carries in, nothing carries out, and the running flag reaches one.
    assert_eq!(trace.row(1)[layout.compare_carry], F::ZERO);
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.timestamp_bits],
        F::ZERO
    );
    assert_eq!(
        trace.row(1)[layout.compare_nonzero + layout.timestamp_bits - 1],
        F::ONE
    );

    // Precondition: read continuity holds, and the multiset claim with it.
    assert_eq!(trace.row(0)[layout.value], trace.row(1)[layout.value]);
    assert!(buses_balance(&air, &trace));

    // Only the reading digits are left.
    //
    // Without them the order between the two rows is empty.
    //
    // The write's value then reaches a read the machine may well have issued before it.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_non_boolean_reading_digit_is_refused() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, 0, value(11)),
        RamAccess::read(5, 1, value(11)),
        RamAccess::read(5, 2, value(11)),
        RamAccess::read(5, 3, value(11)),
    ];
    let mut trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    for row in 0..4 {
        trace.row_mut(row)[layout.timestamp] = F::GENERATOR;
    }

    // Precondition: the multiset claim survives the corruption.
    assert!(buses_balance(&air, &trace));

    // Without digits the comparison of readings stops meaning anything.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn splitting_one_cell_into_two_runs_is_refused() {
    // One write followed by one read of the same cell.
    let statement = single_proof(2, 2, 1);
    let honest = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 1, value(11)),
    ];
    let air = air(statement.clone());
    assert!(air_accepts(&air, &lay_out(&statement, &honest)));

    // Claim the read opens a new run at the same cell, which starts it over at zero.
    let forged = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 1, vec![F::ZERO]),
    ];
    let mut trace = lay_out(&statement, &forged);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    trace.row_mut(1)[layout.same_address] = F::ZERO;
    for bit in 0..layout.compare_bits {
        trace.row_mut(1)[layout.compare_delta + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_carry + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ZERO;
    }
    trace.row_mut(1)[layout.compare_carry + layout.compare_bits] = F::ZERO;

    // Precondition: the forged run flag really is clear on two rows holding one cell.
    assert_eq!(trace.row(1)[layout.same_address], F::ZERO);
    assert_eq!(
        read_bits(&trace, 0, layout.address, layout.address_bits),
        read_bits(&trace, 1, layout.address, layout.address_bits),
    );

    // Precondition: the multiset claim still holds, so only the ordering check can reject.
    assert!(buses_balance(&air, &trace));

    // A strict increase needs a nonzero gap, and a repeated cell number has none.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_wrapped_cell_comparison_is_refused() {
    // Two reads of untouched cells, so nothing but the ordering is in play.
    let statement = single_proof(2, 2, 1);
    let descending = vec![
        RamAccess::read(3, 0, vec![F::ZERO]),
        RamAccess::read(1, 1, vec![F::ZERO]),
    ];
    let trace = lay_out(&statement, &descending);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);

    // Precondition: every adder identity of the comparison still holds.
    let left = read_bits(&trace, 0, layout.address, layout.address_bits);
    let right = read_bits(&trace, 1, layout.address, layout.address_bits);
    let delta = read_bits(&trace, 1, layout.compare_delta, layout.address_bits);
    assert_eq!((left, right), (3, 1));
    assert_eq!((left + delta) % 4, right);
    assert_ne!(delta, 0);
    assert_eq!(
        trace.row(1)[layout.compare_nonzero + layout.address_bits - 1],
        F::ONE
    );

    // Precondition: the wrap is the one thing left. The carry out of the top digit is set.
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.address_bits],
        F::ONE
    );

    // Precondition: the multiset claim still holds.
    assert!(buses_balance(&air, &trace));

    // Refusing the carry out is what makes the comparison unsigned.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_read_placed_before_the_write_it_should_see_is_refused() {
    // Two writes then a read at one cell, plus a read elsewhere to fill the trace.
    let statement = single_proof(4, 2, 1);
    let honest = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::write(1, 1, value(13)),
        RamAccess::read(1, 2, value(13)),
        RamAccess::read(2, 3, vec![F::ZERO]),
    ];
    let air = air(statement.clone());
    assert!(air_accepts(&air, &lay_out(&statement, &honest)));

    // Swap the two writes so the surviving value is the earlier one.
    let forged = vec![
        RamAccess::write(1, 1, value(13)),
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 2, value(11)),
        RamAccess::read(2, 3, vec![F::ZERO]),
    ];
    let trace = lay_out(&statement, &forged);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // Precondition: read continuity now holds, since the row above holds that value.
    assert_eq!(trace.row(1)[layout.value], value(11)[0]);
    assert_eq!(trace.row(2)[layout.value], value(11)[0]);
    assert_eq!(trace.row(2)[layout.same_address], F::ONE);

    // Precondition: the multiset claim still holds.
    assert!(buses_balance(&air, &trace));

    // Only the order of readings within the run is left, and it has wrapped.
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.timestamp_bits],
        F::ONE
    );
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_chosen_carry_chain_cannot_fake_an_order() {
    // The same swapped-write forgery, but with a carry chain the adder never produced.
    //
    // A prover supplies the carries, so nothing about the wrapping witness is forced.
    //
    // It can supply a chain that satisfies every other identity and hides the wrap.
    let statement = single_proof(4, 2, 1);
    let forged = vec![
        RamAccess::write(1, 1, value(13)),
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 2, value(11)),
        RamAccess::read(2, 3, vec![F::ZERO]),
    ];
    let mut trace = lay_out(&statement, &forged);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);

    // Reading one down to reading zero, with every carry clear rather than rippling.
    //
    // The gap that fits a clear chain is one, not the three the honest wrapping witness holds.
    for bit in 0..=layout.compare_bits {
        trace.row_mut(1)[layout.compare_carry + bit] = F::ZERO;
    }
    for bit in 0..layout.compare_bits {
        trace.row_mut(1)[layout.compare_delta + bit] = F::from_bool((1u64 >> bit) & 1 == 1);
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ONE;
    }

    // Precondition: the carry into the lowest digit is clear, as an addition requires.
    assert_eq!(trace.row(1)[layout.compare_carry], F::ZERO);

    // Precondition: no carry leaves the top digit, so nothing looks like a wrap.
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.timestamp_bits],
        F::ZERO
    );

    // Precondition: the gap is nonzero and its running flag reaches one.
    assert_eq!(
        read_bits(&trace, 1, layout.compare_delta, layout.timestamp_bits),
        1
    );
    assert_eq!(
        trace.row(1)[layout.compare_nonzero + layout.timestamp_bits - 1],
        F::ONE
    );

    // Precondition: every sum digit still agrees with the earlier key plus the gap.
    //
    // Exclusive or of three bits is their plain sum here, because the field has characteristic two.
    for bit in 0..layout.timestamp_bits {
        let earlier = trace.row(0)[layout.timestamp + bit];
        let gap = trace.row(1)[layout.compare_delta + bit];
        let carry = trace.row(1)[layout.compare_carry + bit];
        assert_eq!(trace.row(1)[layout.timestamp + bit], earlier + gap + carry);
    }

    // Precondition: the multiset claim still holds.
    assert!(buses_balance(&air, &trace));

    // The carry recurrence is the only thing left, and it is what ties the chain to the keys.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_read_that_ignores_the_last_write_is_refused_under_both_boundaries() {
    // Continuity has to hold whichever boundary the proof carries, so both run here.
    for segment_mode in [false, true] {
        let statement = if segment_mode {
            segment(2, 2, 1)
        } else {
            single_proof(2, 2, 1)
        };
        let honest = if segment_mode {
            vec![
                RamAccess::read(1, 0, value(11)),
                RamAccess::read(1, 1, value(11)),
            ]
        } else {
            vec![
                RamAccess::write(1, 0, value(11)),
                RamAccess::read(1, 1, value(11)),
            ]
        };
        let mut trace =
            RamTrace::build(&statement, &honest).expect("the fixture accesses are consistent");
        let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
        let air = air(statement);
        assert!(air_accepts(&air, &trace));

        // Return a value the cell never held.
        trace.row_mut(1)[layout.value] = value(99)[0];

        // Precondition: the ordering witness is untouched and the multiset claim still holds.
        assert_eq!(trace.row(1)[layout.same_address], F::ONE);
        assert!(buses_balance(&air, &trace));

        // Read continuity is the only thing that makes a read observe memory at all.
        assert!(
            !air_accepts(&air, &trace),
            "continuity went unchecked with segment_mode = {segment_mode}",
        );
    }
}

#[test]
fn a_read_of_an_untouched_cell_must_return_zero() {
    let statement = single_proof(2, 2, 1);

    // Witness generation refuses it outright, with the offending row named.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::read(1, 0, value(11)),
                RamAccess::read(2, 1, vec![F::ZERO]),
            ]
        ),
        Err(RamError::ReadContinuity { index: 0 })
    );

    // So do the constraints, on a trace forged past it.
    let honest = vec![
        RamAccess::read(1, 0, vec![F::ZERO]),
        RamAccess::read(2, 1, vec![F::ZERO]),
    ];
    let mut trace =
        RamTrace::build(&statement, &honest).expect("reads of untouched cells return zero");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    trace.row_mut(0)[layout.value] = value(11)[0];

    // Precondition: the forged row opens its run, and the multiset claim still holds.
    assert_eq!(trace.row(0)[layout.same_address], F::ZERO);
    assert!(buses_balance(&air, &trace));

    // Starting every cell at zero is what stops a proof inventing the memory it begins with.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_run_continued_at_a_different_cell_is_refused() {
    // A write to one cell, then a read of a cell nothing has written.
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(2, 1, vec![F::ZERO]),
    ];
    let mut trace =
        RamTrace::build(&statement, &accesses).expect("a read of an untouched cell returns zero");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Claim the read continues the write's run, which hands it the written value.
    trace.row_mut(1)[layout.value] = value(11)[0];
    trace.row_mut(1)[layout.same_address] = F::ONE;
    for bit in 0..layout.compare_bits {
        trace.row_mut(1)[layout.compare_delta + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_carry + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ZERO;
    }
    trace.row_mut(1)[layout.compare_carry + layout.compare_bits] = F::ZERO;
    for bit in 0..layout.timestamp_bits {
        trace.row_mut(1)[layout.compare_delta + bit] = F::from_bool((1u64 >> bit) & 1 == 1);
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ONE;
    }

    // Precondition: continuity holds, and so does the comparison of readings.
    assert_eq!(trace.row(0)[layout.value], value(11)[0]);
    assert_eq!(trace.row(1)[layout.value], value(11)[0]);
    assert_eq!(
        read_bits(&trace, 1, layout.compare_delta, layout.timestamp_bits),
        1
    );

    // Precondition: the multiset claim still holds.
    assert!(buses_balance(&air, &trace));

    // Only the equal cell number within a run is left.
    assert_ne!(
        read_bits(&trace, 0, layout.address, layout.address_bits),
        read_bits(&trace, 1, layout.address, layout.address_bits),
    );
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn the_bus_is_what_catches_an_access_the_machine_never_issued() {
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 1, value(11)),
    ];
    let trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let air = air(statement);

    // A machine that issued a different schedule entirely.
    let other = vec![
        RamAccess::write(1, 0, value(99)),
        RamAccess::read(1, 1, value(99)),
    ];
    let issued =
        RamTrace::build(air.statement(), &other).expect("the other schedule is consistent too");

    // Precondition: both traces satisfy every constraint on their own.
    assert!(air_accepts(&air, &trace));
    assert!(air_accepts(&air, &issued));

    // Replaying the memory against a machine that issued something else finds the mismatch.
    assert!(!buses_balance_against(&air, &trace, &issued));
    assert_eq!(
        report(&air, &trace, &issued, &[])
            .buses
            .into_iter()
            .map(|bus| bus.bus_name)
            .collect::<Vec<_>>(),
        vec![ACCESS.to_string()]
    );
}

#[test]
fn a_segment_opens_and_closes_every_run_against_its_images() {
    // Each run opens with a read carrying the value the part of the execution inherits.
    let statement = segment(4, 3, 1);
    let accesses = vec![
        RamAccess::read(5, 0, value(11)),
        RamAccess::write(5, 1, value(13)),
        RamAccess::read(2, 2, value(17)),
        RamAccess::read(2, 3, value(17)),
    ];
    let trace = RamTrace::build(&statement, &accesses).expect("every run opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Two cells, so two opening reads and two closing values.
    let openings = (0..4)
        .filter(|&row| trace.row(row)[layout.same_address] == F::ZERO)
        .count();
    let closings = (0..4)
        .filter(|&row| trace.row(row)[layout.group_end] == F::ONE)
        .count();
    assert_eq!((openings, closings), (2, 2));

    // The handed-on image carries the last value at each cell, which for cell five is the write.
    let last_of_five = (0..4)
        .find(|&row| {
            trace.row(row)[layout.group_end] == F::ONE
                && read_bits(&trace, row, layout.address, layout.address_bits) == 5
        })
        .expect("cell five has a closing row");
    assert_eq!(trace.row(last_of_five)[layout.value], value(13)[0]);
}

#[test]
fn an_opening_read_is_bound_to_the_inherited_image() {
    // Two cells and two committed images the memory cannot touch.
    let statement = segment(4, 2, 1);
    let accesses = vec![
        RamAccess::read(1, 0, value(11)),
        RamAccess::write(1, 1, value(13)),
        RamAccess::read(2, 2, value(17)),
        RamAccess::read(2, 3, value(17)),
    ];
    let trace = RamTrace::build(&statement, &accesses).expect("every run opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());

    let incoming = ImageTable {
        bus: INCOMING.to_string(),
        direction: BusDirection::Push,
        width: statement.image_payload_width(),
    };
    let outgoing = ImageTable {
        bus: OUTGOING.to_string(),
        direction: BusDirection::Pull,
        width: statement.image_payload_width(),
    };
    let inherited = image_trace(&statement, &[(1, value(11)), (2, value(17))]);
    let handed_on = image_trace(&statement, &[(1, value(13)), (2, value(17))]);
    fn images<'a>(
        incoming: &ImageTable,
        outgoing: &ImageTable,
        inherited: &'a RamTrace<F>,
        handed_on: &'a RamTrace<F>,
    ) -> Vec<Replay<'a>> {
        vec![
            Replay {
                profile: BusSymbolicBuilder::<F, F>::from_air(
                    incoming,
                    AirLayout::from_air::<F>(incoming),
                ),
                trace: inherited,
            },
            Replay {
                profile: BusSymbolicBuilder::<F, F>::from_air(
                    outgoing,
                    AirLayout::from_air::<F>(outgoing),
                ),
                trace: handed_on,
            },
        ]
    }

    // The honest trace balances against both images.
    assert!(air_accepts(&air, &trace));
    let replays = images(&incoming, &outgoing, &inherited, &handed_on);
    assert!(unbalanced(&air, &trace, &replays).is_empty());

    // Now claim cell one was inherited holding something else.
    let forged_accesses = vec![
        RamAccess::read(1, 0, value(99)),
        RamAccess::write(1, 1, value(13)),
        RamAccess::read(2, 2, value(17)),
        RamAccess::read(2, 3, value(17)),
    ];
    let forged = RamTrace::build(&statement, &forged_accesses)
        .expect("the forged schedule is internally consistent");

    // Precondition: every constraint still holds, and the machine issued exactly this.
    assert!(air_accepts(&air, &forged));
    assert_eq!(forged.row(0)[layout.value], value(99)[0]);

    // The inherited image is what refuses it, because no entry holds that value.
    assert_eq!(
        unbalanced(&air, &forged, &replays),
        vec![INCOMING.to_string()]
    );
}

#[test]
fn a_segment_run_opened_by_a_write_is_refused() {
    let statement = segment(2, 2, 1);

    // A write opening a run would declare the value it stores, not the one it inherited.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::write(1, 0, value(11)),
                RamAccess::read(1, 1, value(11)),
            ]
        ),
        Err(RamError::UnopenedSegmentGroup { index: 0 })
    );

    // The constraints refuse the same thing on a trace forged past witness generation.
    let honest = vec![
        RamAccess::read(1, 0, value(11)),
        RamAccess::write(1, 1, value(13)),
    ];
    let mut trace = RamTrace::build(&statement, &honest).expect("the run opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    trace.row_mut(0)[layout.operation] = F::ONE;

    // Precondition: the multiset claim still holds.
    assert!(buses_balance(&air, &trace));

    // An opening write would declare a stored value where an inherited one belongs.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_forged_closing_marker_is_refused_on_any_row() {
    let statement = segment(4, 2, 1);
    let accesses = vec![
        RamAccess::read(1, 0, value(11)),
        RamAccess::write(1, 1, value(13)),
        RamAccess::read(2, 2, value(17)),
        RamAccess::write(2, 3, value(19)),
    ];
    let honest = RamTrace::build(&statement, &accesses).expect("every run opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &honest));

    // Setting the marker early would hand on a value the run later changed.
    let mut early = honest.clone();
    early.row_mut(0)[layout.group_end] = F::ONE;
    assert_eq!(early.row(1)[layout.same_address], F::ONE);
    assert!(!air_accepts(&air, &early));

    // Clearing it on the last row would hand on nothing for that cell at all.
    //
    // No transition looks past the last row, so only the closing rule catches this one.
    let mut late = honest;
    assert_eq!(late.row(3)[layout.group_end], F::ONE);
    late.row_mut(3)[layout.group_end] = F::ZERO;
    assert!(buses_balance(&air, &late));
    assert!(!air_accepts(&air, &late));
}

#[test]
fn the_first_row_cannot_claim_to_continue_a_run() {
    // The run flag gates the declaration against the inherited image.
    let statement = segment(2, 2, 1);
    let accesses = vec![
        RamAccess::read(1, 0, value(11)),
        RamAccess::read(1, 1, value(11)),
    ];
    let mut trace = RamTrace::build(&statement, &accesses).expect("the run opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    trace.row_mut(0)[layout.same_address] = F::ONE;

    // Precondition: the access channel still balances.
    assert!(buses_balance(&air, &trace));

    // The first row always opens a run, so letting it say otherwise would skip its image entry.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_static_indexed_table_keeps_the_lookup_argument() {
    // One table read through the existing read-only helper.
    struct TableReader {
        bus: ReadOnlyMemoryBus<F>,
    }
    impl BaseAir<F> for TableReader {
        fn width(&self) -> usize {
            4
        }
    }
    impl<AB: ReadOnlyMemoryInteractionBuilder<F = F>> Air<AB> for TableReader {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();
            let (address, count, inverse, value) =
                (row[0].into(), row[1].into(), row[2].into(), row[3].into());
            builder.read_only_memory(&self.bus, address, count, inverse, [value]);
        }
    }

    let reader = TableReader {
        bus: ReadOnlyMemoryBus::new("rom").expect("a 128-bit count orbit is unreachable"),
    };
    let profile = BusSymbolicBuilder::<F, F>::from_air(&reader, AirLayout::from_air::<F>(&reader));
    let bus_plan = BusPlan::build(&[BusPlanInput {
        log_height: 0,
        interactions: profile.interactions(),
    }])
    .expect("the fixture uses a valid bus shape")
    .expect("the fixture declares two tuples");

    // That table is served by the lookup argument, which is happy with it.
    assert!(ReadOnlyMemoryPlan::<F>::new(&bus_plan, "rom", 1, 1).is_ok());

    // A mutable memory cannot be pointed at it.
    let statement = RamStatement {
        access_bus: "rom".to_string(),
        ..single_proof(2, 2, 1)
    };
    assert_eq!(
        statement.check_against(&bus_plan),
        Err(RamError::PayloadWidth {
            name: "rom".to_string(),
            expected: statement.access_payload_width(),
            actual: 3,
        })
    );
}

#[test]
fn a_channel_the_plan_does_not_define_is_refused() {
    let air = air(single_proof(2, 2, 1));
    let bus_plan = plan(&air);
    assert_eq!(air.statement().check_against(&bus_plan), Ok(()));

    let statement = RamStatement {
        access_bus: "absent".to_string(),
        ..air.statement().clone()
    };
    assert_eq!(
        statement.check_against(&bus_plan),
        Err(RamError::UnknownBus {
            name: "absent".to_string()
        })
    );
}

#[test]
fn security_reporting_reads_its_field_size_off_the_challenge_field() {
    let air = air(single_proof(8, 4, 1));
    let bus_plan = plan(&air);
    let statement = air.statement();

    // A wide challenge field and a narrow one give different bounds for one statement.
    let wide = statement
        .security_term::<BinaryField128>(&bus_plan)
        .expect("the fixture plan defines every channel");
    let narrow = statement
        .security_term::<BabyBear>(&bus_plan)
        .expect("the fixture plan defines every channel");
    assert!(wide.bits.bits() > narrow.bits.bits() + 90.0);

    // The term is the plan's own, read at the challenge field's size.
    let field_bits =
        core::num::NonZeroUsize::new(BinaryField128::order().bits() as usize - 1).unwrap();
    assert_eq!(wide, bus_plan.security_term(field_bits));

    // A self-contained memory only consumes; a continuing one adds an entry per row each way.
    assert_eq!(statement.leaf_contribution(), [0, 8]);
    assert_eq!(segment(8, 4, 1).leaf_contribution(), [8, 16]);
}

#[test]
fn witness_generation_refuses_a_witness_that_does_not_match_the_statement() {
    let statement = single_proof(2, 2, 1);

    // The access count is public, so a mismatch is the caller's mistake.
    assert_eq!(
        RamTrace::<F>::build(&statement, &[]),
        Err(RamError::AccessCount {
            expected: 2,
            actual: 0
        })
    );

    // So is the value width.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::read(0, 0, vec![F::ZERO, F::ZERO]),
                RamAccess::read(1, 1, vec![F::ZERO]),
            ]
        ),
        Err(RamError::ValueWidth {
            index: 0,
            expected: 1,
            actual: 2
        })
    );

    // A cell number wider than the statement has no digits to commit.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::read(0, 0, vec![F::ZERO]),
                RamAccess::read(9, 1, vec![F::ZERO]),
            ]
        ),
        Err(RamError::AddressRange {
            index: 1,
            address: 9,
            address_bits: 2
        })
    );

    // So does a reading.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::read(0, 0, vec![F::ZERO]),
                RamAccess::read(1, 9, vec![F::ZERO]),
            ]
        ),
        Err(RamError::TimeRange {
            index: 1,
            time: 9,
            timestamp_bits: 1
        })
    );

    // Two accesses to one cell at one reading would have no order between them.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::write(1, 0, value(11)),
                RamAccess::read(1, 0, value(11)),
            ]
        ),
        Err(RamError::RepeatedTime {
            index: 1,
            address: 1,
            time: 0
        })
    );

    // And a read that ignores the last write is named before any constraint runs.
    assert_eq!(
        RamTrace::build(
            &statement,
            &[
                RamAccess::write(1, 0, value(11)),
                RamAccess::read(1, 1, value(13)),
            ]
        ),
        Err(RamError::ReadContinuity { index: 1 })
    );
}

#[test]
fn the_smallest_memory_balances_and_satisfies_every_constraint() {
    // Two accesses is the smallest legal statement, because one row has no transition.
    let statement = single_proof(2, 1, 1);
    let trace = RamTrace::build(
        &statement,
        &[
            RamAccess::write(1, 0, value(11)),
            RamAccess::read(1, 1, value(11)),
        ],
    )
    .expect("the read returns what the write left");
    let air = air(statement);

    assert_eq!(trace.height(), 2);
    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));
}

#[test]
fn a_wide_memory_keeps_every_value_component() {
    // Four field components per cell, which is what a machine word needs in a small field.
    let statement = single_proof(4, 6, 4);
    let word = |tag: u64| {
        (0..4)
            .map(|part| F::GENERATOR.exp_u64(tag + part))
            .collect::<Vec<_>>()
    };
    let accesses = vec![
        RamAccess::write(40, 0, word(11)),
        RamAccess::read(40, 1, word(11)),
        RamAccess::write(40, 2, word(21)),
        RamAccess::read(40, 3, word(21)),
    ];
    let trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let air = air(statement);

    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));
}

#[test]
fn the_statement_and_the_layout_agree_on_every_tuple_width() {
    let statement = segment(4, 5, 3);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // The declarations read exactly as many columns as the channel reserves slots.
    assert_eq!(
        layout.access_columns().count(),
        statement.access_payload_width()
    );
    assert_eq!(
        layout.image_columns().count(),
        statement.image_payload_width()
    );

    // A continuing proof names both image channels; a self-contained one names none.
    assert_eq!(statement.boundary.image_buses(), Some([INCOMING, OUTGOING]));
    assert_eq!(RamBoundary::SingleProof.image_buses(), None);
}

#[test]
fn the_constraints_hold_over_an_odd_characteristic_field_too() {
    // Exclusive or and majority both carry a correction that vanishes in characteristic two.
    //
    // A test over a binary field alone would accept a wrong correction, so a prime field runs too.
    //
    // The schedule is chosen to reach the inputs that separate the right formula from the wrong.
    //
    // Majority differs from the uncorrected form only when all three inputs are set.
    //
    // It differs from the near-miss only when the middle input is clear and the others are set.
    let word = |tag: u32| vec![BabyBear::from_u32(tag)];
    let statement = single_proof(8, 3, 1);
    let accesses = vec![
        RamAccess::write(4, 0, word(11)),
        RamAccess::read(4, 1, word(11)),
        RamAccess::write(6, 2, word(13)),
        RamAccess::write(3, 3, word(17)),
        RamAccess::write(7, 4, word(19)),
        RamAccess::read(7, 5, word(19)),
        RamAccess::read(3, 6, word(17)),
        RamAccess::read(6, 7, word(13)),
    ];
    let trace =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let air = air(statement);
    let layout = air.layout();

    // Precondition: one run spans readings three and six, so that comparison adds three to three.
    assert_eq!(
        read_bits_in(&trace, 1, layout.timestamp, layout.timestamp_bits),
        6
    );
    assert_eq!(trace.row(1)[layout.same_address], BabyBear::ONE);
    assert_eq!(
        read_bits_in(&trace, 1, layout.compare_delta, layout.timestamp_bits),
        3
    );

    // Precondition: the next run opens right after, so that comparison adds one to three.
    assert_eq!(
        read_bits_in(&trace, 2, layout.address, layout.address_bits),
        4
    );
    assert_eq!(trace.row(2)[layout.same_address], BabyBear::ZERO);
    assert_eq!(
        read_bits_in(&trace, 2, layout.compare_delta, layout.address_bits),
        1
    );

    let main = RowMajorMatrix::new(trace.values().to_vec(), trace.width());
    check_constraints(&air, &main, &[]);

    // The adder must still reject a wrapped comparison here.
    let mut forged = trace;
    forged.row_mut(2)[layout.compare_carry + layout.address_bits] = BabyBear::ONE;
    let main = RowMajorMatrix::new(forged.values().to_vec(), forged.width());
    assert!(catch_unwind(AssertUnwindSafe(|| check_constraints(&air, &main, &[]))).is_err());
}

#[test]
fn every_derived_witness_column_is_pinned_by_its_own_identity() {
    // The gap, its carries and its running flags are all prover-supplied.
    //
    // Each is pinned by an identity rather than merely permitted.
    //
    // A prover free to pick any of them could pick one that makes a false order pass.
    //
    // Equal cell and clock widths keep every shared column active on every row.
    let statement = single_proof(4, 2, 1);
    let accesses = vec![
        RamAccess::write(1, 0, value(11)),
        RamAccess::read(1, 1, value(11)),
        RamAccess::write(2, 2, value(13)),
        RamAccess::read(2, 3, value(13)),
    ];
    let honest =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &honest));
    assert_eq!(layout.compare_bits, layout.address_bits);
    assert_eq!(layout.compare_bits, layout.timestamp_bits);

    let witness = (layout.compare_delta..layout.compare_delta + layout.compare_bits)
        .chain(layout.compare_carry..layout.compare_carry + layout.compare_bits + 1)
        .chain(layout.compare_nonzero..layout.compare_nonzero + layout.compare_bits)
        .collect::<Vec<_>>();
    for row in 1..4 {
        for &column in &witness {
            let mut forged = honest.clone();
            forged.row_mut(row)[column] = F::ONE - forged.row(row)[column];
            assert!(
                !air_accepts(&air, &forged),
                "column {column} of row {row} is not pinned",
            );
        }
    }
}

#[test]
fn the_constraint_count_is_pinned_so_the_load_bearing_note_cannot_drift() {
    // The note on the constraint set says which deletions a test here catches.
    //
    // That claim comes from a sweep run outside the suite, so nothing in-process re-checks it.
    //
    // Pinning the count is the tripwire: changing the constraint set trips this and forces a redo.
    for (statement, expected) in [(single_proof(4, 2, 1), 29), (segment(4, 2, 1), 34)] {
        let air = air(statement);
        let profile = BusSymbolicBuilder::<F, F>::from_air(&air, AirLayout::from_air::<F>(&air));
        assert_eq!(
            profile.base_constraints().len(),
            expected,
            "the constraint set changed, so the note on the constraint set needs a fresh sweep",
        );
    }
}

#[test]
fn a_long_random_schedule_proves_and_balances() {
    // Sixty-four accesses over sixteen cells exercise every run length and both comparisons.
    let statement = single_proof(64, 4, 1);
    let mut rng = Xoroshiro128Plus::seed_from_u64(0x5ea1_1eaf);
    let mut memory = vec![F::ZERO; 16];
    let accesses = (0..64)
        .map(|time| {
            let address = u64::from(rng.random::<u8>() % 16);
            if rng.random::<bool>() {
                let fresh: F = rng.random();
                memory[address as usize] = fresh;
                RamAccess::write(address, time, vec![fresh])
            } else {
                RamAccess::read(address, time, vec![memory[address as usize]])
            }
        })
        .collect::<Vec<_>>();
    let trace = RamTrace::build(&statement, &accesses).expect("the schedule tracks its own memory");
    let air = air(statement);

    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));

    // Every mutation of a single stored value is caught by one of the two mechanisms.
    //
    // The machine keeps issuing the honest schedule, so a rewritten value has to answer for it.
    let layout = air.layout();
    for row in 0..64 {
        let mut forged = trace.clone();
        forged.row_mut(row)[layout.value] += F::ONE;
        assert!(
            !air_accepts(&air, &forged) || !buses_balance_against(&air, &forged, &trace),
            "row {row} was mutated without being rejected",
        );
    }
}

/// Materializes one side of a plan's product tree from concrete traces.
fn materialize(
    bus_plan: &BusPlan,
    profiles: &[&BusSymbolicBuilder<F, F>],
    traces: &[&RamTrace<F>],
    challenges: &BusChallenges<F>,
    direction: BusDirection,
) -> Vec<F> {
    let weights = challenges.fingerprint_weights();
    let mut leaves = Vec::new();
    for block in bus_plan.blocks(direction) {
        let interaction = &profiles[block.owner.air].interactions()[block.owner.declaration];
        let trace = traces[block.owner.air];
        let height = trace.height();
        for row in 0..height {
            leaves.push(
                bus_plan
                    .evaluate_factor(
                        block.bus,
                        interaction,
                        BusEvaluation {
                            main: trace.row(row),
                            preprocessed: &[],
                            public: &[],
                            is_first_row: F::from_bool(row == 0),
                            is_last_row: F::from_bool(row + 1 == height),
                            is_transition: F::from_bool(row + 1 != height),
                        },
                        &weights,
                        challenges.offset,
                    )
                    .expect("the fixture declarations match the plan"),
            );
        }
    }
    leaves
}

#[test]
fn the_plan_reduction_proves_the_trace_holds_the_machine_s_accesses() {
    // A machine would reach this through its own prover.
    //
    // The point here is that the permutation needs no reduction of its own.
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, 0, value(11)),
        RamAccess::read(5, 1, value(11)),
        RamAccess::write(5, 2, value(13)),
        RamAccess::read(2, 3, vec![F::ZERO]),
    ];
    let honest =
        RamTrace::build(&statement, &accesses).expect("the fixture accesses are consistent");
    let air = air(statement);
    let layout = air.layout();

    let memory = BusSymbolicBuilder::<F, F>::from_air(&air, AirLayout::from_air::<F>(&air));
    let machine = MachineTable::mirroring(&air);
    let machine_profile =
        BusSymbolicBuilder::<F, F>::from_air(&machine, AirLayout::from_air::<F>(&machine));
    let profiles = [&memory, &machine_profile];
    let bus_plan = plan(&air);

    // Challenges are drawn inside the reduction, after whatever the caller already bound.
    //
    // For a real prover that is the commitment to this very trace.
    let reduce = |memory_trace: &RamTrace<F>, machine_trace: &RamTrace<F>| {
        let traces = [memory_trace, machine_trace];
        let mut challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash);
        bus_plan.prove::<F, F, _>(
            |challenges| {
                BusDirection::ALL.map(|direction| {
                    materialize(&bus_plan, &profiles, &traces, challenges, direction)
                })
            },
            &mut challenger,
        )
    };

    let (proof, prover_output) = reduce(&honest, &honest).expect("the honest memory balances");
    let mut verifier_challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash);
    let verifier_output = bus_plan
        .verify::<F, F, _>(&proof, &mut verifier_challenger)
        .expect("the honest reduction verifies");
    assert_eq!(prover_output.product.point, verifier_output.product.point);
    assert_eq!(prover_output.product.values, verifier_output.product.values);

    // A memory holding an access the machine never issued no longer balances.
    let mut forged = honest.clone();
    forged.row_mut(0)[layout.value] = value(99)[0];
    assert!(matches!(
        reduce(&forged, &honest),
        Err(BusArgumentError::UnbalancedProducts)
    ));
}
