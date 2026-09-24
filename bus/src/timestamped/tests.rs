use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir, WindowAccess, check_constraints};
use p3_binary_field::{BinaryField32, BinaryField64, BinaryField128};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::{BusDebugInstance, BusDebugReport, BusPlanInput, BusSymbolicBuilder};

/// Bus field.
type F = BinaryField128;

/// Clock field, whose generator ticks the clock.
type C = BinaryField64;

/// Memory under test.
type Memory = TimestampedMemory<C, F>;

/// Channel carrying the memory tuples.
const MEMORY: &str = "tm-memory";

/// Read-only table of low gap factors.
const LOW: &str = "tm-low";

/// Read-only table of high gap factors.
const HIGH: &str = "tm-high";

/// Accesses one row of the machine makes.
const SLOTS: usize = 2;

/// Rows of the machine table.
const ROWS: usize = 4;

/// Cells of the memory, one boundary row each.
const CELLS: usize = 8;

/// Columns of one access: address, previous, old, new, then two range reads of three.
const SLOT_WIDTH: usize = 10;

/// Builds the memory under test, with one value component per cell.
fn memory() -> Memory {
    Memory::new(MEMORY, LOW, HIGH, 1).expect("the fixture names are valid")
}

/// A distinguishable stored value, zero for tag zero.
fn value(tag: u128) -> F {
    F::from_le_bytes(tag.to_le_bytes())
}

/// The clock at exponent `exponent`, which is `g^exponent`.
fn time(exponent: u64) -> F {
    Memory::tick().exp_u64(exponent)
}

/// Address of cell `cell`.
fn cell(cell: usize) -> F {
    TimestampedBoundaryAir::<C, F>::cell_address(cell)
}

/// One access as the attack tests see it, before range counts are assigned.
#[derive(Clone, Debug)]
struct Access {
    /// Cell address.
    address: F,
    /// Time of the last access to that cell.
    previous: F,
    /// Value the last access left.
    old: F,
    /// Value this access leaves.
    new: F,
    /// Low range factor.
    low: F,
    /// High range factor.
    high: F,
}

impl Access {
    /// A padding access, which sits at time zero and reads the smallest range entries.
    fn padding() -> Self {
        Self {
            address: cell(0),
            previous: F::ZERO,
            old: F::ZERO,
            new: F::ZERO,
            low: ClockRangeAir::<C, F>::low(0),
            high: ClockRangeAir::<C, F>::high(0),
        }
    }
}

/// One machine row: its clock and one access per slot.
#[derive(Clone, Debug)]
struct Row {
    /// Clock of the row, zero on padding.
    clock: F,
    /// Accesses in slot order.
    slots: [Access; SLOTS],
}

/// Every table's contents before range counts are assigned.
#[derive(Clone, Debug)]
struct Witness {
    /// Machine rows.
    rows: Vec<Row>,
    /// Starting value of every cell, in cell order.
    starts: Vec<F>,
    /// Last time and final value of every cell, in cell order.
    closes: Vec<(F, F)>,
}

/// One logical operation: a cell and an optional written value.
type Op = (usize, Option<u128>);

/// Runs a program honestly from all-zero cells, padding the machine table to `ROWS`.
///
/// Each entry is a clock exponent and one operation per slot.
fn run(program: &[(u64, [Op; SLOTS])]) -> Witness {
    run_from(&[F::ZERO; CELLS], program)
}

/// Runs a program honestly from the given starting values.
fn run_from(starts: &[F], program: &[(u64, [Op; SLOTS])]) -> Witness {
    // Every cell starts at time `g^0` holding its starting value.
    let mut state = starts
        .iter()
        .map(|&start| (0u64, start))
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for &(clock, ops) in program {
        let slots = core::array::from_fn(|slot| {
            let (target, write) = ops[slot];
            let now = clock + slot as u64;
            let (last, old) = state[target];
            let new = write.map_or(old, value);
            let (low, high) = Memory::gap_factors(now - last).expect("the fixture gaps fit");
            state[target] = (now, new);
            Access {
                address: cell(target),
                previous: time(last),
                old,
                new,
                low,
                high,
            }
        });
        rows.push(Row {
            clock: time(clock),
            slots,
        });
    }
    while rows.len() < ROWS {
        rows.push(Row {
            clock: F::ZERO,
            slots: core::array::from_fn(|_| Access::padding()),
        });
    }
    let closes = state
        .iter()
        .map(|&(last, final_value)| (time(last), final_value))
        .collect();
    Witness {
        rows,
        starts: starts.to_vec(),
        closes,
    }
}

/// The issue's example on cell 5, with two slots per row and one wide gap.
///
/// ```text
///     row 0  clock g^4      write 5 := 7        read 2
///     row 1  clock g^9      read 5              write 5 := 3
///     row 2  clock g^2^20   read 2              write 7 := 9
///     row 3  padding
/// ```
fn honest() -> Witness {
    run(&[
        (4, [(5, Some(7)), (2, None)]),
        (9, [(5, None), (5, Some(3))]),
        (1 << 20, [(2, None), (7, Some(9))]),
    ])
}

/// Stand-in for a machine table making `SLOTS` accesses per row.
struct MachineAir {
    /// Memory the accesses go to.
    memory: Memory,
}

impl<F2> BaseAir<F2> for MachineAir {
    fn width(&self) -> usize {
        1 + SLOTS * SLOT_WIDTH
    }
}

impl<AB: BusInteractionBuilder<F = F>> Air<AB> for MachineAir {
    fn eval(&self, builder: &mut AB) {
        let row: Vec<AB::Expr> = {
            let main = builder.main();
            main.current_slice().iter().map(|&v| v.into()).collect()
        };
        for slot in 0..SLOTS {
            let c = &row[1 + slot * SLOT_WIDTH..1 + (slot + 1) * SLOT_WIDTH];
            let read = |offset: usize| RangeRead {
                value: c[offset].clone(),
                count: c[offset + 1].clone(),
                count_inverse: c[offset + 2].clone(),
            };
            let access = TimestampedAccess {
                address: c[0].clone(),
                previous: c[1].clone(),
                old: vec![c[2].clone()],
                new: vec![c[3].clone()],
                gap: ClockGap {
                    low: read(4),
                    high: read(7),
                },
            };
            builder.timestamped_access(&self.memory, row[0].clone(), slot, access);
        }
    }
}

/// Concrete traces of the three tables.
struct Traces {
    /// Machine table.
    machine: RowMajorMatrix<F>,
    /// Seed and close block.
    boundary: RowMajorMatrix<F>,
    /// Both range tables.
    range: RowMajorMatrix<F>,
}

/// Assigns read-only counts and lays out every table, for a block with this seed.
fn traces(witness: &Witness, seed: &TimestampedSeed<F>) -> Traces {
    let height = ClockRangeAir::<C, F>::HEIGHT;
    let index = |entry: fn(usize) -> F| {
        (0..height)
            .map(|row| (entry(row), row))
            .collect::<HashMap<_, _>>()
    };
    let low_index = index(ClockRangeAir::<C, F>::low);
    let high_index = index(ClockRangeAir::<C, F>::high);
    let mut low_reads = vec![0u64; height];
    let mut high_reads = vec![0u64; height];

    // The k-th read of an entry holds count `G^k`; a forged factor reads no entry.
    let count = |reads: &mut [u64], index: &HashMap<F, usize>, factor: F| {
        index.get(&factor).map_or(F::ONE, |&row| {
            let count = F::GENERATOR.exp_u64(reads[row]);
            reads[row] += 1;
            count
        })
    };

    let mut machine = Vec::new();
    for row in &witness.rows {
        machine.push(row.clock);
        for access in &row.slots {
            let low = count(&mut low_reads, &low_index, access.low);
            let high = count(&mut high_reads, &high_index, access.high);
            machine.extend([
                access.address,
                access.previous,
                access.old,
                access.new,
                access.low,
                low,
                low.inverse(),
                access.high,
                high,
                high.inverse(),
            ]);
        }
    }

    // A private block commits the starting value after the final one.
    let private = matches!(seed, TimestampedSeed::Private(_));
    let boundary = witness
        .closes
        .iter()
        .zip(&witness.starts)
        .enumerate()
        .flat_map(|(row, (&(last, final_value), &start))| {
            [cell(row), last, final_value]
                .into_iter()
                .chain(private.then_some(start))
        })
        .collect();

    let range = (0..height)
        .flat_map(|row| {
            [
                ClockRangeAir::<C, F>::low(row),
                F::GENERATOR.exp_u64(low_reads[row]),
                ClockRangeAir::<C, F>::high(row),
                F::GENERATOR.exp_u64(high_reads[row]),
            ]
        })
        .collect();

    Traces {
        machine: RowMajorMatrix::new(machine, 1 + SLOTS * SLOT_WIDTH),
        boundary: RowMajorMatrix::new(boundary, if private { 4 } else { 3 }),
        range: RowMajorMatrix::new(range, 4),
    }
}

/// Column-major view of a trace, as the bus replay reads it.
fn table(trace: &RowMajorMatrix<F>) -> Table<F> {
    let (height, width) = (trace.values.len() / trace.width, trace.width);
    let columns = (0..width)
        .flat_map(|column| (0..height).map(move |row| trace.values[row * width + column]))
        .collect();
    Table::new(RowMajorMatrix::new(columns, height))
}

/// The three AIRs under test, with a zero seed.
fn airs() -> (
    MachineAir,
    TimestampedBoundaryAir<C, F>,
    ClockRangeAir<C, F>,
) {
    airs_with(TimestampedSeed::Zero)
}

/// The three AIRs under test, with a given seed.
fn airs_with(
    seed: TimestampedSeed<F>,
) -> (
    MachineAir,
    TimestampedBoundaryAir<C, F>,
    ClockRangeAir<C, F>,
) {
    (
        MachineAir { memory: memory() },
        TimestampedBoundaryAir::new(memory(), seed).expect("the fixture seed fits"),
        ClockRangeAir::new(memory()),
    )
}

/// Symbolic declarations of one AIR.
fn profile<A>(air: &A) -> BusSymbolicBuilder<F, F>
where
    A: BaseAir<F> + Air<BusSymbolicBuilder<F, F>>,
{
    BusSymbolicBuilder::from_air(air, AirLayout::from_air::<F>(air))
}

/// Whether every table satisfies its constraints, with a zero seed.
fn constraints_hold(witness: &Witness) -> bool {
    constraints_hold_with(witness, TimestampedSeed::Zero)
}

/// Whether every table satisfies its constraints, with a given seed.
fn constraints_hold_with(witness: &Witness, seed: TimestampedSeed<F>) -> bool {
    let traces = traces(witness, &seed);
    let (machine, boundary, range) = airs_with(seed);
    catch_unwind(AssertUnwindSafe(|| {
        check_constraints(&machine, &traces.machine, &[]);
        check_constraints(&boundary, &traces.boundary, &[]);
        check_constraints(&range, &traces.range, &[]);
    }))
    .is_ok()
}

/// Names of the buses that do not balance, sorted, with a zero seed.
fn unbalanced(witness: &Witness) -> Vec<String> {
    unbalanced_with(witness, TimestampedSeed::Zero)
}

/// Names of the buses that do not balance, sorted, with a given seed.
fn unbalanced_with(witness: &Witness, seed: TimestampedSeed<F>) -> Vec<String> {
    let traces = traces(witness, &seed);
    let (machine, boundary, range) = airs_with(seed);
    let profiles = [profile(&machine), profile(&boundary), profile(&range)];
    let tables = [
        table(&traces.machine),
        table(&traces.boundary),
        table(&traces.range),
    ];

    // The replay reads a public image as the boundary block's periodic columns.
    let periodic = boundary.periodic_columns();
    let periodic = (!periodic.is_empty()).then(|| {
        let values = (0..CELLS)
            .flat_map(|row| periodic.iter().map(move |column| column[row]))
            .collect();
        table(&RowMajorMatrix::new(values, periodic.len()))
    });
    let mut instances = tables
        .iter()
        .zip(&profiles)
        .map(|(table, profile)| {
            BusDebugInstance::new(table, None, &[], profile).expect("the fixture matches")
        })
        .collect::<Vec<_>>();
    if let Some(periodic) = &periodic {
        instances[1] = instances[1].with_periodic(periodic);
    }
    let mut names = BusDebugReport::check(&instances)
        .expect("the fixture declarations are well formed")
        .buses
        .into_iter()
        .map(|bus| bus.bus_name)
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Plan of the three tables at their given heights.
fn plan(log_machine: usize) -> BusPlan {
    let (machine, boundary, range) = airs();
    let profiles = [profile(&machine), profile(&boundary), profile(&range)];
    let heights = [
        log_machine,
        CELLS.trailing_zeros() as usize,
        CLOCK_RANGE_BITS,
    ];
    let inputs = profiles
        .iter()
        .zip(heights)
        .map(|(profile, log_height)| BusPlanInput {
            log_height,
            interactions: profile.interactions(),
        })
        .collect::<Vec<_>>();
    BusPlan::build(&inputs)
        .expect("the fixture uses a valid bus shape")
        .expect("the fixture declares tuples")
}

#[test]
fn an_honest_run_balances_every_bus() {
    let witness = honest();

    // Row 1 touches cell 5 in both slots, so slot times must differ.
    assert_eq!(witness.rows[1].slots[1].previous, time(9));

    // The close of cell 5 is the write in slot 1 of row 1.
    assert_eq!(witness.closes[5], (time(10), value(3)));

    assert!(constraints_hold(&witness));
    assert!(unbalanced(&witness).is_empty());
}

#[test]
fn a_stale_read_leaves_the_memory_unbalanced() {
    let mut witness = honest();

    // The read of cell 5 at `g^9` claims the seed's zero instead of the write's 7.
    let read = &mut witness.rows[1].slots[0];
    read.previous = time(0);
    read.old = F::ZERO;
    read.new = F::ZERO;
    (read.low, read.high) = Memory::gap_factors(9).unwrap();

    // The gap is honest, so only the memory bus can catch it.
    assert!(constraints_hold(&witness));
    assert_eq!(unbalanced(&witness), [MEMORY]);
}

#[test]
fn a_forged_load_leaves_the_memory_unbalanced() {
    let mut witness = honest();

    // The read of cell 5 at `g^9` returns 42, which no access ever wrote.
    let read = &mut witness.rows[1].slots[0];
    read.old = value(42);
    read.new = value(42);

    assert!(constraints_hold(&witness));
    assert_eq!(unbalanced(&witness), [MEMORY]);
}

#[test]
fn a_gap_of_zero_leaves_a_range_table_unbalanced() {
    let mut witness = honest();

    // A real row at `g^20` claims any value of untouched cell 3 in both slots.
    //
    // Each access pulls what it pushes, so the memory bus stays balanced.
    witness.rows[3].clock = time(20);
    for (slot, access) in witness.rows[3].slots.iter_mut().enumerate() {
        let now = time(20 + slot as u64);
        *access = Access {
            address: cell(3),
            previous: now,
            old: value(42),
            new: value(42),
            // `lo = hi` satisfies the gap equation, and `g` sits only in the low table.
            low: Memory::tick(),
            high: Memory::tick(),
        };
    }

    assert!(constraints_hold(&witness));
    assert_eq!(unbalanced(&witness), [HIGH]);
}

#[test]
fn a_padding_row_touching_a_real_cell_leaves_a_range_table_unbalanced() {
    let mut witness = honest();

    // Padding pulls the last tuple of cell 5 and pushes it back at time zero.
    //
    // The close of cell 5 then pulls that zero-time tuple, so the memory bus balances.
    witness.rows[3].slots[0] = Access {
        address: cell(5),
        previous: time(10),
        old: value(3),
        new: value(3),
        // With `now = 0` the gap equation forces `lo = 0`, which no table holds.
        low: F::ZERO,
        high: F::ONE,
    };
    witness.closes[5] = (F::ZERO, value(3));

    assert!(constraints_hold(&witness));
    assert_eq!(unbalanced(&witness), [LOW]);
}

#[test]
fn the_range_table_pins_its_height() {
    let (_, _, range) = airs();

    // Dropping the last entry leaves `g^(2^16 - 1)` on the last row.
    let full = traces(&honest(), &TimestampedSeed::Zero).range;
    let shorter = RowMajorMatrix::new(full.values[..full.values.len() / 2].to_vec(), 4);
    assert!(
        catch_unwind(AssertUnwindSafe(|| check_constraints(
            &range,
            &shorter,
            &[]
        )))
        .is_err()
    );
}

#[test]
fn gap_factors_cover_exactly_one_to_two_to_the_thirty_two() {
    // Both ends of the range, and a gap crossing a digit boundary.
    for gap in [
        1,
        2,
        1 << CLOCK_RANGE_BITS,
        (1 << CLOCK_RANGE_BITS) + 1,
        1 << CLOCK_GAP_BITS,
    ] {
        let (low, high) = Memory::gap_factors(gap).unwrap();
        assert_eq!(time(1) * low, time(1 + gap) * high, "gap {gap}");
    }

    // A zero gap or one past the top has no factors.
    assert_eq!(Memory::gap_factors(0), None);
    assert_eq!(Memory::gap_factors((1 << CLOCK_GAP_BITS) + 1), None);
}

#[test]
fn the_handle_refuses_what_it_cannot_prove() {
    // One channel in two roles would let their tuples cancel.
    assert_eq!(
        Memory::new(MEMORY, MEMORY, HIGH, 1),
        Err(TimestampedMemoryError::DuplicateBus {
            name: MEMORY.into()
        })
    );

    // A cell must carry a value.
    assert_eq!(
        Memory::new(MEMORY, LOW, HIGH, 0),
        Err(TimestampedMemoryError::EmptyValue)
    );

    // A 32-bit clock has order `2^32 - 1`, so a gap of `2^32 - 1` would be zero.
    assert_eq!(
        TimestampedMemory::<BinaryField32, F>::new(MEMORY, LOW, HIGH, 1),
        Err(TimestampedMemoryError::ClockOrbitTooShort {
            orbit_bits: 32,
            minimum_bits: 33
        })
    );

    // A 64-bit bus field is too small for the read-only range lookups.
    assert!(matches!(
        TimestampedMemory::<C, C>::new(MEMORY, LOW, HIGH, 1),
        Err(TimestampedMemoryError::RangeTable(_))
    ));
}

#[test]
fn the_plan_check_bounds_the_accesses() {
    let memory = memory();

    // A small machine fits, and its term is the plan's own.
    let small = plan(ROWS.trailing_zeros() as usize);
    let field_bits = NonZeroUsize::new(128).unwrap();
    assert_eq!(
        memory.security_term::<F>(&small),
        Ok(small.security_term(field_bits))
    );

    // The plan charges every range read, seed and close as a leaf.
    //
    // Pushes: per access one memory tuple and two range reads, per cell one seed, per entry two seeds.
    let accesses = ROWS * SLOTS;
    let range = 2 * ClockRangeAir::<C, F>::HEIGHT;
    assert_eq!(
        small.security_geometry().non_padding_leaf_counts(),
        [3 * accesses + CELLS + range, 3 * accesses + CELLS + range]
    );

    // At `2^32` rows a forged cycle could climb the whole 64-bit orbit.
    let large = plan(32);
    assert!(matches!(
        memory.check_against(&large),
        Err(TimestampedMemoryError::TooManyAccesses { .. })
    ));

    // A memory of a different value width does not fit this plan.
    let wide = Memory::new(MEMORY, LOW, HIGH, 2).unwrap();
    assert_eq!(
        wide.check_against(&small),
        Err(TimestampedMemoryError::PayloadWidth {
            name: MEMORY.into(),
            expected: 4,
            actual: 3
        })
    );
}

/// The public image under test: cell 2 holds 11, cells 5 and 6 hold 7 and 9.
fn image() -> PublicImage<F> {
    PublicImage::new(
        &memory(),
        CELLS.trailing_zeros() as usize,
        vec![(2, vec![value(11)]), (5, vec![value(7), value(9)])],
    )
    .expect("the fixture runs fit")
}

/// A program reading the three image words, then overwriting one.
const IMAGE_PROGRAM: [(u64, [Op; SLOTS]); 2] =
    [(4, [(5, None), (2, None)]), (9, [(6, None), (5, Some(3))])];

#[test]
fn a_public_image_seeds_its_words() {
    // The prover starts from the image the verifier holds.
    let seed = TimestampedSeed::Public(image());
    let witness = run_from(&image().columns()[0], &IMAGE_PROGRAM);
    assert_eq!(witness.rows[0].slots[0].old, value(7));

    assert!(constraints_hold_with(&witness, seed.clone()));
    assert!(unbalanced_with(&witness, seed).is_empty());
}

#[test]
fn a_wrong_public_word_leaves_the_memory_unbalanced() {
    // The prover claims cell 5 started as 5, while the image says 7.
    let mut starts = image().columns().remove(0);
    starts[5] = value(5);
    let witness = run_from(&starts, &IMAGE_PROGRAM);

    // Nothing committed carries the image, so only the memory bus can catch it.
    let seed = TimestampedSeed::Public(image());
    assert!(constraints_hold_with(&witness, seed.clone()));
    assert_eq!(unbalanced_with(&witness, seed), [MEMORY]);
}

#[test]
fn a_seeded_block_pins_its_height() {
    let seed = TimestampedSeed::Private(PrivateRegion::new(0, 3).unwrap());
    let (_, boundary, _) = airs_with(seed.clone());
    let full = traces(&run(&[]), &seed).boundary;
    check_constraints(&boundary, &full, &[]);

    // Dropping the last four cells ends the block before the region does.
    let shorter = RowMajorMatrix::new(full.values[..full.values.len() / 2].to_vec(), 4);
    assert!(
        catch_unwind(AssertUnwindSafe(|| check_constraints(
            &boundary,
            &shorter,
            &[]
        )))
        .is_err()
    );
}

#[test]
fn a_private_region_may_end_at_the_last_machine_word() {
    // A single cell at the top index is its own first and last cell.
    let top = PrivateRegion::new(usize::MAX, 0).unwrap();
    assert_eq!(top.last_cell(), usize::MAX);

    // The widest region ending at the top index.
    let log_cells = usize::BITS as usize - 1;
    let first_cell = usize::MAX - ((1 << log_cells) - 1);
    let widest = PrivateRegion::new(first_cell, log_cells).unwrap();
    assert_eq!(widest.last_cell(), usize::MAX);

    // One cell further would end past the top index.
    assert_eq!(PrivateRegion::new(usize::MAX, 1), None);
    assert_eq!(PrivateRegion::new(first_cell + 1, log_cells), None);

    // A region of `2^BITS` cells cannot be counted in a machine word.
    assert_eq!(PrivateRegion::new(0, usize::BITS as usize), None);
}

#[test]
fn a_public_image_refuses_malformed_runs() {
    let log_cells = CELLS.trailing_zeros() as usize;
    let image = |runs| PublicImage::new(&memory(), log_cells, runs);

    // An empty run holds no word.
    assert_eq!(
        image(vec![(0, vec![])]),
        Err(TimestampedMemoryError::ImageRunWidth {
            run: 0,
            len: 0,
            value_width: 1
        })
    );

    // A run starting inside the previous one would give a cell two starting words.
    assert_eq!(
        image(vec![(2, vec![value(1), value(2)]), (3, vec![value(3)])]),
        Err(TimestampedMemoryError::OverlappingImageRuns { run: 1 })
    );

    // A run ending past the last cell names a cell outside the block.
    assert_eq!(
        image(vec![(7, vec![value(1), value(2)])]),
        Err(TimestampedMemoryError::ImageRunOutOfRange {
            run: 0,
            cells: CELLS
        })
    );

    // An image of two-component words does not fit a one-component memory.
    let wide = Memory::new(MEMORY, LOW, HIGH, 2).unwrap();
    let words = PublicImage::new(&wide, log_cells, vec![(0, vec![value(1), value(2)])]).unwrap();
    assert_eq!(
        TimestampedBoundaryAir::new(memory(), TimestampedSeed::Public(words)),
        Err(TimestampedMemoryError::ImageWidth {
            expected: 1,
            actual: 2
        })
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn a_private_region_accepts_any_start_and_binds_every_read_to_it(
        starts in prop::collection::vec(any::<u128>(), CELLS),
        forged in 1u128..,
    ) {
        let seed = TimestampedSeed::Private(PrivateRegion::new(0, 3).unwrap());
        let starts = starts.into_iter().map(value).collect::<Vec<_>>();

        // Any committed starting values balance an honest run from them.
        let witness = run_from(&starts, &IMAGE_PROGRAM);
        prop_assert!(constraints_hold_with(&witness, seed.clone()));
        prop_assert!(unbalanced_with(&witness, seed.clone()).is_empty());

        // The first read of cell 2 returns something other than its committed start.
        let mut witness = witness;
        let read = &mut witness.rows[0].slots[1];
        read.old += value(forged);
        read.new = read.old;
        prop_assert!(constraints_hold_with(&witness, seed.clone()));
        prop_assert_eq!(unbalanced_with(&witness, seed), [MEMORY]);
    }

    #[test]
    fn the_sparse_image_evaluation_matches_the_dense_column(
        log_cells in 0usize..7,
        value_width in 1usize..3,
        runs in prop::collection::vec((0usize..6, 1usize..10), 0..5),
        seed in any::<u64>(),
    ) {
        // Lay the runs out left to right, with gaps between them, clipped to the image.
        let cells = 1usize << log_cells;
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut next = 0;
        let mut layout = Vec::new();
        for (gap, len) in runs {
            let start = next + gap;
            let len = len.min(cells.saturating_sub(start));
            if len == 0 {
                break;
            }
            let words = (0..len * value_width).map(|_| rng.random::<F>()).collect();
            layout.push((start, words));
            next = start + len;
        }
        let memory = Memory::new(MEMORY, LOW, HIGH, value_width).unwrap();
        let image = PublicImage::new(&memory, log_cells, layout).unwrap();

        // The dense reference sums every cell, zero or not.
        let point = (0..log_cells).map(|_| rng.random::<F>()).collect::<Vec<_>>();
        let weights = Point::new(point.as_slice()).equality_weights_msb();
        let dense = image
            .columns()
            .iter()
            .map(|column| column.iter().zip(&weights).map(|(&v, &w)| v * w).sum::<F>())
            .collect::<Vec<_>>();
        prop_assert_eq!(image.evaluate(&point), dense);
    }
}
