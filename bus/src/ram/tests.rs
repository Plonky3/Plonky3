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

/// Channel carrying the permutation between the two access orders.
const ORDER: &str = "ram-order";

/// Channel carrying the memory image a segment inherits.
const INCOMING: &str = "ram-incoming";

/// Channel carrying the memory image a segment hands on.
const OUTGOING: &str = "ram-outgoing";

/// Builds a self-contained memory statement.
fn single_proof(access_count: usize, address_bits: usize, value_width: usize) -> RamStatement {
    // The clock needs one value per access and nothing wider.
    RamStatement {
        access_bus: ACCESS.to_string(),
        order_bus: ORDER.to_string(),
        access_count,
        address_bits,
        timestamp_bits: access_count.trailing_zeros().max(1) as usize,
        value_width,
        boundary: RamBoundary::SingleProof,
    }
}

/// Builds a segment memory statement over the same dimensions.
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

/// Whether the AIR accepts a concrete trace.
///
/// The constraint checker panics on a violation, so this reports acceptance rather than
/// asserting it, which lets an attack test state its precondition before it states its verdict.
fn air_accepts(air: &RamAir, trace: &RamTrace<F>) -> bool {
    let main = RowMajorMatrix::new(trace.values().to_vec(), trace.width());
    catch_unwind(AssertUnwindSafe(|| check_constraints(air, &main, &[]))).is_ok()
}

/// Stand-in for the machine's chips, and for a segment's committed image tables.
///
/// Every declaration mirrors one of the memory's own, on the opposite side of the multiset and
/// over the same trace columns. That makes the counterparty exactly right by construction, which
/// is what an attack test wants: any imbalance it then reports comes from the memory's own
/// declarations rather than from a mismatched fixture.
struct Counterparty {
    /// Total trace width, shared with the memory under test.
    width: usize,
    /// One mirrored declaration per channel the memory under test uses.
    declarations: Vec<MirroredDeclaration>,
}

impl<F2> BaseAir<F2> for Counterparty {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: BusInteractionBuilder> Air<AB> for Counterparty {
    fn eval(&self, builder: &mut AB) {
        for (bus, direction, columns, gate) in &self.declarations {
            let (fields, activation) = {
                let main = builder.main();
                let row = main.current_slice();
                let fields = columns
                    .iter()
                    .map(|&column| row[column].into())
                    .collect::<Vec<AB::Expr>>();
                let activation = gate.map_or(BusActivation::Always, |(column, complement)| {
                    let selector: AB::Expr = row[column].into();
                    BusActivation::Boolean(if complement {
                        AB::Expr::ONE - selector
                    } else {
                        selector
                    })
                });
                (fields, activation)
            };
            builder.push_bus_interaction(bus, *direction, fields, activation);
        }
    }
}

/// One channel the counterparty mirrors: name, side, payload columns, and activation.
///
/// The activation names a trace column and says whether to take its complement, which is how a
/// group-opening declaration is mirrored against a group-closing one.
type MirroredDeclaration = (String, BusDirection, Vec<usize>, Option<(usize, bool)>);

/// Builds the counterparty for one memory, mirroring every channel it declares on.
fn counterparty(air: &RamAir) -> Counterparty {
    let statement = air.statement();
    let layout = air.layout();
    let mut declarations = vec![(
        statement.access_bus.clone(),
        BusDirection::Push,
        layout.execution_access_columns().collect(),
        None,
    )];
    if let RamBoundary::Segment { incoming, outgoing } = &statement.boundary {
        // The image tables sit on the opposite side of each image channel.
        declarations.push((
            incoming.clone(),
            BusDirection::Push,
            layout.image_columns().collect(),
            Some((layout.same_address, true)),
        ));
        declarations.push((
            outgoing.clone(),
            BusDirection::Pull,
            layout.image_columns().collect(),
            Some((layout.group_end, false)),
        ));
    }
    Counterparty {
        width: layout.width,
        declarations,
    }
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

/// Replays every named multiset this memory and its counterparty declare.
///
/// The incoming-image counterparty is gated by the memory's own group flag, so it mirrors
/// whatever the memory declared. That is deliberate: these tests use the report to ask whether
/// the *two access orders* agree, not whether a fixture image table is correct.
fn bus_report(air: &RamAir, trace: &RamTrace<F>) -> BusDebugReport<F> {
    let memory_profile = BusSymbolicBuilder::<F, F>::from_air(air, AirLayout::from_air::<F>(air));
    let source = counterparty(air);
    let source_profile =
        BusSymbolicBuilder::<F, F>::from_air(&source, AirLayout::from_air::<F>(&source));
    let table = table(trace);
    let instances = [
        BusDebugInstance::new(&table, None, &[], &memory_profile)
            .expect("the fixture trace matches the memory profile"),
        BusDebugInstance::new(&table, None, &[], &source_profile)
            .expect("the fixture trace matches the counterparty profile"),
    ];
    BusDebugReport::check(&instances).expect("the fixture declarations are well formed")
}

/// Whether every named multiset balances.
fn buses_balance(air: &RamAir, trace: &RamTrace<F>) -> bool {
    bus_report(air, trace).is_balanced()
}

/// Names of the multisets that do not balance.
fn unbalanced(air: &RamAir, trace: &RamTrace<F>) -> Vec<String> {
    bus_report(air, trace)
        .buses
        .into_iter()
        .map(|bus| bus.bus_name)
        .collect()
}

/// Builds a plan holding this memory and its counterparty, the way a machine would.
fn plan(air: &RamAir) -> BusPlan {
    let memory_profile = BusSymbolicBuilder::<F, F>::from_air(air, AirLayout::from_air::<F>(air));
    let source = counterparty(air);
    let source_profile =
        BusSymbolicBuilder::<F, F>::from_air(&source, AirLayout::from_air::<F>(&source));
    let log_height = air.statement().access_count.trailing_zeros() as usize;
    BusPlan::build(&[
        BusPlanInput {
            log_height,
            interactions: memory_profile.interactions(),
        },
        BusPlanInput {
            log_height,
            interactions: source_profile.interactions(),
        },
    ])
    .expect("the fixture uses a valid bus shape")
    .expect("the fixture declares at least one tuple")
}

/// Rewrites the address-sorted half of a trace to follow a caller-chosen order.
///
/// The comparison witness is filled modularly, so a descending step produces exactly the wrapped
/// ripple-adder witness an adversary would supply: every full-adder identity still holds and only
/// the refused carry out is left to reject the row.
fn reorder_memory(
    trace: &mut RamTrace<F>,
    statement: &RamStatement,
    accesses: &[RamAccess<F>],
    order: &[usize],
) {
    let clock = (0..accesses.len() as u64).collect::<Vec<_>>();
    reorder_memory_at(trace, statement, accesses, order, &clock);
}

/// Rewrites the address-sorted half of a trace under a caller-chosen order and clock.
fn reorder_memory_at(
    trace: &mut RamTrace<F>,
    statement: &RamStatement,
    accesses: &[RamAccess<F>],
    order: &[usize],
    clock: &[u64],
) {
    let layout = RamLayout::new(statement).expect("the fixture statement is well formed");
    for (sorted, &index) in order.iter().enumerate() {
        let access = &accesses[index];
        let previous = sorted.checked_sub(1).map(|earlier| order[earlier]);
        let same = previous.is_some_and(|earlier| accesses[earlier].address == access.address);
        let row = trace.row_mut(sorted);

        row[layout.memory_write] = F::from_bool(access.write);
        for bit in 0..layout.address_bits {
            row[layout.memory_address + bit] = F::from_bool((access.address >> bit) & 1 == 1);
        }
        for bit in 0..layout.timestamp_bits {
            row[layout.memory_timestamp + bit] = F::from_bool((clock[index] >> bit) & 1 == 1);
        }
        row[layout.memory_value..layout.memory_value + layout.value_width]
            .copy_from_slice(&access.value);
        row[layout.same_address] = F::from_bool(same);

        // Clear the shared comparison witness before the active comparison refills it.
        for bit in 0..layout.compare_bits {
            row[layout.compare_delta + bit] = F::ZERO;
            row[layout.compare_carry + bit] = F::ZERO;
            row[layout.compare_nonzero + bit] = F::ZERO;
        }
        row[layout.compare_carry + layout.compare_bits] = F::ZERO;

        let Some(earlier) = previous else { continue };
        let (left, right, bits) = if same {
            (clock[earlier], clock[index], layout.timestamp_bits)
        } else {
            (
                accesses[earlier].address,
                access.address,
                layout.address_bits,
            )
        };

        // A wrapping difference is the honest witness modulo the bit width.
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
        for sorted in 0..statement.access_count {
            let ends = sorted + 1 == statement.access_count
                || accesses[order[sorted + 1]].address != accesses[order[sorted]].address;
            trace.row_mut(sorted)[layout.group_end] = F::from_bool(ends);
        }
    }
}

/// Reads one little-endian bit column family out of a trace row as an integer.
fn read_bits(trace: &RamTrace<F>, row: usize, offset: usize, bits: usize) -> u64 {
    read_bits_in(trace, row, offset, bits)
}

/// Reads one little-endian bit column family out of a trace row over any field.
fn read_bits_in<K: Field>(trace: &RamTrace<K>, row: usize, offset: usize, bits: usize) -> u64 {
    (0..bits)
        .filter(|&bit| trace.row(row)[offset + bit] == K::ONE)
        .fold(0u64, |value, bit| value | 1 << bit)
}

#[test]
fn the_statement_refuses_every_shape_it_cannot_prove() {
    let base = single_proof(4, 3, 1);

    // A memory with no accesses has no first row and no product tree.
    assert_eq!(
        RamStatement {
            access_count: 0,
            ..base.clone()
        }
        .validate(),
        Err(RamError::EmptyTrace)
    );

    // The permutation rides on the plan's product tree, whose blocks are power-of-two aligned.
    assert_eq!(
        RamStatement {
            access_count: 3,
            ..base.clone()
        }
        .validate(),
        Err(RamError::NonPowerOfTwoAccessCount { access_count: 3 })
    );

    // Both bit widths become explicit columns compared by a ripple adder.
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
            order_bus: ACCESS.to_string(),
            ..base.clone()
        }
        .validate(),
        Err(RamError::DuplicateBus {
            name: ACCESS.to_string()
        })
    );
    assert_eq!(
        RamStatement {
            boundary: RamBoundary::Segment {
                incoming: ORDER.to_string(),
                outgoing: OUTGOING.to_string(),
            },
            ..base
        }
        .validate(),
        Err(RamError::DuplicateBus {
            name: ORDER.to_string()
        })
    );
}

#[test]
fn a_clock_too_narrow_to_count_the_accesses_is_refused_at_declaration() {
    // Eight accesses need eight distinct clock values.
    assert!(
        RamStatement {
            timestamp_bits: 3,
            ..single_proof(8, 3, 1)
        }
        .validate()
        .is_ok()
    );

    // Two bits count four. A ninth access would have to reuse a timestamp, and two accesses at
    // one address sharing a timestamp have no defined order, so a read could be matched against
    // the later write instead of the earlier one. The statement refuses that before any witness.
    assert_eq!(
        RamStatement {
            timestamp_bits: 2,
            ..single_proof(8, 3, 1)
        }
        .validate(),
        Err(RamError::TimestampCapacity {
            access_count: 8,
            timestamp_bits: 2,
            capacity: 4,
        })
    );
}

#[test]
fn every_column_family_occupies_its_own_range() {
    let statement = segment(8, 5, 2);
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // Every family is claimed once, and the claimed ranges tile the trace exactly.
    let families = [
        (layout.execution_write, 1),
        (layout.execution_address, 5),
        (layout.execution_timestamp, 3),
        (layout.execution_value, 2),
        (layout.execution_carry, 3),
        (layout.memory_write, 1),
        (layout.memory_address, 5),
        (layout.memory_timestamp, 3),
        (layout.memory_value, 2),
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

    // A single proof exports no image, so it allocates no group marker.
    let lean =
        RamLayout::new(&single_proof(8, 5, 2)).expect("the fixture statement is well formed");
    assert_eq!(lean.width, layout.width - 1);
}

#[test]
fn an_honest_memory_satisfies_every_constraint_and_balances_every_channel() {
    // Write, read back, write again at one address, then touch a second address.
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, value(11)),
        RamAccess::read(5, value(11)),
        RamAccess::write(5, value(13)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    let trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let air = air(statement);

    // Both access orders live in one matrix of the AIR's width.
    assert_eq!(trace.width(), BaseAir::<F>::width(&air));
    assert_eq!(trace.height(), 4);

    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));
}

#[test]
fn the_execution_clock_is_the_row_index() {
    let statement = single_proof(8, 4, 1);
    let accesses = (0..8)
        .map(|row| RamAccess::write(row, value(row + 3)))
        .collect::<Vec<_>>();
    let trace = statement
        .build_trace(&accesses)
        .expect("distinct addresses need no continuity");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // The clock is what makes timestamps unique, so it is worth reading back directly.
    for row in 0..8 {
        assert_eq!(
            read_bits(
                &trace,
                row,
                layout.execution_timestamp,
                layout.timestamp_bits
            ),
            row as u64
        );
    }
    assert!(air_accepts(&air(statement), &trace));
}

#[test]
fn the_sorted_order_groups_addresses_and_orders_time_inside_a_group() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(6, value(11)),
        RamAccess::write(1, value(13)),
        RamAccess::read(6, value(11)),
        RamAccess::read(1, value(13)),
    ];
    let trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");

    // Address 1's two accesses come first, in timestamp order, then address 6's two.
    let sorted = (0..4)
        .map(|row| {
            (
                read_bits(&trace, row, layout.memory_address, layout.address_bits),
                read_bits(&trace, row, layout.memory_timestamp, layout.timestamp_bits),
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
fn a_non_boolean_address_digit_is_refused() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, value(11)),
        RamAccess::read(5, value(11)),
        RamAccess::write(3, value(13)),
        RamAccess::read(3, value(13)),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Corrupt the same digit on both sides so the permutation still matches tuple for tuple.
    let forged = F::GENERATOR;
    for row in 0..4 {
        trace.row_mut(row)[layout.execution_address] = forged;
        trace.row_mut(row)[layout.memory_address] = forged;
    }

    // Precondition: the multiset claims all still hold, so nothing but the digit check can reject.
    assert!(buses_balance(&air, &trace));

    // A non-Boolean digit makes the ripple-adder comparison meaningless, so ordering stops
    // constraining anything and an adversary can place rows where it likes.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_non_boolean_timestamp_digit_is_refused() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, value(11)),
        RamAccess::read(5, value(11)),
        RamAccess::read(5, value(11)),
        RamAccess::read(5, value(11)),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    let forged = F::GENERATOR;
    for row in 0..4 {
        trace.row_mut(row)[layout.execution_timestamp] = forged;
        trace.row_mut(row)[layout.memory_timestamp] = forged;
    }

    // Precondition: every multiset claim survives the corruption.
    assert!(buses_balance(&air, &trace));

    // Without Boolean digits the clock and the within-group order both stop meaning anything.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_repeated_execution_timestamp_is_refused() {
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, value(11)),
        RamAccess::write(5, value(13)),
        RamAccess::read(5, value(13)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Give row one the same timestamp as row zero, on both sides so the permutation survives.
    for row in [0usize, 1] {
        for bit in 0..layout.timestamp_bits {
            trace.row_mut(row)[layout.execution_timestamp + bit] = F::ZERO;
        }
    }
    let sorted_rows = (0..4)
        .filter(|&row| read_bits(&trace, row, layout.memory_timestamp, layout.timestamp_bits) <= 1)
        .collect::<Vec<_>>();
    for row in sorted_rows {
        for bit in 0..layout.timestamp_bits {
            trace.row_mut(row)[layout.memory_timestamp + bit] = F::ZERO;
        }
    }

    // Precondition: the two orders still hold the same tuples.
    assert!(buses_balance(&air, &trace));

    // Two accesses sharing a timestamp have no order at their address, so the sorted trace could
    // put either first and the read would see whichever write the prover preferred.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn splitting_one_address_into_two_groups_is_refused() {
    // One write followed by one read of the same cell.
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, value(11)),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());
    assert!(air_accepts(&air, &trace));

    // Claim the read opens a new group at the same address, which re-initialises it to zero.
    let forged = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, vec![F::ZERO]),
    ];
    let mut execution = trace.row(1).to_vec();
    execution[layout.execution_value] = F::ZERO;
    trace.row_mut(1).copy_from_slice(&execution);
    reorder_memory(&mut trace, &statement, &forged, &[0, 1]);

    // Claim a new group at the same address, with the zero difference that admits.
    trace.row_mut(1)[layout.same_address] = F::ZERO;
    for bit in 0..layout.compare_bits {
        trace.row_mut(1)[layout.compare_delta + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_carry + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ZERO;
    }
    trace.row_mut(1)[layout.compare_carry + layout.compare_bits] = F::ZERO;

    // Precondition: the forged group flag really is zero on two rows holding one address.
    assert_eq!(trace.row(1)[layout.same_address], F::ZERO);
    assert_eq!(
        read_bits(&trace, 0, layout.memory_address, layout.address_bits),
        read_bits(&trace, 1, layout.memory_address, layout.address_bits),
    );

    // Precondition: both multiset claims still hold, so only the ordering check can reject.
    assert!(buses_balance(&air, &trace));

    // The strict increase needs a nonzero difference, and a repeated address has none. Without
    // it the read returns the initial zero rather than the value the write left.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_wrapped_address_comparison_is_refused() {
    // Two reads of untouched cells, so nothing but the ordering is in play.
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::read(3, vec![F::ZERO]),
        RamAccess::read(1, vec![F::ZERO]),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("reads of untouched cells return zero");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());
    assert!(air_accepts(&air, &trace));

    // Sort descending instead: address three, then address one.
    reorder_memory(&mut trace, &statement, &accesses, &[0, 1]);

    // Precondition: every full-adder identity of the comparison still holds. Three plus the
    // witnessed difference really is one, modulo four, and the difference is nonzero.
    let left = read_bits(&trace, 0, layout.memory_address, layout.address_bits);
    let right = read_bits(&trace, 1, layout.memory_address, layout.address_bits);
    let delta = read_bits(&trace, 1, layout.compare_delta, layout.address_bits);
    assert_eq!(left, 3);
    assert_eq!(right, 1);
    assert_eq!((left + delta) % 4, right);
    assert_ne!(delta, 0);
    assert_eq!(
        trace.row(1)[layout.compare_nonzero + layout.address_bits - 1],
        F::ONE
    );

    // Precondition: the wrap is the one thing left. The carry out of the top bit is set.
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.address_bits],
        F::ONE
    );

    // Precondition: both multiset claims still hold.
    assert!(buses_balance(&air, &trace));

    // Refusing the carry out is what makes the comparison unsigned. Without it the sorted trace
    // need not be sorted, address groups stop being contiguous, and a read can be moved into a
    // group that never saw the write it should have.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_read_placed_before_the_write_it_should_see_is_refused() {
    // Two writes then a read at one address, plus a read of an untouched cell to fill the trace.
    let statement = single_proof(4, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::write(1, value(13)),
        RamAccess::read(1, value(13)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the read returns the value the last write left");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());
    assert!(air_accepts(&air, &trace));

    // Swap the two writes inside the group so the surviving value is the earlier one, and make
    // the read return that earlier value on both sides.
    let forged = vec![
        RamAccess::write(1, value(11)),
        RamAccess::write(1, value(13)),
        RamAccess::read(1, value(11)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    trace.row_mut(2)[layout.execution_value] = value(11)[0];
    reorder_memory(&mut trace, &statement, &forged, &[1, 0, 2, 3]);

    // Precondition: read continuity now holds. Row one leaves value eleven and row two reads it.
    assert_eq!(trace.row(1)[layout.memory_value], value(11)[0]);
    assert_eq!(trace.row(2)[layout.memory_value], value(11)[0]);
    assert_eq!(trace.row(2)[layout.same_address], F::ONE);

    // Precondition: both multiset claims still hold.
    assert!(buses_balance(&air, &trace));

    // Only the within-group timestamp order is left, and it has wrapped from one to zero. That
    // order is the whole reason a read sees the *last* write rather than any write.
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.timestamp_bits],
        F::ONE
    );
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_read_that_ignores_the_last_write_is_refused() {
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, value(11)),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Return a value the cell never held, on both sides so the permutation survives.
    let forged = value(99)[0];
    trace.row_mut(1)[layout.execution_value] = forged;
    trace.row_mut(1)[layout.memory_value] = forged;

    // Precondition: the ordering witness is untouched and both multiset claims still hold.
    assert_eq!(trace.row(1)[layout.same_address], F::ONE);
    assert!(buses_balance(&air, &trace));

    // Read continuity is the only thing that makes a read observe memory at all.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_read_of_an_untouched_cell_must_return_zero() {
    let statement = single_proof(2, 2, 1);

    // Witness generation refuses it outright, with the offending sorted row named.
    let accesses = vec![
        RamAccess::read(1, value(11)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    assert_eq!(
        statement.build_trace(&accesses),
        Err(RamError::ReadContinuity { index: 0 })
    );

    // So does the AIR, on a trace forged past it.
    let honest = vec![
        RamAccess::read(1, vec![F::ZERO]),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    let mut trace = statement
        .build_trace(&honest)
        .expect("reads of untouched cells return zero");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    let forged = value(11)[0];
    trace.row_mut(0)[layout.execution_value] = forged;
    trace.row_mut(0)[layout.memory_value] = forged;

    // Precondition: the forged row opens its group, and both multiset claims still hold.
    assert_eq!(trace.row(0)[layout.same_address], F::ZERO);
    assert!(buses_balance(&air, &trace));

    // Initialization is what stops a proof inventing the memory it starts from.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn the_permutation_is_what_catches_an_access_the_machine_never_issued() {
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, value(11)),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);

    // Rewrite the sorted trace's value only, leaving the execution order alone.
    let forged = value(99)[0];
    trace.row_mut(0)[layout.memory_value] = forged;
    trace.row_mut(1)[layout.memory_value] = forged;

    // Precondition: every constraint still holds. The sorted trace is internally consistent, so
    // the AIR alone cannot tell that it describes a different computation.
    assert!(air_accepts(&air, &trace));

    // The permutation channel is what rejects it, and it names exactly that channel.
    assert_eq!(unbalanced(&air, &trace), vec![ORDER.to_string()]);
}

#[test]
fn a_segment_opens_every_address_group_against_the_incoming_image() {
    // Each address group opens with a read carrying the value the segment inherits.
    let statement = segment(4, 3, 1);
    let accesses = vec![
        RamAccess::read(5, value(11)),
        RamAccess::write(5, value(13)),
        RamAccess::read(2, value(17)),
        RamAccess::read(2, value(17)),
    ];
    let trace = statement
        .build_trace(&accesses)
        .expect("every group opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);

    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));

    // Two addresses, so two opening reads and two closing values.
    let openings = (0..4)
        .filter(|&row| trace.row(row)[layout.same_address] == F::ZERO)
        .count();
    let closings = (0..4)
        .filter(|&row| trace.row(row)[layout.group_end] == F::ONE)
        .count();
    assert_eq!((openings, closings), (2, 2));

    // The outgoing image carries the last value at each address, which for cell five is the
    // written value rather than the inherited one.
    let last_of_five = (0..4)
        .find(|&row| {
            trace.row(row)[layout.group_end] == F::ONE
                && read_bits(&trace, row, layout.memory_address, layout.address_bits) == 5
        })
        .expect("cell five has a closing row");
    assert_eq!(trace.row(last_of_five)[layout.memory_value], value(13)[0]);
}

#[test]
fn a_segment_group_opened_by_a_write_is_refused() {
    let statement = segment(2, 2, 1);

    // A write opening a group would let the segment skip reading what it inherited.
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, value(11)),
    ];
    assert_eq!(
        statement.build_trace(&accesses),
        Err(RamError::UnopenedSegmentGroup { index: 0 })
    );

    // The AIR refuses the same thing on a trace forged past witness generation.
    let honest = vec![
        RamAccess::read(1, value(11)),
        RamAccess::write(1, value(13)),
    ];
    let mut trace = statement
        .build_trace(&honest)
        .expect("the group opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    trace.row_mut(0)[layout.execution_write] = F::ONE;
    trace.row_mut(0)[layout.memory_write] = F::ONE;

    // Precondition: both multiset claims still hold.
    assert!(buses_balance(&air, &trace));

    // Forcing the opening access to be a read is what binds the inherited value to the image.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_forged_group_marker_is_refused() {
    let statement = segment(2, 2, 1);
    let accesses = vec![RamAccess::read(1, value(11)), RamAccess::read(1, value(11))];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the group opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Claim the first row closes its group, which would export a value the group later changed.
    trace.row_mut(0)[layout.group_end] = F::ONE;

    // Precondition: the marker really does disagree with the next row's group flag.
    assert_eq!(trace.row(1)[layout.same_address], F::ONE);

    // The marker is pinned to the next row's group flag, so it cannot be chosen.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_static_indexed_table_keeps_the_lookup_argument() {
    // One AIR reads a static table through the existing read-only helper.
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

    // A mutable memory cannot be pointed at it. Its tuple has no operation marker and no
    // timestamp, because an immutable entry has neither, so the widths cannot agree.
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

    // The memory's own channels are there.
    assert_eq!(air.statement().check_against(&bus_plan), Ok(()));

    // A name nothing declares on is not a channel at all.
    let statement = RamStatement {
        order_bus: "absent".to_string(),
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

    // A 128-bit challenge field and a 31-bit one give different bounds for one statement, and
    // the difference is the field size rather than anything about the memory.
    let wide = statement
        .security_term::<BinaryField128>(&bus_plan)
        .expect("the fixture plan defines every channel");
    let narrow = statement
        .security_term::<BabyBear>(&bus_plan)
        .expect("the fixture plan defines every channel");
    assert!(wide.bits.bits() > narrow.bits.bits() + 90.0);

    // The term is the enclosing plan's own, evaluated at the challenge field's size, so a report
    // that already carries the plan's term must not add this one on top.
    let field_bits =
        core::num::NonZeroUsize::new(BinaryField128::order().bits() as usize - 1).unwrap();
    assert_eq!(wide, bus_plan.security_term(field_bits));

    // The leaf contribution is what a machine designer trades against that bound.
    assert_eq!(statement.leaf_contribution(), [8, 16]);
    assert_eq!(segment(8, 4, 1).leaf_contribution(), [16, 24]);
}

#[test]
fn witness_generation_refuses_a_witness_that_does_not_match_the_statement() {
    let statement = single_proof(2, 2, 1);

    // The access count is public, so a mismatch is the caller's mistake.
    assert_eq!(
        statement.build_trace::<F>(&[]),
        Err(RamError::AccessCount {
            expected: 2,
            actual: 0
        })
    );

    // So is the value width.
    assert_eq!(
        statement.build_trace(&[
            RamAccess::read(0, vec![F::ZERO, F::ZERO]),
            RamAccess::read(1, vec![F::ZERO]),
        ]),
        Err(RamError::ValueWidth {
            index: 0,
            expected: 1,
            actual: 2
        })
    );

    // An address wider than the statement has no bit decomposition to commit.
    assert_eq!(
        statement.build_trace(&[
            RamAccess::read(0, vec![F::ZERO]),
            RamAccess::read(9, vec![F::ZERO]),
        ]),
        Err(RamError::AddressRange {
            index: 1,
            address: 9,
            address_bits: 2
        })
    );

    // And a read that ignores the last write is named before any constraint runs.
    assert_eq!(
        statement.build_trace(&[
            RamAccess::write(1, value(11)),
            RamAccess::read(1, value(13)),
        ]),
        Err(RamError::ReadContinuity { index: 1 })
    );
}

#[test]
fn the_smallest_memory_proves_and_balances() {
    // One access to one cell of one bit is the smallest legal statement.
    let statement = single_proof(1, 1, 1);
    let trace = statement
        .build_trace(&[RamAccess::write(1, value(11))])
        .expect("a single write needs no continuity");
    let air = air(statement);

    assert_eq!(trace.height(), 1);
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
        RamAccess::write(40, word(11)),
        RamAccess::read(40, word(11)),
        RamAccess::write(40, word(21)),
        RamAccess::read(40, word(21)),
    ];
    let trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
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
        layout.execution_access_columns().count(),
        statement.access_payload_width()
    );
    assert_eq!(
        layout.memory_access_columns().count(),
        statement.access_payload_width()
    );
    assert_eq!(
        layout.image_columns().count(),
        statement.image_payload_width()
    );

    // Both orders read the same tuple shape, offset by their own block.
    let execution = layout.execution_access_columns().collect::<Vec<_>>();
    let memory = layout.memory_access_columns().collect::<Vec<_>>();
    let shift = layout.memory_write - layout.execution_write;
    assert_eq!(
        memory,
        execution
            .iter()
            .map(|&column| column + shift)
            .collect::<Vec<_>>(),
        "{execution:?} and {memory:?} must describe one tuple order",
    );

    // A segment names both image channels; a single proof names none.
    assert_eq!(statement.boundary.image_buses(), Some([INCOMING, OUTGOING]));
    assert_eq!(RamBoundary::SingleProof.image_buses(), None);
}

#[test]
fn the_constraints_hold_over_an_odd_characteristic_field_too() {
    // Exclusive or and majority both carry correction terms that vanish in characteristic two:
    // `a + b - 2ab` and `ab + ac + bc - 2abc`. A binary-only test accepts a wrong correction, so
    // the same memory runs over a prime field too.
    //
    // The schedule is chosen so the carry chain actually reaches the inputs that separate the two
    // formulas. Majority differs from `ab + ac + bc` only when all three inputs are one, and from
    // the near-miss `ab + ac + bc - 2ac` only when the middle input is zero while the other two
    // are one. Adding three to four supplies both: bit zero carries out of `1 + 1`, and bit one
    // then sees a set addend, a clear difference, and an incoming carry.
    let word = |tag: u32| vec![BabyBear::from_u32(tag)];
    let statement = single_proof(8, 3, 1);
    let accesses = vec![
        RamAccess::write(4, word(11)),
        RamAccess::read(4, word(11)),
        RamAccess::write(6, word(13)),
        RamAccess::write(3, word(17)),
        RamAccess::write(7, word(19)),
        RamAccess::read(7, word(19)),
        RamAccess::read(3, word(17)),
        RamAccess::read(6, word(13)),
    ];
    let trace = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let air = air(statement);
    let layout = air.layout();

    // Precondition: the two separating input patterns really occur.
    //
    // Cell three's group spans timestamps three and six, so that comparison adds three to three
    // and bit one sees all three inputs set. Cell four's group opens right after cell three's, so
    // that comparison adds one to three and bit one sees a set addend, a clear difference, and an
    // incoming carry. Without both, a wrong correction term would go unnoticed.
    assert_eq!(
        read_bits_in(&trace, 1, layout.memory_timestamp, layout.timestamp_bits),
        6
    );
    assert_eq!(trace.row(1)[layout.same_address], BabyBear::ONE);
    assert_eq!(
        read_bits_in(&trace, 1, layout.compare_delta, layout.timestamp_bits),
        3
    );
    assert_eq!(
        read_bits_in(&trace, 2, layout.memory_address, layout.address_bits),
        4
    );
    assert_eq!(trace.row(2)[layout.same_address], BabyBear::ZERO);
    assert_eq!(
        read_bits_in(&trace, 2, layout.compare_delta, layout.address_bits),
        1
    );

    let main = RowMajorMatrix::new(trace.values().to_vec(), trace.width());
    check_constraints(&air, &main, &[]);

    // The ripple adder must still reject a wrapped comparison here.
    let mut forged = trace;
    forged.row_mut(2)[layout.compare_carry + layout.address_bits] = BabyBear::ONE;
    let main = RowMajorMatrix::new(forged.values().to_vec(), forged.width());
    assert!(catch_unwind(AssertUnwindSafe(|| check_constraints(&air, &main, &[]))).is_err());
}

#[test]
fn a_long_random_schedule_proves_and_balances() {
    // Sixty-four accesses over sixteen cells exercise every group size and both comparisons.
    let statement = single_proof(64, 4, 1);
    let mut rng = Xoroshiro128Plus::seed_from_u64(0x5ea1_1eaf);
    let mut memory = vec![F::ZERO; 16];
    let accesses = (0..64)
        .map(|_| {
            let address = u64::from(rng.random::<u8>() % 16);
            if rng.random::<bool>() {
                let fresh: F = rng.random();
                memory[address as usize] = fresh;
                RamAccess::write(address, vec![fresh])
            } else {
                RamAccess::read(address, vec![memory[address as usize]])
            }
        })
        .collect::<Vec<_>>();
    let trace = statement
        .build_trace(&accesses)
        .expect("the schedule tracks its own memory");
    let air = air(statement);

    assert!(air_accepts(&air, &trace));
    assert!(buses_balance(&air, &trace));

    // Every mutation of a single sorted value is caught by one of the two mechanisms.
    let layout = air.layout();
    for row in 0..64 {
        let mut forged = trace.clone();
        forged.row_mut(row)[layout.memory_value] += F::ONE;
        assert!(
            !air_accepts(&air, &forged) || !buses_balance(&air, &forged),
            "row {row} was mutated without being rejected",
        );
    }
}

#[test]
fn an_execution_clock_that_does_not_start_at_zero_is_refused() {
    // The machine writes a cell and then reads it back.
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, value(11)),
    ];
    let honest = statement
        .build_trace(&accesses)
        .expect("the read returns what the write left");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());
    assert!(air_accepts(&air, &honest));

    // Start the clock one short of its capacity, so it wraps at the first step: 1, then 0. The
    // two timestamps are still distinct, and the increment chain still adds one at every step.
    //
    // The wrap reverses the sorted order inside the group. The read now sorts first, becomes the
    // group opener, and so returns the initial zero rather than the value the write left.
    let forged_accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, vec![F::ZERO]),
    ];
    let clock = [1u64, 0];
    let mut trace = honest;
    for (row, &shifted) in clock.iter().enumerate() {
        for bit in 0..layout.timestamp_bits {
            trace.row_mut(row)[layout.execution_timestamp + bit] =
                F::from_bool((shifted >> bit) & 1 == 1);
        }
        trace.row_mut(row)[layout.execution_carry] = F::ONE;
    }
    trace.row_mut(1)[layout.execution_value] = F::ZERO;
    reorder_memory_at(&mut trace, &statement, &forged_accesses, &[1, 0], &clock);

    // Precondition: the two timestamps are distinct, so uniqueness alone does not object.
    let clocks = (0..2)
        .map(|row| {
            read_bits(
                &trace,
                row,
                layout.execution_timestamp,
                layout.timestamp_bits,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(clocks, vec![1, 0]);

    // Precondition: the sorted order is internally consistent. Its first row opens the group and
    // reads zero, and its second row follows at a strictly later timestamp.
    assert_eq!(trace.row(0)[layout.same_address], F::ZERO);
    assert_eq!(trace.row(0)[layout.memory_value], F::ZERO);
    assert_eq!(trace.row(1)[layout.same_address], F::ONE);
    assert_eq!(
        trace.row(1)[layout.compare_carry + layout.timestamp_bits],
        F::ZERO
    );

    // Precondition: both multiset claims still hold.
    assert!(buses_balance(&air, &trace));

    // Pinning the first timestamp to zero is the only thing left, and together with the
    // statement's capacity check it is what forbids a wrap inside the trace. A wrap reorders a
    // group against the order the machine actually executed it in.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn a_group_continued_at_a_different_address_is_refused() {
    // A write to one cell, then a read of a cell nothing has written.
    let statement = single_proof(2, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("a read of an untouched cell returns zero");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Claim the read continues the write's group, which hands it the written value.
    let stolen = value(11)[0];
    trace.row_mut(1)[layout.execution_value] = stolen;
    trace.row_mut(1)[layout.memory_value] = stolen;
    trace.row_mut(1)[layout.same_address] = F::ONE;

    // Refill the witness as the timestamp comparison the group flag now selects.
    for bit in 0..layout.compare_bits {
        trace.row_mut(1)[layout.compare_delta + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_carry + bit] = F::ZERO;
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ZERO;
    }
    trace.row_mut(1)[layout.compare_carry + layout.compare_bits] = F::ZERO;
    for bit in 0..layout.timestamp_bits {
        let difference = (1u64 >> bit) & 1;
        trace.row_mut(1)[layout.compare_delta + bit] = F::from_bool(difference == 1);
        trace.row_mut(1)[layout.compare_nonzero + bit] = F::ONE;
    }

    // Precondition: read continuity now holds, because the row it copies from really does hold
    // that value, and the timestamp comparison holds too.
    assert_eq!(trace.row(0)[layout.memory_value], stolen);
    assert_eq!(trace.row(1)[layout.memory_value], stolen);
    assert_eq!(
        read_bits(&trace, 1, layout.compare_delta, layout.timestamp_bits),
        1
    );

    // Precondition: both multiset claims still hold.
    assert!(buses_balance(&air, &trace));

    // Only the same-group address equality is left. Without it a read of one cell could inherit
    // another cell's value, which is a read of the wrong address.
    assert_ne!(
        read_bits(&trace, 0, layout.memory_address, layout.address_bits),
        read_bits(&trace, 1, layout.memory_address, layout.address_bits),
    );
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn the_first_sorted_row_cannot_claim_to_continue_a_group() {
    // In segment mode the group flag gates the pull against the incoming image.
    let statement = segment(2, 2, 1);
    let accesses = vec![RamAccess::read(1, value(11)), RamAccess::read(1, value(11))];
    let mut trace = statement
        .build_trace(&accesses)
        .expect("the group opens with a read");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &trace));

    // Claim the first row continues a group that does not exist.
    trace.row_mut(0)[layout.same_address] = F::ONE;

    // Precondition: nothing else changed, so the two access orders still hold the same tuples.
    assert_eq!(
        unbalanced(&air, &trace)
            .into_iter()
            .filter(|bus| bus == ORDER || bus == ACCESS)
            .count(),
        0
    );

    // The first row always opens a group. Letting it say otherwise would skip its pull against
    // the incoming image, and the segment could then invent the value it inherits.
    assert!(!air_accepts(&air, &trace));
}

#[test]
fn every_derived_witness_column_is_pinned_by_its_own_identity() {
    // The comparison difference, its carry chain, its running-or chain, and the clock's carry
    // chain are all prover-supplied. Each is pinned by an identity rather than merely permitted,
    // and a prover free to choose any of them could choose one that makes a false order pass.
    //
    // Equal address and timestamp widths keep every shared witness column active on every row,
    // so nothing below is flipped in a column that no constraint reads.
    let statement = single_proof(4, 2, 1);
    let accesses = vec![
        RamAccess::write(1, value(11)),
        RamAccess::read(1, value(11)),
        RamAccess::write(2, value(13)),
        RamAccess::read(2, value(13)),
    ];
    let honest = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement);
    assert!(air_accepts(&air, &honest));
    assert_eq!(layout.compare_bits, layout.address_bits);
    assert_eq!(layout.compare_bits, layout.timestamp_bits);

    // Comparison witness columns are read on every row after the first.
    let comparison = (layout.compare_delta..layout.compare_delta + layout.compare_bits)
        .chain(layout.compare_carry..layout.compare_carry + layout.compare_bits + 1)
        .chain(layout.compare_nonzero..layout.compare_nonzero + layout.compare_bits)
        .collect::<Vec<_>>();
    for row in 1..4 {
        for &column in &comparison {
            let mut forged = honest.clone();
            forged.row_mut(row)[column] = F::ONE - forged.row(row)[column];
            assert!(
                !air_accepts(&air, &forged),
                "comparison column {column} of row {row} is not pinned",
            );
        }
    }

    // The clock's carry chain is read on every row that has a successor, and the timestamp it
    // produces is pinned just as tightly: the whole point of the clock is that the prover has no
    // say in it.
    for row in 0..4 {
        let carries = if row < 3 {
            layout.execution_carry..layout.execution_carry + layout.timestamp_bits
        } else {
            0..0
        };
        let timestamps =
            layout.execution_timestamp..layout.execution_timestamp + layout.timestamp_bits;
        for column in carries.chain(timestamps) {
            let mut forged = honest.clone();
            forged.row_mut(row)[column] = F::ONE - forged.row(row)[column];
            assert!(
                !air_accepts(&air, &forged),
                "clock column {column} of row {row} is not pinned",
            );
        }
    }
}

#[test]
fn a_clock_whose_carry_chain_is_chosen_freely_can_repeat_a_timestamp() {
    // Four accesses to four cells, so the ordering constraints never look at a timestamp and the
    // clock is the only thing under test.
    let statement = single_proof(4, 2, 1);
    let accesses = (0..4)
        .map(|cell| RamAccess::write(cell, value(cell + 11)))
        .collect::<Vec<_>>();
    let honest = statement
        .build_trace(&accesses)
        .expect("distinct cells need no continuity");
    let layout = RamLayout::new(&statement).expect("the fixture statement is well formed");
    let air = air(statement.clone());
    assert!(air_accepts(&air, &honest));

    // The clock runs 0, 1, 0, 3 instead of 0, 1, 2, 3. It starts at zero, and every step still
    // flips the low bit, because the carry into the low bit is fixed at one. The carries above
    // it are the only freedom, and they are exactly what the recurrence removes.
    let clock = [0u64, 1, 0, 3];
    let mut trace = honest;
    for (row, &tick) in clock.iter().enumerate() {
        for bit in 0..layout.timestamp_bits {
            trace.row_mut(row)[layout.execution_timestamp + bit] =
                F::from_bool((tick >> bit) & 1 == 1);
        }
    }
    for row in 0..3 {
        for bit in 0..layout.timestamp_bits {
            let flips = ((clock[row] ^ clock[row + 1]) >> bit) & 1;
            trace.row_mut(row)[layout.execution_carry + bit] = F::from_bool(flips == 1);
        }
    }
    reorder_memory_at(&mut trace, &statement, &accesses, &[0, 1, 2, 3], &clock);

    // Precondition: the clock starts at zero and every step's sum identity holds.
    assert_eq!(
        read_bits(&trace, 0, layout.execution_timestamp, layout.timestamp_bits),
        0
    );
    for row in 0..3 {
        assert_eq!(trace.row(row)[layout.execution_carry], F::ONE);
        for bit in 0..layout.timestamp_bits {
            let current = trace.row(row)[layout.execution_timestamp + bit];
            let carry = trace.row(row)[layout.execution_carry + bit];
            let next = trace.row(row + 1)[layout.execution_timestamp + bit];
            assert_eq!(next, current + carry - F::TWO * current * carry);
        }
    }

    // Precondition: the two access orders still hold the same tuples.
    assert!(buses_balance(&air, &trace));

    // The carry recurrence is what is left, and it is what stops the clock repeating a value.
    // Two accesses at one address sharing a timestamp would have no order between them.
    assert_eq!(clock[0], clock[2]);
    assert!(!air_accepts(&air, &trace));
}

/// Materializes one side of a plan's product tree from concrete traces.
///
/// Leaves follow the plan's physical block order, which is what its reduction expects.
fn materialize(
    bus_plan: &BusPlan,
    profiles: &[&BusSymbolicBuilder<F, F>],
    trace: &RamTrace<F>,
    challenges: &BusChallenges<F>,
    direction: BusDirection,
) -> Vec<F> {
    let weights = challenges.fingerprint_weights();
    let height = trace.height();
    let mut leaves = Vec::new();
    for block in bus_plan.blocks(direction) {
        let interaction = &profiles[block.owner.air].interactions()[block.owner.declaration];
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
fn the_plan_reduction_proves_the_two_orders_hold_the_same_accesses() {
    // A machine would reach this through its own prover. The point here is that the permutation
    // needs no protocol of its own: the plan's ordinary multiset reduction discharges it.
    let statement = single_proof(4, 3, 1);
    let accesses = vec![
        RamAccess::write(5, value(11)),
        RamAccess::read(5, value(11)),
        RamAccess::write(5, value(13)),
        RamAccess::read(2, vec![F::ZERO]),
    ];
    let honest = statement
        .build_trace(&accesses)
        .expect("the fixture accesses are consistent");
    let air = air(statement);
    let layout = air.layout();

    let memory_profile = BusSymbolicBuilder::<F, F>::from_air(&air, AirLayout::from_air::<F>(&air));
    let source = counterparty(&air);
    let source_profile =
        BusSymbolicBuilder::<F, F>::from_air(&source, AirLayout::from_air::<F>(&source));
    let profiles = [&memory_profile, &source_profile];
    let bus_plan = plan(&air);

    // Challenges are drawn inside the reduction, after whatever the caller already bound into
    // the transcript, which for a real prover is the commitment to this very trace.
    let reduce = |trace: &RamTrace<F>| {
        let mut challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash);
        bus_plan.prove::<F, F, _>(
            |challenges| {
                BusDirection::ALL.map(|direction| {
                    materialize(&bus_plan, &profiles, trace, challenges, direction)
                })
            },
            &mut challenger,
        )
    };

    let (proof, prover_output) = reduce(&honest).expect("the honest memory balances");
    let mut verifier_challenger = Challenger::from_hasher(Vec::new(), Keccak256Hash);
    let verifier_output = bus_plan
        .verify::<F, F, _>(&proof, &mut verifier_challenger)
        .expect("the honest reduction verifies");
    assert_eq!(prover_output.product.point, verifier_output.product.point);
    assert_eq!(prover_output.product.values, verifier_output.product.values);

    // Rewrite one value in the sorted order only. The constraints cannot see it, because the
    // sorted trace stays internally consistent.
    let mut forged = honest;
    for row in 0..4 {
        if forged.row(row)[layout.memory_value] == value(11)[0] {
            forged.row_mut(row)[layout.memory_value] = value(99)[0];
        }
    }
    assert!(air_accepts(&air, &forged));

    // The reduction refuses to prove it, because the two orders no longer hold the same accesses.
    assert_eq!(reduce(&forged), Err(BusArgumentError::UnbalancedProducts));
}
