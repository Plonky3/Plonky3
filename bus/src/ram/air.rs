//! Verifier-enforced constraints for one mutable read-write memory.

use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Algebra, PrimeCharacteristicRing};

use super::{RamBoundary, RamError, RamLayout, RamStatement};
use crate::{BusActivation, BusDirection, BusInteractionBuilder};

/// Constraints and channel declarations of one mutable read-write memory.
///
/// One committed trace holds both access orders.
///
/// Every constraint below is checked by the enclosing proof system.
///
/// Every claim spanning the two orders is a named-channel declaration the plan balances.
///
/// Nothing is left as a handoff a caller has to remember.
///
/// # What each constraint buys
///
/// The digit columns are what make the comparisons mean anything.
///
/// A digit holding something other than a bit empties out "the gap is nonzero and did not wrap".
///
/// An adversary can then order rows however it likes.
///
/// A unique, non-wrapping clock is what makes the sorted order total.
///
/// If two accesses shared a reading, two of them touching one cell would have no order.
///
/// A read could then be matched against the later write instead of the earlier one.
///
/// An unsigned comparison is what keeps a cell's accesses together in one run.
///
/// If a cell number could wrap, one cell would split into two runs.
///
/// The second run starts over, so a read returns the starting value rather than what was written.
///
/// A nonzero gap is the same protection without the wrap.
///
/// Without it two rows could hold one cell number while claiming a fresh run.
///
/// Read continuity is what makes a read see the last write.
///
/// A write is left unconstrained in value, which is how the next read inherits it.
///
/// The two boundaries tie this proof's memory to the world around it.
///
/// # Which constraints are load-bearing
///
/// Deleting a constraint and re-running the tests sorts them into two kinds.
///
/// Some carry the argument; the rest restate something another constraint already forces.
///
/// Removing any of these makes a test in this module fail:
///
/// - the sorted order's cell digits;
/// - the clock's zero start, its sum identity, and its carry recurrence;
/// - the comparison's sum identity, carry recurrence, and running flag;
/// - the refused carry out and the nonzero gap;
/// - the equal cell number within a run, and the first row's run flag;
/// - read continuity;
/// - a continuing proof's opening read and its closing marker.
///
/// The rest are kept on purpose, and it is worth saying why.
///
/// The clock already forces its own digits, and the permutation carries that to the sorted order.
///
/// The sum identity already forces the gap digits, once the compared digits are bits.
///
/// The run flag is forced by the two comparisons never running together.
///
/// The comparison's zero carry in only tightens the witnessed gap by one.
///
/// The clock's unit carry in only fixes which way the clock runs.
///
/// Each stays because it is the plain local form of something this argument has to prove.
///
/// An argument resting on an induction threaded through a separate claim is fragile.
///
/// Each costs one small constraint per column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamAir {
    /// Public shape this AIR enforces.
    statement: RamStatement,
    /// Column offsets derived from that shape.
    layout: RamLayout,
}

impl RamAir {
    /// Builds the constraint set for one validated memory statement.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed statement.
    pub fn new(statement: RamStatement) -> Result<Self, RamError> {
        // The layout constructor validates, so no unchecked statement can reach a column index.
        let layout = RamLayout::new(&statement)?;
        Ok(Self { statement, layout })
    }

    /// Public shape this AIR enforces.
    #[must_use]
    pub const fn statement(&self) -> &RamStatement {
        // Callers build their witness against the same statement the constraints read.
        &self.statement
    }

    /// Column offsets of the committed trace.
    #[must_use]
    pub const fn layout(&self) -> &RamLayout {
        // Witness generation and the bus declarations share one offset table.
        &self.layout
    }

    /// Asserts that every digit column holds a bit.
    ///
    /// Carry and running flags are left out on purpose.
    ///
    /// Each is a product or an or of columns already known to hold bits.
    ///
    /// So each holds a bit wherever anything reads it, and is free where nothing does.
    fn eval_booleanity<AB: AirBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let layout = &self.layout;

        // Both orders carry the same tuple, so both are checked.
        for (write, address, timestamp) in [
            (
                layout.execution_write,
                layout.execution_address,
                layout.execution_timestamp,
            ),
            (
                layout.memory_write,
                layout.memory_address,
                layout.memory_timestamp,
            ),
        ] {
            builder.assert_bool(row[write]);
            for bit in 0..layout.address_bits {
                builder.assert_bool(row[address + bit]);
            }
            for bit in 0..layout.timestamp_bits {
                builder.assert_bool(row[timestamp + bit]);
            }
        }

        // The sorted trace's own witness columns are equally free unless constrained.
        builder.assert_bool(row[layout.same_address]);
        for bit in 0..layout.compare_bits {
            builder.assert_bool(row[layout.compare_delta + bit]);
        }
        if self.statement.boundary.is_segment() {
            builder.assert_bool(row[layout.group_end]);
        }
    }

    /// Asserts that a row's clock reading is its own position in the issuing order.
    ///
    /// Adding one is a carry into the lowest digit.
    ///
    /// Each digit flips for as long as that carry is still travelling.
    ///
    /// With the zero start, a row's reading is its position taken modulo the clock's range.
    ///
    /// The shape check refuses more accesses than the clock can count.
    ///
    /// So the clock never reaches its wrap, and the readings run up from zero without repeating.
    ///
    /// Distinct readings are what give the sorted order a single answer.
    ///
    /// Two accesses at one cell sharing a reading would have no order between them.
    ///
    /// A read could then be matched against the later write instead of the earlier one.
    fn eval_execution_clock<AB: AirBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let next_row = main.next_slice();
        let timestamp = self.layout.execution_timestamp;
        let carry = self.layout.execution_carry;
        let bits = self.layout.timestamp_bits;

        {
            // The execution order starts at time zero.
            let mut first = builder.when_first_row();
            for bit in 0..bits {
                first.assert_zero(row[timestamp + bit]);
            }
        }

        let mut transition = builder.when_transition();

        // Adding one is a carry of one into the least significant bit.
        transition.assert_one(row[carry]);
        for bit in 0..bits {
            let value: AB::Expr = row[timestamp + bit].into();
            let carry_in: AB::Expr = row[carry + bit].into();

            // A half adder: the bit flips exactly while the carry is still travelling.
            transition.assert_eq(
                next_row[timestamp + bit],
                xor::<AB::Expr, AB::F>(value.clone(), carry_in.clone()),
            );
            // The top bit's carry out would only matter to a clock allowed to wrap.
            if bit + 1 < bits {
                transition.assert_eq(row[carry + bit + 1], value * carry_in);
            }
        }
    }

    /// Asserts the sorted order, read continuity, and the opening boundary.
    ///
    /// Row zero opens the first run and has nothing above it, so it takes the opening rules.
    ///
    /// Every later row is compared against the row before it.
    fn eval_memory_order<AB: AirBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let next_row = main.next_slice();
        let layout = &self.layout;

        {
            // The first sorted row cannot continue a group, because there is none.
            let mut first = builder.when_first_row();
            first.assert_zero(row[layout.same_address]);
            self.eval_group_start(&mut first, row);
        }

        let mut transition = builder.when_transition();
        let same: AB::Expr = next_row[layout.same_address].into();

        // A continued group repeats its address exactly.
        for bit in 0..layout.address_bits {
            let previous: AB::Expr = row[layout.memory_address + bit].into();
            let current: AB::Expr = next_row[layout.memory_address + bit].into();
            transition.assert_zero(same.clone() * (current - previous));
        }

        // A new group increases the address; a continued group increases the timestamp.
        //
        // Exactly one of the two holds on any row, so both read one shared difference witness.
        let delta = &next_row[layout.compare_delta..layout.compare_delta + layout.compare_bits];
        let carry = &next_row[layout.compare_carry..layout.compare_carry + layout.compare_bits + 1];
        let nonzero =
            &next_row[layout.compare_nonzero..layout.compare_nonzero + layout.compare_bits];
        assert_strict_increase(
            &mut transition,
            &(AB::Expr::ONE - same.clone()),
            &row[layout.memory_address..layout.memory_address + layout.address_bits],
            &next_row[layout.memory_address..layout.memory_address + layout.address_bits],
            delta,
            carry,
            nonzero,
        );
        assert_strict_increase(
            &mut transition,
            &same,
            &row[layout.memory_timestamp..layout.memory_timestamp + layout.timestamp_bits],
            &next_row[layout.memory_timestamp..layout.memory_timestamp + layout.timestamp_bits],
            delta,
            carry,
            nonzero,
        );

        // A read inside a group returns the value the previous access at that address left.
        let read = AB::Expr::ONE - next_row[layout.memory_write].into();
        for component in 0..layout.value_width {
            let previous: AB::Expr = row[layout.memory_value + component].into();
            let current: AB::Expr = next_row[layout.memory_value + component].into();
            match self.statement.boundary {
                // A single proof starts every address at zero, so a group-opening read sees zero.
                //
                // Folding the group flag into the expected value covers both cases at once.
                RamBoundary::SingleProof => {
                    transition.assert_zero(read.clone() * (current - same.clone() * previous));
                }
                // A continuing proof takes its opening value from the inherited image.
                //
                // So continuity applies only within a cell's run.
                RamBoundary::Segment { .. } => {
                    transition
                        .assert_zero(read.clone() * same.clone() * (current - previous.clone()));
                }
            }
        }

        // A self-contained proof folds its opening rule into the continuity check above.
        //
        // Only a continuing proof needs a separate filter here.
        if self.statement.boundary.is_segment() {
            let mut opening = transition.when_ne(same, AB::Expr::ONE);
            self.eval_group_start(&mut opening, next_row);
        }
    }

    /// Asserts what the first row of a cell's run owes.
    ///
    /// A self-contained proof starts memory empty, so an opening read returns zero.
    ///
    /// An opening write is free, because it only overwrites that zero.
    ///
    /// A continuing proof takes its opening value from a committed image instead.
    ///
    /// So its opening access has to be a read, and the image declaration fixes what it returns.
    ///
    /// That is what stops a continuing proof inventing the memory it starts from.
    fn eval_group_start<AB: AirBuilder>(&self, builder: &mut AB, row: &[AB::Var]) {
        let layout = &self.layout;
        match self.statement.boundary {
            RamBoundary::SingleProof => {
                let read = AB::Expr::ONE - row[layout.memory_write].into();
                for component in 0..layout.value_width {
                    builder.assert_zero(read.clone() * row[layout.memory_value + component]);
                }
            }
            RamBoundary::Segment { .. } => {
                builder.assert_zero(row[layout.memory_write]);
            }
        }
    }

    /// Asserts that the closing marker names the last sorted row of each cell's run.
    ///
    /// Only a continuing proof has this column, because only it hands on an image.
    fn eval_group_marker<AB: AirBuilder>(&self, builder: &mut AB) {
        if !self.statement.boundary.is_segment() {
            return;
        }
        let main = builder.main();
        let row = main.current_slice();
        let next_row = main.next_slice();
        let group_end = self.layout.group_end;
        let same_address = self.layout.same_address;

        // A row closes its run exactly when the row after it opens a new one.
        builder.when_transition().assert_eq(
            row[group_end],
            AB::Expr::ONE - next_row[same_address].into(),
        );

        // The last sorted row closes whichever run it belongs to.
        builder.when_last_row().assert_one(row[group_end]);
    }

    /// Declares this memory's tuples on the named channels the enclosing plan balances.
    ///
    /// Three claims leave here, and none of them is checked here:
    ///
    /// - the issuing order holds exactly the accesses the machine issued;
    /// - the two orders hold the same accesses;
    /// - a continuing proof's opening and closing values match the committed images.
    ///
    /// Each is an ordinary multiset claim on the reduction every other table already uses.
    ///
    /// There is no second memory protocol here.
    fn declare<AB: BusInteractionBuilder>(&self, builder: &mut AB) {
        let layout = &self.layout;

        // The machine's chips produce each access and this memory consumes it.
        let execution = self.tuple::<AB>(builder, layout.execution_access_columns());
        builder.push_bus_interaction(
            &self.statement.access_bus,
            BusDirection::Pull,
            execution.iter().cloned(),
            BusActivation::Always,
        );

        // That same access is the producing side of the permutation.
        builder.push_bus_interaction(
            &self.statement.order_bus,
            BusDirection::Push,
            execution,
            BusActivation::Always,
        );

        // The sorted order consumes it back, so balance proves the two orders agree.
        let memory = self.tuple::<AB>(builder, layout.memory_access_columns());
        builder.push_bus_interaction(
            &self.statement.order_bus,
            BusDirection::Pull,
            memory,
            BusActivation::Always,
        );

        let RamBoundary::Segment { incoming, outgoing } = &self.statement.boundary else {
            return;
        };

        // The value a cell's run opens with comes from the inherited image.
        let image = self.tuple::<AB>(builder, layout.image_columns());
        let opening = AB::Expr::ONE - builder.main().current_slice()[layout.same_address].into();
        builder.push_bus_interaction(
            incoming,
            BusDirection::Pull,
            image.iter().cloned(),
            BusActivation::Boolean(opening),
        );

        // The value a cell's run closes with becomes an entry of the handed-on image.
        let closing: AB::Expr = builder.main().current_slice()[layout.group_end].into();
        builder.push_bus_interaction(
            outgoing,
            BusDirection::Push,
            image,
            BusActivation::Boolean(closing),
        );
    }

    /// Reads one tuple's columns out of the current row.
    fn tuple<AB: AirBuilder>(
        &self,
        builder: &AB,
        columns: impl Iterator<Item = usize>,
    ) -> Vec<AB::Expr> {
        // Collecting releases the trace window before the declaration borrows the builder.
        let main = builder.main();
        let row = main.current_slice();
        columns.map(|column| row[column].into()).collect()
    }
}

impl<F> BaseAir<F> for RamAir {
    fn width(&self) -> usize {
        // One committed trace holds both access orders and every witness column.
        self.layout.width
    }
}

impl<AB: BusInteractionBuilder> Air<AB> for RamAir {
    fn eval(&self, builder: &mut AB) {
        self.eval_booleanity(builder);
        self.eval_execution_clock(builder);
        self.eval_memory_order(builder);
        self.eval_group_marker(builder);
        self.declare(builder);
    }
}

/// Asserts that the later key exceeds the earlier one, whenever this comparison is running.
///
/// The witness is the gap between them, added back to the earlier key by an explicit adder.
///
/// The carry into the lowest digit is zero and the carry out of the highest must be too.
///
/// Refusing that carry out is what makes the comparison unsigned and non-wrapping.
///
/// The later key is then really the larger one, not the smaller one seen through a wrap.
///
/// The running flag forces the gap to be nonzero, which makes the comparison strict.
///
/// Without it two rows could hold one cell number while claiming to open a fresh run.
///
/// The witness may be wider than the keys being compared.
///
/// The two comparisons never run together, so they share one witness sized for the wider.
fn assert_strict_increase<AB: AirBuilder>(
    builder: &mut AB,
    enabled: &AB::Expr,
    left: &[AB::Var],
    right: &[AB::Var],
    delta: &[AB::Var],
    carry: &[AB::Var],
    nonzero: &[AB::Var],
) {
    let bits = left.len();
    debug_assert_eq!(right.len(), bits);
    debug_assert!(delta.len() >= bits);
    debug_assert!(carry.len() > bits);
    debug_assert!(nonzero.len() >= bits);

    // An addition starts with nothing carried in.
    builder.assert_zero(enabled.clone() * carry[0]);

    for bit in 0..bits {
        let addend: AB::Expr = left[bit].into();
        let difference: AB::Expr = delta[bit].into();
        let carry_in: AB::Expr = carry[bit].into();

        // The sum digit and the carry out are the usual adder identities.
        let sum = xor3::<AB::Expr, AB::F>(addend.clone(), difference.clone(), carry_in.clone());
        builder.assert_zero(enabled.clone() * (right[bit].into() - sum));
        builder.assert_zero(
            enabled.clone()
                * (carry[bit + 1].into()
                    - majority::<AB::Expr, AB::F>(addend, difference.clone(), carry_in)),
        );

        // The running flag reaches one exactly when some gap digit is set.
        let running = if bit == 0 {
            difference
        } else {
            or::<AB::Expr, AB::F>(nonzero[bit - 1].into(), difference)
        };
        builder.assert_zero(enabled.clone() * (nonzero[bit].into() - running));
    }

    // A carry out of the highest digit means the addition wrapped, so the order is not unsigned.
    builder.assert_zero(enabled.clone() * carry[bits]);

    // A zero gap would leave the two keys equal rather than ordered.
    builder.assert_zero(enabled.clone() * (AB::Expr::ONE - nonzero[bits - 1].into()));
}

/// Exclusive or of two bit-valued expressions.
///
/// In characteristic two the correction vanishes and this is a plain sum.
fn xor<E: Algebra<F>, F: PrimeCharacteristicRing>(left: E, right: E) -> E {
    left.clone() + right.clone() - E::TWO * left * right
}

/// Exclusive or of three bit-valued expressions.
fn xor3<E: Algebra<F>, F: PrimeCharacteristicRing>(first: E, second: E, third: E) -> E {
    xor::<E, F>(xor::<E, F>(first, second), third)
}

/// Majority of three bit-valued expressions, which is an adder's carry out.
fn majority<E: Algebra<F>, F: PrimeCharacteristicRing>(first: E, second: E, third: E) -> E {
    // Three pairwise products count an all-ones input three times, so two copies come back off.
    first.clone() * second.clone() + first.clone() * third.clone() + second.clone() * third.clone()
        - E::TWO * first * second * third
}

/// Inclusive or of two bit-valued expressions.
fn or<E: Algebra<F>, F: PrimeCharacteristicRing>(left: E, right: E) -> E {
    left.clone() + right.clone() - left * right
}
