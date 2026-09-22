//! Verifier-enforced constraints for one mutable read-write memory.

use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Algebra, PrimeCharacteristicRing};

use super::{RamBoundary, RamError, RamLayout, RamStatement};
use crate::{BusActivation, BusDirection, BusInteractionBuilder};

/// Constraints and bus declarations of one mutable read-write memory.
///
/// One committed matrix holds both access orders. Every constraint below is enforced by the
/// enclosing proof system's constraint check, and every cross-trace claim is a named-bus
/// declaration the enclosing bus plan balances. Nothing is left as a handoff for a caller to
/// remember: an integration that commits this trace and runs the plan has the whole argument.
///
/// # What each constraint buys
///
/// Address and timestamp **bits** are what make the ripple-adder comparisons mean anything. A
/// non-Boolean digit makes "the difference is nonzero and did not wrap" an empty statement, and an
/// adversary can then order rows however it likes.
///
/// A **unique, non-wrapping execution order** is what makes the sorted order total. The clock runs
/// `0, 1, ..., n - 1` with a refused carry-out, so no two accesses share a timestamp. If two did,
/// two accesses at one address would have no defined order and a read could be matched against the
/// later write instead of the earlier one.
///
/// **Unsigned memory order** is what makes address groups contiguous. Without the refused
/// carry-out an address could wrap, splitting one address into two groups; the second group
/// re-initialises, so a read returns the initial value instead of what was written.
///
/// Without the nonzero difference, two rows could hold the *same* address while claiming a new
/// group, which is the same attack with no wrap needed.
///
/// **Read continuity** is what makes a read see the last write. **Write updates** are the absence
/// of a value constraint on a write, so the next read at that address inherits the written value.
///
/// **Initialization** and **final boundaries** are what tie this proof's memory to the world
/// around it, and they differ by [`RamBoundary`].
///
/// # Which constraints are load-bearing
///
/// Mutation testing puts a name on the difference between a constraint that carries the argument
/// and one that restates something another constraint already forces. Removing any of these makes
/// a test in this module fail, so each is doing work on its own:
///
/// - memory-side address Booleanity;
/// - the clock's zero start, its sum identity, and its carry recurrence;
/// - the comparison's sum identity, carry recurrence, and running-or recurrence;
/// - the refused carry out and the nonzero difference;
/// - the same-group address equality and the first row's group flag;
/// - read continuity;
/// - a segment's group-opening read and its group-end marker.
///
/// The rest are kept deliberately, and it is worth saying why rather than leaving a reader to
/// wonder. Timestamp Booleanity is already forced by the clock and carried to the sorted order by
/// the permutation. Difference Booleanity is already forced by the sum identity once the compared
/// digits are Boolean. The group flag's Booleanity is forced by the two comparisons being mutually
/// exclusive. The comparison's zero carry in only tightens the witnessed difference by one, and
/// the clock's unit carry in only fixes which direction the clock runs.
///
/// They stay because each is the local, self-evident form of something the acceptance criteria
/// name outright, and because an argument that rests on an induction threaded through a separate
/// multiset claim is one refactor away from being wrong. They cost one degree-two constraint per
/// column.
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
    /// Carry and running-or columns are left out on purpose: their recurrences are products and
    /// ors of columns already known to be Boolean, so each is Boolean wherever it is constrained
    /// at all, and free where nothing reads it.
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

    /// Asserts that the execution timestamp of row `i` is the integer `i`.
    ///
    /// The clock is a half-adder chain adding one: the carry into the low bit is one, and each bit
    /// flips while the carry is still travelling.
    ///
    /// Together with the zero start this pins row `i` to `i mod 2^timestamp_bits`, and
    /// [`RamStatement::validate`] refuses an access count larger than `2^timestamp_bits`, so the
    /// clock never reaches the wrap. The `access_count` timestamps are therefore exactly
    /// `0, ..., access_count - 1`: unique, in order, and non-wrapping.
    ///
    /// Uniqueness is what gives the address-sorted order a single answer. Two accesses at one
    /// address sharing a timestamp would have no order between them, and a read could then be
    /// matched against the later write instead of the earlier one.
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

    /// Asserts unsigned address-then-timestamp order, read continuity, and the opening boundary.
    ///
    /// Row zero opens the first address group and has no predecessor, so it gets the group-start
    /// rules directly. Every later row is compared against the row before it.
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
                // A segment inherits its opening value from the incoming image instead, so
                // continuity applies only inside a group.
                RamBoundary::Segment { .. } => {
                    transition
                        .assert_zero(read.clone() * same.clone() * (current - previous.clone()));
                }
            }
        }

        // A single proof's group-start rule is already folded into the continuity constraint
        // above, so only a segment needs a separate opening filter here.
        if self.statement.boundary.is_segment() {
            let mut opening = transition.when_ne(same, AB::Expr::ONE);
            self.eval_group_start(&mut opening, next_row);
        }
    }

    /// Asserts what a row that opens an address group owes.
    ///
    /// A single proof initialises memory to zero, so an opening read must return zero. A write may
    /// open a group freely: it overwrites the initial value.
    ///
    /// A segment inherits its opening value from a committed image instead, so an opening access
    /// is forced to be a read. The value that read returns is fixed by the incoming-image
    /// declaration, which is what stops a segment inventing the memory it starts from.
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

    /// Asserts that the group-end marker names the last sorted row of each address group.
    ///
    /// Only a segment allocates this column, because only a segment exports a final image.
    fn eval_group_marker<AB: AirBuilder>(&self, builder: &mut AB) {
        if !self.statement.boundary.is_segment() {
            return;
        }
        let main = builder.main();
        let row = main.current_slice();
        let next_row = main.next_slice();
        let group_end = self.layout.group_end;
        let same_address = self.layout.same_address;

        // A row ends its group exactly when the row after it opens a new one.
        builder.when_transition().assert_eq(
            row[group_end],
            AB::Expr::ONE - next_row[same_address].into(),
        );

        // The last sorted row ends whichever group it belongs to.
        builder.when_last_row().assert_one(row[group_end]);
    }

    /// Declares this memory's tuples on the named channels the enclosing plan balances.
    ///
    /// Three claims leave this AIR, and none of them is checked here:
    ///
    /// - the execution order holds exactly the accesses the machine issued, on `access_bus`;
    /// - the two orders hold the same accesses, on `order_bus`;
    /// - in segment mode, the opening and closing values match the committed images.
    ///
    /// Each is an ordinary multiset claim, discharged by the same product reduction every other
    /// table in the plan uses. There is no second memory protocol here.
    fn declare<AB: BusInteractionBuilder>(&self, builder: &mut AB) {
        let layout = &self.layout;

        // The machine's chips push each access; this memory pulls it.
        let execution = self.tuple::<AB>(builder, layout.execution_access_columns());
        builder.push_bus_interaction(
            &self.statement.access_bus,
            BusDirection::Pull,
            execution.iter().cloned(),
            BusActivation::Always,
        );

        // The same execution tuple is the push side of the permutation.
        builder.push_bus_interaction(
            &self.statement.order_bus,
            BusDirection::Push,
            execution,
            BusActivation::Always,
        );

        // The sorted order pulls it back, so balance proves the two orders agree as multisets.
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

        // The value a group opens with comes from the incoming image.
        let image = self.tuple::<AB>(builder, layout.image_columns());
        let opening = AB::Expr::ONE - builder.main().current_slice()[layout.same_address].into();
        builder.push_bus_interaction(
            incoming,
            BusDirection::Pull,
            image.iter().cloned(),
            BusActivation::Boolean(opening),
        );

        // The value a group closes with becomes an entry of the outgoing image.
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
        // Collecting releases the trace window before the declaration borrows the builder again.
        let main = builder.main();
        let row = main.current_slice();
        columns.map(|column| row[column].into()).collect()
    }
}

impl<F> BaseAir<F> for RamAir {
    fn width(&self) -> usize {
        // One committed matrix holds both access orders and every witness column.
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

/// Asserts `right > left` as unsigned little-endian integers, whenever `enabled` is one.
///
/// The witness is the difference `right - left`, added back with an explicit ripple adder:
/// `right = left + delta`, with the carry into the low bit zero and the carry out of the top bit
/// zero. A refused carry out is what makes the comparison unsigned and non-wrapping, so `right`
/// really is the larger integer rather than the smaller one seen through a modular reduction.
///
/// The running-or chain forces `delta` to be nonzero, which turns the comparison from `>=` into
/// `>`. Without it two rows could hold the same address while claiming to open a new group.
///
/// `delta`, `carry`, and `nonzero` may be wider than the compared values: the two comparisons on a
/// sorted row are mutually exclusive, so they share one witness sized for the wider of the two.
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

    // An addition starts with no carry in.
    builder.assert_zero(enabled.clone() * carry[0]);

    for bit in 0..bits {
        let addend: AB::Expr = left[bit].into();
        let difference: AB::Expr = delta[bit].into();
        let carry_in: AB::Expr = carry[bit].into();

        // The sum bit and the carry out are the full-adder identities.
        let sum = xor3::<AB::Expr, AB::F>(addend.clone(), difference.clone(), carry_in.clone());
        builder.assert_zero(enabled.clone() * (right[bit].into() - sum));
        builder.assert_zero(
            enabled.clone()
                * (carry[bit + 1].into()
                    - majority::<AB::Expr, AB::F>(addend, difference.clone(), carry_in)),
        );

        // The running or reaches one exactly when some difference bit is set.
        let running = if bit == 0 {
            difference
        } else {
            or::<AB::Expr, AB::F>(nonzero[bit - 1].into(), difference)
        };
        builder.assert_zero(enabled.clone() * (nonzero[bit].into() - running));
    }

    // A carry out of the top bit means the addition wrapped, so the order is not unsigned.
    builder.assert_zero(enabled.clone() * carry[bits]);

    // A zero difference would only prove `right >= left`.
    builder.assert_zero(enabled.clone() * (AB::Expr::ONE - nonzero[bits - 1].into()));
}

/// Exclusive or of two Boolean-valued expressions.
///
/// In characteristic two the correction term vanishes and this is a plain sum.
fn xor<E: Algebra<F>, F: PrimeCharacteristicRing>(left: E, right: E) -> E {
    left.clone() + right.clone() - E::TWO * left * right
}

/// Exclusive or of three Boolean-valued expressions.
fn xor3<E: Algebra<F>, F: PrimeCharacteristicRing>(first: E, second: E, third: E) -> E {
    xor::<E, F>(xor::<E, F>(first, second), third)
}

/// Majority of three Boolean-valued expressions, which is a full adder's carry out.
fn majority<E: Algebra<F>, F: PrimeCharacteristicRing>(first: E, second: E, third: E) -> E {
    // Three pairwise products count each all-ones input three times, so two copies come back off.
    first.clone() * second.clone() + first.clone() * third.clone() + second.clone() * third.clone()
        - E::TWO * first * second * third
}

/// Inclusive or of two Boolean-valued expressions.
fn or<E: Algebra<F>, F: PrimeCharacteristicRing>(left: E, right: E) -> E {
    left.clone() + right.clone() - left * right
}
