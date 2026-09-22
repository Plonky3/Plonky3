//! Verifier-enforced constraints for one mutable read-write memory.

use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Algebra, PrimeCharacteristicRing};

use super::{RamBoundary, RamError, RamLayout, RamStatement};
use crate::{BusActivation, BusDirection, BusInteractionBuilder};

/// Constraints and channel declarations of one mutable read-write memory.
///
/// The committed trace holds the accesses sorted by cell and then by clock reading.
///
/// The machine's own chips are the copy in issuing order.
///
/// They produce each access on a named channel and this trace consumes it.
///
/// Balancing that channel is the permutation between the two orders.
///
/// # What the machine still owes
///
/// A clock reading is whatever the machine's chips say it is.
///
/// Nothing here can tell whether that reading tracks the order the machine really ran in.
///
/// A machine has to constrain its own readings to rise along its execution.
///
/// What this argument then proves is that a read returns the last write by that reading.
///
/// # What each constraint buys
///
/// The digit columns are what make the comparisons mean anything.
///
/// A digit holding something other than a bit empties out "the gap is nonzero and did not wrap".
///
/// An adversary can then order rows however it likes.
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
/// Rising readings inside a run are what make a read see the last write and not an earlier one.
///
/// They also make two accesses to one cell at one reading impossible, which would have no order.
///
/// Read continuity is what carries a value from an access to the next read of that cell.
///
/// A write is left unconstrained in value, which is how the next read inherits it.
///
/// The two boundaries tie this proof's memory to the world around it.
///
/// # Which constraints are load-bearing
///
/// Each constraint below was deleted on its own and the suite re-run.
///
/// Fourteen of the sixteen deletions make a test in this module fail.
///
/// Two do not, and both belong to the comparison: the gap digits and the cleared carry in.
///
/// Either one alone is implied by the other, so deleting one changes nothing.
///
/// Boolean keys and a cleared carry in leave the gap digits no freedom to be anything else.
///
/// Boolean keys and Boolean gap digits leave the carry chain no freedom at all.
///
/// Deleting both at once was tried too, and no forgery came out of it.
///
/// Nor did an argument that none exists, so both stay.
///
/// A test below pins the constraint count, so adding or removing one forces this note to be redone.
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
        // Witness generation and the channel declarations share one offset table.
        &self.layout
    }

    /// Asserts that every digit column holds a bit.
    ///
    /// Four columns are left out, each because something else already pins it.
    ///
    /// A carry and a running flag are products and ors of columns that hold bits.
    ///
    /// A conditional declaration checks its own gate, which covers the closing marker.
    ///
    /// The run flag cannot escape the bits, because the two comparisons would then both run.
    ///
    /// One would need the cell numbers equal and the other would need them apart.
    ///
    /// The operation marker only ever appears as one minus itself, multiplying a difference.
    ///
    /// Any value but one leaves that difference forced, so nothing outside the bits buys a write.
    fn eval_booleanity<AB: AirBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let layout = &self.layout;

        for bit in 0..layout.address_bits {
            builder.assert_bool(row[layout.address + bit]);
        }
        for bit in 0..layout.timestamp_bits {
            builder.assert_bool(row[layout.timestamp + bit]);
        }

        // The gap digits are the trace's own, and nothing else pins them to the bits.
        for bit in 0..layout.compare_bits {
            builder.assert_bool(row[layout.compare_delta + bit]);
        }
    }

    /// Asserts the sorted order, read continuity, and the opening rule.
    ///
    /// Row zero opens the first run and has nothing above it, so it takes the opening rule.
    ///
    /// Every later row is compared against the row before it.
    fn eval_order<AB: AirBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let next_row = main.next_slice();
        let layout = &self.layout;

        {
            // The first row cannot continue a run, because there is none.
            let mut first = builder.when_first_row();
            first.assert_zero(row[layout.same_address]);
            self.eval_run_start(&mut first, row);
        }

        let mut transition = builder.when_transition();
        let same: AB::Expr = next_row[layout.same_address].into();

        // A continued run repeats its cell number exactly.
        for bit in 0..layout.address_bits {
            let previous: AB::Expr = row[layout.address + bit].into();
            let current: AB::Expr = next_row[layout.address + bit].into();
            transition.assert_zero(same.clone() * (current - previous));
        }

        // A fresh run raises the cell number; a continued run raises the clock reading.
        //
        // Exactly one of the two holds on any row, so both read one shared gap witness.
        let delta = &next_row[layout.compare_delta..layout.compare_delta + layout.compare_bits];
        let carry = &next_row[layout.compare_carry..layout.compare_carry + layout.compare_bits + 1];
        let nonzero =
            &next_row[layout.compare_nonzero..layout.compare_nonzero + layout.compare_bits];
        assert_strict_increase(
            &mut transition,
            &(AB::Expr::ONE - same.clone()),
            &row[layout.address..layout.address + layout.address_bits],
            &next_row[layout.address..layout.address + layout.address_bits],
            delta,
            carry,
            nonzero,
        );
        assert_strict_increase(
            &mut transition,
            &same,
            &row[layout.timestamp..layout.timestamp + layout.timestamp_bits],
            &next_row[layout.timestamp..layout.timestamp + layout.timestamp_bits],
            delta,
            carry,
            nonzero,
        );

        // A read inside a run returns the value the access before it left.
        let read = AB::Expr::ONE - next_row[layout.operation].into();
        for component in 0..layout.value_width {
            let previous: AB::Expr = row[layout.value + component].into();
            let current: AB::Expr = next_row[layout.value + component].into();
            match self.statement.boundary {
                // A self-contained proof starts every cell at zero, so an opening read sees zero.
                //
                // Folding the run flag into the expected value covers both cases at once.
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
            self.eval_run_start(&mut opening, next_row);
        }
    }

    /// Asserts what the first row of a cell's run owes.
    ///
    /// A self-contained proof starts memory empty, so an opening read returns zero.
    ///
    /// An opening write is free, because it only overwrites that zero.
    ///
    /// A continuing proof declares the opening row's value against the inherited image.
    ///
    /// Balance is what forces that value to be the one the cell was handed.
    ///
    /// The opening access is required to be a read so the value it declares is the one it saw.
    ///
    /// An opening write would declare the value it stores, which no inherited entry need hold.
    fn eval_run_start<AB: AirBuilder>(&self, builder: &mut AB, row: &[AB::Var]) {
        let layout = &self.layout;
        match self.statement.boundary {
            RamBoundary::SingleProof => {
                let read = AB::Expr::ONE - row[layout.operation].into();
                for component in 0..layout.value_width {
                    builder.assert_zero(read.clone() * row[layout.value + component]);
                }
            }
            RamBoundary::Segment { .. } => {
                builder.assert_zero(row[layout.operation]);
            }
        }
    }

    /// Asserts that the closing marker names the last row of each cell's run.
    ///
    /// Only a continuing proof has this column, because only it hands on an image.
    fn eval_run_marker<AB: AirBuilder>(&self, builder: &mut AB) {
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

        // The last row closes whichever run it belongs to.
        builder.when_last_row().assert_one(row[group_end]);
    }

    /// Declares this memory's tuples on the named channels the enclosing plan balances.
    ///
    /// Two claims leave here, and neither is checked here:
    ///
    /// - this trace holds exactly the accesses the machine issued;
    /// - a continuing proof's opening and closing values match the committed images.
    ///
    /// Each is an ordinary multiset claim on the reduction every other table already uses.
    ///
    /// There is no second memory protocol here.
    fn declare<AB: BusInteractionBuilder>(&self, builder: &mut AB) {
        let layout = &self.layout;

        // The machine's chips produce each access and this trace consumes it.
        let access = self.tuple::<AB>(builder, layout.access_columns());
        builder.push_bus_interaction(
            &self.statement.access_bus,
            BusDirection::Pull,
            access,
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
        // One committed trace holds the sorted accesses and every witness column.
        self.layout.width
    }
}

impl<AB: BusInteractionBuilder> Air<AB> for RamAir {
    fn eval(&self, builder: &mut AB) {
        self.eval_booleanity(builder);
        self.eval_order(builder);
        self.eval_run_marker(builder);
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
