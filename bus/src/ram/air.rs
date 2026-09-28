//! Verifier-enforced constraints for one mutable read-write memory.

use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::{Algebra, PrimeCharacteristicRing};

use super::statement::channel;
use super::{RamBoundary, RamError, RamLayout, RamStatement};
use crate::{BusActivation, BusDirection, BusInteractionBuilder};

/// Constraints and channel declarations of one mutable read-write memory.
///
/// The committed trace holds the accesses sorted by cell and then by clock reading.
///
/// The machine's chips produce each access on a channel this trace consumes.
///
/// Balancing it is the permutation between the two orders.
///
/// Two constraints below survive deletion alone; the pull request says which and why.
///
/// A test pins the constraint count, so changing the set forces a fresh sweep.
///
/// That test fails on every deletion, so a sweep has to skip it to learn anything.
///
/// # What the machine owes
///
/// Three obligations sit outside these constraints, for a machine to meet itself.
///
/// A reading is the machine's, so its chips must constrain readings to rise along execution.
///
/// Skip that and the proof still verifies, but of the wrong statement.
///
/// A read then sees the last write by reading, not the last write the machine performed.
///
/// Two accesses to one cell need different readings, and that one the constraints do enforce.
///
/// The access count is the trace height, so it is a power of two and every row pulls a tuple.
///
/// A machine with fewer real accesses pads with real ones its own chips also produce.
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
        &self.statement
    }

    /// Column offsets of the committed trace.
    #[must_use]
    pub const fn layout(&self) -> &RamLayout {
        &self.layout
    }

    /// Asserts that every digit column holds a bit.
    ///
    /// Four columns are left out, each already pinned to the bits by something else:
    ///
    /// - a carry and a running flag, being products and ors of columns that hold bits;
    /// - the closing marker, whose conditional declaration checks its own gate;
    /// - the run flag, which outside the bits would make both comparisons run at once;
    /// - the operation marker, which only ever appears as one minus itself.
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
    /// Row zero opens the first run and has nothing above it, so it only takes the opening rule.
    fn eval_order<AB: AirBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let next_row = main.next_slice();
        let layout = &self.layout;

        {
            let mut first = builder.when_first_row();
            first.assert_zero(row[layout.same_address]);
            self.eval_run_start(&mut first, row);
        }

        let mut transition = builder.when_transition();
        let same: AB::Expr = next_row[layout.same_address].into();

        for bit in 0..layout.address_bits {
            let previous: AB::Expr = row[layout.address + bit].into();
            let current: AB::Expr = next_row[layout.address + bit].into();
            transition.assert_zero(same.clone() * (current - previous));
        }

        // Exactly one holds on a row, so both read one shared gap witness.
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

        // A read inside a run returns what the access before it left.
        let read = AB::Expr::ONE - next_row[layout.operation].into();
        for component in 0..layout.value_width {
            let previous: AB::Expr = row[layout.value + component].into();
            let current: AB::Expr = next_row[layout.value + component].into();
            match self.statement.boundary {
                // A self-contained proof starts every cell at zero, so an opening read sees zero.
                RamBoundary::SingleProof => {
                    transition.assert_zero(read.clone() * (current - same.clone() * previous));
                }
                // A continuing proof takes its opening value from the inherited image.
                RamBoundary::Segment { .. } => {
                    transition
                        .assert_zero(read.clone() * same.clone() * (current - previous.clone()));
                }
            }
        }

        if self.statement.boundary.is_segment() {
            let mut opening = transition.when_ne(same, AB::Expr::ONE);
            self.eval_run_start(&mut opening, next_row);
        }
    }

    /// Asserts what the first row of a cell's run owes.
    ///
    /// A self-contained proof starts memory empty, so an opening read returns zero.
    ///
    /// A continuing proof declares that row's value against the inherited image.
    ///
    /// Balance is what forces it to be the value the cell was handed.
    ///
    /// Requiring a read there keeps the declared value the one the row saw, not one it stored.
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
    /// Only a continuing proof has this column, since only it hands on an image.
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

        builder.when_last_row().assert_one(row[group_end]);
    }

    /// Declares this memory's tuples on the named channels the enclosing plan balances.
    ///
    /// Two claims leave here unchecked, both ordinary multiset claims on the plan's reduction:
    ///
    /// - this trace holds exactly the accesses the machine issued;
    /// - a continuing proof's opening and closing values match the committed images.
    fn declare<AB: BusInteractionBuilder>(&self, builder: &mut AB) {
        let layout = &self.layout;

        let access = self.tuple::<AB>(builder, layout.access_columns());
        builder.push_bus_interaction(
            channel(&self.statement.access_bus),
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
            channel(incoming),
            BusDirection::Pull,
            image.iter().cloned(),
            BusActivation::Boolean(opening),
        );

        let closing: AB::Expr = builder.main().current_slice()[layout.group_end].into();
        builder.push_bus_interaction(
            channel(outgoing),
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
        let main = builder.main();
        let row = main.current_slice();
        columns.map(|column| row[column].into()).collect()
    }
}

impl<F> BaseAir<F> for RamAir {
    fn width(&self) -> usize {
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
/// Two of its rules carry the meaning:
///
/// - the carry out of the highest digit is refused, which makes the comparison unsigned;
/// - the gap is forced nonzero, which makes it strict rather than merely non-decreasing.
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

/// Exclusive or of two bit-valued expressions, a plain sum in characteristic two.
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
