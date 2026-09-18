use p3_word::{
    AndConstraint, ConstraintKind, ConstraintSystem, IntegerMulConstraint, Operand, OperandRole,
    Segment, Shift, ShiftError, ShiftKind, ShiftSequenceError, ShiftedValue, SystemError,
    ValueIndex, VerificationError, Word32, Word64, ZeroConstraint,
};
use proptest::prelude::*;

const FULL_KINDS: [ShiftKind; 4] = [
    ShiftKind::LogicalLeft,
    ShiftKind::LogicalRight,
    ShiftKind::ArithmeticRight,
    ShiftKind::RotateRight,
];

const ALL_KINDS: [ShiftKind; 8] = [
    ShiftKind::LogicalLeft,
    ShiftKind::LogicalRight,
    ShiftKind::ArithmeticRight,
    ShiftKind::RotateRight,
    ShiftKind::Lane32LogicalLeft,
    ShiftKind::Lane32LogicalRight,
    ShiftKind::Lane32ArithmeticRight,
    ShiftKind::Lane32RotateRight,
];

fn public(position: usize) -> ValueIndex {
    ValueIndex::public(position).expect("test position must fit")
}

fn witness(position: usize) -> ValueIndex {
    ValueIndex::witness(position).expect("test position must fit")
}

fn word32_operand(position: usize) -> Operand<Word32> {
    Operand::single(ShiftedValue::plain(witness(position)))
}

fn word64_operand(position: usize) -> Operand<Word64> {
    Operand::single(ShiftedValue::plain(witness(position)))
}

fn reference_shift32(kind: ShiftKind, word: u32, amount: u32) -> u32 {
    // The reference spells each operation directly in primitive integer arithmetic.
    match kind {
        ShiftKind::LogicalLeft => word << amount,
        ShiftKind::LogicalRight => word >> amount,
        ShiftKind::ArithmeticRight => ((word as i32) >> amount) as u32,
        ShiftKind::RotateRight => word.rotate_right(amount),
        _ => unreachable!("32-bit lane operations require a 64-bit container"),
    }
}

fn reference_shift64(kind: ShiftKind, word: u64, amount: u32) -> u64 {
    // Each lane helper prevents bits from crossing the 32-bit boundary.
    let lanes = |operation: fn(u32, u32) -> u32| {
        let low = operation(word as u32, amount);
        let high = operation((word >> 32) as u32, amount);
        u64::from(low) | (u64::from(high) << 32)
    };

    // The reference spells each operation directly in primitive integer arithmetic.
    match kind {
        ShiftKind::LogicalLeft => word << amount,
        ShiftKind::LogicalRight => word >> amount,
        ShiftKind::ArithmeticRight => ((word as i64) >> amount) as u64,
        ShiftKind::RotateRight => word.rotate_right(amount),
        ShiftKind::Lane32LogicalLeft => lanes(|lane, distance| lane << distance),
        ShiftKind::Lane32LogicalRight => lanes(|lane, distance| lane >> distance),
        ShiftKind::Lane32ArithmeticRight => {
            lanes(|lane, distance| ((lane as i32) >> distance) as u32)
        }
        ShiftKind::Lane32RotateRight => lanes(u32::rotate_right),
    }
}

fn shifts32() -> Vec<Shift<Word32>> {
    // Include the canonical identity once and every nonzero full-word shift.
    let mut shifts = vec![Shift::identity()];
    for kind in FULL_KINDS {
        for amount in 1..32 {
            shifts.push(Shift::new(kind, amount).expect("enumerated shift is valid"));
        }
    }
    shifts
}

fn shifts64() -> Vec<Shift<Word64>> {
    // Include the canonical identity once and every valid nonzero shift.
    let mut shifts = vec![Shift::identity()];
    for kind in ALL_KINDS {
        let width = if kind.is_lane32() { 32 } else { 64 };
        for amount in 1..width {
            shifts.push(Shift::new(kind, amount).expect("enumerated shift is valid"));
        }
    }
    shifts
}

fn signature32(inner: Shift<Word32>, outer: Shift<Word32>) -> [u32; 32] {
    // Shifts are F_2-linear maps and are determined by the images of basis bits.
    core::array::from_fn(|bit| outer.apply(inner.apply(Word32::new(1 << bit))).get())
}

fn signature64(inner: Shift<Word64>, outer: Shift<Word64>) -> [u64; 64] {
    // Shifts are F_2-linear maps and are determined by the images of basis bits.
    core::array::from_fn(|bit| outer.apply(inner.apply(Word64::new(1 << bit))).get())
}

proptest! {
    #[test]
    fn word32_shifts_match_primitive_arithmetic(
        word in any::<u32>(),
        kind_index in 0usize..FULL_KINDS.len(),
        amount in 0usize..32,
    ) {
        // Fixture state: every valid full-word operation and distance is constructible.
        let kind = FULL_KINDS[kind_index];
        let shift = Shift::<Word32>::new(kind, amount).expect("valid shift");

        // The checked relation operation must equal an independent integer expression.
        prop_assert_eq!(
            shift.apply(Word32::new(word)).get(),
            reference_shift32(kind, word, amount as u32),
        );
    }

    #[test]
    fn word64_shifts_match_primitive_arithmetic(
        word in any::<u64>(),
        kind_index in 0usize..ALL_KINDS.len(),
        raw_amount in 0usize..64,
    ) {
        // Lane operations use 32 positions while full-word operations use 64.
        let kind = ALL_KINDS[kind_index];
        let width = if kind.is_lane32() { 32 } else { 64 };
        let amount = raw_amount % width;
        let shift = Shift::<Word64>::new(kind, amount).expect("valid shift");

        // The checked relation operation must equal an independent integer expression.
        prop_assert_eq!(
            shift.apply(Word64::new(word)).get(),
            reference_shift64(kind, word, amount as u32),
        );
    }

    #[test]
    fn word32_integer_product_matches_u64(left in any::<u32>(), right in any::<u32>()) {
        // A 64-bit product splits exactly into two 32-bit witness words.
        let product = u64::from(left) * u64::from(right);
        let words = [
            Word32::new(left),
            Word32::new(right),
            Word32::new(product as u32),
            Word32::new((product >> 32) as u32),
        ];
        let relation = IntegerMulConstraint::new(
            word32_operand(0),
            word32_operand(1),
            word32_operand(2),
            word32_operand(3),
        );
        let system = ConstraintSystem::new(0, 4, vec![], vec![], vec![relation])
            .expect("indices match the declared witness");

        // The scalar checker enforces both result limbs.
        prop_assert_eq!(system.verify(&[], &words), Ok(()));
    }

    #[test]
    fn word64_integer_product_matches_u128(left in any::<u64>(), right in any::<u64>()) {
        // A 128-bit product splits exactly into two 64-bit witness words.
        let product = u128::from(left) * u128::from(right);
        let words = [
            Word64::new(left),
            Word64::new(right),
            Word64::new(product as u64),
            Word64::new((product >> 64) as u64),
        ];
        let relation = IntegerMulConstraint::new(
            word64_operand(0),
            word64_operand(1),
            word64_operand(2),
            word64_operand(3),
        );
        let system = ConstraintSystem::new(0, 4, vec![], vec![], vec![relation])
            .expect("indices match the declared witness");

        // The scalar checker enforces both result limbs.
        prop_assert_eq!(system.verify(&[], &words), Ok(()));
    }
}

#[test]
fn zero_distance_has_one_spelling() {
    // Every zero-distance operation denotes the identity map.
    let shift = Shift::<Word64>::new(ShiftKind::RotateRight, 0).expect("zero is valid");

    // Construction normalizes the map before it enters an operand.
    assert_eq!(shift.kind(), ShiftKind::LogicalLeft);
    assert_eq!(shift.amount(), 0);
    assert!(shift.is_identity());
}

#[test]
fn lane_operations_require_word64() {
    // A 32-bit word has only one lane and cannot express paired lane semantics.
    let error = Shift::<Word32>::new(ShiftKind::Lane32LogicalLeft, 1)
        .expect_err("lane operation must be rejected");

    assert_eq!(error, ShiftError::Lane32RequiresWord64);
}

#[test]
fn shift_amounts_are_bounded_by_their_operation_width() {
    // Full words and lanes have different exclusive upper bounds.
    let full = Shift::<Word64>::new(ShiftKind::LogicalLeft, 64)
        .expect_err("distance reaches the word width");
    let lane = Shift::<Word64>::new(ShiftKind::Lane32LogicalLeft, 32)
        .expect_err("distance reaches the lane width");

    assert_eq!(
        full,
        ShiftError::AmountOutOfRange {
            amount: 64,
            width: 64,
        }
    );
    assert_eq!(
        lane,
        ShiftError::AmountOutOfRange {
            amount: 32,
            width: 32,
        }
    );
}

#[test]
fn two_shift_terms_reject_simpler_maps() {
    // Two rotations combine into one rotation.
    let rotate_3 = Shift::<Word64>::new(ShiftKind::RotateRight, 3).expect("valid shift");
    let rotate_5 = Shift::<Word64>::new(ShiftKind::RotateRight, 5).expect("valid shift");
    let reducible = ShiftedValue::pair(witness(0), rotate_3, rotate_5)
        .expect_err("combined rotation has one-shift form");

    // Two logical shifts that cross the word width erase every source bit.
    let left_40 = Shift::<Word64>::new(ShiftKind::LogicalLeft, 40).expect("valid shift");
    let left_24 = Shift::<Word64>::new(ShiftKind::LogicalLeft, 24).expect("valid shift");
    let zero = ShiftedValue::pair(witness(0), left_40, left_24)
        .expect_err("combined shift is identically zero");

    assert_eq!(reducible, ShiftSequenceError::Reducible);
    assert_eq!(zero, ShiftSequenceError::AlwaysZero);
}

#[test]
fn two_shift_classification_is_complete_for_word32() {
    // Precompute every map that has a canonical zero- or one-shift representation.
    let shifts = shifts32();
    let identity = Shift::identity();
    let singles: HashSet<_> = shifts
        .iter()
        .copied()
        .map(|shift| signature32(shift, identity))
        .collect();
    let zero = [0; 32];

    // Exhaust every ordered pair because accepting a reducible pair changes protocol keys.
    for &inner in &shifts {
        for &outer in &shifts {
            let signature = signature32(inner, outer);
            let result = ShiftedValue::pair(witness(0), inner, outer);
            if signature == zero {
                assert_eq!(
                    result,
                    Err(ShiftSequenceError::AlwaysZero),
                    "inner={inner:?}, outer={outer:?}"
                );
            } else if singles.contains(&signature) {
                assert_eq!(
                    result,
                    Err(ShiftSequenceError::Reducible),
                    "inner={inner:?}, outer={outer:?}"
                );
            } else {
                assert!(result.is_ok(), "inner={inner:?}, outer={outer:?}");
            }
        }
    }
}

#[test]
fn two_shift_classification_is_complete_for_word64() {
    // Precompute every map that has a canonical zero- or one-shift representation.
    let shifts = shifts64();
    let identity = Shift::identity();
    let singles: HashSet<_> = shifts
        .iter()
        .copied()
        .map(|shift| signature64(shift, identity))
        .collect();
    let zero = [0; 64];

    // Exhaust every ordered pair because lane crossings have subtle collapse cases.
    for &inner in &shifts {
        for &outer in &shifts {
            let signature = signature64(inner, outer);
            let result = ShiftedValue::pair(witness(0), inner, outer);
            if signature == zero {
                assert_eq!(
                    result,
                    Err(ShiftSequenceError::AlwaysZero),
                    "inner={inner:?}, outer={outer:?}"
                );
            } else if singles.contains(&signature) {
                assert_eq!(
                    result,
                    Err(ShiftSequenceError::Reducible),
                    "inner={inner:?}, outer={outer:?}"
                );
            } else {
                assert!(result.is_ok(), "inner={inner:?}, outer={outer:?}");
            }
        }
    }
}

#[test]
fn irreducible_shift_pair_applies_inner_then_outer() {
    // Moving left discards the high byte before moving the remainder back.
    let left = Shift::<Word64>::new(ShiftKind::LogicalLeft, 8).expect("valid shift");
    let right = Shift::<Word64>::new(ShiftKind::LogicalRight, 8).expect("valid shift");
    let term =
        ShiftedValue::pair(witness(0), left, right).expect("opposite directions need two slots");
    let operand = Operand::single(term);
    let source = Word64::new(0xff12_3456_789a_bcde);

    // The upper byte is lost while every remaining bit returns to its original position.
    assert_eq!(
        operand.evaluate(&[], &[source]),
        Ok(Word64::new(0x0012_3456_789a_bcde))
    );
}

#[test]
fn operand_is_an_xor_sum() {
    // Repeating one term twice cancels every bit over F_2.
    let term = ShiftedValue::plain(public(0));
    let operand = Operand::new(vec![term, term]);

    assert_eq!(
        operand.evaluate(&[Word32::new(u32::MAX)], &[]),
        Ok(Word32::new(0))
    );
}

#[test]
fn checked_system_accepts_all_relation_families() {
    // Public words feed an equality relation through XOR cancellation.
    let public_term = ShiftedValue::plain(public(0));
    let zero = ZeroConstraint::new(Operand::new(vec![public_term, public_term]));

    // Witness layout: [left, right, and, product_low, product_high].
    let left = 0xfedc_ba98_7654_3210_u64;
    let right = 0x1234_5678_9abc_def0_u64;
    let product = u128::from(left) * u128::from(right);
    let words = [
        Word64::new(left),
        Word64::new(right),
        Word64::new(left & right),
        Word64::new(product as u64),
        Word64::new((product >> 64) as u64),
    ];
    let and = AndConstraint::new(word64_operand(0), word64_operand(1), word64_operand(2));
    let mul = IntegerMulConstraint::new(
        word64_operand(0),
        word64_operand(1),
        word64_operand(3),
        word64_operand(4),
    );
    let system = ConstraintSystem::new(1, 5, vec![zero], vec![and], vec![mul])
        .expect("every term is in range");

    assert_eq!(system.verify(&[Word64::new(7)], &words), Ok(()));
}

#[test]
fn checked_system_rejects_out_of_bounds_terms() {
    // The only term selects the second word of a one-word witness.
    let zero = ZeroConstraint::new(word32_operand(1));
    let error = ConstraintSystem::new(0, 1, vec![zero], vec![], vec![])
        .expect_err("term exceeds the declared witness");

    assert_eq!(
        error,
        SystemError::IndexOutOfBounds {
            kind: ConstraintKind::Zero,
            constraint: 0,
            role: OperandRole::Value,
            term: 0,
            segment: Segment::Witness,
            position: 1,
            len: 1,
        }
    );
}

#[test]
fn verifier_rejects_wrong_shape_and_corrupted_output() {
    // Witness layout: [left, right, and].
    let relation = AndConstraint::new(word32_operand(0), word32_operand(1), word32_operand(2));
    let system = ConstraintSystem::new(0, 3, vec![], vec![relation], vec![])
        .expect("every term is in range");

    // Missing a committed word must fail before any relation is read.
    assert_eq!(
        system.verify(&[], &[Word32::new(1), Word32::new(1)]),
        Err(VerificationError::WitnessLength {
            expected: 3,
            actual: 2,
        })
    );

    // Mutation: claim zero for one AND one.
    let corrupt = [Word32::new(1), Word32::new(1), Word32::new(0)];
    assert_eq!(
        system.verify(&[], &corrupt),
        Err(VerificationError::Unsatisfied {
            kind: ConstraintKind::And,
            constraint: 0,
        })
    );
}

#[test]
fn verifier_rejects_zero_and_integer_product_corruption() {
    // A nonzero witness violates a relation that directly reads one word.
    let zero = ZeroConstraint::new(word32_operand(0));
    let zero_system = ConstraintSystem::new(0, 1, vec![zero], vec![], vec![])
        .expect("the witness term is in range");
    assert_eq!(
        zero_system.verify(&[], &[Word32::new(1)]),
        Err(VerificationError::Unsatisfied {
            kind: ConstraintKind::Zero,
            constraint: 0,
        })
    );

    // Mutation: flip one bit in the high limb of the product 2^31 * 2.
    let relation = IntegerMulConstraint::new(
        word32_operand(0),
        word32_operand(1),
        word32_operand(2),
        word32_operand(3),
    );
    let mul_system = ConstraintSystem::new(0, 4, vec![], vec![], vec![relation])
        .expect("all product terms are in range");
    let corrupt = [
        Word32::new(1 << 31),
        Word32::new(2),
        Word32::new(0),
        Word32::new(0),
    ];
    assert_eq!(
        mul_system.verify(&[], &corrupt),
        Err(VerificationError::Unsatisfied {
            kind: ConstraintKind::IntegerMul,
            constraint: 0,
        })
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn indices_reject_positions_above_u32() {
    // Compact relation addresses are bounded independently of host pointer width.
    let error = ValueIndex::public(u32::MAX as usize + 1)
        .expect_err("position exceeds compact representation");

    assert_eq!(
        error,
        p3_word::IndexError::PositionTooLarge {
            position: u32::MAX as usize + 1,
        }
    );
}
use std::collections::HashSet;
