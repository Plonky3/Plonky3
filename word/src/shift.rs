use core::marker::PhantomData;

use thiserror::Error;

use crate::index::ValueIndex;
use crate::word::{Word, sealed};

/// A bit movement supported by a shifted word term.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum ShiftKind {
    /// Moves bits left and fills with zero.
    LogicalLeft = 0,
    /// Moves bits right and fills with zero.
    LogicalRight = 1,
    /// Moves bits right and replicates the sign bit.
    ArithmeticRight = 2,
    /// Rotates bits right across the full word.
    RotateRight = 3,
    /// Moves both 32-bit lanes left independently.
    Lane32LogicalLeft = 4,
    /// Moves both 32-bit lanes right independently.
    Lane32LogicalRight = 5,
    /// Moves both 32-bit lanes right with independent sign extension.
    Lane32ArithmeticRight = 6,
    /// Rotates both 32-bit lanes right independently.
    Lane32RotateRight = 7,
}

impl ShiftKind {
    /// Every movement in compact-code order.
    const ALL: [Self; 8] = [
        Self::LogicalLeft,
        Self::LogicalRight,
        Self::ArithmeticRight,
        Self::RotateRight,
        Self::Lane32LogicalLeft,
        Self::Lane32LogicalRight,
        Self::Lane32ArithmeticRight,
        Self::Lane32RotateRight,
    ];

    /// Returns the stable three-bit operation code.
    #[inline]
    pub const fn code(self) -> u8 {
        // Explicit discriminants define the compact protocol representation.
        self as u8
    }

    /// Recovers a movement from its three-bit operation code.
    #[inline]
    pub const fn from_code(code: u8) -> Option<Self> {
        // The dense table rejects values outside the assigned discriminants.
        if code < Self::ALL.len() as u8 {
            Some(Self::ALL[code as usize])
        } else {
            None
        }
    }

    /// Returns whether the operation acts on two independent 32-bit lanes.
    #[inline]
    pub const fn is_lane32(self) -> bool {
        // Group every operation whose semantics are confined to 32-bit lanes.
        matches!(
            self,
            Self::Lane32LogicalLeft
                | Self::Lane32LogicalRight
                | Self::Lane32ArithmeticRight
                | Self::Lane32RotateRight
        )
    }
}

/// An invalid shift description.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ShiftError {
    /// A lane operation was requested for a word without two 32-bit lanes.
    #[error("32-bit lane shifts require a 64-bit word")]
    Lane32RequiresWord64,
    /// The distance reaches or exceeds the operation's width.
    #[error("shift amount {amount} must be below {width}")]
    AmountOutOfRange {
        /// The rejected distance.
        amount: usize,
        /// The exclusive upper bound.
        width: u32,
    },
}

/// One canonical bit movement.
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct Shift<W: Word> {
    /// The bit movement to apply.
    kind: ShiftKind,
    /// The movement distance in bits.
    amount: u8,
    /// The word width fixed at the type level.
    word: PhantomData<fn() -> W>,
}

impl<W: Word> Copy for Shift<W> {}

impl<W: Word> Clone for Shift<W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<W: Word> Shift<W> {
    /// Creates a checked shift.
    ///
    /// A zero distance is normalized to the unique identity spelling.
    pub fn new(kind: ShiftKind, amount: usize) -> Result<Self, ShiftError> {
        // Two independent lanes require a 64-bit container.
        if kind.is_lane32() && W::BITS != 64 {
            return Err(ShiftError::Lane32RequiresWord64);
        }

        // Full-word shifts use the selected word width.
        let width = if kind.is_lane32() { 32 } else { W::BITS };
        if amount >= width as usize {
            return Err(ShiftError::AmountOutOfRange { amount, width });
        }

        // Every operation is the identity at distance zero.
        let kind = if amount == 0 {
            ShiftKind::LogicalLeft
        } else {
            kind
        };
        Ok(Self {
            kind,
            amount: amount as u8,
            word: PhantomData,
        })
    }

    /// Returns the unique identity shift.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            kind: ShiftKind::LogicalLeft,
            amount: 0,
            word: PhantomData,
        }
    }

    /// Returns the bit movement.
    #[inline]
    pub const fn kind(self) -> ShiftKind {
        self.kind
    }

    /// Returns the distance in bits.
    #[inline]
    pub const fn amount(self) -> u8 {
        self.amount
    }

    /// Returns whether every input word is left unchanged.
    #[inline]
    pub const fn is_identity(self) -> bool {
        self.amount == 0
    }

    /// Applies the movement to one word.
    #[inline]
    pub fn apply(self, word: W) -> W {
        let amount = u32::from(self.amount);
        match self.kind {
            ShiftKind::LogicalLeft => sealed::Sealed::logical_left(word, amount),
            ShiftKind::LogicalRight => sealed::Sealed::logical_right(word, amount),
            ShiftKind::ArithmeticRight => sealed::Sealed::arithmetic_right(word, amount),
            ShiftKind::RotateRight => sealed::Sealed::rotate_right(word, amount),
            ShiftKind::Lane32LogicalLeft => sealed::Sealed::lane32_left(word, amount)
                .expect("lane shifts are constructed only for 64-bit words"),
            ShiftKind::Lane32LogicalRight => sealed::Sealed::lane32_right(word, amount)
                .expect("lane shifts are constructed only for 64-bit words"),
            ShiftKind::Lane32ArithmeticRight => {
                sealed::Sealed::lane32_arithmetic_right(word, amount)
                    .expect("lane shifts are constructed only for 64-bit words")
            }
            ShiftKind::Lane32RotateRight => sealed::Sealed::lane32_rotate_right(word, amount)
                .expect("lane shifts are constructed only for 64-bit words"),
        }
    }
}

/// An invalid two-shift representation.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ShiftSequenceError {
    /// The pair is equivalent to one shift.
    #[error("the shift pair has a canonical single-shift representation")]
    Reducible,
    /// The pair clears every input bit.
    #[error("the shift pair always evaluates to zero")]
    AlwaysZero,
}

/// One indexed word after zero, one, or two canonical shifts.
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct ShiftedValue<W: Word> {
    /// The source word position.
    index: ValueIndex,
    /// The inner and outer movements in evaluation order.
    shifts: [Shift<W>; 2],
}

impl<W: Word> Copy for ShiftedValue<W> {}

impl<W: Word> Clone for ShiftedValue<W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<W: Word> ShiftedValue<W> {
    /// Creates an unshifted word term.
    #[inline]
    pub const fn plain(index: ValueIndex) -> Self {
        Self {
            index,
            shifts: [Shift::identity(), Shift::identity()],
        }
    }

    /// Creates a word term with one checked shift.
    #[inline]
    pub const fn single(index: ValueIndex, shift: Shift<W>) -> Self {
        Self {
            index,
            shifts: [shift, Shift::identity()],
        }
    }

    /// Creates a word term whose two shifts cannot be represented more simply.
    ///
    /// Accepted pairs are irreducible but need not have a unique spelling.
    pub fn pair(
        index: ValueIndex,
        inner: Shift<W>,
        outer: Shift<W>,
    ) -> Result<Self, ShiftSequenceError> {
        // Two slots are reserved only for maps that genuinely need both.
        match Composition::classify(inner, outer) {
            Composition::Pair => Ok(Self {
                index,
                shifts: [inner, outer],
            }),
            Composition::Single => Err(ShiftSequenceError::Reducible),
            Composition::Zero => Err(ShiftSequenceError::AlwaysZero),
        }
    }

    /// Returns the referenced word position.
    #[inline]
    pub const fn index(self) -> ValueIndex {
        self.index
    }

    /// Returns both movements in evaluation order.
    #[inline]
    pub const fn shifts(self) -> [Shift<W>; 2] {
        self.shifts
    }

    /// Applies the inner movement before the outer movement.
    #[inline]
    pub fn apply(self, word: W) -> W {
        // Apply the two slots in their protocol order.
        self.shifts[1].apply(self.shifts[0].apply(word))
    }
}

enum Composition {
    /// The composition has an equivalent single movement.
    Single,
    /// The composition clears every input bit.
    Zero,
    /// The composition irreducibly requires both movements.
    Pair,
}

impl Composition {
    fn classify<W: Word>(inner: Shift<W>, outer: Shift<W>) -> Self {
        // An identity leaves the other movement as a single shift.
        if inner.is_identity() || outer.is_identity() {
            return Self::Single;
        }

        // A saturated arithmetic shift is unchanged by a compatible rotation.
        if Self::is_degenerate(inner, outer) {
            return Self::Single;
        }

        // Movements in one direction combine by adding their distances.
        let Some(kind) = Self::chained_kind(inner, outer) else {
            return Self::Pair;
        };
        let distance = usize::from(inner.amount) + usize::from(outer.amount);

        // Logical shifts erase every bit after crossing their operation width.
        if matches!(
            kind,
            ShiftKind::LogicalLeft
                | ShiftKind::LogicalRight
                | ShiftKind::Lane32LogicalLeft
                | ShiftKind::Lane32LogicalRight
        ) && distance
            >= if kind.is_lane32() {
                32
            } else {
                W::BITS as usize
            }
        {
            return Self::Zero;
        }

        // Arithmetic shifts saturate while rotations wrap.
        // Both cases still have a single-shift representation.
        Self::Single
    }

    fn chained_kind<W: Word>(inner: Shift<W>, outer: Shift<W>) -> Option<ShiftKind> {
        // Equal operations always continue in the same direction.
        if inner.kind == outer.kind {
            return Some(inner.kind);
        }

        // A logical right shift clears the sign before arithmetic extension.
        match (inner.kind, outer.kind) {
            (ShiftKind::LogicalRight, ShiftKind::ArithmeticRight) => Some(ShiftKind::LogicalRight),
            (ShiftKind::Lane32LogicalRight, ShiftKind::Lane32ArithmeticRight) => {
                Some(ShiftKind::Lane32LogicalRight)
            }
            // Crossing 32 bits leaves each lane with bits from only one original half.
            (ShiftKind::LogicalLeft, ShiftKind::Lane32LogicalLeft) if inner.amount >= 32 => {
                Some(ShiftKind::LogicalLeft)
            }
            (ShiftKind::Lane32LogicalLeft, ShiftKind::LogicalLeft) if outer.amount >= 32 => {
                Some(ShiftKind::LogicalLeft)
            }
            (ShiftKind::LogicalRight, ShiftKind::Lane32LogicalRight) if inner.amount >= 32 => {
                Some(ShiftKind::LogicalRight)
            }
            (ShiftKind::Lane32LogicalRight, ShiftKind::LogicalRight) if outer.amount >= 32 => {
                Some(ShiftKind::LogicalRight)
            }
            (ShiftKind::ArithmeticRight, ShiftKind::Lane32ArithmeticRight)
                if inner.amount >= 32 =>
            {
                Some(ShiftKind::ArithmeticRight)
            }
            (ShiftKind::Lane32ArithmeticRight, ShiftKind::ArithmeticRight)
                if outer.amount >= 32 =>
            {
                Some(ShiftKind::ArithmeticRight)
            }
            (ShiftKind::LogicalRight, ShiftKind::Lane32ArithmeticRight) if inner.amount >= 33 => {
                Some(ShiftKind::LogicalRight)
            }
            (ShiftKind::Lane32LogicalRight, ShiftKind::ArithmeticRight) if outer.amount >= 32 => {
                Some(ShiftKind::LogicalRight)
            }
            _ => None,
        }
    }

    const fn is_degenerate<W: Word>(inner: Shift<W>, outer: Shift<W>) -> bool {
        // A full-width arithmetic shift can collapse a word to one repeated sign bit.
        let full_last = W::BITS as u8 - 1;
        match (inner.kind, outer.kind) {
            (ShiftKind::ArithmeticRight, ShiftKind::RotateRight | ShiftKind::Lane32RotateRight)
                if inner.amount == full_last =>
            {
                true
            }
            (ShiftKind::Lane32ArithmeticRight, ShiftKind::Lane32RotateRight)
                if inner.amount == 31 =>
            {
                true
            }
            (
                ShiftKind::ArithmeticRight | ShiftKind::Lane32ArithmeticRight,
                ShiftKind::LogicalRight,
            ) if outer.amount == full_last => true,
            (ShiftKind::Lane32ArithmeticRight, ShiftKind::Lane32LogicalRight)
                if outer.amount == 31 =>
            {
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::{Word32, Word64};

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

    fn witness(position: usize) -> ValueIndex {
        // Test positions remain inside the compact address space.
        ValueIndex::witness(position).expect("test position must fit")
    }

    fn reference_shift32(kind: ShiftKind, word: u32, amount: u32) -> u32 {
        // Spell each operation directly in primitive integer arithmetic.
        match kind {
            ShiftKind::LogicalLeft => word << amount,
            ShiftKind::LogicalRight => word >> amount,
            ShiftKind::ArithmeticRight => ((word as i32) >> amount) as u32,
            ShiftKind::RotateRight => word.rotate_right(amount),
            _ => unreachable!("32-bit lane operations require a 64-bit container"),
        }
    }

    fn reference_shift64(kind: ShiftKind, word: u64, amount: u32) -> u64 {
        // Confine each lane operation to its original 32-bit half.
        let lanes = |operation: fn(u32, u32) -> u32| {
            let low = operation(word as u32, amount);
            let high = operation((word >> 32) as u32, amount);
            u64::from(low) | (u64::from(high) << 32)
        };

        // Spell each operation directly in primitive integer arithmetic.
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
        // Include the unique identity and every nonzero full-word movement.
        let mut shifts = vec![Shift::identity()];
        for kind in FULL_KINDS {
            for amount in 1..32 {
                shifts.push(Shift::new(kind, amount).expect("enumerated shift is valid"));
            }
        }
        shifts
    }

    fn shifts64() -> Vec<Shift<Word64>> {
        // Include the unique identity and every valid nonzero movement.
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
        // An F_2-linear map is determined by the images of all basis bits.
        core::array::from_fn(|bit| {
            let inner_image = reference_shift32(inner.kind(), 1 << bit, inner.amount().into());
            reference_shift32(outer.kind(), inner_image, outer.amount().into())
        })
    }

    fn signature64(inner: Shift<Word64>, outer: Shift<Word64>) -> [u64; 64] {
        // An F_2-linear map is determined by the images of all basis bits.
        core::array::from_fn(|bit| {
            let inner_image = reference_shift64(inner.kind(), 1 << bit, inner.amount().into());
            reference_shift64(outer.kind(), inner_image, outer.amount().into())
        })
    }

    proptest! {
        #[test]
        fn word32_shifts_match_primitive_arithmetic(
            word in any::<u32>(),
            kind_index in 0usize..FULL_KINDS.len(),
            amount in 0usize..32,
        ) {
            // Every valid full-word operation and distance is constructible.
            let kind = FULL_KINDS[kind_index];
            let shift = Shift::<Word32>::new(kind, amount).expect("valid shift");

            // The checked operation must equal the independent integer expression.
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

            // The checked operation must equal the independent integer expression.
            prop_assert_eq!(
                shift.apply(Word64::new(word)).get(),
                reference_shift64(kind, word, amount as u32),
            );
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
    fn operation_codes_round_trip_and_reject_out_of_range_values() {
        // Every assigned three-bit tag recovers its exact movement semantics.
        for kind in ShiftKind::ALL {
            assert_eq!(ShiftKind::from_code(kind.code()), Some(kind));
        }

        // Eight is the first value outside the assigned operation space.
        assert_eq!(ShiftKind::from_code(8), None);
        assert_eq!(ShiftKind::from_code(u8::MAX), None);
    }

    #[test]
    fn lane_operations_require_word64() {
        // A 32-bit word cannot express two independent lanes.
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

        // Crossing the word width erases every source bit.
        let left_40 = Shift::<Word64>::new(ShiftKind::LogicalLeft, 40).expect("valid shift");
        let left_24 = Shift::<Word64>::new(ShiftKind::LogicalLeft, 24).expect("valid shift");
        let zero = ShiftedValue::pair(witness(0), left_40, left_24)
            .expect_err("combined shift is identically zero");

        assert_eq!(reducible, ShiftSequenceError::Reducible);
        assert_eq!(zero, ShiftSequenceError::AlwaysZero);
    }

    #[test]
    fn two_shift_classification_is_complete_for_word32() {
        // Precompute every map with a zero- or one-shift representation.
        let shifts = shifts32();
        let identity = Shift::identity();
        let singles: BTreeSet<_> = shifts
            .iter()
            .copied()
            .map(|shift| signature32(shift, identity))
            .collect();
        let zero = [0; 32];

        // Exhaust every ordered pair against the independent bit-linear model.
        for &inner in &shifts {
            for &outer in &shifts {
                let signature = signature32(inner, outer);
                let result = ShiftedValue::pair(witness(0), inner, outer);
                if signature == zero {
                    assert_eq!(result, Err(ShiftSequenceError::AlwaysZero));
                } else if singles.contains(&signature) {
                    assert_eq!(result, Err(ShiftSequenceError::Reducible));
                } else {
                    assert!(result.is_ok());
                }
            }
        }
    }

    #[test]
    fn two_shift_classification_is_complete_for_word64() {
        // Precompute every map with a zero- or one-shift representation.
        let shifts = shifts64();
        let identity = Shift::identity();
        let singles: BTreeSet<_> = shifts
            .iter()
            .copied()
            .map(|shift| signature64(shift, identity))
            .collect();
        let zero = [0; 64];

        // Exhaust every ordered pair against the independent bit-linear model.
        for &inner in &shifts {
            for &outer in &shifts {
                let signature = signature64(inner, outer);
                let result = ShiftedValue::pair(witness(0), inner, outer);
                if signature == zero {
                    assert_eq!(result, Err(ShiftSequenceError::AlwaysZero));
                } else if singles.contains(&signature) {
                    assert_eq!(result, Err(ShiftSequenceError::Reducible));
                } else {
                    assert!(result.is_ok());
                }
            }
        }
    }

    #[test]
    fn irreducible_pair_applies_inner_then_outer() {
        // Moving left discards the high byte before moving the remainder back.
        let left = Shift::<Word64>::new(ShiftKind::LogicalLeft, 8).expect("valid shift");
        let right = Shift::<Word64>::new(ShiftKind::LogicalRight, 8).expect("valid shift");
        let term = ShiftedValue::pair(witness(0), left, right)
            .expect("opposite directions need two slots");
        let source = Word64::new(0xff12_3456_789a_bcde);

        // The upper byte is lost while every remaining bit returns to its position.
        assert_eq!(term.apply(source), Word64::new(0x0012_3456_789a_bcde));
        assert_eq!(term.shifts(), [left, right]);
    }
}
