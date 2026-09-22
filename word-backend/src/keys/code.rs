//! Compact encodings used by the compiled key layout.

use p3_word::{ConstraintKind, Shift, ShiftKind, Word};

const SHIFT_BITS: u32 = 9;
const SEQUENCE_BITS: u32 = 2 * SHIFT_BITS;
const SEQUENCE_MASK: u32 = (1 << SEQUENCE_BITS) - 1;

/// The compact backend encoding of one relation operation and shift sequence.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct KeyCode(u32);

impl KeyCode {
    /// Encodes one checked operation and its shifts.
    #[inline]
    pub(super) fn new<W: Word>(operation: ConstraintKind, shifts: [Shift<W>; 2]) -> Self {
        // The upper bits select the reduction family.
        let operation = u32::from(operation.code());

        // Each movement occupies nine bits below the operation tag.
        Self(
            operation << SEQUENCE_BITS
                | u32::from(ShiftCode::new(shifts[1]).0) << SHIFT_BITS
                | u32::from(ShiftCode::new(shifts[0]).0),
        )
    }

    /// Decodes the relation operation.
    #[inline]
    pub(super) const fn operation(self) -> ConstraintKind {
        // Construction admits exactly the three supported operation tags.
        let code = (self.0 >> SEQUENCE_BITS) as u8;
        ConstraintKind::from_code(code)
            .expect("compact keys contain only assigned relation-family tags")
    }

    /// Returns the operation-independent shift encoding.
    #[inline]
    pub(super) const fn sequence(self) -> ShiftSequenceCode {
        ShiftSequenceCode(self.0 & SEQUENCE_MASK)
    }
}

/// The compact backend encoding of two ordered shifts.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ShiftSequenceCode(u32);

impl ShiftSequenceCode {
    /// Encodes one shift spelling, independently of any relation family.
    #[inline]
    pub(super) fn new<W: Word>(shifts: [Shift<W>; 2]) -> Self {
        // The family tag occupies higher bits, so any tag gives the same sequence.
        KeyCode::new(ConstraintKind::Zero, shifts).sequence()
    }

    /// Decodes the inner and outer shifts in evaluation order.
    #[inline]
    pub(super) fn shifts<W: Word>(self) -> [Shift<W>; 2] {
        // The inner movement occupies the least-significant slot.
        [
            ShiftCode(self.0 as u16).shift(),
            ShiftCode((self.0 >> SHIFT_BITS) as u16).shift(),
        ]
    }
}

/// The compact backend encoding of one checked shift.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShiftCode(u16);

impl ShiftCode {
    /// Encodes a shift kind and its six-bit amount.
    #[inline]
    fn new<W: Word>(shift: Shift<W>) -> Self {
        // Eight movement kinds fit in the upper three bits.
        let kind = u16::from(shift.kind().code());
        Self((kind << 6) | u16::from(shift.amount()))
    }

    /// Decodes one checked shift.
    #[inline]
    fn shift<W: Word>(self) -> Shift<W> {
        // Ignore neighboring slots when decoding from a wider sequence word.
        let code = self.0 & ((1 << SHIFT_BITS) - 1) as u16;
        let kind = ShiftKind::from_code((code >> 6) as u8)
            .expect("three-bit shift tags always name a supported operation");

        // Every encoded amount originated from the checked shift constructor.
        Shift::new(kind, usize::from(code & 0x3f))
            .expect("encoded shifts retain their checked operation width")
    }
}

#[cfg(test)]
mod tests {
    use p3_word::Word64;

    use super::*;

    #[test]
    fn key_codes_round_trip_every_operation_and_shift_slot() {
        // Distinct movements expose accidental slot reversal.
        let inner = Shift::new(ShiftKind::LogicalLeft, 9).unwrap();
        let outer = Shift::new(ShiftKind::RotateRight, 17).unwrap();

        for operation in [
            ConstraintKind::Zero,
            ConstraintKind::And,
            ConstraintKind::IntegerMul,
        ] {
            // Encoding preserves both the reduction family and evaluation order.
            let code = KeyCode::new(operation, [inner, outer]);
            assert_eq!(code.operation(), operation);
            assert_eq!(code.sequence().shifts::<Word64>(), [inner, outer]);
        }
    }
}
