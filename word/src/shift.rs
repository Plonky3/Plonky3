use core::marker::PhantomData;

use thiserror::Error;

use crate::index::ValueIndex;
use crate::word::{Word, sealed};

/// A bit movement supported by a shifted word term.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShiftKind {
    /// Moves bits left and fills with zero.
    LogicalLeft,
    /// Moves bits right and fills with zero.
    LogicalRight,
    /// Moves bits right and replicates the sign bit.
    ArithmeticRight,
    /// Rotates bits right across the full word.
    RotateRight,
    /// Moves both 32-bit lanes left independently.
    Lane32LogicalLeft,
    /// Moves both 32-bit lanes right independently.
    Lane32LogicalRight,
    /// Moves both 32-bit lanes right with independent sign extension.
    Lane32ArithmeticRight,
    /// Rotates both 32-bit lanes right independently.
    Lane32RotateRight,
}

impl ShiftKind {
    /// Returns whether the operation acts on two independent 32-bit lanes.
    #[inline]
    pub const fn is_lane32(self) -> bool {
        // The lane family is deliberately unavailable to 32-bit words.
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
        // Lane operations have fixed 64-bit container semantics.
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
    pub fn pair(
        index: ValueIndex,
        inner: Shift<W>,
        outer: Shift<W>,
    ) -> Result<Self, ShiftSequenceError> {
        // Two slots are reserved only for maps that genuinely need both.
        match compose(inner, outer) {
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

    /// Returns the first shift.
    #[inline]
    pub const fn inner(self) -> Shift<W> {
        self.shifts[0]
    }

    /// Returns the second shift.
    #[inline]
    pub const fn outer(self) -> Shift<W> {
        self.shifts[1]
    }

    pub(crate) fn evaluate(self, word: W) -> W {
        // Apply the two slots in their protocol order.
        self.outer().apply(self.inner().apply(word))
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

fn compose<W: Word>(inner: Shift<W>, outer: Shift<W>) -> Composition {
    // An identity leaves the other movement as a single shift.
    if inner.is_identity() || outer.is_identity() {
        return Composition::Single;
    }

    // A saturated arithmetic shift is unchanged by a compatible rotation.
    if is_degenerate(inner, outer) {
        return Composition::Single;
    }

    // Movements in one direction combine by adding their distances.
    let Some(kind) = chained_kind(inner, outer) else {
        return Composition::Pair;
    };
    let distance = usize::from(inner.amount) + usize::from(outer.amount);

    // Logical shifts discard every bit once they cross the operation width.
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
        return Composition::Zero;
    }

    // Arithmetic shifts saturate and rotations wrap.
    // Both cases still have a single-shift representation.
    Composition::Single
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
        (ShiftKind::ArithmeticRight, ShiftKind::Lane32ArithmeticRight) if inner.amount >= 32 => {
            Some(ShiftKind::ArithmeticRight)
        }
        (ShiftKind::Lane32ArithmeticRight, ShiftKind::ArithmeticRight) if outer.amount >= 32 => {
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
        (ShiftKind::Lane32ArithmeticRight, ShiftKind::Lane32RotateRight) if inner.amount == 31 => {
            true
        }
        (
            ShiftKind::ArithmeticRight | ShiftKind::Lane32ArithmeticRight,
            ShiftKind::LogicalRight,
        ) if outer.amount == full_last => true,
        (ShiftKind::Lane32ArithmeticRight, ShiftKind::Lane32LogicalRight) if outer.amount == 31 => {
            true
        }
        _ => false,
    }
}
