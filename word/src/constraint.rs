use alloc::vec::Vec;

use thiserror::Error;

use crate::index::{Segment, ValueIndex};
use crate::shift::ShiftedValue;
use crate::word::{Word, sealed};

/// An error returned when an operand addresses a missing word.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("{segment:?} word {position} is out of bounds for segment length {len}")]
pub struct EvaluationError {
    /// The segment selected by the term.
    pub segment: Segment,
    /// The missing position.
    pub position: u32,
    /// The available number of words.
    pub len: usize,
}

/// An XOR sum of shifted words.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Operand<W: Word> {
    /// The shifted words combined by bitwise XOR.
    terms: Vec<ShiftedValue<W>>,
}

impl<W: Word> Operand<W> {
    /// Creates an operand from its shifted word terms.
    #[inline]
    pub const fn new(terms: Vec<ShiftedValue<W>>) -> Self {
        Self { terms }
    }

    /// Creates an operand containing one term.
    #[inline]
    pub fn single(term: ShiftedValue<W>) -> Self {
        Self {
            terms: alloc::vec![term],
        }
    }

    /// Returns the shifted word terms.
    #[inline]
    pub fn terms(&self) -> &[ShiftedValue<W>] {
        &self.terms
    }

    /// Evaluates the XOR sum against public and committed words.
    pub fn evaluate(&self, public: &[W], witness: &[W]) -> Result<W, EvaluationError> {
        let mut value = W::ZERO;
        for term in &self.terms {
            let index = term.index();
            let words = match index.segment() {
                Segment::Public => public,
                Segment::Witness => witness,
            };
            let position = index.position() as usize;
            let word = words
                .get(position)
                .copied()
                .ok_or_else(|| EvaluationError {
                    segment: index.segment(),
                    position: index.position(),
                    len: words.len(),
                })?;
            // Addition in the binary relation is bitwise XOR.
            value = sealed::Sealed::xor(value, term.evaluate(word));
        }
        Ok(value)
    }

    pub(crate) fn indices(&self) -> impl Iterator<Item = (usize, ValueIndex)> + '_ {
        // Preserve the term position so validation can report the precise source.
        self.terms
            .iter()
            .enumerate()
            .map(|(term, value)| (term, value.index()))
    }
}

/// A relation requiring an XOR operand to equal zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroConstraint<W: Word> {
    /// The XOR expression required to vanish.
    value: Operand<W>,
}

impl<W: Word> ZeroConstraint<W> {
    /// Creates a zero relation.
    #[inline]
    pub const fn new(value: Operand<W>) -> Self {
        Self { value }
    }

    /// Returns the operand required to vanish.
    #[inline]
    pub const fn value(&self) -> &Operand<W> {
        &self.value
    }

    pub(crate) fn is_satisfied(&self, public: &[W], witness: &[W]) -> bool {
        // A validated system makes operand evaluation infallible.
        self.value.evaluate(public, witness) == Ok(W::ZERO)
    }
}

/// A relation requiring the bitwise AND of two operands to equal a third.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AndConstraint<W: Word> {
    /// The left bitwise factor.
    left: Operand<W>,
    /// The right bitwise factor.
    right: Operand<W>,
    /// The expected bitwise result.
    output: Operand<W>,
}

impl<W: Word> AndConstraint<W> {
    /// Creates a bitwise AND relation.
    #[inline]
    pub const fn new(left: Operand<W>, right: Operand<W>, output: Operand<W>) -> Self {
        Self {
            left,
            right,
            output,
        }
    }

    /// Returns the left input operand.
    #[inline]
    pub const fn left(&self) -> &Operand<W> {
        &self.left
    }

    /// Returns the right input operand.
    #[inline]
    pub const fn right(&self) -> &Operand<W> {
        &self.right
    }

    /// Returns the output operand.
    #[inline]
    pub const fn output(&self) -> &Operand<W> {
        &self.output
    }

    pub(crate) fn is_satisfied(&self, public: &[W], witness: &[W]) -> bool {
        // A validated system makes all three evaluations infallible.
        let Ok(left) = self.left.evaluate(public, witness) else {
            return false;
        };
        let Ok(right) = self.right.evaluate(public, witness) else {
            return false;
        };
        let Ok(output) = self.output.evaluate(public, witness) else {
            return false;
        };
        sealed::Sealed::and(left, right) == output
    }
}

/// A relation requiring a full-width unsigned integer product.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntegerMulConstraint<W: Word> {
    /// The left unsigned factor.
    left: Operand<W>,
    /// The right unsigned factor.
    right: Operand<W>,
    /// The low word of the expected product.
    low: Operand<W>,
    /// The high word of the expected product.
    high: Operand<W>,
}

impl<W: Word> IntegerMulConstraint<W> {
    /// Creates an unsigned integer multiplication relation.
    #[inline]
    pub const fn new(
        left: Operand<W>,
        right: Operand<W>,
        low: Operand<W>,
        high: Operand<W>,
    ) -> Self {
        Self {
            left,
            right,
            low,
            high,
        }
    }

    /// Returns the left factor.
    #[inline]
    pub const fn left(&self) -> &Operand<W> {
        &self.left
    }

    /// Returns the right factor.
    #[inline]
    pub const fn right(&self) -> &Operand<W> {
        &self.right
    }

    /// Returns the low result limb.
    #[inline]
    pub const fn low(&self) -> &Operand<W> {
        &self.low
    }

    /// Returns the high result limb.
    #[inline]
    pub const fn high(&self) -> &Operand<W> {
        &self.high
    }

    pub(crate) fn is_satisfied(&self, public: &[W], witness: &[W]) -> bool {
        // A validated system makes all four evaluations infallible.
        let Ok(left) = self.left.evaluate(public, witness) else {
            return false;
        };
        let Ok(right) = self.right.evaluate(public, witness) else {
            return false;
        };
        let Ok(low) = self.low.evaluate(public, witness) else {
            return false;
        };
        let Ok(high) = self.high.evaluate(public, witness) else {
            return false;
        };
        // Native widening preserves all 2 * word_bits product bits.
        sealed::Sealed::wide_mul(left, right) == (low, high)
    }
}
