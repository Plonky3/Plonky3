//! Operand claims entering the shift reduction.

use alloc::vec::Vec;

use p3_field::Field;

/// Operand-column evaluations at one shared constraint and bit point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShiftClaim<F> {
    /// Point selecting a padded constraint row.
    constraint_point: Vec<F>,
    /// Point selecting a bit within each word.
    bit_point: Vec<F>,
    /// Evaluation of the operand required to vanish.
    zero: [F; 1],
    /// Evaluations of the left input, right input, and output.
    bitwise_and: [F; 3],
    /// Evaluations of the factors followed by the low and high result limbs.
    integer_mul: [F; 4],
}

impl<F> ShiftClaim<F> {
    /// Creates the complete set of claims consumed by one reduction.
    #[must_use]
    pub const fn new(
        constraint_point: Vec<F>,
        bit_point: Vec<F>,
        zero: [F; 1],
        bitwise_and: [F; 3],
        integer_mul: [F; 4],
    ) -> Self {
        // Fixed-size arrays make every relation family carry its exact arity.
        Self {
            constraint_point,
            bit_point,
            zero,
            bitwise_and,
            integer_mul,
        }
    }

    /// Returns the point selecting a padded constraint row.
    #[inline]
    pub fn constraint_point(&self) -> &[F] {
        // The statement owns the point for the complete reduction.
        &self.constraint_point
    }

    /// Returns the point selecting a bit within each word.
    #[inline]
    pub fn bit_point(&self) -> &[F] {
        // Every operand claim uses the same bit coordinate.
        &self.bit_point
    }

    /// Returns the zero-relation operand evaluation.
    #[inline]
    pub const fn zero(&self) -> &[F; 1] {
        // The family has exactly one semantic operand.
        &self.zero
    }

    /// Returns the three bitwise-product operand evaluations.
    #[inline]
    pub const fn bitwise_and(&self) -> &[F; 3] {
        // The order is left, right, then output.
        &self.bitwise_and
    }

    /// Returns the four unsigned-product operand evaluations.
    #[inline]
    pub const fn integer_mul(&self) -> &[F; 4] {
        // The order is left, right, low, then high.
        &self.integer_mul
    }

    /// Flattens claims in the relation and operand order fixed by the protocol.
    pub(crate) fn flattened(&self) -> Vec<F>
    where
        F: Copy,
    {
        // The transcript binds the same order used by batching and compiled keys.
        self.zero
            .iter()
            .chain(&self.bitwise_and)
            .chain(&self.integer_mul)
            .copied()
            .collect()
    }
}

/// One evaluation claim on the committed bit trace.
///
/// The leading coordinates select a padded word and the trailing ones select a bit within it.
#[must_use = "leaf claims must be tied to committed columns"]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShiftOpeningClaim<F> {
    /// Point over the word index followed by the within-word bit index.
    point: Vec<F>,
    /// Claimed multilinear evaluation of the committed bits.
    value: F,
}

impl<F> ShiftOpeningClaim<F> {
    /// Creates an opening claim in the Boolean PCS coordinate order.
    pub(crate) const fn new(point: Vec<F>, value: F) -> Self {
        // The reduction constructs the point from transcript-derived coordinates.
        Self { point, value }
    }

    /// Returns the point over word and bit coordinates.
    #[inline]
    pub fn point(&self) -> &[F] {
        // Word coordinates precede the trailing within-word coordinates.
        &self.point
    }

    /// Returns the claimed committed-trace evaluation.
    #[inline]
    pub const fn value(&self) -> F
    where
        F: Copy,
    {
        // The value is tied to the returned point by both sumchecks.
        self.value
    }
}

impl<F: Field> ShiftClaim<F> {
    /// Returns the claim after batching both public axes by equality weights.
    pub(crate) fn batched(&self, operation: &[F], operand: &[F]) -> F {
        // Three relation families occupy the first three vertices of a two-bit cube.
        let families = [
            self.zero.as_slice(),
            self.bitwise_and.as_slice(),
            self.integer_mul.as_slice(),
        ];

        // Each family reads the prefix matching its semantic arity.
        families
            .into_iter()
            .enumerate()
            .map(|(family, claims)| {
                let family_claim = claims
                    .iter()
                    .zip(operand)
                    .map(|(&claim, &weight)| claim * weight)
                    .sum::<F>();
                operation[family] * family_claim
            })
            .sum()
    }
}
