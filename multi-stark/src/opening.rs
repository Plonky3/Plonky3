//! Reduce a finished AIR zerocheck to evaluation claims about the committed columns.
//!
//! The sumcheck binds every variable to a single point `r`.
//! Satisfying the AIR then reduces to a handful of evaluation claims at `r`:
//!
//! - every column's value at `r` — the current-row claims, opened as `Eq`,
//! - the successor value of each column read ahead — the next-row claims, opened as `Next`.
//!
//! Boundary selectors carry no claim: the verifier evaluates them in closed form at `r`.
//! The next-row claims open the same committed columns, so they need no second commitment.

use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::point::Point;

/// The evaluation claims an AIR zerocheck reduces to, all at one point.
///
/// A commitment scheme later opens this bundle against the trace commitment.
/// Until then the values are trusted and feed only the verifier's constraint recompute.
// TODO: produce this from the prover instead of only reconstructing it inside the verifier.
// The commitment-opening step will consume it directly.
#[derive(Clone, Debug)]
pub struct OpeningClaims<EF> {
    /// Point `r` at which every claim is made: the bound sumcheck challenges.
    pub point: Point<EF>,
    /// Current-row value of each main column, in column order.
    pub local: Vec<EF>,
    /// Next-row claims, each pairing a column index with its repeat-last successor value at `r`.
    ///
    /// Only the columns the AIR reads on the next row appear here.
    pub next: Vec<(usize, EF)>,
}

impl<EF: Field> OpeningClaims<EF> {
    /// Bundle the reduced claims from the bound point and the opened column values.
    ///
    /// # Arguments
    ///
    /// - `point`: the bound sumcheck challenges.
    /// - `local`: current-row value of each main column, in column order.
    /// - `next_columns`: indices of the columns the AIR reads on the next row.
    /// - `next_values`: successor value of each such column, aligned with `next_columns`.
    ///
    /// # Panics
    ///
    /// Panics if `next_columns` and `next_values` have different lengths.
    #[must_use]
    pub fn new(
        point: Point<EF>,
        local: Vec<EF>,
        next_columns: &[usize],
        next_values: &[EF],
    ) -> Self {
        // One claimed successor value per declared next-row column, or the pairing is undefined.
        assert_eq!(next_columns.len(), next_values.len());

        // Pair each declared column index with its claimed successor value.
        let next = next_columns
            .iter()
            .copied()
            .zip(next_values.iter().copied())
            .collect();

        Self { point, local, next }
    }

    /// Reconstruct the full-width next-row evaluation vector the folder reads from.
    ///
    /// Columns the AIR never reads ahead stay zero, since their value never reaches the constraint.
    ///
    /// # Arguments
    ///
    /// - `width`: number of main columns.
    #[must_use]
    pub fn next_row(&self, width: usize) -> Vec<EF> {
        // Start from all zeros: undeclared columns are never read, so zero is a safe filler.
        let mut row = EF::zero_vec(width);

        // Scatter each claimed successor value into its column slot.
        for &(column, value) in &self.next {
            row[column] = value;
        }

        row
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn next_row_scatters_declared_columns_and_zeroes_the_rest() {
        // Fixture state: a width-4 trace whose AIR reads only columns 0 and 2 on the next row.
        //
        //     next_columns: [0, 2]
        //     next_values : [7, 9]
        //
        // The reconstructed row places each value in its slot and leaves the rest zero:
        //
        //     row index:  0    1    2    3
        //     value:      7    0    9    0
        let point = Point::new(vec![EF::from_u64(1), EF::from_u64(2)]);
        let local = vec![EF::from_u64(3); 4];
        let claims = OpeningClaims::new(point, local, &[0, 2], &[EF::from_u64(7), EF::from_u64(9)]);

        let row = claims.next_row(4);

        assert_eq!(
            row,
            vec![EF::from_u64(7), EF::ZERO, EF::from_u64(9), EF::ZERO]
        );
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn new_rejects_mismatched_next_lengths() {
        // Two declared columns but only one claimed value: the pairing is undefined.
        let point = Point::new(vec![EF::from_u64(1)]);
        let local = vec![EF::from_u64(2)];
        let _ = OpeningClaims::new(point, local, &[0, 1], &[EF::from_u64(5)]);
    }
}
