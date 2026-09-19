//! Sparse-to-dense selector polynomial from the basic jagged construction.

use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::{JaggedError, JaggedLayout, JaggedPoint};

/// Selector polynomial induced by one validated sparse layout.
pub(super) struct JaggedSelector<'a> {
    /// Sparse geometry that determines every dense-to-sparse mapping.
    layout: &'a JaggedLayout,
}

impl<'a> JaggedSelector<'a> {
    /// Borrows the geometry shared by selector materialization and evaluation.
    pub(super) const fn new(layout: &'a JaggedLayout) -> Self {
        // The layout constructor already validated every boundary.
        Self { layout }
    }

    /// Validates that a sparse evaluation point matches this layout.
    pub(super) const fn validate_point<F: Field>(
        &self,
        point: &JaggedPoint<F>,
    ) -> Result<(), JaggedError> {
        // The row coordinates address the provisioned row space.
        if point.row().num_variables() != self.layout.row_variables() {
            return Err(JaggedError::RowPointWidthMismatch {
                expected: self.layout.row_variables(),
                actual: point.row().num_variables(),
            });
        }

        // The column coordinates address exactly the power-of-two column count.
        if point.column().num_variables() != self.layout.column_variables() {
            return Err(JaggedError::ColumnPointWidthMismatch {
                expected: self.layout.column_variables(),
                actual: point.column().num_variables(),
            });
        }

        Ok(())
    }

    /// Materializes the Boolean-cube selector for the dense witness.
    ///
    /// At a live dense index the value is the equality weight of its sparse row and column.
    ///
    /// Every virtual padding cell has weight zero.
    pub(super) fn table<F: Field>(&self, point: &JaggedPoint<F>) -> Vec<F> {
        // Generate only row weights reached by a live cell.
        // A large public row bound must not turn sparse commitment back into rectangular allocation.
        let max_height = self
            .layout
            .cumulative_heights()
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
            .max()
            .unwrap_or(0);
        let row_weights = equality_prefix(point.row().as_slice(), max_height);
        let column_weights = Poly::new_from_point(point.column().as_slice(), F::ONE);
        let mut selector = F::zero_vec(self.layout.dense_capacity());

        // One segment is one column in the column-major dense representation.
        for column in 0..self.layout.num_columns() {
            let start = self.layout.cumulative_heights()[column];
            let end = self.layout.cumulative_heights()[column + 1];
            let column_weight = column_weights.as_slice()[column];

            // The row index restarts at zero at every column boundary.
            for (slot, &row_weight) in selector[start..end].iter_mut().zip(&row_weights) {
                *slot = row_weight * column_weight;
            }
        }

        selector
    }
}

/// Materializes the first equality weights without allocating the full row cube.
fn equality_prefix<F: Field>(point: &[F], length: usize) -> Vec<F> {
    // Each coordinate doubles the lexicographically ordered table.
    let mut weights = Vec::with_capacity(length);
    if length == 0 {
        return weights;
    }
    weights.push(F::ONE);

    for &coordinate in point {
        let previous = weights.len();
        let next = previous.saturating_mul(2).min(length);
        weights.resize(next, F::ZERO);

        // A new least-significant coordinate interleaves its zero and one branches.
        // Descending order preserves each source entry until both children are written.
        for index in (0..next.div_ceil(2)).rev() {
            let weight = weights[index];
            if 2 * index + 1 < next {
                weights[2 * index + 1] = weight * coordinate;
            }
            weights[2 * index] = weight * (F::ONE - coordinate);
        }
    }

    weights
}

impl JaggedSelector<'_> {
    /// Evaluates the selector polynomial at an arbitrary dense point.
    ///
    /// The computation is the width-four read-once branching program from ePrint 2025/917.
    ///
    /// It checks both `dense_index = row + column_start` and `dense_index < column_end`.
    pub(super) fn evaluate<F: Field>(
        &self,
        sparse_point: &JaggedPoint<F>,
        dense_point: &Point<F>,
    ) -> F {
        // The column equality table supplies the coefficient of each boundary pair.
        let column_weights = Poly::new_from_point(sparse_point.column().as_slice(), F::ONE);

        self.layout
            .cumulative_heights()
            .windows(2)
            .zip(column_weights.as_slice())
            .map(|(bounds, &weight)| {
                // Each column asks the same automaton about its own start and end.
                weight * boundary_evaluation(sparse_point.row(), dense_point, bounds[0], bounds[1])
            })
            .sum()
    }
}

/// Evaluates one boundary pair through the branching program's multilinear extension.
fn boundary_evaluation<F: Field>(
    row_point: &Point<F>,
    dense_point: &Point<F>,
    start: usize,
    end: usize,
) -> F {
    // States encode `(carry, less_than)` in the low and high bits.
    // The accepting state has no remaining carry and a proven strict inequality.
    const INITIAL: usize = 0;
    const ACCEPT: usize = 2;

    let mut suffix = [F::ZERO; 4];
    suffix[ACCEPT] = F::ONE;

    // One extra zero layer checks the final carry and the endpoint's overflow bit.
    let top = row_point.num_variables().max(dense_point.num_variables());
    for layer in (0..=top).rev() {
        let coordinates = [
            point_coordinate_from_low(row_point, layer),
            point_coordinate_from_low(dense_point, layer),
            integer_bit(start, layer),
            integer_bit(end, layer),
        ];
        let mut prefix = [F::ZERO; 4];

        // Multilinear extension of one automaton layer.
        for (state, next_weight) in prefix.iter_mut().enumerate() {
            for symbol in 0usize..16 {
                if let Some(next_state) = transition(symbol, state) {
                    *next_weight += cube_weight(&coordinates, symbol) * suffix[next_state];
                }
            }
        }

        suffix = prefix;
    }

    suffix[INITIAL]
}

/// Returns one point coordinate counted from the least-significant variable.
fn point_coordinate_from_low<F: Field>(point: &Point<F>, layer: usize) -> F {
    // Points use big-endian coordinate order throughout the multilinear crates.
    point
        .num_variables()
        .checked_sub(layer + 1)
        .map_or(F::ZERO, |coordinate| point[coordinate])
}

/// Embeds one bit of a public machine integer into the field.
fn integer_bit<F: Field>(value: usize, layer: usize) -> F {
    // Shifts at or above the machine width represent leading zero bits.
    let bit = (layer < usize::BITS as usize) && ((value >> layer) & 1 == 1);
    F::from_bool(bit)
}

/// Evaluates the equality basis weight of one four-bit symbol.
fn cube_weight<F: Field>(point: &[F; 4], symbol: usize) -> F {
    // Each factor selects the coordinate or its Boolean complement.
    point
        .iter()
        .enumerate()
        .map(|(coordinate, &value)| {
            if (symbol >> coordinate) & 1 == 1 {
                value
            } else {
                F::ONE - value
            }
        })
        .product()
}

/// Advances the integer addition and strict-comparison automaton.
const fn transition(symbol: usize, state: usize) -> Option<usize> {
    // Symbol bits are `(row, dense, start, end)` from low to high.
    let row = symbol & 1;
    let dense = (symbol >> 1) & 1;
    let start = (symbol >> 2) & 1;
    let end = (symbol >> 3) & 1;
    let carry = state & 1;
    let less_than = state >> 1;

    // Equality with `row + start` fixes the dense-index bit.
    let sum = row + start + carry;
    if dense != sum & 1 {
        return None;
    }

    // Higher bits overwrite a comparison when they differ.
    // Equal bits preserve the decision made by the lower prefix.
    let next_less_than = if dense == end { less_than } else { end };
    Some((sum >> 1) | (next_less_than << 1))
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    fn field_point(values: &[u32]) -> Point<F> {
        // Small canonical values make failures easy to reproduce.
        Point::new(values.iter().copied().map(F::from_u32).collect())
    }

    #[test]
    fn branching_program_matches_the_materialized_selector() {
        // Fixture state:
        //
        //     rows       2^3
        //     heights    [3, 0, 5, 1]
        //     dense      9 live cells in a 16-cell envelope
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let sparse = JaggedPoint::new(field_point(&[2, 3, 5]), field_point(&[7, 11]));
        let dense = field_point(&[13, 17, 19, 23]);

        let materialized = Poly::new(JaggedSelector::new(&layout).table(&sparse));
        assert_eq!(
            JaggedSelector::new(&layout).evaluate(&sparse, &dense),
            materialized.eval_base(&dense)
        );
    }

    #[test]
    fn equality_prefix_matches_the_full_table() {
        // Every prefix, including the empty one, agrees with the canonical dense constructor.
        let point = field_point(&[2, 3, 5, 7]);
        let full = Poly::new_from_point(point.as_slice(), F::ONE);
        for length in 0..=full.num_evals() {
            assert_eq!(
                equality_prefix(point.as_slice(), length),
                full.as_slice()[..length]
            );
        }
    }

    #[test]
    fn sparse_row_weights_do_not_allocate_the_row_bound() {
        // One live row under a forty-variable bound needs one weight, not a tebibyte table.
        let layout = JaggedLayout::new(40, &[1]).unwrap();
        let row = field_point(&[2; 40]);
        let sparse = JaggedPoint::new(row.clone(), Point::new(vec![]));
        let selector = JaggedSelector::new(&layout).table(&sparse);
        let expected = row
            .iter()
            .fold(F::ONE, |weight, &value| weight * (F::ONE - value));

        assert_eq!(selector, vec![expected]);
    }

    #[test]
    fn the_selector_identity_holds_in_characteristic_two() {
        // A binary-field point exercises subtraction as addition.
        // The automaton must remain an algebraic MLE rather than use prime-field signs.
        let layout = JaggedLayout::new(2, &[1, 3]).unwrap();
        let sparse = JaggedPoint::new(
            Point::new(vec![
                BinaryField128::from_u64(5),
                BinaryField128::from_u64(9),
            ]),
            Point::new(vec![BinaryField128::from_u64(7)]),
        );
        let dense = Point::new(vec![
            BinaryField128::from_u64(11),
            BinaryField128::from_u64(13),
        ]);

        let materialized = Poly::new(JaggedSelector::new(&layout).table(&sparse));
        assert_eq!(
            JaggedSelector::new(&layout).evaluate(&sparse, &dense),
            materialized.eval_base(&dense)
        );
    }

    proptest! {
        #[test]
        fn arbitrary_points_match_direct_multilinear_evaluation(
            heights in prop::collection::vec(0usize..=8, 4),
            row in prop::collection::vec(any::<u32>(), 3),
            column in prop::collection::vec(any::<u32>(), 2),
            dense in prop::collection::vec(any::<u32>(), 5),
        ) {
            // Four columns of at most eight rows satisfy the shared row bound.
            let layout = JaggedLayout::new(3, &heights).unwrap();
            let sparse = JaggedPoint::new(field_point(&row), field_point(&column));

            // The dense point is truncated to the arity derived from this random area.
            let dense = field_point(&dense[..layout.dense_variables()]);
            let materialized = Poly::new(JaggedSelector::new(&layout).table(&sparse));

            prop_assert_eq!(
                JaggedSelector::new(&layout).evaluate(&sparse, &dense),
                materialized.eval_base(&dense)
            );
        }
    }
}
