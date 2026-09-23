//! Sparse-to-dense selector polynomial from the basic jagged construction.

use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_ceil_usize;

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

/// Materializes the first equality weights without visiting the full row cube.
fn equality_prefix<F: Field>(point: &[F], length: usize) -> Vec<F> {
    if length == 0 {
        return Vec::new();
    }

    // Every index below the requested length leaves the leading coordinates on their zero branch.
    // Those coordinates contribute one shared scalar instead of a pass over the whole table.
    let expanded = point.len().min(log2_ceil_usize(length));
    let (leading, trailing) = point.split_at(point.len() - expanded);
    let scale = leading
        .iter()
        .fold(F::ONE, |weight, &value| weight * (F::ONE - value));

    // The remaining coordinates span at most twice the requested length.
    let mut weights = Poly::new_from_point(trailing, scale).into_evals();
    weights.truncate(length);
    weights
}

impl JaggedSelector<'_> {
    /// Evaluates the selector polynomial at an arbitrary dense point.
    ///
    /// The computation is the width-four read-once branching program from ePrint 2025/917.
    ///
    /// It checks both `dense_index = row + column_start` and `dense_index < column_end`.
    ///
    /// The automaton reads one layer per variable, least-significant first:
    ///
    /// ```text
    ///   layer:      0      1      2     ...    top
    ///   row  y:    y_0    y_1    y_2          zero above the row bound
    ///   dense z:   z_0    z_1    z_2          zero above the dense arity
    ///   start  :   s_0    s_1    s_2          public bit
    ///   end    :   e_0    e_1    e_2          public bit
    ///
    ///   carry:     z_k must equal (y_k + s_k + carry) & 1,  carry' = sum >> 1
    ///   less :     less' = if z_k == e_k { less } else { e_k }
    ///
    ///   INITIAL = (carry 0, less 0)          ACCEPT = (carry 0, less 1)
    /// ```
    ///
    /// Accepting only on a set comparison bit gives a strict inequality, and the lower bound is free because the row index is non-negative.
    ///
    /// Reading one layer above the wider point leaves both points at zero there, which absorbs the overflow bit of an endpoint at the dense capacity and forces the last carry to vanish.
    ///
    /// The two conditions together say exactly that the row index is below the column height.
    ///
    /// Above the dense arity both endpoints vanish, every surviving transition keeps its state, and the block collapses to one shared scalar.
    ///
    /// The cost is therefore the dense arity per column plus one pass over the row bound.
    pub(super) fn evaluate<F: Field>(
        &self,
        sparse_point: &JaggedPoint<F>,
        dense_point: &Point<F>,
    ) -> F {
        let row_point = sparse_point.row();

        // One extra zero layer checks the final carry and the endpoint's overflow bit.
        let top = row_point.num_variables().max(dense_point.num_variables());

        // Every boundary lies inside the envelope, so no public bit above this layer is set.
        let split = self
            .layout
            .dense_variables()
            .max(dense_point.num_variables())
            .min(top);

        // The four products of a row and a dense coordinate depend only on the layer, so every column shares them.
        let layer_weights = (0..=split)
            .map(|layer| {
                let row = point_coordinate_from_low(row_point, layer);
                let dense = point_coordinate_from_low(dense_point, layer);
                [
                    (F::ONE - row) * (F::ONE - dense),
                    row * (F::ONE - dense),
                    (F::ONE - row) * dense,
                    row * dense,
                ]
            })
            .collect::<Vec<_>>();

        // Reaching the accepting state through the collapsed block asks every leading row bit to vanish.
        let leading = (split + 1..=top)
            .map(|layer| F::ONE - point_coordinate_from_low(row_point, layer))
            .product::<F>();

        // The column equality table supplies the coefficient of each boundary pair.
        let column_weights = Poly::new_from_point(sparse_point.column().as_slice(), F::ONE);

        self.layout
            .cumulative_heights()
            .windows(2)
            .zip(column_weights.as_slice())
            .map(|(bounds, &weight)| {
                // Each column asks the same automaton about its own start and end.
                weight * boundary_evaluation(&layer_weights, leading, bounds[0], bounds[1])
            })
            .sum()
    }
}

/// Evaluates one boundary pair through the branching program's multilinear extension.
fn boundary_evaluation<F: Field>(
    layer_weights: &[[F; 4]],
    leading: F,
    start: usize,
    end: usize,
) -> F {
    // States encode `(carry, less_than)` in the low and high bits.
    // The accepting state has no remaining carry and a proven strict inequality.
    const INITIAL: usize = 0;
    const ACCEPT: usize = 2;

    let mut suffix = [F::ZERO; 4];
    suffix[ACCEPT] = leading;

    for (layer, weights) in layer_weights.iter().enumerate().rev() {
        // Both endpoints are public, so twelve of the sixteen symbols have an identically zero factor.
        let public = (integer_bit(start, layer) << 2) | (integer_bit(end, layer) << 3);
        let mut prefix = [F::ZERO; 4];

        // Multilinear extension of one automaton layer.
        for (state, next_weight) in prefix.iter_mut().enumerate() {
            for (secret, &weight) in weights.iter().enumerate() {
                if let Some(next_state) = transition(public | secret, state) {
                    *next_weight += weight * suffix[next_state];
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

/// Extracts one bit of a public machine integer.
const fn integer_bit(value: usize, layer: usize) -> usize {
    // Shifts at or above the machine width represent leading zero bits.
    if layer < usize::BITS as usize {
        (value >> layer) & 1
    } else {
        0
    }
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
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    // Reference automaton: the unpruned layer evaluation, enumerating all sixteen symbols.
    // The shipped loop must agree with it while skipping the symbols whose factor is zero.
    fn reference_cube_weight(coordinates: &[F; 4], symbol: usize) -> F {
        coordinates
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if (symbol >> index) & 1 == 1 {
                    value
                } else {
                    F::ONE - value
                }
            })
            .product()
    }

    fn reference_boundary(row: &Point<F>, dense: &Point<F>, start: usize, end: usize) -> F {
        // Index two is the accepting state and index zero the initial one.
        let mut suffix = [F::ZERO; 4];
        suffix[2] = F::ONE;
        let top = row.num_variables().max(dense.num_variables());
        for layer in (0..=top).rev() {
            let bit = |value: usize| {
                F::from_bool((layer < usize::BITS as usize) && (value >> layer) & 1 == 1)
            };
            let coordinates = [
                point_coordinate_from_low(row, layer),
                point_coordinate_from_low(dense, layer),
                bit(start),
                bit(end),
            ];
            let mut prefix = [F::ZERO; 4];
            for (state, next_weight) in prefix.iter_mut().enumerate() {
                for symbol in 0usize..16 {
                    if let Some(next_state) = transition(symbol, state) {
                        *next_weight +=
                            reference_cube_weight(&coordinates, symbol) * suffix[next_state];
                    }
                }
            }
            suffix = prefix;
        }
        suffix[0]
    }

    fn reference_evaluate(layout: &JaggedLayout, sparse: &JaggedPoint<F>, dense: &Point<F>) -> F {
        let column_weights = Poly::new_from_point(sparse.column().as_slice(), F::ONE);
        layout
            .cumulative_heights()
            .windows(2)
            .zip(column_weights.as_slice())
            .map(|(bounds, &weight)| {
                weight * reference_boundary(sparse.row(), dense, bounds[0], bounds[1])
            })
            .sum()
    }

    // Equality weight of one Boolean index, written straight from the product definition.
    // Coordinates are most-significant first, so the trailing coordinate owns the low index bit.
    fn definitional_weight(point: &[F], index: usize) -> F {
        point
            .iter()
            .rev()
            .enumerate()
            .map(|(bit, &value)| {
                if (index >> bit) & 1 == 1 {
                    value
                } else {
                    F::ONE - value
                }
            })
            .product()
    }

    fn field_point(values: &[u32]) -> Point<F> {
        // Small canonical values make failures easy to reproduce.
        Point::new(values.iter().copied().map(F::from_u32).collect())
    }

    fn binary_point(values: &[u128]) -> Point<BinaryField128> {
        // The raw tower representation is the only constructor that escapes the prime subfield.
        // An integer constructor would reduce every coordinate modulo two.
        let point = Point::new(
            values
                .iter()
                .copied()
                .map(BinaryField128::from_repr)
                .collect(),
        );
        assert!(
            point
                .iter()
                .all(|c| *c != BinaryField128::ZERO && *c != BinaryField128::ONE)
        );
        point
    }

    // Multilinear extension of the paper's own function, summed straight over its Boolean domain.
    // It reads as "the second part equals the first plus the third, and is below the fourth".
    //
    // No automaton, carry or comparison register appears, so it shares no model with the shipped code.
    fn definitional_g(row: &Point<F>, dense: &Point<F>, start: usize, end: usize) -> F {
        let mut total = F::ZERO;
        for a in 0..(1usize << row.num_variables()) {
            for b in 0..(1usize << dense.num_variables()) {
                if b < end && b == a + start {
                    total += definitional_weight(row.as_slice(), a)
                        * definitional_weight(dense.as_slice(), b);
                }
            }
        }
        total
    }

    #[test]
    fn branching_program_matches_the_materialized_selector() {
        // Fixture state:
        //
        //     Rows       2^3
        //     Heights    [3, 0, 5, 1]
        //     Dense      9 live cells in a 16-cell envelope
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

        // A forty-variable point never materializes its cube, so the definition is the only reference.
        // Lengths straddling a power of two catch an off-by-one in the count of expanded coordinates.
        let wide = field_point(&[3; 40]);
        for length in [1, 2, 3, 8, 17] {
            let expected = (0..length)
                .map(|index| definitional_weight(wide.as_slice(), index))
                .collect::<Vec<_>>();
            assert_eq!(equality_prefix(wide.as_slice(), length), expected);
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
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let sparse = JaggedPoint::new(
            binary_point(&[
                0x2545_f491_4f6c_dd1d_853c_49e6_748f_ea9b,
                0x0139_408d_cbbf_7a44_0ca4_5b0f_286a_dd56,
                0x1234_5678_9abc_def0_0fed_cba9_8765_4321,
            ]),
            binary_point(&[
                0x8000_0000_0000_0000_0000_0000_0000_0001,
                0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
            ]),
        );
        let dense = binary_point(&[
            0x9e37_79b9_7f4a_7c15_f39c_c060_5ceb_c860,
            0xa076_1d64_78bd_642f_e703_7ed1_a0b4_28db,
            0x8ebc_6af0_9c88_c6e3_5899_65cc_7537_4cc3,
            0xc2b2_ae3d_27d4_eb4f_1656_67b1_9c3f_2ee7,
        ]);

        // Both sides would agree vacuously at a point where the selector vanishes.
        let selector = JaggedSelector::new(&layout);
        let evaluation = selector.evaluate(&sparse, &dense);
        assert_ne!(evaluation, BinaryField128::ZERO);
        assert_eq!(
            evaluation,
            Poly::new(selector.table(&sparse)).eval_base(&dense)
        );
    }

    #[test]
    fn the_automaton_computes_the_function_the_paper_defines() {
        // Fixture state: one column spanning rows 3 to 8, read under two row bounds.
        //
        // ```text
        //     start   3
        //     end     9
        // ```
        //
        // The second bound is above the dense arity, which the paper assumes away.
        let dense = field_point(&[7, 11, 13, 17]);
        let narrow = field_point(&[2, 3, 5]);
        let wide = field_point(&[2, 3, 5, 19, 23, 29]);

        for row in [&narrow, &wide] {
            for (start, end) in [(0usize, 1usize), (3, 9), (0, 16), (5, 5), (9, 16)] {
                let top = row.num_variables().max(dense.num_variables());
                let split = dense.num_variables();
                let layers = (0..=split)
                    .map(|layer| {
                        let r = point_coordinate_from_low(row, layer);
                        let z = point_coordinate_from_low(&dense, layer);
                        [
                            (F::ONE - r) * (F::ONE - z),
                            r * (F::ONE - z),
                            (F::ONE - r) * z,
                            r * z,
                        ]
                    })
                    .collect::<Vec<_>>();

                // Reaching the accepting state above the dense arity asks every leading row bit to vanish.
                let leading = (split + 1..=top)
                    .map(|layer| F::ONE - point_coordinate_from_low(row, layer))
                    .product::<F>();

                assert_eq!(
                    boundary_evaluation(&layers, leading, start, end),
                    definitional_g(row, &dense, start, end),
                    "boundary pair {start}..{end}"
                );
            }
        }
    }

    #[test]
    fn collapsing_the_leading_row_layers_changes_no_value() {
        // A forty-variable row bound over sixteen live cells is the shape the collapse exists for.
        // The reference below walks every layer of every column, which is what the collapse replaces.
        let layout = JaggedLayout::new(40, &[7, 0, 6, 3]).unwrap();
        let sparse = JaggedPoint::new(
            field_point(&(0..40).map(|i| 2 + 3 * i).collect::<Vec<_>>()),
            field_point(&[7, 11]),
        );
        let dense = field_point(&[13, 17, 19, 23]);

        let evaluation = JaggedSelector::new(&layout).evaluate(&sparse, &dense);
        assert_ne!(evaluation, F::ZERO);
        assert_eq!(evaluation, reference_evaluate(&layout, &sparse, &dense));
    }

    #[test]
    fn the_selector_is_the_column_sum_the_paper_writes_down() {
        // Equation five of the paper weights one boundary pair per column by the column equality table.
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let sparse = JaggedPoint::new(field_point(&[2, 3, 5]), field_point(&[7, 11]));
        let dense = field_point(&[13, 17, 19, 23]);

        let expected = layout
            .cumulative_heights()
            .windows(2)
            .enumerate()
            .map(|(column, bounds)| {
                definitional_weight(sparse.column().as_slice(), column)
                    * definitional_g(sparse.row(), &dense, bounds[0], bounds[1])
            })
            .sum::<F>();

        assert_ne!(expected, F::ZERO);
        assert_eq!(
            JaggedSelector::new(&layout).evaluate(&sparse, &dense),
            expected
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

        #[test]
        fn pruning_the_zero_symbols_changes_no_value(
            row_variables in 3usize..=9,
            heights in prop::collection::vec(0usize..=8, 4),
            row in prop::collection::vec(any::<u32>(), 9),
            column in prop::collection::vec(any::<u32>(), 2),
            dense in prop::collection::vec(any::<u32>(), 5),
        ) {
            // A row bound above the dense arity is the shape the shipped loop hoists the most work out of.
            let layout = JaggedLayout::new(row_variables, &heights).unwrap();
            let sparse = JaggedPoint::new(field_point(&row[..row_variables]), field_point(&column));
            let dense = field_point(&dense[..layout.dense_variables()]);

            prop_assert_eq!(
                JaggedSelector::new(&layout).evaluate(&sparse, &dense),
                reference_evaluate(&layout, &sparse, &dense)
            );
        }
    }
}
