//! Batched vanishing check over the shared constraint and within-word cube.
//!
//! Both local families hold on one constraint row and one bit.
//!
//! ```text
//!     linear    value(row, bit) = 0
//!     bitwise   left(row, bit) * right(row, bit) = output(row, bit)
//! ```
//!
//! A statement with products adds five more terms under further powers of the same coefficient.
//!
//! ```text
//!     low bit   a(row, 0) * b(row, 0) = lo(row, 0)
//!     claims    a, b, lo, hi at the points the multiplication reduction ended on
//! ```
//!
//! The low bit closes the one collision the exponent lift leaves open.
//!
//! The four claims share this sumcheck, so every operand claim ends at one point.
//!
//! Padding rows hold the empty operand, which satisfies every shape.
//!
//! That shared point is the one the shift reduction consumes.

use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_field::Field;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;

/// Per-variable degree of the batched relation polynomial.
///
/// One equality factor multiplies a product of two operand columns.
pub(super) const ZEROCHECK_DEGREE: usize = 3;

/// Terms batched when the statement declares no product: linear, then bitwise.
pub(super) const BASE_TERMS: usize = 2;

/// Terms batched when it does: the two above, the low bit, and four operand claims.
pub(super) const PRODUCT_TERMS: usize = 7;

/// Operand evaluations the vanishing check leaves for the shift reduction.
///
/// The vanishing operand comes first, then the bitwise left, right, and output operands.
///
/// The left factor, right factor, low limb, and high limb follow.
pub(super) const OPERAND_EVALUATIONS: usize = 8;

/// Equality weights of the four product terms, in low-bit, left, right, then limb order.
pub(super) type ProductWeights<F> = [F; 4];

/// Bound product columns beside their bound term weights.
type ProductPolys<F> = ([Poly<F>; 4], ProductWeights<Poly<F>>);

/// Product columns and their term weights over the shared cube.
pub(super) struct ProductTables<F> {
    /// Left factor, right factor, low limb, then high limb bit columns.
    pub(super) columns: [Vec<F>; 4],
    /// Equality tables of the four points the product terms are weighted at.
    pub(super) weights: ProductWeights<Vec<F>>,
}

/// Prover state for the batched relation vanishing check.
pub(super) struct RelationZerocheck<F> {
    /// Equality weights of the sampled vanishing point.
    equality: Poly<F>,
    /// The operand every linear relation requires to vanish.
    linear: Poly<F>,
    /// The left input, right input, and output of every bitwise product.
    bitwise: [Poly<F>; 3],
    /// Product columns, then their weights, when the statement declares products.
    product: Option<ProductPolys<F>>,
    /// Powers of the coefficient separating every term, starting at one.
    powers: [F; PRODUCT_TERMS],
}

impl<F: Field> RelationZerocheck<F> {
    /// Creates a prover state over one shared Boolean cube.
    pub(super) fn new(
        equality: Vec<F>,
        linear: Vec<F>,
        bitwise: [Vec<F>; 3],
        product: Option<ProductTables<F>>,
        batching: F,
    ) -> Self {
        // Every column spans the padded constraint rows times the word width.
        debug_assert!(bitwise.iter().all(|column| column.len() == equality.len()));
        debug_assert_eq!(linear.len(), equality.len());
        Self {
            equality: Poly::new(equality),
            linear: Poly::new(linear),
            bitwise: bitwise.map(Poly::new),
            product: product
                .map(|tables| (tables.columns.map(Poly::new), tables.weights.map(Poly::new))),
            powers: batching_powers(batching),
        }
    }

    /// Returns the operand evaluations left once every variable is bound.
    pub(super) fn terminal_operands(&self) -> [F; OPERAND_EVALUATIONS] {
        let constant = |poly: &Poly<F>| {
            poly.as_constant()
                .expect("every zerocheck variable was bound")
        };
        let mut operands = [F::ZERO; OPERAND_EVALUATIONS];
        operands[0] = constant(&self.linear);
        for (slot, column) in operands[1..4].iter_mut().zip(&self.bitwise) {
            *slot = constant(column);
        }
        if let Some((columns, _)) = &self.product {
            for (slot, column) in operands[4..].iter_mut().zip(columns) {
                *slot = constant(column);
            }
        }
        operands
    }

    /// Evaluates the round polynomial at one interpolation node.
    fn round_at(&self, node: F) -> F {
        // Prefix binding pairs the lower and upper half of every table.
        let half = self.equality.num_evals() / 2;
        let interpolate = |poly: &Poly<F>, row: usize| {
            let values = poly.as_slice();
            values[row] + (values[row + half] - values[row]) * node
        };
        (0..half)
            .map(|row| {
                let mut operands = [F::ZERO; OPERAND_EVALUATIONS];
                operands[0] = interpolate(&self.linear, row);
                for (slot, column) in operands[1..4].iter_mut().zip(&self.bitwise) {
                    *slot = interpolate(column, row);
                }
                let mut weights = [F::ZERO; 4];
                if let Some((columns, tables)) = &self.product {
                    for (slot, column) in operands[4..].iter_mut().zip(columns) {
                        *slot = interpolate(column, row);
                    }
                    weights = array::from_fn(|term| interpolate(&tables[term], row));
                }
                batched(
                    interpolate(&self.equality, row),
                    &weights,
                    &operands,
                    &self.powers,
                )
            })
            .sum()
    }
}

impl<F: Field> RoundProver<F> for RelationZerocheck<F> {
    fn fold(&mut self, challenge: F) {
        // Every column binds the same most-significant active variable.
        self.equality.fix_prefix_var_mut(challenge);
        self.linear.fix_prefix_var_mut(challenge);
        for column in &mut self.bitwise {
            column.fix_prefix_var_mut(challenge);
        }
        if let Some((columns, weights)) = &mut self.product {
            for column in columns.iter_mut().chain(weights.iter_mut()) {
                column.fix_prefix_var_mut(challenge);
            }
        }
    }

    fn round_poly(&self) -> Vec<F> {
        // The omitted value at one is reconstructed from the running claim.
        vec![
            self.round_at(F::ZERO),
            self.round_at(F::interpolation_node(2)),
            self.round_at(F::interpolation_node(3)),
        ]
    }
}

/// Returns `1, beta, beta^2, ..., beta^6`.
pub(super) fn batching_powers<F: Field>(batching: F) -> [F; PRODUCT_TERMS] {
    let mut powers = [F::ONE; PRODUCT_TERMS];
    for index in 1..PRODUCT_TERMS {
        powers[index] = powers[index - 1] * batching;
    }
    powers
}

/// Evaluates the batched relation polynomial at one point of the cube or its extension.
///
/// ```text
///     eq_zeta * (linear + beta * (left * right - output))
///       + beta^2 * eq_low   * (a * b - lo)
///       + beta^3 * eq_left  * a
///       + beta^4 * eq_right * b
///       + eq_limb * (beta^5 * lo + beta^6 * hi)
/// ```
fn batched<F: Field>(
    equality: F,
    weights: &ProductWeights<F>,
    operands: &[F; OPERAND_EVALUATIONS],
    powers: &[F; PRODUCT_TERMS],
) -> F {
    let [linear, left, right, output, a, b, low, high] = *operands;
    let [low_bit, left_claim, right_claim, limb_claim] = *weights;
    equality * (linear + powers[1] * (left * right - output))
        + powers[2] * low_bit * (a * b - low)
        + powers[3] * left_claim * a
        + powers[4] * right_claim * b
        + limb_claim * (powers[5] * low + powers[6] * high)
}

/// Returns the sum the batched check must reach, fixed by the multiplication claims.
///
/// The local terms vanish on the cube, and each claim term sums to its claimed value.
pub(super) fn claimed_sum<F: Field>(claims: &[F; 4], batching: F) -> F {
    let powers = batching_powers(batching);
    powers[3] * claims[0] + powers[4] * claims[1] + powers[5] * claims[2] + powers[6] * claims[3]
}

/// Closes the vanishing check against the operand evaluations the prover supplied.
pub(super) fn closing_value<F: Field>(
    equality: F,
    weights: &ProductWeights<F>,
    operands: &[F; OPERAND_EVALUATIONS],
    batching: F,
) -> F {
    // The verifier evaluates every equality weight itself, so only the operands are claimed.
    batched(equality, weights, operands, &batching_powers(batching))
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryChallenger, BinaryField128, TowerLevel};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::point::Point;

    use super::*;

    type F = BinaryField128;

    fn challenger() -> BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>> {
        BinaryChallenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    // A column whose entries are distinct so no reordering can survive unnoticed.
    fn column(seed: u128, len: usize) -> Vec<F> {
        (0..len)
            .map(|row| F::from_repr(seed + row as u128))
            .collect()
    }

    #[test]
    fn round_message_matches_direct_interpolation() {
        // Fixture state: a four-row cube with four distinct operand columns.
        let equality = column(1, 4);
        let linear = column(100, 4);
        let bitwise = [column(200, 4), column(300, 4), column(400, 4)];
        let prover = RelationZerocheck::new(equality, linear, bitwise, None, F::from_repr(7));

        // The helper transmits the values at the nodes zero, two, and three.
        assert_eq!(
            prover.round_poly(),
            vec![
                prover.round_at(F::ZERO),
                prover.round_at(F::interpolation_node(2)),
                prover.round_at(F::interpolation_node(3)),
            ]
        );
    }

    #[test]
    fn the_bound_equality_column_matches_the_verifier_formula() {
        // Fixture state: a two-variable vanishing point and its expanded weights.
        let point = [F::from_repr(11), F::from_repr(13)];
        let weights = Point::new(point.as_slice()).equality_weights_msb();
        let zeros = F::zero_vec(4);
        let mut prover = RelationZerocheck::new(
            weights,
            zeros.clone(),
            [zeros.clone(), zeros.clone(), zeros],
            None,
            F::ONE,
        );

        // Binding both variables must reproduce the closed-form equality evaluation.
        let challenges = [F::from_repr(17), F::from_repr(19)];
        for challenge in challenges {
            prover.fold(challenge);
        }
        assert_eq!(
            prover.equality.as_constant().unwrap(),
            Point::eval_eq(&point, &challenges)
        );
    }

    #[test]
    fn a_product_zerocheck_closes_on_the_verifier_formula() {
        // Fixture state: two-variable cube, distinct columns, four distinct weight points.
        let batching = F::from_repr(5);
        let point = |seed: u128| [F::from_repr(seed), F::from_repr(seed + 1)];
        let vanishing = point(11);
        let product_points = [point(21), point(31), point(41), point(51)];
        let equality = Point::new(vanishing.as_slice()).equality_weights_msb();
        let linear = column(100, 4);
        let bitwise = [column(200, 4), column(300, 4), column(400, 4)];
        let columns = [
            column(500, 4),
            column(600, 4),
            column(700, 4),
            column(800, 4),
        ];
        let weights =
            product_points.map(|point| Point::new(point.as_slice()).equality_weights_msb());

        // The claimed sum is computed cell by cell from the batched definition.
        let powers = batching_powers(batching);
        let sum = (0..4)
            .map(|cell| {
                let operands = [
                    linear[cell],
                    bitwise[0][cell],
                    bitwise[1][cell],
                    bitwise[2][cell],
                    columns[0][cell],
                    columns[1][cell],
                    columns[2][cell],
                    columns[3][cell],
                ];
                let cell_weights = weights.each_ref().map(|table| table[cell]);
                batched(equality[cell], &cell_weights, &operands, &powers)
            })
            .sum::<F>();

        let mut prover = RelationZerocheck::new(
            equality,
            linear,
            bitwise,
            Some(ProductTables { columns, weights }),
            batching,
        );
        let mut sponge = challenger();
        let (proof, _) = prover.prove::<F, _>(&mut sponge, 2, ZEROCHECK_DEGREE, 0, sum);
        let operands = prover.terminal_operands();

        // The verifier evaluates every weight itself and closes on the prover's operands.
        let (end, final_claim) = proof
            .verify(&mut challenger(), 2, ZEROCHECK_DEGREE, 0)
            .unwrap();
        let equality_at = Point::eval_eq(&vanishing, end.as_slice());
        let weights_at = product_points.map(|point| Point::eval_eq(&point, end.as_slice()));
        assert_eq!(
            final_claim,
            closing_value(equality_at, &weights_at, &operands, batching)
        );
    }
}
