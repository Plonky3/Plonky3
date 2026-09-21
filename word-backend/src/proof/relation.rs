//! Batched vanishing check over the shared constraint and within-word cube.
//!
//! Both proved families are local to one constraint row and one bit.
//!
//! ```text
//!     linear    value(row, bit) = 0
//!     bitwise   left(row, bit) * right(row, bit) = output(row, bit)
//! ```
//!
//! One check therefore covers them, under one coefficient separating the two.
//!
//! Padding rows hold the empty operand, which satisfies both shapes.
//!
//! The check ends at one point shared by every operand claim the shift reduction consumes.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;

/// Per-variable degree of the batched relation polynomial.
///
/// One equality factor multiplies a product of two operand columns.
pub(super) const ZEROCHECK_DEGREE: usize = 3;

/// Relation families combined under one batching coefficient.
pub(super) const BATCHED_FAMILIES: usize = 2;

/// Operand evaluations the vanishing check leaves for the shift reduction.
///
/// The order is the vanishing operand, then the left, right, and output operands.
pub(super) const OPERAND_EVALUATIONS: usize = 4;

/// Prover state for the batched relation vanishing check.
pub(super) struct RelationZerocheck<F> {
    /// Equality weights of the sampled vanishing point.
    equality: Poly<F>,
    /// The operand every linear relation requires to vanish.
    linear: Poly<F>,
    /// The left input, right input, and output of every bitwise product.
    bitwise: [Poly<F>; 3],
    /// Coefficient separating the linear family from the bitwise one.
    batching: F,
}

impl<F: Field> RelationZerocheck<F> {
    /// Creates a prover state over one shared Boolean cube.
    pub(super) fn new(equality: Vec<F>, linear: Vec<F>, bitwise: [Vec<F>; 3], batching: F) -> Self {
        // Every column spans the padded constraint rows times the word width.
        debug_assert!(bitwise.iter().all(|column| column.len() == equality.len()));
        debug_assert_eq!(linear.len(), equality.len());
        Self {
            equality: Poly::new(equality),
            linear: Poly::new(linear),
            bitwise: bitwise.map(Poly::new),
            batching,
        }
    }

    /// Returns the operand evaluations left once every variable is bound.
    pub(super) fn terminal_operands(&self) -> [F; OPERAND_EVALUATIONS] {
        let constant = |poly: &Poly<F>| {
            poly.as_constant()
                .expect("every zerocheck variable was bound")
        };
        [
            constant(&self.linear),
            constant(&self.bitwise[0]),
            constant(&self.bitwise[1]),
            constant(&self.bitwise[2]),
        ]
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
                let linear = interpolate(&self.linear, row);
                let left = interpolate(&self.bitwise[0], row);
                let right = interpolate(&self.bitwise[1], row);
                let output = interpolate(&self.bitwise[2], row);
                let combined = linear + self.batching * (left * right - output);
                interpolate(&self.equality, row) * combined
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

/// Closes the vanishing check against the operand evaluations the prover supplied.
pub(super) fn closing_value<F: Field>(
    equality: F,
    operands: &[F; OPERAND_EVALUATIONS],
    batching: F,
) -> F {
    // The bitwise family contributes the product of its inputs against its output.
    let [linear, left, right, output] = *operands;
    equality * (linear + batching * (left * right - output))
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;

    use super::*;

    type F = BinaryField128;

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
        let prover = RelationZerocheck::new(equality, linear, bitwise, F::from_repr(7));

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
        let weights = crate::shift::transcript::equality_weights(&point);
        let zeros = F::zero_vec(4);
        let mut prover = RelationZerocheck::new(
            weights,
            zeros.clone(),
            [zeros.clone(), zeros.clone(), zeros],
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
}
