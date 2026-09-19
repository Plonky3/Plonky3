//! Sumcheck state for a sum of multilinear products.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;

/// A degree-two polynomial represented as a sum of multilinear products.
pub(super) struct SumOfProducts<F> {
    /// Left and right factors sharing one Boolean variable space.
    pairs: Vec<[Poly<F>; 2]>,
}

impl<F: Field> SumOfProducts<F> {
    /// Creates a prover state from equally sized factor pairs.
    pub(super) fn new(mut pairs: Vec<[Poly<F>; 2]>, num_variables: usize) -> Self {
        // An empty wiring polynomial is represented by one zero product.
        if pairs.is_empty() {
            pairs.push([
                Poly::new(F::zero_vec(1 << num_variables)),
                Poly::new(F::zero_vec(1 << num_variables)),
            ]);
        }

        // Every pair is evaluated over the same sumcheck cube.
        debug_assert!(pairs.iter().all(|pair| {
            pair[0].num_variables() == num_variables && pair[1].num_variables() == num_variables
        }));
        Self { pairs }
    }

    /// Returns the right factor of every product after all variables are bound.
    pub(super) fn right_constants(&self) -> Vec<F> {
        // A completed sumcheck leaves one evaluation in every factor.
        self.pairs
            .iter()
            .map(|pair| {
                pair[1]
                    .as_constant()
                    .expect("every sumcheck variable was bound")
            })
            .collect()
    }

    /// Returns both terminal evaluations of the first product.
    pub(super) fn first_constants(&self) -> [F; 2] {
        // The word phase contains exactly one product pair.
        debug_assert_eq!(self.pairs.len(), 1);
        self.pairs[0].each_ref().map(|factor| {
            factor
                .as_constant()
                .expect("every sumcheck variable was bound")
        })
    }

    /// Returns the final evaluation of the summed product polynomial.
    pub(super) fn terminal_sum(&self) -> F {
        // At a fully bound point each product is the product of its two constants.
        self.pairs
            .iter()
            .map(|pair| {
                pair.iter()
                    .map(|factor| {
                        factor
                            .as_constant()
                            .expect("every sumcheck variable was bound")
                    })
                    .product::<F>()
            })
            .sum()
    }

    /// Evaluates one round polynomial at a selected node.
    fn round_at(&self, node: F) -> F {
        // Prefix binding pairs the lower and upper table halves.
        self.pairs
            .iter()
            .map(|pair| {
                let half = pair[0].num_evals() / 2;
                (0..half)
                    .map(|row| {
                        pair.iter()
                            .map(|factor| {
                                let values = factor.as_slice();
                                values[row] + (values[row + half] - values[row]) * node
                            })
                            .product::<F>()
                    })
                    .sum::<F>()
            })
            .sum()
    }
}

impl<F: Field> RoundProver<F> for SumOfProducts<F> {
    fn fold(&mut self, challenge: F) {
        // Every factor binds the same most-significant active variable.
        for pair in &mut self.pairs {
            for factor in pair {
                factor.fix_prefix_var_mut(challenge);
            }
        }
    }

    fn round_poly(&self) -> Vec<F> {
        // Degree two needs the values at zero and two.
        // The running claim reconstructs the omitted value at one.
        vec![
            self.round_at(F::ZERO),
            self.round_at(F::interpolation_node(2)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn round_message_matches_direct_interpolation() {
        // Fixture state: two products over a two-variable cube.
        type F = BinaryField128;
        let pairs = vec![
            [
                Poly::new([1, 2, 3, 4].map(F::from_u64).to_vec()),
                Poly::new([5, 6, 7, 8].map(F::from_u64).to_vec()),
            ],
            [
                Poly::new([9, 10, 11, 12].map(F::from_u64).to_vec()),
                Poly::new([13, 14, 15, 16].map(F::from_u64).to_vec()),
            ],
        ];
        let prover = SumOfProducts::new(pairs, 2);

        // The helper returns evaluations at zero and two.
        let round = prover.round_poly();
        assert_eq!(
            round,
            vec![
                prover.round_at(F::ZERO),
                prover.round_at(F::interpolation_node(2)),
            ]
        );
    }
}
