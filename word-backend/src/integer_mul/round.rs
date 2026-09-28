//! One cubic sumcheck prover over a fixed composition of equally sized tables.

use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_field::Field;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::RoundProver;

/// Per-variable degree of every sumcheck the multiplication reduction runs.
///
/// Each composition is an equality factor times at most two tables varying in any one variable.
pub(super) const DEGREE: usize = 3;

/// Prover state for `sum_z combine(tables(z))` over one Boolean cube.
pub(super) struct Composite<F, const N: usize> {
    /// Tables bound in lockstep, most significant variable first.
    tables: [Poly<F>; N],
    /// Pointwise composition, cubic in every variable.
    combine: fn(&[F; N]) -> F,
}

impl<F: Field, const N: usize> Composite<F, N> {
    /// Creates a prover state over tables of one shared length.
    pub(super) fn new(tables: [Vec<F>; N], combine: fn(&[F; N]) -> F) -> Self {
        // Every table spans the same cube.
        debug_assert!(tables.iter().all(|table| table.len() == tables[0].len()));
        Self {
            tables: tables.map(Poly::new),
            combine,
        }
    }

    /// Returns every table's value once all variables are bound.
    pub(super) fn terminal(&self) -> [F; N] {
        array::from_fn(|index| {
            self.tables[index]
                .as_constant()
                .expect("every sumcheck variable was bound")
        })
    }

    /// Evaluates the round polynomial at one interpolation node.
    fn round_at(&self, node: F) -> F {
        // Prefix binding pairs the lower and upper half of every table.
        let half = self.tables[0].num_evals() / 2;
        (0..half)
            .map(|row| {
                let values = array::from_fn(|index| {
                    let table = self.tables[index].as_slice();
                    table[row] + (table[row + half] - table[row]) * node
                });
                (self.combine)(&values)
            })
            .sum()
    }
}

impl<F: Field, const N: usize> RoundProver<F> for Composite<F, N> {
    fn fold(&mut self, challenge: F) {
        // Every table binds the same most significant active variable.
        for table in &mut self.tables {
            table.fix_prefix_var_mut(challenge);
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

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::integer_mul::exponent::evaluate;

    type F = BinaryField128;

    fn product(values: &[F; 3]) -> F {
        values[0] * values[1] * values[2]
    }

    #[test]
    fn binding_every_variable_reads_each_table_at_the_challenges() {
        // Fixture state: three distinct tables over a two-variable cube.
        let table = |seed: u128| {
            (0..4)
                .map(|row| F::from_repr(seed + row))
                .collect::<Vec<_>>()
        };
        let tables = [table(1), table(10), table(20)];
        let mut prover = Composite::new(tables.clone(), product);

        // Binding both variables must land on each table's extension at the challenge point.
        let challenges = [F::from_repr(7), F::from_repr(9)];
        for challenge in challenges {
            prover.fold(challenge);
        }
        let expected = tables.map(|table| evaluate(&table, &challenges));
        assert_eq!(prover.terminal(), expected);
    }

    #[test]
    fn the_first_round_splits_the_sum_at_zero_and_one() {
        // Invariant: the round polynomial at zero plus its value at one is the full sum.
        let table = |seed: u128| {
            (0..8)
                .map(|row| F::from_repr(seed * row + 3))
                .collect::<Vec<_>>()
        };
        let prover = Composite::new([table(2), table(5), table(11)], product);
        let total = (0..8)
            .map(|row| product(&[table(2)[row], table(5)[row], table(11)[row]]))
            .sum::<F>();
        assert_eq!(prover.round_at(F::ZERO) + prover.round_at(F::ONE), total);
    }
}
