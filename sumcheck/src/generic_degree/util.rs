//! Per-round polynomial evaluation used by both the prover and the verifier.

use alloc::vec::Vec;

use p3_field::{Algebra, Dup, Field};
use p3_matrix::interpolation::barycentric_weights;

/// Evaluates each round polynomial at the sampled challenge.
///
/// Every round shares the same interpolation domain: the integer nodes `0, 1, …, degree`.
///
/// - The barycentric weights depend only on that domain.
/// - They are precomputed once and reused across all rounds.
/// - Each round then contributes only the per-challenge difference products.
pub struct RoundPolyInterpolator<EF> {
    /// Integer domain nodes `0, 1, …, degree`, lifted into the field.
    x_coords: Vec<EF>,
    /// Barycentric weight `w_i = 1 / prod_{j != i} (x_i - x_j)` of each node.
    weights: Vec<EF>,
}

impl<EF: Field> RoundPolyInterpolator<EF> {
    /// Return the node encoded by one transmitted round-polynomial entry.
    ///
    /// Entry zero carries `h(0)`; every later entry skips the derived value `h(1)`.
    #[must_use]
    pub fn transmitted_node(index: usize) -> EF {
        EF::interpolation_node(if index == 0 { 0 } else { index + 1 })
    }

    /// Precomputes the domain and barycentric weights for a degree-`degree` round polynomial.
    ///
    /// # Arguments
    ///
    /// - `degree`: the per-variable degree, so the domain has `degree + 1` nodes.
    pub fn new(degree: usize) -> Self {
        // Domain: the first `degree + 1` interpolation nodes of the field.
        let x_coords: Vec<EF> = (0..=degree).map(EF::interpolation_node).collect();

        // Interpolation nodes are pairwise distinct, so the weights always exist.
        let weights =
            barycentric_weights(&x_coords).expect("interpolation nodes are pairwise distinct");

        Self { x_coords, weights }
    }

    /// Evaluates the round polynomial at the sampled challenge.
    ///
    /// # Arguments
    ///
    /// - `evals`: transmitted evaluations `h(0), h(2), …, h(degree)`.
    /// - `sum_constraint`: the running claimed sum, equal to `h(0) + h(1)`.
    /// - `r`: the verifier's challenge point.
    ///
    /// # Returns
    ///
    /// `h(r)`, computed by Lagrange interpolation over the precomputed domain.
    pub fn eval(&self, evals: &[EF], sum_constraint: EF, r: EF) -> EF {
        self.eval_in(evals, sum_constraint, &r)
    }

    /// As [`Self::eval`], over any algebra of the field.
    ///
    /// The Lagrange basis is formed from prefix and suffix products of `r - x_j`, so nothing is inverted.
    pub fn eval_in<E: Algebra<EF> + Dup>(&self, evals: &[E], sum_constraint: E, r: &E) -> E {
        // Width contract, enforced upstream by the round polynomial's fixed-length step.
        // A bounded length there would replace that rejection with a silent mis-interpolation.
        debug_assert_eq!(evals.len() + 1, self.x_coords.len());

        // The evaluations at the integer nodes 0, 1, …, degree, where h(1) = sum_constraint - h(0).
        let h1 = sum_constraint - evals[0].dup();
        let full = core::iter::once(&evals[0])
            .chain(core::iter::once(&h1))
            .chain(&evals[1..]);

        // L_i(r) = w_i * prod_{j < i} (r - x_j) * prod_{j > i} (r - x_j).
        let diffs: Vec<E> = self.x_coords.iter().map(|&x| r.dup() - x).collect();
        let mut suffix = Vec::with_capacity(diffs.len());
        suffix.push(E::ONE);
        for diff in diffs.iter().rev() {
            suffix.push(suffix[suffix.len() - 1].dup() * diff.dup());
        }
        let mut prefix = E::ONE;
        let mut value = E::ZERO;
        for (i, (h, (&weight, diff))) in full.zip(self.weights.iter().zip(&diffs)).enumerate() {
            value += prefix.dup() * suffix[diffs.len() - 1 - i].dup() * (h.dup() * weight);
            prefix *= diff.dup();
        }
        value
    }

    /// Extend a round polynomial's transmitted evaluations up to `target_len` entries.
    ///
    /// The layout matches [`Self::eval`]: entry `0` is `h(0)`, entry `i >= 1` is `h(i + 1)`.
    /// Entries past the input are the same polynomial extrapolated at the higher nodes.
    ///
    /// # Arguments
    ///
    /// - `evals`: transmitted evaluations `h(0), h(2), …, h(degree)`.
    /// - `sum_constraint`: the running claimed sum, equal to `h(0) + h(1)`.
    /// - `target_len`: length of the returned vector; must be at least the input length.
    pub fn extend_evals(&self, evals: &[EF], sum_constraint: EF, target_len: usize) -> Vec<EF> {
        let mut extended = Vec::with_capacity(target_len);
        extended.extend_from_slice(evals);
        // Each appended entry `i` carries node `i + 1`, extrapolated from the input.
        extended.extend(
            (evals.len()..target_len)
                .map(|i| self.eval(evals, sum_constraint, EF::interpolation_node(i + 1))),
        );
        extended
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn poly_eval(coeffs: &[EF], x: EF) -> EF {
        coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * x + c)
    }

    #[test]
    fn round_poly_eval_at_known_points_matches_definition() {
        // Fixture state:
        //
        //     polynomial: random degree-3, four coefficients
        //     known evals: h(0), h(1), h(2), h(3)
        //
        // Invariant:
        //
        //     the interpolator recovers h(r) from [h(0), h(2), h(3)] and the sum h(0) + h(1).
        let mut rng = SmallRng::seed_from_u64(11);
        let coeffs: Vec<EF> = (0..4)
            .map(|_| EF::from_u64(rng.random_range(0..(1 << 30))))
            .collect();

        let h_at = |x: EF| poly_eval(&coeffs, x);
        let h0 = h_at(EF::ZERO);
        let h1 = h_at(EF::ONE);
        let h2 = h_at(EF::from_u64(2));
        let h3 = h_at(EF::from_u64(3));

        // Transmit only [h(0), h(2), h(3)]; the sum carries h(1).
        let evals = vec![h0, h2, h3];
        let sum = h0 + h1;
        let r = EF::from_u64(rng.random_range(0..(1 << 30)));

        // The interpolator must reproduce direct polynomial evaluation at r.
        let interp = RoundPolyInterpolator::new(3);
        assert_eq!(interp.eval(&evals, sum, r), h_at(r));
    }

    #[test]
    fn round_poly_eval_on_domain_node_is_exact() {
        // Invariant: a challenge on a domain node returns that node's value.
        //
        // Fixture state: random degree-3 polynomial.
        //
        //     r = 2  → transmitted node, must return h(2)
        //     r = 1  → recovered node,   must return h(1)
        let mut rng = SmallRng::seed_from_u64(7);
        let coeffs: Vec<EF> = (0..4)
            .map(|_| EF::from_u64(rng.random_range(0..(1 << 30))))
            .collect();

        let h_at = |x: EF| poly_eval(&coeffs, x);
        let evals = vec![h_at(EF::ZERO), h_at(EF::from_u64(2)), h_at(EF::from_u64(3))];
        let sum = h_at(EF::ZERO) + h_at(EF::ONE);

        let interp = RoundPolyInterpolator::new(3);
        assert_eq!(
            interp.eval(&evals, sum, EF::from_u64(2)),
            h_at(EF::from_u64(2))
        );
        assert_eq!(interp.eval(&evals, sum, EF::ONE), h_at(EF::ONE));
    }
}
