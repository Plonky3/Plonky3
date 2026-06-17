//! Per-round polynomial evaluation used by both the prover and the verifier.

use alloc::vec::Vec;

use p3_field::{Field, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::interpolation::{InterpolateArbitrary, barycentric_weights};

/// Evaluates each round polynomial at the sampled challenge.
///
/// Every round shares the same interpolation domain: the integer nodes `0, 1, …, degree`.
///
/// - The barycentric weights depend only on that domain.
/// - They are precomputed once and reused across all rounds.
/// - Each round then contributes only the per-challenge difference inverses.
pub(super) struct RoundPolyInterpolator<EF> {
    /// Integer domain nodes `0, 1, …, degree`, lifted into the field.
    x_coords: Vec<EF>,
    /// Barycentric weight `w_i = 1 / prod_{j != i} (x_i - x_j)` of each node.
    weights: Vec<EF>,
}

impl<EF: Field> RoundPolyInterpolator<EF> {
    /// Precomputes the domain and barycentric weights for a degree-`degree` round polynomial.
    ///
    /// # Arguments
    ///
    /// - `degree`: the per-variable degree, so the domain has `degree + 1` nodes.
    pub(super) fn new(degree: usize) -> Self {
        // Domain: the integer nodes 0, 1, …, degree lifted into the field.
        let x_coords: Vec<EF> = (0..=degree).map(EF::from_usize).collect();

        // Integer nodes are pairwise distinct, so the weights always exist.
        let weights = barycentric_weights(&x_coords).expect("integer nodes are pairwise distinct");

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
    /// `h(r)`, computed by barycentric Lagrange interpolation over the precomputed domain.
    pub(super) fn eval(&self, evals: &[EF], sum_constraint: EF, r: EF) -> EF {
        debug_assert_eq!(evals.len() + 1, self.x_coords.len());

        // Reconstruct the full evaluation vector at the integer nodes 0, 1, …, degree.
        //
        //     evals[0] = h(0)
        //     evals[i] = h(i + 1) for i in 1..degree
        //     h(1)     = sum_constraint - h(0)        (sumcheck invariant)
        let mut full: Vec<EF> = Vec::with_capacity(self.x_coords.len());
        full.push(evals[0]);
        full.push(sum_constraint - evals[0]);
        full.extend_from_slice(&evals[1..]);

        // Distance from the challenge to each domain node.
        let diffs: Vec<EF> = self.x_coords.iter().map(|&x| r - x).collect();

        // A challenge that lands on a domain node is evaluated exactly there.
        //
        // This both returns the right answer and avoids inverting a zero below.
        if let Some(i) = diffs.iter().position(|d| d.is_zero()) {
            return full[i];
        }

        // Off-domain: invert the differences and apply the precomputed weights.
        let diff_invs = batch_multiplicative_inverse(&diffs);
        RowMajorMatrix::new_col(full)
            .interpolate_arbitrary_with_precomputation(&self.weights, &diff_invs)[0]
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
        // Invariant: a challenge on a domain node returns that node's value via the zero-difference shortcut, not the inversion path.
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
