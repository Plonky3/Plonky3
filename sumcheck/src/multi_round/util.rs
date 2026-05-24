//! Per-round polynomial evaluation used by both the prover and the verifier.

use alloc::vec::Vec;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::interpolation::InterpolateArbitrary;

/// Evaluate the round polynomial at the sampled challenge.
///
/// # Arguments
///
/// - `evals`: transmitted evaluations `h(0), h(2), …, h(degree)`.
/// - `sum_constraint`: the running claimed sum, equal to `h(0) + h(1)`.
/// - `r`: the verifier's challenge point.
///
/// # Returns
///
/// `h(r)`, computed by barycentric Lagrange interpolation over the integer nodes `0, 1, …, degree`.
#[inline]
pub(super) fn evaluate_round_poly_at<EF: Field>(evals: &[EF], sum_constraint: EF, r: EF) -> EF {
    let d = evals.len();
    debug_assert!(d > 0);

    // Reconstruct the full evaluation vector at the integer nodes 0, 1, …, d.
    //
    //     evals[0] = h(0)
    //     evals[i] = h(i + 1) for i in 1..d
    //     h(1)     = sum_constraint - h(0)        (sumcheck invariant)
    let mut full: Vec<EF> = Vec::with_capacity(d + 1);
    full.push(evals[0]);
    full.push(sum_constraint - evals[0]);
    full.extend_from_slice(&evals[1..]);

    // Domain: integer nodes lifted into the extension field.
    let x_coords: Vec<EF> = (0..=d).map(EF::from_usize).collect();

    // Build a one-column matrix view since the workspace interpolator works on matrices.
    //
    // Integer domain points 0, 1, …, d are pairwise distinct, so the helper never returns `None`.
    RowMajorMatrix::new_col(full)
        .interpolate_arbitrary_point(&x_coords, r)
        .expect("integer nodes are pairwise distinct")[0]
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
        // Horner-style accumulation of a polynomial in coefficient form.
        let mut acc = EF::ZERO;
        let mut pow = EF::ONE;
        for &c in coeffs {
            acc += c * pow;
            pow *= x;
        }
        acc
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
        //     the wrapper must recover h(r) from the transmitted subset
        //     [h(0), h(2), h(3)] together with the sum h(0) + h(1).
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

        // Wrapper must reproduce direct polynomial evaluation at r.
        let got = evaluate_round_poly_at(&evals, sum, r);
        assert_eq!(got, h_at(r));
    }
}
