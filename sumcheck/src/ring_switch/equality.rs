//! The tensor-algebra equality element (Remark 3.4) and its characteristic-2 recurrence.

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::tensor::TensorAlgebra;

/// `e = eq̃(φ0(r_high), φ1(r'))`, the tensor-algebra equality element the verifier needs to
/// close the sumcheck without evaluating the weight multilinear from its definition.
///
/// Computed by the char-2 recurrence of Remark 3.4: starting from `1 ⊗ 1`, for each `i`,
/// `e ← e + φ0(r_high_i)·e + φ1(r'_i)·e`. Costs `2·ℓ'·d` extension multiplications.
///
/// # Panics
/// Panics unless `r_high` and `r_challenge` have the same length, and unless `F` has
/// characteristic 2 — in odd characteristic the recurrence needs a third term and this
/// function does not implement it.
pub fn equality_element<F: Field, EF: ExtensionField<F>>(
    r_high: &Point<EF>,
    r_challenge: &Point<EF>,
) -> TensorAlgebra<F, EF> {
    assert_eq!(
        r_high.num_variables(),
        r_challenge.num_variables(),
        "r_high and r_challenge must name the same number of variables"
    );
    assert_eq!(
        F::PrimeSubfield::ONE + F::PrimeSubfield::ONE,
        F::PrimeSubfield::ZERO,
        "the Remark 3.4 recurrence relies on eq̃(X, Y) = Π(1 − X_i − Y_i), which holds only \
         when 2 = 0; in odd characteristic the factor is (1 − X_i − Y_i + 2·X_i·Y_i) and needs \
         a third term this recurrence does not implement — use equality_element_reference there"
    );

    let mut e = TensorAlgebra::one();
    for i in 0..r_high.num_variables() {
        let mut by_high = e.clone();
        by_high.scale_columns(r_high[i]);
        let mut by_chal = e.clone();
        by_chal.scale_rows(r_challenge[i]);
        e += by_high;
        e += by_chal;
    }
    e
}

/// `e` from its definition: `Σ_w φ0(eq̃(r_high, w)) · φ1(eq̃(w, r'))`. Exponential in the
/// number of variables; the oracle the recurrence is checked against, and the route for fields
/// of odd characteristic.
///
/// # Panics
/// Panics unless `r_high` and `r_challenge` have the same length.
pub fn equality_element_reference<F: Field, EF: ExtensionField<F>>(
    r_high: &Point<EF>,
    r_challenge: &Point<EF>,
) -> TensorAlgebra<F, EF> {
    assert_eq!(
        r_high.num_variables(),
        r_challenge.num_variables(),
        "r_high and r_challenge must name the same number of variables"
    );

    let eq_high = Poly::<EF>::new_from_point(r_high.as_slice(), EF::ONE);
    let eq_chal = Poly::<EF>::new_from_point(r_challenge.as_slice(), EF::ONE);
    let mut e = TensorAlgebra::zero();
    for (&a, &b) in eq_high.as_slice().iter().zip(eq_chal.as_slice()) {
        e += TensorAlgebra::exterior_product(a, b);
    }
    e
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::ring_switch::test_util::{EF, F};
    use crate::ring_switch::weights::batched_weights;

    /// The recurrence agrees with the definition summed over the hypercube.
    #[test]
    fn the_recurrence_matches_the_definition() {
        let mut rng = SmallRng::seed_from_u64(11);
        for ell_prime in 0..=4 {
            let r_high = Point::<EF>::rand(&mut rng, ell_prime);
            let r_chal = Point::<EF>::rand(&mut rng, ell_prime);
            assert_eq!(
                equality_element::<F, EF>(&r_high, &r_chal),
                equality_element_reference::<F, EF>(&r_high, &r_chal),
                "ell' = {ell_prime}"
            );
        }
    }

    /// Row `u` of `e` is the `A_{·,u}`-weighted sum of `eq̃(w, r')` — the identity that makes
    /// the batched rows of `e` equal `A(r')`, which is the sumcheck's endpoint.
    #[test]
    fn equality_rows_reproduce_the_batched_weights() {
        let mut rng = SmallRng::seed_from_u64(12);
        let ell_prime = 3;
        let r_high = Point::<EF>::rand(&mut rng, ell_prime);
        let r_chal = Point::<EF>::rand(&mut rng, ell_prime);
        let r_batch = Point::<EF>::rand(&mut rng, 4);

        let rows = equality_element::<F, EF>(&r_high, &r_chal).rows();
        let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
        let batched: EF = rows
            .iter()
            .zip(eq_batch.as_slice())
            .map(|(e, b)| *e * *b)
            .sum();

        let weights = batched_weights::<F, EF>(&r_high, &r_batch);
        assert_eq!(batched, weights.eval_ext::<F>(&r_chal));
    }

    /// The degenerate case the whole recurrence starts from.
    #[test]
    fn empty_point_gives_the_identity() {
        let e = equality_element::<F, EF>(&Point::new(vec![]), &Point::new(vec![]));
        assert_eq!(e, TensorAlgebra::<F, EF>::one());
    }
}
