//! The tensor-algebra equality element (Remark 3.4) and its recurrence.

use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;

use super::tensor::TensorAlgebra;

/// `e = eq̃(φ0(r_high), φ1(r'))`, the tensor-algebra equality element the verifier needs to
/// close the sumcheck without evaluating the weight multilinear from its definition.
///
/// Remark 3.4's recurrence, over any characteristic. `eq̃(X, Y) = Π_i (X_i·Y_i + (1 − X_i)(1 − Y_i))`,
/// so starting from `1 ⊗ 1` each variable contributes
/// `e ← φ0(a)·φ1(b)·e + φ0(1 − a)·φ1(1 − b)·e`, formed by scaling a copy by `a` then `b` and
/// the running element by `1 − a` then `1 − b`. Costs `O(ℓ' · d)` extension multiplications.
///
/// # Panics
/// Panics unless `r_high` and `r_prime` have the same length.
pub fn equality_element<F: Field, EF: ExtensionField<F>>(
    r_high: &Point<EF>,
    r_prime: &Point<EF>,
) -> TensorAlgebra<F, EF> {
    assert_eq!(
        r_high.num_variables(),
        r_prime.num_variables(),
        "r_high and r_prime must name the same number of variables"
    );

    let mut e = TensorAlgebra::one();
    for i in 0..r_high.num_variables() {
        let (a, b) = (r_high[i], r_prime[i]);

        // `φ0(a)·φ1(b)·e`; the two scalings commute, so either order gives the term.
        let mut agree = e.clone();
        agree.scale_columns(a);
        agree.scale_rows(b);

        e.scale_columns(EF::ONE - a);
        e.scale_rows(EF::ONE - b);
        e += agree;
    }
    e
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::ring_switch::test_util::{EF, F};
    use crate::ring_switch::weights::batched_weights;

    /// `e` from its definition: `Σ_w φ0(eq̃(r_high, w)) · φ1(eq̃(w, r'))`. Exponential in the
    /// number of variables; the oracle [`equality_element`] is checked against.
    ///
    /// # Panics
    /// Panics unless `r_high` and `r_prime` have the same length.
    fn equality_element_reference<G: Field, EG: ExtensionField<G>>(
        r_high: &Point<EG>,
        r_prime: &Point<EG>,
    ) -> TensorAlgebra<G, EG> {
        assert_eq!(
            r_high.num_variables(),
            r_prime.num_variables(),
            "r_high and r_prime must name the same number of variables"
        );

        let eq_high = Poly::<EG>::new_from_point(r_high.as_slice(), EG::ONE);
        let eq_prime = Poly::<EG>::new_from_point(r_prime.as_slice(), EG::ONE);
        let mut e = TensorAlgebra::zero();
        for (&a, &b) in eq_high.as_slice().iter().zip(eq_prime.as_slice()) {
            e += TensorAlgebra::exterior_product(a, b);
        }
        e
    }

    /// The recurrence agrees with the definition summed over the hypercube.
    #[test]
    fn the_recurrence_matches_the_definition() {
        let mut rng = SmallRng::seed_from_u64(11);
        for ell_prime in 0..=4 {
            let r_high = Point::<EF>::rand(&mut rng, ell_prime);
            let r_prime = Point::<EF>::rand(&mut rng, ell_prime);
            assert_eq!(
                equality_element::<F, EF>(&r_high, &r_prime),
                equality_element_reference::<F, EF>(&r_high, &r_prime),
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
        let r_prime = Point::<EF>::rand(&mut rng, ell_prime);
        let r_batch = Point::<EF>::rand(&mut rng, 4);

        let rows = equality_element::<F, EF>(&r_high, &r_prime).rows();
        let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
        let batched: EF = rows
            .iter()
            .zip(eq_batch.as_slice())
            .map(|(e, b)| *e * *b)
            .sum();

        let weights = batched_weights::<F, EF>(&r_high, &r_batch);
        assert_eq!(batched, weights.eval_ext::<F>(&r_prime));
    }

    /// The same agreement in odd characteristic, where `1 − a` and `1 + a` differ: a
    /// factorisation of `eq̃`'s per-variable factor that only held when `2 = 0` would part
    /// company with the definition here.
    #[test]
    fn the_recurrence_matches_the_definition_in_odd_characteristic() {
        type G = BabyBear;
        type EG = BinomialExtensionField<BabyBear, 4>;

        let mut rng = SmallRng::seed_from_u64(13);
        for ell_prime in 0..=5 {
            let r_high = Point::<EG>::rand(&mut rng, ell_prime);
            let r_prime = Point::<EG>::rand(&mut rng, ell_prime);
            assert_eq!(
                equality_element::<G, EG>(&r_high, &r_prime),
                equality_element_reference::<G, EG>(&r_high, &r_prime),
                "ell' = {ell_prime}"
            );
        }
    }

    /// The degenerate case the whole recurrence starts from.
    #[test]
    fn empty_point_gives_the_identity() {
        let e = equality_element::<F, EF>(&Point::new(vec![]), &Point::new(vec![]));
        assert_eq!(e, TensorAlgebra::<F, EF>::one());
    }

    proptest! {
        /// The recurrence agrees with the definition at random lengths and points, not only at
        /// the fixed seeds the cases above pin. Six variables keep the exponential reference at
        /// `2^6` terms.
        #[test]
        fn the_recurrence_matches_the_definition_everywhere(ell_prime in 0usize..6, seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let r_high = Point::<EF>::rand(&mut rng, ell_prime);
            let r_prime = Point::<EF>::rand(&mut rng, ell_prime);
            prop_assert_eq!(
                equality_element::<F, EF>(&r_high, &r_prime),
                equality_element_reference::<F, EF>(&r_high, &r_prime),
            );
        }

        /// The same sweep in odd characteristic, which is where a factorisation that held only
        /// when `2 = 0` would show.
        #[test]
        fn the_recurrence_matches_the_definition_everywhere_in_odd_characteristic(
            ell_prime in 0usize..6,
            seed: u64,
        ) {
            type G = BabyBear;
            type EG = BinomialExtensionField<BabyBear, 4>;

            let mut rng = SmallRng::seed_from_u64(seed);
            let r_high = Point::<EG>::rand(&mut rng, ell_prime);
            let r_prime = Point::<EG>::rand(&mut rng, ell_prime);
            prop_assert_eq!(
                equality_element::<G, EG>(&r_high, &r_prime),
                equality_element_reference::<G, EG>(&r_high, &r_prime),
            );
        }
    }
}
