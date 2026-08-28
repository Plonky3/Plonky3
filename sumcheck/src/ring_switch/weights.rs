//! The batched weight multilinear `A(X)` and the batched row reading of a tensor element
//! (Construction 3.1).

use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::tensor::TensorAlgebra;

/// The batched weight multilinear `A(w) = Σ_u eq̃(hypercube(u, κ), r'') · A_{w,u}`, where
/// `eq̃(r_high, w) = Σ_u A_{w,u} · β_u` decomposes the equality table over the base field.
///
/// # Panics
/// Panics unless `r_batch` names exactly the `κ = log2([EF:F])` batching variables.
pub fn batched_weights<F: Field, EF: ExtensionField<F>>(
    r_high: &Point<EF>,
    r_batch: &Point<EF>,
) -> Poly<EF> {
    assert_eq!(
        1 << r_batch.num_variables(),
        TensorAlgebra::<F, EF>::DIMENSION,
        "r_batch must name exactly the kappa batching variables"
    );
    let eq = Poly::<EF>::new_from_point(r_high.as_slice(), EF::ONE);
    let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
    let weights: Vec<EF> = eq
        .as_slice()
        .par_iter()
        .map(|e| {
            BasedVectorSpace::<F>::as_basis_coefficients_slice(e)
                .iter()
                .zip(eq_batch.as_slice())
                .map(|(&a, &w)| w * EF::from(a))
                .sum()
        })
        .collect();
    Poly::new(weights)
}

/// `Σ_u eq̃(hypercube(u, κ), r'') · m^u`, where `m^u` is row `u` of `tensor` read as an `EF`:
/// the rows batched against the equality table of `r_batch`.
///
/// Both ends of the sumcheck are this same operation on a different tensor element. On `ŝ` it
/// gives the initial claim `s_0`, which the verifier derives rather than receives. On the
/// equality element `e` it gives `A(r')`, the weight the surviving claim is scaled by.
///
/// # Panics
/// Panics unless `r_batch` names exactly the `κ = log2([EF:F])` batching variables.
pub fn batch_rows<F: Field, EF: ExtensionField<F>>(
    tensor: &TensorAlgebra<F, EF>,
    r_batch: &Point<EF>,
) -> EF {
    assert_eq!(
        1 << r_batch.num_variables(),
        TensorAlgebra::<F, EF>::DIMENSION,
        "r_batch must name exactly the kappa batching variables"
    );
    let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
    tensor
        .rows()
        .iter()
        .zip(eq_batch.as_slice())
        .map(|(&row, &w)| row * w)
        .sum()
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::ring_switch::packing::{compute_s_hat, pack};
    use crate::ring_switch::test_util::{EF, F, base_poly};

    /// `Poly::new_from_point` indexes its table in the same MSB-first order as
    /// `Point::hypercube`: entry `u` is `eq̃(hypercube(u, κ), point)`.
    #[test]
    fn new_from_point_matches_hypercube_indexing() {
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(1), 4);
        let eq = Poly::<EF>::new_from_point(point.as_slice(), EF::ONE);
        for u in 0..16 {
            let expected =
                Point::eval_eq(Point::<EF>::hypercube(u, 4).as_slice(), point.as_slice());
            assert_eq!(eq.as_slice()[u], expected, "entry {u}");
        }
    }

    /// Row `u` of `ŝ` is the `A_{·,u}`-weighted sum of the packed values, and the batched
    /// weights reproduce the batched row sum. This is the identity the sumcheck proves.
    #[test]
    fn rows_and_weights_agree_with_the_packed_values() {
        let ell = 6;
        let t = base_poly(ell, 7);
        let packed = pack::<F, EF>(&t);
        let mut rng = SmallRng::seed_from_u64(8);
        let r = Point::<EF>::rand(&mut rng, ell);
        let (r_high, _) = r.split_at(ell - 4);
        let r_batch = Point::<EF>::rand(&mut rng, 4);

        let s_hat = compute_s_hat::<F, EF>(&packed, &r_high);
        let weights = batched_weights::<F, EF>(&r_high, &r_batch);
        assert_eq!(weights.num_variables(), ell - 4);

        // The sumcheck's claim: Σ_w A(w) · t'(w) equals the batched row reading.
        let dot: EF = weights
            .as_slice()
            .iter()
            .zip(packed.as_slice())
            .map(|(a, p)| *a * *p)
            .sum();
        assert_eq!(dot, batch_rows::<F, EF>(&s_hat, &r_batch));
    }

    /// The unbatched identity, checked row by row so a failure localises.
    #[test]
    fn each_row_is_its_own_weighted_sum() {
        let ell = 5;
        let t = base_poly(ell, 9);
        let packed = pack::<F, EF>(&t);
        let r_high = Point::<EF>::rand(&mut SmallRng::seed_from_u64(10), ell - 4);

        let rows = compute_s_hat::<F, EF>(&packed, &r_high).rows();
        let eq = Poly::<EF>::new_from_point(r_high.as_slice(), EF::ONE);

        for (u, row) in rows.iter().enumerate() {
            let expected: EF = eq
                .as_slice()
                .iter()
                .zip(packed.as_slice())
                .map(|(e, p)| {
                    *p * EF::from(BasedVectorSpace::<F>::as_basis_coefficients_slice(e)[u])
                })
                .sum();
            assert_eq!(*row, expected, "row {u}");
        }
    }

    /// A wrong-length `r_batch` must panic rather than silently drop rows of `ŝ` — the
    /// `zip` in `batched_weights` would otherwise truncate to the shorter side, and the
    /// dropped rows would go undetected because both sides of the sumcheck's identity are
    /// wrong in the same direction.
    #[test]
    #[should_panic(expected = "r_batch must name exactly the kappa batching variables")]
    fn batched_weights_rejects_a_short_r_batch() {
        let r_high = Point::<EF>::rand(&mut SmallRng::seed_from_u64(11), 2);
        let r_batch = Point::<EF>::rand(&mut SmallRng::seed_from_u64(12), 3);
        let _ = batched_weights::<F, EF>(&r_high, &r_batch);
    }

    /// The same guard on `batch_rows`: a wrong-length `r_batch` must panic rather than
    /// silently drop rows of `ŝ` from the sumcheck's initial claim.
    #[test]
    #[should_panic(expected = "r_batch must name exactly the kappa batching variables")]
    fn batch_rows_rejects_a_short_r_batch() {
        let r_high = Point::<EF>::rand(&mut SmallRng::seed_from_u64(13), 0);
        let t = base_poly(4, 14);
        let packed = pack::<F, EF>(&t);
        let s_hat = compute_s_hat::<F, EF>(&packed, &r_high);
        let r_batch = Point::<EF>::rand(&mut SmallRng::seed_from_u64(15), 3);
        let _ = batch_rows::<F, EF>(&s_hat, &r_batch);
    }
}
