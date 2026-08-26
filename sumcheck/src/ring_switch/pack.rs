//! Packing a base-field multilinear into an extension-field one (Construction 3.1), and
//! reading `ŝ` by columns.

use core::ops::AddAssign;

use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use super::tensor::TensorAlgebra;

/// The number of Boolean variables one packed element absorbs, `log2([EF:F])`.
///
/// # Panics
/// Panics unless `[EF:F]` is a power of two.
pub fn packed_vars<F: Field, EF: ExtensionField<F>>() -> usize {
    let d = <EF as BasedVectorSpace<F>>::DIMENSION;
    assert!(
        d.is_power_of_two(),
        "ring switching needs a power-of-two extension degree, got {d}"
    );
    log2_strict_usize(d)
}

/// Packs `2^ℓ` base evaluations into `2^(ℓ − κ)` extension ones: entry `d·w + j` becomes
/// basis coefficient `j` of packed element `w`.
pub fn pack<F: Field, EF: ExtensionField<F>>(t: &Poly<F>) -> Poly<EF> {
    Poly::new(<EF as BasedVectorSpace<F>>::reconstitute_from_base(
        t.as_slice().to_vec(),
    ))
}

/// `ŝ = Σ_w eq̃(r_high, w) ⊗ t'(w)`.
///
/// # Panics
/// Panics unless `r_high` names exactly `packed`'s variables.
pub fn compute_s_hat<F: Field, EF: ExtensionField<F>>(
    packed: &Poly<EF>,
    r_high: &Point<EF>,
) -> TensorAlgebra<F, EF> {
    assert_eq!(
        r_high.num_variables(),
        packed.num_variables(),
        "r_high must name exactly the packed polynomial's variables"
    );
    let eq = Poly::<EF>::new_from_point(r_high.as_slice(), EF::ONE);
    let mut s_hat = TensorAlgebra::zero();
    for (&e, &p) in eq.as_slice().iter().zip(packed.as_slice()) {
        s_hat.add_assign(TensorAlgebra::exterior_product(e, p));
    }
    s_hat
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::ring_switch::test_util::{EF, F, base_poly};

    /// Packing groups the SUFFIX variables: entry `16w + j` is coefficient `j` of element `w`.
    #[test]
    fn packing_consumes_the_suffix_variables() {
        let t = base_poly(6, 1);
        let packed = pack::<F, EF>(&t);
        assert_eq!(packed.num_variables(), 6 - 4);
        for (w, e) in packed.as_slice().iter().enumerate() {
            for j in 0..16 {
                assert_eq!(
                    BasedVectorSpace::<F>::as_basis_coefficients_slice(e)[j],
                    t.as_slice()[16 * w + j]
                );
            }
        }
    }

    /// Column `v` of `ŝ` is `t` with its packed variables fixed to `v` and the rest at `r_high`.
    #[test]
    fn s_hat_columns_are_the_partial_evaluations() {
        let ell = 6;
        let t = base_poly(ell, 2);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(3), ell);
        let (r_high, r_low) = r.split_at(ell - 4);

        let s_hat = compute_s_hat::<F, EF>(&packed, &r_high);
        let cols = s_hat.columns();

        #[allow(clippy::needless_range_loop)]
        for v in 0..16 {
            // `t` evaluated at (r_high, hypercube(v, 4)) — packed variables are the suffix.
            let mut point = r_high.clone();
            point.extend(&Point::<EF>::hypercube(v, 4));
            assert_eq!(cols[v], t.eval_base(&point), "column {v}");
        }

        // And the claim itself is the `eq̃(·, r_low)`-combination of those columns.
        let s = t.eval_base(&r);
        let combined: EF = (0..16)
            .map(|v| {
                Point::eval_eq(Point::<EF>::hypercube(v, 4).as_slice(), r_low.as_slice()) * cols[v]
            })
            .sum();
        assert_eq!(combined, s);
    }

    /// Evaluating the packed polynomial at `r_high` is the `β`-combination of the same columns,
    /// and it is NOT the claim. This is why the reduction exists (Remark 3.3); deleting it in
    /// favour of a direct opening is unsound.
    #[test]
    fn the_packed_evaluation_is_lossy() {
        let ell = 6;
        let t = base_poly(ell, 4);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(5), ell);
        let (r_high, _) = r.split_at(ell - 4);

        let cols = compute_s_hat::<F, EF>(&packed, &r_high).columns();
        let beta_combination: EF = (0..16)
            .map(|v| cols[v] * EF::from_basis_coefficients_fn(|i| F::from_bool(i == v)))
            .sum();

        assert_eq!(beta_combination, packed.eval_ext::<F>(&r_high));
        assert_ne!(beta_combination, t.eval_base(&r));
    }
}
