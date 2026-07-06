//! Deep quotient computation for Circle STARKs.
//!
//! This module implements the DEEP (Domain Extension for Eliminating Pretenders) quotient
//! computation as described in the Circle STARKs paper. This allows the verifier to check
//! polynomial constraints by evaluating them at random points outside the original domain.
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_field::extension::ComplexExtendable;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, batch_multiplicative_inverse, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::CircleEvaluations;
use crate::domain::CircleDomain;
use crate::ordering::cfft_permute_index;
use crate::point::Point;

/// Compute the "vanishing part" of the DEEP quotient numerator and denominator.
///
/// See Section 6, Remark 21 of the Circle STARKs paper (page 30 of first edition PDF).
///
/// # Mathematical Background
///
/// The DEEP quotient has two parts:
/// 1. The vanishing part: `Re(1/v_gamma) + alpha^L * Im(1/v_gamma)` (computed here)
/// 2. The constraint part: `\bar g - \bar v_gamma` (computed elsewhere)
///
/// Where `v_gamma` is the vanishing polynomial at point `gamma = zeta`.
///
/// # Parameters
///
/// - `x`: A point on the circle domain
/// - `zeta`: The random challenge point (outside the original domain)
/// - `alpha_pow_width`: The challenge `alpha` raised to the power of the polynomial width
///
/// # Returns
/// The numerator and denominator for the vanishing part.
pub(crate) fn deep_quotient_vanishing_part<F: ComplexExtendable, EF: ExtensionField<F>>(
    x: Point<F>,
    zeta: Point<EF>,
    alpha_pow_width: EF,
) -> (EF, EF) {
    // Compute the vanishing polynomial v_p(zeta) = (x - zeta) * (x - zeta_bar)
    let (re_v_zeta, im_v_zeta) = x.v_p(zeta);

    // Numerator: Re(1/v_gamma) + alpha^L * Im(1/v_gamma)
    let numerator = re_v_zeta - alpha_pow_width * im_v_zeta;

    // Denominator: |v_gamma|^2 = Re(v_gamma)^2 + Im(v_gamma)^2
    let denominator = re_v_zeta.square() + im_v_zeta.square();

    (numerator, denominator)
}

/// Compute the DEEP quotient for a single row of polynomial evaluations.
///
/// This function computes the DEEP quotient for a single point `x` on the circle domain,
/// given the polynomial evaluations at that point and at the challenge point `zeta`.
///
/// # Mathematical Background
///
/// The DEEP quotient for a polynomial `f` at point `x` is:
/// `(f(x) - f(zeta)) / (x - zeta)`
///
/// This function computes this quotient by:
/// 1. Computing the vanishing part (handles the `(x - zeta)` denominator)
/// 2. Computing the constraint part (handles the `f(x) - f(zeta)` numerator)
///
/// # Parameters
///
/// - `alpha`: The random challenge scalar
/// - `x`: A point on the circle domain
/// - `zeta`: The random challenge point (outside the original domain)
/// - `ps_at_x`: Polynomial evaluations at point `x` (one per polynomial)
/// - `ps_at_zeta`: Polynomial evaluations at challenge point `zeta`
///
/// # Returns
///
/// - `Some(value)`: the DEEP quotient value for this row.
/// - `None`: the opening point coincides with this query point, so the denominator vanishes.
pub(crate) fn deep_quotient_reduce_row<F: ComplexExtendable, EF: ExtensionField<F>>(
    alpha: EF,
    x: Point<F>,
    zeta: Point<EF>,
    ps_at_x: &[F],
    ps_at_zeta: &[EF],
) -> Option<EF> {
    // Compute the vanishing part: handles the (x - zeta) denominator
    let (vp_num, vp_denom) =
        deep_quotient_vanishing_part(x, zeta, alpha.exp_u64(ps_at_x.len() as u64));

    // On the circle, the denominator `|v_p(zeta)|^2` reduces to `2 * (1 - (x - zeta).x)`.
    // This is zero exactly when `x == zeta`.
    // Return `None` there so the caller rejects the opening instead of dividing by zero.
    let vp_denom_inv = vp_denom.try_inverse()?;

    // Compute the constraint part: handles the f(x) - f(zeta) numerator
    let constraint_part = dot_product::<EF, _, _>(
        alpha.powers(),
        izip!(ps_at_x, ps_at_zeta).map(|(&p_at_x, &p_at_zeta)| -p_at_zeta + p_at_x),
    );

    // Combine vanishing part and constraint part
    Some(vp_num * vp_denom_inv * constraint_part)
}

/// The point-dependent part of the DEEP quotient on a fixed domain.
///
/// Holds `v_p(zeta) = re + im * i` for every domain point, along with the inverse of the
/// squared magnitude `|v_p(zeta)|^2`. These depend only on `(domain, zeta)`, so they are
/// shared by every matrix opened at `zeta` on that domain.
pub(crate) struct VanishingParts<EF> {
    re: Vec<EF>,
    im: Vec<EF>,
    denom_inv: Vec<EF>,
}

/// Compute the [`VanishingParts`] of the DEEP quotient at `zeta` for the given domain points.
///
/// `points` must be the domain points in CFFT order.
#[instrument(skip_all, fields(n = points.len()))]
pub(crate) fn compute_vanishing_parts<F: ComplexExtendable, EF: ExtensionField<F>>(
    points: &[Point<F>],
    zeta: Point<EF>,
) -> VanishingParts<EF> {
    let (re, im): (Vec<_>, Vec<_>) = points.par_iter().map(|&x| x.v_p(zeta)).unzip();
    let denoms = re
        .par_iter()
        .zip(&im)
        .map(|(&re, &im)| re.square() + im.square())
        .collect::<Vec<_>>();
    let denom_inv = batch_multiplicative_inverse(&denoms);
    VanishingParts { re, im, denom_inv }
}

/// Accumulate one matrix/point DEEP quotient into a running reduced opening:
///
/// `ro[i] += alpha_offset * (re[i] - alpha^W * im[i]) / |v_p(zeta)|^2[i] * (r[i] - c)`
///
/// where `r[i] = sum_j(alpha^j * p_j[x_i])` are the alpha-reduced rows of the matrix,
/// `c = sum_j(alpha^j * p_j[zeta])` is `reduced_ps_at_zeta` and `W` is the matrix width.
#[instrument(skip_all, fields(n = ro.len()), level = "debug")]
pub(crate) fn accumulate_deep_quotient<EF: Field>(
    ro: &mut [EF],
    alpha_offset: EF,
    alpha_pow_width: EF,
    reduced_rows: &[EF],
    vp: &VanishingParts<EF>,
    reduced_ps_at_zeta: EF,
) {
    ro.par_iter_mut()
        .zip(reduced_rows)
        .zip(&vp.re)
        .zip(&vp.im)
        .zip(&vp.denom_inv)
        .for_each(|((((ro, &reduced_ps_at_x), &re), &im), &denom_inv)| {
            *ro += alpha_offset
                * (re - alpha_pow_width * im)
                * denom_inv
                * (reduced_ps_at_x - reduced_ps_at_zeta);
        });
}

impl<F: ComplexExtendable, M: Matrix<F>> CircleEvaluations<F, M> {
    /// Reduce each row to a single value with powers of `alpha`: `r[i] = sum_j(alpha^j * m[i][j])`.
    ///
    /// This is the only part of the DEEP quotient that traverses the matrix, and it does not
    /// depend on the opening point, so it is computed once per matrix and shared by all points.
    #[instrument(skip_all, fields(dims = %self.values.dimensions()), level = "debug")]
    pub(crate) fn rowwise_alpha_reduce<EF: ExtensionField<F>>(&self, alpha: EF) -> Vec<EF> {
        let packed_alpha_powers =
            EF::ExtensionPacking::packed_ext_powers_capped(alpha, self.values.width())
                .collect_vec();
        self.values
            .rowwise_packed_dot_product::<EF>(&packed_alpha_powers)
            .collect()
    }

    /// Alpha-reduce evaluations over a subdomain, then lift the reduced column to
    /// `target_domain` with a narrow CFFT extrapolation.
    ///
    /// When `self` holds the trace-size subdomain prefix of a committed LDE (see
    /// `eval_at_point_on_subdomain_prefix_matches_full`), the reduced column
    /// `r = sum_j(alpha^j * p_j)` lies coordinate-wise in the pre-blow-up polynomial space,
    /// so its values on the prefix determine it. Reducing the prefix and extrapolating a
    /// single extension-field column costs `1 / blowup` of the full-matrix traversal plus
    /// a narrow CFFT, instead of a full traversal.
    #[instrument(skip_all, fields(dims = %self.values.dimensions()))]
    pub(crate) fn rowwise_alpha_reduce_lifted<EF: ExtensionField<F>>(
        &self,
        alpha: EF,
        target_domain: CircleDomain<F>,
    ) -> Vec<EF> {
        let reduced = self.rowwise_alpha_reduce(alpha);
        let flat = RowMajorMatrix::new_col(reduced).flatten_to_base();
        let lifted = CircleEvaluations::from_cfft_order(self.domain, flat)
            .extrapolate(target_domain)
            .to_cfft_order();
        EF::reconstitute_from_base(lifted.values)
    }
}

/// Extract and remove the vanishing polynomial component from LDE evaluations.
///
/// This function implements the lambda extraction algorithm described in Section 4.3, Lemma 6
/// of the Circle STARKs paper. It finds and removes the "error" component that is a scalar
/// multiple of the vanishing polynomial of the original domain.
///
/// The key insight is that `<v_n, f> = 0` for any polynomial `f` in the FFT space, so we can
/// use this orthogonality to extract `lambda`.
///
/// # Parameters
///
/// - `lde`: Mutable slice of LDE evaluations (will be modified in-place)
/// - `log_blowup`: Log of the blowup factor (how much larger the LDE domain is)
///
/// # Returns
///
/// The extracted coefficient `lambda` that was removed from the LDE evaluations.
#[instrument(skip_all, fields(bits = log2_strict_usize(lde.len())), level = "debug")]
pub fn extract_lambda<F: ComplexExtendable, EF: ExtensionField<F>>(
    lde: &mut [EF],
    log_blowup: usize,
) -> EF {
    let log_lde_size = log2_strict_usize(lde.len());

    // The vanishing polynomial v_n is constant on cosets of the same size as the original domain.
    // We only need to compute the unique values, which correspond to the number of cosets.
    let v_d_init = CircleDomain::<F>::standard(log_lde_size)
        .points()
        .take(1 << log_blowup)
        .map(|p| p.v_n(log_lde_size - log_blowup))
        .collect_vec();

    // The unique values are repeated over the rest of the domain in the pattern:
    // 0 1 2 .. n-1 n n n-1 .. 1 0 0 1 ..
    // Look the pattern up through the CFFT permutation instead of materializing
    // and permuting a domain-sized vector.
    let b = 1 << log_blowup;
    let v_d_at = |i: usize| {
        let m = cfft_permute_index(i, log_lde_size) & (2 * b - 1);
        v_d_init[if m < b { m } else { 2 * b - 1 - m }]
    };

    // Compute the squared norm <v_d, v_d> of the vanishing polynomial
    let v_d_2 = F::TWO.exp_u64(log_lde_size as u64 - 1);

    // Extract lambda using the orthogonality property: lambda = <lde, v_d> / <v_d, v_d>
    let lambda = lde
        .par_iter()
        .enumerate()
        .map(|(i, &y)| y * v_d_at(i))
        .sum::<EF>()
        * v_d_2.inverse();

    // Remove the vanishing polynomial component from the LDE evaluations
    let lambda_v_d: Vec<EF> = v_d_init.iter().map(|&v| lambda * v).collect();
    lde.par_iter_mut().enumerate().for_each(|(i, y)| {
        let m = cfft_permute_index(i, log_lde_size) & (2 * b - 1);
        *y -= lambda_v_d[if m < b { m } else { 2 * b - 1 - m }];
    });

    lambda
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::ordering::cfft_permute_slice;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    impl<F: ComplexExtendable, M: Matrix<F>> CircleEvaluations<F, M> {
        /// Compute DEEP quotients for all rows in the matrix efficiently using batch operations.
        ///
        /// This is an optimized version of `deep_quotient_reduce_row` that processes the entire
        /// matrix at once.
        ///
        /// # Mathematical Background
        ///
        /// For each row `i` in the matrix, this computes:
        /// `DEEP_quotient[i] = (f(x[i]) - f(zeta)) / (x[i] - zeta)`
        ///
        /// # Parameters
        ///
        /// - `alpha`: The random challenge scalar
        /// - `zeta`: The random challenge point (outside the original domain)
        /// - `ps_at_zeta`: Polynomial evaluations at challenge point `zeta`
        ///
        /// # Returns
        ///
        /// A vector of DEEP quotient values, one for each row in the matrix.
        pub(crate) fn deep_quotient_reduce<EF: ExtensionField<F>>(
            &self,
            alpha: EF,
            zeta: Point<EF>,
            ps_at_zeta: &[EF],
        ) -> Vec<EF> {
            let points = cfft_permute_slice(&self.domain.points().collect_vec());
            let vp = compute_vanishing_parts(&points, zeta);
            let reduced_rows = self.rowwise_alpha_reduce(alpha);

            let alpha_pow_width = alpha.exp_u64(self.values.width() as u64);
            // sum_j(alpha^j * p_j[zeta]), the same for all rows.
            let reduced_ps_at_zeta: EF = dot_product(alpha.powers(), ps_at_zeta.iter().copied());

            let mut ro = EF::zero_vec(reduced_rows.len());
            accumulate_deep_quotient(
                &mut ro,
                EF::ONE,
                alpha_pow_width,
                &reduced_rows,
                &vp,
                reduced_ps_at_zeta,
            );
            ro
        }
    }

    #[test]
    fn reduce_row_same_as_reduce_matrix() {
        let mut rng = SmallRng::seed_from_u64(1);
        let domain = CircleDomain::standard(5);
        let evals = CircleEvaluations::from_cfft_order(
            domain,
            RowMajorMatrix::<F>::rand(&mut rng, 1 << domain.log_n, 1 << 3),
        );

        let alpha: EF = rng.random();
        let zeta: Point<EF> = Point::from_projective_line(rng.random());
        let ps_at_zeta = evals.evaluate_at_point(zeta);

        let mat_reduced = evals.deep_quotient_reduce(alpha, zeta, &ps_at_zeta);
        let row_reduced = evals
            .to_natural_order()
            .rows()
            .zip(domain.points())
            .map(|(ps_at_x, x)| {
                deep_quotient_reduce_row(alpha, x, zeta, &ps_at_x.collect_vec(), &ps_at_zeta)
                    .unwrap()
            })
            .collect_vec();
        assert_eq!(cfft_permute_slice(&mat_reduced), row_reduced);
    }

    #[test]
    fn reduce_evaluations_low_degree() {
        let mut rng = SmallRng::seed_from_u64(1);
        let log_n = 5;
        let log_blowup = 1;
        let evals = CircleEvaluations::from_cfft_order(
            CircleDomain::standard(log_n),
            RowMajorMatrix::<F>::rand(&mut rng, 1 << log_n, 1 << 3),
        );
        let lde = evals
            .clone()
            .extrapolate(CircleDomain::standard(log_n + log_blowup));
        assert!(lde.dim() <= (1 << log_n));

        let alpha: EF = rng.random();
        let zeta: Point<EF> = Point::from_projective_line(rng.random());

        let ps_at_zeta = evals.evaluate_at_point(zeta);
        let reduced0 = CircleEvaluations::<F>::from_cfft_order(
            CircleDomain::standard(log_n + log_blowup),
            RowMajorMatrix::new_col(lde.deep_quotient_reduce(alpha, zeta, &ps_at_zeta))
                .flatten_to_base(),
        );
        assert!(reduced0.dim() <= (1 << log_n) + 1);

        let not_ps_at_zeta = evals.evaluate_at_point(zeta.double());
        let reduced1 = CircleEvaluations::<F>::from_cfft_order(
            CircleDomain::standard(log_n + log_blowup),
            RowMajorMatrix::new_col(lde.deep_quotient_reduce(alpha, zeta, &not_ps_at_zeta))
                .flatten_to_base(),
        );
        assert!(reduced1.dim() > (1 << log_n) + 1);
    }

    #[test]
    fn reduce_multiple_evaluations() {
        let mut rng = SmallRng::seed_from_u64(1);
        let domain = CircleDomain::standard(5);
        let lde_domain = CircleDomain::standard(8);

        let alpha: EF = rng.random();
        let zeta: Point<EF> = Point::from_projective_line(rng.random());

        let mut alpha_offset = EF::ONE;
        let mut ros = EF::zero_vec(1 << lde_domain.log_n);

        for _ in 0..4 {
            let evals = CircleEvaluations::from_cfft_order(
                domain,
                RowMajorMatrix::<F>::rand(&mut rng, 1 << domain.log_n, 1 << 3),
            );
            let ps_at_zeta = evals.evaluate_at_point(zeta);
            let lde = evals.extrapolate(lde_domain);
            assert!(lde.dim() <= (1 << domain.log_n) + 1);
            let mat_ros = lde.deep_quotient_reduce(alpha, zeta, &ps_at_zeta);
            for (ro, mat_ro) in izip!(&mut ros, mat_ros) {
                *ro += alpha_offset * mat_ro;
            }
            alpha_offset *= alpha.exp_u64(2 * lde.values.width() as u64);
        }

        let ros = CircleEvaluations::from_cfft_order(
            lde_domain,
            RowMajorMatrix::new_col(ros).flatten_to_base(),
        );
        assert!(ros.dim() <= (1 << domain.log_n) + 1);
    }

    /// The reduced column is a linear combination of committed polynomials, so it lies in the
    /// pre-blow-up polynomial space: reducing the trace-size subdomain prefix and lifting the
    /// column back with a CFFT must match reducing the full LDE.
    #[test]
    fn alpha_reduce_lifted_matches_full() {
        let mut rng = SmallRng::seed_from_u64(1);
        for log_n in 2..8 {
            for log_blowup in [1, 2] {
                let lde_domain = CircleDomain::standard(log_n + log_blowup);
                let lde = CircleEvaluations::<F>::from_natural_order(
                    CircleDomain::standard(log_n),
                    RowMajorMatrix::rand(&mut rng, 1 << log_n, 11),
                )
                .extrapolate(lde_domain);

                let alpha: EF = rng.random();
                let full = lde.rowwise_alpha_reduce(alpha);

                let sub_domain = CircleDomain::new(log_n, lde_domain.shift);
                let prefix = lde.values.split_rows(1 << log_n).0;
                let lifted = CircleEvaluations::from_cfft_order(sub_domain, prefix)
                    .rowwise_alpha_reduce_lifted(alpha, lde_domain);
                assert_eq!(full, lifted);
            }
        }
    }

    #[test]
    fn test_extract_lambda() {
        let mut rng = SmallRng::seed_from_u64(1);
        let log_n = 5;
        for log_blowup in [1, 2, 3] {
            let mut coeffs = RowMajorMatrix::<F>::rand(&mut rng, (1 << log_n) + 1, 1);
            coeffs.pad_to_height(1 << (log_n + log_blowup), F::ZERO);

            let domain = CircleDomain::standard(log_n + log_blowup);
            let mut lde = CircleEvaluations::evaluate(domain, coeffs.clone()).values;

            let lambda = extract_lambda(&mut lde.values, log_blowup);
            assert_eq!(lambda, coeffs.get(1 << log_n, 0).unwrap());

            let coeffs2 =
                CircleEvaluations::from_cfft_order(domain, RowMajorMatrix::new_col(lde.values))
                    .interpolate()
                    .values;
            assert_eq!(&coeffs2[..(1 << log_n)], &coeffs.values[..(1 << log_n)]);
            assert_eq!(lambda, coeffs.values[1 << log_n]);
            assert_eq!(coeffs2[1 << log_n], F::ZERO);
            assert_eq!(
                &coeffs2[(1 << log_n) + 1..],
                &coeffs.values[(1 << log_n) + 1..]
            );
        }
    }

    #[test]
    fn reduce_row_rejects_opening_point_on_query_point() {
        // Invariant: the DEEP-quotient denominator is `2 * (1 - (x - zeta).x)`.
        // It vanishes exactly when the opening point `zeta` equals the query point `x`.
        //
        //     x == zeta  ->  denominator 0     ->  None
        //     x != zeta  ->  denominator != 0  ->  Some(quotient)

        // A query point on the circle domain, in the base field.
        let x: Point<F> = Point::from_projective_line(F::from_u8(5));

        // Challenge scalar and dummy column evaluations.
        // Their values never affect whether the denominator vanishes.
        let alpha = EF::from_u8(7);
        let ps_at_x = [F::from_u8(1), F::from_u8(2)];
        let ps_at_zeta = [EF::from_u8(3), EF::from_u8(4)];

        // Lift the query point's coordinates into the extension field.
        // The opening point is now the same point, so `x - zeta` is the group identity.
        let zeta_on_x: Point<EF> = Point::new(EF::from(x.x), EF::from(x.y));
        assert!(deep_quotient_reduce_row(alpha, x, zeta_on_x, &ps_at_x, &ps_at_zeta).is_none());

        // A distinct opening point keeps the denominator nonzero and reduces normally.
        let zeta_off: Point<EF> = Point::from_projective_line(EF::from_u8(9));
        assert!(deep_quotient_reduce_row(alpha, x, zeta_off, &ps_at_x, &ps_at_zeta).is_some());
    }
}
