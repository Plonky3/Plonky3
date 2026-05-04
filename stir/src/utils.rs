//! Polynomial arithmetic utilities for the STIR protocol.
//!
//! Provides the core operations over coefficient-form polynomials needed by the
//! STIR prover and verifier: Horner evaluation, synthetic division, polynomial
//! addition, Newton interpolation, shake polynomial construction, and the
//! verifier-side shake consistency check.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, batch_multiplicative_inverse,
};

/// Evaluate a polynomial at a point using Horner's method.
///
/// `poly` is given in **coefficient form** with `poly[0]` the constant term.
///
/// Returns `poly[0] + poly[1]*point + ... + poly[n]*point^n`.
pub fn eval_poly<F: Field>(poly: &[F], point: F) -> F {
    poly.iter()
        .rev()
        .fold(F::ZERO, |acc, &coeff| acc * point + coeff)
}

/// Divide a coefficient-form polynomial by the linear factor `(X - point)`.
///
/// Returns `(quotient, remainder)` via synthetic (Horner) division.
/// The remainder equals `eval_poly(poly, point)`.
///
/// For a polynomial `p(X)` and point `a`:
/// `p(X) = (X - a) * q(X) + p(a)`
pub fn divide_by_linear<F: Field>(poly: &[F], point: F) -> (Vec<F>, F) {
    if poly.is_empty() {
        return (vec![], F::ZERO);
    }

    let n = poly.len();
    let mut quotient = vec![F::ZERO; n - 1];
    let mut carry = F::ZERO;

    // Process from highest degree to lowest (Horner-style).
    for i in (0..n - 1).rev() {
        carry = poly[i + 1] + carry * point;
        quotient[i] = carry;
    }

    let remainder = poly[0] + carry * point;
    (quotient, remainder)
}

/// Add two coefficient-form polynomials.
///
/// The result has degree `max(deg(a), deg(b))`.
pub fn add_polys<F: PrimeCharacteristicRing + Copy>(a: &[F], b: &[F]) -> Vec<F> {
    let len = a.len().max(b.len());
    let mut result = vec![F::ZERO; len];
    for (i, r) in result.iter_mut().enumerate() {
        let ai = if i < a.len() { a[i] } else { F::ZERO };
        let bi = if i < b.len() { b[i] } else { F::ZERO };
        *r = ai + bi;
    }
    result
}

/// Scale every coefficient of a polynomial by `scalar`.
pub fn scale_poly<F: Field>(poly: &[F], scalar: F) -> Vec<F> {
    poly.iter().map(|&c| c * scalar).collect()
}

/// Multiply two coefficient-form polynomials.
pub fn multiply_polys<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let mut result = vec![F::ZERO; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }

    while result.last() == Some(&F::ZERO) && result.len() > 1 {
        result.pop();
    }

    result
}

/// Divide `poly` by the vanishing polynomial over `roots`.
///
/// Panics in debug builds if the division is not exact.
pub fn quotient_by_roots<F: Field>(poly: &[F], roots: &[F]) -> Vec<F> {
    let mut quotient = poly.to_vec();
    for &root in roots {
        let (q, remainder) = divide_by_linear(&quotient, root);
        debug_assert!(
            remainder == F::ZERO,
            "quotient_by_roots: non-zero remainder when dividing by (X - root)"
        );
        quotient = q;
    }
    quotient
}

/// Return the geometric-sum polynomial `sum_{ell=0}^{gap} (r_comb X)^ell`.
pub fn degree_correction_poly<F: Field>(r_comb: F, gap: usize) -> Vec<F> {
    let mut coeffs = Vec::with_capacity(gap + 1);
    let mut power = F::ONE;
    for _ in 0..=gap {
        coeffs.push(power);
        power *= r_comb;
    }
    coeffs
}

/// Apply `DegCor(d*, r_comb, f, d)` with `gap = d* - d`.
pub fn degree_correct<F: Field>(poly: &[F], r_comb: F, gap: usize) -> Vec<F> {
    multiply_polys(poly, &degree_correction_poly(r_comb, gap))
}

/// Evaluate the degree-correction factor at a point and multiply it into `value`.
pub fn eval_degree_correction<F: Field>(value: F, point: F, r_comb: F, gap: usize) -> F {
    let step = point * r_comb;
    let geom = if step == F::ONE {
        F::from_usize(gap + 1)
    } else {
        (F::ONE - step.exp_u64((gap + 1) as u64)) * (F::ONE - step).inverse()
    };
    value * geom
}

/// Evaluate the vanishing polynomial `prod_{y in roots} (point - y)` at `point`.
pub fn eval_vanishing_at_roots<F: Field>(roots: &[F], point: F) -> F {
    roots.iter().fold(F::ONE, |acc, &root| acc * (point - root))
}

/// Shift for the next committed domain.
///
/// The folded polynomial naturally lives on `current_shift^(2^log_arity) * H`. We commit it on
/// a disjoint coset of the next-round domain by multiplying that natural shift by the field's
/// multiplicative generator.
pub fn next_domain_shift<F: Field>(current_shift: F, log_arity: usize) -> F {
    current_shift.exp_power_of_2(log_arity) * F::GENERATOR
}

/// Compute the shake polynomial for a set of evaluation points.
///
/// Given an answer polynomial `ans(X)` and the set `P = {y_1, ..., y_m}` of evaluation
/// points where `ans(y_i)` is known, the **shake polynomial** is defined as:
///
/// ```text
/// S(X) = sum_{y in P} (ans(X) - ans(y)) / (X - y)
/// ```
///
/// Each term `(ans(X) - ans(y)) / (X - y)` is the result of synthetic division of
/// `ans(X) - ans(y)` (a polynomial with a known root at `y`) by `(X - y)`.
///
/// The shake polynomial enables the verifier to check that `ans` correctly interpolates
/// all `(y_i, ans(y_i))` pairs without recomputing a full Lagrange interpolation.
pub fn compute_shake_polynomial<F: Field>(ans: &[F], points: &[F]) -> Vec<F> {
    points
        .iter()
        .map(|&y| divide_by_linear(ans, y).0)
        .fold(vec![], |acc, q| add_polys(&acc, &q))
}

/// Interpolate a polynomial through the given `(points, values)` pairs.
///
/// Uses Newton's divided-difference method.
/// Returns coefficients of a polynomial of degree `< points.len()` passing through
/// all `(points[i], values[i])` pairs.
///
/// # Panics
///
/// Panics if `points.len() != values.len()`. In debug builds, also panics if any two
/// points are equal (duplicate points yield division by zero).
pub fn interpolate_poly<F: Field>(points: &[F], values: &[F]) -> Vec<F> {
    assert_eq!(
        points.len(),
        values.len(),
        "interpolate_poly: points and values must have the same length"
    );
    let n = points.len();

    #[cfg(debug_assertions)]
    {
        for i in 0..n {
            for j in 0..i {
                assert!(
                    points[i] != points[j],
                    "all interpolation points must be distinct"
                );
            }
        }
    }

    if n == 0 {
        return vec![];
    }

    // Newton's divided differences table.
    let mut dd = values.to_vec();
    for k in 1..n {
        for i in (k..n).rev() {
            let denom = points[i] - points[i - k];
            dd[i] = (dd[i] - dd[i - 1]) * denom.inverse();
        }
    }

    // Evaluate Newton's forward difference expansion into coefficient form.
    // poly(x) = dd[0] + dd[1]*(x-p[0]) + dd[2]*(x-p[0])*(x-p[1]) + ...
    // We expand this iteratively.
    let mut coeffs = vec![F::ZERO; n];
    // Build coefficient form by Horner: accumulate from the highest term.
    // basis[k] = prod_{j<k} (X - points[j]) in coefficient form.
    let mut basis: Vec<F> = vec![F::ONE];
    coeffs[0] = dd[0];

    for k in 1..n {
        // Multiply current basis by (X - points[k-1]).
        let pk = points[k - 1];
        let old_len = basis.len();
        let mut new_basis = vec![F::ZERO; old_len + 1];
        for (i, &b) in basis.iter().enumerate() {
            new_basis[i + 1] += b;
            new_basis[i] -= b * pk;
        }
        basis = new_basis;

        // Add dd[k] * basis to coeffs.
        for (i, &b) in basis.iter().enumerate() {
            coeffs[i] += dd[k] * b;
        }
    }

    // Remove trailing zeros.
    while coeffs.last() == Some(&F::ZERO) && coeffs.len() > 1 {
        coeffs.pop();
    }

    coeffs
}

/// Verify shake polynomial consistency at a random point `rho`.
///
/// Checks that `S(rho) == sum_{y in P} (ans(rho) - val_y) / (rho - y)` using batch inversion.
///
/// Returns `true` if the check passes.
pub fn check_shake_consistency<F: Field>(
    ans: &[F],
    shake: &[F],
    points: &[F],
    values: &[F],
    rho: F,
) -> bool {
    debug_assert_eq!(points.len(), values.len());

    // If rho coincides with one of the evaluation points the denominator (rho - y_i) would be
    // zero.  This is negligible for a random rho but we handle it defensively: the shake identity
    // does not apply at such a degenerate rho, so we report failure.
    if points.contains(&rho) {
        return false;
    }

    let ans_rho = eval_poly(ans, rho);
    let shake_rho = eval_poly(shake, rho);

    // Compute (rho - y_i) for all i and batch-invert.
    let diffs: Vec<F> = points.iter().map(|&y| rho - y).collect();
    let diff_invs = batch_multiplicative_inverse(&diffs);

    // sum_i (ans(rho) - val_i) / (rho - y_i)
    let expected: F = values
        .iter()
        .zip(diff_invs.iter())
        .map(|(&val, &inv)| (ans_rho - val) * inv)
        .sum();

    shake_rho == expected
}

/// Fold an entire natural-order codeword of size `N` by arity `k = 2^log_arity`.
///
/// The codeword is on domain `shift * <g>` where `g = two_adic_generator(log_domain_size)`.
/// For each new-domain index `j` (in `0..new_height` where `new_height = N/k`), the fiber is:
///
/// ```text
/// fiber[j] = [ codeword[j],  codeword[j + new_height],  …,  codeword[j + (k-1)*new_height] ]
/// ```
///
/// The fold at challenge `beta` is the Lagrange interpolation of `(x_l, fiber[j][l])`
/// evaluated at `beta`, using **subgroup** x-coordinates:
///
/// ```text
/// x_l = g^{j + l * new_height}   (l = 0, …, k-1)
/// ```
///
/// (The coset shift is NOT included in the x-coordinates — consistent with the FRI convention.)
///
/// # Optimisation for arity 2
///
/// When `log_arity == 1`, the Lagrange formula simplifies to:
///
/// ```text
/// fold(j) = (lo + hi) / 2 + beta * (lo - hi) / (2 * g^j)
/// ```
///
/// which is computed via `halve_inv_powers[j] = (1/2) * g^{-j}`.
pub fn fold_codeword<F: TwoAdicField, EF: ExtensionField<F>>(
    codeword: &[EF],
    beta: EF,
    log_arity: usize,
    log_domain_size: usize,
) -> Vec<EF> {
    let arity = 1 << log_arity;
    let new_height = codeword.len() / arity;
    assert!(new_height > 0);

    if log_arity == 1 {
        // Arity-2: x_j = g^j, x_{j+new_height} = -g^j  (since g^{new_height} = -1).
        // fold(j) = (lo + hi)/2 + beta * (lo - hi) / (2 * g^j)
        //         = (lo + hi)/2 + (beta/2) * g_inv^j * (lo - hi)
        //
        // g_orig has order `domain_size`, so g_inv = g_orig.inverse() has the same order.
        // halve_inv_powers[j] = (1/2) * g_orig^{-j}
        let g_orig_inv = F::two_adic_generator(log_domain_size).inverse();
        let halve_inv_powers: Vec<F> = g_orig_inv
            .shifted_powers(F::ONE.halve())
            .take(new_height)
            .collect();

        (0..new_height)
            .map(|j| {
                let lo = codeword[j];
                let hi = codeword[j + new_height];
                let hip = EF::from(halve_inv_powers[j]);
                (lo + hi).halve() + (lo - hi) * beta * hip
            })
            .collect()
    } else {
        // General arity k = 2^log_arity: evaluate Lagrange interpolation at `beta`
        // through all k fiber values for each new-domain index j.
        //
        // x-coordinates for new-domain index j:
        //   x_l = g^{j + l * new_height}  for l = 0..k-1
        // where g = two_adic_generator(log_domain_size).
        //
        // Key identity: x_l^{(j)} = g^j * x_l^{(0)} for all l. The barycentric weight
        // for node (j, l) satisfies w_l^{(j)} = g^{-j*(k-1)} * w_l^{(0)}. Since this
        // factor is common to numerator and denominator, it cancels in the ratio. Thus
        // the weights from j=0 (xs_0 = [1, step, step^2, ...]) can be reused for all j,
        // and only the x-coordinates need to be rescaled by g^j.
        let g = F::two_adic_generator(log_domain_size);
        let step = g.exp_u64(new_height as u64); // arity-th root of unity

        // Precompute base x-coordinates xs_0[l] = step^l (for j = 0).
        let xs_0: Vec<F> = {
            let mut acc = F::ONE;
            (0..arity)
                .map(|_| {
                    let v = acc;
                    acc *= step;
                    v
                })
                .collect()
        };

        // Precompute barycentric weights for xs_0 (reused for all j after cancellation).
        let barycentric_weights: Vec<F> = {
            let mut w = vec![F::ONE; arity];
            for i in 0..arity {
                for j in 0..arity {
                    if i != j {
                        w[i] *= xs_0[i] - xs_0[j];
                    }
                }
            }
            batch_multiplicative_inverse(&w)
        };

        // Precompute g^j for j = 0..new_height.
        let g_powers: Vec<F> = {
            let mut acc = F::ONE;
            (0..new_height)
                .map(|_| {
                    let v = acc;
                    acc *= g;
                    v
                })
                .collect()
        };

        (0..new_height)
            .map(|j| {
                let gj = g_powers[j];
                // xs_j[l] = gj * xs_0[l]
                let ys_j: Vec<EF> = (0..arity).map(|l| codeword[j + l * new_height]).collect();

                // Check if beta equals one of the xs_j nodes (exact match).
                for l in 0..arity {
                    if beta == EF::from(gj * xs_0[l]) {
                        return ys_j[l];
                    }
                }

                // Barycentric Lagrange eval: use precomputed weights (cancellation of g^{j*(k-1)})
                // num = sum_l w_l * ys_j[l] / (beta - gj * xs_0[l])
                // den = sum_l w_l / (beta - gj * xs_0[l])
                let mut num = EF::ZERO;
                let mut den = EF::ZERO;
                for l in 0..arity {
                    let diff = beta - EF::from(gj * xs_0[l]);
                    let term = EF::from(barycentric_weights[l]) / diff;
                    num += term * ys_j[l];
                    den += term;
                }
                num / den
            })
            .collect()
    }
}

/// Compute the expected folded value for a single fiber (used by the verifier).
///
/// Given:
/// - `fiber`: `k = 2^log_arity` evaluations `f(shift * g^{j + l*new_height})` for `l=0..k-1`
/// - `j`: index of the new-domain query point
/// - `log_new_height`: log2 of the new domain size  
/// - `log_arity`: log2 of the folding arity
/// - `beta`: folding challenge
///
/// Returns the expected value of `f_new` at the j-th new-domain point using
/// the same **subgroup** x-coordinate convention as [`fold_codeword`].
pub fn fold_fiber<F: TwoAdicField, EF: ExtensionField<F>>(
    fiber: &[EF],
    j: usize,
    log_new_height: usize,
    log_arity: usize,
    beta: EF,
) -> EF {
    let arity = 1 << log_arity;
    assert_eq!(fiber.len(), arity);

    let new_height = 1 << log_new_height;
    let log_domain_size = log_new_height + log_arity;

    // Build subgroup x-coordinates: g^j, g^{j+new_height}, ..., g^{j+(arity-1)*new_height}
    // where g = two_adic_generator(log_domain_size).
    let g = F::two_adic_generator(log_domain_size);
    let x0 = g.exp_u64(j as u64);
    let step = g.exp_u64(new_height as u64); // = zeta = arity-th root of unity
    let xs: Vec<F> = step.shifted_powers(x0).take(arity).collect();

    lagrange_eval_at(&xs, fiber, beta)
}

/// Evaluate the Lagrange interpolating polynomial through `(xs[i], ys[i])` at `point`.
///
/// Uses the barycentric form. Both `xs` and `ys` must have the same length.
/// If `point` equals one of the `xs[i]` exactly, returns `ys[i]` directly.
pub fn lagrange_eval_at<F: Field, EF: ExtensionField<F>>(xs: &[F], ys: &[EF], point: EF) -> EF {
    let n = xs.len();
    assert_eq!(ys.len(), n);

    // Short-circuit: if point coincides with one of the nodes, return the known value directly.
    for i in 0..n {
        if point == EF::from(xs[i]) {
            return ys[i];
        }
    }

    // Barycentric weights w_i = 1 / prod_{j != i} (x_i - x_j).
    let mut weights = vec![F::ONE; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                weights[i] *= xs[i] - xs[j];
            }
        }
    }
    let weights = batch_multiplicative_inverse(&weights);

    // L(point) = [sum_i w_i * y_i / (point - x_i)] / [sum_i w_i / (point - x_i)]
    let diffs: Vec<EF> = xs.iter().map(|&x| point - EF::from(x)).collect();

    let mut num = EF::ZERO;
    let mut den = EF::ZERO;
    for i in 0..n {
        let term = EF::from(weights[i]) / diffs[i];
        num += term * ys[i];
        den += term;
    }
    num / den
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_eval_poly_zero() {
        let poly: Vec<F> = vec![];
        assert_eq!(eval_poly(&poly, F::from_u64(3)), F::ZERO);
    }

    #[test]
    fn test_eval_poly_constant() {
        let poly = vec![F::from_u64(7)];
        assert_eq!(eval_poly(&poly, F::from_u64(100)), F::from_u64(7));
    }

    #[test]
    fn test_eval_poly_linear() {
        // poly(x) = 2 + 3x
        let poly = vec![F::from_u64(2), F::from_u64(3)];
        assert_eq!(eval_poly(&poly, F::from_u64(4)), F::from_u64(14));
    }

    #[test]
    fn test_divide_by_linear_exact_root() {
        // p(x) = x^2 - 1 = (x-1)(x+1); divide by (x - 1)
        // coefficients: [-1, 0, 1]
        let one = F::ONE;
        let neg_one = -one;
        let poly = vec![neg_one, F::ZERO, one];
        let (q, r) = divide_by_linear(&poly, one);
        assert_eq!(r, F::ZERO, "remainder should be 0 for a root");
        // quotient should be x + 1: [1, 1]
        assert_eq!(q, vec![neg_one + one + one, one]); // == [1, 1]
    }

    #[test]
    fn test_divide_by_linear_remainder() {
        // p(x) = x^2 + 2x + 1; divide by (x - 2) -> remainder = p(2) = 9
        let poly = vec![F::ONE, F::from_u64(2), F::ONE];
        let (_, r) = divide_by_linear(&poly, F::from_u64(2));
        assert_eq!(r, eval_poly(&poly, F::from_u64(2)));
    }

    #[test]
    fn test_add_polys() {
        let a = vec![F::ONE, F::from_u64(2)];
        let b = vec![F::from_u64(3), F::ZERO, F::from_u64(5)];
        let c = add_polys(&a, &b);
        assert_eq!(c, vec![F::from_u64(4), F::from_u64(2), F::from_u64(5)]);
    }

    #[test]
    fn test_interpolate_poly_degree_1() {
        // Through (1, 3) and (2, 5): y = 2x + 1
        let pts = vec![F::ONE, F::from_u64(2)];
        let vals = vec![F::from_u64(3), F::from_u64(5)];
        let poly = interpolate_poly(&pts, &vals);
        assert_eq!(eval_poly(&poly, F::ONE), F::from_u64(3));
        assert_eq!(eval_poly(&poly, F::from_u64(2)), F::from_u64(5));
        // Eval at a third point: y(3) = 7
        assert_eq!(eval_poly(&poly, F::from_u64(3)), F::from_u64(7));
    }

    #[test]
    fn test_shake_polynomial_consistency() {
        // Build ans poly through some points, compute shake, verify consistency.
        let pts = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let vals = vec![F::from_u64(4), F::from_u64(9), F::from_u64(16)];

        let ans = interpolate_poly(&pts, &vals);
        let shake = compute_shake_polynomial(&ans, &pts);

        // Check at a random point (use a field element not in pts).
        let rho = F::from_u64(7);
        assert!(
            check_shake_consistency(&ans, &shake, &pts, &vals, rho),
            "shake consistency should pass"
        );
    }

    #[test]
    fn test_shake_consistency_fails_on_wrong_ans() {
        let pts = vec![F::from_u64(1), F::from_u64(2)];
        let vals = vec![F::from_u64(4), F::from_u64(9)];

        let ans = interpolate_poly(&pts, &vals);
        let shake = compute_shake_polynomial(&ans, &pts);

        // Corrupt ans.
        let mut bad_ans = ans;
        bad_ans[0] += F::ONE;

        let rho = F::from_u64(5);
        assert!(
            !check_shake_consistency(&bad_ans, &shake, &pts, &vals, rho),
            "shake consistency should fail on bad ans"
        );
    }

    #[test]
    fn test_lagrange_eval_at() {
        // Interpolate y = x^2 through (1,1),(2,4),(3,9) and eval at 4 -> 16.
        let xs: Vec<F> = vec![F::ONE, F::from_u64(2), F::from_u64(3)];
        let ys: Vec<EF> = vec![
            EF::from(F::ONE),
            EF::from(F::from_u64(4)),
            EF::from(F::from_u64(9)),
        ];
        let result = lagrange_eval_at(&xs, &ys, EF::from(F::from_u64(4)));
        assert_eq!(result, EF::from(F::from_u64(16)));
    }
}
