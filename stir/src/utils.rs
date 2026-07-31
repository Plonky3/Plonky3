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
use p3_maybe_rayon::prelude::*;

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

/// Evaluate a polynomial at a point using a chunk-parallel Horner scheme.
///
/// Splits `poly` into contiguous chunks, evaluates each chunk's local Horner sum in
/// parallel, then combines the per-chunk results with one more Horner pass over powers
/// of `point^chunk_len`:
/// `value = chunk_0 + point^L * (chunk_1 + point^L * (chunk_2 + ...))`.
///
/// Falls back to plain [`eval_poly`] for inputs too small to amortize the parallel
/// dispatch.
pub fn eval_poly_parallel<F: Field>(poly: &[F], point: F) -> F {
    const MIN_PARALLEL_LEN: usize = 4096;
    if poly.len() < MIN_PARALLEL_LEN {
        return eval_poly(poly, point);
    }

    let num_chunks = current_num_threads().max(1);
    let chunk_size = poly.len().div_ceil(num_chunks);
    let point_pow_chunk = point.exp_u64(chunk_size as u64);

    poly.par_chunks(chunk_size)
        .map(|chunk| eval_poly(chunk, point))
        .collect::<Vec<F>>()
        .into_iter()
        .rev()
        .fold(F::ZERO, |acc, chunk_val| acc * point_pow_chunk + chunk_val)
}

/// Divide a coefficient-form polynomial by the linear factor `(X - point)`.
///
/// Returns `(quotient, remainder)` via synthetic (Horner) division.
/// The remainder equals `eval_poly(poly, point)`.
///
/// For a polynomial `p(X)` and point `a`:
/// `p(X) = (X - a) * q(X) + p(a)`
///
/// # Panics
///
/// Panics if `poly` is empty. Callers handle the empty case explicitly upstream
/// (`quotient_by_roots` checks `poly.len() > roots.len()`).
pub fn divide_by_linear<F: Field>(poly: &[F], point: F) -> (Vec<F>, F) {
    assert!(
        !poly.is_empty(),
        "divide_by_linear: cannot divide an empty polynomial by a linear factor"
    );

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
/// `poly` is required to vanish on every root (true by construction inside the STIR prover,
/// where the numerator is `g_i − Ans_i` and `Ans_i` interpolates `g_i` on every root).
///
/// Each per-root remainder is asserted to be zero in **release builds too** — a silently
/// dropped remainder would yield the wrong virtual oracle, and the cost of one field equality
/// per round is negligible.
///
/// Returns the empty vector when the quotient reduces to the zero polynomial (legitimately,
/// when `poly.len() ≤ roots.len()` and `poly` vanishes on all roots: any polynomial of degree
/// `< n` that vanishes on `n` distinct points is identically zero). In that case downstream
/// callers (`multiply_polys`, `codeword_from_coeffs`) treat the empty input as the zero
/// polynomial.
pub fn quotient_by_roots<F: Field>(poly: &[F], roots: &[F]) -> Vec<F> {
    let mut quotient = poly.to_vec();
    for &root in roots {
        if quotient.is_empty() {
            // Already reduced to the zero polynomial; further linear divisions stay zero.
            break;
        }
        let (q, remainder) = divide_by_linear(&quotient, root);
        assert!(
            remainder == F::ZERO,
            "quotient_by_roots: non-zero remainder when dividing by (X - root). \
             The input polynomial does not vanish on every root — this should not happen \
             with an honest STIR prover."
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
///
/// Equivalent to `multiply_polys(poly, &degree_correction_poly(r_comb, gap))`, but computed in
/// `O(poly.len() + gap)` instead of `O(poly.len() * gap)` by exploiting the multiplier's
/// geometric-series structure: `1 + rX + ... + r^gap X^gap = (1 - r^{gap+1} X^{gap+1}) / (1 - rX)`.
///
/// Let `A(X) = poly(X) / (1 - rX)` as a power series, so `a_k = poly[k] + r * a_{k-1}`. Then
/// `poly(X) * (1 + rX + ... + r^gap X^gap) = A(X) - r^{gap+1} X^{gap+1} A(X)`, i.e.
/// `result[k] = a_k - r^{gap+1} * a_{k - gap - 1}`.
pub fn degree_correct<F: Field>(poly: &[F], r_comb: F, gap: usize) -> Vec<F> {
    if poly.is_empty() {
        return vec![];
    }

    let n = poly.len();
    let len = n + gap;

    let mut a = Vec::with_capacity(len);
    let mut acc = F::ZERO;
    for k in 0..len {
        let p_k = poly.get(k).copied().unwrap_or(F::ZERO);
        acc = p_k + r_comb * acc;
        a.push(acc);
    }

    let r_pow = r_comb.exp_u64((gap + 1) as u64);
    let mut result: Vec<F> = (0..len)
        .map(|k| {
            if k > gap {
                a[k] - r_pow * a[k - gap - 1]
            } else {
                a[k]
            }
        })
        .collect();

    while result.last() == Some(&F::ZERO) && result.len() > 1 {
        result.pop();
    }

    result
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

/// Coefficients of the vanishing polynomial `prod_{y in roots} (X - y)`.
///
/// The result is in ascending coefficient order, has length `roots.len() + 1`,
/// and is monic. The empty root set produces the constant polynomial one.
pub fn vanishing_poly_from_roots<F: Field>(roots: &[F]) -> Vec<F> {
    let mut coeffs = Vec::with_capacity(roots.len() + 1);
    coeffs.push(F::ONE);

    for &root in roots {
        coeffs.push(F::ZERO);
        for i in (1..coeffs.len()).rev() {
            coeffs[i] = coeffs[i - 1] - coeffs[i] * root;
        }
        coeffs[0] = -coeffs[0] * root;
    }

    coeffs
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
/// Panics if `points.len() != values.len()` or if any two points are equal
/// (duplicate points yield division by zero).
pub fn interpolate_poly<F: Field>(points: &[F], values: &[F]) -> Vec<F> {
    assert_eq!(
        points.len(),
        values.len(),
        "interpolate_poly: points and values must have the same length"
    );
    let n = points.len();

    for i in 0..n {
        for j in 0..i {
            assert!(
                points[i] != points[j],
                "all interpolation points must be distinct"
            );
        }
    }

    if n == 0 {
        return vec![];
    }

    // Newton's divided differences table. The denominators depend only on the
    // interpolation points, so invert the whole triangular table at once.
    let mut dd = values.to_vec();
    let denominators: Vec<F> = (1..n)
        .flat_map(|k| (k..n).map(move |i| points[i] - points[i - k]))
        .collect();
    let denominator_inverses = batch_multiplicative_inverse(&denominators);
    let mut denominator_offset = 0;
    for k in 1..n {
        let width = n - k;
        let inverses = &denominator_inverses[denominator_offset..denominator_offset + width];
        for i in (k..n).rev() {
            dd[i] = (dd[i] - dd[i - 1]) * inverses[i - k];
        }
        denominator_offset += width;
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
    if points.len() != values.len() {
        return false;
    }

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
/// # Subgroup coordinates are an internal optimization
///
/// Construction 4.5 defines `Fold(f, β)(y)` over **coset** preimages
/// `α·g^{j + l·new_height}` (`α` = domain shift). Interpolating at subgroup
/// coordinates instead (no `α`) lets the fold be decomposed into `log_arity`
/// sequential arity-2 folds (below), which only works cleanly without a coset shift.
///
/// Because the y-values are identical and the x-nodes scale by `α`,
/// `P_coset(X) = P_subgroup(X/α)` exactly for any arity. Callers therefore pass
/// `β = γ/α` to obtain Construction 4.5's coset fold at challenge `γ`; this crate's
/// prover and verifier both do so, so the realized fold matches the paper literally.
///
/// # Binary-pass decomposition
///
/// An arity-`2^k` fold at challenge `beta` equals `k` sequential arity-2 folds at
/// challenges `beta, beta^2, beta^4, …, beta^{2^(k-1)}` (the standard FFT decimation
/// identity underlying FRI-style low-degree tests). Each pass halves the natural-order
/// codeword, pairing index `j` with `j + height/2`: since `g^{height/2} = -1` on the
/// pass's current domain, this pairs conjugate points `x` and `-x`.
pub fn fold_codeword<F: TwoAdicField, EF: ExtensionField<F>>(
    codeword: &[EF],
    beta: EF,
    log_arity: usize,
    log_domain_size: usize,
) -> Vec<EF> {
    let arity = 1 << log_arity;
    let new_height = codeword.len() / arity;
    assert!(new_height > 0);

    let mut data = codeword.to_vec();
    let mut current_beta = beta;
    let mut cur_log_domain = log_domain_size;

    for _ in 0..log_arity {
        let height = data.len() / 2;

        // fold(j) = (lo + hi)/2 + beta * (lo - hi) / (2 * g^j)
        //         = (lo + hi)/2 + (beta/2) * g_inv^j * (lo - hi)
        //
        // g_orig has order `2^cur_log_domain`, so g_inv = g_orig.inverse() has the same
        // order. halve_inv_powers[j] = (1/2) * g_orig^{-j}.
        let g_orig_inv = F::two_adic_generator(cur_log_domain).inverse();
        let halve_inv_powers: Vec<F> = g_orig_inv
            .shifted_powers(F::ONE.halve())
            .take(height)
            .collect();

        data = (0..height)
            .into_par_iter()
            .map(|j| {
                let lo = data[j];
                let hi = data[j + height];
                let hip = EF::from(halve_inv_powers[j]);
                (lo + hi).halve() + (lo - hi) * current_beta * hip
            })
            .collect();

        current_beta = current_beta.square();
        cur_log_domain -= 1;
    }

    debug_assert_eq!(data.len(), new_height);
    data
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
    fn test_eval_poly_parallel_matches_eval_poly() {
        // Exercise chunk boundaries around and past the parallel-dispatch threshold.
        for len in [0, 1, 4095, 4096, 4097, 10_000] {
            let poly: Vec<F> = (0..len).map(|i| F::from_u64(i as u64 + 1)).collect();
            let point = F::from_u64(12345);
            assert_eq!(
                eval_poly_parallel(&poly, point),
                eval_poly(&poly, point),
                "mismatch at len={len}"
            );
        }
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

    #[test]
    fn test_degree_correct_matches_naive_multiply() {
        let r_comb = F::from_u64(7);
        for poly_len in [0, 1, 2, 5, 8] {
            for gap in [0, 1, 3, 6] {
                let poly: Vec<F> = (0..poly_len).map(|i| F::from_u64(i as u64 + 1)).collect();
                let expected = multiply_polys(&poly, &degree_correction_poly(r_comb, gap));
                let actual = degree_correct(&poly, r_comb, gap);
                assert_eq!(
                    actual, expected,
                    "mismatch for poly_len={poly_len}, gap={gap}"
                );
            }
        }
    }

    #[test]
    fn test_vanishing_poly_from_roots() {
        let roots = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let poly = vanishing_poly_from_roots(&roots);

        assert_eq!(poly.len(), roots.len() + 1);
        assert_eq!(poly.last(), Some(&F::ONE));
        for root in roots {
            assert_eq!(eval_poly(&poly, root), F::ZERO);
        }
    }
}
