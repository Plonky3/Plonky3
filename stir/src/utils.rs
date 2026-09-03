//! Polynomial arithmetic utilities for the STIR protocol.
//!
//! Provides the core operations over coefficient-form polynomials needed by the
//! STIR prover and verifier: Horner evaluation, synthetic division, polynomial
//! addition, Newton interpolation, and the verifier-side check that the prover's
//! answer polynomial interpolates the round's claimed values.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, batch_multiplicative_inverse,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

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
/// Panics if `poly` is empty.
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

/// Evaluate the batch-degree-correction `Combine` (§4.5, Definition 4.11) at every point of
/// the coset `shift · ⟨g⟩` of size `2^log_domain`:
/// `Combine(x) = Σᵢ rᵢ · fᵢ(x) · GeomSum(r_comb·x, gapᵢ)`.
///
/// `groups[i] = (r_i, gap_i, values_i)`: `r_i` is Definition 4.11's own per-group shifting
/// coefficient (distinct from `r_comb`, the random point the geometric sum runs in), `gap_i =
/// d* − dᵢ` is a degree count (not log), and `values_i[p]` is group `i`'s reduced-opening
/// evaluated at `shift · g^p`, `p` in the same natural order as the returned codeword.
///
/// The geometric sum's numerator and denominator both advance by a fixed ratio (a power of
/// `g`) across the coset, so each is seeded once per `POWER_CHUNK`-sized chunk (one
/// exponentiation) and then advanced by a base-field multiply — mirroring the
/// degree-correction sweep in `RoundProver::finish`. The `(1 − r_comb·x)` denominators do not
/// depend on the group, so they are swept, inverted, and applied once for all of `groups`
/// rather than once per group.
pub fn combine_on_coset<F, EF>(
    groups: &[(EF, usize, &[EF])],
    r_comb: EF,
    shift: F,
    log_domain: usize,
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let n = 1usize << log_domain;
    let g = F::two_adic_generator(log_domain);
    let step_start = r_comb * shift;

    const POWER_CHUNK: usize = 1 << 12;
    let mut result = EF::zero_vec(n);

    let mut denoms = EF::zero_vec(n);
    denoms
        .par_chunks_mut(POWER_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut step = step_start * g.exp_u64((chunk_idx * POWER_CHUNK) as u64);
            for d in chunk.iter_mut() {
                *d = EF::ONE - step;
                step *= g;
            }
        });

    // `p ↦ r_comb·shift·g^p` is injective over the coset, so at most one lane can reach
    // `step = 1`, where the geometric sum degenerates to `gap + 1` and the denominator
    // vanishes. Substituting a unit keeps the batch inversion defined and makes that lane's
    // numerator `1 − step^(gap+1) = 0`, so the sweep below contributes nothing there and the
    // closed form is added back afterwards — matching `eval_degree_correction` pointwise.
    let degenerate = denoms.iter().position(|d| d.is_zero());
    if let Some(p) = degenerate {
        denoms[p] = EF::ONE;
    }
    let inv_denoms = batch_multiplicative_inverse(&denoms);
    drop(denoms);

    for &(r_i, gap, values) in groups {
        assert_eq!(values.len(), n, "group values must span the full coset");

        // `r_i · (1 − step^(gap+1))` is swept directly, folding the group's shifting
        // coefficient into the geometric numerator so that each point costs one product with
        // its value rather than a separate multiplication by `r_i` and by the denominator.
        let g_hi = g.exp_u64((gap + 1) as u64);
        let r_step_start_hi = r_i * step_start.exp_u64((gap + 1) as u64);
        result
            .par_chunks_mut(POWER_CHUNK)
            .zip(values.par_chunks(POWER_CHUNK))
            .enumerate()
            .for_each(|(chunk_idx, (res_chunk, val_chunk))| {
                let mut r_step_hi =
                    r_step_start_hi * g_hi.exp_u64((chunk_idx * POWER_CHUNK) as u64);
                for (res, &val) in res_chunk.iter_mut().zip(val_chunk) {
                    *res += val * (r_i - r_step_hi);
                    r_step_hi *= g_hi;
                }
            });
    }

    result
        .par_iter_mut()
        .zip(inv_denoms.par_iter())
        .for_each(|(res, &inv_denom)| *res *= inv_denom);

    if let Some(p) = degenerate {
        for &(r_i, gap, values) in groups {
            result[p] += r_i * values[p] * EF::from_usize(gap + 1);
        }
    }
    result
}

/// Horner evaluation of an extension-coefficient polynomial at a **base-field** point.
///
/// Identical to [`eval_poly`] on a lifted point, but each step is an extension-by-base
/// product rather than a full extension multiplication.
pub fn eval_poly_at_base<F: Field, EF: ExtensionField<F>>(poly: &[EF], point: F) -> EF {
    poly.iter()
        .rev()
        .fold(EF::ZERO, |acc, &coeff| acc * point + coeff)
}

/// Reduce `poly` modulo `X^n - c`, returning the `n` remainder coefficients.
///
/// Writing `i = q*n + r` gives `X^i = (X^n)^q * X^r ≡ c^q * X^r`, so every coefficient folds
/// onto its index mod `n` scaled by a power of `c`: one pass over `poly`, independent of `n`.
///
/// Evaluating the reduced polynomial agrees with the original at any point `x` with
/// `x^n = c`, which for a coset of the `n`-th roots of unity is every point at once.
pub fn reduce_mod_x_pow_minus_c<F: Field, EF: ExtensionField<F>>(
    poly: &[EF],
    n: usize,
    c: F,
) -> Vec<EF> {
    let mut remainder = EF::zero_vec(n);
    let mut c_pow = F::ONE;
    for block in poly.chunks(n) {
        for (slot, &coeff) in remainder.iter_mut().zip(block) {
            *slot += coeff * c_pow;
        }
        c_pow *= c;
    }
    remainder
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

/// Domain `(log_size, shift)` that a fold of arity `2^log_arity` produces before any
/// re-commitment shift is applied: `current_shift^(2^log_arity) * H`, `|H| = 2^(log_domain -
/// log_arity)`.
pub fn fold_domain_params<F: Field>(
    current_shift: F,
    current_log_domain: usize,
    log_arity: usize,
) -> (usize, F) {
    (
        current_log_domain - log_arity,
        current_shift.exp_power_of_2(log_arity),
    )
}

/// Sample `num_ood_samples` distinct out-of-domain points for a STIR round from the
/// transcript.
///
/// `excluded_domains` gives the `(shift, log_size)` of the current, next, and fold-query
/// domains for the round: each candidate is drawn until it lies outside all three cosets,
/// so that it cannot collide with the interpolation nodes used elsewhere in the round.
/// Prover and verifier both call this to derive identical points from identical
/// transcript state.
pub fn sample_ood_points<F, EF, Challenger>(
    challenger: &mut Challenger,
    excluded_domains: [(F, usize); 3],
    num_ood_samples: usize,
) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    // Nothing to sample, and nothing to precompute for it: keeps the contract identical to
    // evaluating the predicate lazily, which did no inversions at all in this case.
    if num_ood_samples == 0 {
        return Vec::new();
    }

    // `shift^{-2^log_size}` per excluded domain, computed once in the base field (an
    // inversion and an exponentiation cheaper than their extension-field counterparts) and
    // reused for every candidate below, rather than an extension-field inversion redone once
    // per domain per candidate.
    let shift_inv_pows: [F; 3] =
        excluded_domains.map(|(shift, log_size)| shift.inverse().exp_power_of_2(log_size));
    let max_log_size = excluded_domains
        .iter()
        .map(|&(_, log_size)| log_size)
        .max()
        .expect("three excluded domains");

    let mut ood_points: Vec<EF> = Vec::with_capacity(num_ood_samples);
    while ood_points.len() < num_ood_samples {
        let z: EF = challenger.sample_algebra_element();

        // One doubling chain `z, z^2, z^4, ..., z^(2^max_log_size)` walked once, recording
        // `z^(2^log_size)` for each excluded domain as it passes — instead of a fresh
        // `exp_power_of_2` call (its own squaring chain from scratch) per domain. There are
        // exactly three domains, so the recording slots are a fixed-size array.
        let mut z_pows = [EF::ONE; 3];
        let mut acc = z;
        for l in 0..=max_log_size {
            for (slot, &(_, log_size)) in z_pows.iter_mut().zip(&excluded_domains) {
                if log_size == l {
                    *slot = acc;
                }
            }
            if l < max_log_size {
                acc = acc.square();
            }
        }

        let outside_all_domains = excluded_domains
            .iter()
            .zip(&shift_inv_pows)
            .zip(&z_pows)
            .all(|((&(_, log_size), &shift_inv_pow), &z_pow)| {
                log_size == 0 || z_pow * shift_inv_pow != EF::ONE
            });
        // Deduplicate OOD points.
        let not_dup = ood_points.iter().all(|&existing| existing != z);
        if outside_all_domains && not_dup {
            ood_points.push(z);
        }
    }
    ood_points
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

/// Verify at a random point `rho` that `ans` interpolates `(points, values)`.
///
/// `ans` is compared against the barycentric form of the unique polynomial `I` of degree
/// `< n` through the `n` pairs `(y_i, v_i)`:
///
/// ```text
/// I(rho) = (sum_i w_i v_i / (rho - y_i)) / (sum_i w_i / (rho - y_i)),
/// w_i = 1 / prod_{j != i} (y_i - y_j)
/// ```
///
/// The identity `ans(rho) == I(rho)` is checked cross-multiplied, so the barycentric
/// denominator is never inverted. That denominator equals `1 / prod_i (rho - y_i)` — the
/// Lagrange basis sums to one — and so is non-zero for every `rho` outside the node set,
/// which is exactly what the rejection below enforces.
///
/// Returns `true` if the check passes. Both `ans` (by the caller's length bound) and `I` have
/// degree `< n`, so a caller that binds `ans` and the node set into the transcript before
/// drawing `rho` gets soundness error at most `(n - 1) / |F|`.
pub fn check_ans_interpolates<F: Field>(ans: &[F], points: &[F], values: &[F], rho: F) -> bool {
    if points.len() != values.len() {
        return false;
    }

    // At an interpolation node the barycentric denominators vanish and the identity says
    // nothing, so report failure. A `rho` drawn after `ans` is bound lands there only with
    // negligible probability.
    if points.contains(&rho) {
        return false;
    }

    let n = points.len();
    // The barycentric weight denominators `prod_{j != i} (y_i - y_j)` followed by the node
    // distances `rho - y_i`, inverted in a single batch.
    let mut denominators: Vec<F> = points
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            points
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &z)| y - z)
                .product()
        })
        .collect();
    denominators.extend(points.iter().map(|&y| rho - y));
    // A vanishing weight denominator means two nodes coincide, so no interpolant exists.
    if denominators[..n].contains(&F::ZERO) {
        return false;
    }
    let inverses = batch_multiplicative_inverse(&denominators);
    let (weights, diff_invs) = inverses.split_at(n);

    let mut numerator = F::ZERO;
    let mut denominator = F::ZERO;
    for ((&w, &diff_inv), &v) in weights.iter().zip(diff_invs).zip(values) {
        let term = w * diff_inv;
        numerator += term * v;
        denominator += term;
    }

    eval_poly(ans, rho) * denominator == numerator
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

    if log_arity == 0 {
        return codeword.to_vec();
    }

    let mut current_beta = beta;
    let mut cur_log_domain = log_domain_size;
    // The first pass reads the caller's slice and every later pass reads its predecessor's
    // output, so the input is never copied.
    let mut data = fold_pass::<F, EF>(codeword, current_beta, cur_log_domain);
    for _ in 1..log_arity {
        current_beta = current_beta.square();
        cur_log_domain -= 1;
        data = fold_pass::<F, EF>(&data, current_beta, cur_log_domain);
    }

    debug_assert_eq!(data.len(), new_height);
    data
}

/// One arity-2 pass of [`fold_codeword`] over a natural-order codeword on a domain of size
/// `2^log_domain_size`, pairing index `j` with `j + height`.
fn fold_pass<F: TwoAdicField, EF: ExtensionField<F>>(
    src: &[EF],
    beta: EF,
    log_domain_size: usize,
) -> Vec<EF> {
    let height = src.len() / 2;

    // fold(j) = (lo + hi)/2 + beta * (lo - hi) / (2 * g^j)
    //         = (lo + hi)/2 + (beta/2) * g_inv^j * (lo - hi)
    //
    // g_orig has order `2^log_domain_size`, so g_inv = g_orig.inverse() has the same
    // order. halve_inv_powers[j] = (1/2) * g_orig^{-j}.
    let g_orig_inv = F::two_adic_generator(log_domain_size).inverse();
    let halve_inv_powers: Vec<F> = g_orig_inv
        .shifted_powers(F::ONE.halve())
        .take(height)
        .collect();

    (0..height)
        .into_par_iter()
        .map(|j| {
            let lo = src[j];
            let hi = src[j + height];
            // The base-field twiddle multiplies first so that it dispatches through the cheap
            // extension-by-base product, leaving `beta` as the pass's only extension product.
            (lo + hi).halve() + (lo - hi) * halve_inv_powers[j] * beta
        })
        .collect()
}

/// Fold a coefficient-form polynomial at challenge `gamma` with arity `k = 2^log_arity`.
///
/// Writing `f(X) = Σ_{l<k} X^l · q_l(X^k)`, Construction 4.5's fold is `Σ_{l<k} gamma^l · q_l`,
/// whose degree-`m` coefficient is `Σ_{l<k} gamma^l · f[l + k·m]` — the Horner evaluation at
/// `gamma` of the `m`-th length-`k` block of `f`.
///
/// Unlike [`fold_codeword`], which interpolates at subgroup coordinates and therefore takes the
/// rescaled challenge `gamma / shift`, this form carries no domain shift and takes `gamma`
/// itself.
pub fn fold_poly_coeffs<F: Field>(coeffs: &[F], gamma: F, log_arity: usize) -> Vec<F> {
    coeffs
        .par_chunks(1 << log_arity)
        .map(|block| eval_poly(block, gamma))
        .collect()
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

    lagrange_interpolate_at(&xs, fiber, beta)
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

/// Evaluate, at `point`, the degree-`< n` polynomial through `(xs[i], ys[i])`, where `xs` is
/// a coset of the `n`-th roots of unity (`n` a power of two).
///
/// On a full root-of-unity coset the barycentric weights collapse to a closed form
/// `w_i = x_i / (n * xs[0]^n)`, so the barycentric denominator `sum_i w_i / (point - x_i)`
/// simplifies to `prod_i (point - x_i)` and needs no separate accumulation. One batch
/// inversion covers every `(point - x_i)`, against [`lagrange_eval_at`]'s `O(n^2)` weight
/// construction plus `n` separate divisions.
///
/// # Panics
///
/// Panics if `xs.len() != ys.len()`, or if `xs.len()` is not a power of two.
pub fn lagrange_interpolate_at<F: Field, EF: ExtensionField<F>>(
    xs: &[F],
    ys: &[EF],
    point: EF,
) -> EF {
    let n = xs.len();
    assert_eq!(ys.len(), n);
    if n == 0 {
        return EF::ZERO;
    }

    // Short-circuit: if point coincides with one of the nodes, return the known value directly.
    for i in 0..n {
        if point == EF::from(xs[i]) {
            return ys[i];
        }
    }

    let log_n = log2_strict_usize(n);

    // All xs lie in a coset of the 2^log_n roots of unity, so every x_i shares the same
    // x_i^n = xs[0]^n.
    let coset_power = xs[0].exp_power_of_2(log_n);
    let weight_scale = (F::from_usize(n) * coset_power).inverse();

    let diffs: Vec<EF> = xs.iter().map(|&x| point - EF::from(x)).collect();
    let diff_invs = batch_multiplicative_inverse(&diffs);
    let l_point = diffs.iter().copied().product::<EF>();

    let mut result = EF::ZERO;
    for ((&x, &y), &diff_inv) in xs.iter().zip(ys).zip(diff_invs.iter()) {
        result += y * (x * weight_scale) * diff_inv;
    }
    result * l_point
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn test_eval_poly_zero() {
        let poly: Vec<F> = vec![];
        assert_eq!(eval_poly(&poly, F::from_u64(3)), F::ZERO);
    }

    #[test]
    fn eval_degree_correction_matches_the_geometric_sum_it_stands_for() {
        // Three places evaluate this same sum: `eval_degree_correction`, the chunked sweep in
        // `combine_on_coset`, and the split form `materialize_virtual_fiber` inlines. Only the
        // first two are compared to each other (by the tests above), and their shared
        // reference is this function — so it is pinned here to an independent oracle, plain
        // summation, rather than to another implementation of itself.
        //
        // `geom = Σ_{l=0}^{gap} step^l`, so `gap = 2`, `step = 2` gives `1 + 2 + 4 = 7`.
        assert_eq!(
            eval_degree_correction(EF::ONE, EF::from_u64(2), EF::ONE, 2),
            EF::from_u64(7)
        );

        let x = EF::from_u64(5);
        // The last case makes `step = x * r` exactly one, the branch an honest transcript
        // reaches with probability ~1/|EF| and which end-to-end coverage therefore never hits.
        let cases = [
            (x, EF::from_u64(3)),
            (x, EF::ONE),
            (EF::from_u64(7), EF::from_u64(11)),
            (x, x.inverse()),
        ];

        for gap in [0usize, 1, 5, 40] {
            for (point, r_comb) in cases {
                let step = point * r_comb;

                let mut expected = EF::ZERO;
                let mut power = EF::ONE;
                for _ in 0..=gap {
                    expected += power;
                    power *= step;
                }
                assert_eq!(
                    eval_degree_correction(EF::ONE, point, r_comb, gap),
                    expected,
                    "gap={gap} point={point:?} r_comb={r_comb:?}"
                );

                // The factored form `materialize_virtual_fiber` evaluates, which splits
                // `step^(gap+1)` into a round-constant `r_comb^(gap+1)` and a per-point
                // `x^(gap+1)`.
                let gap_plus_1 = (gap + 1) as u64;
                let split = if step == EF::ONE {
                    EF::from_usize(gap + 1)
                } else {
                    (EF::ONE - r_comb.exp_u64(gap_plus_1) * point.exp_u64(gap_plus_1))
                        * (EF::ONE - step).inverse()
                };
                assert_eq!(
                    split, expected,
                    "gap={gap} point={point:?} r_comb={r_comb:?}"
                );
            }
        }
    }

    #[test]
    fn sample_ood_points_returns_nothing_for_zero_samples() {
        let mut rng = SmallRng::seed_from_u64(3);
        let perm = Perm::new_from_rng_128(&mut rng);
        let mut challenger = TestChallenger::new(perm);
        let excluded = [(F::GENERATOR, 4), (F::GENERATOR, 3), (F::GENERATOR, 2)];
        let points: Vec<EF> = sample_ood_points(&mut challenger, excluded, 0);
        assert!(points.is_empty());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The hoisted `shift^{-2^log_size}` and the shared squaring chain must compose back
        /// into `(z / shift)^{2^log_size}`. The oracle is the original three-line predicate,
        /// evaluated independently below; `log_sizes` are drawn freely so that equal sizes, a
        /// zero size, and the maximum sitting at each of the three positions all occur.
        #[test]
        fn sample_ood_points_avoids_every_excluded_domain(
            log_sizes in prop::collection::vec(0usize..=8, 3..=3),
            shift_seeds in prop::collection::vec(1u64..(1 << 20), 3..=3),
            num_ood_samples in 1usize..=3,
            seed: u64,
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let perm = Perm::new_from_rng_128(&mut rng);
            let mut challenger = TestChallenger::new(perm);

            let excluded: [(F, usize); 3] =
                core::array::from_fn(|i| (F::from_u64(shift_seeds[i]), log_sizes[i]));

            let points: Vec<EF> = sample_ood_points(&mut challenger, excluded, num_ood_samples);
            prop_assert_eq!(points.len(), num_ood_samples);

            for (i, &z) in points.iter().enumerate() {
                for &(shift, log_size) in &excluded {
                    let outside = (z * EF::from(shift).inverse()).exp_power_of_2(log_size)
                        != EF::ONE;
                    prop_assert!(log_size == 0 || outside);
                }
                prop_assert!(!points[..i].contains(&z));
            }
        }
    }

    /// Reference implementation of `combine_on_coset`: `eval_degree_correction` called
    /// pointwise, with no chunked-power-sweep optimization.
    fn naive_combine_on_coset(
        groups: &[(EF, usize, &[EF])],
        r_comb: EF,
        shift: F,
        log_domain: usize,
    ) -> Vec<EF> {
        let n = 1usize << log_domain;
        let g = F::two_adic_generator(log_domain);
        let mut x = shift;
        let mut result = vec![EF::ZERO; n];
        for (p, slot) in result.iter_mut().enumerate() {
            let point = EF::from(x);
            for &(r_i, gap, values) in groups {
                *slot += eval_degree_correction(r_i * values[p], point, r_comb, gap);
            }
            x *= g;
        }
        result
    }

    #[test]
    fn test_combine_on_coset_matches_naive_reference() {
        let mut rng = SmallRng::seed_from_u64(7);
        let shift = F::GENERATOR;

        // log_domain=13 (2 chunks at POWER_CHUNK=4096) with 3 groups of varying gaps,
        // including gap=0 (d*=d_i, the identity-DegCor case) and a large gap.
        let log_domain = 13usize;
        let n = 1usize << log_domain;
        let values_a: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let values_b: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let values_c: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let r_a: EF = rng.random();
        let r_b: EF = rng.random();
        let r_c: EF = rng.random();
        let r_comb: EF = rng.random();

        let groups: Vec<(EF, usize, &[EF])> = vec![
            (r_a, 0, values_a.as_slice()),
            (r_b, 917_504, values_b.as_slice()),
            (r_c, 12, values_c.as_slice()),
        ];

        let fast = combine_on_coset(&groups, r_comb, shift, log_domain);
        let naive = naive_combine_on_coset(&groups, r_comb, shift, log_domain);
        assert_eq!(fast, naive);
    }

    #[test]
    fn test_combine_on_coset_matches_reference_at_degenerate_step() {
        // `combine_on_coset` and `eval_degree_correction` must agree pointwise, including on
        // the one lane where `step = r_comb·shift·g^p` can equal 1 and the geometric sum
        // collapses to `gap + 1`. Reaching it needs `r_comb = (shift·g^p)^{-1}`, which an
        // honest transcript hits with probability ~n/|F|, so only a constructed input gets
        // here — and the naive reference alone would never reveal a divergence.
        let log_domain = 4usize;
        let n = 1usize << log_domain;
        let g = F::two_adic_generator(log_domain);
        let shift = F::GENERATOR;

        let degenerate_p = 3usize;
        let r_comb = EF::from(shift * g.exp_u64(degenerate_p as u64)).inverse();

        let mut values: Vec<EF> = (0..n).map(|i| EF::from_u64(i as u64)).collect();
        values[degenerate_p] = EF::from_u64(4);
        let groups: Vec<(EF, usize, &[EF])> = vec![(EF::ONE, 5, values.as_slice())];

        let fast = combine_on_coset(&groups, r_comb, shift, log_domain);
        let naive = naive_combine_on_coset(&groups, r_comb, shift, log_domain);
        assert_eq!(fast, naive);
        // `value * (gap + 1)` = 4 * 6 at the degenerate lane.
        assert_eq!(fast[degenerate_p], EF::from_u64(24));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// `combine_on_coset`'s chunked power sweep must stay in step with the true
        /// `p`-indexed powers for every coset size and gap, not just the one seed
        /// `test_combine_on_coset_matches_naive_reference` fixes. A failure here means a
        /// `POWER_CHUNK` boundary desync or a wrong `g_hi` seed: `log_domain` spans cosets
        /// below, exactly at, and straddling the chunk size, and `snap_to_period` forces
        /// `gap + 1 ≡ 0 (mod n)`, where `g_hi` collapses to 1.
        #[test]
        fn combine_on_coset_matches_naive_reference_over_shapes(
            log_domain in 2usize..=14,
            gap_specs in prop::collection::vec((0usize..(1 << 20), any::<bool>()), 1..=4),
            seed: u64,
        ) {
            let n = 1usize << log_domain;
            let mut rng = SmallRng::seed_from_u64(seed);
            let shift = F::GENERATOR;
            let r_comb: EF = rng.random();

            let values: Vec<Vec<EF>> = gap_specs
                .iter()
                .map(|_| (0..n).map(|_| rng.random()).collect())
                .collect();
            let coeffs: Vec<EF> = gap_specs.iter().map(|_| rng.random()).collect();

            let groups: Vec<(EF, usize, &[EF])> = gap_specs
                .iter()
                .zip(&coeffs)
                .zip(&values)
                .map(|((&(raw_gap, snap_to_period), &r_i), vals)| {
                    let gap = if snap_to_period {
                        (raw_gap % 32 + 1) * n - 1
                    } else {
                        raw_gap
                    };
                    (r_i, gap, vals.as_slice())
                })
                .collect();

            prop_assert_eq!(
                combine_on_coset(&groups, r_comb, shift, log_domain),
                naive_combine_on_coset(&groups, r_comb, shift, log_domain)
            );
        }
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
    fn test_check_ans_interpolates_accepts_the_interpolant() {
        let pts = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let vals = vec![F::from_u64(4), F::from_u64(9), F::from_u64(16)];

        let ans = interpolate_poly(&pts, &vals);

        // Check at a field element outside `pts`.
        let rho = F::from_u64(7);
        assert!(
            check_ans_interpolates(&ans, &pts, &vals, rho),
            "the interpolant must satisfy the barycentric identity"
        );
    }

    #[test]
    fn test_check_ans_interpolates_rejects_a_corrupted_answer() {
        let pts = vec![F::from_u64(1), F::from_u64(2)];
        let vals = vec![F::from_u64(4), F::from_u64(9)];

        let mut bad_ans = interpolate_poly(&pts, &vals);
        bad_ans[0] += F::ONE;

        let rho = F::from_u64(5);
        assert!(
            !check_ans_interpolates(&bad_ans, &pts, &vals, rho),
            "a shifted answer polynomial must fail the identity"
        );
    }

    #[test]
    fn test_check_ans_interpolates_rejects_a_perturbed_value_set() {
        // The prover's `ans` interpolates the honest values; the verifier reconstructs the
        // interpolant from the values it derived itself, so any disagreement must be caught.
        let pts = vec![F::from_u64(1), F::from_u64(2), F::from_u64(3)];
        let vals = vec![F::from_u64(4), F::from_u64(9), F::from_u64(16)];
        let ans = interpolate_poly(&pts, &vals);

        let rho = F::from_u64(7);
        for i in 0..vals.len() {
            let mut perturbed = vals.clone();
            perturbed[i] += F::ONE;
            assert!(
                !check_ans_interpolates(&ans, &pts, &perturbed, rho),
                "perturbing value {i} must fail the identity"
            );
        }
    }

    #[test]
    fn test_check_ans_interpolates_rejects_degenerate_inputs() {
        let pts = vec![F::from_u64(1), F::from_u64(2)];
        let vals = vec![F::from_u64(4), F::from_u64(9)];
        let ans = interpolate_poly(&pts, &vals);

        // `rho` on a node leaves the barycentric denominators undefined.
        assert!(!check_ans_interpolates(&ans, &pts, &vals, pts[0]));
        // Mismatched lengths pin no interpolation problem at all.
        assert!(!check_ans_interpolates(
            &ans,
            &pts,
            &vals[..1],
            F::from_u64(5)
        ));
        // Repeated nodes leave the barycentric weights undefined.
        let dup_pts = vec![F::from_u64(1), F::from_u64(1)];
        assert!(!check_ans_interpolates(
            &ans,
            &dup_pts,
            &vals,
            F::from_u64(5)
        ));
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
    fn test_lagrange_interpolate_at_agrees_with_lagrange_eval_at_on_coset() {
        // xs = a coset of the 8th roots of unity: shift * g^i for i in 0..8.
        let log_n = 3;
        let n = 1 << log_n;
        let g = F::two_adic_generator(log_n);
        let shift = F::GENERATOR;
        let xs: Vec<F> = g.shifted_powers(shift).take(n).collect();
        let ys: Vec<EF> = (0..n)
            .map(|i| EF::from(F::from_u64(i as u64 * i as u64 + 1)))
            .collect();

        for z in [
            EF::from(F::from_u64(1000)),
            EF::ZERO,
            -EF::from(F::from_u64(7)),
        ] {
            assert_eq!(
                lagrange_interpolate_at(&xs, &ys, z),
                lagrange_eval_at(&xs, &ys, z),
                "closed-form and general barycentric forms must agree"
            );
        }
    }

    #[test]
    fn test_lagrange_interpolate_at_returns_node_value_on_exact_match() {
        let log_n = 2;
        let n = 1 << log_n;
        let g = F::two_adic_generator(log_n);
        let shift = F::GENERATOR;
        let xs: Vec<F> = g.shifted_powers(shift).take(n).collect();
        let ys: Vec<EF> = (0..n)
            .map(|i| EF::from(F::from_u64(i as u64 + 10)))
            .collect();

        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(lagrange_interpolate_at(&xs, &ys, EF::from(x)), ys[i]);
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
