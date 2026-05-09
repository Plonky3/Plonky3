//! Recursive composition algorithm from WARP paper Claim 6.5
//! (eprint 2025/753, Lemma 6.4 and Claim 6.5, lines 2090–2225).
//!
//! Given a vector of `m` linear univariate polynomials
//! `ĉ(X) = (ĉ_0(X), …, ĉ_{m-1}(X))` and a list of `2^m` univariate polynomials
//! `q̂(X) = (q̂_b(X))_{b ∈ {0,1}^m}` of degree at most `d`, this module computes
//!
//! ```text
//!     P(X) := Σ_{b ∈ {0,1}^m} eq(ĉ(X), b) · q̂_b(X)
//! ```
//!
//! in `O(2^m · d)` field operations (vs `O(2^m · d · m)` for the naive
//! "evaluate at `m + d + 1` integer points + Lagrange-interpolate" pattern).
//!
//! The optimal-cost algorithm is the bottleneck inner kernel of the WARP
//! §6.3 twin-constraint sumcheck prover (Lemma 6.4); applying it on both
//! the codeword side (`m = log n`, `d = 1`) and the constraint side
//! (`m = log M`, `d = PESAT degree`) brings the per-round cost from
//! `O((n + M·d) · log(n·M·d))` down to `O(n + M·d)`.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};

/// Reusable buffers for the specialised Claim 6.5 kernels.
///
/// The hot WARP prover path invokes Claim 6.5 many times per accumulation
/// step. Keeping the large intermediate coefficient tables here lets callers
/// reuse allocations across `(round, i)` work items while preserving the same
/// coefficient-level algorithm.
pub struct Claim65Scratch<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub(crate) scalar_current: Vec<EF>,
    pub(crate) scalar_next: Vec<EF>,
    pub(crate) packed_current: Vec<EF::ExtensionPacking>,
    pub(crate) packed_next: Vec<EF::ExtensionPacking>,
    pub(crate) lane_buf: Vec<EF>,
    pub(crate) broadcast_buf: Vec<EF>,
}

impl<F, EF> Claim65Scratch<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub const fn new() -> Self {
        Self {
            scalar_current: Vec::new(),
            scalar_next: Vec::new(),
            packed_current: Vec::new(),
            packed_next: Vec::new(),
            lane_buf: Vec::new(),
            broadcast_buf: Vec::new(),
        }
    }
}

/// Recursive composition algorithm (WARP paper Claim 6.5, lines 2148–2190).
///
/// Computes the polynomial
/// ```text
///     P(X) := Σ_{b ∈ {0,1}^m} eq(c(X), b) · q_b(X)
/// ```
/// where:
/// - `c[j]` for `j ∈ [0, m)` is a linear univariate polynomial in `X`,
///   given as a 2-element coefficient slice `[c_const, c_linear]` so that
///   `c_j(X) = c_const + c_linear · X`. (In the WARP §6.3 application,
///   each `c_j` is a lerp `(1 − X) · c_lo + X · c_hi`, which encodes as
///   `[c_lo, c_hi − c_lo]`.)
/// - `q[b]` for `b ∈ [0, 2^m)` is a polynomial of degree at most `d`,
///   given as a coefficient slice of length `d + 1`.
///
/// Returns the coefficient vector of `P` (length `m + d + 1`).
///
/// **Bit convention**: the first `c[0]` absorbs the **MSB** of `b`'s integer
/// index. So at the first recursive step we pair `q[idx]` with
/// `q[idx + 2^{m-1}]` for `idx ∈ [0, 2^{m-1})`. This matches Plonky3's
/// `Poly::new_from_point` (big-endian / MSB-first) hypercube indexing.
///
/// # Cost
/// `O(2^m · d + 2^m · m)` field ops, dominated by `O(2^m · d)` when `d ≳ m`.
///
/// # Panics
/// - `q.len() == 2^m` where `m = c.len()`.
/// - All `c[j]` slices are length 2 (linear poly).
/// - All `q[b]` slices share the same length `d + 1`.
pub fn eq_dot_q_recursive<EF: Field>(c: &[[EF; 2]], q: Vec<Vec<EF>>) -> Vec<EF> {
    let m = c.len();
    assert_eq!(q.len(), 1 << m, "q must have 2^m elements");
    if m == 0 {
        return q.into_iter().next().unwrap();
    }
    let d_plus_1 = q[0].len();
    debug_assert!(q.iter().all(|qi| qi.len() == d_plus_1));

    let mut current = q;
    for i in 0..m {
        let half = current.len() / 2;
        let c_i = c[i];
        let mut next: Vec<Vec<EF>> = Vec::with_capacity(half);
        for b in 0..half {
            let q_lo = &current[b];
            let q_hi = &current[b + half];
            next.push(poly_lerp_via_linear::<EF>(q_lo, q_hi, c_i));
        }
        current = next;
    }

    debug_assert_eq!(current.len(), 1);
    current.into_iter().next().unwrap()
}

/// Compute `q_lo + (q_hi − q_lo) · (c[0] + c[1] · X)` in coefficient form.
///
/// Output has length `q_lo.len() + 1` (degree increases by 1 from the
/// linear factor). Cost: `O(q_lo.len())` field ops.
#[inline]
pub fn poly_lerp_via_linear<EF: Field>(q_lo: &[EF], q_hi: &[EF], c: [EF; 2]) -> Vec<EF> {
    debug_assert_eq!(q_lo.len(), q_hi.len());
    let n = q_lo.len();
    let mut out = vec![EF::ZERO; n + 1];
    let c_const = c[0];
    let c_linear = c[1];
    for k in 0..n {
        let q_diff_k = q_hi[k] - q_lo[k];
        out[k] += q_lo[k] + q_diff_k * c_const;
        out[k + 1] += q_diff_k * c_linear;
    }
    out
}

#[inline]
pub(crate) fn packed_ext_scalar_with_scratch<F, EF>(
    value: EF,
    _broadcast_buf: &mut Vec<EF>,
) -> EF::ExtensionPacking
where
    F: Field,
    EF: ExtensionField<F>,
{
    value.into()
}

pub(crate) fn fold_claim_6_5_scalar_round<EF>(
    current: &[EF],
    width: usize,
    len: usize,
    c0: EF,
    c1: EF,
    next: &mut [EF],
) where
    EF: Field,
{
    debug_assert_eq!(current.len(), len * width);
    debug_assert_eq!(next.len(), (len / 2) * (width + 1));
    debug_assert!(width > 0);
    let half = len / 2;
    let next_width = width + 1;
    for b in 0..half {
        let lo_base = b * width;
        let hi_base = (b + half) * width;
        let out_base = b * next_width;

        let lo = current[lo_base];
        let diff = current[hi_base] - lo;
        next[out_base] = lo + diff * c0;
        let mut carry = diff * c1;

        for k in 1..width {
            let lo = current[lo_base + k];
            let diff = current[hi_base + k] - lo;
            next[out_base + k] = carry + lo + diff * c0;
            carry = diff * c1;
        }
        next[out_base + width] = carry;
    }
}

pub(crate) fn fold_claim_6_5_packed_round<F, EF>(
    current: &[EF::ExtensionPacking],
    width: usize,
    len: usize,
    c0: EF::ExtensionPacking,
    c1: EF::ExtensionPacking,
    next: &mut [EF::ExtensionPacking],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(current.len(), len * width);
    debug_assert_eq!(next.len(), (len / 2) * (width + 1));
    debug_assert!(width > 0);
    let half = len / 2;
    let next_width = width + 1;
    for b in 0..half {
        let lo_base = b * width;
        let hi_base = (b + half) * width;
        let out_base = b * next_width;

        let lo = current[lo_base];
        let diff = current[hi_base] - lo;
        next[out_base] = lo + diff * c0;
        let mut carry = diff * c1;

        for k in 1..width {
            let lo = current[lo_base + k];
            let diff = current[hi_base + k] - lo;
            next[out_base + k] = carry + lo + diff * c0;
            carry = diff * c1;
        }
        next[out_base + width] = carry;
    }
}

pub(crate) fn unpack_packed_coeffs_to_scalar<F, EF>(
    packed: &[EF::ExtensionPacking],
    width: usize,
    out: &mut Vec<EF>,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let pack_w = F::Packing::WIDTH;
    debug_assert_eq!(packed.len(), width);
    out.clear();
    out.resize(pack_w * width, EF::ZERO);
    for (coeff_idx, &coeff) in packed.iter().enumerate() {
        for (lane, value) in EF::ExtensionPacking::to_ext_iter([coeff]).enumerate() {
            out[lane * width + coeff_idx] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::eq_dot_q_recursive;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Naive reference: compute `Σ_b eq(c(α), b) · q_b(α)` at one point `α`.
    fn naive_eval_at_alpha(c: &[[EF; 2]], q: &[Vec<EF>], alpha: EF) -> EF {
        let m = c.len();
        let n = 1usize << m;
        debug_assert_eq!(q.len(), n);

        let c_at_alpha: Vec<EF> = c.iter().map(|cj| cj[0] + cj[1] * alpha).collect();
        let eq = Poly::<EF>::new_from_point(&c_at_alpha, EF::ONE);

        let mut acc = EF::ZERO;
        for b in 0..n {
            let qb_at_alpha = q[b]
                .iter()
                .rev()
                .copied()
                .fold(EF::ZERO, |s, ck| s * alpha + ck);
            acc += eq.as_slice()[b] * qb_at_alpha;
        }
        acc
    }

    fn poly_eval(coeffs: &[EF], alpha: EF) -> EF {
        coeffs
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |s, c| s * alpha + c)
    }

    fn random_inputs(rng: &mut SmallRng, m: usize, d: usize) -> (Vec<[EF; 2]>, Vec<Vec<EF>>) {
        use rand::RngExt;
        let c: Vec<[EF; 2]> = (0..m)
            .map(|_| [rng.random::<EF>(), rng.random::<EF>()])
            .collect();
        let q: Vec<Vec<EF>> = (0..(1 << m))
            .map(|_| (0..=d).map(|_| rng.random::<EF>()).collect())
            .collect();
        (c, q)
    }

    #[test]
    fn claim_6_5_agrees_with_naive_eval_random_inputs() {
        for m in 1..=5usize {
            for d in 1..=3usize {
                let mut rng = SmallRng::seed_from_u64(((m as u64) << 16) | d as u64);
                let (c, q) = random_inputs(&mut rng, m, d);
                let p = eq_dot_q_recursive::<EF>(&c, q.clone());
                assert_eq!(
                    p.len(),
                    m + d + 1,
                    "output poly length mismatch at m={m}, d={d}"
                );
                use rand::RngExt;
                for _ in 0..4 {
                    let alpha: EF = rng.random();
                    let from_recursive = poly_eval(&p, alpha);
                    let from_naive = naive_eval_at_alpha(&c, &q, alpha);
                    assert_eq!(
                        from_recursive, from_naive,
                        "Claim 6.5 disagrees with naive at m={m}, d={d}, α={alpha:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn claim_6_5_m_zero_returns_q0() {
        let q = vec![vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]];
        let p = eq_dot_q_recursive::<EF>(&[], q.clone());
        assert_eq!(p, q[0]);
    }

    #[test]
    fn claim_6_5_alpha_zero_at_hypercube_vertex() {
        let m = 3usize;
        let n = 1 << m;
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let q: Vec<Vec<EF>> = (0..n)
            .map(|_| {
                use rand::RngExt;
                vec![rng.random::<EF>(), rng.random::<EF>()]
            })
            .collect();

        let b_star_int = 5usize;
        let bits_msb_first: Vec<EF> = (0..m)
            .map(|j| {
                if (b_star_int >> (m - 1 - j)) & 1 == 1 {
                    EF::ONE
                } else {
                    EF::ZERO
                }
            })
            .collect();
        let c: Vec<[EF; 2]> = bits_msb_first
            .iter()
            .map(|&bit| [bit, EF::from_u64(42)])
            .collect();
        let p = eq_dot_q_recursive::<EF>(&c, q.clone());
        let expected = q[b_star_int][0];
        let got = poly_eval(&p, EF::ZERO);
        assert_eq!(got, expected, "Claim 6.5 indexing convention mismatch");
    }
}
