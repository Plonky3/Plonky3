//! PESAT relation abstraction for WARP.
//!
//! WARP is an accumulation scheme for the **PESAT** relation
//! (polynomial equation satisfiability) of [WARP, Definition 5.1].
//! A PESAT system is a tuple `(p̂, M, N, k)` where `p̂ = (p̂_1, …, p̂_M)`
//! is a list of constant-degree polynomials in `N` variables, with `k`
//! of those variables forming the witness and `κ = N − k` forming the
//! explicit instance.
//!
//! The protocol only ever interacts with PESAT through the **bundling**
//! polynomial (Definition 5.5):
//!
//! ```text
//!     Pb(τ, z) = Σ_{i ∈ {0,1}^{log M}} eq(τ, i) · p̂_i(z)
//! ```
//!
//! Hence the trait surface required by WARP reduces to two methods:
//!
//! - `evaluate_bundled(τ_eq, z)` for the decider and the §6.3 oracle check.
//! - `bundled_round_poly(b_lo, b_hi, w_lo, w_hi)` for the §6.3 sumcheck
//!   prover, which folds two adjacent (β, w) pairs along the round axis.
//!
//! The trait is intentionally protocol-agnostic. The Plonky3-native benchmark
//! path uses [`BooleanPesat`](boolean::BooleanPesat), a direct Boolean relation
//! with one quadratic constraint per witness coordinate.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};

pub mod boolean;
pub mod claim_6_5;

pub use boolean::BooleanPesat;
pub use claim_6_5::{Claim65Scratch, eq_dot_q_recursive, poly_lerp_via_linear};

/// Shape parameters of a PESAT instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PesatShape {
    /// `log_2 M` where `M` is the number of bundled constraints (must be a
    /// power of two; pad with zero-constraints if necessary).
    pub log_constraints: usize,
    /// `log_2 k` where `k` is the witness length (must be a power of two).
    pub log_witness: usize,
    /// Length `κ = N − k` of the explicit instance (no power-of-two
    /// requirement; the explicit instance is read directly).
    pub explicit_len: usize,
    /// Maximum total degree `d` of any individual `p̂_i` in the variables `z`.
    pub max_degree: usize,
}

impl PesatShape {
    /// Number of bundled constraints `M = 2^{log_constraints}`.
    #[inline]
    pub const fn num_constraints(&self) -> usize {
        1 << self.log_constraints
    }

    /// Witness length `k = 2^{log_witness}`.
    #[inline]
    pub const fn witness_len(&self) -> usize {
        1 << self.log_witness
    }

    /// Total `(z = (x, w))` length `N = κ + k`.
    #[inline]
    pub const fn total_vars(&self) -> usize {
        self.explicit_len + self.witness_len()
    }

    /// Length `m = log M + κ` of the per-instance `β` vector
    /// (the bundling τ part plus the explicit instance part).
    #[inline]
    pub const fn beta_len(&self) -> usize {
        self.log_constraints + self.explicit_len
    }
}

/// A bundled PESAT relation as seen by the WARP protocol.
///
/// This is the only interface WARP needs from a constraint system. Any type
/// implementing this trait can be accumulated.
pub trait BundledPesat<F: Field, EF: ExtensionField<F>>: Sync + Send {
    /// Shape parameters of this PESAT instance.
    fn shape(&self) -> PesatShape;

    /// Evaluate `Pb(τ, z) = Σ_i eq(τ, i) · p̂_i(z)` at a single point.
    ///
    /// - `tau_eq` is the precomputed eq-table `[eq(τ, i) : i ∈ [0, M)]`,
    ///   length `M`.
    /// - `z = (x, w)` is the full assignment, length `N = κ + k`.
    fn evaluate_bundled(&self, tau_eq: &[EF], z: &[EF]) -> EF;

    /// Per-constraint polynomial slices in `α` along the §6.3 round axis.
    ///
    /// Linearise the explicit instance and witness as
    /// ```text
    ///     b_x(α) = (1 − α) · b_x_lo + α · b_x_hi   (length κ)
    ///     w(α)   = (1 − α) · w_lo  + α · w_hi      (length k)
    /// ```
    /// and return `M` polynomials in `α` (one per constraint), each in
    /// coefficient form of length `max_degree + 1`. The `c`-th entry is
    /// the polynomial `α ↦ p̂_c(b_x(α), w(α))`.
    ///
    /// **Indexing**. The `c`-th polynomial corresponds to the same constraint
    /// index `c ∈ [0, M)` that [`evaluate_bundled`](Self::evaluate_bundled)
    /// weights with `tau_eq[c]`. Implementors must keep the orderings in
    /// lock-step.
    ///
    /// # Why this is on the trait
    ///
    /// This is the inner-loop kernel of the optimal §6.3 prover (paper
    /// Lemma 6.4): given per-constraint polys, the prover applies
    /// [`eq_dot_q_recursive`] (Claim 6.5) to compute
    /// `Σ_c eq(B_τ(X), c) · p̂_c(B_x(X), W(X))` in `O(M · d)` time, vs
    /// `O(M · d · log M)` for the naive "evaluate at `D + 1` points then
    /// Lagrange-interpolate" pattern previously used by the default impl.
    ///
    /// # Panics
    ///
    /// - `b_x_lo.len() == b_x_hi.len() == κ` from [`shape().explicit_len`].
    /// - `w_lo.len() == w_hi.len() == k` from [`shape().witness_len`].
    fn iter_constraint_polys_at_lerp(
        &self,
        b_x_lo: &[EF],
        b_x_hi: &[EF],
        w_lo: &[EF],
        w_hi: &[EF],
    ) -> Vec<Vec<EF>>;

    /// Compute the §6.3 sumcheck round contribution from a single `(lo, hi)`
    /// index pair.
    ///
    /// Given the lo / hi values of `(β, w)` along the round axis, returns
    /// the coefficients of the univariate polynomial
    ///
    /// ```text
    ///     Q(X) = Σ_c eq(B_τ(X), c) · p̂_c(B_x(X), W(X))
    /// ```
    ///
    /// in `X`, where:
    /// - `B(X) = (1−X)·b_lo + X·b_hi` of length `log M + κ`
    /// - `B_τ(X)` is the first `log M` coordinates (the bundling τ)
    /// - `B_x(X)` is the remaining `κ` coordinates (the explicit instance)
    /// - `W(X) = (1−X)·w_lo + X·w_hi` of length `k`
    ///
    /// Returned vector has length `1 + log M + d` (the polynomial degree
    /// plus one), in coefficient form `[c_0, c_1, …, c_D]` so that
    /// `Q(X) = Σ_j c_j · X^j`.
    ///
    /// # Default impl: paper Lemma 6.4 / Claim 6.5
    ///
    /// The default impl combines [`iter_constraint_polys_at_lerp`] with
    /// [`eq_dot_q_recursive`] for `O(M · d)` cost. Implementors that have a
    /// faster bespoke kernel can override.
    ///
    /// # Panics
    ///
    /// - All four input slices must have the right length per `shape()`.
    fn bundled_round_poly(&self, b_lo: &[EF], b_hi: &[EF], w_lo: &[EF], w_hi: &[EF]) -> Vec<EF> {
        default_round_poly_via_claim_6_5(self, b_lo, b_hi, w_lo, w_hi)
    }

    /// Scratch-buffer variant of [`bundled_round_poly`](Self::bundled_round_poly).
    ///
    /// The default implementation preserves compatibility for arbitrary
    /// `BundledPesat` implementations. Hot relations, such as
    /// [`BooleanPesat`](crate::relation::BooleanPesat), override this to avoid
    /// allocating a fresh coefficient vector and Claim 6.5 work tables for every
    /// WARP §6.3 `(round, i)` contribution.
    fn bundled_round_poly_into(
        &self,
        b_lo: &[EF],
        b_hi: &[EF],
        w_lo: &[EF],
        w_hi: &[EF],
        out: &mut Vec<EF>,
        _scratch: &mut Claim65Scratch<F, EF>,
    ) {
        out.clear();
        out.extend(self.bundled_round_poly(b_lo, b_hi, w_lo, w_hi));
    }

    /// Bytes that bind this constraint system into the Fiat–Shamir transcript.
    ///
    /// Should be a deterministic encoding of the relation's structure (e.g.,
    /// constraint matrices or relation metadata). Two distinct
    /// PESAT instances must produce distinct descriptions.
    fn description(&self) -> Vec<u8>;

    /// Maximum degree of the `Q(X)` polynomial returned by
    /// [`bundled_round_poly`](Self::bundled_round_poly).
    ///
    /// Equal to `log M + d` for the standard `eq`-zero-evader.
    #[inline]
    fn round_poly_degree(&self) -> usize {
        let s = self.shape();
        s.log_constraints + s.max_degree
    }
}

/// Default `bundled_round_poly` impl: Lemma 6.4 / Claim 6.5 algorithm.
///
/// Splits `B(X)` into `B_τ(X)` (the first `log M` coordinates) and `B_x(X)`
/// (the remaining `κ` coordinates), invokes
/// [`BundledPesat::iter_constraint_polys_at_lerp`] to get per-constraint
/// polys in `α`, then folds them with `B_τ(X)` via
/// [`eq_dot_q_recursive`] (Claim 6.5).
///
/// Cost: `O(M · d)` field ops plus the cost of one
/// `iter_constraint_polys_at_lerp` call (`O(d · |p̂| · M)` for the typical
/// "evaluate at `d + 1` points + Lagrange-interpolate per constraint" impl).
///
/// # Panics
///
/// - `b_lo.len() == b_hi.len() == log M + κ`.
/// - `w_lo.len() == w_hi.len() == k`.
pub fn default_round_poly_via_claim_6_5<P, F, EF>(
    pesat: &P,
    b_lo: &[EF],
    b_hi: &[EF],
    w_lo: &[EF],
    w_hi: &[EF],
) -> Vec<EF>
where
    P: BundledPesat<F, EF> + ?Sized,
    F: Field,
    EF: ExtensionField<F>,
{
    let shape = pesat.shape();
    let log_m = shape.log_constraints;
    let k = shape.witness_len();

    assert_eq!(b_lo.len(), shape.beta_len(), "b_lo length");
    assert_eq!(b_hi.len(), shape.beta_len(), "b_hi length");
    assert_eq!(w_lo.len(), k, "w_lo length");
    assert_eq!(w_hi.len(), k, "w_hi length");

    // Split B(X) into B_τ(X) (first log_m) and B_x(X) (remaining κ).
    let (b_tau_lo, b_x_lo) = b_lo.split_at(log_m);
    let (b_tau_hi, b_x_hi) = b_hi.split_at(log_m);

    // Per-constraint polys in α (M polys, each degree d).
    let constraint_polys = pesat.iter_constraint_polys_at_lerp(b_x_lo, b_x_hi, w_lo, w_hi);
    debug_assert_eq!(constraint_polys.len(), 1usize << log_m, "M polys expected");

    // Build B_τ(X) as `log_m` linear polys [b_τ_lo[j], b_τ_hi[j] − b_τ_lo[j]],
    // matching the encoding `eq_dot_q_recursive` consumes.
    let b_tau_linears: Vec<[EF; 2]> = b_tau_lo
        .iter()
        .zip(b_tau_hi.iter())
        .map(|(&l, &r)| [l, r - l])
        .collect();

    eq_dot_q_recursive(&b_tau_linears, constraint_polys)
}

/// Lagrange-interpolate a polynomial of degree `D` from its evaluations at
/// the integer points `{0, 1, …, D}`, returning monomial coefficients
/// `[c_0, c_1, …, c_D]` so that `Q(X) = Σ_j c_j · X^j`.
///
/// # Cost
///
/// `O(D²)` field operations per call. For our typical `D ≈ log M + d`
/// (around 25 for `M = 2^20`, `d = 5`) this is negligible compared to
/// the per-point evaluation cost.
///
/// # Panics
///
/// - `evals` must be non-empty.
/// - The integer differences `(i − j)` for `i ≠ j ∈ [0, D]` must be
///   invertible in `EF` (true for any field of characteristic `> D`).
pub fn lagrange_interpolate_int_points<EF: Field>(evals: &[EF]) -> Vec<EF> {
    let n = evals.len();
    assert!(n > 0, "lagrange_interpolate_int_points: empty evals");
    if n == 1 {
        return vec![evals[0]];
    }

    // Newton forward interpolation on the grid {0, 1, ..., n - 1}:
    // P(X) = sum_k Δ^k P(0) * binom(X, k).
    //
    // This replaces the previous per-basis Lagrange construction, which did
    // O(n^2) polynomial work and n extension inversions per call. The grid
    // scalars are fixed small integers, so we batch-invert 1..n once and
    // convert the falling-factorial basis to monomials incrementally.
    let mut diffs = evals.to_vec();
    let mut newton = vec![EF::ZERO; n];
    for k in 0..n {
        newton[k] = diffs[0];
        for i in 0..(n - k - 1) {
            diffs[i] = diffs[i + 1] - diffs[i];
        }
    }

    let invs =
        batch_multiplicative_inverse(&(1..n).map(|i| EF::from_u64(i as u64)).collect::<Vec<_>>());
    assert!(
        invs.iter().all(|&inv| inv != EF::ZERO),
        "lagrange_interpolate_int_points: characteristic too small",
    );

    let mut result = vec![EF::ZERO; n];
    let mut falling = vec![EF::ONE];
    let mut inv_factorial = EF::ONE;
    for k in 0..n {
        let factor = newton[k] * inv_factorial;
        for j in 0..=k {
            result[j] += falling[j] * factor;
        }

        if k + 1 < n {
            inv_factorial *= invs[k];
            let k_ef = EF::from_u64(k as u64);
            falling.push(EF::ZERO);
            for j in (0..=k).rev() {
                let old = falling[j];
                falling[j + 1] += old;
                falling[j] = -k_ef * old;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Sanity check: interpolate the polynomial Q(X) = 3 + 5X + 7X² from its
    /// evaluations at {0, 1, 2}; round-trip recovers the coefficients.
    #[test]
    fn lagrange_recovers_coefficients() {
        // Q(0) = 3, Q(1) = 3 + 5 + 7 = 15, Q(2) = 3 + 10 + 28 = 41.
        let evals = vec![EF::from_u64(3), EF::from_u64(15), EF::from_u64(41)];
        let coeffs = lagrange_interpolate_int_points(&evals);
        assert_eq!(
            coeffs,
            vec![EF::from_u64(3), EF::from_u64(5), EF::from_u64(7)]
        );
    }

    /// Constant polynomial Q(X) = 42 round-trips through evaluation +
    /// interpolation as a single coefficient.
    #[test]
    fn lagrange_constant() {
        let evals = vec![EF::from_u64(42)];
        let coeffs = lagrange_interpolate_int_points(&evals);
        assert_eq!(coeffs, vec![EF::from_u64(42)]);
    }

    /// Higher-degree round trip: build a random degree-7 polynomial,
    /// evaluate at {0..8}, interpolate, compare with original coeffs.
    #[test]
    fn lagrange_degree_7_roundtrip() {
        let coeffs = (0..8)
            .map(|i| EF::from_u64((i * 17 + 1) as u64))
            .collect::<Vec<_>>();
        let evals: Vec<EF> = (0..8)
            .map(|x| {
                // Evaluate via Horner.
                let x_ef = EF::from_u64(x as u64);
                coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * x_ef + c)
            })
            .collect();
        let recovered = lagrange_interpolate_int_points(&evals);
        assert_eq!(recovered, coeffs);
    }
}
