//! Deferred matrix evaluation: the lincheck's circuit form as one claim on the circuit's stacked matrix.
//!
//! The stacked matrix `M(s, r, c)` is `A` at `s = 0` and `B` at `s = 1`, so `M(t, x) = (1 + t) A(x) + t B(x)` in characteristic two.
//!
//! The form weighs rows by `L(skip) ⊗ eq(inner)` and columns by `eq(point) ⊗ slices`, so two 64-entry restrictions and one stacking reduce it to `M` at one point.
//!
//! Each of the three steps is a Schwartz-Zippel check on a multilinear polynomial: soundness error `(6 + 6 + 1) / |F|`.
//!
//! Two claims on one matrix fold into one on the line through their points (Thaler, "Proofs, Arguments, and Zero-Knowledge", Section 4.5.2): error `n / |F|` for `n` variables.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{Dup, Field, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_verifier_arith::Lift;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::circuit::Circuit;
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::witness::Program;
use crate::{Challenge, Val};

/// The circuit form a lincheck left to a deferred claim, with the row fold the prover supplied.
#[derive(Clone, Debug)]
pub struct PendingForm<E> {
    /// Bits per instance of the circuit.
    pub k_log: usize,
    /// The rows' skip coordinate.
    pub skip: E,
    /// The rows' multilinear coordinates.
    pub inner: Vec<E>,
    /// The columns' multilinear coordinates.
    pub point: Vec<E>,
    /// The weight of `B`.
    pub alpha: E,
    /// `h(v)`: the row-weighted matrix at column word `point` and low bits `v`.
    pub h: Vec<E>,
}

/// A claim `M(point) = value` on a circuit's stacked matrix.
///
/// The point is the stacking coordinate, then the row's `k_log` coordinates, then the column's, most significant first.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatrixClaim<E = Challenge> {
    /// The point.
    pub point: Vec<E>,
    /// The claimed value.
    pub value: E,
}

/// The prover's messages reducing a circuit form to a matrix claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormProof {
    /// `g(u)`: the matrix at row word `inner`, low bits `u`, and the column point.
    pub g: Vec<Challenge>,
    /// `A` and `B` at the row and column points.
    pub ab: [Challenge; 2],
}

/// Why a deferral or a fold is rejected.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DeferralError {
    /// A message has the wrong length.
    #[error("a message has the wrong length")]
    Shape,
    /// A message does not reproduce the claim it reduces.
    #[error("a message does not reproduce the claim it reduces")]
    Mismatch,
}

/// Lanes one walk of a [`Matrix`] carries.
const LANES: usize = 8;

/// Bytes the register files of a [`Matrix`]'s parallel walks may take together.
const REGISTER_BUDGET: usize = 1 << 31;

/// `eq(point, s)` for every index `s`, as two half tables whose product is the entry.
struct SplitEq {
    /// Entries of the high bits.
    hi: Vec<Challenge>,
    /// Entries of the low bits.
    lo: Vec<Challenge>,
    /// Low bits.
    lo_bits: usize,
}

impl SplitEq {
    /// The tables of a point, the first coordinate the most significant.
    fn new(point: &[Challenge]) -> Self {
        let lo_bits = point.len() / 2;
        let (hi, lo) = point.split_at(point.len() - lo_bits);
        Self {
            hi: eq(hi),
            lo: eq(lo),
            lo_bits,
        }
    }

    /// The tables of `eq(point, s / 64) low[s % 64]`, the 64 low entries folded into the low table.
    fn with_low(point: &[Challenge], low: &[Challenge]) -> Self {
        let mut split = Self::new(point);
        split.lo = split
            .lo
            .iter()
            .flat_map(|&e| low.iter().map(move |&l| e * l))
            .collect();
        split.lo_bits += K_SKIP;
        split
    }

    /// `eq(point, s)`.
    #[inline]
    fn at(&self, s: usize) -> Challenge {
        self.hi[s >> self.lo_bits] * self.lo[s & ((1 << self.lo_bits) - 1)]
    }
}

/// `eq(point, x)` for every hypercube point `x`, the first coordinate the most significant.
fn eq(point: &[Challenge]) -> Vec<Challenge> {
    Poly::new_from_point(point, Challenge::ONE).into_evals()
}

/// A circuit's stacked matrix, compiled once for repeated evaluation.
///
/// Every evaluation streams the gates with live wires in registers, several points per walk.
pub struct Matrix {
    /// The compiled gate walk.
    program: Arc<Program>,
    /// Bits per instance.
    k_log: usize,
    /// The circuit's digest.
    digest: [u8; 32],
}

impl core::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Matrix")
            .field("k_log", &self.k_log)
            .field("registers", &self.program.registers())
            .finish_non_exhaustive()
    }
}

impl Matrix {
    /// Compile a circuit's matrices.
    #[must_use]
    pub fn new(circuit: &Circuit) -> Self {
        Self {
            program: circuit.program(),
            k_log: circuit.k_log(),
            digest: *circuit.digest(),
        }
    }

    /// Bits per instance: each of the row and column indices has this many coordinates.
    #[must_use]
    pub const fn k_log(&self) -> usize {
        self.k_log
    }

    /// The circuit's digest.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// `M` at every point, each of `1 + 2 k` coordinates for some `k >= k_log`.
    ///
    /// Row and column coordinates past `k_log` are the most significant, where the matrix is zero-extended.
    ///
    /// # Panics
    ///
    /// When a point has an even length or names fewer than `k_log` row coordinates.
    #[must_use]
    pub fn evaluate(&self, points: &[Vec<Challenge>]) -> Vec<Challenge> {
        let k = self.k_log;
        // The zero extension contributes `eq(high, 0)` on each side.
        let parts: Vec<(Challenge, Challenge, SplitEq, SplitEq)> = points
            .iter()
            .map(|point| {
                assert_eq!(
                    point.len() % 2,
                    1,
                    "the stacking, row and column coordinates"
                );
                let pad = (point.len() - 1) / 2;
                assert!(pad >= k, "a point names every coordinate of the matrix");
                let (rows, cols) = point[1..].split_at(pad);
                let extension: Challenge = rows[..pad - k]
                    .iter()
                    .chain(&cols[..pad - k])
                    .map(|&c| Challenge::ONE + c)
                    .product();
                (
                    point[0],
                    extension,
                    SplitEq::new(&rows[pad - k..]),
                    SplitEq::new(&cols[pad - k..]),
                )
            })
            .collect();
        // Each walk holds `LANES` values per register, so at most a budget's worth of walks run at once.
        let per_walk = self.program.registers() * size_of::<[Challenge; LANES]>();
        let walks = (REGISTER_BUDGET / per_walk).max(1);
        parts
            .chunks(LANES * walks)
            .flat_map(|batch| self.evaluate_parts(batch))
            .collect()
    }

    /// The values of prepared points, `LANES` per walk, the walks in parallel.
    fn evaluate_parts(&self, parts: &[(Challenge, Challenge, SplitEq, SplitEq)]) -> Vec<Challenge> {
        parts
            .par_chunks(LANES)
            .flat_map_iter(|group| {
                // Rows sum per high entry of their weight, which scales each sum once.
                let mut sums: Vec<[(Challenge, Challenge); LANES]> =
                    vec![[(Challenge::ZERO, Challenge::ZERO); LANES]; group[0].2.hi.len()];
                self.program.walk_rows_lanes::<Challenge, LANES>(
                    |s| {
                        core::array::from_fn(|l| {
                            group.get(l).map_or(Challenge::ZERO, |p| p.3.at(s))
                        })
                    },
                    |s, a, b| {
                        let block = &mut sums[s >> group[0].2.lo_bits];
                        for (l, part) in group.iter().enumerate() {
                            let low = part.2.lo[s & ((1 << part.2.lo_bits) - 1)];
                            block[l].0 += low * a[l];
                            block[l].1 += low * b[l];
                        }
                    },
                );
                group
                    .iter()
                    .enumerate()
                    .map(|(l, p)| {
                        let (sa, sb) = sums
                            .iter()
                            .zip(&p.2.hi)
                            .fold((Challenge::ZERO, Challenge::ZERO), |(x, y), (block, &h)| {
                                (x + h * block[l].0, y + h * block[l].1)
                            });
                        p.1 * (sa + p.0 * (sa + sb))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// `h(v)` of a circuit form: rows weighted by `L(skip) ⊗ eq(inner)`, columns by `eq(point) ⊗ e_v`.
    fn form_rows(
        &self,
        skip: Challenge,
        inner: &[Challenge],
        point: &[Challenge],
        alpha: Challenge,
    ) -> Vec<Challenge> {
        // One backward walk: the rows' weights flow to the columns, each column summed into its low bits' entry.
        let rows = SplitEq::with_low(inner, &lagrange_weights(&skip, SKIP_SIZE));
        let words = SplitEq::new(point);
        let mask = (1 << words.lo_bits) - 1;
        // Per high entry of the column word, the 64 sums still to be scaled by it.
        let mut sums = vec![[Challenge::ZERO; SKIP_SIZE]; words.hi.len()];
        self.program.fold_transpose_by(
            Challenge::ONE,
            alpha,
            |s| rows.at(s),
            |c, v| {
                let word = c / SKIP_SIZE;
                sums[word >> words.lo_bits][c % SKIP_SIZE] += words.lo[word & mask] * v;
            },
        );
        (0..SKIP_SIZE)
            .map(|v| sums.iter().zip(&words.hi).map(|(s, &h)| h * s[v]).sum())
            .collect()
    }

    /// `sum_s rows(s) (A, B)[s]` for row weights given per slot, the columns weighted by `eq(columns)`.
    fn row_sums(&self, columns: &[Challenge], mut rows: impl FnMut(usize, Challenge, Challenge)) {
        let columns = SplitEq::new(columns);
        self.program
            .walk_rows_lanes::<Challenge, 1>(|s| [columns.at(s)], |s, a, b| rows(s, a[0], b[0]));
    }
}

/// `h(v)` of a circuit form, for the arithmetic that defers it.
#[tracing::instrument(skip_all, name = "form rows")]
pub(crate) fn form_rows(
    matrix: &Matrix,
    skip: Challenge,
    inner: &[Challenge],
    point: &[Challenge],
    alpha: Challenge,
) -> Vec<Challenge> {
    matrix.form_rows(skip, inner, point, alpha)
}

/// `M(point)` on a circuit's stacked matrix: the native final check of a deferred claim.
///
/// # Panics
///
/// When the point does not name `1 + 2 k` coordinates for some `k >= k_log`.
pub fn evaluate_claim(circuit: &Circuit, point: &[Challenge]) -> Challenge {
    Matrix::new(circuit).evaluate(&[point.to_vec()])[0]
}

/// Bind values the verifier holds into the transcript, as their native values.
fn observe_all<A, C>(arith: &mut A, values: &[A::EF], challenger: &mut C)
where
    A: Lift<Val = Val, Challenge = Challenge>,
    C: FieldChallenger<Val>,
{
    for value in values {
        arith.will_observe_ext(value);
        challenger.observe_algebra_element(arith.value_ext(value));
    }
}

/// Draw `count` challenge-field coordinates.
fn sample_all<A, C>(arith: &mut A, count: usize, challenger: &mut C) -> Vec<A::EF>
where
    A: Lift<Val = Val, Challenge = Challenge>,
    C: FieldChallenger<Val>,
{
    (0..count)
        .map(|_| {
            let value: Challenge = challenger.sample_algebra_element();
            arith.sampled_ext(value)
        })
        .collect()
}

/// Reduce a circuit form to one claim on the stacked matrix, within an arithmetic.
///
/// The transcript binds `h`, draws the column bits `rho`, binds `g` and checks it against `h(rho)`, draws the row bits `rho'`, binds `A` and `B` and checks them against `g(rho')`, and draws the stacking coordinate.
///
/// # Errors
///
/// When a message has the wrong length or does not reproduce the claim it reduces.
pub fn reduce_form_in<A, C>(
    arith: &mut A,
    form: &PendingForm<A::EF>,
    proof: &FormProof,
    challenger: &mut C,
) -> Result<MatrixClaim<A::EF>, DeferralError>
where
    A: Lift<Val = Val, Challenge = Challenge>,
    C: FieldChallenger<Val>,
{
    if form.h.len() != SKIP_SIZE || proof.g.len() != SKIP_SIZE {
        return Err(DeferralError::Shape);
    }
    observe_all(arith, &form.h, challenger);
    let rho = sample_all(arith, K_SKIP, challenger);
    let h_rho = dot(&arith.eq_table(&rho), &form.h);

    let g: Vec<A::EF> = proof.g.iter().map(|&g| arith.load_ext(g)).collect();
    observe_all(arith, &g, challenger);
    let lagrange = lagrange_weights(&form.skip, SKIP_SIZE);
    arith.assert_zero(dot(&lagrange, &g) - h_rho, || DeferralError::Mismatch)?;
    let rho_rows = sample_all(arith, K_SKIP, challenger);
    let g_rho = dot(&arith.eq_table(&rho_rows), &g);

    let [a, b] = proof.ab.map(|v| arith.load_ext(v));
    observe_all(arith, &[a.dup(), b.dup()], challenger);
    arith.assert_zero(a.dup() + form.alpha.dup() * b.dup() - g_rho, || {
        DeferralError::Mismatch
    })?;
    let stack = sample_all(arith, 1, challenger).remove(0);
    let value = a.dup() + stack.dup() * (a + b);
    let point = core::iter::once(stack)
        .chain(form.inner.iter().map(Dup::dup))
        .chain(rho_rows)
        .chain(form.point.iter().map(Dup::dup))
        .chain(rho)
        .collect();
    Ok(MatrixClaim { point, value })
}

/// The prover's side of [`reduce_form_in`], on a copy of the transcript it will replay.
#[must_use]
#[tracing::instrument(skip_all, name = "prove form")]
pub fn prove_form<C: FieldChallenger<Val>>(
    matrix: &Matrix,
    form: &PendingForm<Challenge>,
    challenger: &mut C,
) -> FormProof {
    let observe = |values: &[Challenge], challenger: &mut C| {
        for &v in values {
            challenger.observe_algebra_element(v);
        }
    };
    let sample = |count: usize, challenger: &mut C| -> Vec<Challenge> {
        (0..count)
            .map(|_| challenger.sample_algebra_element())
            .collect()
    };
    observe(&form.h, challenger);
    let rho = sample(K_SKIP, challenger);
    let columns: Vec<Challenge> = form.point.iter().chain(&rho).copied().collect();
    // Rows sum per high entry of their weight, which scales each sum once.
    let zero = (Challenge::ZERO, Challenge::ZERO);
    let words = SplitEq::new(&form.inner);
    let mask = (1 << words.lo_bits) - 1;
    let mut sums = vec![[zero; SKIP_SIZE]; words.hi.len()];
    matrix.row_sums(&columns, |s, a, b| {
        let word = s / SKIP_SIZE;
        let low = words.lo[word & mask];
        let sum = &mut sums[word >> words.lo_bits][s % SKIP_SIZE];
        sum.0 += low * a;
        sum.1 += low * b;
    });
    let g: Vec<Challenge> = (0..SKIP_SIZE)
        .map(|v| {
            sums.iter()
                .zip(&words.hi)
                .map(|(s, &h)| h * (s[v].0 + form.alpha * s[v].1))
                .sum()
        })
        .collect();
    observe(&g, challenger);
    let rho_rows = sample(K_SKIP, challenger);
    let row_point: Vec<Challenge> = form.inner.iter().chain(&rho_rows).copied().collect();
    let rows = SplitEq::new(&row_point);
    let mask = (1 << rows.lo_bits) - 1;
    let mut sums = vec![zero; rows.hi.len()];
    matrix.row_sums(&columns, |s, a, b| {
        let low = rows.lo[s & mask];
        let sum = &mut sums[s >> rows.lo_bits];
        sum.0 += low * a;
        sum.1 += low * b;
    });
    let ab = sums
        .iter()
        .zip(&rows.hi)
        .fold([Challenge::ZERO; 2], |[x, y], (s, &h)| {
            [x + h * s.0, y + h * s.1]
        });
    observe(&ab, challenger);
    let _stack: Vec<Challenge> = sample(1, challenger);
    FormProof { g, ab }
}

/// `sum_i a_i b_i`.
fn dot<E: PrimeCharacteristicRing>(a: &[E], b: &[E]) -> E {
    a.iter().zip(b).map(|(x, y)| x.dup() * y.dup()).sum()
}

/// Fold two claims on one matrix into one, on the line through their points, within an arithmetic.
///
/// `q` is the restriction of the matrix to the line `l(t) = x1 + t (x1 + x2)`, by its coefficients, lowest first.
///
/// # Errors
///
/// When the points differ in length, `q` has the wrong degree, or `q(0)`, `q(1)` miss the two values.
pub fn fold_claims_in<A, C>(
    arith: &mut A,
    first: &MatrixClaim<A::EF>,
    second: &MatrixClaim<A::EF>,
    q: &[Challenge],
    challenger: &mut C,
) -> Result<MatrixClaim<A::EF>, DeferralError>
where
    A: Lift<Val = Val, Challenge = Challenge>,
    C: FieldChallenger<Val>,
{
    let n = first.point.len();
    if second.point.len() != n || q.len() != n + 1 {
        return Err(DeferralError::Shape);
    }
    let q: Vec<A::EF> = q.iter().map(|&c| arith.load_ext(c)).collect();
    observe_all(arith, &q, challenger);
    arith.assert_zero(q[0].dup() - first.value.dup(), || DeferralError::Mismatch)?;
    let at_one: A::EF = q.iter().map(Dup::dup).sum();
    arith.assert_zero(at_one - second.value.dup(), || DeferralError::Mismatch)?;
    let t = sample_all(arith, 1, challenger).remove(0);
    let value = q
        .iter()
        .rev()
        .fold(A::EF::ZERO, |acc, c| acc * t.dup() + c.dup());
    let point = first
        .point
        .iter()
        .zip(&second.point)
        .map(|(x, y)| x.dup() + t.dup() * (x.dup() + y.dup()))
        .collect();
    Ok(MatrixClaim { point, value })
}

/// The prover's side of [`fold_claims_in`]: the coefficients of `M` on the line through the two points.
///
/// The restriction has degree at most the number of coordinates the points differ in, so that many plus one evaluations suffice.
#[must_use]
#[tracing::instrument(skip_all, name = "line polynomial")]
pub fn line_polynomial(
    matrix: &Matrix,
    first: &MatrixClaim,
    second: &MatrixClaim,
) -> Vec<Challenge> {
    let n = first.point.len();
    let degree = first
        .point
        .iter()
        .zip(&second.point)
        .filter(|(x, y)| x != y)
        .count();
    let nodes: Vec<Challenge> = (0..=degree as u64)
        .map(|i| Challenge::from(Val::new(i)))
        .collect();
    let points: Vec<Vec<Challenge>> = nodes
        .iter()
        .map(|&t| {
            first
                .point
                .iter()
                .zip(&second.point)
                .map(|(&x, &y)| x + t * (x + y))
                .collect()
        })
        .collect();
    let values = matrix.evaluate(&points);
    let mut coefficients = interpolate_coefficients(&nodes, &values);
    coefficients.resize(n + 1, Challenge::ZERO);
    coefficients
}

/// The coefficients, lowest first, of the polynomial of degree below `nodes.len()` through the points.
fn interpolate_coefficients(nodes: &[Challenge], values: &[Challenge]) -> Vec<Challenge> {
    let n = nodes.len();
    let mut out = vec![Challenge::ZERO; n];
    for (i, (&xi, &yi)) in nodes.iter().zip(values).enumerate() {
        // The basis polynomial prod_{j != i} (t - x_j) / (x_i - x_j), expanded.
        let mut basis = vec![Challenge::ONE];
        let mut denominator = Challenge::ONE;
        for (j, &xj) in nodes.iter().enumerate() {
            if j == i {
                continue;
            }
            let mut next = vec![Challenge::ZERO; basis.len() + 1];
            for (k, &c) in basis.iter().enumerate() {
                next[k + 1] += c;
                next[k] += c * xj;
            }
            basis = next;
            denominator *= xi - xj;
        }
        let scale = yi * denominator.inverse();
        for (o, c) in out.iter_mut().zip(basis) {
            *o += scale * c;
        }
    }
    out
}

/// The native value of a pending form, for the prover.
#[must_use]
pub fn native_form<A: Lift<Val = Val, Challenge = Challenge>>(
    arith: &A,
    form: &PendingForm<A::EF>,
) -> PendingForm<Challenge> {
    let native = |v: &[A::EF]| v.iter().map(|x| arith.value_ext(x)).collect::<Vec<_>>();
    PendingForm {
        k_log: form.k_log,
        skip: arith.value_ext(&form.skip),
        inner: native(&form.inner),
        point: native(&form.point),
        alpha: arith.value_ext(&form.alpha),
        h: native(&form.h),
    }
}
