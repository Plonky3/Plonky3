//! Sumcheck helpers: variable ordering, round coefficients, and the prover state.
//!
//! # Layout
//!
//! - `sumcheck_coefficients_{prefix,suffix}`: the two round-coefficient routines.
//! - `VariableOrder`: tag enum carrying inherent methods that dispatch to either routine.
//! - `SumcheckProver`: drives rounds over a paired product polynomial.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePackedView};

use crate::constraints::{Constraint, Statements};
use crate::product_polynomial::ProductPolynomial;
use crate::{SumcheckData, extrapolate_01inf};

/// Input size at which the round-coefficient routines switch from serial to parallel execution.
///
/// # Why this value
///
/// - Below `2^14` paired elements, the rayon splitting and join overhead outweighs the parallel work.
/// - Above it, the fold-reduce amortises the splitting cost.
const PAR_THRESHOLD: usize = 1 << 14;

/// Tile size for the chunked round-coefficient kernel.
///
/// On Monty-31 packings, hand-written delayed-reduction primitives exist for tile sizes `2, 4, 5, 8`;
///
/// `8` is the deepest available on every supported target.
///
/// - Larger overruns the integer-multiply pipeline depth;
/// - Smaller dilutes the delayed-reduction win.
const K: usize = 8;

/// Per-tile MAC: extends a `(constant, leading)` accumulator pair.
///
/// # Algorithm
///
/// Folding the active variable in `h(X) = sum_b f(X, b) * w(X, b)` gives:
///
/// ```text
///     constant += sum_i  w_lo[i] * e_lo[i]
///     leading  += sum_i  (w_hi[i] - w_lo[i]) * (e_hi[i] - e_lo[i])
/// ```
///
/// where `lo`, `hi` are the two faces of the active variable. Each sum is
/// one delayed-reduction dot product over `K` pairs, collapsing `K`
/// widening multiplies into one Montgomery reduce per output coordinate.
#[inline(always)]
fn chunk_round_step<B, A>(e_lo: &[B; K], e_hi: &[B; K], w_lo: &[A; K], w_hi: &[A; K]) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy,
    A: Algebra<B> + Copy,
{
    // Constant term: one delayed-reduction dot product over the b_0 = 0 face.
    let acc0 = A::mixed_dot_product::<K>(w_lo, e_lo);

    // Materialise the differences (b_0 = 1 minus b_0 = 0) tile-locally so
    // they can feed the same primitive. `K` base subs, no reductions.
    let diffs_e: [B; K] = core::array::from_fn(|i| e_hi[i] - e_lo[i]);
    let diffs_w: [A; K] = core::array::from_fn(|i| w_hi[i] - w_lo[i]);

    // Leading coefficient: dot product of the differences.
    let acc_inf = A::mixed_dot_product::<K>(&diffs_w, &diffs_e);

    (acc0, acc_inf)
}

/// Per-pair MAC for the streaming tail (at most `K - 1` leftover pairs).
#[inline(always)]
fn round_step<B, A>((acc0, acc_inf): (A, A), e0: B, e1: B, w0: A, w1: A) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy,
    A: Algebra<B> + Copy,
{
    (acc0 + w0 * e0, acc_inf + (w1 - w0) * (e1 - e0))
}

/// Component-wise sum of two `(constant, leading)` accumulator pairs.
#[inline(always)]
fn round_reduce<A: Copy + PrimeCharacteristicRing>(a: (A, A), b: (A, A)) -> (A, A) {
    (a.0 + b.0, a.1 + b.1)
}

/// Projective per-tile MAC (eprint 2026/762, Fig. 3).
///
/// Like [`chunk_round_step`], but the tables are interpreted as monomial
/// coefficients, so the round message is `[s(1), s(inf)]` and the verifier
/// derives `s(0) := C - s(inf)` from the projective round identity. The
/// `X = 1` evaluation of a coefficient pair is `lo + hi`, so the differences
/// of the evaluation basis become sums:
///
/// ```text
///     at_one  += sum_i  (w_lo[i] + w_hi[i]) * (e_lo[i] + e_hi[i])
///     leading += sum_i  w_hi[i] * e_hi[i]
/// ```
#[inline(always)]
fn chunk_round_step_projective<B, A>(
    e_lo: &[B; K],
    e_hi: &[B; K],
    w_lo: &[A; K],
    w_hi: &[A; K],
) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy,
    A: Algebra<B> + Copy,
{
    // Materialise the X = 1 evaluations (lo + hi) tile-locally so they can
    // feed the same delayed-reduction primitive. `K` base adds, no reductions.
    let ones_e: [B; K] = core::array::from_fn(|i| e_lo[i] + e_hi[i]);
    let ones_w: [A; K] = core::array::from_fn(|i| w_lo[i] + w_hi[i]);

    let acc1 = A::mixed_dot_product::<K>(&ones_w, &ones_e);

    // Leading coefficient: dot product of the high (coefficient) faces.
    let acc_inf = A::mixed_dot_product::<K>(w_hi, e_hi);

    (acc1, acc_inf)
}

/// Projective per-pair MAC for the streaming tail (no subtraction).
#[inline(always)]
fn round_step_projective<B, A>((acc1, acc_inf): (A, A), e0: B, e1: B, w0: A, w1: A) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy,
    A: Algebra<B> + Copy,
{
    (acc1 + (w0 + w1) * (e0 + e1), acc_inf + w1 * e1)
}

/// The two prover-sent values of a quadratic sumcheck round.
///
/// The round polynomial has three unknowns; the verifier recovers the third
/// from the running claim `C`, so only two values cross the transcript. Which
/// finite point is sent differs by basis:
///
/// | basis      | `c_a`  | `c_inf`  | derived             |
/// |------------|--------|----------|---------------------|
/// | evaluation | `h(0)` | `h(inf)` | `h(1) = C - h(0)`   |
/// | projective | `s(1)` | `s(inf)` | `s(0) = C - s(inf)` |
///
/// The struct itself is basis-agnostic: `c_a` has the same name and type in
/// both rows, so it carries no evidence of which kernel produced it. What
/// keeps the rows from being crossed is that a message is only ever produced
/// by [`Basis::sumcheck_coefficients`] and only ever consumed by
/// [`Basis::reduce_claim`], each under the same tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundMessage<A> {
    /// The finite-point value: `h(0)` (evaluation basis) or `s(1)` (projective).
    pub c_a: A,
    /// The leading coefficient of the round polynomial: `h(inf)` / `s(inf)`.
    pub c_inf: A,
}

/// Shared prefix round-coefficient scaffold, parameterised over the basis steps.
///
/// The tiling, `PAR_THRESHOLD` par-vs-serial split, and `K`-tail fold are
/// identical across bases; only the per-tile and per-pair MAC differ. The
/// evaluation and projective kernels supply their `chunk_step` / `pair_step`
/// pair so this delicate delayed-reduction loop exists exactly once.
#[inline]
fn sumcheck_coefficients_prefix_with<B, A, Chunk, Pair>(
    evals: &[B],
    weights: &[A],
    chunk_step: Chunk,
    pair_step: Pair,
) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<B> + Copy + Send + Sync,
    Chunk: Fn(&[B; K], &[B; K], &[A; K], &[A; K]) -> (A, A) + Sync,
    Pair: Fn((A, A), B, B, A, A) -> (A, A),
{
    // Precondition: paired slices must be aligned; half-and-half split addresses the prefix bit.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len().is_multiple_of(2));
    let half = evals.len() / 2;
    let (e_lo, e_hi) = evals.split_at(half);
    let (w_lo, w_hi) = weights.split_at(half);

    let body = (half / K) * K;
    let (e_lo_main, e_lo_tail) = e_lo.split_at(body);
    let (e_hi_main, e_hi_tail) = e_hi.split_at(body);
    let (w_lo_main, w_lo_tail) = w_lo.split_at(body);
    let (w_hi_main, w_hi_tail) = w_hi.split_at(body);

    // Main chunked loop: K pairs per iteration via delayed-reduction dot products.
    let main: (A, A) = if half > PAR_THRESHOLD {
        e_lo_main
            .par_chunks_exact(K)
            .zip(e_hi_main.par_chunks_exact(K))
            .zip(
                w_lo_main
                    .par_chunks_exact(K)
                    .zip(w_hi_main.par_chunks_exact(K)),
            )
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |acc, ((e_lo_c, e_hi_c), (w_lo_c, w_hi_c))| {
                    let chunk = chunk_step(
                        e_lo_c.try_into().unwrap(),
                        e_hi_c.try_into().unwrap(),
                        w_lo_c.try_into().unwrap(),
                        w_hi_c.try_into().unwrap(),
                    );
                    round_reduce(acc, chunk)
                },
                round_reduce,
            )
    } else {
        e_lo_main
            .as_chunks::<K>()
            .0
            .iter()
            .zip(e_hi_main.as_chunks::<K>().0.iter())
            .zip(
                w_lo_main
                    .as_chunks::<K>()
                    .0
                    .iter()
                    .zip(w_hi_main.as_chunks::<K>().0.iter()),
            )
            .fold(
                (A::ZERO, A::ZERO),
                |acc, ((e_lo_c, e_hi_c), (w_lo_c, w_hi_c))| {
                    let chunk = chunk_step(e_lo_c, e_hi_c, w_lo_c, w_hi_c);
                    round_reduce(acc, chunk)
                },
            )
    };

    // Tail: at most K-1 pairs; streaming fold with eager reduction is fine.
    let tail = e_lo_tail
        .iter()
        .zip(e_hi_tail.iter())
        .zip(w_lo_tail.iter().zip(w_hi_tail.iter()))
        .fold((A::ZERO, A::ZERO), |acc, ((&e0, &e1), (&w0, &w1))| {
            pair_step(acc, e0, e1, w0, w1)
        });

    round_reduce(main, tail)
}

/// Computes the round message for a prefix-binding sumcheck round.
///
/// # Inputs
///
/// - `evals`   — multilinear evaluations of `f(X)` over the hypercube.
/// - `weights` — multilinear evaluations of `w(X)` over the hypercube.
///
/// # Returns
///
/// - `c_a` = `h(0)`     = sum_{b in {0,1}^{n-1}} f(0, b) * w(0, b)
/// - `c_inf` = `h(inf)` = sum_{b} (f(1, b) - f(0, b)) * (w(1, b) - w(0, b))
///
/// # Complexity
///
/// O(2^n). Parallelised above a 2^14 threshold. The main loop is tiled by
/// `K` over a delayed-reduction dot product; the `half mod K` tail uses a
/// streaming fold.
pub fn sumcheck_coefficients_prefix<B, A>(evals: &[B], weights: &[A]) -> RoundMessage<A>
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<B> + Copy + Send + Sync,
{
    let (c_a, c_inf) = sumcheck_coefficients_prefix_with(
        evals,
        weights,
        chunk_round_step::<B, A>,
        round_step::<B, A>,
    );
    RoundMessage { c_a, c_inf }
}

/// Projective (monomial-basis) variant of [`sumcheck_coefficients_prefix`]
/// (eprint 2026/762, Fig. 3).
///
/// The tables are interpreted as monomial coefficients. The round message is
/// `[s(1), s(inf)]`; the verifier derives `s(0) := C - s(inf)` from the
/// projective round identity `s(0) + s(inf) = C`. Returned as:
///
/// - `c_a` = `s(1)`     = sum_{b} (w(0,b) + w(inf,b)) * (f(0,b) + f(inf,b))
/// - `c_inf` = `s(inf)` = sum_{b} w(inf, b) * f(inf, b)   (leading coefficient)
///
/// The evaluation-basis kernel's per-pair subtractions (`hi - lo`) become
/// additions (`lo + hi`); same `K`-tiled, delayed-reduction structure.
pub fn sumcheck_coefficients_prefix_projective<B, A>(evals: &[B], weights: &[A]) -> RoundMessage<A>
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<B> + Copy + Send + Sync,
{
    let (c_a, c_inf) = sumcheck_coefficients_prefix_with(
        evals,
        weights,
        chunk_round_step_projective::<B, A>,
        round_step_projective::<B, A>,
    );
    RoundMessage { c_a, c_inf }
}

/// Target byte size of one bound face inside a fused block.
///
/// A block binds four faces, then measures them straight afterwards.
/// All four have to stay in first-level cache across those two steps.
///
/// The budget is in bytes rather than elements.
/// A SIMD-packed element is an order of magnitude wider than a scalar one.
const FUSED_BLOCK_BYTES: usize = 4096;

/// Number of index positions one fused block binds before measuring them.
///
/// Never below the tile width, so a measurement always gets one full tile.
#[inline]
const fn fused_block<A>() -> usize {
    // Positions that fit the byte budget at this element width.
    let by_bytes = FUSED_BLOCK_BYTES / core::mem::size_of::<A>();

    // A wide element can exhaust the budget below one tile.
    // The tile width wins there.
    if by_bytes < K { K } else { by_bytes }
}

/// Binds one face of a table in place.
///
/// The destination holds the round variable at 0.
/// The source holds it at 1.
///
/// Each entry becomes the line through the two, sampled at the challenge.
#[inline]
fn bind_face<A, Ch>(dst: &mut [A], src: &[A], r: Ch)
where
    A: Algebra<Ch> + Copy,
    Ch: Copy,
{
    // The bound value overwrites the 0 face.
    // The 1 face is only read.
    for (lo, &hi) in dst.iter_mut().zip(src) {
        *lo += (hi - *lo) * r;
    }
}

/// Round message of a pair already split into the two faces of the round variable.
///
/// The measuring scaffold with the split hoisted out.
/// Parallelism is left to the caller, which owns the outer loop.
///
/// The tiled body and the streaming tail are the ones the unsplit routine uses.
#[inline]
fn round_coefficients_faces<A>(e_lo: &[A], e_hi: &[A], w_lo: &[A], w_hi: &[A]) -> (A, A)
where
    A: Algebra<A> + Copy,
{
    // Whole tiles first, leftovers after.
    // The four faces split at the same place.
    let (e_lo_main, e_lo_tail) = e_lo.as_chunks::<K>();
    let (e_hi_main, e_hi_tail) = e_hi.as_chunks::<K>();
    let (w_lo_main, w_lo_tail) = w_lo.as_chunks::<K>();
    let (w_hi_main, w_hi_tail) = w_hi.as_chunks::<K>();

    // Main loop: K pairs per iteration through delayed-reduction dot products.
    let main = e_lo_main
        .iter()
        .zip(e_hi_main)
        .zip(w_lo_main.iter().zip(w_hi_main))
        .fold(
            (A::ZERO, A::ZERO),
            |acc, ((e_lo_c, e_hi_c), (w_lo_c, w_hi_c))| {
                round_reduce(acc, chunk_round_step(e_lo_c, e_hi_c, w_lo_c, w_hi_c))
            },
        );

    // Tail: fewer than K pairs, so a streaming fold with eager reduction is fine.
    let tail = e_lo_tail
        .iter()
        .zip(e_hi_tail)
        .zip(w_lo_tail.iter().zip(w_hi_tail))
        .fold((A::ZERO, A::ZERO), |acc, ((&e0, &e1), (&w0, &w1))| {
            round_step(acc, e0, e1, w0, w1)
        });

    round_reduce(main, tail)
}

/// Binds a prefix variable and measures the bound pair's round message in one pass.
///
/// # Overview
///
/// The bound tables land in the lower half of each input.
/// The inputs keep their original length, so the caller drops the upper halves itself.
///
/// The message is the one a separate measuring pass over the bound tables returns.
///
/// # Algorithm
///
/// The pass touches two variables at once.
/// One is bound now.
/// The other is the one the returned message sums over.
///
/// Together they cut each table into four quadrants:
///
/// ```text
///     bound = 0, summed = 0 : q0        bound = 1, summed = 0 : q2
///     bound = 0, summed = 1 : q1        bound = 1, summed = 1 : q3
/// ```
///
/// Binding leaves the two faces of the variable still to be summed:
///
/// ```text
///     lo = q0 + (q2 - q0) * r     written back over q0
///     hi = q1 + (q3 - q1) * r     written back over q1
/// ```
///
/// Those two faces are what the message needs.
/// Working a block at a time keeps them in cache across the two steps.
///
/// The bound table is therefore written once and never read back from memory.
///
/// # Arguments
///
/// - `evals` - evaluation table, before this binding.
/// - `weights` - weight table, before this binding.
/// - `r` - challenge the round variable binds to.
///
/// # Returns
///
/// - `c_a` - the bound pair's round polynomial at 0.
/// - `c_inf` - its leading coefficient.
///
/// # Performance
///
/// O(2^n), at the same multiply count as binding and measuring separately.
/// What it saves is one pass over the bound tables.
///
/// # Panics
///
/// - The two tables must have the same length.
/// - The length must be a multiple of four.
///   The bound table then keeps the variable the message sums over.
pub fn fold_and_round_coefficients_prefix<A, Ch>(
    evals: &mut [A],
    weights: &mut [A],
    r: Ch,
) -> RoundMessage<A>
where
    A: Algebra<Ch> + Copy + Send + Sync,
    Ch: Copy + Send + Sync,
{
    // Precondition: paired tables, with a variable left over for the message.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len().is_multiple_of(4));
    let evals_len = evals.len();

    // Cut each table into the four quadrants of the two variables this pass touches.
    //
    //     [ q0 | q1 | q2 | q3 ]
    //       written   only read
    let quarter = evals.len() / 4;
    let (e_bound, e_free) = evals.split_at_mut(2 * quarter);
    let (e_q0, e_q1) = e_bound.split_at_mut(quarter);
    let (e_q2, e_q3) = e_free.split_at(quarter);
    let (w_bound, w_free) = weights.split_at_mut(2 * quarter);
    let (w_q0, w_q1) = w_bound.split_at_mut(quarter);
    let (w_q2, w_q3) = w_free.split_at(quarter);

    // One block: bind the four faces, then measure them while they are still hot.
    let block = |e_q0: &mut [A],
                 e_q1: &mut [A],
                 e_q2: &[A],
                 e_q3: &[A],
                 w_q0: &mut [A],
                 w_q1: &mut [A],
                 w_q2: &[A],
                 w_q3: &[A]| {
        bind_face(e_q0, e_q2, r);
        bind_face(e_q1, e_q3, r);
        bind_face(w_q0, w_q2, r);
        bind_face(w_q1, w_q3, r);
        round_coefficients_faces(e_q0, e_q1, w_q0, w_q1)
    };

    let len = fused_block::<A>();

    // The pass covers the whole table, not just the bound half.
    //
    // So the par-vs-serial split is gated on the whole table.
    // That puts about as much work in one task as a measuring pass does at its own gate.
    let (c_a, c_inf) = if evals_len > PAR_THRESHOLD {
        e_q0.par_chunks_mut(len)
            .zip(e_q1.par_chunks_mut(len))
            .zip(e_q2.par_chunks(len))
            .zip(e_q3.par_chunks(len))
            .zip(w_q0.par_chunks_mut(len))
            .zip(w_q1.par_chunks_mut(len))
            .zip(w_q2.par_chunks(len))
            .zip(w_q3.par_chunks(len))
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |acc, (((((((e0, e1), e2), e3), w0), w1), w2), w3)| {
                    round_reduce(acc, block(e0, e1, e2, e3, w0, w1, w2, w3))
                },
                round_reduce,
            )
    } else {
        e_q0.chunks_mut(len)
            .zip(e_q1.chunks_mut(len))
            .zip(e_q2.chunks(len))
            .zip(e_q3.chunks(len))
            .zip(w_q0.chunks_mut(len))
            .zip(w_q1.chunks_mut(len))
            .zip(w_q2.chunks(len))
            .zip(w_q3.chunks(len))
            .fold(
                (A::ZERO, A::ZERO),
                |acc, (((((((e0, e1), e2), e3), w0), w1), w2), w3)| {
                    round_reduce(acc, block(e0, e1, e2, e3, w0, w1, w2, w3))
                },
            )
    };

    RoundMessage { c_a, c_inf }
}

/// Computes the round message for a suffix-binding sumcheck round.
///
/// # Inputs
///
/// - `evals`   — multilinear evaluations of `f(X)` over the hypercube.
/// - `weights` — multilinear evaluations of `w(X)` over the hypercube.
///
/// # Returns
///
/// - `c_a` = `h(0)`     = sum_{b in {0,1}^{n-1}} f(b, 0) * w(b, 0)
/// - `c_inf` = `h(inf)` = sum_{b} (f(b, 1) - f(b, 0)) * (w(b, 1) - w(b, 0))
///
/// # Complexity
///
/// O(2^n). Parallelised above a 2^14 threshold. The main loop walks the
/// buffer in `2K`-wide chunks: each chunk gathers `K` adjacent
/// `(b_n=0, b_n=1)` pairs and dispatches to a delayed-reduction dot
/// product.
pub fn sumcheck_coefficients_suffix<B, A>(evals: &[B], weights: &[A]) -> RoundMessage<A>
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<B> + Copy + Send + Sync,
{
    // Precondition: paired slices must be aligned; adjacent pairs address the suffix bit.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len().is_multiple_of(2));

    let half = evals.len() / 2;
    // Each chunk consumes 2K consecutive elements (K pairs).
    let body_pairs = (half / K) * K;
    let body_elems = body_pairs * 2;
    let (evals_main, evals_tail) = evals.split_at(body_elems);
    let (weights_main, weights_tail) = weights.split_at(body_elems);

    #[inline(always)]
    fn gather_pairs<T: Copy>(chunk: &[T]) -> ([T; K], [T; K]) {
        // Layout: [t0, t1, t2, t3, ...]; even indices = "0", odd indices = "1".
        let lo: [T; K] = core::array::from_fn(|i| chunk[2 * i]);
        let hi: [T; K] = core::array::from_fn(|i| chunk[2 * i + 1]);
        (lo, hi)
    }

    let main: (A, A) = if evals.len() > PAR_THRESHOLD {
        evals_main
            .par_chunks_exact(2 * K)
            .zip(weights_main.par_chunks_exact(2 * K))
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |acc, (e_chunk, w_chunk)| {
                    let (e_lo, e_hi) = gather_pairs::<B>(e_chunk);
                    let (w_lo, w_hi) = gather_pairs::<A>(w_chunk);
                    let chunk = chunk_round_step::<B, A>(&e_lo, &e_hi, &w_lo, &w_hi);
                    round_reduce(acc, chunk)
                },
                round_reduce,
            )
    } else {
        evals_main
            .as_chunks::<{ 2 * K }>()
            .0
            .iter()
            .zip(weights_main.as_chunks::<{ 2 * K }>().0.iter())
            .fold((A::ZERO, A::ZERO), |acc, (e_chunk, w_chunk)| {
                let (e_lo, e_hi) = gather_pairs::<B>(e_chunk);
                let (w_lo, w_hi) = gather_pairs::<A>(w_chunk);
                let chunk = chunk_round_step::<B, A>(&e_lo, &e_hi, &w_lo, &w_hi);
                round_reduce(acc, chunk)
            })
    };

    // Tail: at most K-1 pairs; streaming fold over adjacent (0,1) chunks.
    let tail = evals_tail
        .chunks(2)
        .zip(weights_tail.chunks(2))
        .fold((A::ZERO, A::ZERO), |acc, (e, w)| {
            round_step(acc, e[0], e[1], w[0], w[1])
        });

    let (c_a, c_inf) = round_reduce(main, tail);
    RoundMessage { c_a, c_inf }
}

/// How the sumcheck tables are interpreted, and with it the round arithmetic.
///
/// The table bytes are identical in both bases; the tag selects which
/// polynomial those bytes describe, and one round arithmetic follows from
/// each choice (eprint 2026/762, Section 3):
///
/// | per round                  | [`Basis::Evaluation`]              | [`Basis::Projective`]                           |
/// |----------------------------|------------------------------------|-------------------------------------------------|
/// | a table entry is           | a value on the hypercube `{0,1}^n` | a monomial coefficient (a value on `{0,inf}^n`) |
/// | binding `X = r`            | `a0 + (a1 - a0) * r`               | `a0 + a1 * r`                                   |
/// | message sent               | `[h(0), h(inf)]`                   | `[s(1), s(inf)]`                                |
/// | value the verifier derives | `h(1) := C - h(0)`                 | `s(0) := C - s(inf)`                            |
///
/// The rows are one package per column: a consumer must take an entire
/// column, never a mix. That is what the tag buys. A [`RoundMessage`] alone
/// cannot say which column produced it, so the two values only ever leave or
/// re-enter the transcript through the basis that defines them.
///
/// The claim invariant `C = dot(evals, weights)` is the same in both bases:
/// the `{0,1}`-sum of products in the evaluation basis and the `{0,inf}`-sum
/// in the projective basis are both the dot product of the two tables, so
/// the running-sum bookkeeping does not change. Like [`VariableOrder`], the
/// tag is consulted once per round in the outer frame, never inside the
/// O(2^n) inner loops.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Basis {
    /// The tables hold values over the boolean hypercube `{0,1}^n`; rounds
    /// sum over `{0,1}` and bind by linear interpolation. The default.
    #[default]
    Evaluation,
    /// The tables hold monomial coefficients, equivalently values over
    /// `{0,inf}^n`; rounds sum over `{0,inf}` and bind subtraction-free
    /// (eprint 2026/762).
    ///
    /// Prefix order only: the projective kernels are implemented for
    /// prefix-bound variables (WHIR's path).
    Projective,
}

impl Basis {
    /// Computes the two-element round message for one quadratic sumcheck round.
    ///
    /// - [`Basis::Evaluation`]: `[h(0), h(inf)]`, dispatching on `order`.
    /// - [`Basis::Projective`]: `[s(1), s(inf)]` (prefix only); the verifier
    ///   derives `s(0) := C - s(inf)` from the projective round identity.
    ///
    /// # Panics
    ///
    /// Panics if the projective basis is paired with suffix binding.
    pub fn sumcheck_coefficients<B, A>(
        self,
        order: VariableOrder,
        evals: &[B],
        weights: &[A],
    ) -> RoundMessage<A>
    where
        B: PrimeCharacteristicRing + Copy + Send + Sync,
        A: Algebra<B> + Copy + Send + Sync,
    {
        match self {
            Self::Evaluation => order.sumcheck_coefficients(evals, weights),
            Self::Projective => {
                assert_eq!(
                    order,
                    VariableOrder::Prefix,
                    "the projective basis is prefix-only"
                );
                sumcheck_coefficients_prefix_projective(evals, weights)
            }
        }
    }

    /// Reduces the running claim to the round polynomial at `r`, from the two
    /// sent message elements.
    ///
    /// One source of truth for the round identity, shared by the prover and
    /// the verifier:
    ///
    /// - [`Basis::Evaluation`]: message `[h(0), h(inf)]`; the identity
    ///   `h(0) + h(1) = C` supplies `h(1) = C - h(0)`.
    /// - [`Basis::Projective`]: message `[s(1), s(inf)]`; the identity
    ///   `s(0) + s(inf) = C` supplies `s(0) = C - s(inf)`
    ///   (eprint 2026/762, Fig. 3).
    ///
    /// Both reconstruct the quadratic through `{0, 1, inf}` and evaluate it
    /// at `r`. Pairing a message with the wrong basis reduces to the wrong
    /// claim, so this is the only place either identity is written down.
    pub fn reduce_claim<EF: Field>(self, c_a: EF, c_inf: EF, r: EF, claimed_sum: EF) -> EF {
        match self {
            Self::Evaluation => extrapolate_01inf(c_a, claimed_sum - c_a, c_inf, r),
            Self::Projective => extrapolate_01inf(claimed_sum - c_inf, c_a, c_inf, r),
        }
    }
}

/// Which side of the variable order is bound first by the sumcheck rounds.
///
/// # Role
///
/// - Round-coefficient math differs in which axis is summed over.
/// - Variable binding differs in which coordinate is fixed to the challenge.
/// - Verifier constraint evaluation differs in how the final challenge is spliced.
///
/// All three dispatches go through inherent methods below, so the runtime
/// branch sits in the outer frame and never inside the O(2^n) inner loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableOrder {
    /// Prefix variables are bound first (round `i` binds `X_i`).
    Prefix,
    /// Suffix variables are bound first (round `i` binds `X_{n-i}`).
    Suffix,
}

impl VariableOrder {
    /// Computes the [`RoundMessage`] for one quadratic sumcheck round.
    pub fn sumcheck_coefficients<B, A>(self, evals: &[B], weights: &[A]) -> RoundMessage<A>
    where
        B: PrimeCharacteristicRing + Copy + Send + Sync,
        A: Algebra<B> + Copy + Send + Sync,
    {
        match self {
            Self::Prefix => sumcheck_coefficients_prefix(evals, weights),
            Self::Suffix => sumcheck_coefficients_suffix(evals, weights),
        }
    }

    /// Binds the active round variable of `poly` to challenge `r`.
    pub fn fix_var<A, Ch>(self, poly: &mut Poly<A>, r: Ch)
    where
        A: Algebra<Ch> + Copy + Send + Sync,
        Ch: Copy + Send + Sync,
    {
        match self {
            Self::Prefix => poly.fix_prefix_var_mut(r),
            Self::Suffix => poly.fix_suffix_var_mut(r),
        }
    }

    /// Evaluates the batched verifier constraints at the final challenge point.
    ///
    /// # Slicing rule
    ///
    /// - Prefix binding folds variables low-to-high, so each constraint sees
    ///   the last `k` original variables of the challenge.
    /// - Suffix binding folds variables high-to-low, so each constraint sees
    ///   the last `k` original variables of the challenge, reversed.
    pub fn eval_constraints_poly<F, EF>(
        self,
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // Reverse once outside the per-constraint loop; both branches reuse it.
        let reversed = challenge.reversed();

        constraints
            .iter()
            .map(|constraint| {
                // Slice the reversed challenge to the constraint arity; flip back for prefix binding.
                let local_challenge = match self {
                    Self::Prefix => reversed
                        .get_subpoint_over_range(..constraint.num_variables())
                        .reversed(),
                    Self::Suffix => reversed.get_subpoint_over_range(..constraint.num_variables()),
                };

                // The batched weight polynomial is one big random combination
                // of all statement weights against successive challenge powers.
                //
                //     value = sum_g sum_i weight_{g,i} * chi^{shift_g + i}
                //
                // Each statement group contributes a contiguous block of powers.
                // The running shift is where the next group's powers begin.
                //
                //     group 0: chi^0       chi^1   ... chi^{l_0 - 1}
                //     group 1: chi^{l_0}   ...         chi^{l_0 + l_1 - 1}
                //     ...
                let mut shift = 0;
                let mut acc = EF::ZERO;
                // Each statement group exposes its weights evaluated at the
                // local challenge; the kinds differ only in how weights are formed.
                for statement in constraint.statements() {
                    match statement {
                        // Equality weights: one term per recorded equality point.
                        Statements::Eq(eq_statement) => {
                            // Pair this group's weights with powers starting at the shift.
                            acc += dot_product::<EF, _, _>(
                                eq_statement.weights_at(&local_challenge),
                                constraint.challenge_powers(shift),
                            );
                        }
                        // Successor-view weights: equality through the repeat-last view.
                        Statements::Next(next_statement) => {
                            acc += dot_product::<EF, _, _>(
                                next_statement.weights_at(&local_challenge),
                                constraint.challenge_powers(shift),
                            );
                        }
                        // Selector weights: one term per single-variable selector.
                        Statements::Select(sel_statement) => {
                            acc += dot_product::<EF, _, _>(
                                sel_statement.weights_at(&local_challenge),
                                constraint.challenge_powers(shift),
                            );
                        }
                    }
                    // Advance past this group's block so the next group's powers
                    // begin one beyond the last power consumed here.
                    shift += statement.len();
                }
                acc
            })
            .sum()
    }
}

/// Sumcheck prover: drives rounds of the quadratic sumcheck protocol.
///
/// # Invariant
///
/// At every point during the protocol:
///
/// ```text
///     sum == sum_{x in {0,1}^n} f(x) * w(x)
/// ```
///
/// where `n` is the number of remaining unbound variables. It decreases by
/// one per round as variables are bound to verifier challenges.
#[derive(Debug, Clone)]
pub struct SumcheckProver<F: Field, EF: ExtensionField<F>> {
    /// Paired evaluation and weight polynomials for the quadratic sumcheck.
    poly: ProductPolynomial<F, EF>,
    /// Current claimed sum over the remaining unbound variables.
    sum: EF,
}

impl<F: Field, EF: ExtensionField<F>> SumcheckProver<F, EF> {
    /// Creates a prover state from a product polynomial and its claimed sum.
    pub fn new(poly: ProductPolynomial<F, EF>, sum: EF) -> Self {
        // Sanity: the claimed sum must match the polynomial pair's dot product.
        debug_assert_eq!(poly.dot_product(), sum);
        Self { poly, sum }
    }

    /// Returns the current claimed sum over the remaining unbound variables.
    pub const fn claimed_sum(&self) -> EF {
        self.sum
    }

    /// Returns the number of remaining (unbound) variables.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Extracts the current evaluation polynomial as scalar extension-field elements.
    #[tracing::instrument(skip_all)]
    pub fn evals(&self) -> Poly<EF> {
        self.poly.evals()
    }

    /// Borrows the current evaluation polynomial in its live representation.
    ///
    /// No unpacking or copying takes place.
    pub fn evals_view(&self) -> PolyMaybePackedView<'_, F, EF> {
        self.poly.evals_view()
    }

    /// Evaluates `f` at a given multilinear point via interpolation.
    pub fn eval(&self, point: &Point<EF>) -> EF {
        self.poly.eval(point)
    }

    /// Measures the current round, first applying a binding held back from the last one.
    ///
    /// A held-back binding is absorbed into the measuring pass.
    /// The round then reads its tables once instead of twice.
    ///
    /// The slot is cleared here.
    /// The caller puts this round's own challenge back into it.
    pub(crate) fn measure_round(&mut self, pending: &mut Option<EF>) -> (EF, EF) {
        match pending.take() {
            // A challenge is waiting, so bind and measure in one pass.
            Some(r) => self.poly.fold_round_coefficients(r),
            // Nothing waiting, so this is a plain measuring pass.
            None => self.poly.round_coefficients(),
        }
    }

    /// Applies a binding that no measuring pass absorbed.
    ///
    /// The last round of a batch has no successor to fuse with.
    /// Any later reader of the tables must still see them bound.
    pub(crate) fn bind_pending(&mut self, pending: Option<EF>) {
        if let Some(r) = pending {
            self.poly.fold_round(r);
        }
    }

    /// Advances the running claim to the round polynomial at the challenge.
    ///
    /// This is the quadratic extrapolation through 0, 1 and infinity the verifier applies.
    /// The binding itself is left to the caller.
    pub(crate) fn reduce_claim_with_coefficients(&mut self, c0: EF, c_inf: EF, gamma: EF) {
        self.sum = extrapolate_01inf(c0, self.sum - c0, c_inf, gamma);
    }

    /// Asserts that the claim is the inner product of the bound pair.
    ///
    /// Only meaningful once every binding has been applied.
    /// A driver that holds bindings back therefore calls this at the end, not per round.
    pub(crate) fn debug_assert_claim(&self) {
        debug_assert_eq!(self.sum, self.poly.dot_product());
    }

    /// Applies a scalar to the weight side and the matching residual claim.
    ///
    /// Leaves the evaluation side untouched, so downstream reductions can
    /// reuse it as the honest folded message.
    pub(crate) fn scale_weights_and_claim(&mut self, scale: EF) {
        self.poly.scale_weights(scale);
        self.sum *= scale;
    }

    /// Extracts the current weight polynomial as scalar extension-field elements.
    pub fn weights(&self) -> Poly<EF> {
        self.poly.weights()
    }

    /// Folds a dense weight increment and its claim contribution into the prover.
    ///
    /// # Invariant
    ///
    /// The caller guarantees `sum_delta == <evals, weights_delta>`, restoring
    /// the running invariant `sum == dot_product` after the update.
    pub fn accumulate_claim(&mut self, weights_delta: &[EF], sum_delta: EF) {
        self.poly.accumulate_weights(weights_delta);
        self.sum += sum_delta;
        debug_assert_eq!(self.sum, self.poly.dot_product());
    }

    /// Runs additional sumcheck rounds, optionally incorporating a new constraint.
    ///
    /// # Phases
    ///
    /// - Constraint folding (optional): fold an extra constraint into the weight
    ///   polynomial and update the claimed sum before any rounds.
    /// - Round execution: perform `folding_factor` rounds of one-variable-per-round
    ///   sumcheck; each round emits coefficients, absorbs a challenge, and folds.
    ///
    /// # Returns
    ///
    /// The verifier challenges sampled during this batch.
    ///
    /// # Panics
    ///
    /// - Folding factor must not exceed the current number of remaining variables.
    #[tracing::instrument(skip_all, level = "debug")]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        constraint: Option<Constraint<F, EF>>,
    ) -> Point<EF>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Optional constraint absorption: fold into the weight polynomial and update the sum.
        if let Some(constraint) = constraint {
            self.poly.combine(&mut self.sum, &constraint);
        }

        // A challenge is not applied on the spot.
        // It is handed to the next round, which binds and measures in one pass.
        //
        //     round i:  bind r_{i-1}  +  measure h_i     (one pass)
        //     after:    bind r_{k-1}                     (one pass)
        let mut pending: Option<EF> = None;
        let mut challenges = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            // Measure this round, absorbing whatever binding the last one left behind.
            let (c_a, c_inf) = self.measure_round(&mut pending);

            // Commit to the transcript, do the optional grinding, take the challenge.
            let r = sumcheck_data.observe_and_sample(challenger, c_a, c_inf, pow_bits);

            // Advance the claim through the round identity the verifier applies.
            self.sum = Basis::Evaluation.reduce_claim(c_a, c_inf, r, self.sum);

            challenges.push(r);

            // Hand this round's challenge to the next one.
            pending = Some(r);
        }

        // The last challenge has no successor to fuse with, so it binds on its own.
        self.bind_pending(pending);

        // Invariant: the claim is the inner product of the bound pair.
        self.debug_assert_claim();

        Point::new(challenges)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing, dot_product};
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{Basis, RoundMessage, VariableOrder};
    use crate::constraints::statement::{EqStatement, NextStatement, SelectStatement};
    use crate::constraints::{Constraint, Statements};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    // Reference implementation: evaluate each constraint's combined polynomial at
    // the appropriately sliced challenge and sum. Used to cross-check the fast path.
    fn eval_constraints_poly_reference(
        order: VariableOrder,
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF {
        constraints
            .iter()
            .map(|constraint| {
                // Combine eq + sel contributions into one weight polynomial.
                let mut combined = Poly::zero(constraint.num_variables());
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);

                // Slice the challenge per binding direction; evaluate at that local point.
                let point = match order {
                    VariableOrder::Prefix => challenge
                        .reversed()
                        .get_subpoint_over_range(..constraint.num_variables())
                        .reversed(),
                    VariableOrder::Suffix => challenge
                        .reversed()
                        .get_subpoint_over_range(..constraint.num_variables()),
                };

                combined.eval_ext::<F>(&point)
            })
            .sum()
    }

    // Generates a random list of constraints for fuzzing the evaluator.
    fn random_constraints(
        rng: &mut SmallRng,
        num_variables: usize,
        rounds: usize,
    ) -> Vec<Constraint<F, EF>> {
        (0..rounds)
            .map(|_| {
                let num_variables = rng.random_range(1..=num_variables);
                let gamma = rng.random();

                // Up to 3 equality constraints at random points.
                let mut eq_statement = EqStatement::initialize(num_variables);
                (0..rng.random_range(0..=3)).for_each(|_| {
                    eq_statement
                        .add_evaluated_constraint(Point::rand(rng, num_variables), rng.random());
                });

                // Up to 3 selector constraints at random variables.
                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_variables);
                (0..rng.random_range(0..=3))
                    .for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));

                // Up to 3 successor-view equality constraints at random points.
                // The empty prefix point means each one spans the full space.
                let mut next_statement = NextStatement::initialize(num_variables);
                (0..rng.random_range(0..=3)).for_each(|_| {
                    next_statement.add_evaluated_constraint(
                        Point::new(Vec::new()),
                        Point::rand(rng, num_variables),
                        rng.random(),
                        VariableOrder::Prefix,
                    );
                });

                // Bundle the three statement groups into one constraint.
                // Order fixes the challenge-power layout: equality, then
                // successor-view, then selector blocks.
                Constraint::new(
                    gamma,
                    num_variables,
                    vec![
                        Statements::Eq(eq_statement),
                        Statements::Next(next_statement),
                        Statements::Select(sel_statement),
                    ],
                )
            })
            .collect()
    }

    #[test]
    fn test_eval_constraints_poly_prefix() {
        // Fixture: 6 random constraints over 20 variables.
        let mut rng = SmallRng::seed_from_u64(0);
        let constraints = random_constraints(&mut rng, 20, 6);
        let challenge = Point::rand(&mut rng, 20);

        // Fast path vs reference implementation must agree.
        let got = VariableOrder::Prefix.eval_constraints_poly(&constraints, &challenge);
        let expected =
            eval_constraints_poly_reference(VariableOrder::Prefix, &constraints, &challenge);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_eval_constraints_poly_suffix() {
        // Fixture: 6 random constraints over 20 variables.
        let mut rng = SmallRng::seed_from_u64(1);
        let constraints = random_constraints(&mut rng, 20, 6);
        let challenge = Point::rand(&mut rng, 20);

        // Fast path vs reference implementation must agree.
        let got = VariableOrder::Suffix.eval_constraints_poly(&constraints, &challenge);
        let expected =
            eval_constraints_poly_reference(VariableOrder::Suffix, &constraints, &challenge);
        assert_eq!(got, expected);
    }

    proptest! {
        // Invariant:
        //     VariableOrder::eval_constraints_poly must agree with the reference
        //     implementation across random constraint sets and challenge points.
        #[test]
        fn prop_eval_constraints_poly_matches_reference(
            total_num_variables in 2usize..=20,
            rounds in 1usize..=8,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let constraints = random_constraints(&mut rng, total_num_variables, rounds);
            let challenge = Point::rand(&mut rng, total_num_variables);

            prop_assert_eq!(
                VariableOrder::Prefix.eval_constraints_poly(&constraints, &challenge),
                eval_constraints_poly_reference(VariableOrder::Prefix, &constraints, &challenge),
            );
            prop_assert_eq!(
                VariableOrder::Suffix.eval_constraints_poly(&constraints, &challenge),
                eval_constraints_poly_reference(VariableOrder::Suffix, &constraints, &challenge),
            );
        }
    }

    proptest! {
        // Projective (monomial-basis) prefix round message (eprint 2026/762,
        // Fig. 3) is [s(1), s(inf)] = [dot(lo + hi, lo + hi), dot(hi, hi)];
        // the per-pair subtractions of the evaluation basis become additions.
        #[test]
        fn prop_sumcheck_coefficients_prefix_projective_matches_reference(
            k in 1usize..=12,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let n = 1usize << k;
            let evals: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let weights: Vec<EF> = (0..n).map(|_| rng.random()).collect();

            let RoundMessage { c_a: h1, c_inf: h_inf } =
                super::sumcheck_coefficients_prefix_projective(&evals, &weights);

            let half = n / 2;
            // s(1): the X = 1 evaluation of each coefficient pair is lo + hi.
            let h1_ref: EF = (0..half)
                .map(|i| (weights[i] + weights[half + i]) * (evals[i] + evals[half + i]))
                .sum();
            // s(inf): dot product of the high (leading-coefficient) faces.
            let h_inf_ref: EF = (0..half).map(|i| weights[half + i] * evals[half + i]).sum();

            prop_assert_eq!(h1, h1_ref);
            prop_assert_eq!(h_inf, h_inf_ref);
        }

        // The projective round identity, both protocol sides together: derive
        // s(0) := C - s(inf) as the verifier does, evaluate the quadratic at
        // the challenge, and compare against the dot product of the tables
        // bound in the monomial basis. Unlike the reference check above, this
        // fails if the sent message did not determine the round polynomial
        // (e.g. the insufficient [s(0), s(inf)] message passes the reference
        // check but not this one).
        #[test]
        fn prop_projective_round_message_satisfies_round_identity(
            k in 1usize..=12,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let n = 1usize << k;
            let evals: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let weights: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let r: EF = rng.random();

            let claim: EF = dot_product(evals.iter().copied(), weights.iter().copied());
            let RoundMessage { c_a: s1, c_inf: s_inf } =
                super::sumcheck_coefficients_prefix_projective(&evals, &weights);

            // Verifier side: s(0) is derived, never sent. The quadratic is
            // s(X) = s(0) + (s(1) - s(0) - s(inf)) * X + s(inf) * X^2.
            let s0 = claim - s_inf;
            let s_at_r = s0 + (s1 - s0 - s_inf) * r + s_inf * r.square();

            // The shipped reduction must agree with the identity written out above.
            prop_assert_eq!(Basis::Projective.reduce_claim(s1, s_inf, r, claim), s_at_r);

            // Prover side: bind the round variable at r in the monomial basis.
            let (mut bound_evals, mut bound_weights) = (Poly::new(evals), Poly::new(weights));
            bound_evals.fix_prefix_var_mut_monomial(r);
            bound_weights.fix_prefix_var_mut_monomial(r);

            prop_assert_eq!(
                s_at_r,
                dot_product(
                    bound_evals.as_slice().iter().copied(),
                    bound_weights.as_slice().iter().copied(),
                )
            );
        }

        // A `RoundMessage` carries no evidence of its basis, so the tag is the
        // only thing keeping the two round identities apart. Pin that they are
        // genuinely different reductions: reading a projective message with the
        // evaluation identity (or the reverse) lands on another claim entirely,
        // which is why production only ever pairs the two through `Basis`.
        #[test]
        fn prop_the_two_round_identities_disagree_on_the_same_message(
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let (c_a, c_inf, r, claim): (EF, EF, EF, EF) =
                (rng.random(), rng.random(), rng.random(), rng.random());

            // Both reductions are quadratics through {0, 1, inf}; they differ
            // in which of the three values the round identity supplies.
            prop_assume!(c_a + c_inf != claim);

            prop_assert_ne!(
                Basis::Evaluation.reduce_claim(c_a, c_inf, r, claim),
                Basis::Projective.reduce_claim(c_a, c_inf, r, claim),
            );
        }
    }

    proptest! {
        #[test]
        fn prop_fold_and_round_coefficients_prefix_matches_bind_then_measure(
            k in 2usize..=16,
            seed in any::<u64>(),
        ) {
            // Invariant: the fused pass is bind-then-measure, in one traversal.
            //
            //     two passes: bind the tables, then measure the bound pair
            //     fused     : one pass doing both
            //
            // Both the bound tables and the message have to come out identical.
            // A prover on the fused path would otherwise send a different transcript.
            //
            // Fixture state: 2^k paired random entries, one random challenge.
            // The range straddles the 8-wide tiled body and the par-vs-serial split.
            let mut rng = SmallRng::seed_from_u64(seed);
            let n = 1usize << k;
            let evals: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let weights: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let r: EF = rng.random();

            // Reference arm: bind both tables, then measure the bound pair.
            let mut want_evals = Poly::new(evals.clone());
            let mut want_weights = Poly::new(weights.clone());
            want_evals.fix_prefix_var_mut(r);
            want_weights.fix_prefix_var_mut(r);
            let want = super::sumcheck_coefficients_prefix(
                want_evals.as_slice(),
                want_weights.as_slice(),
            );

            // Fused arm: one pass writes the bound tables into the lower halves.
            let mut got_evals = Poly::new(evals);
            let mut got_weights = Poly::new(weights);
            let got = super::fold_and_round_coefficients_prefix(
                got_evals.as_mut_slice(),
                got_weights.as_mut_slice(),
                r,
            );

            // The fused pass leaves the upper halves in place, so drop them here.
            got_evals.truncate_to_half();
            got_weights.truncate_to_half();

            // The bound tables must agree entry for entry.
            prop_assert_eq!(got_evals.as_slice(), want_evals.as_slice());
            prop_assert_eq!(got_weights.as_slice(), want_weights.as_slice());

            // And so must the two values the round sends.
            prop_assert_eq!(got.c_a, want.c_a);
            prop_assert_eq!(got.c_inf, want.c_inf);
        }
    }

    #[test]
    fn deferred_binding_drives_the_same_rounds_as_round_at_a_time() {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_challenger::DuplexChallenger;
        use p3_util::log2_strict_usize;

        use crate::SumcheckData;
        use crate::product_polynomial::ProductPolynomial;

        type Perm = Poseidon2BabyBear<16>;
        type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

        // Both arms start from the same transcript.
        // Their challenges can only differ if a round message did.
        let challenger = || {
            let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(42));
            TestChallenger::new(perm)
        };

        let mut rng = SmallRng::seed_from_u64(0xD1FF);

        // A pair below one SIMD lane group has nothing to pack, so it is built scalar.
        // The lane count is a target property, so the split has to be computed, not fixed.
        let log_width = log2_strict_usize(<F as Field>::Packing::WIDTH);

        // Invariant: holding a binding back a round changes nothing the verifier sees.
        //
        // Fixture state: 1, 2, 4 and 9 variables, both binding orders.
        // Nine covers the packed path, the unpacking handoff and the scalar tail.
        // One and two land on the fallback that binds and measures separately.
        for num_variables in [1usize, 2, 4, 9] {
            for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                let evals = Poly::<EF>::rand(&mut rng, num_variables);
                let weights = Poly::<EF>::rand(&mut rng, num_variables);
                let poly = if num_variables >= log_width {
                    ProductPolynomial::<F, EF>::new_packed(
                        order,
                        evals.pack::<F, EF>(),
                        weights.pack::<F, EF>(),
                    )
                } else {
                    ProductPolynomial::<F, EF>::new_unpacked(order, evals, weights)
                };
                let sum = poly.dot_product();

                // Reference arm: bind on the spot, one round at a time.
                let mut want_data = SumcheckData::<F, EF>::default();
                let mut want_poly = poly.clone();
                let mut want_sum = sum;
                let mut want_challenger = challenger();
                let want_challenges: Vec<EF> = (0..num_variables)
                    .map(|_| {
                        want_poly.round(&mut want_data, &mut want_challenger, &mut want_sum, 0)
                    })
                    .collect();

                // Arm under test: the driver, which holds each binding back a round.
                let mut got_data = SumcheckData::<F, EF>::default();
                let mut prover = super::SumcheckProver::new(poly, sum);
                let mut got_challenger = challenger();
                let got_challenges = prover.compute_sumcheck_polynomials(
                    &mut got_data,
                    &mut got_challenger,
                    num_variables,
                    0,
                    None,
                );

                // Every round message on the wire, not just the final claim.
                assert_eq!(
                    got_data.polynomial_evaluations(),
                    want_data.polynomial_evaluations(),
                    "{order:?}, {num_variables} variables"
                );

                // The transcript is shared, so the challenges follow the messages.
                assert_eq!(got_challenges.as_slice(), want_challenges.as_slice());

                // The prover state left behind must match too.
                assert_eq!(prover.claimed_sum(), want_sum);
                assert_eq!(prover.evals().as_slice(), want_poly.evals().as_slice());
                assert_eq!(prover.weights().as_slice(), want_poly.weights().as_slice());
            }
        }
    }

    #[test]
    #[should_panic(expected = "the projective basis is prefix-only")]
    fn projective_basis_rejects_suffix_binding() {
        // The projective kernels are prefix-only; pairing them with suffix
        // binding would silently run prefix math on suffix-laid-out data.
        let evals = [EF::ONE; 4];
        let weights = [EF::ONE; 4];
        let _ = Basis::Projective.sumcheck_coefficients(VariableOrder::Suffix, &evals, &weights);
    }
}
