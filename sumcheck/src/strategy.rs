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

/// Splits a `2K`-wide chunk into the two faces of the suffix round variable.
///
/// The suffix variable is the low index bit.
///
/// So the two faces of a point are adjacent entries:
///
/// ```text
///     chunk : [ t0, t1, t2, t3, ... ]
///     lo    : [ t0, t2, ... ]           the variable at 0
///     hi    : [ t1, t3, ... ]           the variable at 1
/// ```
#[inline(always)]
fn gather_pairs<T: Copy>(chunk: &[T]) -> ([T; K], [T; K]) {
    let lo: [T; K] = core::array::from_fn(|i| chunk[2 * i]);
    let hi: [T; K] = core::array::from_fn(|i| chunk[2 * i + 1]);
    (lo, hi)
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
/// Always a whole number of tiles.
/// A block that ended mid-tile would send its remainder down the eager per-pair path.
///
/// That remainder is paid once per block, not once per table, so it has to be zero.
#[inline]
const fn fused_block<A>() -> usize {
    // Whole tiles that fit the byte budget at this element width.
    let whole_tiles = FUSED_BLOCK_BYTES / core::mem::size_of::<A>() / K * K;

    // A wide element can exhaust the budget below one tile.
    // One tile is then the floor.
    if whole_tiles == 0 { K } else { whole_tiles }
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
/// Both tables come back bound: the lower half holds the bound values, and the
/// upper half is dropped before returning.
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
/// - The length must be at least four and a multiple of four.
///   The bound table then keeps the variable the message sums over.
pub fn fold_and_round_coefficients_prefix<A, Ch>(
    evals: &mut Poly<A>,
    weights: &mut Poly<A>,
    r: Ch,
) -> RoundMessage<A>
where
    A: Algebra<Ch> + Copy + Send + Sync,
    Ch: Copy + Send + Sync,
{
    let message = bind_and_measure(evals.as_mut_slice(), weights.as_mut_slice(), r);

    // The bound values sit in the lower half of each table.
    evals.truncate_to_half();
    weights.truncate_to_half();

    message
}

/// The pass behind the binding above, over the raw tables.
///
/// The lower half of each table is left holding the bound values.
/// Dropping the upper half is the caller's, so the half-done state stays private.
fn bind_and_measure<A, Ch>(evals: &mut [A], weights: &mut [A], r: Ch) -> RoundMessage<A>
where
    A: Algebra<Ch> + Copy + Send + Sync,
    Ch: Copy + Send + Sync,
{
    // Precondition: paired tables, with a variable left over for the message.
    //
    // Zero is a multiple of four.
    //
    // So an empty pair would otherwise pass and return a zero message instead of panicking.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len() >= 4 && evals.len().is_multiple_of(4));
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

/// Round message of an interleaved pair, serial.
///
/// The suffix counterpart of the split-face scaffold.
///
/// The two faces of the round variable are adjacent entries, not separate slices.
///
/// So each tile gathers them itself.
///
/// Parallelism is left to the caller, which owns the outer loop.
#[inline]
fn round_coefficients_pairs<A>(evals: &[A], weights: &[A]) -> (A, A)
where
    A: Algebra<A> + Copy,
{
    // Whole tiles first, leftovers after.
    // A tile is `K` pairs, so `2K` consecutive entries.
    let (e_main, e_tail) = evals.as_chunks::<{ 2 * K }>();
    let (w_main, w_tail) = weights.as_chunks::<{ 2 * K }>();

    // Main loop: K pairs per iteration through delayed-reduction dot products.
    let main = e_main
        .iter()
        .zip(w_main)
        .fold((A::ZERO, A::ZERO), |acc, (e_chunk, w_chunk)| {
            let (e_lo, e_hi) = gather_pairs::<A>(e_chunk);
            let (w_lo, w_hi) = gather_pairs::<A>(w_chunk);
            round_reduce(acc, chunk_round_step(&e_lo, &e_hi, &w_lo, &w_hi))
        });

    // Tail: fewer than K pairs, so a streaming fold with eager reduction is fine.
    let tail = e_tail
        .chunks(2)
        .zip(w_tail.chunks(2))
        .fold((A::ZERO, A::ZERO), |acc, (e, w)| {
            round_step(acc, e[0], e[1], w[0], w[1])
        });

    round_reduce(main, tail)
}

/// Binds the low index bit of a table into a half-size destination.
///
/// Each output entry is the line through its input pair, sampled at the challenge:
///
/// ```text
///     src : [ a0, a1 | a2, a3 | a4, a5 | ... ]
///     dst : [ b0     | b1     | b2     | ... ]
///
///     b_g = a_{2g} + (a_{2g+1} - a_{2g}) * r
/// ```
///
/// The destination is never the source, so no entry is read after it is written.
#[inline]
fn bind_pairs<A, Ch>(dst: &mut [A], src: &[A], r: Ch)
where
    A: Algebra<Ch> + Copy,
    Ch: Copy,
{
    // Every destination entry is written, so nothing it held before can be read back.
    debug_assert_eq!(2 * dst.len(), src.len());

    for (out, pair) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
        *out = pair[0] + (pair[1] - pair[0]) * r;
    }
}

/// Destination buffers for the out-of-place suffix fused pass.
///
/// One pair of buffers serves every round of a sumcheck.
///
/// A round writes its bound tables into the buffers, then the two trade places.
///
/// ```text
///     round 1:  tables 2^n     buffers  0        -> destination 2^{n-1}
///     round 2:  tables 2^{n-1} buffers 2^n       -> destination 2^{n-2}
///     round 3:  tables 2^{n-2} buffers 2^{n-1}   -> destination 2^{n-3}
/// ```
///
/// The two slots alternate and every round halves.
///
/// So the storage a round hands over is four times the length the next round writes.
///
/// The resize below releases an overshoot that large rather than holding it.
/// The footprint therefore follows the tables down.
///
/// The first round's full-size allocation is released rather than pinned until drop.
#[derive(Debug, Clone)]
pub struct FoldBuffers<A> {
    /// Destination for the bound evaluation table.
    evals: Vec<A>,
    /// Destination for the bound weight table.
    weights: Vec<A>,
}

impl<A> Default for FoldBuffers<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A> FoldBuffers<A> {
    /// Creates empty buffers, to be sized by the first round that uses them.
    pub const fn new() -> Self {
        Self {
            evals: Vec::new(),
            weights: Vec::new(),
        }
    }
}

/// Binds a suffix variable and measures the bound pair's round message in one pass.
///
/// # Overview
///
/// Both tables come back bound to half their length.
///
/// The message is the one a separate measuring pass over the bound tables returns.
///
/// # Algorithm
///
/// The pass touches two variables at once.
/// One is bound now.
/// The other is the one the returned message sums over.
///
/// The suffix variable is the low index bit.
///
/// So the four points of those two variables are four consecutive entries.
///
/// Binding compacts each group of four into two:
///
/// ```text
///     in  : [ a0, a1, a2, a3 | a4, a5, a6, a7 | ... ]
///     out : [ b0, b1         | b2, b3         | ... ]
///
///     b_{2g}   = a_{4g}   + (a_{4g+1} - a_{4g})   * r
///     b_{2g+1} = a_{4g+2} + (a_{4g+3} - a_{4g+2}) * r
/// ```
///
/// Those two entries are the two faces the message sums over.
///
/// So a block binds its own output and measures it straight afterwards.
///
/// The output is still in cache when the measuring pass reads it.
///
/// # Why the destination is a separate buffer
///
/// Output index `g` reads input indices `2g` and `2g+1`, both at or above `g`.
/// The fold is therefore a compaction.
///
/// Writes land at indices no higher than the reads they depend on.
///
/// So one serial forward sweep could safely write in place.
///
/// Blocked parallelism breaks that.
///
/// Cut the output into blocks at `G_0 = 0 < G_1 < ...`:
///
/// ```text
///     block 0 : writes [ 0,   G_1 )     reads [ 0,     2 G_1 )
///     block 1 : writes [ G_1, G_2 )     reads [ 2 G_1, 2 G_2 )
/// ```
///
/// Block 1 writes from `G_1`, and block 0 reads up to `2 G_1`.
///
/// Any non-empty first block has `G_1 < 2 G_1`, so those ranges always overlap.
/// One task would be overwriting entries another task has yet to read.
///
/// A separate half-size destination removes the overlap outright.
/// The pass then stays single and stays parallel.
///
/// # Arguments
///
/// - `evals` - evaluation table, before this binding.
/// - `weights` - weight table, before this binding.
///
/// - `buffers` - destination buffers, resized here and traded with the tables.
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
/// - The length must be at least four and a multiple of four.
///   The bound table then keeps the variable the message sums over.
pub fn fold_and_round_coefficients_suffix<A, Ch>(
    evals: &mut Poly<A>,
    weights: &mut Poly<A>,
    buffers: &mut FoldBuffers<A>,
    r: Ch,
) -> RoundMessage<A>
where
    A: Algebra<Ch> + Copy + Send + Sync,
    Ch: Copy + Send + Sync,
{
    let message = bind_and_measure_pairs(
        evals.as_slice(),
        weights.as_slice(),
        &mut buffers.evals,
        &mut buffers.weights,
        r,
    );

    // Trade places.
    //
    //     bound buffers  ->  become the tables
    //     old storage    ->  becomes the next round's destination
    swap_storage(evals, &mut buffers.evals);
    swap_storage(weights, &mut buffers.weights);

    message
}

/// Hands a buffer to a table and takes the table's old storage as the buffer.
///
/// The old storage is twice the length of the buffer replacing it.
///
/// The next round writes only a quarter of that.
///
/// Whether so much room is worth keeping is decided when the destination is resized.
#[inline]
fn swap_storage<A>(table: &mut Poly<A>, buffer: &mut Vec<A>) {
    let bound = core::mem::take(buffer);
    *buffer = core::mem::replace(table, Poly::new(bound)).into_evals();
}

/// Gives a destination buffer the length one round writes.
///
/// A buffer handed back by an earlier round is longer than this round needs.
/// Keeping the whole of it is what turns a reused buffer into retained memory:
///
/// ```text
///     capacity <= 2 * len : length update, nothing allocated
///     capacity >  2 * len : released, then a fresh `len`-entry allocation
/// ```
///
/// Two is the smallest factor that still lets a buffer survive one halving.
///
/// That single halving is the reuse the exchange above is built around.
///
/// The first round hands over storage four times the next destination.
///
/// So that round is the one whose full-size allocation is released rather than held.
///
/// Footprint per side is then the live table plus at most twice its length:
///
/// ```text
///     after round i:  table 2^{n-i}  +  buffer <= 2^{n-i+1}
/// ```
///
/// A replacement is allocated already zeroed, so no userspace pass fills it.
///
/// Every entry is overwritten before anything reads it, so the values do not matter.
#[inline]
fn resize_destination<A>(buffer: &mut Vec<A>, len: usize)
where
    A: PrimeCharacteristicRing,
{
    // Reuse: the buffer already covers this round and does not overshoot it badly.
    if buffer.len() >= len && buffer.capacity() <= 2 * len {
        buffer.truncate(len);
        return;
    }

    // Released before the replacement is asked for, so the two never coexist.
    drop(core::mem::take(buffer));
    *buffer = A::zero_vec(len);
}

/// The pass behind the binding above, over the raw tables.
///
/// The destinations are resized to the bound length and fully overwritten.
///
/// So whatever they held before is never read.
fn bind_and_measure_pairs<A, Ch>(
    evals: &[A],
    weights: &[A],
    evals_out: &mut Vec<A>,
    weights_out: &mut Vec<A>,
    r: Ch,
) -> RoundMessage<A>
where
    A: Algebra<Ch> + Copy + Send + Sync,
    Ch: Copy + Send + Sync,
{
    // Precondition: paired tables, with a variable left over for the message.
    //
    // Zero is a multiple of four.
    //
    // So an empty pair would otherwise pass and return a zero message instead of panicking.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len() >= 4 && evals.len().is_multiple_of(4));

    // Binding halves the length.
    let half = evals.len() / 2;

    // The pass covers the whole table, not just the bound half.
    //
    // So the par-vs-serial split is gated on the whole table.
    // That puts about as much work in one task as a measuring pass does at its own gate.
    let threaded = evals.len() > PAR_THRESHOLD;

    // Size the destinations to the bound length.
    resize_destination(evals_out, half);
    resize_destination(weights_out, half);

    // Bound index positions one block writes before measuring them.
    //
    // A block keeps one bound face of each table hot across the two steps.
    //
    //     prefix block : four faces, half the table apart
    //     suffix block : one face, adjacent entries
    //
    // So the same byte budget buys twice as many positions.
    //
    // Twice a whole number of tiles is still a whole number of tiles.
    //
    // So no block ends mid-tile.
    let block_len = 2 * fused_block::<A>();

    // One block: bind its own slice of the destination, then measure it while hot.
    let block = |e_in: &[A], w_in: &[A], e_out: &mut [A], w_out: &mut [A]| {
        bind_pairs(e_out, e_in, r);
        bind_pairs(w_out, w_in, r);
        round_coefficients_pairs(e_out, w_out)
    };

    // Each destination block reads the twice-as-long input block at the same position.
    //
    // No block writes where another reads.
    let (c_a, c_inf) = if threaded {
        evals_out
            .par_chunks_mut(block_len)
            .zip(weights_out.par_chunks_mut(block_len))
            .zip(evals.par_chunks(2 * block_len))
            .zip(weights.par_chunks(2 * block_len))
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |acc, (((e_out, w_out), e_in), w_in)| {
                    round_reduce(acc, block(e_in, w_in, e_out, w_out))
                },
                round_reduce,
            )
    } else {
        evals_out
            .chunks_mut(block_len)
            .zip(weights_out.chunks_mut(block_len))
            .zip(evals.chunks(2 * block_len))
            .zip(weights.chunks(2 * block_len))
            .fold((A::ZERO, A::ZERO), |acc, (((e_out, w_out), e_in), w_in)| {
                round_reduce(acc, block(e_in, w_in, e_out, w_out))
            })
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
/// where `n` is the number of remaining unbound variables.
/// It decreases by one per round as variables are bound to verifier challenges.
///
/// # Outstanding challenge
///
/// A round's challenge is not applied to the tables the moment it is sampled.
/// It is held in the prover instead, and the next measuring pass binds it on the way through.
///
/// ```text
///     round i:   bind r_{i-1}  +  measure h_i     one pass
///     round i+1: bind r_i      +  measure h_{i+1} one pass
/// ```
///
/// The claim is always up to date, because it only ever needs the round message.
/// The tables lag by one binding for as long as a challenge is outstanding.
///
/// Anything that reads the tables therefore applies the outstanding binding first.
///
/// The arity is the one exception.
/// It is a length, so subtracting the outstanding binding answers it exactly.
///
/// The challenge lives here rather than in a driver's local.
///
/// So a caller asking for one round at a time still gets one pass per round.
///
/// Such a caller interleaves its own work between rounds.
/// It cannot ask for several rounds at once.
#[derive(Debug, Clone)]
pub struct SumcheckProver<F: Field, EF: ExtensionField<F>> {
    /// Paired evaluation and weight polynomials for the quadratic sumcheck.
    poly: ProductPolynomial<F, EF>,
    /// Current claimed sum over the remaining unbound variables.
    sum: EF,
    /// Challenge sampled by the last round and not yet applied to the tables.
    ///
    /// Empty means the tables are current with the claim.
    outstanding: Option<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SumcheckProver<F, EF> {
    /// Creates a prover state from a product polynomial and its claimed sum.
    pub fn new(poly: ProductPolynomial<F, EF>, sum: EF) -> Self {
        // Sanity: the claimed sum must match the polynomial pair's dot product.
        debug_assert_eq!(poly.dot_product(), sum);
        Self {
            poly,
            sum,
            outstanding: None,
        }
    }

    /// Returns the current claimed sum over the remaining unbound variables.
    pub const fn claimed_sum(&self) -> EF {
        self.sum
    }

    /// Returns the number of remaining (unbound) variables.
    ///
    /// An outstanding binding has already consumed a variable.
    ///
    /// The tables are still the length they had before it.
    /// Subtracting it is exact, so the answer costs no pass over the data.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables() - usize::from(self.outstanding.is_some())
    }

    /// Applies an outstanding binding, so the tables are current with the claim.
    ///
    /// Every reader of the tables starts here.
    /// A reader that skipped it would silently see the tables one round behind.
    ///
    /// A debug build checks the claim against the pair this binding produced.
    /// That is the only place a held binding is ever validated.
    ///
    /// Idempotent, and free when nothing is outstanding.
    pub fn settle(&mut self) {
        if let Some(r) = self.outstanding.take() {
            self.poly.fold_round(r);

            // Invariant: the claim is the inner product of the now-current pair.
            debug_assert_eq!(self.sum, self.poly.dot_product());
        }
    }

    /// Extracts the current evaluation polynomial as scalar extension-field elements.
    #[tracing::instrument(skip_all)]
    pub fn evals(&mut self) -> Poly<EF> {
        self.settle();
        self.poly.evals()
    }

    /// Borrows the current evaluation polynomial in its live representation.
    ///
    /// No unpacking or copying takes place.
    pub fn evals_view(&mut self) -> PolyMaybePackedView<'_, F, EF> {
        self.settle();
        self.poly.evals_view()
    }

    /// Evaluates `f` at a given multilinear point via interpolation.
    pub fn eval(&mut self, point: &Point<EF>) -> EF {
        self.settle();
        self.poly.eval(point)
    }

    /// Measures the current round, first applying any outstanding binding.
    ///
    /// An outstanding binding is absorbed into the measuring pass.
    /// The round then reads its tables once instead of twice.
    ///
    /// The slot is cleared here.
    /// The caller puts this round's own challenge back into it.
    pub(crate) fn measure_round(&mut self) -> (EF, EF) {
        let message = match self.outstanding.take() {
            // A challenge is waiting, so bind and measure in one pass.
            Some(r) => self.poly.fold_round_coefficients(r),
            // Nothing waiting, so this is a plain measuring pass.
            None => self.poly.round_coefficients(),
        };

        // Invariant: the claim is the inner product of the pair this round measured.
        //
        // The claim describes the table this pass measured.
        // The binding this pass absorbed is what brought the tables up to it.
        //
        // A stale table, or a binding that landed wrong, breaks the equality here.
        debug_assert_eq!(self.sum, self.poly.dot_product());

        message
    }

    /// Holds a challenge back for the next measuring pass to absorb.
    ///
    /// # Panics
    ///
    /// Panics if a challenge is already outstanding.
    /// Two unapplied bindings cannot be fused into one pass.
    ///
    /// The second would overwrite the first and lose a variable.
    pub(crate) fn hold(&mut self, r: EF) {
        assert!(
            self.outstanding.replace(r).is_none(),
            "a challenge is already outstanding"
        );
    }

    /// Advances the running claim to the round polynomial at the challenge.
    ///
    /// This is the quadratic extrapolation through 0, 1 and infinity the verifier applies.
    /// The binding itself is left to the caller.
    pub(crate) fn reduce_claim_with_coefficients(&mut self, c0: EF, c_inf: EF, gamma: EF) {
        self.sum = extrapolate_01inf(c0, self.sum - c0, c_inf, gamma);
    }

    /// Applies a scalar to the weight side and the matching residual claim.
    ///
    /// Leaves the evaluation side untouched, so downstream reductions can
    /// reuse it as the honest folded message.
    pub(crate) fn scale_weights_and_claim(&mut self, scale: EF) {
        // Scaling every entry commutes with binding, so the order does not change the result.
        // Settling first halves the number of entries to scale.
        self.settle();
        self.poly.scale_weights(scale);
        self.sum *= scale;
    }

    /// Extracts the current weight polynomial as scalar extension-field elements.
    pub fn weights(&mut self) -> Poly<EF> {
        self.settle();
        self.poly.weights()
    }

    /// Folds a dense weight increment and its claim contribution into the prover.
    ///
    /// # Invariant
    ///
    /// The caller guarantees `sum_delta == <evals, weights_delta>`, restoring
    /// the running invariant `sum == dot_product` after the update.
    pub fn accumulate_claim(&mut self, weights_delta: &[EF], sum_delta: EF) {
        // The increment is indexed by the current hypercube.
        //
        // So the tables have to be the length the caller sized it against.
        self.settle();
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
        //
        // The constraint is sized against the current hypercube.
        //
        // So an outstanding binding has to land before the weights grow by it.
        if let Some(constraint) = constraint {
            self.settle();
            self.poly.combine(&mut self.sum, &constraint);
        }

        let mut challenges = Vec::with_capacity(folding_factor);

        for _ in 0..folding_factor {
            // Measure this round, absorbing whatever binding the last one left behind.
            let (c_a, c_inf) = self.measure_round();

            // Commit to the transcript, do the optional grinding, take the challenge.
            let r = sumcheck_data.observe_and_sample(challenger, c_a, c_inf, pow_bits);

            // Advance the claim through the round identity the verifier applies.
            self.sum = Basis::Evaluation.reduce_claim(c_a, c_inf, r, self.sum);

            challenges.push(r);

            // Hand this round's challenge on, for the next measuring pass to absorb.
            //
            // The last one of this batch stays outstanding on return.
            // A caller that comes straight back for another round fuses across the call;
            // one that reads the tables instead settles it on the way in.
            self.hold(r);
        }

        Point::new(challenges)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

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

            // Fused arm: one pass binds both tables and measures the bound pair.
            let mut got_evals = Poly::new(evals);
            let mut got_weights = Poly::new(weights);
            let got = super::fold_and_round_coefficients_prefix(
                &mut got_evals,
                &mut got_weights,
                r,
            );

            // The bound tables must agree entry for entry.
            prop_assert_eq!(got_evals.as_slice(), want_evals.as_slice());
            prop_assert_eq!(got_weights.as_slice(), want_weights.as_slice());

            // And so must the two values the round sends.
            prop_assert_eq!(got.c_a, want.c_a);
            prop_assert_eq!(got.c_inf, want.c_inf);
        }
    }

    proptest! {
        #[test]
        fn prop_fold_and_round_coefficients_suffix_matches_bind_then_measure(
            k in 2usize..=16,
            seed in any::<u64>(),
        ) {
            // Invariant: the fused suffix pass is bind-then-measure, in one traversal.
            //
            //     two passes: bind the tables, then measure the bound pair
            //     fused     : one pass writing a half-size destination and measuring it
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
            want_evals.fix_suffix_var_mut(r);
            want_weights.fix_suffix_var_mut(r);
            let want = super::sumcheck_coefficients_suffix(
                want_evals.as_slice(),
                want_weights.as_slice(),
            );

            // Fused arm: one pass binds both tables and measures the bound pair.
            let mut got_evals = Poly::new(evals);
            let mut got_weights = Poly::new(weights);
            let mut buffers = super::FoldBuffers::new();
            let got = super::fold_and_round_coefficients_suffix(
                &mut got_evals,
                &mut got_weights,
                &mut buffers,
                r,
            );

            // The bound tables must agree entry for entry.
            prop_assert_eq!(got_evals.as_slice(), want_evals.as_slice());
            prop_assert_eq!(got_weights.as_slice(), want_weights.as_slice());

            // And so must the two values the round sends.
            prop_assert_eq!(got.c_a, want.c_a);
            prop_assert_eq!(got.c_inf, want.c_inf);
        }
    }

    #[test]
    fn reused_fold_buffers_bind_every_suffix_round_correctly() {
        // Invariant: one pair of buffers serves a whole sumcheck.
        //
        // The tables and the buffers trade places each round.
        // A buffer therefore arrives holding entries from two rounds ago.
        //
        // Every destination entry is written before it is read.
        // Those stale entries can never reach a round message.
        //
        // The footprint is checked alongside the values.
        // A buffer never holds more than twice the live table.
        //
        // So what the pair retains halves with the rounds, rather than staying at round one's.
        //
        // Fixture state: 2^15 paired entries, bound down to 4.
        // The first rounds run the threaded branch, the last ones the serial branch.
        //
        //     round 1: tables 2^15 -> 2^14      buffers sized
        //     round 2: tables 2^14 -> 2^13      round-1 storage handed back
        //     ...
        //     round 13: tables 4 -> 2           the shortest fusable table
        const NUM_VARIABLES: usize = 15;

        let mut rng = SmallRng::seed_from_u64(0xB0FFE7);
        let evals = Poly::<EF>::rand(&mut rng, NUM_VARIABLES);
        let weights = Poly::<EF>::rand(&mut rng, NUM_VARIABLES);

        // Reference arm: bind on the spot, measure the bound pair separately.
        let mut want_evals = evals.clone();
        let mut want_weights = weights.clone();

        // Arm under test: one pass per round, into buffers reused throughout.
        let mut got_evals = evals;
        let mut got_weights = weights;
        let mut buffers = super::FoldBuffers::new();

        // Stop with four entries left: below that the pass has no variable to measure.
        for round in 0..NUM_VARIABLES - 1 {
            let r: EF = rng.random();

            want_evals.fix_suffix_var_mut(r);
            want_weights.fix_suffix_var_mut(r);
            let want =
                super::sumcheck_coefficients_suffix(want_evals.as_slice(), want_weights.as_slice());

            let got = super::fold_and_round_coefficients_suffix(
                &mut got_evals,
                &mut got_weights,
                &mut buffers,
                r,
            );

            assert_eq!(got_evals.as_slice(), want_evals.as_slice(), "round {round}");
            assert_eq!(
                got_weights.as_slice(),
                want_weights.as_slice(),
                "round {round}"
            );
            assert_eq!(got.c_a, want.c_a, "round {round}");
            assert_eq!(got.c_inf, want.c_inf, "round {round}");

            // The buffer now holds the storage the table just handed over.
            // Keeping more than twice the live table would pin memory no round reaches.
            let live = got_evals.as_slice().len();
            assert!(
                buffers.evals.capacity() <= 2 * live,
                "round {round}: evals buffer holds {} for a table of {live}",
                buffers.evals.capacity()
            );
            assert!(
                buffers.weights.capacity() <= 2 * live,
                "round {round}: weights buffer holds {} for a table of {live}",
                buffers.weights.capacity()
            );
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn suffix_binding_rejects_a_pair_with_nothing_left_to_measure() {
        // Invariant: a pair too short to leave a variable is rejected, not measured.
        //
        // Two entries bind down to one, and a one-entry table has nothing to sum over.
        // Producing no tiles and reporting a zero message would look like a real round.
        let mut evals = Poly::new(vec![EF::ONE; 2]);
        let mut weights = Poly::new(vec![EF::ONE; 2]);
        let mut buffers = super::FoldBuffers::new();
        let _ = super::fold_and_round_coefficients_suffix(
            &mut evals,
            &mut weights,
            &mut buffers,
            EF::ONE,
        );
    }

    #[test]
    fn a_fused_block_is_a_whole_number_of_tiles() {
        // Invariant: a block never ends mid-tile.
        //
        // The measurement splits a block into K-wide tiles and sends the remainder
        // down the eager per-pair path.
        // A block remainder would be paid once per block instead of once per table.
        //
        // Fixture state: element widths that do and do not divide the byte budget.
        //
        //     1 B   -> budget / 1  is already a multiple of K
        //     320 B -> budget / 320 = 12, which is not
        //     192 B -> budget / 192 = 21, which is not
        //     8192B -> budget / 8192 = 0, so the tile width is the floor
        assert!(super::fused_block::<u8>().is_multiple_of(super::K));
        assert!(super::fused_block::<[u8; 320]>().is_multiple_of(super::K));
        assert!(super::fused_block::<[u8; 192]>().is_multiple_of(super::K));
        assert_eq!(super::fused_block::<[u8; 8192]>(), super::K);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn binding_rejects_a_pair_with_nothing_left_to_measure() {
        // Invariant: a pair too short to leave a variable is rejected, not measured.
        //
        // Two entries bind down to one, and a one-entry table has nothing to sum over.
        // Producing no tiles and reporting a zero message would look like a real round.
        let mut evals = Poly::new(vec![EF::ONE; 2]);
        let mut weights = Poly::new(vec![EF::ONE; 2]);
        let _ = super::fold_and_round_coefficients_prefix(&mut evals, &mut weights, EF::ONE);
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
    fn every_table_accessor_settles_an_outstanding_binding() {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_challenger::DuplexChallenger;
        use p3_util::log2_strict_usize;

        use crate::SumcheckData;
        use crate::product_polynomial::ProductPolynomial;

        type Perm = Poseidon2BabyBear<16>;
        type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;

        let challenger = || {
            let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(11));
            TestChallenger::new(perm)
        };

        let log_width = log2_strict_usize(<F as Field>::Packing::WIDTH);
        let mut rng = SmallRng::seed_from_u64(0xACCE55);

        // Invariant: any accessor applies the binding the last round left outstanding.
        //
        // A reader that skipped it would see the tables one round behind the claim.
        //
        // It would then silently produce the previous round's answer.
        //
        // Fixture state: a batch of rounds runs and returns with its last challenge held.
        //
        // Every accessor is then read on a fresh copy of that state.
        //
        //     driven : one binding outstanding, accessor settles it
        //     settled: the same state with the binding already applied
        //
        // Fixture shapes:
        //
        //     variables : 3, 5, 9
        //     orders    : both
        //     batches   : stopping short of the last variable, and consuming every one
        for (num_variables, rounds) in [(3usize, 1usize), (3, 3), (5, 2), (9, 4), (9, 9)] {
            for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                let evals = Poly::<EF>::rand(&mut rng, num_variables);
                let weights = Poly::<EF>::rand(&mut rng, num_variables);

                // A pair below one SIMD lane group has nothing to pack.
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

                let mut driven = super::SumcheckProver::new(poly, sum);
                let mut data = SumcheckData::<F, EF>::default();
                driven.compute_sumcheck_polynomials(&mut data, &mut challenger(), rounds, 0, None);

                // Reference arm: the same state with the outstanding binding applied.
                let mut settled = driven.clone();
                settled.settle();

                let shape = format!("{order:?}, {num_variables} variables, {rounds} rounds");

                // The arity is answered from lengths, so it never triggers a binding pass.
                // It still has to report the variable the outstanding challenge consumed.
                assert_eq!(driven.num_variables(), settled.num_variables(), "{shape}");
                assert_eq!(driven.num_variables(), num_variables - rounds, "{shape}");

                // The claim never lags, so it reads the same on both arms.
                assert_eq!(driven.claimed_sum(), settled.claimed_sum(), "{shape}");

                // Each accessor is read on its own copy, so it meets the binding outstanding.
                assert_eq!(
                    driven.clone().evals().as_slice(),
                    settled.evals().as_slice(),
                    "{shape}"
                );
                assert_eq!(
                    driven.clone().weights().as_slice(),
                    settled.weights().as_slice(),
                    "{shape}"
                );

                // The live-representation borrow, unpacked so the two storages compare.
                let mut got_view = EF::zero_vec(1 << driven.num_variables());
                let mut want_view = EF::zero_vec(1 << driven.num_variables());
                driven.clone().evals_view().unpack_into(&mut got_view);
                settled.evals_view().unpack_into(&mut want_view);
                assert_eq!(got_view, want_view, "{shape}");

                // Interpolation at a point of the current arity.
                let point = Point::<EF>::rand(&mut rng, driven.num_variables());
                assert_eq!(driven.clone().eval(&point), settled.eval(&point), "{shape}");

                // Scaling the weight side and the claim together.
                let scale: EF = rng.random();
                let mut got_scaled = driven.clone();
                let mut want_scaled = settled.clone();
                got_scaled.scale_weights_and_claim(scale);
                want_scaled.scale_weights_and_claim(scale);
                assert_eq!(
                    got_scaled.weights().as_slice(),
                    want_scaled.weights().as_slice(),
                    "{shape}"
                );
                assert_eq!(
                    got_scaled.claimed_sum(),
                    want_scaled.claimed_sum(),
                    "{shape}"
                );

                // A dense weight increment, indexed by the current hypercube.
                // Its claim contribution is derived from the settled evaluation table,
                // so an accessor that read a stale table would break the running invariant.
                let delta: Vec<EF> = (0..1 << driven.num_variables())
                    .map(|_| rng.random())
                    .collect();
                let delta_sum = dot_product::<EF, _, _>(
                    settled.evals().as_slice().iter().copied(),
                    delta.iter().copied(),
                );
                let mut got_acc = driven.clone();
                let mut want_acc = settled.clone();
                got_acc.accumulate_claim(&delta, delta_sum);
                want_acc.accumulate_claim(&delta, delta_sum);
                assert_eq!(
                    got_acc.weights().as_slice(),
                    want_acc.weights().as_slice(),
                    "{shape}"
                );
                assert_eq!(got_acc.claimed_sum(), want_acc.claimed_sum(), "{shape}");

                // Settling twice binds once.
                let mut twice = driven.clone();
                twice.settle();
                twice.settle();
                assert_eq!(
                    twice.evals().as_slice(),
                    settled.evals().as_slice(),
                    "{shape}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "a challenge is already outstanding")]
    fn holding_a_second_challenge_without_settling_is_rejected() {
        use crate::product_polynomial::ProductPolynomial;

        // Invariant: at most one binding is ever outstanding.
        //
        // Two unapplied bindings cannot be fused into a single pass.
        //
        // The second would overwrite the first and lose a variable with nothing noticing.
        let evals = Poly::new(vec![EF::ONE; 4]);
        let weights = Poly::new(vec![EF::TWO; 4]);
        let poly = ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, evals, weights);
        let sum = poly.dot_product();
        let mut prover = super::SumcheckProver::new(poly, sum);

        prover.hold(EF::ONE);
        prover.hold(EF::TWO);
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
