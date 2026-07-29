use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::Mmcs;
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_fri::FriFoldingStrategy;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::domain::CircleDomain;
use crate::{CircleInputProof, InputError};

pub(crate) struct CircleFriFolding<F, InputProof, InputError>(
    pub(crate) PhantomData<(F, InputProof, InputError)>,
);

pub(crate) type CircleFriFoldingForMmcs<Val, Challenge, InputMmcs, FriMmcs> = CircleFriFolding<
    Val,
    CircleInputProof<Val, Challenge, InputMmcs, FriMmcs>,
    InputError<<InputMmcs as Mmcs<Val>>::Error, <FriMmcs as Mmcs<Challenge>>::Error>,
>;

impl<F: ComplexExtendable, EF: ExtensionField<F>, InputProof, InputError: Debug>
    FriFoldingStrategy<F, EF> for CircleFriFolding<F, InputProof, InputError>
{
    type InputProof = InputProof;
    type InputError = InputError;

    fn extra_query_index_bits(&self) -> usize {
        1
    }

    fn fold_row(
        &self,
        index: usize,
        log_folded_height: usize,
        log_arity: usize,
        beta: EF,
        evals: impl Iterator<Item = EF>,
    ) -> EF {
        fold_x_row(index, log_folded_height, log_arity, beta, evals)
    }

    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, log_arity: usize, m: M) -> Vec<EF> {
        fold_x(beta, log_arity, &m)
    }
}

fn fold<F: ComplexExtendable, EF: ExtensionField<F>>(
    evals: &impl Matrix<EF>,
    beta: EF,
    twiddles: &[F],
) -> Vec<EF> {
    debug_assert_eq!(evals.width(), 2);
    debug_assert_eq!(evals.height(), twiddles.len());

    // Rows are folded independently, so the matrix splits into parallel chunks. The chunk
    // size keeps the per-task work well above the fork-join overhead.
    const FOLD_CHUNK_ROWS: usize = 1 << 10;

    let mut out = EF::zero_vec(evals.height());
    out.par_chunks_mut(FOLD_CHUNK_ROWS)
        .zip(twiddles.par_chunks(FOLD_CHUNK_ROWS))
        .enumerate()
        .for_each(|(chunk_idx, (out_chunk, twiddle_chunk))| {
            let first_row = chunk_idx * FOLD_CHUNK_ROWS;
            for (i, (o, &t)) in out_chunk.iter_mut().zip(twiddle_chunk).enumerate() {
                // SAFETY: the chunks cover exactly `evals.height()` rows.
                let row = unsafe { evals.row_slice_unchecked(first_row + i) };
                let (lo, hi) = (row[0], row[1]);
                let sum = lo + hi;
                let diff = (lo - hi) * t;
                *o = (sum + beta * diff).halve();
            }
        });
    out
}

pub(crate) fn fold_y<F: ComplexExtendable, EF: ExtensionField<F>>(
    beta: EF,
    evals: &impl Matrix<EF>,
) -> Vec<EF> {
    assert_eq!(evals.width(), 2);
    let log_n = log2_strict_usize(evals.height()) + 1;
    fold(
        evals,
        beta,
        &batch_multiplicative_inverse(&CircleDomain::standard(log_n).y_twiddles()),
    )
}

/// Reference implementation for [`fold_y`], kept only to cross-check it against row-wise
/// folding in tests; the verifier reuses the lambda-correction point instead (see
/// `CirclePcs::open`'s `open_input` closure) rather than recomputing this per-row twiddle.
#[cfg(test)]
pub(crate) fn fold_y_row<F: ComplexExtendable, EF: ExtensionField<F>>(
    index: usize,
    log_folded_height: usize,
    beta: EF,
    evals: impl Iterator<Item = EF>,
) -> EF {
    let evals = evals.collect_vec();
    assert_eq!(evals.len(), 2);
    let t = CircleDomain::<F>::standard(log_folded_height + 1)
        .nth_y_twiddle(index)
        .inverse();
    let sum = evals[0] + evals[1];
    let diff = (evals[0] - evals[1]) * t;
    (sum + beta * diff).halve()
}

pub(crate) fn fold_x<F: ComplexExtendable, EF: ExtensionField<F>>(
    beta: EF,
    log_arity: usize,
    evals: &impl Matrix<EF>,
) -> Vec<EF> {
    // Currently only arity 2 is supported for Circle PCS
    assert_eq!(log_arity, 1, "Circle PCS currently only supports arity 2");
    let log_n = log2_strict_usize(evals.width() * evals.height());
    // +1 because twiddles after the first layer come from the x coordinates of the larger domain.
    let domain = CircleDomain::standard(log_n + 1);
    fold(
        evals,
        beta,
        &batch_multiplicative_inverse(&domain.x_twiddles(0)),
    )
}

pub(crate) fn fold_x_row<F: ComplexExtendable, EF: ExtensionField<F>>(
    index: usize,
    log_folded_height: usize,
    log_arity: usize,
    beta: EF,
    evals: impl Iterator<Item = EF>,
) -> EF {
    // Currently only arity 2 is supported for Circle PCS
    assert_eq!(log_arity, 1, "Circle PCS currently only supports arity 2");
    let evals = evals.collect_vec();
    assert_eq!(evals.len(), 2);

    let t = CircleDomain::<F>::standard(log_folded_height + log_arity + 1)
        .nth_x_twiddle(reverse_bits_len(index, log_folded_height))
        .inverse();

    let sum = evals[0] + evals[1];
    let diff = (evals[0] - evals[1]) * t;
    (sum + beta * diff).halve()
}

/// Fold a pair of sibling evaluations given an already-inverted twiddle, shared by both the
/// first-layer (y) fold and every FRI (x) fold round: `fold_*_row` differ only in how `t` is
/// derived, not in this arithmetic.
pub(crate) fn fold_row_with_inv_twiddle<F: Field, EF: ExtensionField<F>>(
    inv_twiddle: F,
    beta: EF,
    evals: impl Iterator<Item = EF>,
) -> EF {
    let mut it = evals;
    let (e0, e1) = (it.next().unwrap(), it.next().unwrap());
    assert!(it.next().is_none());

    let sum = e0 + e1;
    let diff = (e0 - e1) * inv_twiddle;
    (sum + beta * diff).halve()
}

/// Precompute the per-query chain of x-fold twiddles, already batch-inverted.
///
/// `top_level_index` is the FRI-domain index (the query index with
/// [`FriFoldingStrategy::extra_query_index_bits`] shifted off), and `log_max_height` is
/// [`crate::verifier::verify`]'s own top height (one less than the tallest committed matrix's
/// LDE height, since the first-layer bivariate fold has already consumed one bit of height by
/// this point).
///
/// Round `r`'s twiddle is exactly what `fold_x_row` recomputes from scratch on every call
/// (`CircleDomain::standard(log_max_height - r + 1).nth_x_twiddle(..)`). After the first round,
/// each successive twiddle is the circle's squaring map applied to the previous one
/// (`x -> 2x^2 - 1`), with a sign flip determined by a fixed bit of `top_level_index` - no
/// further scalar multiplications or domain constructions are needed. All `num_rounds`
/// twiddles are inverted in a single batch instead of one inversion per round.
pub(crate) fn query_x_twiddles_inv<F: ComplexExtendable>(
    top_level_index: usize,
    log_max_height: usize,
    num_rounds: usize,
) -> Vec<F> {
    if num_rounds == 0 {
        return Vec::new();
    }

    let seed_log_folded_height = log_max_height - 1;
    let seed_idx = reverse_bits_len(top_level_index >> 1, seed_log_folded_height);
    let mut x = CircleDomain::<F>::standard(log_max_height + 1).nth_x_twiddle(seed_idx);

    let mut twiddles = Vec::with_capacity(num_rounds);
    twiddles.push(x);
    for r in 0..num_rounds - 1 {
        x = x.square().double() - F::ONE;
        if (top_level_index >> (r + 1)) & 1 == 1 {
            x = -x;
        }
        twiddles.push(x);
    }

    batch_multiplicative_inverse(&twiddles)
}

#[cfg(test)]
mod tests {
    use itertools::iproduct;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::CircleEvaluations;
    use crate::ordering::cfft_permute_index;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    /// Exhaustively checks the closed-form twiddle chain that `verify_query`'s fold loop
    /// could use instead of recomputing `nth_point`/`nth_x_twiddle` from scratch each round.
    ///
    /// Reproduces, bit-for-bit, the index arithmetic in `CirclePcs::open`'s `open_input`
    /// closure (the `p`/`orig_idx`/`bits_reduced` computation feeding the lambda correction
    /// and the first-layer `fold_y_row` call) and in `verify_query` (the FRI round loop's
    /// `index >>= log_arity` bookkeeping feeding `fold_x_row`), for a matrix committed at
    /// `log_height` inside a global FRI instance of `log_max_height`.
    ///
    /// Claims checked, for `p = CircleDomain::standard(log_height).nth_point(orig_idx)`:
    /// - the first-layer (bivariate) y-twiddle is `p.y`, sign-flipped when the query's bit
    ///   at `bits_reduced` is 1 (the two members of a fold pair are negations of each other).
    /// - the x-twiddle chain for every FRI round this matrix participates in is exactly
    ///   `p.x, 2p.x^2-1, 2(2p.x^2-1)^2-1, ..`, i.e. repeated application of the circle's
    ///   squaring map with no further scalar multiplications.
    #[test]
    fn scratch_generator_doubling() {
        use crate::point::Point;
        for k in 2..8 {
            let g = Point::<F>::generator(k);
            assert_eq!(g.double(), Point::<F>::generator(k - 1), "k={k}");
        }
        // nth_x_twiddle doubling identity, same idx, domain one size smaller.
        for log_n in 4..8 {
            let d = CircleDomain::<F>::standard(log_n);
            let d2 = CircleDomain::<F>::standard(log_n - 1);
            for idx in 0..(1usize << (log_n - 2)) {
                let t = d.nth_x_twiddle(idx);
                let doubled_x = t.square().double() - F::ONE;
                let t2 = d2.nth_x_twiddle(idx);
                assert_eq!(doubled_x, t2, "log_n={log_n} idx={idx}");
            }
        }
    }

    #[test]
    fn query_twiddle_chain_matches_naive_recomputation() {
        for log_blowup in 1..4 {
            for matrix_log_n in 2..5 {
                for extra_rounds_above in 0..3 {
                    let log_height = matrix_log_n + log_blowup;
                    // The first-layer (bivariate) fold always consumes one bit of height
                    // before FRI's own x-folding rounds begin, even for the tallest matrix.
                    let log_max_height = log_height - 1 + extra_rounds_above;
                    let log_global_max_height = log_max_height + 1;
                    let total_rounds = log_max_height - log_blowup;

                    let lde_domain = CircleDomain::<F>::standard(log_height);
                    let bits_reduced = log_global_max_height - log_height;

                    for global_sampled_index in 0..(1usize << log_global_max_height) {
                        let m = global_sampled_index >> bits_reduced;
                        let orig_idx = cfft_permute_index(m, log_height);
                        let p = lde_domain.nth_point(orig_idx);

                        // First-layer (bivariate) y-twiddle: shared across the fold pair,
                        // canonically the "b=0" member's y-coordinate.
                        let idx_y_true = global_sampled_index >> (bits_reduced + 1);
                        let t_y_true = lde_domain.nth_y_twiddle(idx_y_true);
                        let b = m & 1;
                        let y_twiddle_hyp = if b == 0 { p.y } else { -p.y };
                        assert_eq!(
                            y_twiddle_hyp, t_y_true,
                            "y-twiddle mismatch: log_blowup={log_blowup} matrix_log_n={matrix_log_n} \
                             extra_rounds_above={extra_rounds_above} global_sampled_index={global_sampled_index}"
                        );

                        // x-twiddle chain: shared by the whole query (independent of which
                        // matrix rolls in where), seeded once at FRI round 0.
                        let top_level_index = global_sampled_index >> 1;

                        let seed_index_r = top_level_index >> 1;
                        let seed_log_folded_height = log_max_height - 1;
                        let seed_domain_log_n = seed_log_folded_height + 2;
                        let seed_idx = reverse_bits_len(seed_index_r, seed_log_folded_height);
                        let mut x_closed =
                            CircleDomain::<F>::standard(seed_domain_log_n).nth_x_twiddle(seed_idx);

                        for r in 0..total_rounds {
                            let index_r = top_level_index >> (r + 1);
                            let log_current_height_r = log_max_height - r;
                            let log_folded_height_r = log_current_height_r - 1;
                            let domain_log_n = log_folded_height_r + 2; // + log_arity(1) + 1
                            let idx_for_twiddle = reverse_bits_len(index_r, log_folded_height_r);
                            let t_x_true = CircleDomain::<F>::standard(domain_log_n)
                                .nth_x_twiddle(idx_for_twiddle);

                            assert_eq!(
                                x_closed, t_x_true,
                                "x-twiddle mismatch at round {r}: log_blowup={log_blowup} \
                                 matrix_log_n={matrix_log_n} extra_rounds_above={extra_rounds_above} \
                                 global_sampled_index={global_sampled_index}"
                            );
                            x_closed = x_closed.square().double() - F::ONE;
                            if (top_level_index >> (r + 1)) & 1 == 1 {
                                x_closed = -x_closed;
                            }
                        }
                        let _ = p;
                    }
                }
            }
        }
    }

    #[test]
    fn fold_matrix_same_as_row() {
        let mut rng = SmallRng::seed_from_u64(1);
        let log_folded_height = 5;
        let log_arity = 1; // arity 2
        let m = RowMajorMatrix::<EF>::rand(&mut rng, 1 << log_folded_height, 2);
        let beta: EF = rng.random();

        let mat_y_folded = fold_y::<F, EF>(beta, &m);
        let row_y_folded = (0..(1 << log_folded_height))
            .map(|i| fold_y_row::<F, EF>(i, log_folded_height, beta, m.row(i).unwrap().into_iter()))
            .collect_vec();
        assert_eq!(mat_y_folded, row_y_folded);

        let mat_x_folded = fold_x::<F, EF>(beta, log_arity, &m);
        let row_x_folded = (0..(1 << log_folded_height))
            .map(|i| {
                fold_x_row::<F, EF>(
                    i,
                    log_folded_height,
                    log_arity,
                    beta,
                    m.row(i).unwrap().into_iter(),
                )
            })
            .collect_vec();
        assert_eq!(mat_x_folded, row_x_folded);
    }

    #[test]
    fn folded_matrix_remains_low_degree() {
        let vec_dim = |evals: &[F]| {
            CircleEvaluations::from_cfft_order(
                CircleDomain::standard(log2_strict_usize(evals.len())),
                RowMajorMatrix::new_col(evals.to_vec()),
            )
            .dim()
        };

        let log_arity = 1; // arity 2
        let mut rng = SmallRng::seed_from_u64(1);
        for (log_n, log_blowup) in iproduct!(3..6, 1..4) {
            let mut values = CircleEvaluations::evaluate(
                CircleDomain::standard(log_n + log_blowup),
                RowMajorMatrix::rand(&mut rng, 1 << log_n, 1),
            )
            .to_cfft_order()
            .values;

            values = fold_y(rng.random(), &RowMajorMatrix::new(values, 2));
            assert_eq!(vec_dim(&values), values.len() >> log_blowup);
            for _ in 0..(log_n - 1) {
                values = fold_x(rng.random(), log_arity, &RowMajorMatrix::new(values, 2));
                assert_eq!(vec_dim(&values), values.len() >> log_blowup);
            }
        }
    }
}
