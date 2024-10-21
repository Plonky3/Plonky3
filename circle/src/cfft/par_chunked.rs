use core::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_commit::PolynomialSpace;
use p3_dft::{divide_by_height, Butterfly, DifButterfly, DitButterfly};
use p3_field::{batch_multiplicative_inverse, extension::ComplexExtendable, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::{debug_span, instrument};

use crate::{cfft::compute_twiddles, CircleDomain};

use super::{CfftAlgorithm, CircleEvaluations};

#[derive(Default)]
pub struct ParChunkedCfft<F>(PhantomData<F>);

impl<F: ComplexExtendable> CfftAlgorithm<F> for ParChunkedCfft<F> {
    #[instrument(skip_all, fields(dims = %evals.values.dimensions()))]
    fn interpolate<M: Matrix<F>>(&self, evals: CircleEvaluations<F, M>) -> RowMajorMatrix<F> {
        let CircleEvaluations { domain, values } = evals;
        let mut values = debug_span!("to_rmm").in_scope(|| values.to_row_major_matrix());

        let mut twiddles = debug_span!("twiddles").in_scope(|| {
            compute_twiddles(domain)
                .into_iter()
                .map(|ts| {
                    batch_multiplicative_inverse(&ts)
                        .into_iter()
                        .map(|t| DifButterfly(t))
                        .collect_vec()
                })
                .peekable()
        });

        assert_eq!(twiddles.len(), domain.log_n);

        let par_twiddles = twiddles
            .peeking_take_while(|ts| ts.len() >= desired_num_jobs())
            .collect_vec();
        if let Some(min_blks) = par_twiddles.last().map(|ts| ts.len()) {
            let max_blk_sz = values.height() / min_blks;
            debug_span!("par_layers", log_min_blks = log2_strict_usize(min_blks)).in_scope(|| {
                values
                    .par_row_chunks_exact_mut(max_blk_sz)
                    .enumerate()
                    .for_each(|(chunk_i, submat)| {
                        for ts in &par_twiddles {
                            let twiddle_chunk_sz = ts.len() / min_blks;
                            let twiddle_chunk = &ts
                                [(twiddle_chunk_sz * chunk_i)..(twiddle_chunk_sz * (chunk_i + 1))];
                            serial_layer(submat.values, twiddle_chunk);
                        }
                    });
            });
        }

        for ts in twiddles {
            par_within_blk_layer(&mut values.values, &ts);
        }

        // TODO: omit this?
        divide_by_height(&mut values);
        values
    }

    #[instrument(skip_all, fields(dims = %coeffs.dimensions()))]
    fn evaluate(
        &self,
        domain: CircleDomain<F>,
        mut coeffs: RowMajorMatrix<F>,
    ) -> CircleEvaluations<F> {
        let log_n = log2_strict_usize(coeffs.height());
        assert!(log_n <= domain.log_n);

        if log_n < domain.log_n {
            // We could simply pad coeffs like this:
            // coeffs.pad_to_height(target_domain.size(), F::zero());
            // But the first `added_bits` layers will simply fill out the zeros
            // with the lower order values. (In `DitButterfly`, `x_2` is 0, so
            // both `x_1` and `x_2` are set to `x_1`).
            // So instead we directly repeat the coeffs and skip the initial layers.
            debug_span!("extend coeffs").in_scope(|| {
                coeffs.values.reserve(domain.size() * coeffs.width());
                for _ in log_n..domain.log_n {
                    coeffs.values.extend_from_within(..);
                }
            });
        }
        assert_eq!(coeffs.height(), 1 << domain.log_n);

        let mut twiddles = debug_span!("twiddles").in_scope(|| {
            compute_twiddles(domain)
                .into_iter()
                .map(|ts| ts.into_iter().map(|t| DitButterfly(t)).collect_vec())
                .rev()
                .skip(domain.log_n - log_n)
                .peekable()
        });

        for ts in twiddles.peeking_take_while(|ts| ts.len() < desired_num_jobs()) {
            par_within_blk_layer(&mut coeffs.values, &ts);
        }

        let par_twiddles = twiddles.collect_vec();
        if let Some(min_blks) = par_twiddles.first().map(|ts| ts.len()) {
            let max_blk_sz = coeffs.height() / min_blks;
            debug_span!("par_layers", log_min_blks = log2_strict_usize(min_blks)).in_scope(|| {
                coeffs
                    .par_row_chunks_exact_mut(max_blk_sz)
                    .enumerate()
                    .for_each(|(chunk_i, submat)| {
                        for ts in &par_twiddles {
                            let twiddle_chunk_sz = ts.len() / min_blks;
                            let twiddle_chunk = &ts
                                [(twiddle_chunk_sz * chunk_i)..(twiddle_chunk_sz * (chunk_i + 1))];
                            serial_layer(submat.values, twiddle_chunk);
                        }
                    });
            });
        }

        CircleEvaluations::from_cfft_order(domain, coeffs)
    }
}

#[inline]
fn serial_layer<F: Field, B: Butterfly<F>>(values: &mut [F], twiddles: &[B]) {
    let blk_sz = values.len() / twiddles.len();
    for (&t, blk) in izip!(twiddles, values.chunks_exact_mut(blk_sz)) {
        let (lo, hi) = blk.split_at_mut(blk_sz / 2);
        t.apply_to_rows(lo, hi);
    }
}

#[inline]
#[instrument(level = "debug", skip_all, fields(log_blks = log2_strict_usize(twiddles.len())))]
fn par_within_blk_layer<F: Field, B: Butterfly<F>>(values: &mut [F], twiddles: &[B]) {
    let blk_sz = values.len() / twiddles.len();
    for (&t, blk) in izip!(twiddles, values.chunks_exact_mut(blk_sz)) {
        let (lo, hi) = blk.split_at_mut(blk_sz / 2);
        let job_sz = core::cmp::max(1, lo.len() >> log2_ceil_usize(desired_num_jobs()));
        lo.par_chunks_mut(job_sz)
            .zip(hi.par_chunks_mut(job_sz))
            .for_each(|(lo_job, hi_job)| t.apply_to_rows(lo_job, hi_job));
    }
}

#[inline]
fn desired_num_jobs() -> usize {
    16 * current_num_threads()
}
