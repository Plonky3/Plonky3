//! Per-round AIR zerocheck state.
//!
//! Builds round polynomials for `sum_x eq(tau, x) * g(x)` and folds state across challenges.

use alloc::vec::Vec;

use p3_air::Air;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::folder::MultilinearFolder;
use crate::packed_ext::PackedExt;
use crate::selectors::BoundaryEvals;

/// Sumcheck prover state for the AIR zerocheck.
///
/// Stores the base trace as one transposed row-major matrix: each matrix row is one trace column.
#[derive(Debug)]
pub(crate) struct RoundStateBase<'a, A, F, EF> {
    /// AIR whose alpha-batched constraint is being evaluated.
    air: &'a A,
    /// Public inputs forwarded to the AIR.
    public_values: &'a [F],
    /// Random scalar batching the AIR constraints.
    alpha: EF,
    /// Equality weight over the current round's suffix variables.
    eq_suffix: Poly<EF>,
    /// Main trace columns, stored as a row-major matrix with one original column per row.
    columns: RowMajorMatrix<F>,
    /// Per-round sumcheck degree.
    degree: usize,
}

/// Extension-field sumcheck state after the first base-field round.
pub(crate) struct RoundStateExt<'a, A, F, EF> {
    /// AIR whose alpha-batched constraint is being evaluated.
    air: &'a A,
    /// Public inputs forwarded to the AIR, in the base field.
    public_values: &'a [F],
    /// Random scalar batching the AIR constraints.
    alpha: EF,
    /// Equality weight over the current round's suffix variables.
    eq_suffix: Poly<EF>,
    /// Folded boundary-selector values at the current sumcheck prefix.
    boundary: BoundaryEvals<EF>,
    /// Main trace columns after the first base-field fold.
    columns: Vec<Poly<EF>>,
    /// Repeat-last successor values for each main column at the folded tail row.
    next_tail: Vec<EF>,
    /// Per-round sumcheck degree.
    degree: usize,
}

/// Per-worker scratch for the extension-field round-polynomial fold.
///
/// `out` accumulates the round-polynomial evaluations; the row buffers hold one
/// residual row's interpolation node and its per-step difference. One instance
/// is allocated per worker and reused across that worker's rows.
struct ExtScratch<EF> {
    out: Vec<EF>,
    local_point: Vec<EF>,
    local_diff: Vec<EF>,
    next_point: Vec<EF>,
    next_diff: Vec<EF>,
}

/// Per-worker scratch for the packed base-field first-round fold.
///
/// Mirrors [`ExtScratch`] with packed base-field row buffers and a pair of
/// per-lane equality-weight buffers. One instance is allocated per worker and
/// reused across that worker's packed blocks.
struct PackedScratch<P, EF> {
    out: Vec<EF>,
    local_point: Vec<P>,
    local_diff: Vec<P>,
    next_point: Vec<P>,
    next_diff: Vec<P>,
}

impl<EF: PrimeCharacteristicRing> ExtScratch<EF> {
    fn new(degree: usize, width: usize) -> Self {
        Self {
            out: EF::zero_vec(degree),
            local_point: EF::zero_vec(width),
            local_diff: EF::zero_vec(width),
            next_point: EF::zero_vec(width),
            next_diff: EF::zero_vec(width),
        }
    }
}

impl<P, EF> PackedScratch<P, EF>
where
    P: PrimeCharacteristicRing,
    EF: PrimeCharacteristicRing,
{
    fn new(degree: usize, width: usize) -> Self {
        Self {
            out: EF::zero_vec(degree),
            local_point: P::zero_vec(width),
            local_diff: P::zero_vec(width),
            next_point: P::zero_vec(width),
            next_diff: P::zero_vec(width),
        }
    }
}

impl<'a, A, F, EF> RoundStateBase<'a, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the prover state from a trace and a sampled zerocheck point.
    #[tracing::instrument(skip_all)]
    pub(crate) fn new(
        air: &'a A,
        public_values: &'a [F],
        alpha: EF,
        tau: &Point<EF>,
        trace: &'a RowMajorMatrix<F>,
        degree: usize,
    ) -> Self {
        // TODO: we may want to send cm_trace directly since PCS needs Vec<Poly> representation of witneses
        let columns = tracing::info_span!("transpose").in_scope(|| trace.transpose());
        Self {
            air,
            public_values,
            alpha,
            eq_suffix: Poly::new_from_point(&tau.as_slice()[1..], EF::ONE),
            columns,
            degree,
        }
    }

    const fn num_evals(&self) -> usize {
        self.eq_suffix.num_evals() * 2
    }

    fn width(&self) -> usize {
        self.columns.height()
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn round_poly(&self) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        if self.num_evals() / 2 < F::Packing::WIDTH {
            self.round_poly_unpacked()
        } else {
            self.round_poly_packed()
        }
    }

    #[tracing::instrument(skip_all)]
    fn round_poly_packed(&self) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        let width = self.width();
        let height = self.num_evals();
        let scalar_half = height / 2;
        let packing_width = F::Packing::WIDTH;
        let packed_half = scalar_half / packing_width;
        let degree = self.degree;
        let alpha = EF::ExtensionPacking::from(self.alpha);
        assert_ne!(packed_half, 0);

        self.eq_suffix
            .as_slice()
            .par_chunks_exact(packing_width)
            .enumerate()
            .par_fold_reduce(
                || PackedScratch::new(degree, width),
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;

                    for ((((local, local_delta), next), next_delta), column) in scratch
                        .local_point
                        .iter_mut()
                        .zip(scratch.local_diff.iter_mut())
                        .zip(scratch.next_point.iter_mut())
                        .zip(scratch.next_diff.iter_mut())
                        .zip(self.columns.row_slices())
                    {
                        let local_lo = *F::Packing::from_slice(&column[s..s + packing_width]);
                        let local_hi = *F::Packing::from_slice(
                            &column[s + scalar_half..s + scalar_half + packing_width],
                        );
                        *local = local_lo;
                        *local_delta = local_hi - local_lo;

                        let next_lo =
                            *F::Packing::from_slice(&column[s + 1..s + 1 + packing_width]);
                        let next_hi_start = s + scalar_half + 1;
                        let next_hi = if next_hi_start + packing_width <= height {
                            *F::Packing::from_slice(
                                &column[next_hi_start..next_hi_start + packing_width],
                            )
                        } else {
                            F::Packing::from_fn(|lane| {
                                let row = next_hi_start + lane;
                                if row < height {
                                    column[row]
                                } else {
                                    column[height - 1]
                                }
                            })
                        };
                        *next = next_lo;
                        *next_delta = next_hi - next_lo;
                    }

                    let (mut boundary, boundary_diff) =
                        BoundaryEvals::<F::Packing>::row_pair_packed(s, scalar_half, height);

                    scratch
                        .local_point
                        .iter_mut()
                        .zip(scratch.local_diff.iter())
                        .zip(scratch.next_point.iter_mut())
                        .zip(scratch.next_diff.iter())
                        .for_each(|(((local, local_diff), next), next_diff)| {
                            *local += *local_diff;
                            *next += *next_diff;
                        });
                    boundary += boundary_diff;

                    for acc in &mut scratch.out[1..] {
                        scratch
                            .local_point
                            .iter_mut()
                            .zip(scratch.local_diff.iter())
                            .zip(scratch.next_point.iter_mut())
                            .zip(scratch.next_diff.iter())
                            .for_each(|(((local, local_diff), next), next_diff)| {
                                *local += *local_diff;
                                *next += *next_diff;
                            });
                        boundary += boundary_diff;

                        let g = MultilinearFolder::new(
                            &scratch.local_point,
                            &scratch.next_point,
                            boundary,
                            self.public_values,
                            alpha,
                        )
                        .eval_air(self.air);
                        *acc += dot_product::<EF, _, _>(
                            eq_suffix.iter().copied(),
                            EF::ExtensionPacking::to_ext_iter([g]),
                        );
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.out
                        .iter_mut()
                        .zip(rhs.out)
                        .for_each(|(lhs, rhs)| *lhs += rhs);
                    lhs
                },
            )
            .out
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn round_poly_unpacked(&self) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>,
    {
        let width = self.width();
        let height = self.num_evals();
        let half = height / 2;

        let mut out = EF::zero_vec(self.degree);
        let mut local_point = F::zero_vec(width);
        let mut local_diff = F::zero_vec(width);
        let mut next_point = F::zero_vec(width);
        let mut next_diff = F::zero_vec(width);

        for (s, &eq_suffix) in self.eq_suffix.as_slice().iter().enumerate() {
            local_point
                .iter_mut()
                .zip(local_diff.iter_mut())
                .zip(next_point.iter_mut())
                .zip(next_diff.iter_mut())
                .zip(self.columns.row_slices())
                .for_each(|((((local, local_delta), next), next_delta), column)| {
                    let local_lo = column[s];
                    let local_hi = column[s + half];
                    *local = local_lo;
                    *local_delta = local_hi - local_lo;

                    let next_lo = column[s + 1];
                    let next_hi = if s + half + 1 < height {
                        column[s + half + 1]
                    } else {
                        column[height - 1]
                    };
                    *next = next_lo;
                    *next_delta = next_hi - next_lo;
                });

            let (mut boundary, boundary_diff) = BoundaryEvals::<F>::row_pair(s, half, height);

            F::add_slices(&mut local_point, &local_diff);
            F::add_slices(&mut next_point, &next_diff);
            boundary += boundary_diff;

            for acc in &mut out[1..] {
                F::add_slices(&mut local_point, &local_diff);
                F::add_slices(&mut next_point, &next_diff);
                boundary += boundary_diff;

                let g = MultilinearFolder::new(
                    &local_point,
                    &next_point,
                    boundary,
                    self.public_values,
                    self.alpha,
                )
                .eval_air(self.air);
                *acc += eq_suffix * g;
            }
        }

        out
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn fold(self, r: EF) -> RoundStateExt<'a, A, F, EF> {
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let next_tail = self
            .columns
            .row_slices()
            .map(|col| EF::from(col[half]) + r * (col[num_evals - 1] - col[half]))
            .collect::<Vec<_>>();

        let mut eq_suffix = self.eq_suffix;
        eq_suffix.sum_prefix_var_mut();

        let columns = self
            .columns
            .par_row_slices()
            .map(|col| Poly::fix_prefix_var_from_evals(col, r))
            .collect();

        RoundStateExt {
            air: self.air,
            public_values: self.public_values,
            alpha: self.alpha,
            eq_suffix,
            degree: self.degree,
            columns,
            next_tail,
            boundary: BoundaryEvals::new(EF::ONE - r, r, EF::ONE - r),
        }
    }
}

impl<'a, A, F, EF> RoundStateExt<'a, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    const fn num_evals(&self) -> usize {
        self.eq_suffix.num_evals() * 2
    }

    const fn width(&self) -> usize {
        self.columns.len()
    }

    pub(crate) fn evals(self) -> (Vec<EF>, Vec<EF>, BoundaryEvals<EF>) {
        let columns = self
            .columns
            .iter()
            .map(|poly| poly.as_constant().unwrap())
            .collect();
        (columns, self.next_tail, self.boundary)
    }

    /// Computes the round polynomial evaluations, dispatching to the
    /// SIMD-packed kernel once there are enough residual rows to fill a
    /// packed lane, mirroring [`RoundStateBase::round_poly`].
    #[tracing::instrument(skip_all)]
    pub(crate) fn round_poly(&self) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>>
            + for<'b> Air<
                MultilinearFolder<
                    'b,
                    F,
                    PackedExt<F, EF::ExtensionPacking>,
                    PackedExt<F, EF::ExtensionPacking>,
                >,
            >,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        if self.num_evals() / 2 < F::Packing::WIDTH {
            self.round_poly_unpacked()
        } else {
            self.round_poly_packed()
        }
    }

    #[tracing::instrument(skip_all)]
    fn round_poly_unpacked(&self) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>>,
    {
        let width = self.width();
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let degree = self.degree;

        self.eq_suffix
            .as_slice()
            .par_iter()
            .enumerate()
            .par_fold_reduce(
                || ExtScratch::new(degree, width),
                |mut scratch, (s, &eq_suffix)| {
                    for (((((local, local_delta), next), next_delta), column), next_tail) in scratch
                        .local_point
                        .iter_mut()
                        .zip(scratch.local_diff.iter_mut())
                        .zip(scratch.next_point.iter_mut())
                        .zip(scratch.next_diff.iter_mut())
                        .zip(self.columns.iter())
                        .zip(self.next_tail.iter())
                    {
                        let column = column.as_slice();
                        let local_lo = column[s];
                        let local_hi = column[s + half];
                        *local = local_lo;
                        *local_delta = local_hi - local_lo;

                        let next_lo = column[s + 1];
                        let next_hi_row = s + half;
                        let next_hi = if next_hi_row + 1 < num_evals {
                            column[next_hi_row + 1]
                        } else {
                            *next_tail
                        };
                        *next = next_lo;
                        *next_delta = next_hi - next_lo;
                    }

                    let (mut boundary, boundary_diff) =
                        BoundaryEvals::row_pair_with_prefix(s, half, num_evals, self.boundary);

                    let g = MultilinearFolder::new(
                        &scratch.local_point,
                        &scratch.next_point,
                        boundary,
                        self.public_values,
                        self.alpha,
                    )
                    .eval_air(self.air);
                    scratch.out[0] += eq_suffix * g;

                    EF::add_slices(&mut scratch.local_point, &scratch.local_diff);
                    EF::add_slices(&mut scratch.next_point, &scratch.next_diff);
                    boundary += boundary_diff;

                    for acc in &mut scratch.out[1..] {
                        EF::add_slices(&mut scratch.local_point, &scratch.local_diff);
                        EF::add_slices(&mut scratch.next_point, &scratch.next_diff);
                        boundary += boundary_diff;

                        let g = MultilinearFolder::new(
                            &scratch.local_point,
                            &scratch.next_point,
                            boundary,
                            self.public_values,
                            self.alpha,
                        )
                        .eval_air(self.air);
                        *acc += eq_suffix * g;
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.out
                        .iter_mut()
                        .zip(rhs.out)
                        .for_each(|(lhs, rhs)| *lhs += rhs);
                    lhs
                },
            )
            .out
    }

    /// SIMD-packed twin of [`Self::round_poly_unpacked`].
    ///
    /// Every column is already extension-valued (folded by earlier rounds),
    /// so packing groups `F::Packing::WIDTH` consecutive residual rows of
    /// each `Poly<EF>` column into one `EF::ExtensionPacking`, via an
    /// unaligned [`PackedFieldExtension::from_ext_slice`] read exactly like
    /// [`RoundStateBase::round_poly_packed`] does for its base-field
    /// columns. The AIR is driven through [`PackedExt`], which wraps the
    /// packed extension value so it supports arithmetic against the base
    /// field `F` directly (see that type's docs for why this is needed).
    #[tracing::instrument(skip_all)]
    fn round_poly_packed(&self) -> Vec<EF>
    where
        A: for<'b> Air<
            MultilinearFolder<
                'b,
                F,
                PackedExt<F, EF::ExtensionPacking>,
                PackedExt<F, EF::ExtensionPacking>,
            >,
        >,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        let width = self.width();
        let height = self.num_evals();
        let scalar_half = height / 2;
        let packing_width = F::Packing::WIDTH;
        let packed_half = scalar_half / packing_width;
        let degree = self.degree;
        let alpha = PackedExt::new(EF::ExtensionPacking::from(self.alpha));
        assert_ne!(packed_half, 0);

        self.eq_suffix
            .as_slice()
            .par_chunks_exact(packing_width)
            .enumerate()
            .par_fold_reduce(
                || PackedScratch::<PackedExt<F, EF::ExtensionPacking>, EF>::new(degree, width),
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;

                    for (((((local, local_delta), next), next_delta), column), next_tail) in scratch
                        .local_point
                        .iter_mut()
                        .zip(scratch.local_diff.iter_mut())
                        .zip(scratch.next_point.iter_mut())
                        .zip(scratch.next_diff.iter_mut())
                        .zip(self.columns.iter())
                        .zip(self.next_tail.iter())
                    {
                        let column = column.as_slice();
                        let local_lo = PackedExt::new(EF::ExtensionPacking::from_ext_slice(
                            &column[s..s + packing_width],
                        ));
                        let local_hi = PackedExt::new(EF::ExtensionPacking::from_ext_slice(
                            &column[s + scalar_half..s + scalar_half + packing_width],
                        ));
                        *local = local_lo;
                        *local_delta = local_hi - local_lo;

                        let next_lo = PackedExt::new(EF::ExtensionPacking::from_ext_slice(
                            &column[s + 1..s + 1 + packing_width],
                        ));
                        let next_hi_start = s + scalar_half + 1;
                        let next_hi = if next_hi_start + packing_width <= height {
                            PackedExt::new(EF::ExtensionPacking::from_ext_slice(
                                &column[next_hi_start..next_hi_start + packing_width],
                            ))
                        } else {
                            PackedExt::new(EF::ExtensionPacking::from_ext_fn(|lane| {
                                let row = next_hi_start + lane;
                                if row < height {
                                    column[row]
                                } else {
                                    *next_tail
                                }
                            }))
                        };
                        *next = next_lo;
                        *next_delta = next_hi - next_lo;
                    }

                    let (raw_boundary, raw_boundary_diff) =
                        BoundaryEvals::row_pair_with_prefix_packed::<F>(
                            s,
                            scalar_half,
                            height,
                            self.boundary,
                        );
                    let mut boundary = BoundaryEvals::new(
                        PackedExt::new(raw_boundary.first),
                        PackedExt::new(raw_boundary.last),
                        PackedExt::new(raw_boundary.transition),
                    );
                    let boundary_diff = BoundaryEvals::new(
                        PackedExt::new(raw_boundary_diff.first),
                        PackedExt::new(raw_boundary_diff.last),
                        PackedExt::new(raw_boundary_diff.transition),
                    );

                    let g = MultilinearFolder::new(
                        &scratch.local_point,
                        &scratch.next_point,
                        boundary,
                        self.public_values,
                        alpha,
                    )
                    .eval_air(self.air);
                    scratch.out[0] += dot_product::<EF, _, _>(
                        eq_suffix.iter().copied(),
                        EF::ExtensionPacking::to_ext_iter([g.0]),
                    );

                    scratch
                        .local_point
                        .iter_mut()
                        .zip(scratch.local_diff.iter())
                        .zip(scratch.next_point.iter_mut())
                        .zip(scratch.next_diff.iter())
                        .for_each(|(((local, local_diff), next), next_diff)| {
                            *local += *local_diff;
                            *next += *next_diff;
                        });
                    boundary += boundary_diff;

                    for acc in &mut scratch.out[1..] {
                        scratch
                            .local_point
                            .iter_mut()
                            .zip(scratch.local_diff.iter())
                            .zip(scratch.next_point.iter_mut())
                            .zip(scratch.next_diff.iter())
                            .for_each(|(((local, local_diff), next), next_diff)| {
                                *local += *local_diff;
                                *next += *next_diff;
                            });
                        boundary += boundary_diff;

                        let g = MultilinearFolder::new(
                            &scratch.local_point,
                            &scratch.next_point,
                            boundary,
                            self.public_values,
                            alpha,
                        )
                        .eval_air(self.air);
                        *acc += dot_product::<EF, _, _>(
                            eq_suffix.iter().copied(),
                            EF::ExtensionPacking::to_ext_iter([g.0]),
                        );
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.out
                        .iter_mut()
                        .zip(rhs.out)
                        .for_each(|(lhs, rhs)| *lhs += rhs);
                    lhs
                },
            )
            .out
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn fold(&mut self, r: EF) {
        let num_evals = self.num_evals();
        let half = num_evals / 2;

        self.next_tail = self
            .columns
            .iter()
            .zip(self.next_tail.iter())
            .map(|(col, &next_tail)| {
                let lo = col.as_slice()[half];
                lo + r * (next_tail - lo)
            })
            .collect();

        self.eq_suffix.sum_prefix_var_mut();

        // Both halves already live in the extension field.
        // So each column folds in place, reusing its buffer.
        // No per-round reallocation is needed.
        self.columns
            .par_iter_mut()
            .for_each(|col| col.fix_prefix_var_mut(r));

        self.boundary.apply(r);
    }
}
