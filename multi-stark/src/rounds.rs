//! Per-round AIR zerocheck state.
//!
//! Builds round polynomials for `sum_x eq(tau, x) * g(x)` and folds state across challenges.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
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
/// Stores the trace as one transposed row-major matrix: each matrix row is one trace column.
///
/// Three column groups fold identically and share one matrix, laid out in this order:
/// - main columns, then
/// - preprocessed columns, then
/// - periodic columns, each materialized to full trace height (`col[i mod period]`).
///
/// Two stored split indices separate the three views when a folder is built.
/// Periodic columns are uncommitted: they carry no opening claim.
/// The verifier recomputes them in closed form at the bound point.
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
    /// Main columns, then preprocessed columns, then periodic columns.
    /// Each matrix row is one original column.
    columns: RowMajorMatrix<F>,
    /// Count of leading rows that are main columns.
    main_width: usize,
    /// Count of rows following the main columns that are preprocessed columns.
    /// Every row after the first `main_width + preprocessed_width` is a periodic column.
    preprocessed_width: usize,
    /// Per-round sumcheck degree.
    degree: usize,
}

/// Extension-round column storage.
///
/// Columns stay SIMD-packed (one lane per residual row) as long as there are
/// enough residual rows left to fill a packed lane. Once a round's fold would
/// leave fewer than `F::Packing::WIDTH` residual rows, columns unpack to
/// scalar form; that transition happens at most once, since the residual row
/// count only ever shrinks.
enum ExtColumns<F: Field, EF: ExtensionField<F>> {
    Packed(Vec<Poly<EF::ExtensionPacking>>),
    Scalar(Vec<Poly<EF>>),
}

impl<F: Field, EF: ExtensionField<F>> ExtColumns<F, EF> {
    const fn len(&self) -> usize {
        match self {
            Self::Packed(cols) => cols.len(),
            Self::Scalar(cols) => cols.len(),
        }
    }

    /// Column view expected by [`RoundStateExt::round_poly_packed`].
    ///
    /// # Panics
    ///
    /// Panics if the columns have already unpacked to scalar form. Callers
    /// gate on the same width threshold that decides the storage variant, so
    /// this never fires.
    fn as_packed(&self) -> &[Poly<EF::ExtensionPacking>] {
        match self {
            Self::Packed(cols) => cols,
            Self::Scalar(_) => unreachable!("round_poly_packed requires packed columns"),
        }
    }

    /// Column view expected by [`RoundStateExt::round_poly_unpacked`].
    ///
    /// # Panics
    ///
    /// Panics if the columns are still packed. Callers gate on the same
    /// width threshold that decides the storage variant, so this never fires.
    fn as_scalar(&self) -> &[Poly<EF>] {
        match self {
            Self::Scalar(cols) => cols,
            Self::Packed(_) => unreachable!("round_poly_unpacked requires scalar columns"),
        }
    }

    /// Folds the prefix variable of every column at `r`, unpacking to scalar
    /// form when `want_packed` is false.
    ///
    /// # Panics
    ///
    /// Panics if `want_packed` is true while the columns are already scalar:
    /// the residual row count only shrinks round to round, so a fold can
    /// never make packed storage viable again once it stopped being viable.
    fn fold(self, r: EF, want_packed: bool) -> Self {
        match self {
            Self::Packed(mut cols) => {
                cols.par_iter_mut()
                    .for_each(|col| col.fix_prefix_var_mut(r));
                if want_packed {
                    Self::Packed(cols)
                } else {
                    Self::Scalar(cols.par_iter().map(Poly::unpack::<F, EF>).collect())
                }
            }
            Self::Scalar(mut cols) => {
                assert!(!want_packed, "columns cannot transition scalar -> packed");
                cols.par_iter_mut()
                    .for_each(|col| col.fix_prefix_var_mut(r));
                Self::Scalar(cols)
            }
        }
    }

    /// Reads the scalar value at flat residual index `idx` from every column.
    fn scalar_row_at(&self, idx: usize) -> Vec<EF> {
        match self {
            Self::Scalar(cols) => cols.iter().map(|col| col.as_slice()[idx]).collect(),
            Self::Packed(cols) => {
                let packing_width = F::Packing::WIDTH;
                let (group, lane) = (idx / packing_width, idx % packing_width);
                cols.iter()
                    .map(|col| col.as_slice()[group].extract(lane))
                    .collect()
            }
        }
    }
}

/// Reads `F::Packing::WIDTH` consecutive residual rows of a packed column
/// starting at scalar row `start`, falling back to `tail` past `len`.
///
/// The packed groups in `column` are aligned to multiples of the packing
/// width, so an offset window generally spans two adjacent groups; each lane
/// is reconstructed independently via [`PackedFieldExtension::extract`]
/// rather than assuming any particular memory layout.
#[inline]
fn packed_window<F: Field, EF: ExtensionField<F>>(
    column: &[EF::ExtensionPacking],
    start: usize,
    len: usize,
    tail: EF,
) -> EF::ExtensionPacking {
    let packing_width = F::Packing::WIDTH;
    EF::ExtensionPacking::from_ext_fn(|lane| {
        let row = start + lane;
        if row < len {
            column[row / packing_width].extract(row % packing_width)
        } else {
            tail
        }
    })
}

/// Extension-field sumcheck state after the first base-field round.
pub(crate) struct RoundStateExt<'a, A, F: Field, EF: ExtensionField<F>> {
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
    /// Main columns, then preprocessed columns, then periodic columns, after the first base-field fold.
    columns: ExtColumns<F, EF>,
    /// Count of leading columns that are main columns.
    main_width: usize,
    /// Count of columns following the main columns that are preprocessed columns.
    /// Every later column is a periodic column.
    preprocessed_width: usize,
    /// Repeat-last successor value for each column at the folded tail row.
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
///
/// `out` accumulates each round-polynomial evaluation as a packed extension
/// value, one SIMD lane per row in the current chunk, deferring the
/// horizontal lane reduction to a single pass after all chunks are folded in.
struct PackedScratch<P, Acc> {
    out: Vec<Acc>,
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

impl<P, Acc> PackedScratch<P, Acc>
where
    P: PrimeCharacteristicRing,
    Acc: PrimeCharacteristicRing,
{
    fn new(degree: usize, width: usize) -> Self {
        Self {
            out: Acc::zero_vec(degree),
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
    ///
    /// The preprocessed columns, when present, are appended after the main columns.
    /// Periodic columns are appended last, each materialized to the full trace height.
    /// The main and preprocessed traces must share the same height.
    /// So every column has the same arity.
    ///
    /// A period-`p` periodic column repeats every `p` rows.
    /// Its full-height column therefore holds `col[i mod p]` at row `i`.
    /// This full column is a genuine multilinear polynomial over the trace variables.
    /// So it folds exactly like a committed column.
    /// Its fold to the bound point equals the column's multilinear extension there.
    ///
    /// # Arguments
    ///
    /// - `air`: the AIR whose alpha-batched constraint is evaluated.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `alpha`: random scalar batching the constraints.
    /// - `tau`: the sampled zerocheck point.
    /// - `trace`: the main execution trace, one column per main AIR column.
    /// - `preprocessed`: the preprocessed trace, or `None` when the AIR declares none.
    /// - `degree`: the per-round sumcheck degree.
    ///
    /// # Panics
    ///
    /// Panics if the preprocessed trace height differs from the main trace height.
    /// Panics if a periodic column's period is not a power of two dividing the trace height.
    #[tracing::instrument(skip_all)]
    pub(crate) fn new(
        air: &'a A,
        public_values: &'a [F],
        alpha: EF,
        tau: &Point<EF>,
        trace: &'a RowMajorMatrix<F>,
        preprocessed: Option<&RowMajorMatrix<F>>,
        degree: usize,
    ) -> Self
    where
        A: BaseAir<F>,
    {
        // TODO: we may want to send cm_trace directly since PCS needs Vec<Poly> representation of witneses
        // One matrix row per column: transpose lays each column out contiguously.
        let main_width = trace.width;
        let preprocessed_width = preprocessed.map_or(0, |p| p.width);
        let columns = tracing::info_span!("transpose").in_scope(|| {
            // Transposed main trace: one row per column, each row `trace_height` long.
            let main = trace.transpose();
            let trace_height = main.width;
            let mut values = main.values;

            // Append the preprocessed columns as extra rows of the same length.
            if let Some(preprocessed) = preprocessed {
                let prep = preprocessed.transpose();
                assert_eq!(
                    prep.width, trace_height,
                    "preprocessed trace height must match the main trace height"
                );
                values.extend_from_slice(&prep.values);
            }

            // Append each periodic column, expanded to the full trace height.
            for col in air.periodic_columns().iter() {
                let period = col.len();
                assert!(
                    period.is_power_of_two() && trace_height.is_multiple_of(period),
                    "periodic column period must be a power of two dividing the trace height"
                );
                // Row i reads value i mod period.
                // So the column cycles through the p values.
                values.extend((0..trace_height).map(|i| col[i % period]));
            }

            RowMajorMatrix::new(values, trace_height)
        });
        Self {
            air,
            public_values,
            alpha,
            eq_suffix: Poly::new_from_point(&tau.as_slice()[1..], EF::ONE),
            columns,
            main_width,
            preprocessed_width,
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
        let main_width = self.main_width;
        // End of the preprocessed group.
        // Every column after this is a periodic column.
        let prep_end = main_width + self.preprocessed_width;
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
                || PackedScratch::<F::Packing, EF::ExtensionPacking>::new(degree, width),
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;
                    let eq_suffix = EF::ExtensionPacking::from_ext_slice(eq_suffix);

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
                            &scratch.local_point[..main_width],
                            &scratch.next_point[..main_width],
                            boundary,
                            self.public_values,
                            alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[main_width..prep_end],
                            &scratch.next_point[main_width..prep_end],
                        )
                        .with_periodic(&scratch.local_point[prep_end..])
                        .eval_air(self.air);
                        *acc += g * eq_suffix;
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
            .into_iter()
            .map(|acc| EF::ExtensionPacking::to_ext_iter([acc]).sum())
            .collect()
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn round_poly_unpacked(&self) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>,
    {
        let width = self.width();
        let main_width = self.main_width;
        // End of the preprocessed group.
        // Every column after this is a periodic column.
        let prep_end = main_width + self.preprocessed_width;
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
                    &local_point[..main_width],
                    &next_point[..main_width],
                    boundary,
                    self.public_values,
                    self.alpha,
                )
                .with_preprocessed(
                    &local_point[main_width..prep_end],
                    &next_point[main_width..prep_end],
                )
                .with_periodic(&local_point[prep_end..])
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

        let want_packed = (half / 2) >= F::Packing::WIDTH;
        let columns = if want_packed {
            ExtColumns::Packed(
                self.columns
                    .par_row_slices()
                    .map(|col| Poly::fix_prefix_var_to_packed_from_evals(col, r))
                    .collect(),
            )
        } else {
            ExtColumns::Scalar(
                self.columns
                    .par_row_slices()
                    .map(|col| Poly::fix_prefix_var_from_evals(col, r))
                    .collect(),
            )
        };

        RoundStateExt {
            air: self.air,
            public_values: self.public_values,
            alpha: self.alpha,
            eq_suffix,
            degree: self.degree,
            columns,
            main_width: self.main_width,
            preprocessed_width: self.preprocessed_width,
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

    /// Extracts the final scalar value of each column.
    ///
    /// # Panics
    ///
    /// Panics if the columns are still SIMD-packed. Callers only reach this
    /// once the sumcheck has bound every variable (`num_evals() == 1`), well
    /// below the packing width, so the columns have always unpacked by then.
    pub(crate) fn evals(self) -> (Vec<EF>, Vec<EF>, BoundaryEvals<EF>) {
        let columns = self
            .columns
            .as_scalar()
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
        let main_width = self.main_width;
        // End of the preprocessed group.
        // Every column after this is a periodic column.
        let prep_end = main_width + self.preprocessed_width;
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
                        .zip(self.columns.as_scalar().iter())
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
                        &scratch.local_point[..main_width],
                        &scratch.next_point[..main_width],
                        boundary,
                        self.public_values,
                        self.alpha,
                    )
                    .with_preprocessed(
                        &scratch.local_point[main_width..prep_end],
                        &scratch.next_point[main_width..prep_end],
                    )
                    .with_periodic(&scratch.local_point[prep_end..])
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
                            &scratch.local_point[..main_width],
                            &scratch.next_point[..main_width],
                            boundary,
                            self.public_values,
                            self.alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[main_width..prep_end],
                            &scratch.next_point[main_width..prep_end],
                        )
                        .with_periodic(&scratch.local_point[prep_end..])
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
        let main_width = self.main_width;
        // End of the preprocessed group.
        // Every column after this is a periodic column.
        let prep_end = main_width + self.preprocessed_width;
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
                || {
                    PackedScratch::<PackedExt<F, EF::ExtensionPacking>, EF::ExtensionPacking>::new(
                        degree, width,
                    )
                },
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;
                    let eq_suffix = EF::ExtensionPacking::from_ext_slice(eq_suffix);

                    for (((((local, local_delta), next), next_delta), column), next_tail) in scratch
                        .local_point
                        .iter_mut()
                        .zip(scratch.local_diff.iter_mut())
                        .zip(scratch.next_point.iter_mut())
                        .zip(scratch.next_diff.iter_mut())
                        .zip(self.columns.as_packed().iter())
                        .zip(self.next_tail.iter())
                    {
                        let column = column.as_slice();
                        // Aligned reads: `packed_s` indexes this chunk's own packed group directly.
                        let local_lo = PackedExt::new(column[packed_s]);
                        let local_hi = PackedExt::new(column[packed_s + packed_half]);
                        *local = local_lo;
                        *local_delta = local_hi - local_lo;

                        // Shifted-by-one reads cross packed-group boundaries, so they
                        // reconstruct their window lane by lane instead of indexing directly.
                        let next_lo = PackedExt::new(packed_window::<F, EF>(
                            column,
                            s + 1,
                            height,
                            *next_tail,
                        ));
                        let next_hi = PackedExt::new(packed_window::<F, EF>(
                            column,
                            s + scalar_half + 1,
                            height,
                            *next_tail,
                        ));
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
                        &scratch.local_point[..main_width],
                        &scratch.next_point[..main_width],
                        boundary,
                        self.public_values,
                        alpha,
                    )
                    .with_preprocessed(
                        &scratch.local_point[main_width..prep_end],
                        &scratch.next_point[main_width..prep_end],
                    )
                    .with_periodic(&scratch.local_point[prep_end..])
                    .eval_air(self.air);
                    scratch.out[0] += g.0 * eq_suffix;

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
                            &scratch.local_point[..main_width],
                            &scratch.next_point[..main_width],
                            boundary,
                            self.public_values,
                            alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[main_width..prep_end],
                            &scratch.next_point[main_width..prep_end],
                        )
                        .with_periodic(&scratch.local_point[prep_end..])
                        .eval_air(self.air);
                        *acc += g.0 * eq_suffix;
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
            .into_iter()
            .map(|acc| EF::ExtensionPacking::to_ext_iter([acc]).sum())
            .collect()
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn fold(&mut self, r: EF) {
        let num_evals = self.num_evals();
        let half = num_evals / 2;

        self.next_tail = self
            .columns
            .scalar_row_at(half)
            .into_iter()
            .zip(self.next_tail.iter())
            .map(|(lo, &next_tail)| lo + r * (next_tail - lo))
            .collect();

        self.eq_suffix.sum_prefix_var_mut();

        // Packed columns fold in packed form (one SIMD op per lane group instead
        // of one scalar EF mul per row), unpacking to scalar once residual rows
        // drop below the packing width.
        let want_packed = (half / 2) >= F::Packing::WIDTH;
        self.columns = core::mem::replace(&mut self.columns, ExtColumns::Scalar(Vec::new()))
            .fold(r, want_packed);

        self.boundary.apply(r);
    }
}
