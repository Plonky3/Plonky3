//! Per-round AIR zerocheck state.
//!
//! Builds round polynomials for `sum_x eq(tau, x) * g(x)` and folds state across challenges.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyView};
use p3_sumcheck::layout::Table;

use crate::folder::MultilinearFolder;
use crate::packed_ext::PackedExt;
use crate::selectors::BoundaryEvals;

/// Sumcheck prover state for the AIR zerocheck.
///
/// Stores the trace as one transposed row-major matrix: each matrix row is one trace column.
///
/// Preprocessed columns fold exactly like main columns.
/// They are laid out immediately after the main columns in the same matrix.
/// A stored split index separates the two views when a folder is built.
#[derive(Debug)]
pub(crate) struct RoundStateBase<'a, A, F: Field, EF> {
    /// AIR whose alpha-batched constraint is being evaluated.
    air: &'a A,
    /// Public inputs forwarded to the AIR.
    public_values: &'a [F],
    /// Random scalar batching the AIR constraints.
    alpha: EF,
    /// Equality weight over the current round's suffix variables.
    eq_suffix: Poly<EF>,
    /// Main trace columns, stored as one table row per original trace column.
    table: &'a Table<F>,
    /// Preprocessed columns
    preprocessed: Option<&'a Table<F>>,
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
                if want_packed {
                    cols.par_iter_mut()
                        .for_each(|col| col.fix_prefix_var_mut(r));
                    Self::Packed(cols)
                } else {
                    // Fold and unpack each column in a single pass.
                    Self::Scalar(
                        cols.into_par_iter()
                            .map(|mut col| {
                                col.fix_prefix_var_mut(r);
                                col.unpack::<F, EF>()
                            })
                            .collect(),
                    )
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
    /// Main columns then preprocessed columns, after the first base-field fold.
    columns: ExtColumns<F, EF>,
    /// Count of leading columns that are main columns.
    /// Every later column is a preprocessed column.
    main_width: usize,
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

/// Per-worker scratch for round-polynomial folds.
///
/// `out` accumulates the round-polynomial evaluations; the row buffers hold one
/// interpolation point and its per-step difference. One instance is allocated per
/// worker and reused across that worker's rows or packed blocks.
struct RoundScratch<P, Acc> {
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

impl<EF: Field> ExtScratch<EF> {
    fn add_diffs(&mut self) {
        EF::add_slices(&mut self.local_point, &self.local_diff);
        EF::add_slices(&mut self.next_point, &self.next_diff);
    }
}

impl<P, Acc> RoundScratch<P, Acc>
where
    P: PrimeCharacteristicRing + Copy,
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

    fn add_diffs(&mut self) {
        self.local_point
            .iter_mut()
            .zip(self.local_diff.iter())
            .zip(self.next_point.iter_mut())
            .zip(self.next_diff.iter())
            .for_each(|(((local, local_diff), next), next_diff)| {
                *local += *local_diff;
                *next += *next_diff;
            });
    }
}

impl<'a, A, F, EF> RoundStateBase<'a, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Build the prover state from trace columns and a sampled zerocheck point.
    #[tracing::instrument(skip_all)]
    pub(crate) fn new(
        air: &'a A,
        public_values: &'a [F],
        alpha: EF,
        tau: &Point<EF>,
        table: &'a Table<F>,
        preprocessed: Option<&'a Table<F>>,
        degree: usize,
    ) -> Self {
        assert_eq!(tau.num_variables(), table.num_variables());
        assert_eq!(air.width(), table.num_polys());
        assert_eq!(
            air.preprocessed_width(),
            preprocessed.map_or(0, Table::num_polys)
        );
        // Both traces bind the same zerocheck point, so they must share an arity.
        // The fold reads preprocessed columns at offsets derived from the main height.
        if let Some(preprocessed) = preprocessed {
            assert_eq!(
                preprocessed.num_variables(),
                table.num_variables(),
                "preprocessed trace height must match the main trace height"
            );
        }
        if let Some(air_degree) = air.max_constraint_degree() {
            assert_eq!(degree, air_degree);
        }
        Self {
            air,
            public_values,
            alpha,
            eq_suffix: Poly::new_from_point(&tau.as_slice()[1..], EF::ONE),
            preprocessed,
            table,
            degree,
        }
    }

    fn num_evals(&self) -> usize {
        self.eq_suffix.num_evals() * 2
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
        A: for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        let main_width = self.table.num_polys();
        let preprocessed_width = self.preprocessed.map_or(0, Table::num_polys);
        let width = main_width + preprocessed_width;
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
                || RoundScratch::<F::Packing, EF::ExtensionPacking>::new(degree, width),
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;
                    let eq_suffix = EF::ExtensionPacking::from_ext_slice(eq_suffix);
                    let fill_columns =
                        |scratch: &mut RoundScratch<F::Packing, EF::ExtensionPacking>,
                         offset: usize,
                         columns: &Table<F>| {
                            let end = offset + columns.num_polys();
                            let local_point = &mut scratch.local_point[offset..end];
                            let local_diff = &mut scratch.local_diff[offset..end];
                            let next_point = &mut scratch.next_point[offset..end];
                            let next_diff = &mut scratch.next_diff[offset..end];

                            for ((((local, local_diff), next), next_diff), column) in local_point
                                .iter_mut()
                                .zip(local_diff.iter_mut())
                                .zip(next_point.iter_mut())
                                .zip(next_diff.iter_mut())
                                .zip(columns.iter_polys())
                            {
                                let local_lo =
                                    *F::Packing::from_slice(&column[s..s + packing_width]);
                                let local_hi = *F::Packing::from_slice(
                                    &column[s + scalar_half..s + scalar_half + packing_width],
                                );
                                *local = local_lo;
                                *local_diff = local_hi - local_lo;

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
                                *next_diff = next_hi - next_lo;
                            }
                        };

                    fill_columns(&mut scratch, 0, self.table);

                    if let Some(preprocessed) = self.preprocessed {
                        fill_columns(&mut scratch, main_width, preprocessed);
                    }

                    let (mut boundary, boundary_diff) =
                        BoundaryEvals::<F::Packing>::row_pair_packed(s, scalar_half, height);

                    scratch.add_diffs();
                    boundary += boundary_diff;

                    for acc_idx in 1..scratch.out.len() {
                        scratch.add_diffs();
                        boundary += boundary_diff;

                        let g = MultilinearFolder::new(
                            &scratch.local_point[..main_width],
                            &scratch.next_point[..main_width],
                            boundary,
                            self.public_values,
                            alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[main_width..],
                            &scratch.next_point[main_width..],
                        )
                        .eval_air(self.air);
                        scratch.out[acc_idx] += g * eq_suffix;
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
        let main_width = self.table.num_polys();
        let preprocessed_width = self.preprocessed.map_or(0, Table::num_polys);
        let width = main_width + preprocessed_width;
        let height = self.num_evals();
        let half = height / 2;

        let mut scratch = RoundScratch::<F, EF>::new(self.degree, width);

        for (s, &eq_suffix) in self.eq_suffix.as_slice().iter().enumerate() {
            let fill_columns =
                |scratch: &mut RoundScratch<F, EF>, offset: usize, columns: &Table<F>| {
                    let end = offset + columns.num_polys();
                    let local_point = &mut scratch.local_point[offset..end];
                    let local_diff = &mut scratch.local_diff[offset..end];
                    let next_point = &mut scratch.next_point[offset..end];
                    let next_diff = &mut scratch.next_diff[offset..end];

                    for ((((local, local_diff), next), next_diff), column) in local_point
                        .iter_mut()
                        .zip(local_diff.iter_mut())
                        .zip(next_point.iter_mut())
                        .zip(next_diff.iter_mut())
                        .zip(columns.iter_polys())
                    {
                        let local_lo = column[s];
                        let local_hi = column[s + half];
                        *local = local_lo;
                        *local_diff = local_hi - local_lo;

                        let next_lo = column[s + 1];
                        let next_hi = if s + half + 1 < height {
                            column[s + half + 1]
                        } else {
                            column[height - 1]
                        };
                        *next = next_lo;
                        *next_diff = next_hi - next_lo;
                    }
                };

            fill_columns(&mut scratch, 0, self.table);

            if let Some(preprocessed) = self.preprocessed {
                fill_columns(&mut scratch, main_width, preprocessed);
            }

            let (mut boundary, boundary_diff) = BoundaryEvals::<F>::row_pair(s, half, height);

            scratch.add_diffs();
            boundary += boundary_diff;

            for acc_idx in 1..scratch.out.len() {
                scratch.add_diffs();
                boundary += boundary_diff;

                let g = MultilinearFolder::new(
                    &scratch.local_point[..main_width],
                    &scratch.next_point[..main_width],
                    boundary,
                    self.public_values,
                    self.alpha,
                )
                .with_preprocessed(
                    &scratch.local_point[main_width..],
                    &scratch.next_point[main_width..],
                )
                .eval_air(self.air);
                scratch.out[acc_idx] += eq_suffix * g;
            }
        }

        scratch.out
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn fold(self, r: EF) -> RoundStateExt<'a, A, F, EF> {
        let main_width = self.table.num_polys();
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let mut next_tail = self
            .table
            .iter_polys()
            .map(|col| EF::from(col[half]) + r * (col[num_evals - 1] - col[half]))
            .collect::<Vec<_>>();

        let mut eq_suffix = self.eq_suffix;
        eq_suffix.sum_prefix_var_mut();

        let want_packed = (half / 2) >= F::Packing::WIDTH;
        let columns = if want_packed {
            let mut columns = self
                .table
                .par_iter_polys()
                .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                .collect::<Vec<_>>();
            if let Some(preprocessed) = self.preprocessed {
                columns.extend(
                    preprocessed
                        .par_iter_polys()
                        .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                        .collect::<Vec<_>>(),
                );
            }
            ExtColumns::Packed(columns)
        } else {
            let mut columns = self
                .table
                .par_iter_polys()
                .map(|col| PolyView::new(col).fix_prefix_var(r))
                .collect::<Vec<_>>();
            if let Some(preprocessed) = self.preprocessed {
                columns.extend(
                    preprocessed
                        .par_iter_polys()
                        .map(|col| PolyView::new(col).fix_prefix_var(r))
                        .collect::<Vec<_>>(),
                );
            }
            ExtColumns::Scalar(columns)
        };

        if let Some(preprocessed) = self.preprocessed {
            next_tail.extend(
                preprocessed
                    .iter_polys()
                    .map(|col| EF::from(col[half]) + r * (col[num_evals - 1] - col[half])),
            );
        }

        RoundStateExt {
            air: self.air,
            public_values: self.public_values,
            alpha: self.alpha,
            eq_suffix,
            degree: self.degree,
            columns,
            main_width,
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
    fn num_evals(&self) -> usize {
        self.eq_suffix.num_evals() * 2
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
        let width = self.columns.len();
        let main_width = self.main_width;
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
                    for (((((local, local_diff), next), next_diff), column), next_tail) in scratch
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
                        *local_diff = local_hi - local_lo;

                        let next_lo = column[s + 1];
                        let next_hi_row = s + half;
                        let next_hi = if next_hi_row + 1 < num_evals {
                            column[next_hi_row + 1]
                        } else {
                            *next_tail
                        };
                        *next = next_lo;
                        *next_diff = next_hi - next_lo;
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
                        &scratch.local_point[main_width..],
                        &scratch.next_point[main_width..],
                    )
                    .eval_air(self.air);
                    scratch.out[0] += eq_suffix * g;

                    scratch.add_diffs();
                    boundary += boundary_diff;

                    for acc_idx in 1..scratch.out.len() {
                        scratch.add_diffs();
                        boundary += boundary_diff;

                        let g = MultilinearFolder::new(
                            &scratch.local_point[..main_width],
                            &scratch.next_point[..main_width],
                            boundary,
                            self.public_values,
                            self.alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[main_width..],
                            &scratch.next_point[main_width..],
                        )
                        .eval_air(self.air);
                        scratch.out[acc_idx] += eq_suffix * g;
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
        let width = self.columns.len();
        let main_width = self.main_width;
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
                    RoundScratch::<PackedExt<F, EF::ExtensionPacking>, EF::ExtensionPacking>::new(
                        degree, width,
                    )
                },
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;
                    let eq_suffix = EF::ExtensionPacking::from_ext_slice(eq_suffix);

                    for (((((local, local_diff), next), next_diff), column), next_tail) in scratch
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
                        *local_diff = local_hi - local_lo;

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
                        *next_diff = next_hi - next_lo;
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
                        &scratch.local_point[main_width..],
                        &scratch.next_point[main_width..],
                    )
                    .eval_air(self.air);
                    scratch.out[0] += g.0 * eq_suffix;

                    scratch.add_diffs();
                    boundary += boundary_diff;

                    for acc_idx in 1..scratch.out.len() {
                        scratch.add_diffs();
                        boundary += boundary_diff;

                        let g = MultilinearFolder::new(
                            &scratch.local_point[..main_width],
                            &scratch.next_point[..main_width],
                            boundary,
                            self.public_values,
                            alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[main_width..],
                            &scratch.next_point[main_width..],
                        )
                        .eval_air(self.air);
                        scratch.out[acc_idx] += g.0 * eq_suffix;
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

        // Fold each column's repeat-last tail in place with the value at row `half`.
        // Read that row straight from the current storage, no per-column temporary.
        match &self.columns {
            ExtColumns::Scalar(cols) => {
                for (next_tail, col) in self.next_tail.iter_mut().zip(cols) {
                    let lo = col.as_slice()[half];
                    *next_tail = lo + r * (*next_tail - lo);
                }
            }
            ExtColumns::Packed(cols) => {
                let packing_width = F::Packing::WIDTH;
                let (group, lane) = (half / packing_width, half % packing_width);
                for (next_tail, col) in self.next_tail.iter_mut().zip(cols) {
                    let lo = col.as_slice()[group].extract(lane);
                    *next_tail = lo + r * (*next_tail - lo);
                }
            }
        }

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
