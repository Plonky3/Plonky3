//! Per-round AIR zerocheck state.
//!
//! Builds round polynomials for `sum_x eq(tau, x) * g(x)` and folds state across challenges.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyView};
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use p3_sumcheck::layout::Table;

use crate::folder::MultilinearFolder;
use crate::packed_ext::PackedExt;
use crate::selectors::{BoundaryEvals, periodic_num_variables};

/// Per-variable constraint degree and asserted base-constraint count of one AIR.
///
/// Both values come from the same symbolic pass and are threaded everywhere together,
/// one instance per AIR in a stage.
#[derive(Clone, Copy, Debug)]
pub(super) struct AirShape {
    /// Per-variable constraint degree, with the eq factor stripped.
    pub(super) degree: usize,
    /// Number of base constraints the AIR asserts.
    pub(super) num_constraints: usize,
}

/// One batch of AIRs that share a single trace height.
///
/// A stage activates when the global sumcheck cube shrinks to its height.
/// Every table inside a stage has the same number of variables.
pub(super) struct Stage<'air, 'data, A, F: Field> {
    /// Original caller indices of these AIRs, used to return openings in caller order.
    pub(super) indices: Vec<usize>,
    /// The AIRs in this stage.
    pub(super) airs: Vec<&'air A>,
    /// Public inputs forwarded to each AIR.
    pub(super) public_values: Vec<&'data [F]>,
    /// Optional preprocessed table for each AIR.
    pub(super) preprocessed: Vec<Option<&'data Table<F>>>,
    /// Transposed main trace table for each AIR.
    pub(super) tables: Vec<&'data Table<F>>,
    /// Degree and constraint count of each AIR, in the same order as `airs`.
    pub(super) shapes: Vec<AirShape>,
    /// Shared variable count, equal to the base-two logarithm of the common height.
    pub(super) num_vars: usize,
}

impl<'air, 'data, A, F: Field> Stage<'air, 'data, A, F> {
    /// Build a stage from AIRs that all share one trace height.
    ///
    /// # Panics
    ///
    /// Panics if the tables do not all have the same height.
    /// Panics if a preprocessed table's height differs from the stage height.
    pub(super) fn new(
        airs: &[&'air A],
        public_values: &[&'data [F]],
        indices: &[usize],
        preprocessed: &[Option<&'data Table<F>>],
        tables: &[&'data Table<F>],
        shapes: &[AirShape],
    ) -> Self {
        // Every table in a stage binds the same zerocheck variables, so heights must agree.
        let num_vars = tables
            .iter()
            .map(|table| table.num_variables())
            .all_equal_value()
            .expect("all tables must have the same height");

        // Preprocessed columns fold alongside the main columns, so they share the height.
        assert!(
            preprocessed
                .iter()
                .flatten()
                .all(|table| table.num_variables() == num_vars),
            "preprocessed tables must match the stage height"
        );

        Self {
            num_vars,
            indices: indices.to_vec(),
            airs: airs.to_vec(),
            public_values: public_values.to_vec(),
            preprocessed: preprocessed.to_vec(),
            tables: tables.to_vec(),
            shapes: shapes.to_vec(),
        }
    }
}

/// Sumcheck prover state for the AIR zerocheck.
///
/// Stores already-transposed trace tables.
///
/// Each AIR contributes three column groups, folded identically and laid out in this order:
///
/// ```text
///     main         : committed, opened at the bound point
///     preprocessed : committed, opened at the bound point
///     periodic     : uncommitted, materialized to the full trace height
/// ```
///
/// A materialized periodic column is a genuine multilinear polynomial.
/// It therefore folds exactly like a committed column.
///
/// The verifier recomputes each periodic column in closed form instead of opening it.
pub(crate) struct RoundStateBase<'air, 'data, A, F: Field, EF: ExtensionField<F>> {
    /// AIR whose alpha-batched constraint is being evaluated.
    airs: Vec<&'air A>,
    /// Public inputs forwarded to the AIR.
    public_values: Vec<&'data [F]>,
    /// Random scalar batching the AIR constraints.
    alpha: EF,
    /// Descending alpha powers for each AIR, one entry per constraint that AIR asserts.
    ///
    /// Broadcast across the SIMD lanes so the packed folder can weight a constraint directly.
    /// `None` when this round does not batch: only the packed first round reads these, so the
    /// scalar path leaves it unset rather than building an empty vector per AIR, which would be
    /// indistinguishable from an AIR that genuinely asserts zero constraints.
    alpha_powers: Option<Vec<Vec<EF::ExtensionPacking>>>,
    /// Optional preprocessed tables, one per AIR.
    preprocessed: Vec<Option<&'data Table<F>>>,
    /// Periodic tables, one per AIR, each materialized to the full trace height.
    ///
    /// `None` when the AIR declares no periodic columns.
    /// Owned rather than borrowed: the values come from the AIR, not from committed data.
    periodic: Vec<Option<Table<F>>>,
    /// Main trace tables, one row per original trace column.
    tables: Vec<&'data Table<F>>,
    /// Beta power for each AIR in canonical input order.
    betas: Vec<EF>,
    /// AIRs grouped by their internal round-polynomial degree.
    degree_groups: Vec<DegreeGroup<EF>>,
    /// Per-AIR column spans and degrees, in stage-local order, driving the per-round column fold.
    ///
    /// Derived once from the degree groups, since the layout is fixed for the state's lifetime.
    slots: Vec<AirSlot>,
    /// Zerocheck point coordinates, used to recover the omitted node one for each AIR.
    tau: Point<EF>,
}

/// Extension-round column storage.
///
/// Columns stay SIMD-packed as long as there are enough residual rows to fill a packed lane.
/// Once a fold would leave fewer rows than a lane, columns unpack to scalar form.
enum ExtColumns<F: Field, EF: ExtensionField<F>> {
    /// One SIMD lane per residual row, holding several rows per stored element.
    Packed(Vec<Poly<EF::ExtensionPacking>>),
    /// One extension element per residual row.
    Scalar(Vec<Poly<EF>>),
}

impl<F: Field, EF: ExtensionField<F>> ExtColumns<F, EF> {
    /// Number of stored columns.
    const fn len(&self) -> usize {
        match self {
            Self::Packed(cols) => cols.len(),
            Self::Scalar(cols) => cols.len(),
        }
    }

    /// Number of residual rows across every column.
    ///
    /// A packed column stores one lane group per `F::Packing::WIDTH` rows.
    /// Multiplying the stored-element count by the packing width recovers the scalar row count.
    ///
    /// # Panics
    ///
    /// Panics if there are no columns, since the row count is read from the first column.
    fn num_evals(&self) -> usize {
        match self {
            Self::Packed(cols) => {
                cols.first()
                    .expect("round state requires at least one column")
                    .num_evals()
                    * F::Packing::WIDTH
            }
            Self::Scalar(cols) => cols
                .first()
                .expect("round state requires at least one column")
                .num_evals(),
        }
    }

    /// Borrow the columns as packed lanes.
    ///
    /// # Panics
    ///
    /// Panics if the columns have already unpacked to scalar form.
    /// Callers gate on the same width threshold that decides the storage variant, so this never fires.
    fn as_packed(&self) -> &[Poly<EF::ExtensionPacking>] {
        match self {
            Self::Packed(cols) => cols,
            Self::Scalar(_) => unreachable!("round_poly_packed requires packed columns"),
        }
    }

    /// Borrow the columns as scalar extension elements.
    ///
    /// # Panics
    ///
    /// Panics if the columns are still packed.
    /// Callers gate on the same width threshold that decides the storage variant, so this never fires.
    fn as_scalar(&self) -> &[Poly<EF>] {
        match self {
            Self::Scalar(cols) => cols,
            Self::Packed(_) => unreachable!("round_poly_unpacked requires scalar columns"),
        }
    }

    /// Fold the prefix variable of every column at `r`.
    ///
    /// Stays packed when `want_packed` holds; otherwise unpacks to scalar form in the same pass.
    ///
    /// The residual row count only shrinks round to round.
    /// So a fold can never make packed storage viable again once it stopped being viable.
    ///
    /// # Panics
    ///
    /// Panics if `want_packed` is true while the columns are already scalar.
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

/// Read one packed lane group of consecutive residual rows, starting at scalar row `start`.
///
/// Rows at or past `len` fall back to `tail`, the repeat-last successor value.
///
/// `column` must hold exactly `len` rows, that is `len / F::Packing::WIDTH` groups with no
/// padding lanes. The fast path below relies on it: it decides that a window cannot reach the
/// tail from the group count alone, which is only sound when every stored lane is a real row.
///
/// The stored groups are aligned to multiples of the packing width, so an offset window
/// straddles two adjacent groups.
/// Whenever the window's successor group exists, each basis coefficient of the result is one
/// unaligned read across the two groups' coefficient lanes.
/// The final block has no successor group, so its lanes are gathered one at a time instead.
#[inline]
fn packed_window<F: Field, EF: ExtensionField<F>>(
    column: &[EF::ExtensionPacking],
    start: usize,
    len: usize,
    tail: EF,
) -> EF::ExtensionPacking {
    let packing_width = F::Packing::WIDTH;
    debug_assert_eq!(
        column.len() * packing_width,
        len,
        "packed_window assumes a fully populated column"
    );
    let group = start / packing_width;
    let offset = start % packing_width;

    // With a successor group in range the window stays inside `group` and `group + 1`,
    // so no lane can reach the tail.
    if group + 1 < column.len() {
        return EF::ExtensionPacking::from_basis_coefficients_fn(|d| {
            // The two groups' lanes for this coefficient are contiguous once laid side by side.
            let pair = [
                column[group].as_basis_coefficients_slice()[d],
                column[group + 1].as_basis_coefficients_slice()[d],
            ];
            let flat = F::Packing::unpack_slice(&pair);
            *F::Packing::from_slice(&flat[offset..offset + packing_width])
        });
    }

    EF::ExtensionPacking::from_ext_fn(|lane| {
        // Scalar row this lane maps to inside the window.
        let row = start + lane;
        if row < len {
            // Locate the row's group, then pull its lane out of that group.
            column[row / packing_width].extract(row % packing_width)
        } else {
            // Past the last real row: repeat the tail value.
            tail
        }
    })
}

/// Extension-field sumcheck state after the first base-field round.
///
/// Owns the folded trace columns, boundary selectors, and repeat-last next-row tail values needed
/// by the remaining rounds.
pub(crate) struct RoundStateExt<'air, 'data, A, F: Field, EF: ExtensionField<F>> {
    /// AIR whose alpha-batched constraint is being evaluated.
    airs: Vec<&'air A>,
    /// Public inputs forwarded to the AIR.
    public_values: Vec<&'data [F]>,
    /// Random scalar batching the AIR constraints.
    alpha: EF,
    /// Folded boundary-selector values at the current sumcheck prefix.
    boundary: BoundaryEvals<EF>,
    /// Main and preprocessed columns after the first base-field fold.
    columns: ExtColumns<F, EF>,
    /// Beta power for each AIR in canonical input order.
    betas: Vec<EF>,
    /// AIRs grouped by their internal round-polynomial degree.
    degree_groups: Vec<DegreeGroup<EF>>,
    /// Per-AIR column spans and degrees, in stage-local order, driving the per-round column fold.
    ///
    /// Derived once from the degree groups, since the layout is fixed for the state's lifetime.
    slots: Vec<AirSlot>,
    /// Zerocheck point coordinates, used to recover the omitted node one for each AIR.
    tau: Point<EF>,
    /// Number of already-bound prefix coordinates.
    round: usize,
    /// Repeat-last successor values for each main column at the folded tail row.
    next_tail: Vec<EF>,
}

/// Scratch for scalar round-polynomial folds.
///
/// The base path uses one instance; the extension path allocates one per worker.
struct Scratch<F, EF> {
    /// Unweighted per-node evaluation accumulator for each AIR.
    air_evals: Vec<Vec<EF>>,
    /// Current-row value of each column at the active interpolation node.
    local_point: Vec<F>,
    /// Step added to advance each current-row value to the next node.
    local_diff: Vec<F>,
    /// Successor-row value of each column at the active interpolation node.
    next_point: Vec<F>,
    /// Step added to advance each successor-row value to the next node.
    next_diff: Vec<F>,
}

/// Per-worker scratch for the packed base-field first-round fold.
///
/// Mirrors the scalar scratch with packed row buffers, so each element covers one SIMD lane group.
/// One instance is allocated per worker and reused across that worker's packed blocks.
struct PackedScratch<P, Acc> {
    /// Per-node evaluation accumulator for each AIR, one lane per row of the covered blocks.
    ///
    /// Each lane already carries its own eq weight, so the lanes are summed once at the end.
    air_evals: Vec<Vec<Acc>>,
    /// Current-row lanes of each column at the active interpolation node.
    local_point: Vec<P>,
    /// Step added to advance each current-row lane to the next node.
    local_diff: Vec<P>,
    /// Successor-row lanes of each column at the active interpolation node.
    next_point: Vec<P>,
    /// Step added to advance each successor-row lane to the next node.
    next_diff: Vec<P>,
}

impl<F, EF> Scratch<F, EF>
where
    F: PrimeCharacteristicRing,
    EF: PrimeCharacteristicRing,
{
    fn new(air_degrees: &[usize], width: usize) -> Self {
        Self {
            air_evals: air_degrees.iter().copied().map(EF::zero_vec).collect(),
            local_point: F::zero_vec(width),
            local_diff: F::zero_vec(width),
            next_point: F::zero_vec(width),
            next_diff: F::zero_vec(width),
        }
    }
}

impl<F: Field, EF> Scratch<F, EF> {
    fn add_diffs(&mut self) {
        F::add_slices(&mut self.local_point, &self.local_diff);
        F::add_slices(&mut self.next_point, &self.next_diff);
    }
}

impl<P, Acc> PackedScratch<P, Acc>
where
    P: PrimeCharacteristicRing,
    Acc: PrimeCharacteristicRing,
{
    fn new(air_degrees: &[usize], width: usize) -> Self {
        Self {
            air_evals: air_degrees.iter().copied().map(Acc::zero_vec).collect(),
            local_point: P::zero_vec(width),
            local_diff: P::zero_vec(width),
            next_point: P::zero_vec(width),
            next_diff: P::zero_vec(width),
        }
    }

    fn add_diffs(&mut self)
    where
        P: Copy,
    {
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

    /// Merge another worker's per-node accumulators into this one, lane by lane.
    fn add_air_evals(&mut self, other: Self) {
        debug_assert_eq!(self.air_evals.len(), other.air_evals.len());
        self.air_evals
            .iter_mut()
            .zip(other.air_evals)
            .for_each(|(lhs, rhs)| {
                debug_assert_eq!(lhs.len(), rhs.len());
                lhs.iter_mut()
                    .zip(rhs)
                    .for_each(|(acc, value)| *acc += value);
            });
    }
}

/// Collapse each packed accumulator to the scalar sum of its lanes.
///
/// Every lane already carries the eq weight of the row it stands for, so one horizontal sum
/// per AIR per interpolation node finishes the fold.
fn reduce_air_evals<F: Field, EF: ExtensionField<F>>(
    air_evals: Vec<Vec<EF::ExtensionPacking>>,
) -> Vec<Vec<EF>> {
    air_evals
        .into_iter()
        .map(|evals| {
            evals
                .into_iter()
                .map(|acc| EF::ExtensionPacking::to_ext_iter([acc]).sum())
                .collect()
        })
        .collect()
}

/// Per-AIR fold metadata, in stage-local order.
///
/// Flattens the degree groups so the hot loop drives one AIR at a time.
/// Interpolation nodes above an AIR's own degree are still skipped.
#[derive(Clone, Copy, Default)]
struct AirSlot {
    /// Position of this AIR within its stage, in caller order.
    air_index: usize,
    /// First main column of this AIR inside the merged column buffer.
    main_offset: usize,
    /// Number of main columns this AIR owns.
    main_width: usize,
    /// First preprocessed column of this AIR inside the merged column buffer.
    preprocessed_offset: usize,
    /// Number of preprocessed columns this AIR owns.
    preprocessed_width: usize,
    /// First periodic column of this AIR inside the merged column buffer.
    periodic_offset: usize,
    /// Number of periodic columns this AIR owns.
    periodic_width: usize,
    /// Per-variable degree of this AIR's eq-stripped round polynomial.
    degree: usize,
}

impl AirSlot {
    /// Flatten the degree groups into one slot per AIR, indexed by stage-local position.
    ///
    /// The groups iterate in degree order, but the hot loop wants AIR order.
    /// Each AIR belongs to exactly one group, so every output slot is written exactly once.
    fn flatten<EF>(degree_groups: &[DegreeGroup<EF>]) -> Vec<Self> {
        // The stage-local indices run `0..num_airs`, so the total AIR count sizes the output.
        let num_airs = degree_groups.iter().map(|group| group.airs.len()).sum();
        let mut slots = alloc::vec![Self::default(); num_airs];
        // A group fixes the degree; each AIR in it fixes its own column span.
        for group in degree_groups {
            for air in &group.airs {
                // Scatter to the stage-local position so callers index by AIR, not group.
                slots[air.air_index] = Self {
                    air_index: air.air_index,
                    main_offset: air.main_offset,
                    main_width: air.main_width,
                    preprocessed_offset: air.preprocessed_offset,
                    preprocessed_width: air.preprocessed_width,
                    periodic_offset: air.periodic_offset,
                    periodic_width: air.periodic_width,
                    degree: group.degree,
                };
            }
        }
        slots
    }
}

/// One AIR's column placement inside its degree group.
///
/// The group fixes the degree; this records where the AIR's columns live in the merged buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DegreeGroupAir {
    /// Position of this AIR within its stage, in caller order.
    air_index: usize,
    /// First main column of this AIR inside the merged column buffer.
    main_offset: usize,
    /// Number of main columns this AIR owns.
    main_width: usize,
    /// First preprocessed column of this AIR inside the merged column buffer.
    preprocessed_offset: usize,
    /// Number of preprocessed columns this AIR owns.
    preprocessed_width: usize,
    /// First periodic column of this AIR inside the merged column buffer.
    periodic_offset: usize,
    /// Number of periodic columns this AIR owns.
    periodic_width: usize,
}

/// AIRs sharing one per-variable degree, folded into a single beta-weighted round polynomial.
///
/// Grouping by degree lets a lower-degree AIR skip the interpolation nodes that only a
/// higher-degree AIR needs.
struct DegreeGroup<EF> {
    /// Common per-variable degree of every AIR in this group, with the eq factor stripped.
    degree: usize,
    /// The AIRs in this group, each carrying its own column span.
    airs: Vec<DegreeGroupAir>,
    /// Current reduced sumcheck claim for this group's round polynomial.
    claim: EF,
    /// This round's beta-weighted evaluations at nodes `0, 2, 3, ..., degree`.
    last_evals: Vec<EF>,
    /// Barycentric helper for this group's degree, reused across rounds.
    interpolator: RoundPolyInterpolator<EF>,
}

impl<EF: Field> DegreeGroup<EF> {
    /// Bucket AIRs by degree and assign each AIR its column span in the merged buffer.
    ///
    /// Columns are laid out per AIR as main, then preprocessed, then periodic columns, in caller order:
    ///
    /// ```text
    ///     [ air0 main | air0 preproc | air0 periodic | air1 main | air1 preproc | air1 periodic | ... ]
    /// ```
    ///
    /// # Arguments
    ///
    /// - `degrees`: per-variable constraint degree of each AIR, in stage-local order.
    /// - `main_widths`: main column count of each AIR, in stage-local order.
    /// - `preprocessed_widths`: preprocessed column count of each AIR, in stage-local order.
    /// - `periodic_widths`: periodic column count of each AIR, in stage-local order.
    fn build(
        degrees: &[usize],
        main_widths: &[usize],
        preprocessed_widths: &[usize],
        periodic_widths: &[usize],
    ) -> Vec<Self> {
        // A btree keyed by degree gives deterministic group order across prover and verifier.
        let mut groups = BTreeMap::<usize, Vec<DegreeGroupAir>>::new();
        // Running column cursor into the merged buffer, advanced AIR by AIR.
        let mut column_offset = 0;
        for (air_index, (((&degree, &main_width), &preprocessed_width), &periodic_width)) in degrees
            .iter()
            .zip(main_widths)
            .zip(preprocessed_widths)
            .zip(periodic_widths)
            .enumerate()
        {
            // Main columns come first for this AIR.
            let main_offset = column_offset;
            column_offset += main_width;
            // Preprocessed columns follow immediately after.
            let preprocessed_offset = column_offset;
            column_offset += preprocessed_width;
            // Periodic columns come last for this AIR.
            let periodic_offset = column_offset;
            column_offset += periodic_width;
            // File the AIR under its degree, keeping its column span.
            groups.entry(degree).or_default().push(DegreeGroupAir {
                air_index,
                main_offset,
                main_width,
                preprocessed_offset,
                preprocessed_width,
                periodic_offset,
                periodic_width,
            });
        }

        // Each degree becomes one group with a zero starting claim and a prebuilt interpolator.
        groups
            .into_iter()
            .map(|(degree, airs)| Self {
                degree,
                airs,
                claim: EF::ZERO,
                last_evals: EF::zero_vec(degree),
                interpolator: RoundPolyInterpolator::new(degree),
            })
            .collect()
    }

    /// Evaluate this group's eq-stripped round polynomial `q` at an interpolation node.
    ///
    /// The prover never stores `q(1)`; it is recovered from the sumcheck claim relation:
    ///
    /// ```text
    ///     claim = (1 - tau) * q(0) + tau * q(1)
    ///  => q(1)  = (claim - (1 - tau) * q(0)) / tau
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the group degree is zero, since a constant has no round polynomial.
    fn eval(&self, tau: EF, point: EF) -> EF {
        debug_assert_eq!(self.last_evals.len(), self.degree);
        assert!(self.degree > 0, "round polynomial degree must be positive");
        // Node 0 is stored directly as the first evaluation.
        if point.is_zero() {
            return self.last_evals[0];
        }

        // Recover the dropped node 1 from the inter-round claim relation.
        let q1 = (self.claim - (EF::ONE - tau) * self.last_evals[0]) * tau.inverse();
        if point == EF::ONE {
            return q1;
        }

        // Any higher node is a barycentric extrapolation of the stored evaluations.
        self.interpolator
            .eval(&self.last_evals, self.last_evals[0] + q1, point)
    }

    /// Add this group's per-node evaluations into the stage's shared accumulator.
    ///
    /// The accumulator carries the stage's max degree, so a lower-degree group is
    /// extrapolated up to the missing top nodes.
    ///
    /// ```text
    ///     out index : 0    1    2    3    ...
    ///     node      : 0    2    3    4    ...   (node 1 is never stored)
    /// ```
    fn combine_evals(&self, out: &mut [EF], tau: EF) {
        for (idx, acc) in out.iter_mut().enumerate() {
            // Map the dense output index onto the sparse node set {0, 2, 3, ...}.
            let node = if idx == 0 { 0 } else { idx + 1 };
            let value = if node == 0 || node <= self.degree {
                // Within this group's own degree: read the stored evaluation directly.
                let index = if node == 0 { 0 } else { node - 1 };
                self.last_evals[index]
            } else {
                // Above this group's degree: extrapolate to the stage's top nodes.
                self.eval(tau, EF::interpolation_node(node))
            };
            *acc += value;
        }
    }

    /// Beta-weight each AIR's per-node sums and store them as this round's evaluations.
    ///
    /// The per-row fold accumulates `sum_x eq(x) * g_air(x)` with no beta factor.
    /// Beta enters here, once per AIR per node, since the sum is linear:
    ///
    /// ```text
    ///     sum_x beta * eq(x) * g(x) = beta * sum_x eq(x) * g(x)
    /// ```
    ///
    /// This keeps beta out of the row loop.
    /// It saves one extension multiply per node per row.
    /// That multiply dominates the fold for cheap AIRs.
    fn write_last_evals(&mut self, betas: &[EF], air_evals: &[Vec<EF>]) {
        // One accumulator per interpolation node this group carries.
        let mut last = EF::zero_vec(self.degree);
        for air in &self.airs {
            // beta^i for this AIR, applied once here rather than per row.
            let beta = betas[air.air_index];
            // Fold the AIR's unweighted per-node sums in, scaled by beta.
            last.iter_mut()
                .zip(air_evals[air.air_index].iter())
                .for_each(|(acc, &value)| *acc += beta * value);
        }
        self.last_evals = last;
    }
}

impl<'air, 'data, A, F, EF> RoundStateBase<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Build the prover state when different AIRs have different internal degrees.
    ///
    /// `degrees[i]` is the degree of AIR `i` after stripping the active eq factor.
    /// Every entry must be positive. The round degree is the maximum AIR degree.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn new(
        stage: &Stage<'air, 'data, A, F>,
        alpha: EF,
        betas: Vec<EF>,
        tau: &Point<EF>,
    ) -> Self {
        assert!(!stage.tables.is_empty(),);
        assert_eq!(stage.airs.len(), stage.tables.len(),);
        assert_eq!(stage.preprocessed.len(), stage.tables.len(),);
        assert_eq!(stage.public_values.len(), stage.tables.len(),);
        assert_eq!(stage.shapes.len(), stage.tables.len(),);
        assert!(stage.shapes.iter().all(|shape| shape.degree > 0),);

        let num_vars = stage
            .tables
            .iter()
            .map(|table| table.num_variables())
            .all_equal_value()
            .expect("all tables in a RoundStateBase must have the same height");
        assert!(
            stage
                .preprocessed
                .iter()
                .flatten()
                .all(|table| table.num_variables() == num_vars),
            "preprocessed tables must match the main trace height"
        );

        let main_widths = stage
            .tables
            .iter()
            .map(|table| table.num_polys())
            .collect::<Vec<_>>();
        let preprocessed_widths = stage
            .preprocessed
            .iter()
            .map(|table| table.map_or(0, Table::num_polys))
            .collect::<Vec<_>>();

        // Materialize each AIR's periodic columns to the full trace height.
        //
        //     period vector  : [v_0, v_1]
        //     trace height 8 : [v_0, v_1, v_0, v_1, v_0, v_1, v_0, v_1]
        //
        // The full-height column is a genuine multilinear polynomial.
        // It therefore folds through the sumcheck exactly like a committed column.
        let trace_height = 1 << num_vars;
        let periodic = stage
            .airs
            .iter()
            .map(|air| {
                let cols = air.periodic_columns();
                if cols.is_empty() {
                    return None;
                }

                // Reject a declaration the trace cannot hold, matching the verifier's own check.
                let num_variables =
                    periodic_num_variables(air.num_periodic_columns(), &cols, num_vars)
                        .expect("periodic column declaration must fit the trace height");

                let mut values = Vec::with_capacity(cols.len() * trace_height);
                for (col, j) in cols.iter().zip(num_variables) {
                    // Copy the whole period vector once per cycle it spans.
                    for _ in 0..trace_height >> j {
                        values.extend_from_slice(col);
                    }
                }
                Some(Table::new(RowMajorMatrix::new(values, trace_height)))
            })
            .collect::<Vec<_>>();
        let periodic_widths = periodic
            .iter()
            .map(|table| table.as_ref().map_or(0, Table::num_polys))
            .collect::<Vec<_>>();

        let num_airs = stage.airs.len();
        assert_eq!(
            betas.len(),
            num_airs,
            "one beta power is required for each AIR"
        );

        for (((air, public_values), &main_width), &preprocessed_width) in stage
            .airs
            .iter()
            .zip(stage.public_values.iter())
            .zip(main_widths.iter())
            .zip(preprocessed_widths.iter())
        {
            assert_eq!(main_width, air.width(), "trace width must match AIR width");
            assert_eq!(
                preprocessed_width,
                air.preprocessed_width(),
                "preprocessed width must match AIR preprocessed width"
            );
            assert_eq!(public_values.len(), air.num_public_values());
        }

        // Constraint `i` of an AIR asserting `n` constraints is batched with weight `alpha^(n-1-i)`,
        // the same weight the folder's Horner fold assigns it.
        //
        // Only the packed first round reads these, so the scalar path leaves this `None`.
        let alpha_powers = ((1 << num_vars) / 2 >= F::Packing::WIDTH).then(|| {
            stage
                .shapes
                .iter()
                .map(|shape| {
                    let mut powers = alpha.powers().collect_n(shape.num_constraints);
                    powers.reverse();
                    powers
                        .into_iter()
                        .map(EF::ExtensionPacking::from)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });

        // Build the degree grouping once, then flatten it into the AIR-ordered fold view.
        let degrees = stage
            .shapes
            .iter()
            .map(|shape| shape.degree)
            .collect::<Vec<_>>();
        let degree_groups = DegreeGroup::build(
            &degrees,
            &main_widths,
            &preprocessed_widths,
            &periodic_widths,
        );
        let slots = AirSlot::flatten(&degree_groups);

        Self {
            airs: stage.airs.clone(),
            public_values: stage.public_values.clone(),
            alpha,
            alpha_powers,
            preprocessed: stage.preprocessed.clone(),
            periodic,
            tables: stage.tables.clone(),
            betas,
            degree_groups,
            slots,
            tau: tau.clone(),
        }
    }

    fn num_evals(&self) -> usize {
        1 << self
            .tables
            .first()
            .expect("round state requires at least one table")
            .num_variables()
    }

    fn total_width(&self) -> usize {
        self.tables
            .iter()
            .map(|table| table.num_polys())
            .sum::<usize>()
            + self
                .preprocessed
                .iter()
                .map(|table| table.map_or(0, Table::num_polys))
                .sum::<usize>()
            + self
                .periodic
                .iter()
                .map(|table| table.as_ref().map_or(0, Table::num_polys))
                .sum::<usize>()
    }

    fn degree(&self) -> usize {
        self.degree_groups
            .iter()
            .map(|group| group.degree)
            .max()
            .unwrap()
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        if self.num_evals() / 2 < F::Packing::WIDTH {
            self.round_poly_unpacked(eq_suffix)
        } else {
            self.round_poly_packed(eq_suffix)
        }
    }

    #[tracing::instrument(skip_all, level = "debug")]
    fn round_poly_packed(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        let width = self.total_width();
        let height = self.num_evals();
        let scalar_half = height / 2;
        let packing_width = F::Packing::WIDTH;
        let packed_half = scalar_half / packing_width;
        let degree = self.degree();
        let alpha = EF::ExtensionPacking::from(self.alpha);
        let air_degrees = self
            .slots
            .iter()
            .map(|slot| slot.degree)
            .collect::<Vec<_>>();
        assert_ne!(packed_half, 0);

        // A block's selector pair depends on the block only through the residual rows `0` and
        // `height - 1`. Row `0` sits in the low half of block `0`; row `height - 1` sits in the
        // high half of block `packed_half - 1`. Every other block reads the same constant pair,
        // so only these two need the per-row construction.
        let first_block = BoundaryEvals::<F::Packing>::row_pair_packed(0, scalar_half, height);
        let last_block = BoundaryEvals::<F::Packing>::row_pair_packed(
            (packed_half - 1) * packing_width,
            scalar_half,
            height,
        );
        let interior_block = (
            BoundaryEvals::new(F::Packing::ZERO, F::Packing::ZERO, F::Packing::ONE),
            BoundaryEvals::new(F::Packing::ZERO, F::Packing::ZERO, F::Packing::ZERO),
        );

        let air_evals = eq_suffix
            .as_slice()
            .par_chunks_exact(packing_width)
            .enumerate()
            .par_fold_reduce(
                || PackedScratch::new(&air_degrees, width),
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;
                    let eq_suffix = EF::ExtensionPacking::from_ext_slice(eq_suffix);

                    let fill_columns =
                        |scratch: &mut PackedScratch<F::Packing, EF::ExtensionPacking>,
                         offset: usize,
                         table: &Table<F>| {
                            let end = offset + table.num_polys();
                            for ((((local, local_delta), next), next_delta), column) in scratch
                                .local_point[offset..end]
                                .iter_mut()
                                .zip(scratch.local_diff[offset..end].iter_mut())
                                .zip(scratch.next_point[offset..end].iter_mut())
                                .zip(scratch.next_diff[offset..end].iter_mut())
                                .zip(table.iter_polys())
                            {
                                let local_lo =
                                    *F::Packing::from_slice(&column[s..s + packing_width]);
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
                        };
                    for slot in &self.slots {
                        fill_columns(&mut scratch, slot.main_offset, self.tables[slot.air_index]);
                        if let Some(preprocessed) = self.preprocessed[slot.air_index] {
                            fill_columns(&mut scratch, slot.preprocessed_offset, preprocessed);
                        }
                        if let Some(periodic) = self.periodic[slot.air_index].as_ref() {
                            fill_columns(&mut scratch, slot.periodic_offset, periodic);
                        }
                    }

                    let (mut boundary, boundary_diff) = if packed_s == 0 {
                        first_block
                    } else if packed_s == packed_half - 1 {
                        last_block
                    } else {
                        interior_block
                    };

                    // Node 0 is skipped.
                    // Invariant: a stage's first round runs on an unfolded trace.
                    //     X = 0 and X = 1 are then real boolean rows.
                    //     A satisfying trace makes g vanish on every row.
                    //     So q(0) = q(1) = 0.
                    // The node-0 accumulator stays zero.
                    // The sweep below starts at node 2.
                    scratch.add_diffs();
                    boundary += boundary_diff;

                    for node in 2..=degree {
                        scratch.add_diffs();
                        boundary += boundary_diff;

                        self.slots
                            .iter()
                            .zip(scratch.air_evals.iter_mut())
                            .for_each(|(slot, evals)| {
                                if node <= slot.degree {
                                    let g = MultilinearFolder::new(
                                        &scratch.local_point
                                            [slot.main_offset..slot.main_offset + slot.main_width],
                                        &scratch.next_point
                                            [slot.main_offset..slot.main_offset + slot.main_width],
                                        boundary,
                                        self.public_values[slot.air_index],
                                        alpha,
                                    )
                                    .with_preprocessed(
                                        &scratch.local_point[slot.preprocessed_offset
                                            ..slot.preprocessed_offset + slot.preprocessed_width],
                                        &scratch.next_point[slot.preprocessed_offset
                                            ..slot.preprocessed_offset + slot.preprocessed_width],
                                    )
                                    .with_periodic(
                                        &scratch.local_point[slot.periodic_offset
                                            ..slot.periodic_offset + slot.periodic_width],
                                    );
                                    let g = match &self.alpha_powers {
                                        Some(alpha_powers) => {
                                            g.with_alpha_powers(&alpha_powers[slot.air_index])
                                        }
                                        None => g,
                                    };
                                    let g = g.eval_air(self.airs[slot.air_index]);
                                    evals[node - 1] += eq_suffix * g;
                                }
                            });
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.add_air_evals(rhs);
                    lhs
                },
            )
            .air_evals;
        let air_evals = reduce_air_evals::<F, EF>(air_evals);
        let mut out = EF::zero_vec(degree);
        let tau = self.tau.as_slice()[0];
        for group in &mut self.degree_groups {
            group.write_last_evals(&self.betas, &air_evals);
        }
        self.degree_groups
            .iter()
            .for_each(|group| group.combine_evals(&mut out, tau));
        out
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_unpacked(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>,
    {
        let width = self.total_width();
        let height = self.num_evals();
        let half = height / 2;
        let degree = self.degree();

        let air_degrees = self
            .slots
            .iter()
            .map(|slot| slot.degree)
            .collect::<Vec<_>>();
        let mut scratch = Scratch::<F, EF>::new(&air_degrees, width);

        for (s, &eq_suffix) in eq_suffix.as_slice().iter().enumerate() {
            let fill_columns = |scratch: &mut Scratch<F, EF>, offset: usize, table: &Table<F>| {
                let end = offset + table.num_polys();
                scratch.local_point[offset..end]
                    .iter_mut()
                    .zip(scratch.local_diff[offset..end].iter_mut())
                    .zip(scratch.next_point[offset..end].iter_mut())
                    .zip(scratch.next_diff[offset..end].iter_mut())
                    .zip(table.iter_polys())
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
            };
            for slot in &self.slots {
                fill_columns(&mut scratch, slot.main_offset, self.tables[slot.air_index]);
                if let Some(preprocessed) = self.preprocessed[slot.air_index] {
                    fill_columns(&mut scratch, slot.preprocessed_offset, preprocessed);
                }
                if let Some(periodic) = self.periodic[slot.air_index].as_ref() {
                    fill_columns(&mut scratch, slot.periodic_offset, periodic);
                }
            }

            let (mut boundary, boundary_diff) = BoundaryEvals::<F>::row_pair(s, half, height);

            // Node 0 is skipped.
            // Invariant: a stage's first round runs on an unfolded trace.
            //     X = 0 and X = 1 are then real boolean rows.
            //     A satisfying trace makes g vanish on every row.
            //     So q(0) = q(1) = 0.
            // The node-0 accumulator stays zero.
            // The sweep below starts at node 2.
            scratch.add_diffs();
            boundary += boundary_diff;

            for node in 2..=degree {
                scratch.add_diffs();
                boundary += boundary_diff;

                self.slots
                    .iter()
                    .zip(scratch.air_evals.iter_mut())
                    .for_each(|(slot, evals)| {
                        if node <= slot.degree {
                            let g = MultilinearFolder::new(
                                &scratch.local_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                &scratch.next_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                boundary,
                                self.public_values[slot.air_index],
                                self.alpha,
                            )
                            .with_preprocessed(
                                &scratch.local_point[slot.preprocessed_offset
                                    ..slot.preprocessed_offset + slot.preprocessed_width],
                                &scratch.next_point[slot.preprocessed_offset
                                    ..slot.preprocessed_offset + slot.preprocessed_width],
                            )
                            .with_periodic(
                                &scratch.local_point[slot.periodic_offset
                                    ..slot.periodic_offset + slot.periodic_width],
                            )
                            .eval_air(self.airs[slot.air_index]);
                            evals[node - 1] += eq_suffix * g;
                        }
                    });
            }
        }

        let mut out = EF::zero_vec(degree);
        let tau = self.tau.as_slice()[0];
        for group in &mut self.degree_groups {
            group.write_last_evals(&self.betas, &scratch.air_evals);
        }
        self.degree_groups
            .iter()
            .for_each(|group| group.combine_evals(&mut out, tau));
        out
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold(mut self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>,
    {
        let tau = self.tau.as_slice()[0];
        self.degree_groups
            .iter_mut()
            .for_each(|group| group.claim = group.eval(tau, r));

        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let width = self.total_width();
        let mut next_tail = Vec::with_capacity(width);
        for slot in &self.slots {
            next_tail.extend(
                self.tables[slot.air_index]
                    .iter_polys()
                    .map(|col| r * (col[num_evals - 1] - col[half]) + col[half]),
            );
            if let Some(preprocessed) = self.preprocessed[slot.air_index] {
                next_tail.extend(
                    preprocessed
                        .iter_polys()
                        .map(|col| r * (col[num_evals - 1] - col[half]) + col[half]),
                );
            }
            if let Some(periodic) = self.periodic[slot.air_index].as_ref() {
                next_tail.extend(
                    periodic
                        .iter_polys()
                        .map(|col| r * (col[num_evals - 1] - col[half]) + col[half]),
                );
            }
        }

        let want_packed = (half / 2) >= F::Packing::WIDTH;
        let columns = if want_packed {
            let mut columns = Vec::with_capacity(width);
            for slot in &self.slots {
                columns.extend(
                    self.tables[slot.air_index]
                        .par_iter_polys()
                        .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                        .collect::<Vec<_>>(),
                );
                if let Some(preprocessed) = self.preprocessed[slot.air_index] {
                    columns.extend(
                        preprocessed
                            .par_iter_polys()
                            .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                            .collect::<Vec<_>>(),
                    );
                }
                if let Some(periodic) = self.periodic[slot.air_index].as_ref() {
                    columns.extend(
                        periodic
                            .par_iter_polys()
                            .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                            .collect::<Vec<_>>(),
                    );
                }
            }
            ExtColumns::Packed(columns)
        } else {
            let mut columns = Vec::with_capacity(width);
            for slot in &self.slots {
                columns.extend(
                    self.tables[slot.air_index]
                        .par_iter_polys()
                        .map(|col| PolyView::new(col).fix_prefix_var(r))
                        .collect::<Vec<_>>(),
                );
                if let Some(preprocessed) = self.preprocessed[slot.air_index] {
                    columns.extend(
                        preprocessed
                            .par_iter_polys()
                            .map(|col| PolyView::new(col).fix_prefix_var(r))
                            .collect::<Vec<_>>(),
                    );
                }
                if let Some(periodic) = self.periodic[slot.air_index].as_ref() {
                    columns.extend(
                        periodic
                            .par_iter_polys()
                            .map(|col| PolyView::new(col).fix_prefix_var(r))
                            .collect::<Vec<_>>(),
                    );
                }
            }
            ExtColumns::Scalar(columns)
        };

        RoundStateExt {
            airs: self.airs,
            public_values: self.public_values,
            alpha: self.alpha,
            betas: self.betas,
            degree_groups: self.degree_groups,
            slots: self.slots,
            tau: self.tau,
            round: 1,
            columns,
            next_tail,
            boundary: BoundaryEvals::new(EF::ONE - r, r, EF::ONE - r),
        }
    }
}

impl<'air, 'data, A, F, EF> RoundStateExt<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn num_evals(&self) -> usize {
        self.columns.num_evals()
    }

    const fn width(&self) -> usize {
        self.columns.len()
    }

    fn degree(&self) -> usize {
        self.degree_groups
            .iter()
            .map(|group| group.degree)
            .max()
            .unwrap()
    }

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
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
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
            self.round_poly_unpacked(eq_suffix)
        } else {
            self.round_poly_packed(eq_suffix)
        }
    }

    #[tracing::instrument(skip_all, level = "debug")]
    fn round_poly_unpacked(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>>,
    {
        let width = self.width();
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let degree = self.degree();
        let air_degrees = self
            .slots
            .iter()
            .map(|slot| slot.degree)
            .collect::<Vec<_>>();

        let air_evals = eq_suffix
            .as_slice()
            .par_iter()
            .enumerate()
            .par_fold_reduce(
                || Scratch::<EF, EF>::new(&air_degrees, width),
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

                    self.slots
                        .iter()
                        .zip(scratch.air_evals.iter_mut())
                        .for_each(|(slot, evals)| {
                            let g = MultilinearFolder::new(
                                &scratch.local_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                &scratch.next_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                boundary,
                                self.public_values[slot.air_index],
                                self.alpha,
                            )
                            .with_preprocessed(
                                &scratch.local_point[slot.preprocessed_offset
                                    ..slot.preprocessed_offset + slot.preprocessed_width],
                                &scratch.next_point[slot.preprocessed_offset
                                    ..slot.preprocessed_offset + slot.preprocessed_width],
                            )
                            .with_periodic(
                                &scratch.local_point[slot.periodic_offset
                                    ..slot.periodic_offset + slot.periodic_width],
                            )
                            .eval_air(self.airs[slot.air_index]);
                            evals[0] += eq_suffix * g;
                            debug_assert!(slot.degree > 0);
                        });

                    scratch.add_diffs();
                    boundary += boundary_diff;

                    for node in 2..=degree {
                        scratch.add_diffs();
                        boundary += boundary_diff;

                        self.slots
                            .iter()
                            .zip(scratch.air_evals.iter_mut())
                            .for_each(|(slot, evals)| {
                                if node <= slot.degree {
                                    let g = MultilinearFolder::new(
                                        &scratch.local_point
                                            [slot.main_offset..slot.main_offset + slot.main_width],
                                        &scratch.next_point
                                            [slot.main_offset..slot.main_offset + slot.main_width],
                                        boundary,
                                        self.public_values[slot.air_index],
                                        self.alpha,
                                    )
                                    .with_preprocessed(
                                        &scratch.local_point[slot.preprocessed_offset
                                            ..slot.preprocessed_offset + slot.preprocessed_width],
                                        &scratch.next_point[slot.preprocessed_offset
                                            ..slot.preprocessed_offset + slot.preprocessed_width],
                                    )
                                    .with_periodic(
                                        &scratch.local_point[slot.periodic_offset
                                            ..slot.periodic_offset + slot.periodic_width],
                                    )
                                    .eval_air(self.airs[slot.air_index]);
                                    evals[node - 1] += eq_suffix * g;
                                }
                            });
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.air_evals
                        .iter_mut()
                        .zip(rhs.air_evals)
                        .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                    lhs
                },
            )
            .air_evals;
        let mut out = EF::zero_vec(degree);
        let tau = self.tau.as_slice()[self.round];
        for group in &mut self.degree_groups {
            group.write_last_evals(&self.betas, &air_evals);
        }
        self.degree_groups
            .iter()
            .for_each(|group| group.combine_evals(&mut out, tau));

        out
    }

    /// SIMD-packed twin of [`Self::round_poly_unpacked`].
    ///
    /// Every column is already extension-valued (folded by earlier rounds),
    /// so one stored `EF::ExtensionPacking` already groups `F::Packing::WIDTH`
    /// consecutive residual rows. The current-row window is such a group read
    /// directly; the successor window sits one row later and is assembled by
    /// [`packed_window`]. The AIR is driven through [`PackedExt`], which wraps
    /// the packed extension value so it supports arithmetic against the base
    /// field `F` directly (see that type's docs for why this is needed).
    #[tracing::instrument(skip_all, level = "debug")]
    fn round_poly_packed(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
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
        let degree = self.degree();
        let alpha = PackedExt::new(EF::ExtensionPacking::from(self.alpha));
        let air_degrees = self
            .slots
            .iter()
            .map(|slot| slot.degree)
            .collect::<Vec<_>>();
        assert_ne!(packed_half, 0);

        // `packed_window` assumes every stored column holds exactly `height` rows with no
        // padding lane. Checking that once per round here, instead of once per `(block, column)`
        // read inside the fold, keeps the cost independent of the block count.
        assert!(
            self.columns
                .as_packed()
                .iter()
                .all(|column| column.as_slice().len() * packing_width == height),
            "packed_window assumes a fully populated column"
        );

        // A block's selector pair depends on the block only through the residual rows `0` and
        // `height - 1`. Row `0` sits in the low half of block `0`; row `height - 1` sits in the
        // high half of block `packed_half - 1`. Every other block reads the same constant pair,
        // so only these two need the per-row construction.
        let first_block =
            BoundaryEvals::row_pair_with_prefix_packed::<F>(0, scalar_half, height, self.boundary);
        let last_block = BoundaryEvals::row_pair_with_prefix_packed::<F>(
            (packed_half - 1) * packing_width,
            scalar_half,
            height,
            self.boundary,
        );
        let interior_block = (
            BoundaryEvals::new(
                EF::ExtensionPacking::ZERO,
                EF::ExtensionPacking::ZERO,
                EF::ExtensionPacking::ONE,
            ),
            BoundaryEvals::new(
                EF::ExtensionPacking::ZERO,
                EF::ExtensionPacking::ZERO,
                EF::ExtensionPacking::ZERO,
            ),
        );

        let air_evals = eq_suffix
            .as_slice()
            .par_chunks_exact(packing_width)
            .enumerate()
            .par_fold_reduce(
                || {
                    PackedScratch::<PackedExt<F, EF::ExtensionPacking>, EF::ExtensionPacking>::new(
                        &air_degrees,
                        width,
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
                        let local_lo = PackedExt::new(column[packed_s]);
                        let local_hi = PackedExt::new(column[packed_s + packed_half]);
                        *local = local_lo;
                        *local_delta = local_hi - local_lo;

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

                    let (raw_boundary, raw_boundary_diff) = if packed_s == 0 {
                        first_block
                    } else if packed_s == packed_half - 1 {
                        last_block
                    } else {
                        interior_block
                    };
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

                    self.slots
                        .iter()
                        .zip(scratch.air_evals.iter_mut())
                        .for_each(|(slot, evals)| {
                            let g = MultilinearFolder::new(
                                &scratch.local_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                &scratch.next_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                boundary,
                                self.public_values[slot.air_index],
                                alpha,
                            )
                            .with_preprocessed(
                                &scratch.local_point[slot.preprocessed_offset
                                    ..slot.preprocessed_offset + slot.preprocessed_width],
                                &scratch.next_point[slot.preprocessed_offset
                                    ..slot.preprocessed_offset + slot.preprocessed_width],
                            )
                            .with_periodic(
                                &scratch.local_point[slot.periodic_offset
                                    ..slot.periodic_offset + slot.periodic_width],
                            )
                            .eval_air(self.airs[slot.air_index]);
                            evals[0] += eq_suffix * g.0;
                            debug_assert!(slot.degree > 0);
                        });

                    scratch.add_diffs();
                    boundary += boundary_diff;

                    for node in 2..=degree {
                        scratch.add_diffs();
                        boundary += boundary_diff;

                        self.slots
                            .iter()
                            .zip(scratch.air_evals.iter_mut())
                            .for_each(|(slot, evals)| {
                                if node <= slot.degree {
                                    let g = MultilinearFolder::new(
                                        &scratch.local_point
                                            [slot.main_offset..slot.main_offset + slot.main_width],
                                        &scratch.next_point
                                            [slot.main_offset..slot.main_offset + slot.main_width],
                                        boundary,
                                        self.public_values[slot.air_index],
                                        alpha,
                                    )
                                    .with_preprocessed(
                                        &scratch.local_point[slot.preprocessed_offset
                                            ..slot.preprocessed_offset + slot.preprocessed_width],
                                        &scratch.next_point[slot.preprocessed_offset
                                            ..slot.preprocessed_offset + slot.preprocessed_width],
                                    )
                                    .with_periodic(
                                        &scratch.local_point[slot.periodic_offset
                                            ..slot.periodic_offset + slot.periodic_width],
                                    )
                                    .eval_air(self.airs[slot.air_index]);
                                    evals[node - 1] += eq_suffix * g.0;
                                }
                            });
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.add_air_evals(rhs);
                    lhs
                },
            )
            .air_evals;
        let air_evals = reduce_air_evals::<F, EF>(air_evals);
        let mut out = EF::zero_vec(degree);
        let tau = self.tau.as_slice()[self.round];
        for group in &mut self.degree_groups {
            group.write_last_evals(&self.betas, &air_evals);
        }
        self.degree_groups
            .iter()
            .for_each(|group| group.combine_evals(&mut out, tau));
        out
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold(&mut self, r: EF)
    where
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>>,
    {
        let tau = self.tau.as_slice()[self.round];
        self.degree_groups
            .iter_mut()
            .for_each(|group| group.claim = group.eval(tau, r));

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

        let want_packed = (half / 2) >= F::Packing::WIDTH;
        self.columns = core::mem::replace(&mut self.columns, ExtColumns::Scalar(Vec::new()))
            .fold(r, want_packed);

        self.boundary.apply(r);
        self.round += 1;
    }
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;

    use p3_air::{AirBuilder, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Packing = <EF as ExtensionField<F>>::ExtensionPacking;

    #[test]
    fn packed_window_matches_the_scalar_gather() {
        // Invariant: lane `l` of the window starting at `start` carries scalar row
        // `start + l`, or the repeat-last tail once that row runs past the column.
        //
        // Fixture state: columns of 1 through 8 packed groups, each exercised at every
        // start offset. `num_groups == 1` and `== 2` pin the case where every window falls
        // in the final-block gather branch (no successor group ever exists); `num_groups >= 3`
        // reaches the two-group unaligned-read branch as well.
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        let packing_width = <F as Field>::Packing::WIDTH;

        for num_groups in 1..=8 {
            let len = num_groups * packing_width;

            let rows = (0..len).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
            let tail: EF = rng.random();
            let column = rows
                .chunks_exact(packing_width)
                .map(<Packing as PackedFieldExtension<F, EF>>::from_ext_slice)
                .collect::<Vec<_>>();

            for start in 0..len {
                let window = packed_window::<F, EF>(&column, start, len, tail);
                for lane in 0..packing_width {
                    let row = start + lane;
                    let expected = if row < len { rows[row] } else { tail };
                    assert_eq!(
                        <Packing as PackedFieldExtension<F, EF>>::extract(&window, lane),
                        expected,
                        "num_groups={num_groups} start={start} lane={lane}"
                    );
                }
            }
        }
    }

    /// Width-3 AIR with a degree-1 first-row constraint and a degree-2 transition product.
    ///
    /// No preprocessed or periodic columns.
    struct QuadAir;

    impl BaseAir<F> for QuadAir {
        fn width(&self) -> usize {
            3
        }
        fn num_public_values(&self) -> usize {
            1
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for QuadAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            let (local0, local1, next2) = (local[0], local[1], next[2]);
            let pi0 = builder.public_values()[0];

            builder.when_first_row().assert_eq(local0, pi0);
            builder.when_transition().assert_eq(local0 * local1, next2);
        }
    }

    /// Single period-2 periodic column shared by [`CubicMixedAir`].
    fn cubic_mixed_periodic_columns() -> Vec<Vec<F>> {
        [[F::ONE, F::TWO].to_vec()].to_vec()
    }

    /// Width-2 AIR with one preprocessed column, one period-2 periodic column, and a
    /// degree-3 transition product.
    struct CubicMixedAir;

    impl BaseAir<F> for CubicMixedAir {
        fn width(&self) -> usize {
            2
        }
        fn preprocessed_width(&self) -> usize {
            1
        }
        fn num_periodic_columns(&self) -> usize {
            cubic_mixed_periodic_columns().len()
        }
        fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
            Cow::Owned(cubic_mixed_periodic_columns())
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for CubicMixedAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            let (local0, local1, next0) = (local[0], local[1], next[0]);
            let preprocessed0 = builder.preprocessed().current_slice()[0];
            let periodic0 = builder.periodic_values()[0];

            builder
                .when_transition()
                .assert_eq(local0 * local1 * preprocessed0, next0);
            builder.assert_eq(local1, periodic0);
        }
    }

    /// Unifies [`QuadAir`] and [`CubicMixedAir`] into one type, since a [`Stage`] batches AIRs
    /// of a single concrete type.
    enum EquivalenceAir {
        Quad(QuadAir),
        CubicMixed(CubicMixedAir),
    }

    impl BaseAir<F> for EquivalenceAir {
        fn width(&self) -> usize {
            match self {
                Self::Quad(air) => air.width(),
                Self::CubicMixed(air) => air.width(),
            }
        }
        fn preprocessed_width(&self) -> usize {
            match self {
                Self::Quad(air) => air.preprocessed_width(),
                Self::CubicMixed(air) => air.preprocessed_width(),
            }
        }
        fn num_public_values(&self) -> usize {
            match self {
                Self::Quad(air) => air.num_public_values(),
                Self::CubicMixed(air) => air.num_public_values(),
            }
        }
        fn num_periodic_columns(&self) -> usize {
            match self {
                Self::Quad(air) => air.num_periodic_columns(),
                Self::CubicMixed(air) => air.num_periodic_columns(),
            }
        }
        fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
            match self {
                Self::Quad(air) => air.periodic_columns(),
                Self::CubicMixed(air) => air.periodic_columns(),
            }
        }
    }

    impl<AB: AirBuilder<F = F>> Air<AB> for EquivalenceAir {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::Quad(air) => air.eval(builder),
                Self::CubicMixed(air) => air.eval(builder),
            }
        }
    }

    /// A `height x width` table of independent random field elements.
    fn random_table(rng: &mut SmallRng, height: usize, width: usize) -> Table<F> {
        let values = (0..height * width)
            .map(|_| rng.random())
            .collect::<Vec<_>>();
        Table::new(RowMajorMatrix::new(values, width).transpose())
    }

    #[test]
    fn round_poly_packed_matches_unpacked_base_round() {
        // Property: for the same base-round state, `round_poly_packed` and
        // `round_poly_unpacked` compute the same eq-weighted per-node evaluations.
        //
        // The state batches two AIRs of differing degree and width, one with a
        // preprocessed column and a periodic column and one with neither.
        //
        // `num_vars` sweeps `log2(2W) ..= log2(2W) + 3`, so `packed_half` covers every block
        // shape the packed path can see: a single combined first/last block, two blocks with
        // no interior, and four or eight blocks with at least one interior block.
        let mut rng = SmallRng::seed_from_u64(0xEA915E);
        let packing_width = <F as Field>::Packing::WIDTH;
        let base_num_vars = packing_width.trailing_zeros() as usize + 1;

        for extra in 0..4 {
            let num_vars = base_num_vars + extra;
            let height = 1usize << num_vars;

            let quad = EquivalenceAir::Quad(QuadAir);
            let cubic = EquivalenceAir::CubicMixed(CubicMixedAir);
            let airs = [&quad, &cubic];

            let main_quad = random_table(&mut rng, height, 3);
            let main_cubic = random_table(&mut rng, height, 2);
            let preprocessed_cubic = random_table(&mut rng, height, 1);
            let tables = [&main_quad, &main_cubic];
            let preprocessed = [None, Some(&preprocessed_cubic)];

            let public_values_quad = [rng.random::<F>()];
            let public_values_cubic: [F; 0] = [];
            let public_values: [&[F]; 2] = [&public_values_quad, &public_values_cubic];

            let shapes = [
                AirShape {
                    degree: 2,
                    num_constraints: 2,
                },
                AirShape {
                    degree: 3,
                    num_constraints: 2,
                },
            ];
            let indices = [0, 1];

            let stage = Stage::new(
                &airs,
                &public_values,
                &indices,
                &preprocessed,
                &tables,
                &shapes,
            );

            let alpha: EF = rng.random();
            let betas = [EF::ONE, rng.random()].to_vec();
            let tau = Point::new((0..num_vars).map(|_| rng.random()).collect::<Vec<EF>>());
            let eq_point = (0..num_vars - 1).map(|_| rng.random()).collect::<Vec<EF>>();
            let eq_suffix = Poly::new_from_point(&eq_point, EF::ONE);

            let mut packed_state = RoundStateBase::new(&stage, alpha, betas.clone(), &tau);
            let mut unpacked_state = RoundStateBase::new(&stage, alpha, betas, &tau);

            let packed = packed_state.round_poly_packed(&eq_suffix);
            let unpacked = unpacked_state.round_poly_unpacked(&eq_suffix);

            assert_eq!(packed, unpacked, "num_vars={num_vars}");
        }
    }
}
