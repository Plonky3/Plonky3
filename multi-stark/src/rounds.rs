//! Per-round AIR zerocheck state.
//!
//! Builds round polynomials for `sum_x eq(tau, x) * g(x)` and folds state across challenges.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    dot_product,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyView};
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use p3_sumcheck::layout::Table;

use crate::folder::{FolderEvaluations, InteractionMultilinearFolder, MultilinearFolder};
use crate::lookup::AirLinkInstance;
use crate::packed_ext::PackedExt;
use crate::selectors::{BoundaryEvals, periodic_num_variables};

/// Native per-variable degrees of one AIR's two zerocheck expression families.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub(super) struct AirDegrees {
    /// Alpha-batched ordinary constraint degree, or zero when the AIR asserts none.
    pub(super) constraints: usize,
    /// Lookup-link degree, or zero when the AIR declares no interactions.
    pub(super) interactions: usize,
}

impl AirDegrees {
    /// Highest per-variable degree this AIR reaches in either family.
    ///
    /// The round state evaluates the AIR up to this node, and stops accumulating the
    /// lower-degree family once its own final node is past.
    pub(super) const fn max(self) -> usize {
        if self.constraints > self.interactions {
            self.constraints
        } else {
            self.interactions
        }
    }
}

/// One batch of AIRs that share a single trace height.
///
/// A stage activates when the global sumcheck cube shrinks to its height.
/// Every table inside a stage has the same number of variables.
pub(super) struct Stage<'air, 'data, A, F: Field, EF> {
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
    /// Native ordinary-constraint and lookup-link degrees for each AIR.
    pub(super) degrees: Vec<AirDegrees>,
    /// Shared variable count, equal to the base-two logarithm of the common height.
    pub(super) num_vars: usize,
    /// One-time lookup initialization consumed when this stage activates.
    coupling: StageCoupling<EF>,
}

impl<'air, 'data, A, F: Field, EF: Field> Stage<'air, 'data, A, F, EF> {
    /// Build a stage from AIRs that all share one trace height.
    ///
    /// # Panics
    ///
    /// Panics if the stage is empty or its per-AIR input lengths disagree.
    /// Panics if an AIR has no positive-degree expression family.
    /// Panics if the tables do not all have the same height.
    /// Panics if a preprocessed table's height differs from the stage height.
    /// Panics if a trace width or public-value count differs from its AIR declaration.
    pub(super) fn new(
        airs: Vec<&'air A>,
        public_values: Vec<&'data [F]>,
        indices: Vec<usize>,
        preprocessed: Vec<Option<&'data Table<F>>>,
        tables: Vec<&'data Table<F>>,
        degrees: Vec<AirDegrees>,
        coupling: StageCoupling<EF>,
    ) -> Self
    where
        A: BaseAir<F>,
    {
        assert!(!tables.is_empty());
        assert_eq!(airs.len(), tables.len());
        assert_eq!(preprocessed.len(), tables.len());
        assert_eq!(public_values.len(), tables.len());
        assert_eq!(degrees.len(), tables.len());
        assert!(degrees.iter().all(|degrees| degrees.max() > 0));

        // Every table in a stage binds the same zerocheck variables, so heights must agree.
        let num_vars = tables
            .iter()
            .map(|table| table.num_variables())
            .all_equal_value()
            .unwrap();

        // Preprocessed columns fold alongside the main columns, so they share the height.
        assert!(
            preprocessed
                .iter()
                .flatten()
                .all(|table| table.num_variables() == num_vars),
            "preprocessed tables must match the main trace height"
        );

        for (((air, public_values), preprocessed), table) in airs
            .iter()
            .zip(&public_values)
            .zip(&preprocessed)
            .zip(&tables)
        {
            assert_eq!(table.num_polys(), air.width());
            assert_eq!(
                preprocessed.map_or(0, Table::num_polys),
                air.preprocessed_width(),
            );
            assert_eq!(public_values.len(), air.num_public_values());
        }

        Self {
            num_vars,
            indices,
            airs,
            public_values,
            preprocessed,
            tables,
            degrees,
            coupling,
        }
    }

    /// The part of the folded lookup claim that becomes live when this stage activates.
    ///
    /// The global sumcheck holds the remainder as a dormant constant until then.
    pub(super) fn lookup_claim(&self, eta: EF) -> EF {
        eta * self.coupling.claims.values().copied().sum::<EF>()
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
pub(crate) struct RoundStateBase<'air, 'data, A, F: Field, EF> {
    /// Public inputs forwarded to the AIR.
    public_values: Vec<&'data [F]>,
    /// Random scalar batching the AIR constraints.
    alpha: EF,
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
    /// Ordinary AIR constraints grouped by their native round-polynomial degree.
    constraint_groups: Vec<DegreeGroup<EF>>,
    /// Lookup links grouped independently by their native round-polynomial degree.
    interaction_groups: Vec<DegreeGroup<EF>>,
    /// Per-AIR column spans and native family degrees driving the per-round column fold.
    slots: Vec<AirSlot<'air, A>>,
    /// Zerocheck point coordinates, used to recover the omitted node one for each AIR.
    tau: Point<EF>,
    /// Lookup coefficients for this stage.
    ///
    /// Empty when no AIR of this stage declares a lookup.
    coupling: InteractionCoupling<EF>,
    /// Common scalar applied to lookup claims and evaluations after grouping.
    eta: EF,
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
/// The stored groups are aligned to multiples of the packing width.
/// An offset window generally straddles two adjacent groups.
/// So each lane is reconstructed independently rather than assuming a contiguous layout.
#[inline]
fn packed_window<F: Field, EF: ExtensionField<F>>(
    column: &[EF::ExtensionPacking],
    start: usize,
    len: usize,
    tail: EF,
) -> EF::ExtensionPacking {
    let packing_width = F::Packing::WIDTH;
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
    /// Ordinary AIR constraints grouped by their native round-polynomial degree.
    constraint_groups: Vec<DegreeGroup<EF>>,
    /// Lookup links grouped independently by their native round-polynomial degree.
    interaction_groups: Vec<DegreeGroup<EF>>,
    /// Per-AIR column spans and native family degrees driving the per-round column fold.
    slots: Vec<AirSlot<'air, A>>,
    /// Zerocheck point coordinates, used to recover the omitted node one for each AIR.
    tau: Point<EF>,
    /// Number of already-bound prefix coordinates.
    round: usize,
    /// Repeat-last successor values for each main column at the folded tail row.
    next_tail: Vec<EF>,
    /// Lookup/AIR-link coefficients retained from this stage's base-field round.
    coupling: InteractionCoupling<EF>,
    /// Common scalar applied to lookup claims and evaluations after grouping.
    lookup_scale: EF,
}

/// One AIR's column values at the fully bound sumcheck point.
///
/// Stages activate by trace height, not in caller order, so each set of openings travels
/// alongside the caller position it has to be scattered back to.
pub(super) struct AirOpenings<EF> {
    /// Current-row value of each main column.
    pub(super) local: Vec<EF>,
    /// Successor value of each main column the AIR reads on the next row.
    pub(super) next: Vec<EF>,
    /// Current-row value of each preprocessed column.
    pub(super) preprocessed_local: Vec<EF>,
    /// Successor value of each preprocessed column the AIR reads on the next row.
    pub(super) preprocessed_next: Vec<EF>,
}

/// Lookup coefficients held in whatever representation the current round kernel uses.
///
/// A packed kernel rebuilds this once per round with every scalar lifted into a lane group.
/// The row loop then never broadcasts a scalar again.
struct InteractionCoupling<A> {
    /// One entry per lookup-declaring AIR of this stage, in stage order.
    ///
    /// Each slot records its own position here, so no map lookup happens in the row loop.
    links: Vec<AirLinkInstance<A>>,
    /// Coefficient `theta * beta^k` applied to payload coordinate `k`.
    theta_beta_powers: Vec<A>,
}

/// Everything a stage needs from the lookup reduction, consumed once when it activates.
///
/// Both maps are keyed by caller order and are sparse: only lookup-declaring AIRs appear.
/// Activation turns them into the compact stage-ordered form the round kernels use.
pub(crate) struct StageCoupling<A> {
    /// Each AIR's private share of the folded lookup claim.
    claims: BTreeMap<usize, A>,
    /// Each AIR's block weights and bus offsets.
    links: BTreeMap<usize, AirLinkInstance<A>>,
    /// Coefficient `theta * beta^k` applied to payload coordinate `k`.
    theta_beta_powers: Vec<A>,
}

impl<A> StageCoupling<A> {
    /// Pair the private claims with the link data they belong to.
    ///
    /// # Panics
    ///
    /// Panics if the two maps do not cover exactly the same AIRs.
    pub(crate) fn new(
        claims: BTreeMap<usize, A>,
        links: BTreeMap<usize, AirLinkInstance<A>>,
        theta_beta_powers: Vec<A>,
    ) -> Self {
        // A claim without link data, or the reverse, would leave one half of the fold undefined.
        assert!(claims.keys().eq(links.keys()));
        Self {
            claims,
            links,
            theta_beta_powers,
        }
    }
}

/// Run one AIR at one interpolation node and return the families this node needs.
///
/// The ordinary folder is used whenever no lookup value is wanted.
/// That keeps the constraint-only path free of every lookup-related branch and read.
#[inline]
fn evaluate_air_families<'a, F, Var, Acc, A>(
    folder: MultilinearFolder<'a, F, Var, Acc>,
    coupling: &'a InteractionCoupling<Acc>,
    enabled: EnabledFamilies,
    air: &A,
) -> FolderEvaluations<Acc>
where
    F: PrimeCharacteristicRing + Copy + Sync,
    Var: Algebra<F> + Copy + Send + Sync,
    Acc: Algebra<Var> + Copy,
    A: Air<MultilinearFolder<'a, F, Var, Acc>> + Air<InteractionMultilinearFolder<'a, F, Var, Acc>>,
{
    match enabled.interaction {
        Some(interaction) => {
            let link = &coupling.links[interaction.link_index];
            InteractionMultilinearFolder::new(
                folder,
                link,
                &coupling.theta_beta_powers,
                enabled.constraints,
            )
            .eval_air(air)
        }
        None => {
            debug_assert!(enabled.constraints);
            FolderEvaluations {
                constraints: folder.eval_air(air),
                interactions: Acc::ZERO,
            }
        }
    }
}

/// Scratch for scalar round-polynomial folds.
///
/// The base path uses one instance; the extension path allocates one per worker.
struct Scratch<F, EF> {
    /// Unweighted ordinary-constraint evaluations for each AIR at its native nodes.
    constraint_evals: Vec<Vec<EF>>,
    /// Unweighted lookup-link evaluations, already summed within each interaction group.
    interaction_evals: Vec<Vec<EF>>,
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
struct PackedScratch<P, EF> {
    /// Unweighted ordinary-constraint evaluations for each AIR at its native nodes.
    constraint_evals: Vec<Vec<EF>>,
    /// Unweighted lookup-link evaluations, already summed within each interaction group.
    interaction_evals: Vec<Vec<EF>>,
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
    fn new(constraint_degrees: &[usize], interaction_degrees: &[usize], width: usize) -> Self {
        Self {
            constraint_evals: constraint_degrees
                .iter()
                .copied()
                .map(EF::zero_vec)
                .collect(),
            interaction_evals: interaction_degrees
                .iter()
                .copied()
                .map(EF::zero_vec)
                .collect(),
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

impl<P, EF> PackedScratch<P, EF>
where
    P: PrimeCharacteristicRing,
    EF: PrimeCharacteristicRing,
{
    fn new(constraint_degrees: &[usize], interaction_degrees: &[usize], width: usize) -> Self {
        Self {
            constraint_evals: constraint_degrees
                .iter()
                .copied()
                .map(EF::zero_vec)
                .collect(),
            interaction_evals: interaction_degrees
                .iter()
                .copied()
                .map(EF::zero_vec)
                .collect(),
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
}

/// Where one AIR's lookup link lands, for an AIR that declares one.
#[derive(Clone, Copy, Default)]
struct AirInteractionSlot {
    /// Native per-variable degree of this AIR's lookup-link expression.
    degree: usize,
    /// Degree group collecting this AIR's lookup evaluations.
    group_index: usize,
    /// Position of this AIR's coefficients in the stage's compact link vector.
    link_index: usize,
}

/// Which expression families one AIR contributes at one interpolation node.
#[derive(Clone, Copy)]
struct EnabledFamilies {
    /// Whether the ordinary constraints are wanted here.
    constraints: bool,
    /// Where to send the lookup link, absent when it is not wanted here.
    interaction: Option<AirInteractionSlot>,
}

/// How many columns of each group one AIR owns.
#[derive(Clone, Copy)]
struct AirColumnWidths {
    /// Committed main trace columns.
    main: usize,
    /// Committed preprocessed columns, zero when the AIR declares none.
    preprocessed: usize,
    /// Materialized periodic columns, zero when the AIR declares none.
    periodic: usize,
}

/// One AIR's slice of the stage's merged column buffer, plus its fold metadata.
///
/// Every AIR of a stage keeps its columns in one shared buffer laid out group by group:
///
/// ```text
///     | air 0: main | prep | periodic | air 1: main | prep | periodic | ...
///       ^ main_offset      ^ periodic_offset
/// ```
///
/// The offsets stay valid for the whole sumcheck.
/// Folding shrinks every column by the same factor and never reorders them.
struct AirSlot<'air, A> {
    /// AIR evaluated by this slot.
    air: &'air A,
    /// Position of this AIR within its stage, in caller order.
    stage_index: usize,
    /// Original caller index used to return this AIR's openings.
    caller_index: usize,
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
    /// Native degree of the ordinary constraint family.
    constraint_degree: usize,
    /// Lookup metadata, absent when this AIR declares no interactions.
    interaction: Option<AirInteractionSlot>,
}

impl<'air, A> AirSlot<'air, A> {
    /// Lay out every AIR's columns once and attach its fold metadata.
    ///
    /// The sparse caller-keyed link map is flattened into a dense stage-ordered vector.
    /// The row loop then indexes a slice instead of searching a map.
    ///
    /// ```text
    ///     stage caller indices : [ 7,  2,  9,  4 ]
    ///     input links          : {     2 -> L2,      4 -> L4 }
    ///     returned links       : [ L2, L4 ]
    ///     recorded link index  :       0            1
    /// ```
    ///
    /// Every later representation of the coefficients keeps this order.
    /// A recorded index therefore stays valid for the whole sumcheck.
    ///
    /// # Returns
    ///
    /// - one slot per AIR of the stage, in stage order;
    /// - the dense link vector the slots point into.
    ///
    /// # Panics
    ///
    /// Panics if a lookup-declaring AIR has no link, or a link names an AIR outside the stage.
    fn build<EF>(
        airs: &[&'air A],
        caller_indices: &[usize],
        degrees: &[AirDegrees],
        column_widths: &[AirColumnWidths],
        mut links: BTreeMap<usize, AirLinkInstance<EF>>,
        interaction_group_by_degree: &BTreeMap<usize, usize>,
    ) -> (Vec<Self>, Vec<AirLinkInstance<EF>>) {
        assert_eq!(airs.len(), caller_indices.len());
        assert_eq!(airs.len(), degrees.len());
        assert_eq!(airs.len(), column_widths.len());

        let mut column_offset = 0;
        let mut active_links = Vec::with_capacity(links.len());
        let slots = (0..airs.len())
            .map(|stage_index| {
                let degrees = degrees[stage_index];
                let caller_index = caller_indices[stage_index];
                let AirColumnWidths {
                    main: main_width,
                    preprocessed: preprocessed_width,
                    periodic: periodic_width,
                } = column_widths[stage_index];
                let main_offset = column_offset;
                column_offset += main_width;
                let preprocessed_offset = column_offset;
                column_offset += preprocessed_width;
                let periodic_offset = column_offset;
                column_offset += periodic_width;
                let interaction = match interaction_group_by_degree.get(&degrees.interactions) {
                    Some(&group_index) => {
                        let link = links
                            .remove(&caller_index)
                            .expect("lookup-active AIR requires link metadata");
                        let interaction = AirInteractionSlot {
                            degree: degrees.interactions,
                            group_index,
                            link_index: active_links.len(),
                        };
                        active_links.push(link);
                        Some(interaction)
                    }
                    None => {
                        debug_assert!(!links.contains_key(&caller_index));
                        None
                    }
                };
                debug_assert_eq!(interaction.is_some(), degrees.interactions > 0);
                Self {
                    air: airs[stage_index],
                    stage_index,
                    caller_index,
                    main_offset,
                    main_width,
                    preprocessed_offset,
                    preprocessed_width,
                    periodic_offset,
                    periodic_width,
                    constraint_degree: degrees.constraints,
                    interaction,
                }
            })
            .collect();
        assert!(
            links.is_empty(),
            "stage coupling contains links for AIRs outside this stage"
        );
        (slots, active_links)
    }

    /// Select which expression families this AIR contributes at one interpolation node.
    ///
    /// A family is skipped past its own degree, since it is already pinned down there.
    ///
    /// The first base-field round also skips the ordinary node-zero evaluation:
    /// the rows are still boolean there, so satisfied constraints are known to vanish.
    const fn enabled_families(
        &self,
        node: usize,
        include_constraint_node_zero: bool,
    ) -> EnabledFamilies {
        let constraints = self.constraint_degree > 0
            && node <= self.constraint_degree
            && (node != 0 || include_constraint_node_zero);
        let interaction = match self.interaction {
            Some(interaction) if node <= interaction.degree => Some(interaction),
            _ => None,
        };
        EnabledFamilies {
            constraints,
            interaction,
        }
    }
}

/// Expression families sharing one per-variable degree and one reduced claim.
///
/// Grouping by degree lets a lower-degree family skip the nodes only a higher one needs.
struct DegreeGroup<EF> {
    /// Common per-variable degree of every AIR in this group, with the eq factor stripped.
    degree: usize,
    /// Stage-local indices of the AIRs contributing to this group.
    air_indices: Vec<usize>,
    /// Current reduced sumcheck claim for this group's round polynomial.
    claim: EF,
    /// This round's scaled evaluations at nodes `0, 2, 3, ..., degree`.
    last_evals: Vec<EF>,
    /// Barycentric helper for this group's degree, reused across rounds.
    interpolator: RoundPolyInterpolator<EF>,
}

impl<EF: Field> DegreeGroup<EF> {
    /// Bucket nonempty expression families by their native degree.
    fn build(degrees: impl IntoIterator<Item = usize>) -> Vec<Self> {
        // A btree keyed by degree gives deterministic group order across prover and verifier.
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        for (air_index, degree) in degrees.into_iter().enumerate() {
            if degree > 0 {
                groups.entry(degree).or_default().push(air_index);
            }
        }

        // Each degree becomes one group with a zero starting claim and a prebuilt interpolator.
        groups
            .into_iter()
            .map(|(degree, air_indices)| Self {
                degree,
                air_indices,
                claim: EF::ZERO,
                last_evals: EF::zero_vec(degree),
                interpolator: RoundPolyInterpolator::new(degree),
            })
            .collect()
    }

    /// Build the groups and seed each one with the claims of the AIRs it holds.
    ///
    /// Ordinary constraints start from zero, so only the lookup groups need this.
    ///
    /// # Panics
    ///
    /// Panics if the degrees and the caller indices disagree on length.
    /// Panics if an AIR in a nonempty group has no claim.
    fn build_with_claims<I>(
        degrees: I,
        caller_indices: &[usize],
        claims: &BTreeMap<usize, EF>,
        claim_scale: EF,
    ) -> Vec<Self>
    where
        I: IntoIterator<Item = usize>,
        I::IntoIter: ExactSizeIterator,
    {
        let degrees = degrees.into_iter();
        assert_eq!(degrees.len(), caller_indices.len());

        let mut groups = Self::build(degrees);
        for group in &mut groups {
            group.claim = claim_scale
                * group
                    .air_indices
                    .iter()
                    .map(|&stage_index| {
                        let caller_index = caller_indices[stage_index];
                        *claims
                            .get(&caller_index)
                            .expect("lookup-active AIR requires an initial claim")
                    })
                    .sum::<EF>();
        }
        groups
    }

    /// Evaluate this group's eq-stripped round polynomial `q` at an interpolation node.
    ///
    /// The prover never stores `q(1)`.
    /// It is recovered from the sumcheck claim relation instead:
    ///
    /// ```text
    ///     claim = (1 - tau) * q(0) + tau * q(1)
    ///  => q(1)  = (claim - (1 - tau) * q(0)) / tau
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the group degree is zero, since a constant has no round polynomial.
    /// Panics if `tau` is zero, since recovering the dropped node divides by it.
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
        // Reuse the group's node accumulators across rounds.
        self.last_evals.fill(EF::ZERO);
        for &air_index in &self.air_indices {
            // beta^i for this AIR, applied once here rather than per row.
            let beta = betas[air_index];
            // Fold the AIR's unweighted per-node sums in, scaled by beta.
            self.last_evals
                .iter_mut()
                .zip(air_evals[air_index].iter())
                .for_each(|(acc, &value)| *acc += beta * value);
        }
    }

    /// Store one lookup group's per-node sums, scaled by the lookup separation scalar.
    ///
    /// The row scan already summed the group's AIRs together, so no beta weighting is left.
    fn write_interaction_last_evals(&mut self, scale: EF, evals: &[EF]) {
        debug_assert_eq!(evals.len(), self.degree);
        self.last_evals
            .iter_mut()
            .zip(evals)
            .for_each(|(out, &value)| *out = scale * value);
    }
}

/// Turn one round's raw per-node sums into the stage's transmitted round polynomial.
///
/// Beta weighting and the lookup scalar are applied here rather than inside the row loop.
/// That saves one extension multiply per node per row.
///
/// Groups below the stage's top degree are extrapolated up to it before being summed.
///
/// # Panics
///
/// Panics if the stage has no degree group at all.
fn finish_round<EF: Field>(
    constraint_groups: &mut [DegreeGroup<EF>],
    interaction_groups: &mut [DegreeGroup<EF>],
    betas: &[EF],
    interaction_scale: EF,
    constraint_evals: &[Vec<EF>],
    interaction_evals: &[Vec<EF>],
    tau: EF,
) -> Vec<EF> {
    constraint_groups
        .iter_mut()
        .for_each(|group| group.write_last_evals(betas, constraint_evals));
    interaction_groups
        .iter_mut()
        .zip(interaction_evals)
        .for_each(|(group, evals)| group.write_interaction_last_evals(interaction_scale, evals));

    let degree = constraint_groups
        .iter()
        .chain(interaction_groups.iter())
        .map(|group| group.degree)
        .max()
        .expect("round state requires at least one degree group");
    let mut out = EF::zero_vec(degree);
    constraint_groups
        .iter()
        .chain(interaction_groups.iter())
        .for_each(|group| group.combine_evals(&mut out, tau));
    out
}

impl<'air, 'data, A, F, EF> RoundStateBase<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Activate a stage: materialize its periodic columns and lay out its degree groups.
    ///
    /// Ordinary constraints and lookup links are bucketed separately.
    /// One AIR can carry two different native degrees, and the lower must not pay for the higher.
    ///
    /// # Arguments
    ///
    /// - `stage`: the AIRs sharing this trace height, with their traces and lookup state.
    /// - `alpha`: batches the constraints of one AIR.
    /// - `eta`: separates lookup links from ordinary constraints.
    /// - `betas`: one power per AIR, batching the AIRs against each other.
    /// - `tau`: the zerocheck point coordinates this stage still has to bind.
    ///
    /// # Panics
    ///
    /// Panics if the zerocheck point does not match the stage height.
    /// Panics if a periodic column's period is not a power of two dividing the trace height.
    /// Panics if the beta powers do not number one per AIR.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn new(
        stage: Stage<'air, 'data, A, F, EF>,
        alpha: EF,
        eta: EF,
        betas: Vec<EF>,
        tau: Point<EF>,
    ) -> Self {
        assert_eq!(
            tau.num_variables(),
            stage.num_vars,
            "zerocheck point must match the stage height"
        );

        let Stage {
            indices,
            airs,
            public_values,
            preprocessed,
            tables,
            degrees,
            num_vars,
            coupling,
        } = stage;
        // Materialize each AIR's periodic columns to the full trace height.
        //
        //     period vector  : [v_0, v_1]
        //     trace height 8 : [v_0, v_1, v_0, v_1, v_0, v_1, v_0, v_1]
        //
        // The full-height column is a genuine multilinear polynomial.
        // It therefore folds through the sumcheck exactly like a committed column.
        let trace_height = 1 << num_vars;
        let periodic = airs
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
        let column_widths = tables
            .iter()
            .zip(&preprocessed)
            .zip(&periodic)
            .map(|((main, preprocessed), periodic)| AirColumnWidths {
                main: main.num_polys(),
                preprocessed: preprocessed.map_or(0, Table::num_polys),
                periodic: periodic.as_ref().map_or(0, Table::num_polys),
            })
            .collect::<Vec<_>>();

        let num_airs = airs.len();
        assert_eq!(
            betas.len(),
            num_airs,
            "one beta power is required for each AIR"
        );

        // Ordinary constraints keep their native degrees and their post-scan beta weighting.
        let constraint_groups =
            DegreeGroup::build(degrees.iter().map(|degrees| degrees.constraints));

        // Lookup links form their own native-degree groups.
        // Only the groups holding a lookup-declaring AIR start from a nonzero claim.
        let StageCoupling {
            claims,
            links,
            theta_beta_powers,
        } = coupling;

        let interaction_groups = DegreeGroup::build_with_claims(
            degrees.iter().map(|degrees| degrees.interactions),
            &indices,
            &claims,
            eta,
        );

        let interaction_group_by_degree = interaction_groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| (group.degree, group_index))
            .collect::<BTreeMap<_, _>>();

        let (slots, links) = AirSlot::build(
            &airs,
            &indices,
            &degrees,
            &column_widths,
            links,
            &interaction_group_by_degree,
        );

        let coupling = InteractionCoupling {
            links,
            theta_beta_powers,
        };

        Self {
            public_values,
            alpha,
            periodic,
            preprocessed,
            tables,
            betas,
            constraint_groups,
            interaction_groups,
            slots,
            tau,
            coupling,
            eta,
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
        self.constraint_groups
            .iter()
            .chain(&self.interaction_groups)
            .map(|group| group.degree)
            .max()
            .unwrap()
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
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
            + for<'b> Air<MultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, F::Packing, EF::ExtensionPacking>>,
        EF::ExtensionPacking: From<EF> + From<F::Packing>,
    {
        let width = self.total_width();
        let height = self.num_evals();
        let scalar_half = height / 2;
        let packing_width = F::Packing::WIDTH;
        let packed_half = scalar_half / packing_width;
        let degree = self.degree();
        let alpha = EF::ExtensionPacking::from(self.alpha);

        let coupling = InteractionCoupling {
            links: self
                .coupling
                .links
                .iter()
                .map(|link| link.map(EF::ExtensionPacking::from))
                .collect(),
            theta_beta_powers: self
                .coupling
                .theta_beta_powers
                .iter()
                .copied()
                .map(EF::ExtensionPacking::from)
                .collect(),
        };
        let constraint_degrees = self
            .slots
            .iter()
            .map(|slot| slot.constraint_degree)
            .collect::<Vec<_>>();

        let interaction_degrees = self
            .interaction_groups
            .iter()
            .map(|group| group.degree)
            .collect::<Vec<_>>();

        assert_ne!(packed_half, 0);

        let scratch = eq_suffix
            .as_slice()
            .par_chunks_exact(packing_width)
            .enumerate()
            .par_fold_reduce(
                || PackedScratch::new(&constraint_degrees, &interaction_degrees, width),
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;

                    let fill_columns = |scratch: &mut PackedScratch<F::Packing, EF>,
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
                    };
                    for slot in &self.slots {
                        fill_columns(
                            &mut scratch,
                            slot.main_offset,
                            self.tables[slot.stage_index],
                        );
                        if let Some(preprocessed) = self.preprocessed[slot.stage_index] {
                            fill_columns(&mut scratch, slot.preprocessed_offset, preprocessed);
                        }
                        if let Some(periodic) = self.periodic[slot.stage_index].as_ref() {
                            fill_columns(&mut scratch, slot.periodic_offset, periodic);
                        }
                    }

                    let (mut boundary, boundary_diff) =
                        BoundaryEvals::<F::Packing>::row_pair_packed(s, scalar_half, height);

                    for node in 0..=degree {
                        if node != 1 {
                            for slot in &self.slots {
                                let enabled = slot.enabled_families(node, false);
                                if !enabled.constraints && enabled.interaction.is_none() {
                                    continue;
                                }
                                let folder = MultilinearFolder::new(
                                    &scratch.local_point
                                        [slot.main_offset..slot.main_offset + slot.main_width],
                                    &scratch.next_point
                                        [slot.main_offset..slot.main_offset + slot.main_width],
                                    boundary,
                                    self.public_values[slot.stage_index],
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
                                let evaluations =
                                    evaluate_air_families(folder, &coupling, enabled, slot.air);
                                let eval_index = if node == 0 { 0 } else { node - 1 };
                                if enabled.constraints {
                                    scratch.constraint_evals[slot.stage_index][eval_index] +=
                                        dot_product::<EF, _, _>(
                                            eq_suffix.iter().copied(),
                                            EF::ExtensionPacking::to_ext_iter([
                                                evaluations.constraints
                                            ]),
                                        );
                                }
                                if let Some(interaction) = enabled.interaction {
                                    scratch.interaction_evals[interaction.group_index]
                                        [eval_index] += dot_product::<EF, _, _>(
                                        eq_suffix.iter().copied(),
                                        EF::ExtensionPacking::to_ext_iter([
                                            evaluations.interactions
                                        ]),
                                    );
                                }
                            }
                        }
                        if node != degree {
                            scratch.add_diffs();
                            boundary += boundary_diff;
                        }
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.constraint_evals
                        .iter_mut()
                        .zip(rhs.constraint_evals)
                        .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                    lhs.interaction_evals
                        .iter_mut()
                        .zip(rhs.interaction_evals)
                        .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                    lhs
                },
            );
        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.eta,
            &scratch.constraint_evals,
            &scratch.interaction_evals,
            self.tau.as_slice()[0],
        )
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_unpacked(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, F, EF>>,
    {
        let width = self.total_width();
        let height = self.num_evals();
        let half = height / 2;
        let degree = self.degree();

        let constraint_degrees = self
            .slots
            .iter()
            .map(|slot| slot.constraint_degree)
            .collect::<Vec<_>>();

        let interaction_degrees = self
            .interaction_groups
            .iter()
            .map(|group| group.degree)
            .collect::<Vec<_>>();

        let mut scratch = Scratch::<F, EF>::new(&constraint_degrees, &interaction_degrees, width);

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
                fill_columns(
                    &mut scratch,
                    slot.main_offset,
                    self.tables[slot.stage_index],
                );
                if let Some(preprocessed) = self.preprocessed[slot.stage_index] {
                    fill_columns(&mut scratch, slot.preprocessed_offset, preprocessed);
                }
                if let Some(periodic) = self.periodic[slot.stage_index].as_ref() {
                    fill_columns(&mut scratch, slot.periodic_offset, periodic);
                }
            }

            let (mut boundary, boundary_diff) = BoundaryEvals::<F>::row_pair(s, half, height);

            for node in 0..=degree {
                if node != 1 {
                    for slot in &self.slots {
                        let enabled = slot.enabled_families(node, false);
                        if !enabled.constraints && enabled.interaction.is_none() {
                            continue;
                        }
                        let folder = MultilinearFolder::new(
                            &scratch.local_point
                                [slot.main_offset..slot.main_offset + slot.main_width],
                            &scratch.next_point
                                [slot.main_offset..slot.main_offset + slot.main_width],
                            boundary,
                            self.public_values[slot.stage_index],
                            self.alpha,
                        )
                        .with_preprocessed(
                            &scratch.local_point[slot.preprocessed_offset
                                ..slot.preprocessed_offset + slot.preprocessed_width],
                            &scratch.next_point[slot.preprocessed_offset
                                ..slot.preprocessed_offset + slot.preprocessed_width],
                        )
                        .with_periodic(
                            &scratch.local_point
                                [slot.periodic_offset..slot.periodic_offset + slot.periodic_width],
                        );

                        let evaluations =
                            evaluate_air_families(folder, &self.coupling, enabled, slot.air);

                        let eval_index = if node == 0 { 0 } else { node - 1 };
                        if enabled.constraints {
                            scratch.constraint_evals[slot.stage_index][eval_index] +=
                                eq_suffix * evaluations.constraints;
                        }
                        if let Some(interaction) = enabled.interaction {
                            scratch.interaction_evals[interaction.group_index][eval_index] +=
                                eq_suffix * evaluations.interactions;
                        }
                    }
                }
                if node != degree {
                    scratch.add_diffs();
                    boundary += boundary_diff;
                }
            }
        }

        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.eta,
            &scratch.constraint_evals,
            &scratch.interaction_evals,
            self.tau.as_slice()[0],
        )
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold(mut self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF>
    where
        A: for<'b> Air<MultilinearFolder<'b, F, F, EF>>,
    {
        let tau = self.tau.as_slice()[0];
        self.constraint_groups
            .iter_mut()
            .chain(self.interaction_groups.iter_mut())
            .for_each(|group| group.claim = group.eval(tau, r));

        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let width = self.total_width();
        let mut next_tail = Vec::with_capacity(width);
        for slot in &self.slots {
            next_tail.extend(
                self.tables[slot.stage_index]
                    .iter_polys()
                    .map(|col| r * (col[num_evals - 1] - col[half]) + col[half]),
            );
            if let Some(preprocessed) = self.preprocessed[slot.stage_index] {
                next_tail.extend(
                    preprocessed
                        .iter_polys()
                        .map(|col| r * (col[num_evals - 1] - col[half]) + col[half]),
                );
            }
            if let Some(periodic) = self.periodic[slot.stage_index].as_ref() {
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
                    self.tables[slot.stage_index]
                        .par_iter_polys()
                        .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                        .collect::<Vec<_>>(),
                );
                if let Some(preprocessed) = self.preprocessed[slot.stage_index] {
                    columns.extend(
                        preprocessed
                            .par_iter_polys()
                            .map(|col| PolyView::new(col).fix_prefix_var_to_packed(r))
                            .collect::<Vec<_>>(),
                    );
                }
                if let Some(periodic) = self.periodic[slot.stage_index].as_ref() {
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
                    self.tables[slot.stage_index]
                        .par_iter_polys()
                        .map(|col| PolyView::new(col).fix_prefix_var(r))
                        .collect::<Vec<_>>(),
                );
                if let Some(preprocessed) = self.preprocessed[slot.stage_index] {
                    columns.extend(
                        preprocessed
                            .par_iter_polys()
                            .map(|col| PolyView::new(col).fix_prefix_var(r))
                            .collect::<Vec<_>>(),
                    );
                }
                if let Some(periodic) = self.periodic[slot.stage_index].as_ref() {
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
            public_values: self.public_values,
            alpha: self.alpha,
            betas: self.betas,
            constraint_groups: self.constraint_groups,
            interaction_groups: self.interaction_groups,
            slots: self.slots,
            tau: self.tau,
            round: 1,
            columns,
            next_tail,
            coupling: self.coupling,
            lookup_scale: self.eta,
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
        self.constraint_groups
            .iter()
            .chain(&self.interaction_groups)
            .map(|group| group.degree)
            .max()
            .unwrap()
    }

    /// Split the final folded columns back into per-AIR openings in original caller order.
    pub(crate) fn into_openings(self) -> Vec<(usize, AirOpenings<EF>)>
    where
        A: BaseAir<F>,
    {
        let local = self
            .columns
            .as_scalar()
            .iter()
            .map(|poly| poly.as_constant().unwrap())
            .collect::<Vec<_>>();
        let all_next = self.next_tail;

        self.slots
            .into_iter()
            .map(|slot| {
                let main_end = slot.main_offset + slot.main_width;
                let preprocessed_end = slot.preprocessed_offset + slot.preprocessed_width;
                let next = slot
                    .air
                    .main_next_row_columns()
                    .into_iter()
                    .map(|column| all_next[slot.main_offset + column])
                    .collect();
                let preprocessed_next = slot
                    .air
                    .preprocessed_next_row_columns()
                    .into_iter()
                    .map(|column| all_next[slot.preprocessed_offset + column])
                    .collect();

                (
                    slot.caller_index,
                    AirOpenings {
                        local: local[slot.main_offset..main_end].to_vec(),
                        next,
                        preprocessed_local: local[slot.preprocessed_offset..preprocessed_end]
                            .to_vec(),
                        preprocessed_next,
                    },
                )
            })
            .collect()
    }

    /// Evaluate this round's polynomial at every interpolation node.
    ///
    /// The packed kernel runs while a fold still leaves enough residual rows to fill a lane.
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
            > + for<'b> Air<InteractionMultilinearFolder<'b, F, EF, EF>>
            + for<'b> Air<
                InteractionMultilinearFolder<
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
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, EF, EF>>,
    {
        let width = self.width();
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let degree = self.degree();
        let constraint_degrees = self
            .slots
            .iter()
            .map(|slot| slot.constraint_degree)
            .collect::<Vec<_>>();
        let interaction_degrees = self
            .interaction_groups
            .iter()
            .map(|group| group.degree)
            .collect::<Vec<_>>();

        let scratch = eq_suffix.as_slice().par_iter().enumerate().par_fold_reduce(
            || Scratch::<EF, EF>::new(&constraint_degrees, &interaction_degrees, width),
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

                for node in 0..=degree {
                    if node != 1 {
                        for slot in &self.slots {
                            let enabled = slot.enabled_families(node, true);
                            if !enabled.constraints && enabled.interaction.is_none() {
                                continue;
                            }
                            let folder = MultilinearFolder::new(
                                &scratch.local_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                &scratch.next_point
                                    [slot.main_offset..slot.main_offset + slot.main_width],
                                boundary,
                                self.public_values[slot.stage_index],
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
                            );
                            let evaluations =
                                evaluate_air_families(folder, &self.coupling, enabled, slot.air);
                            let eval_index = if node == 0 { 0 } else { node - 1 };
                            if enabled.constraints {
                                scratch.constraint_evals[slot.stage_index][eval_index] +=
                                    eq_suffix * evaluations.constraints;
                            }
                            if let Some(interaction) = enabled.interaction {
                                scratch.interaction_evals[interaction.group_index][eval_index] +=
                                    eq_suffix * evaluations.interactions;
                            }
                        }
                    }
                    if node != degree {
                        scratch.add_diffs();
                        boundary += boundary_diff;
                    }
                }

                scratch
            },
            |mut lhs, rhs| {
                lhs.constraint_evals
                    .iter_mut()
                    .zip(rhs.constraint_evals)
                    .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                lhs.interaction_evals
                    .iter_mut()
                    .zip(rhs.interaction_evals)
                    .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                lhs
            },
        );
        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.lookup_scale,
            &scratch.constraint_evals,
            &scratch.interaction_evals,
            self.tau.as_slice()[self.round],
        )
    }

    /// SIMD-packed twin of the scalar kernel above.
    ///
    /// Earlier rounds already lifted every column into the extension field.
    /// One lane group therefore holds as many consecutive residual rows as the base field packs:
    ///
    /// ```text
    ///     rows   : x_0  x_1  x_2  x_3 ...
    ///     lanes  : |------- one packed element -------|
    /// ```
    ///
    /// A wrapper supplies the mixed base-times-extension multiply the AIR interface needs.
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
            > + for<'b> Air<
                InteractionMultilinearFolder<
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
        let coupling = InteractionCoupling {
            links: self
                .coupling
                .links
                .iter()
                .map(|link| link.map(|value| PackedExt::new(EF::ExtensionPacking::from(value))))
                .collect(),
            theta_beta_powers: self
                .coupling
                .theta_beta_powers
                .iter()
                .copied()
                .map(|value| PackedExt::new(EF::ExtensionPacking::from(value)))
                .collect(),
        };
        let constraint_degrees = self
            .slots
            .iter()
            .map(|slot| slot.constraint_degree)
            .collect::<Vec<_>>();
        let interaction_degrees = self
            .interaction_groups
            .iter()
            .map(|group| group.degree)
            .collect::<Vec<_>>();
        assert_ne!(packed_half, 0);

        let scratch = eq_suffix
            .as_slice()
            .par_chunks_exact(packing_width)
            .enumerate()
            .par_fold_reduce(
                || {
                    PackedScratch::<PackedExt<F, EF::ExtensionPacking>, EF>::new(
                        &constraint_degrees,
                        &interaction_degrees,
                        width,
                    )
                },
                |mut scratch, (packed_s, eq_suffix)| {
                    let s = packed_s * packing_width;

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

                    for node in 0..=degree {
                        if node != 1 {
                            for slot in &self.slots {
                                let enabled = slot.enabled_families(node, true);
                                if !enabled.constraints && enabled.interaction.is_none() {
                                    continue;
                                }
                                let folder = MultilinearFolder::new(
                                    &scratch.local_point
                                        [slot.main_offset..slot.main_offset + slot.main_width],
                                    &scratch.next_point
                                        [slot.main_offset..slot.main_offset + slot.main_width],
                                    boundary,
                                    self.public_values[slot.stage_index],
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
                                let evaluations =
                                    evaluate_air_families(folder, &coupling, enabled, slot.air);
                                let eval_index = if node == 0 { 0 } else { node - 1 };
                                if enabled.constraints {
                                    scratch.constraint_evals[slot.stage_index][eval_index] +=
                                        dot_product::<EF, _, _>(
                                            eq_suffix.iter().copied(),
                                            EF::ExtensionPacking::to_ext_iter([evaluations
                                                .constraints
                                                .0]),
                                        );
                                }
                                if let Some(interaction) = enabled.interaction {
                                    scratch.interaction_evals[interaction.group_index]
                                        [eval_index] += dot_product::<EF, _, _>(
                                        eq_suffix.iter().copied(),
                                        EF::ExtensionPacking::to_ext_iter([evaluations
                                            .interactions
                                            .0]),
                                    );
                                }
                            }
                        }
                        if node != degree {
                            scratch.add_diffs();
                            boundary += boundary_diff;
                        }
                    }

                    scratch
                },
                |mut lhs, rhs| {
                    lhs.constraint_evals
                        .iter_mut()
                        .zip(rhs.constraint_evals)
                        .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                    lhs.interaction_evals
                        .iter_mut()
                        .zip(rhs.interaction_evals)
                        .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
                    lhs
                },
            );
        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.lookup_scale,
            &scratch.constraint_evals,
            &scratch.interaction_evals,
            self.tau.as_slice()[self.round],
        )
    }

    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold(&mut self, r: EF)
    where
        A: for<'b> Air<MultilinearFolder<'b, F, EF, EF>>,
    {
        let tau = self.tau.as_slice()[self.round];
        self.constraint_groups
            .iter_mut()
            .chain(self.interaction_groups.iter_mut())
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
