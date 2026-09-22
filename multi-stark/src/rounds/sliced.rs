//! Zerocheck rounds evaluated on bit-sliced `GF(4)` planes of the trace.
//!
//! A stage whose cells all lie in `S = GF(4)` is repacked once, two bits per cell:
//!
//! ```text
//!     column c, rows 64w .. 64w + 63   ->   one (low, high) pair of words
//! ```
//!
//! A round then evaluates the AIR once per sixty-four residual rows, see [`crate::sliced`].
//!
//! Round `j` evaluates the columns at points whose first `j` coordinates are interpolation nodes
//! rather than the challenges already drawn. Every column value there is a `GF(4)` combination
//! of `2^(j+1)` corner cells, and the round polynomial follows by interpolating over those nodes:
//!
//! ```text
//!     q_j(t) = sum_x eq(x) * C(r_0, .., r_(j-1), t, x)
//!            = sum_v L_v(r_0, .., r_(j-1)) * sum_x eq(x) * C(v, t, x)      v in {nodes}^j
//! ```
//!
//! `C` has degree at most `d` in each variable, so `d + 1` nodes interpolate it exactly.
//! Every value is the one the generic kernel computes, so the round polynomial is the same.

#[cfg(test)]
extern crate std;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use p3_air::{Air, BaseAir};
use p3_field::{
    Algebra, ExtensionField, Field, HasSubfield, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::Table;

use super::repr::{lane_group, sum_lanes};
use super::{
    AirSlot, ExtColumns, InteractionCoupling, NodeStep, PackedScratch, RoundStateBase,
    RoundStateExt, Scratch, add_slice, evaluated_nodes, finish_round, lower_evals, next_row_runs,
    node_schedule, rows_per_task,
};
use crate::folder::{InteractionMultilinearFolder, MultilinearFolder};
use crate::packed_ext::{PackedExt, PackedRepr};
use crate::selectors::BoundaryEvals;
use crate::sliced::{LaneSums, SLICED_LANES, SlicedFolder, SlicedGf4, gf4_coordinates, is_gf4};

/// Most rounds a stage may evaluate on its planes.
///
/// Each round doubles both the plane work and the corner words a residual row folds, so the
/// count has a ceiling. A stage is also capped by the row variables its words leave unbound.
/// With its late-materialization parameter set, [`ReprBackend`](crate::ReprBackend) serves a
/// stage that qualifies one further round from its planes, outside this cap.
pub const MAX_SLICED_ROUNDS: usize = 4;

/// Longest prefix a plane fold binds.
///
/// A stage evaluates rounds on its planes with at most [`MAX_SLICED_ROUNDS`] challenges bound.
/// The delayed boundary path binds the challenge of its last such round as it unslices, so a
/// plane fold binds at most one challenge more.
const MAX_PLANE_FOLD_ROUNDS: usize = MAX_SLICED_ROUNDS + 1;

/// How a sliced first round is used by a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlicedStrategy {
    /// Evaluate only the sparse nodes needed by the current sliced round.
    Sequential,
    /// Build the four-variable tensor used by the representation backend lookahead.
    TensorBoundary,
    /// Build the tensor, then retain its planes for one later representation round.
    TensorBoundaryLate,
}

/// Row variables one word's lanes span.
const LANE_VARIABLES: usize = SLICED_LANES.trailing_zeros() as usize;

/// Fewest row variables a stage takes the delayed boundary path with.
///
/// Its round-four fold binds [`MAX_PLANE_FOLD_ROUNDS`] challenges, which must leave a whole
/// word of residual rows. Its round-four evaluation binds one challenge fewer, which must leave
/// a whole word pair. Either way the stage needs [`LANE_VARIABLES`] past the fold's prefix.
const MIN_LATE_BOUNDARY_VARS: usize = MAX_PLANE_FOLD_ROUNDS + LANE_VARIABLES;

#[cfg(test)]
std::thread_local! {
    /// Delayed boundary rounds the planes have served on this thread.
    ///
    /// A proof's round loop runs on the thread that calls it, so a test reads this around one
    /// proof to see whether the dispatch took the delayed round.
    pub(crate) static LATE_BOUNDARY_ROUNDS: core::cell::Cell<usize> =
        const { core::cell::Cell::new(0) };
}

/// A stage's cells as bit planes, laid out word by word.
pub(super) struct SlicedTrace {
    /// Number of variables of the stage.
    num_vars: usize,
    /// Number of columns, in merged-buffer order.
    width: usize,
    /// `cells[w * width + c]`: the planes of column `c` over rows `64 w .. 64 w + 63`.
    cells: Vec<[u64; 2]>,
    /// The successor planes, laid out like `cells`.
    ///
    /// Row `s` holds the cell at row `min(s + 1, height - 1)`, the repeat-last successor.
    /// Zero for every column no AIR reads on the next row.
    successors: Vec<[u64; 2]>,
    /// The first-row, last-row, and transition selectors of each word, one bit per row.
    boundary: Vec<[u64; 3]>,
    /// Number of rounds evaluated on the planes.
    rounds: usize,
}

/// Pack sixty-four cells into their two coordinate planes.
///
/// # Returns
///
/// `None` when a cell lies outside `S`.
#[inline]
fn pack_word<F, S>(cells: &[F; SLICED_LANES]) -> Option<[u64; 2]>
where
    S: Field,
    F: HasSubfield<S>,
{
    let mut planes = [0_u64; 2];
    for (lane, cell) in cells.iter().enumerate() {
        let (low, high) = cell.as_subfield().and_then(gf4_coordinates)?;
        planes[0] |= u64::from(low) << lane;
        planes[1] |= u64::from(high) << lane;
    }
    Some(planes)
}

/// Copy packed Boolean source tables into the final word-major plane layout.
///
/// Every source matrix already stores one row-block per physical row, so this path only adds the
/// zero high plane and interleaves table segments in their merged-buffer order. `None` keeps the
/// caller on the representation-independent path when any source is dense or has an unexpected
/// number of word blocks.
fn direct_packed_cells<F: Field>(tables: &[&Table<F>], words: usize) -> Option<Vec<[u64; 2]>> {
    if tables.is_empty() {
        return None;
    }
    let width = tables.iter().map(|table| table.num_polys()).sum();
    if tables.iter().any(|table| {
        let Some(packed) = table.packed_bits() else {
            return true;
        };
        words.checked_mul(packed.width) != Some(packed.values.len())
    }) {
        return None;
    }

    let mut cells = vec![[0; 2]; words * width];
    cells
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(word, output)| {
            let mut offset = 0;
            for table in tables {
                let packed = table
                    .packed_bits()
                    .expect("direct packed ingestion checked every source table");
                let source = &packed.values[word * packed.width..(word + 1) * packed.width];
                for (destination, &value) in
                    output[offset..offset + packed.width].iter_mut().zip(source)
                {
                    *destination = [value, 0];
                }
                offset += packed.width;
            }
        });
    Some(cells)
}

/// The successor planes of one word: every lane reads the next row.
///
/// The top lane reads `carry`, the coordinates of the cell after the word.
#[inline]
fn successor_word(planes: [u64; 2], carry: (bool, bool)) -> [u64; 2] {
    let top = SLICED_LANES - 1;
    [
        (planes[0] >> 1) | (u64::from(carry.0) << top),
        (planes[1] >> 1) | (u64::from(carry.1) << top),
    ]
}

/// The multilinear value of one column at `(v, t)` for `t = 0` and `t = 1`, over one word of rows.
///
/// `corners[c]` holds corner `c`, whose bits are the prefix bits, first variable highest, then
/// `t`. Each prefix variable folds as `lo + v * (hi - lo)` with `v` a node of `GF(4)`, which
/// leaves the values at `t = 0` and `t = 1` in the first two entries.
#[inline]
fn fold_corners<F, S>(
    corners: &mut [SlicedGf4<F, S>],
    prefix: &[(bool, bool)],
) -> (SlicedGf4<F, S>, SlicedGf4<F, S>) {
    let mut len = corners.len();
    for &(low, high) in prefix {
        len /= 2;
        let (lo, hi) = corners.split_at_mut(len);
        for (lo, &hi) in lo.iter_mut().zip(hi.iter()) {
            *lo += (*lo + hi).scale(low, high);
        }
    }
    (corners[0], corners[1])
}

/// What every task of one sliced round shares.
struct SlicedRound<'a, 'air, A, F, S, R> {
    /// The stage's planes.
    trace: &'a SlicedTrace,
    /// The stage's AIRs and their column spans.
    slots: &'a [AirSlot<'air, A>],
    /// Public inputs of each AIR.
    public_values: &'a [&'a [F]],
    /// Descending alpha powers of each AIR, in the accumulation field.
    alpha_powers: &'a [Vec<R>],
    /// Every prefix of this round: node coordinates of each bound variable, first variable first.
    prefixes: Vec<Vec<(bool, bool)>>,
    /// Nodes this round evaluates, each with the step that reaches it.
    schedule: Vec<(usize, NodeStep<(bool, bool)>)>,
    /// Every successor column run of the stage.
    next_columns: Vec<Range<usize>>,
    /// Words each corner block spans.
    words: usize,
    /// The lane weights: the eq factor of the variables inside a word.
    lanes: LaneSums<R>,
    /// The eq factor of the word variables, one per word of a corner block.
    word_weights: Vec<R>,
    /// Whether node zero of the ordinary constraints is evaluated.
    include_node_zero: bool,
    /// Whether this pass stores every tensor node, including node one.
    tensor: bool,
    /// The subfield the planes hold.
    _subfield: core::marker::PhantomData<fn() -> S>,
}

/// Per-worker sums of a sliced round.
struct SlicedScratch<F, S, R> {
    /// `sums[air][prefix][node]`: eq-weighted, alpha-batched constraint sums.
    sums: Vec<Vec<Vec<R>>>,
    /// Column values at the current node.
    local: Vec<SlicedGf4<F, S>>,
    /// Column steps from `t = 0` to `t = 1`.
    local_diff: Vec<SlicedGf4<F, S>>,
    /// Successor values at the current node, zero outside the successor runs.
    next: Vec<SlicedGf4<F, S>>,
    /// Successor steps, zero outside the successor runs.
    next_diff: Vec<SlicedGf4<F, S>>,
    /// Corner buffer of one column.
    corners: Vec<SlicedGf4<F, S>>,
    /// Whether any evaluation was poisoned.
    poisoned: bool,
}

impl<F, S, R: Field> SlicedScratch<F, S, R> {
    fn new(degrees: &[usize], prefixes: usize, width: usize, corners: usize, tensor: bool) -> Self {
        Self {
            sums: degrees
                .iter()
                .map(|&degree| vec![R::zero_vec(if tensor { 3 } else { degree }); prefixes])
                .collect(),
            local: vec![SlicedGf4::default(); width],
            local_diff: vec![SlicedGf4::default(); width],
            next: vec![SlicedGf4::default(); width],
            next_diff: vec![SlicedGf4::default(); width],
            corners: vec![SlicedGf4::default(); corners],
            poisoned: false,
        }
    }

    /// Add another worker's sums into this one, poison included.
    fn merge(mut self, other: Self) -> Self {
        for (lhs, rhs) in self.sums.iter_mut().zip(other.sums) {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                R::add_slices(lhs, &rhs);
            }
        }
        self.poisoned |= other.poisoned;
        self
    }
}

impl<'a, 'air, A, F, S, R> SlicedRound<'a, 'air, A, F, S, R>
where
    F: HasSubfield<S>,
    S: Field,
    R: Field,
    A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
{
    /// Fold one column's corners at the prefix, over word `word` of each corner block.
    #[inline]
    fn fold_column(
        &self,
        planes: &[[u64; 2]],
        column: usize,
        word: usize,
        prefix: &[(bool, bool)],
        corners: &mut [SlicedGf4<F, S>],
    ) -> (SlicedGf4<F, S>, SlicedGf4<F, S>) {
        let width = self.trace.width;
        for (corner, value) in corners.iter_mut().enumerate() {
            let [low, high] = planes[(corner * self.words + word) * width + column];
            *value = SlicedGf4::from_planes(low, high);
        }
        fold_corners(corners, prefix)
    }

    /// Add one word of residual rows at one prefix to the scratch sums.
    fn accumulate(&self, scratch: &mut SlicedScratch<F, S, R>, word: usize, prefix_index: usize) {
        let prefix = &self.prefixes[prefix_index];
        let trace = self.trace;
        let SlicedScratch {
            local,
            local_diff,
            next,
            next_diff,
            corners,
            ..
        } = scratch;

        for column in 0..trace.width {
            let (lo, hi) = self.fold_column(&trace.cells, column, word, prefix, corners);
            local[column] = lo;
            local_diff[column] = lo + hi;
        }
        for run in &self.next_columns {
            for column in run.clone() {
                let (lo, hi) = self.fold_column(&trace.successors, column, word, prefix, corners);
                next[column] = lo;
                next_diff[column] = lo + hi;
            }
        }
        let mut selector = |index: usize| {
            for (corner, value) in corners.iter_mut().enumerate() {
                let plane = trace.boundary[corner * self.words + word][index];
                *value = SlicedGf4::from_planes(plane, 0);
            }
            let (lo, hi) = fold_corners(corners, prefix);
            (lo, lo + hi)
        };
        let (first, first_diff) = selector(0);
        let (last, last_diff) = selector(1);
        let (transition, transition_diff) = selector(2);
        let boundary = BoundaryEvals::new(first, last, transition);
        let boundary_diff = BoundaryEvals::new(first_diff, last_diff, transition_diff);

        self.evaluate_nodes(
            scratch,
            boundary,
            boundary_diff,
            self.word_weights[word],
            prefix_index,
        );
    }

    /// Step the folded word through every scheduled node, adding each AIR's value there.
    ///
    /// Never inlined: the AIR evaluation needs a large stack frame, which inside the parallel
    /// fold would be reserved again at every level of Rayon's recursive split.
    #[inline(never)]
    fn evaluate_nodes(
        &self,
        scratch: &mut SlicedScratch<F, S, R>,
        mut boundary: BoundaryEvals<SlicedGf4<F, S>>,
        boundary_diff: BoundaryEvals<SlicedGf4<F, S>>,
        weight: R,
        prefix_index: usize,
    ) {
        let SlicedScratch {
            sums,
            local,
            local_diff,
            next,
            next_diff,
            poisoned,
            ..
        } = scratch;
        for &(node, step) in &self.schedule {
            match step {
                NodeStep::Unit(count) => {
                    for _ in 0..count {
                        add_step(local, local_diff, next, next_diff, &self.next_columns, None);
                        boundary.first += boundary_diff.first;
                        boundary.last += boundary_diff.last;
                        boundary.transition += boundary_diff.transition;
                    }
                }
                NodeStep::Scaled(step) => {
                    add_step(
                        local,
                        local_diff,
                        next,
                        next_diff,
                        &self.next_columns,
                        Some(step),
                    );
                    boundary.first += boundary_diff.first.scale(step.0, step.1);
                    boundary.last += boundary_diff.last.scale(step.0, step.1);
                    boundary.transition += boundary_diff.transition.scale(step.0, step.1);
                }
            }
            let eval_index = if self.tensor {
                node
            } else if node == 0 {
                0
            } else {
                node - 1
            };
            for slot in self.slots {
                let enabled = if self.tensor {
                    slot.constraint_degree > 0 && node <= 2
                } else {
                    slot.enabled_families(node, self.include_node_zero)
                        .constraints
                };
                if !enabled {
                    continue;
                }
                let main = slot.main_offset..slot.main_offset + slot.main_width;
                let preprocessed =
                    slot.preprocessed_offset..slot.preprocessed_offset + slot.preprocessed_width;
                let periodic = slot.periodic_offset..slot.periodic_offset + slot.periodic_width;
                let evaluation = SlicedFolder::new(
                    &local[main.clone()],
                    &next[main],
                    boundary,
                    self.public_values[slot.stage_index],
                    &self.alpha_powers[slot.stage_index],
                    &self.lanes,
                )
                .with_preprocessed(&local[preprocessed.clone()], &next[preprocessed])
                .with_periodic(&local[periodic])
                .eval_air(slot.air);
                *poisoned |= evaluation.poisoned;
                sums[slot.stage_index][prefix_index][eval_index] += weight * evaluation.value;
            }
        }
    }
}

/// Step every column, and the successor columns inside `next_columns`, to the next node.
#[inline]
fn add_step<F, S>(
    local: &mut [SlicedGf4<F, S>],
    local_diff: &[SlicedGf4<F, S>],
    next: &mut [SlicedGf4<F, S>],
    next_diff: &[SlicedGf4<F, S>],
    next_columns: &[Range<usize>],
    scale: Option<(bool, bool)>,
) {
    let step = |value: &mut SlicedGf4<F, S>, &diff: &SlicedGf4<F, S>| {
        *value += scale.map_or(diff, |(low, high)| diff.scale(low, high));
    };
    local
        .iter_mut()
        .zip(local_diff)
        .for_each(|(v, d)| step(v, d));
    for run in next_columns {
        next[run.clone()]
            .iter_mut()
            .zip(&next_diff[run.clone()])
            .for_each(|(v, d)| step(v, d));
    }
}

/// The Lagrange basis over the nodes `0 ..= degree`, evaluated at `point`.
fn lagrange_weights<EF: Field>(degree: usize, point: EF) -> Vec<EF> {
    let nodes = (0..=degree).map(EF::interpolation_node).collect::<Vec<_>>();
    nodes
        .iter()
        .enumerate()
        .map(|(m, &node)| {
            let (numerator, denominator) = nodes
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != m)
                .fold((EF::ONE, EF::ONE), |(num, den), (_, &other)| {
                    (num * (point - other), den * (node - other))
                });
            numerator * denominator.inverse()
        })
        .collect()
}

/// The coordinates of a step between two interpolation nodes, if it lies in `S`.
fn step_coordinates<S, EF>(step: NodeStep<EF>) -> Option<NodeStep<(bool, bool)>>
where
    S: Field,
    EF: HasSubfield<S>,
{
    Some(match step {
        NodeStep::Unit(count) => NodeStep::Unit(count),
        NodeStep::Scaled(step) => NodeStep::Scaled(gf4_coordinates(step.as_subfield()?)?),
    })
}

/// Raw output of one sliced row pass, retaining every prefix and node sum.
struct SlicedRaw<R> {
    sums: Vec<Vec<Vec<R>>>,
}

/// Evaluate one sliced row pass, retaining every prefix and node sum.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, level = "debug", fields(round = challenges.len()))]
fn sliced_raw<A, F, EF, S, R>(
    eq_suffix: Option<&Poly<EF>>,
    trace: &SlicedTrace,
    slots: &[AirSlot<'_, A>],
    public_values: &[&[F]],
    alpha_powers: &[Vec<R>],
    tau: &[EF],
    challenges: &[EF],
    degree: usize,
    tensor: bool,
) -> Option<SlicedRaw<R>>
where
    F: HasSubfield<S>,
    EF: ExtensionField<F> + HasSubfield<S> + From<R>,
    S: Field,
    R: Field + From<EF>,
    A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
{
    // Tensor mode evaluates the fourth variable as the active node, so the existing
    // round-three layout provides 3^3 prefixes and 2^4 corners for its 81 entries.
    let round = if tensor { 3 } else { challenges.len() };
    let num_vars = trace.num_vars;
    debug_assert!(
        (tensor && trace.rounds == 3 || !tensor && round < trace.rounds) && tau.len() == num_vars
    );
    debug_assert!(is_gf4::<S>(), "sliced values hold GF(4)");

    let nodes = (0..=degree)
        .map(|node| gf4_coordinates(EF::interpolation_node(node).as_subfield()?))
        .collect::<Option<Vec<_>>>()?;
    let include_node_zero = tensor || round > 0;
    let schedule_nodes = if tensor {
        (0..=degree).collect::<Vec<_>>()
    } else {
        evaluated_nodes(slots, degree, include_node_zero).collect::<Vec<_>>()
    };
    let schedule = node_schedule::<EF>(schedule_nodes)
        .into_iter()
        .map(|(node, step)| Some((node, step_coordinates::<S, EF>(step)?)))
        .collect::<Option<Vec<_>>>()?;

    // Row variables split three ways: prefix and t, the words of a corner block, and the lanes.
    let lane_point = &tau[num_vars - LANE_VARIABLES..];
    let word_point = &tau[round + 1..num_vars - LANE_VARIABLES];
    let lane_weights = Poly::new_from_point(lane_point, EF::ONE);
    let word_weights = Poly::new_from_point(word_point, EF::ONE);
    // The row weights factor as word weight times lane weight, as a table the caller passes must.
    debug_assert!(eq_suffix.is_some_and(|eq_suffix| {
        eq_suffix.num_evals() == word_weights.num_evals() * SLICED_LANES
            && eq_suffix
                .as_slice()
                .iter()
                .enumerate()
                .all(|(row, &weight)| {
                    weight
                        == word_weights.as_slice()[row / SLICED_LANES]
                            * lane_weights.as_slice()[row % SLICED_LANES]
                })
    }));
    let lift = |values: &[EF]| {
        values
            .iter()
            .map(|&value| R::from(value))
            .collect::<Vec<_>>()
    };
    let generator = R::from(EF::from(S::GENERATOR));
    let lanes = LaneSums::new(&lift(lane_weights.as_slice()), generator);
    let word_weights = lift(word_weights.as_slice());

    // The prefixes in index order, the last variable varying fastest.
    let prefixes = (0..nodes.len().pow(round as u32))
        .map(|mut index| {
            let mut prefix = vec![(false, false); round];
            for coordinate in prefix.iter_mut().rev() {
                *coordinate = nodes[index % nodes.len()];
                index /= nodes.len();
            }
            prefix
        })
        .collect::<Vec<_>>();
    let context = SlicedRound {
        trace,
        slots,
        public_values,
        alpha_powers,
        prefixes,
        schedule,
        next_columns: next_row_runs(slots),
        words: word_weights.len(),
        lanes,
        word_weights,
        include_node_zero,
        tensor,
        _subfield: core::marker::PhantomData,
    };

    let degrees = slots
        .iter()
        .map(|slot| slot.constraint_degree)
        .collect::<Vec<_>>();
    let prefixes = context.prefixes.len();
    let corners = 2 << round;
    let scratch = (0..context.words * prefixes)
        .into_par_iter()
        .par_fold_reduce(
            || SlicedScratch::new(&degrees, prefixes, trace.width, corners, tensor),
            |mut scratch, task| {
                context.accumulate(&mut scratch, task / prefixes, task % prefixes);
                scratch
            },
            SlicedScratch::merge,
        );
    if scratch.poisoned {
        tracing::debug!("an AIR constant outside the subfield reached a sliced round");
        return None;
    }

    Some(SlicedRaw { sums: scratch.sums })
}

/// Evaluate round `challenges.len()` of a stage on its planes at sparse nodes.
#[allow(clippy::too_many_arguments)]
fn sliced_round<A, F, EF, S, R>(
    eq_suffix: &Poly<EF>,
    trace: &SlicedTrace,
    slots: &[AirSlot<'_, A>],
    public_values: &[&[F]],
    alpha_powers: &[Vec<R>],
    tau: &[EF],
    challenges: &[EF],
    degree: usize,
) -> Option<Vec<Vec<EF>>>
where
    F: HasSubfield<S>,
    EF: ExtensionField<F> + HasSubfield<S> + From<R>,
    S: Field,
    R: Field + From<EF>,
    A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
{
    let raw = sliced_raw(
        Some(eq_suffix),
        trace,
        slots,
        public_values,
        alpha_powers,
        tau,
        challenges,
        degree,
        false,
    )?;
    let weights = challenges
        .iter()
        .map(|&challenge| lagrange_weights(degree, challenge))
        .collect::<Vec<_>>();
    let prefix_weight = |mut prefix: usize| {
        let mut weight = EF::ONE;
        for weights in weights.iter().rev() {
            weight *= weights[prefix % weights.len()];
            prefix /= weights.len();
        }
        weight
    };
    Some(
        raw.sums
            .into_iter()
            .map(|prefix_sums| {
                let mut evals = EF::zero_vec(prefix_sums.first().map_or(0, Vec::len));
                for (prefix, sums) in prefix_sums.into_iter().enumerate() {
                    let weight = prefix_weight(prefix);
                    for (eval, sum) in evals.iter_mut().zip(sums) {
                        *eval += weight * EF::from(sum);
                    }
                }
                evals
            })
            .collect(),
    )
}

/// The complete four-variable tensor retained by representation-field lookahead.
struct SlicedTensor<EF> {
    /// Tensor entries in AIR, base-three prefix, node order.
    values: Vec<Vec<EF>>,
    /// Number of variables covered by the tensor.
    depth: usize,
}

/// Evaluate all 81 tensor entries before the first sliced fold.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, level = "debug", fields(round = 3))]
fn sliced_tensor<A, F, EF, S, R>(
    eq_suffix: Option<&Poly<EF>>,
    trace: &SlicedTrace,
    slots: &[AirSlot<'_, A>],
    public_values: &[&[F]],
    alpha_powers: &[Vec<R>],
    tau: &[EF],
) -> Option<SlicedTensor<EF>>
where
    F: HasSubfield<S>,
    EF: ExtensionField<F> + HasSubfield<S> + From<R>,
    S: Field,
    R: Field + From<EF>,
    A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
{
    debug_assert!(
        slots.iter().all(|slot| slot.constraint_degree <= 2),
        "the tensor holds nodes 0, 1 and 2 of each variable, which pin at most a quadratic"
    );
    let raw = sliced_raw(
        eq_suffix,
        trace,
        slots,
        public_values,
        alpha_powers,
        tau,
        &[],
        2,
        true,
    )?;
    Some(SlicedTensor {
        values: raw
            .sums
            .into_iter()
            .map(|prefix_sums| {
                prefix_sums
                    .into_iter()
                    .flat_map(|sums| sums.into_iter().map(EF::from))
                    .collect()
            })
            .collect(),
        depth: 4,
    })
}

/// Contract a cached tensor at one round's prefix and active interpolation node.
fn tensor_round<EF: Field, A>(
    tensor: &SlicedTensor<EF>,
    slots: &[AirSlot<'_, A>],
    tau: &[EF],
    challenges: &[EF],
    round: usize,
) -> Vec<Vec<EF>> {
    debug_assert!(round < tensor.depth && challenges.len() == round);
    debug_assert!(tau.len() >= tensor.depth);
    debug_assert!(
        slots.iter().all(|slot| slot.constraint_degree <= 2),
        "the contraction interpolates quadratics and fills nodes 0 and 2 only"
    );
    let prefix_weights = challenges
        .iter()
        .map(|&challenge| lagrange_weights(2, challenge))
        .collect::<Vec<_>>();
    slots
        .iter()
        .enumerate()
        .map(|(air, slot)| {
            if slot.constraint_degree == 0 {
                return Vec::new();
            }
            let mut evals = EF::zero_vec(slot.constraint_degree);
            for node in [0, 2] {
                if node > slot.constraint_degree {
                    continue;
                }
                let mut value = EF::ZERO;
                for index in 0..81 {
                    let mut coordinates = [0usize; 4];
                    let mut remaining = index;
                    for coordinate in coordinates.iter_mut().rev() {
                        *coordinate = remaining % 3;
                        remaining /= 3;
                    }
                    if coordinates[round] != node {
                        continue;
                    }
                    let mut weight = EF::ONE;
                    for (coordinate, weights) in
                        coordinates.iter().zip(prefix_weights.iter()).take(round)
                    {
                        weight *= weights[*coordinate];
                    }
                    for (coordinate, &tau) in coordinates
                        .iter()
                        .skip(round + 1)
                        .zip(&tau[round + 1..tensor.depth])
                    {
                        weight *= match *coordinate {
                            0 => EF::ONE - tau,
                            1 => tau,
                            _ => EF::ZERO,
                        };
                    }
                    value += weight * tensor.values[air][index];
                }
                if round == 0 && node == 0 {
                    value = EF::ZERO;
                }
                evals[if node == 0 { 0 } else { 1 }] = value;
            }
            evals
        })
        .collect()
}

impl<'air, 'data, A, F, EF> RoundStateBase<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Repack every cell of the stage into bit planes of `S`.
    ///
    /// # Returns
    ///
    /// `None` when `S` is not `GF(4)`, the stage is too short to fill a word per half, or a cell
    /// lies outside `S`.
    #[tracing::instrument(skip_all, level = "debug")]
    fn sliced_trace<S>(&self) -> Option<SlicedTrace>
    where
        S: Field,
        F: HasSubfield<S>,
    {
        let num_vars = self.num_evals().trailing_zeros() as usize;
        if !is_gf4::<S>() || num_vars <= LANE_VARIABLES {
            return None;
        }
        let rounds = self
            .sliced_rounds
            .min(MAX_SLICED_ROUNDS)
            .min(num_vars - LANE_VARIABLES);
        // A stage with no round to run on its planes is not repacked at all, so the caller
        // reaches for another kernel as it does for a stage that does not fit.
        if rounds == 0 {
            return None;
        }

        // Every column in merged-buffer order: main, preprocessed, periodic, AIR by AIR.
        let mut tables: Vec<&Table<F>> = Vec::new();
        for slot in &self.slots {
            tables.push(self.tables[slot.stage_index]);
            tables.extend(self.preprocessed[slot.stage_index]);
            tables.extend(self.periodic[slot.stage_index].as_ref());
        }
        let columns = tables
            .iter()
            .flat_map(|table| table.columns())
            .collect::<Vec<_>>();
        let width = columns.len();
        let next_columns = next_row_runs(&self.slots);
        let mut is_successor = vec![false; width];
        for column in next_columns.iter().flat_map(|run| run.clone()) {
            is_successor[column] = true;
        }

        // Each table's packed matrix is already word-major. Copy directly into the final layout
        // when no successor planes are needed; otherwise retain the generic column path, which
        // also computes the repeat-last successor words.
        let words = 1 << (num_vars - LANE_VARIABLES);
        let (cells, successors) = if next_columns.is_empty() {
            if let Some(cells) = direct_packed_cells(&tables, words) {
                (cells, vec![[0; 2]; words * width])
            } else {
                pack_sliced_columns::<F, S>(&columns, &is_successor, words, width)?
            }
        } else {
            pack_sliced_columns::<F, S>(&columns, &is_successor, words, width)?
        };

        let last = SLICED_LANES - 1;
        let boundary = (0..words)
            .map(|word| {
                let first = u64::from(word == 0);
                let last = if word + 1 == words { 1 << last } else { 0 };
                [first, last, !last]
            })
            .collect();

        Some(SlicedTrace {
            num_vars,
            width,
            cells,
            successors,
            boundary,
            rounds,
        })
    }

    /// Evaluate the first round polynomial on bit-sliced `GF(4)` planes, when the stage fits.
    ///
    /// # Returns
    ///
    /// - `Some`: exactly what [`Self::round_poly`] returns.
    /// - `None`: the stage does not fit, or an AIR constant outside `S` poisoned a value.
    ///   No round group has changed, so the caller runs another kernel instead.
    ///
    /// Whether the stage's cells fit `S` is recorded for [`Self::fits_subfield`].
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_sliced<S, R>(&mut self, eq_suffix: &Poly<EF>) -> Option<Vec<EF>>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S> + From<R>,
        R: Field + From<EF>,
        A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
    {
        self.round_poly_sliced_with_strategy::<S, R>(eq_suffix, SlicedStrategy::Sequential)
    }

    /// Evaluate the first round on planes, optionally retaining the tensor lookahead cache.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_sliced_with_strategy<S, R>(
        &mut self,
        eq_suffix: &Poly<EF>,
        strategy: SlicedStrategy,
    ) -> Option<Vec<EF>>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S> + From<R>,
        R: Field + From<EF>,
        A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
    {
        self.subfield_schedule::<S>()?;
        let trace = self.sliced_trace::<S>()?;
        self.fits_subfield = true;
        let alpha_powers = self
            .alpha_powers
            .iter()
            .map(|powers| powers.iter().map(|&power| R::from(power)).collect())
            .collect::<Vec<Vec<R>>>();
        let tensor_eligible = matches!(
            strategy,
            SlicedStrategy::TensorBoundary | SlicedStrategy::TensorBoundaryLate
        ) && trace.rounds == 3
            && trace.num_vars >= 10
            && self.degree() == 2
            && self
                .slots
                .iter()
                .all(|slot| slot.constraint_degree <= 2 && slot.interaction.is_none())
            && next_row_runs(&self.slots).is_empty();
        if tensor_eligible {
            // Only the factorization check inside the tensor pass reads this table.
            let tensor_eq_suffix = cfg!(debug_assertions)
                .then(|| Poly::new_from_point(&self.tau.as_slice()[4..], EF::ONE));
            if let Some(tensor) = sliced_tensor::<A, F, EF, S, R>(
                tensor_eq_suffix.as_ref(),
                &trace,
                &self.slots,
                &self.public_values,
                &alpha_powers,
                self.tau.as_slice(),
            ) {
                let evals = tensor_round(&tensor, &self.slots, self.tau.as_slice(), &[], 0);
                let late_boundary = strategy == SlicedStrategy::TensorBoundaryLate
                    && trace.num_vars >= MIN_LATE_BOUNDARY_VARS
                    && SLICED_LANES.is_multiple_of(R::Packing::WIDTH);
                self.sliced = Some(SlicedColumns {
                    trace,
                    challenges: Vec::new(),
                    tensor: Some(tensor),
                    late_boundary,
                });
                return Some(finish_round(
                    &mut self.constraint_groups,
                    &mut self.interaction_groups,
                    &self.betas,
                    self.eta,
                    &evals,
                    &[],
                    self.tau.as_slice()[0],
                ));
            }
        }
        let evals = sliced_round::<A, F, EF, S, R>(
            eq_suffix,
            &trace,
            &self.slots,
            &self.public_values,
            &alpha_powers,
            self.tau.as_slice(),
            &[],
            self.degree(),
        )?;
        self.sliced = Some(SlicedColumns {
            trace,
            challenges: Vec::new(),
            tensor: None,
            late_boundary: false,
        });

        // A sliced stage declares no lookup, so it has no lookup group to fill.
        Some(finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.eta,
            &evals,
            &[],
            self.tau.as_slice()[0],
        ))
    }

    /// Whether the first round ran on the stage's planes.
    pub(crate) const fn is_sliced(&self) -> bool {
        self.sliced.is_some()
    }

    /// Whether the representation-specific four-variable cache was installed.
    #[cfg(test)]
    pub(crate) fn has_sliced_tensor(&self) -> bool {
        self.sliced
            .as_ref()
            .is_some_and(|columns| columns.tensor.is_some())
    }

    /// Bind the first variable at `r`, keeping every column on the stage's planes.
    ///
    /// The alpha powers, the lookup coefficients, and the selector prefix cross into `R` here.
    /// The repeat-last tails are computed when the stage leaves its planes.
    ///
    /// # Panics
    ///
    /// Panics if the first round did not run on the planes, as [`Self::is_sliced`] reports.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_sliced<R>(mut self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        R: Field + From<EF>,
    {
        let mut columns = self
            .sliced
            .take()
            .expect("the first round ran on the planes");
        self.fold_claims(r);
        columns.challenges.push(r);
        let lift = |values: &[EF]| values.iter().map(|&value| R::from(value)).collect();
        RoundStateExt {
            public_values: self.public_values,
            alpha: R::from(self.alpha),
            alpha_powers: self
                .alpha_powers
                .iter()
                .map(|powers| lift(powers))
                .collect(),
            betas: self.betas,
            constraint_groups: self.constraint_groups,
            interaction_groups: self.interaction_groups,
            slots: self.slots,
            tau: self.tau,
            round: 1,
            next_tail: R::zero_vec(columns.width()),
            columns: ExtColumns::Sliced(columns),
            coupling: InteractionCoupling {
                links: self
                    .coupling
                    .links
                    .iter()
                    .map(|link| link.map(R::from))
                    .collect(),
                theta_beta_powers: lift(&self.coupling.theta_beta_powers),
            },
            lookup_scale: self.eta,
            boundary: BoundaryEvals::new(R::from(EF::ONE - r), R::from(r), R::from(EF::ONE - r)),
        }
    }
}

/// Pack representation-independent columns into word-major cell and successor planes.
#[allow(clippy::type_complexity)]
fn pack_sliced_columns<F, S>(
    columns: &[p3_sumcheck::layout::ColumnView<'_, F>],
    is_successor: &[bool],
    words: usize,
    width: usize,
) -> Option<(Vec<[u64; 2]>, Vec<[u64; 2]>)>
where
    F: HasSubfield<S>,
    S: Field,
{
    // Pack each column on its own, successor planes included, reading it contiguously.
    let top = SLICED_LANES - 1;
    let packed = columns
        .par_iter()
        .zip(is_successor)
        .map(|(column, &is_successor)| {
            let column_planes = if let Some(values) = column.as_dense() {
                values
                    .as_chunks::<SLICED_LANES>()
                    .0
                    .iter()
                    .map(pack_word::<F, S>)
                    .collect::<Option<Vec<_>>>()?
            } else {
                (0..column.len() / SLICED_LANES)
                    .map(|word| Some([column.boolean_word(word)?, 0]))
                    .collect::<Option<Vec<_>>>()?
            };
            let successor_planes = if is_successor {
                // Each word's top lane reads the lowest lane of the next word, and the last
                // word's top lane repeats itself.
                let lane = |planes: [u64; 2], lane: usize| {
                    ((planes[0] >> lane) & 1 == 1, (planes[1] >> lane) & 1 == 1)
                };
                let last = column_planes[column_planes.len() - 1];
                let carries = column_planes[1..]
                    .iter()
                    .map(|&next| lane(next, 0))
                    .chain([lane(last, top)]);
                column_planes
                    .iter()
                    .zip(carries)
                    .map(|(&planes, carry)| successor_word(planes, carry))
                    .collect()
            } else {
                Vec::new()
            };
            Some((column_planes, successor_planes))
        })
        .collect::<Option<Vec<_>>>()?;

    // Lay the words out word by word, every column of a word side by side.
    let mut cells = vec![[0; 2]; words * width];
    let mut successors = vec![[0; 2]; words * width];
    cells
        .par_chunks_mut(width)
        .zip(successors.par_chunks_mut(width))
        .enumerate()
        .for_each(|(word, (word_cells, word_successors))| {
            for ((cell, successor), (column_planes, successor_planes)) in word_cells
                .iter_mut()
                .zip(word_successors.iter_mut())
                .zip(&packed)
            {
                *cell = column_planes[word];
                if let Some(&planes) = successor_planes.get(word) {
                    *successor = planes;
                }
            }
        });
    Some((cells, successors))
}

/// A sliced stage's planes and the challenges bound so far.
pub(super) struct SlicedColumns<EF> {
    /// The stage's planes, none of whose columns has folded yet.
    trace: SlicedTrace,
    /// Every challenge bound so far, first variable first.
    challenges: Vec<EF>,
    /// Optional four-variable tensor retained by the representation backend.
    tensor: Option<SlicedTensor<EF>>,
    /// Whether the representation backend may defer materialization through round four.
    late_boundary: bool,
}

impl<EF> SlicedColumns<EF> {
    /// Number of columns.
    pub(super) const fn width(&self) -> usize {
        self.trace.width
    }

    /// Number of residual rows, one per assignment of the unbound variables.
    pub(super) const fn num_evals(&self) -> usize {
        1 << (self.trace.num_vars - self.challenges.len())
    }

    /// Whether the stage's sliced rounds are spent and its planes can still serve a round.
    ///
    /// This admits the boundary round alone. The delayed boundary path evaluates its round four
    /// with one more challenge bound, so this is false there, and
    /// [`RoundStateExt::round_poly_late_boundary`] gates that round instead.
    ///
    /// A word pair is the shortest run of residual rows that holds both halves of a row pair.
    const fn at_boundary(&self) -> bool {
        self.challenges.len() == self.trace.rounds && self.num_evals() >= ROW_HALVES * SLICED_LANES
    }
}

/// Transpose an 8 x 8 bit matrix held one row per byte.
///
/// Bit `j` of byte `i` becomes bit `i` of byte `j`.
#[inline]
const fn transpose_bytes(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^ t ^ (t << 28)
}

/// For each lane, the byte whose bit `i` is that lane's bit in `words[i]`, for up to eight words.
#[inline]
fn lane_masks(words: &[u64]) -> [u8; SLICED_LANES] {
    debug_assert!(words.len() <= 8);
    let mut masks = [0; SLICED_LANES];
    for (byte, masks) in masks.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let rows = words.iter().enumerate().fold(0_u64, |rows, (i, &word)| {
            rows | (((word >> (8 * byte)) & 0xff) << (8 * i))
        });
        masks.copy_from_slice(&transpose_bytes(rows).to_le_bytes());
    }
    masks
}

/// For each lane, the byte whose bit `i` is that lane's bit in `words[i]`, for up to eight words.
///
/// This is [`lane_masks`] restricted to the top lane, which no transpose is needed to read.
#[inline]
fn top_lane_mask(words: &[u64]) -> u8 {
    debug_assert!(words.len() <= GROUP_CORNERS);
    words.iter().enumerate().fold(0, |mask, (index, &word)| {
        mask | (((word >> (SLICED_LANES - 1)) as u8) << index)
    })
}

/// Subset sums of up to eight weights, indexed by the byte of the weights they include.
fn subset_sums<R: Field>(weights: &[R]) -> Vec<R> {
    let mut sums = R::zero_vec(1 << weights.len());
    for mask in 1..sums.len() {
        sums[mask] = sums[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
    }
    sums
}

/// Corners one subset-sum table covers, one bit of a mask byte each.
const GROUP_CORNERS: usize = 8;

/// Entries of one corner group's subset-sum table, one per value of a mask byte.
const GROUP_ENTRIES: usize = 1 << u8::BITS;

/// Corners the bound variables of a sliced stage can range over.
const MAX_CORNERS: usize = 1 << MAX_PLANE_FOLD_ROUNDS;

/// Mask bytes one corner group of one residual row reads, one per plane.
const PLANE_BYTES: usize = 2;

/// Halves of the residual rows a round reads side by side: the low one and the high one.
const ROW_HALVES: usize = 2;

/// The byte-indexed tables that carry a stage's planes into residual rows of `R`.
///
/// Each residual row of a column combines the cells the bound variables range over:
///
/// ```text
///     column(x) = sum_b eq(r, b) * cell(b, x)       b in {0, 1}^k
/// ```
///
/// A cell is `low + high * g`, so each row takes byte-indexed subset sums of `eq(r, .)`, one
/// lookup per plane per group of eight `b`. The tables do not depend on the column or the row,
/// so one set serves a whole round.
struct PlaneFold<'a, R> {
    /// The stage's planes.
    trace: &'a SlicedTrace,
    /// Subset sums of the eq weights, one table per corner group.
    low_sums: Vec<[R; GROUP_ENTRIES]>,
    /// The same sums scaled by the generator of `S`, indexed the same way.
    high_sums: Vec<[R; GROUP_ENTRIES]>,
    /// Corners the bound variables range over.
    corners: usize,
    /// Corner groups one residual row reads.
    groups: usize,
    /// Words one corner block spans.
    words: usize,
}

impl<'a, R: Field> PlaneFold<'a, R> {
    /// Tabulate the fold of `trace` at every challenge bound so far.
    fn new<S, EF>(trace: &'a SlicedTrace, challenges: &[EF]) -> Self
    where
        S: Field,
        EF: Field + HasSubfield<S>,
        R: From<EF>,
    {
        assert!(
            challenges.len() <= MAX_PLANE_FOLD_ROUNDS,
            "a plane fold's corner buffers must hold every corner of its bound prefix"
        );
        assert!(
            trace.num_vars >= challenges.len() + LANE_VARIABLES,
            "a bound prefix must leave a whole residual word, or its corners collapse onto one"
        );
        let generator = R::from(EF::from(S::GENERATOR));
        let weights = Poly::new_from_point(challenges, EF::ONE)
            .as_slice()
            .iter()
            .map(|&weight| R::from(weight))
            .collect::<Vec<_>>();
        let corners = weights.len();
        let groups = corners.div_ceil(GROUP_CORNERS);
        let mut low_sums = vec![[R::ZERO; GROUP_ENTRIES]; groups];
        let mut high_sums = vec![[R::ZERO; GROUP_ENTRIES]; groups];
        for ((low, high), weights) in low_sums
            .iter_mut()
            .zip(&mut high_sums)
            .zip(weights.chunks(GROUP_CORNERS))
        {
            for ((low, high), &sum) in low.iter_mut().zip(high).zip(&subset_sums(weights)) {
                (*low, *high) = (sum, generator * sum);
            }
        }
        Self {
            trace,
            low_sums,
            high_sums,
            corners,
            groups,
            words: (trace.cells.len() / trace.width) / corners,
        }
    }

    /// The corner words of one `(column, word)`, one array per plane.
    #[inline]
    fn corner_words(
        &self,
        planes: &[[u64; 2]],
        column: usize,
        word: usize,
    ) -> ([u64; MAX_CORNERS], [u64; MAX_CORNERS]) {
        let mut low = [0; MAX_CORNERS];
        let mut high = [0; MAX_CORNERS];
        let base = word * self.trace.width + column;
        let stride = self.words * self.trace.width;
        for (corner, (low, high)) in low[..self.corners]
            .iter_mut()
            .zip(&mut high[..self.corners])
            .enumerate()
        {
            let planes = planes[base + corner * stride];
            *low = planes[0];
            *high = planes[1];
        }
        (low, high)
    }

    /// The corners of one group, one array per plane.
    #[inline]
    fn group_words<'b>(
        &self,
        low: &'b [u64; MAX_CORNERS],
        high: &'b [u64; MAX_CORNERS],
        group: usize,
    ) -> (&'b [u64], &'b [u64]) {
        let start = group * GROUP_CORNERS;
        let end = (start + GROUP_CORNERS).min(self.corners);
        (&low[start..end], &high[start..end])
    }

    /// The value at every residual row of one word.
    fn fold_word(&self, planes: &[[u64; 2]], column: usize, word: usize, out: &mut [R]) {
        let (low, high) = self.corner_words(planes, column, word);
        out.fill(R::ZERO);
        for group in 0..self.groups {
            let (low_table, high_table) = (&self.low_sums[group], &self.high_sums[group]);
            let (low, high) = self.group_words(&low, &high, group);
            let (low, high) = (lane_masks(low), lane_masks(high));
            for (value, (&low, &high)) in out.iter_mut().zip(low.iter().zip(&high)) {
                *value += low_table[usize::from(low)] + high_table[usize::from(high)];
            }
        }
    }

    /// The value one half of one residual row's mask bytes stands for.
    ///
    /// # Panics
    ///
    /// Debug builds panic unless `bytes` holds one plane pair per corner group.
    #[inline]
    fn row_value(&self, bytes: &[u8]) -> R {
        debug_assert_eq!(bytes.len(), self.groups * PLANE_BYTES);
        let mut value = R::ZERO;
        for ((low_table, high_table), masks) in self
            .low_sums
            .iter()
            .zip(&self.high_sums)
            .zip(bytes.as_chunks::<PLANE_BYTES>().0)
        {
            value += low_table[usize::from(masks[0])] + high_table[usize::from(masks[1])];
        }
        value
    }

    /// The low-half and high-half values one residual row's mask bytes stand for.
    #[inline]
    fn row_pair(&self, bytes: &[u8]) -> (R, R) {
        let half = self.groups * PLANE_BYTES;
        (
            self.row_value(&bytes[..half]),
            self.row_value(&bytes[half..]),
        )
    }

    /// Write one `(column, word)`'s mask bytes lane by lane, `stride` bytes apart.
    fn write_lane_masks(
        &self,
        planes: &[[u64; 2]],
        column: usize,
        word: usize,
        stride: usize,
        out: &mut [u8],
    ) {
        let (low, high) = self.corner_words(planes, column, word);
        for group in 0..self.groups {
            let (low, high) = self.group_words(&low, &high, group);
            let (low, high) = (lane_masks(low), lane_masks(high));
            for (lane, (&low, &high)) in low.iter().zip(&high).enumerate() {
                let at = lane * stride + group * PLANE_BYTES;
                out[at] = low;
                out[at + 1] = high;
            }
        }
    }

    /// Write one `(column, word)`'s top-lane mask bytes.
    fn write_top_lane_mask(&self, planes: &[[u64; 2]], column: usize, word: usize, out: &mut [u8]) {
        let (low, high) = self.corner_words(planes, column, word);
        for group in 0..self.groups {
            let (low, high) = self.group_words(&low, &high, group);
            out[group * PLANE_BYTES] = top_lane_mask(low);
            out[group * PLANE_BYTES + 1] = top_lane_mask(high);
        }
    }
}

/// One word pair's mask bytes, transposed so each residual row reads its columns in order.
///
/// ```text
///     lane l : | column 0 | column 1 | ... |    residual rows 64 p + l and 64 p + l + half
///     column : the low half's mask bytes, group by group, then the high half's
/// ```
///
/// The lane past the last holds the successor planes, whose lane `l` is the cell of lane `l + 1`.
/// A row's next-row values are therefore the following lane's bytes, inside the word or not.
struct RowTile {
    /// The mask bytes, lane by lane.
    bytes: Vec<u8>,
    /// Bytes one column spans inside a lane.
    column_stride: usize,
    /// Bytes one lane spans.
    lane_stride: usize,
}

impl RowTile {
    /// An empty tile for a stage of `width` columns whose rows read `groups` corner groups.
    fn new(groups: usize, width: usize) -> Self {
        let column_stride = ROW_HALVES * groups * PLANE_BYTES;
        let lane_stride = column_stride * width;
        Self {
            bytes: vec![0; (SLICED_LANES + 1) * lane_stride],
            column_stride,
            lane_stride,
        }
    }

    /// One lane's mask bytes, column by column.
    #[inline]
    fn lane(&self, lane: usize) -> &[u8] {
        &self.bytes[lane * self.lane_stride..][..self.lane_stride]
    }

    /// One column's mask bytes inside one lane.
    #[inline]
    fn cell(&self, lane: usize, column: usize) -> &[u8] {
        &self.lane(lane)[column * self.column_stride..][..self.column_stride]
    }

    /// Lay out the mask bytes of word pair `pair`.
    ///
    /// # Panics
    ///
    /// Debug builds panic unless the tile was laid out for `fold`'s corner groups and width.
    fn fill<R: Field>(
        &mut self,
        fold: &PlaneFold<'_, R>,
        pair: usize,
        next_columns: &[Range<usize>],
    ) {
        debug_assert_eq!(self.column_stride, ROW_HALVES * fold.groups * PLANE_BYTES);
        debug_assert_eq!(self.lane_stride, self.column_stride * fold.trace.width);
        let words = [pair, pair + fold.words / ROW_HALVES];
        let half_bytes = fold.groups * PLANE_BYTES;
        for column in 0..fold.trace.width {
            for (half, &word) in words.iter().enumerate() {
                let at = column * self.column_stride + half * half_bytes;
                fold.write_lane_masks(
                    &fold.trace.cells,
                    column,
                    word,
                    self.lane_stride,
                    &mut self.bytes[at..],
                );
            }
        }

        // Only a successor column is ever read one row on, so only it needs the extra lane.
        let extra = SLICED_LANES * self.lane_stride;
        for run in next_columns {
            for column in run.clone() {
                for (half, &word) in words.iter().enumerate() {
                    let at = extra + column * self.column_stride + half * half_bytes;
                    fold.write_top_lane_mask(
                        &fold.trace.successors,
                        column,
                        word,
                        &mut self.bytes[at..],
                    );
                }
            }
        }
    }

    /// Read one residual row pair of every column into the buffers a node walk steps.
    fn read_row<R: Field>(
        &self,
        fold: &PlaneFold<'_, R>,
        lane: usize,
        next_columns: &[Range<usize>],
        scratch: &mut Scratch<R, R>,
    ) {
        let Scratch {
            local_point,
            local_diff,
            next_point,
            next_diff,
            ..
        } = scratch;
        for ((local, local_delta), bytes) in local_point
            .iter_mut()
            .zip(local_diff.iter_mut())
            .zip(self.lane(lane).chunks_exact(self.column_stride))
        {
            let (lo, hi) = fold.row_pair(bytes);
            *local = lo;
            *local_delta = hi - lo;
        }
        for run in next_columns {
            for ((column, next), next_delta) in run
                .clone()
                .zip(next_point.fill()[run.clone()].iter_mut())
                .zip(next_diff.fill()[run.clone()].iter_mut())
            {
                let (lo, hi) = fold.row_pair(self.cell(lane + 1, column));
                *next = lo;
                *next_delta = hi - lo;
            }
        }
    }

    /// One lane group of one column's residual row pairs, low halves then high halves.
    #[inline]
    fn lane_pair<F, R: Field>(
        &self,
        fold: &PlaneFold<'_, R>,
        lane: usize,
        column: usize,
    ) -> (PackedRepr<F, R>, PackedRepr<F, R>) {
        let (mut low, mut high) = (R::Packing::ZERO, R::Packing::ZERO);
        for (step, (l, h)) in low
            .as_slice_mut()
            .iter_mut()
            .zip(high.as_slice_mut())
            .enumerate()
        {
            (*l, *h) = fold.row_pair(self.cell(lane + step, column));
        }
        (PackedExt::new(low), PackedExt::new(high))
    }

    /// Read one lane group of residual row pairs of every column into a packed node walk's buffers.
    fn read_lane_group<F, R: Field>(
        &self,
        fold: &PlaneFold<'_, R>,
        lane: usize,
        next_columns: &[Range<usize>],
        scratch: &mut PackedScratch<PackedRepr<F, R>, PackedRepr<F, R>>,
    ) {
        let PackedScratch {
            local_point,
            local_diff,
            next_point,
            next_diff,
            ..
        } = scratch;
        for (column, (local, local_delta)) in local_point
            .iter_mut()
            .zip(local_diff.iter_mut())
            .enumerate()
        {
            let (lo, hi) = self.lane_pair(fold, lane, column);
            *local = lo;
            *local_delta = hi - lo;
        }
        for run in next_columns {
            for ((column, next), next_delta) in run
                .clone()
                .zip(next_point.fill()[run.clone()].iter_mut())
                .zip(next_diff.fill()[run.clone()].iter_mut())
            {
                let (lo, hi) = self.lane_pair(fold, lane + 1, column);
                *next = lo;
                *next_delta = hi - lo;
            }
        }
    }
}

impl<'air, 'data, A, F, EF, R> RoundStateExt<'air, 'data, A, F, EF, R>
where
    F: Field,
    EF: ExtensionField<F> + From<R>,
    R: Field + From<EF>,
{
    /// Whether the representation-specific four-variable cache is still installed.
    #[cfg(test)]
    pub(crate) const fn has_sliced_tensor(&self) -> bool {
        matches!(&self.columns, ExtColumns::Sliced(columns) if columns.tensor.is_some())
    }

    /// Evaluate this round's polynomial on the stage's planes, while they have rounds left.
    ///
    /// # Returns
    ///
    /// - `Some`: exactly what the scalar and packed kernels return.
    /// - `None`: the stage is not on its planes, its sliced rounds are spent, or an AIR constant
    ///   outside `S` poisoned a value. No round group has changed; see [`Self::unslice`].
    pub(crate) fn round_poly_sliced<S>(&mut self, eq_suffix: &Poly<EF>) -> Option<Vec<EF>>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S>,
        A: for<'b> Air<SlicedFolder<'b, F, S, R>>,
    {
        let ExtColumns::Sliced(columns) = &self.columns else {
            return None;
        };
        debug_assert_eq!(columns.challenges.len(), self.round);
        let evals = if let Some(tensor) = &columns.tensor {
            if columns.challenges.len() >= tensor.depth {
                return None;
            }
            tensor_round(
                tensor,
                &self.slots,
                self.tau.as_slice(),
                &columns.challenges,
                columns.challenges.len(),
            )
        } else {
            if columns.challenges.len() >= columns.trace.rounds {
                return None;
            }
            sliced_round::<A, F, EF, S, R>(
                eq_suffix,
                &columns.trace,
                &self.slots,
                &self.public_values,
                &self.alpha_powers,
                self.tau.as_slice(),
                &columns.challenges,
                self.degree(),
            )?
        };

        // A sliced stage declares no lookup, so it has no lookup group to fill.
        Some(finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.lookup_scale,
            &evals,
            &[],
            self.tau.as_slice()[self.round],
        ))
    }

    /// Bind the next variable at `r` while the stage is on its planes.
    ///
    /// # Returns
    ///
    /// Whether the stage is on its planes; when it is not, nothing has changed.
    pub(crate) fn fold_sliced(&mut self, r: EF) -> bool {
        let ExtColumns::Sliced(columns) = &mut self.columns else {
            return false;
        };
        if columns.challenges.len() >= columns.trace.rounds {
            return false;
        }
        columns.challenges.push(r);
        self.fold_claims(r);
        self.boundary.apply(R::from(r));
        self.round += 1;
        true
    }

    /// Take the stage's planes off the round state, leaving it with no columns.
    ///
    /// # Returns
    ///
    /// `None` when the stage is not on its planes, which then stay where they are.
    fn take_planes(&mut self) -> Option<(SlicedTrace, Vec<EF>)> {
        match core::mem::replace(&mut self.columns, ExtColumns::Scalar(Vec::new())) {
            ExtColumns::Sliced(SlicedColumns {
                trace,
                challenges,
                tensor: _,
                late_boundary: _,
            }) => Some((trace, challenges)),
            columns => {
                self.columns = columns;
                None
            }
        }
    }

    /// Read each successor column at the last residual row into its repeat-last tail.
    fn read_next_tails(&mut self, fold: &PlaneFold<'_, R>) {
        let mut last = [R::ZERO; SLICED_LANES];
        for run in next_row_runs(&self.slots) {
            for column in run {
                fold.fold_word(&fold.trace.successors, column, fold.words - 1, &mut last);
                self.next_tail[column] = last[SLICED_LANES - 1];
            }
        }
    }

    /// Fold a stage off its planes into scalar columns in `R`, at every challenge bound so far.
    ///
    /// Every residual row of every column is written out, see [`PlaneFold`]. The repeat-last
    /// tails read the successor planes at the last residual row the same way.
    ///
    /// Does nothing when the stage is not on its planes.
    pub(crate) fn unslice<S>(&mut self)
    where
        S: Field,
        EF: HasSubfield<S>,
    {
        let Some((trace, challenges)) = self.take_planes() else {
            return;
        };
        let _span = tracing::debug_span!("unslice").entered();
        let fold = PlaneFold::<R>::new::<S, EF>(&trace, &challenges);
        let rows = fold.words * SLICED_LANES;

        let scalar = (0..trace.width)
            .into_par_iter()
            .map(|column| {
                let mut values = R::zero_vec(rows);
                for (word, out) in values
                    .as_chunks_mut::<SLICED_LANES>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    fold.fold_word(&trace.cells, column, word, out);
                }
                Poly::new(values)
            })
            .collect();

        self.read_next_tails(&fold);
        self.columns = ExtColumns::Scalar(scalar);
    }

    /// Bind the next variable at `r`, folding the stage off its planes in the same pass.
    ///
    /// Each folded row reads the two residual rows the bound variable joins straight from the
    /// planes, so only the half-size result is ever written out.
    ///
    /// # Returns
    ///
    /// Whether the stage was still on its planes with its sliced rounds spent.
    pub(crate) fn fold_boundary<S>(&mut self, r: EF) -> bool
    where
        S: Field,
        EF: HasSubfield<S>,
    {
        let ExtColumns::Sliced(columns) = &self.columns else {
            return false;
        };
        if !columns.at_boundary() {
            return false;
        }
        let _span = tracing::debug_span!("fold_boundary").entered();
        let Some((trace, challenges)) = self.take_planes() else {
            unreachable!("the stage holds its planes")
        };
        let fold = PlaneFold::<R>::new::<S, EF>(&trace, &challenges);
        let half = fold.words * SLICED_LANES / ROW_HALVES;
        let challenge = R::from(r);

        self.fold_claims(r);

        // Each tail folds with the column value at the first residual row of the high half,
        // which is the first lane of the first word of that half.
        let high_words = fold.words / ROW_HALVES;
        let mut buffer = [R::ZERO; SLICED_LANES];
        for run in next_row_runs(&self.slots) {
            for column in run {
                fold.fold_word(&trace.successors, column, fold.words - 1, &mut buffer);
                let tail = buffer[SLICED_LANES - 1];
                fold.fold_word(&trace.cells, column, high_words, &mut buffer);
                let lo = buffer[0];
                self.next_tail[column] = lo + (tail - lo) * challenge;
            }
        }

        let scalar = (0..trace.width)
            .into_par_iter()
            .map(|column| {
                let mut values = R::zero_vec(half);
                let mut lo = [R::ZERO; SLICED_LANES];
                let mut hi = [R::ZERO; SLICED_LANES];
                for (word, out) in values
                    .as_chunks_mut::<SLICED_LANES>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    fold.fold_word(&trace.cells, column, word, &mut lo);
                    fold.fold_word(&trace.cells, column, word + high_words, &mut hi);
                    for (value, (&lo, &hi)) in out.iter_mut().zip(lo.iter().zip(&hi)) {
                        *value = lo + (hi - lo) * challenge;
                    }
                }
                Poly::new(values)
            })
            .collect();

        self.columns = ExtColumns::Scalar(scalar);
        self.boundary.apply(challenge);
        self.round += 1;
        true
    }

    /// Bind round three or four of the delayed boundary path without materializing an
    /// intermediate residual column.
    ///
    /// The round-three fold only drops the tensor. The round-four fold consumes all five
    /// recorded challenges through [`Self::unslice`], so its first scalar columns have length
    /// `N / 32`.
    pub(crate) fn fold_late_boundary<S>(&mut self, r: EF) -> bool
    where
        S: Field,
        EF: HasSubfield<S>,
    {
        let (round, prefix_len, tensor_present, late) = match &self.columns {
            ExtColumns::Sliced(columns) => (
                self.round,
                columns.challenges.len(),
                columns.tensor.is_some(),
                columns.late_boundary,
            ),
            _ => return false,
        };
        let valid = late
            && match round {
                3 => prefix_len == 3 && tensor_present,
                4 => prefix_len == 4 && !tensor_present,
                _ => false,
            };
        if !valid {
            return false;
        }

        self.fold_claims(r);
        match round {
            3 => {
                let ExtColumns::Sliced(columns) = &mut self.columns else {
                    unreachable!("late boundary gate checked sliced columns")
                };
                columns.challenges.push(r);
                columns.tensor = None;
                self.boundary.apply(R::from(r));
                self.round += 1;
            }
            4 => {
                let ExtColumns::Sliced(columns) = &mut self.columns else {
                    unreachable!("late boundary gate checked sliced columns")
                };
                columns.challenges.push(r);
                self.unslice::<S>();
                self.boundary.apply(R::from(r));
                self.round += 1;
            }
            _ => unreachable!("late boundary gate checked round three or four"),
        }
        true
    }

    /// Evaluate this round's polynomial straight from the stage's planes, its sliced rounds spent.
    ///
    /// One word pair of residual rows is expanded at a time, into the mask bytes its rows read
    /// rather than into their values, and dropped once those rows are done. The residual columns
    /// themselves are never written out; [`Self::fold_boundary`] reads the planes again.
    ///
    /// # Returns
    ///
    /// - `Some`: exactly what [`Self::round_poly_repr`] returns once the stage has unsliced.
    /// - `None`: the stage is not on its planes, still has a sliced round left, or is too short
    ///   for a word pair. Nothing has changed; see [`Self::unslice`].
    pub(crate) fn round_poly_boundary<S>(&mut self, eq_suffix: &Poly<EF>) -> Option<Vec<EF>>
    where
        S: Field,
        EF: HasSubfield<S>,
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, R, R>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, R, R>>
            + for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let ExtColumns::Sliced(columns) = &self.columns else {
            return None;
        };
        let lanes = R::Packing::WIDTH;
        // A lane group covers consecutive rows of one word, so it must not straddle two.
        if !columns.at_boundary() || !SLICED_LANES.is_multiple_of(lanes) {
            return None;
        }
        let _span = tracing::debug_span!("round_poly_boundary").entered();
        Some(self.round_poly_planes::<S>(eq_suffix))
    }

    /// Evaluate one representation-field round directly from the stage's planes.
    ///
    /// This is shared by the incumbent boundary round and the delayed fifth-challenge path so
    /// both use the same row/packed evaluator, node schedule, and finish-round semantics.
    ///
    /// # Panics
    ///
    /// When the residual rows do not fill a word pair.
    fn round_poly_planes<S>(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        S: Field,
        EF: HasSubfield<S>,
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, R, R>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, R, R>>
            + for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let ExtColumns::Sliced(columns) = &self.columns else {
            unreachable!("plane round requires sliced columns")
        };
        let num_evals = self.num_evals();
        let (constraints, interactions) = {
            let fold = PlaneFold::<R>::new::<S, EF>(&columns.trace, &columns.challenges);
            assert!(
                fold.words >= ROW_HALVES,
                "a plane round needs a whole word pair, or its round polynomial sums no rows"
            );
            if R::Packing::WIDTH > 1 && num_evals / 2 >= R::Packing::WIDTH {
                self.boundary_evals_lanes(eq_suffix, &fold)
            } else {
                self.boundary_evals_rows(eq_suffix, &fold)
            }
        };

        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.lookup_scale,
            &constraints,
            &interactions,
            self.tau.as_slice()[self.round],
        )
    }

    /// Evaluate round four of the delayed boundary path from the retained planes.
    pub(crate) fn round_poly_late_boundary<S>(&mut self, eq_suffix: &Poly<EF>) -> Option<Vec<EF>>
    where
        S: Field,
        EF: HasSubfield<S>,
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, R, R>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, R, R>>
            + for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let ExtColumns::Sliced(columns) = &self.columns else {
            return None;
        };
        if !columns.late_boundary
            || columns.tensor.is_some()
            || self.round != 4
            || columns.challenges.len() != 4
            || !SLICED_LANES.is_multiple_of(R::Packing::WIDTH)
        {
            return None;
        }
        let _span = tracing::debug_span!("round_poly_late_boundary").entered();
        #[cfg(test)]
        LATE_BOUNDARY_ROUNDS.with(|rounds| rounds.set(rounds.get() + 1));
        Some(self.round_poly_planes::<S>(eq_suffix))
    }

    /// Accumulate this round's node sums one residual row at a time, straight from the planes.
    fn boundary_evals_rows(
        &self,
        eq_suffix: &Poly<EF>,
        fold: &PlaneFold<'_, R>,
    ) -> (Vec<Vec<EF>>, Vec<Vec<EF>>)
    where
        R: Algebra<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, R, R>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, R, R>>,
    {
        let width = self.width();
        let pairs = fold.words / ROW_HALVES;
        let schedule = node_schedule::<EF>(evaluated_nodes(&self.slots, self.degree(), true))
            .into_iter()
            .map(|(node, step)| (node, step.map(R::from)))
            .collect::<Vec<_>>();
        let next_columns = next_row_runs(&self.slots);
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
        let weights = eq_suffix.as_slice();
        // Every worker of a stage that reads no successor row reads the same zeros.
        let next_zeros = next_columns
            .is_empty()
            .then(|| Arc::from(R::zero_vec(width)));

        let (scratch, _) = (0..pairs)
            .into_par_iter()
            .with_min_len(rows_per_task(pairs))
            .par_fold_reduce(
                || {
                    (
                        Scratch::<R, R>::new(
                            &constraint_degrees,
                            &interaction_degrees,
                            width,
                            next_zeros.as_ref(),
                        ),
                        RowTile::new(fold.groups, width),
                    )
                },
                |(mut scratch, mut tile), pair| {
                    tile.fill(fold, pair, &next_columns);
                    for lane in 0..SLICED_LANES {
                        let s = pair * SLICED_LANES + lane;
                        tile.read_row(fold, lane, &next_columns, &mut scratch);
                        self.walk_row_nodes(
                            &mut scratch,
                            s,
                            R::from(weights[s]),
                            &schedule,
                            &next_columns,
                        );
                    }
                    (scratch, tile)
                },
                |(mut lhs, tile), (rhs, _)| {
                    lhs.constraint_evals
                        .iter_mut()
                        .zip(rhs.constraint_evals)
                        .for_each(|(lhs, rhs)| R::add_slices(lhs, &rhs));
                    lhs.interaction_evals
                        .iter_mut()
                        .zip(rhs.interaction_evals)
                        .for_each(|(lhs, rhs)| R::add_slices(lhs, &rhs));
                    (lhs, tile)
                },
            );
        (
            lower_evals(scratch.constraint_evals),
            lower_evals(scratch.interaction_evals),
        )
    }

    /// Accumulate this round's node sums one lane group at a time, straight from the planes.
    fn boundary_evals_lanes(
        &self,
        eq_suffix: &Poly<EF>,
        fold: &PlaneFold<'_, R>,
    ) -> (Vec<Vec<EF>>, Vec<Vec<EF>>)
    where
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let lanes = R::Packing::WIDTH;
        let width = self.width();
        let pairs = fold.words / ROW_HALVES;
        let round = self.lane_round();
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
        let weights = eq_suffix.as_slice();
        // Every worker of a stage that reads no successor row reads the same zeros.
        let next_zeros = round
            .next_columns
            .is_empty()
            .then(|| Arc::from(PackedRepr::<F, R>::zero_vec(width)));

        let (scratch, _) = (0..pairs)
            .into_par_iter()
            .with_min_len(rows_per_task(pairs))
            .par_fold_reduce(
                || {
                    (
                        PackedScratch::<PackedRepr<F, R>, PackedRepr<F, R>>::new(
                            &constraint_degrees,
                            &interaction_degrees,
                            width,
                            next_zeros.as_ref(),
                        ),
                        RowTile::new(fold.groups, width),
                    )
                },
                |(mut scratch, mut tile), pair| {
                    tile.fill(fold, pair, &round.next_columns);
                    for lane in (0..SLICED_LANES).step_by(lanes) {
                        let s = pair * SLICED_LANES + lane;
                        tile.read_lane_group(fold, lane, &round.next_columns, &mut scratch);
                        let eq_suffix = lane_group(|step| R::from(weights[s + step]));
                        self.walk_lane_nodes(&mut scratch, s, eq_suffix, &round);
                    }
                    (scratch, tile)
                },
                |(mut lhs, tile), (rhs, _)| {
                    lhs.constraint_evals
                        .iter_mut()
                        .zip(rhs.constraint_evals)
                        .for_each(|(lhs, rhs)| add_slice(lhs, &rhs));
                    lhs.interaction_evals
                        .iter_mut()
                        .zip(rhs.interaction_evals)
                        .for_each(|(lhs, rhs)| add_slice(lhs, &rhs));
                    (lhs, tile)
                },
            );
        (
            sum_lanes::<F, R, EF>(scratch.constraint_evals),
            sum_lanes::<F, R, EF>(scratch.interaction_evals),
        )
    }
}

impl<'air, 'data, A, F, EF> RoundStateExt<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Fold a stage off its planes into the storage the challenge-field kernels read.
    ///
    /// The columns pack into lanes while half the residual rows still fill one, as a fold does.
    pub(crate) fn unslice_packed<S>(&mut self)
    where
        S: Field,
        EF: HasSubfield<S>,
    {
        if !matches!(self.columns, ExtColumns::Sliced(_)) {
            return;
        }
        let want_packed = self.num_evals() / 2 >= F::Packing::WIDTH;
        self.unslice::<S>();
        if want_packed && let ExtColumns::Scalar(columns) = &self.columns {
            let width = F::Packing::WIDTH;
            let packed = columns
                .par_iter()
                .map(|column| {
                    let rows = column.as_slice();
                    Poly::new(
                        (0..rows.len() / width)
                            .map(|group| {
                                EF::ExtensionPacking::from_ext_fn(|lane| rows[group * width + lane])
                            })
                            .collect(),
                    )
                })
                .collect();
            self.columns = ExtColumns::Packed(packed);
        }
    }
}

#[cfg(test)]
mod tests;
