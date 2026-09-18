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

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field, HasSubfield, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::Table;

use super::{
    AirSlot, ExtColumns, InteractionCoupling, NodeStep, RoundStateBase, RoundStateExt,
    evaluated_nodes, finish_round, next_row_runs, node_schedule,
};
use crate::selectors::BoundaryEvals;
use crate::sliced::{LaneSums, SLICED_LANES, SlicedFolder, SlicedGf4, gf4_coordinates, is_gf4};

/// Rounds a stage evaluates on its planes.
const SLICED_ROUNDS: usize = 3;

/// Row variables one word's lanes span.
const LANE_VARIABLES: usize = SLICED_LANES.trailing_zeros() as usize;

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
    fn new(degrees: &[usize], prefixes: usize, width: usize, corners: usize) -> Self {
        Self {
            sums: degrees
                .iter()
                .map(|&degree| vec![R::zero_vec(degree); prefixes])
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
            sums,
            local,
            local_diff,
            next,
            next_diff,
            corners,
            poisoned,
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
        let mut boundary = BoundaryEvals::new(first, last, transition);
        let boundary_diff = BoundaryEvals::new(first_diff, last_diff, transition_diff);

        let weight = self.word_weights[word];
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
            let eval_index = if node == 0 { 0 } else { node - 1 };
            for slot in self.slots {
                if !slot
                    .enabled_families(node, self.include_node_zero)
                    .constraints
                {
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

/// Evaluate round `challenges.len()` of a stage on its planes.
///
/// # Returns
///
/// Each AIR's eq-weighted, alpha-batched constraint sums at its native nodes `0, 2, 3, ...`, or
/// `None` when an interpolation node lies outside `S` or an AIR constant poisoned a value.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, level = "debug", fields(round = challenges.len()))]
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
    let round = challenges.len();
    let num_vars = trace.num_vars;
    debug_assert!(round < trace.rounds && tau.len() == num_vars);

    let nodes = (0..=degree)
        .map(|node| gf4_coordinates(EF::interpolation_node(node).as_subfield()?))
        .collect::<Option<Vec<_>>>()?;
    let include_node_zero = round > 0;
    let schedule = node_schedule::<EF>(evaluated_nodes(slots, degree, include_node_zero))
        .into_iter()
        .map(|(node, step)| Some((node, step_coordinates::<S, EF>(step)?)))
        .collect::<Option<Vec<_>>>()?;

    // Row variables split three ways: prefix and t, the words of a corner block, and the lanes.
    let lane_point = &tau[num_vars - LANE_VARIABLES..];
    let word_point = &tau[round + 1..num_vars - LANE_VARIABLES];
    let lane_weights = Poly::new_from_point(lane_point, EF::ONE);
    let word_weights = Poly::new_from_point(word_point, EF::ONE);
    // The row weights factor as word weight times lane weight, the table the caller holds.
    debug_assert!(
        eq_suffix
            .as_slice()
            .iter()
            .enumerate()
            .all(|(row, &weight)| {
                weight
                    == word_weights.as_slice()[row / SLICED_LANES]
                        * lane_weights.as_slice()[row % SLICED_LANES]
            })
    );
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
            || SlicedScratch::new(&degrees, prefixes, trace.width, corners),
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

    // Interpolate each prefix sum at the challenges already drawn.
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
        scratch
            .sums
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
        let rounds = SLICED_ROUNDS.min(num_vars - LANE_VARIABLES);

        // Every column in merged-buffer order: main, preprocessed, periodic, AIR by AIR.
        let mut tables: Vec<&Table<F>> = Vec::new();
        for slot in &self.slots {
            tables.push(self.tables[slot.stage_index]);
            tables.extend(self.preprocessed[slot.stage_index]);
            tables.extend(self.periodic[slot.stage_index].as_ref());
        }
        let columns = tables
            .iter()
            .flat_map(|table| table.iter_polys())
            .collect::<Vec<_>>();
        let width = columns.len();
        let mut is_successor = vec![false; width];
        for column in next_row_runs(&self.slots).into_iter().flatten() {
            is_successor[column] = true;
        }

        // Pack each column on its own, successor planes included, reading it contiguously.
        let packed = columns
            .par_iter()
            .zip(&is_successor)
            .map(|(column, &is_successor)| {
                let column = column.as_chunks::<SLICED_LANES>().0;
                let planes = column
                    .iter()
                    .map(pack_word::<F, S>)
                    .collect::<Option<Vec<_>>>()?;
                let successors = if is_successor {
                    // The last row repeats itself; every other row reads the next one.
                    let last = planes[planes.len() - 1];
                    let last_carry = (last[0] >> 63 == 1, last[1] >> 63 == 1);
                    let carries = column[1..]
                        .iter()
                        .map(|next| next[0].as_subfield().and_then(gf4_coordinates))
                        .chain([Some(last_carry)]);
                    planes
                        .iter()
                        .zip(carries)
                        .map(|(&planes, carry)| Some(successor_word(planes, carry?)))
                        .collect::<Option<Vec<_>>>()?
                } else {
                    Vec::new()
                };
                Some((planes, successors))
            })
            .collect::<Option<Vec<_>>>()?;

        // Lay the words out word by word, every column of a word side by side.
        let words = 1 << (num_vars - LANE_VARIABLES);
        let mut cells = vec![[0; 2]; words * width];
        let mut successors = vec![[0; 2]; words * width];
        cells
            .par_chunks_mut(width)
            .zip(successors.par_chunks_mut(width))
            .enumerate()
            .for_each(|(word, (cells, successors))| {
                for ((cell, successor), (planes, next)) in
                    cells.iter_mut().zip(successors.iter_mut()).zip(&packed)
                {
                    *cell = planes[word];
                    if let Some(&next) = next.get(word) {
                        *successor = next;
                    }
                }
            });

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
        self.subfield_schedule::<S>()?;
        let trace = self.sliced_trace::<S>()?;
        self.fits_subfield = true;
        let alpha_powers = self
            .alpha_powers
            .iter()
            .map(|powers| powers.iter().map(|&power| R::from(power)).collect())
            .collect::<Vec<Vec<R>>>();
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
        self.sliced = Some(trace);

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
        let trace = self
            .sliced
            .take()
            .expect("the first round ran on the planes");
        self.fold_claims(r);
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
            next_tail: R::zero_vec(trace.width),
            columns: ExtColumns::Sliced(SlicedColumns {
                trace,
                challenges: vec![r],
            }),
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

/// A sliced stage's planes and the challenges bound so far.
pub(super) struct SlicedColumns<EF> {
    /// The stage's planes, none of whose columns has folded yet.
    trace: SlicedTrace,
    /// Every challenge bound so far, first variable first.
    challenges: Vec<EF>,
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

/// Subset sums of up to eight weights, indexed by the byte of the weights they include.
fn subset_sums<R: Field>(weights: &[R]) -> Vec<R> {
    let mut sums = R::zero_vec(1 << weights.len());
    for mask in 1..sums.len() {
        sums[mask] = sums[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
    }
    sums
}

impl<'air, 'data, A, F, EF, R> RoundStateExt<'air, 'data, A, F, EF, R>
where
    F: Field,
    EF: ExtensionField<F> + From<R>,
    R: Field + From<EF>,
{
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
        if columns.challenges.len() == columns.trace.rounds {
            return None;
        }
        debug_assert_eq!(columns.challenges.len(), self.round);
        let evals = sliced_round::<A, F, EF, S, R>(
            eq_suffix,
            &columns.trace,
            &self.slots,
            &self.public_values,
            &self.alpha_powers,
            self.tau.as_slice(),
            &columns.challenges,
            self.degree(),
        )?;

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
        columns.challenges.push(r);
        self.fold_claims(r);
        self.boundary.apply(R::from(r));
        self.round += 1;
        true
    }

    /// Fold a stage off its planes into scalar columns in `R`, at every challenge bound so far.
    ///
    /// Each residual row of a column combines the cells the bound variables range over:
    ///
    /// ```text
    ///     column(x) = sum_b eq(r, b) * cell(b, x)       b in {0, 1}^k
    /// ```
    ///
    /// A cell is `low + high * g`, so each row takes byte-indexed subset sums of `eq(r, .)`,
    /// one lookup per plane per group of eight `b`. The repeat-last tails read the successor
    /// planes at the last residual row the same way.
    ///
    /// Does nothing when the stage is not on its planes.
    pub(crate) fn unslice<S>(&mut self)
    where
        S: Field,
        EF: HasSubfield<S>,
    {
        let (trace, challenges) =
            match core::mem::replace(&mut self.columns, ExtColumns::Scalar(Vec::new())) {
                ExtColumns::Sliced(SlicedColumns { trace, challenges }) => (trace, challenges),
                columns => {
                    self.columns = columns;
                    return;
                }
            };
        let _span = tracing::debug_span!("unslice").entered();
        let generator = R::from(EF::from(S::GENERATOR));
        let weights = Poly::new_from_point(&challenges, EF::ONE)
            .as_slice()
            .iter()
            .map(|&weight| R::from(weight))
            .collect::<Vec<_>>();
        let low_sums = weights.chunks(8).map(subset_sums).collect::<Vec<_>>();
        let high_sums = low_sums
            .iter()
            .map(|sums| sums.iter().map(|&sum| generator * sum).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let corners = weights.len();
        let words = (trace.cells.len() / trace.width) / corners;
        let width = trace.width;

        // The value at every residual row of one word, from the corner words of both planes.
        let fold_word = |planes: &[[u64; 2]], column: usize, word: usize, out: &mut [R]| {
            let corner_words = |plane: usize| {
                let mut words_of_plane = [0; 1 << SLICED_ROUNDS];
                for (corner, value) in words_of_plane[..corners].iter_mut().enumerate() {
                    *value = planes[(corner * words + word) * width + column][plane];
                }
                words_of_plane
            };
            let (low, high) = (corner_words(0), corner_words(1));
            out.fill(R::ZERO);
            for (group, (low, high)) in low[..corners]
                .chunks(8)
                .zip(high[..corners].chunks(8))
                .enumerate()
            {
                let (low, high) = (lane_masks(low), lane_masks(high));
                for (value, (&low, &high)) in out.iter_mut().zip(low.iter().zip(&high)) {
                    *value +=
                        low_sums[group][usize::from(low)] + high_sums[group][usize::from(high)];
                }
            }
        };

        let scalar = (0..width)
            .into_par_iter()
            .map(|column| {
                let mut values = R::zero_vec(words * SLICED_LANES);
                for (word, out) in values
                    .as_chunks_mut::<SLICED_LANES>()
                    .0
                    .iter_mut()
                    .enumerate()
                {
                    fold_word(&trace.cells, column, word, out);
                }
                Poly::new(values)
            })
            .collect();

        // Each tail is the successor column at the last residual row.
        let mut last = [R::ZERO; SLICED_LANES];
        for run in next_row_runs(&self.slots) {
            for column in run {
                fold_word(&trace.successors, column, words - 1, &mut last);
                self.next_tail[column] = last[SLICED_LANES - 1];
            }
        }
        self.columns = ExtColumns::Scalar(scalar);
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
