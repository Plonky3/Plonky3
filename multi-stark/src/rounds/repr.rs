//! Later zerocheck rounds computed in a field isomorphic to the challenge field.
//!
//! Two representations of one field can multiply at very different costs:
//!
//! ```text
//!     GF(2^128), tower basis      : three changes of basis around one carryless multiply
//!     GF(2^128), polynomial basis : one carryless multiply
//! ```
//!
//! A stage bound into `R` keeps every value its row loop reads in `R`.
//! The claims, the zerocheck point, and the interpolators stay in the challenge field.
//! Each round's per-node sums cross back into it once.
//!
//! The conversions are field isomorphisms, so every round polynomial is the challenge field's.

use alloc::vec::Vec;
use core::ops::Range;

use p3_air::{Air, BaseAir};
use p3_field::{Algebra, ExtensionField, Field, HasSubfield, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::ColumnView;

use super::subfield::PARALLEL_FOLD_CELLS;
use super::{
    ExtColumns, InteractionCoupling, NodeStep, PackedScratch, RoundStateBase, RoundStateExt,
    add_slice, evaluate_air_families, evaluated_nodes, finish_round, next_row_runs, node_schedule,
    rows_per_task,
};
use crate::folder::{InteractionMultilinearFolder, MultilinearFolder};
use crate::packed_ext::{PackedExt, PackedRepr};
use crate::selectors::BoundaryEvals;

/// Entries of a subfield fold table, one per value of a byte.
const TABLE_ENTRIES: usize = 1 << u8::BITS;

/// One lane group of the representation field, holding the value each lane reads.
#[inline]
pub(super) fn lane_group<F, R: Field>(value: impl FnMut(usize) -> R) -> PackedRepr<F, R> {
    PackedExt::new(R::Packing::from_fn(value))
}

/// Carry per-node sums from lane groups back into the challenge field.
///
/// Each lane of a sum covers residual rows of its own, so a node's value is the sum of its lanes.
pub(super) fn sum_lanes<F, R: Field, EF: From<R>>(
    evals: Vec<Vec<PackedRepr<F, R>>>,
) -> Vec<Vec<EF>> {
    evals
        .into_iter()
        .map(|evals| {
            evals
                .into_iter()
                .map(|value| EF::from(value.0.as_slice().iter().copied().sum::<R>()))
                .collect()
        })
        .collect()
}

/// The round-wide values a lane-group row loop reads, the same in every lane.
pub(super) struct LaneRound<P> {
    /// The interpolation nodes, and the step from each to the next.
    schedule: Vec<(usize, NodeStep<P>)>,
    /// The column runs any AIR reads on the next row.
    pub(super) next_columns: Vec<Range<usize>>,
    /// The constraint batching challenge.
    alpha: P,
    /// Its powers, per AIR of the stage.
    alpha_powers: Vec<Vec<P>>,
    /// The lookup coupling of the stage.
    coupling: InteractionCoupling<P>,
}

/// The table entry of a trace-field element: the first byte of its serialization.
#[inline]
fn table_index<F: Field>(value: F) -> usize {
    value.into_bytes().into_iter().next().map_or(0, usize::from)
}

/// The fold of every cell pair of a stage that fits the subfield `S`, tabulated in `R`.
///
/// Every cell and every difference of two cells lies in `S`, so two lookups replace the product:
///
/// ```text
///     R::from(lo + r * (hi - lo)) = low[index(lo)] + scaled[index(hi - lo)]
///
///     low[index(s)]    = R::from(s)
///     scaled[index(s)] = R::from(r * s)          for every s in S
/// ```
///
/// A table exists only when the index tells every element of `S` apart.
struct SubfieldFoldTables<R> {
    /// Each element of `S`, carried into `R`.
    low: [R; TABLE_ENTRIES],
    /// Each element of `S` scaled by the challenge, carried into `R`.
    scaled: [R; TABLE_ENTRIES],
}

impl<R: Field> SubfieldFoldTables<R> {
    /// Tabulate the fold at `r` over every element of `S`.
    ///
    /// The elements are zero and the powers of the generator of `S`.
    ///
    /// # Returns
    ///
    /// `None` when two elements of `S` share an index, or when those elements are not all of `S`.
    fn new<S, F, EF>(r: EF) -> Option<Self>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: ExtensionField<F>,
        R: From<EF>,
    {
        let mut tables = Self {
            low: [R::ZERO; TABLE_ENTRIES],
            scaled: [R::ZERO; TABLE_ENTRIES],
        };
        let mut filled = [false; TABLE_ENTRIES];
        let mut insert = |element: S| {
            let cell = F::from(element);
            let index = table_index(cell);
            if core::mem::replace(&mut filled[index], true) {
                return false;
            }
            let value = EF::from(cell);
            tables.low[index] = R::from(value);
            tables.scaled[index] = R::from(r * value);
            true
        };

        // A repeated index ends the walk, so it takes at most one step per entry.
        let mut count = 1_u64;
        if !insert(S::ZERO) {
            return None;
        }
        let mut power = S::ONE;
        loop {
            if !insert(power) {
                return None;
            }
            count += 1;
            power *= S::GENERATOR;
            if power == S::ONE {
                break;
            }
        }
        (S::order().to_u64_digits() == [count]).then_some(tables)
    }
}

impl<'air, 'data, A, F, EF> RoundStateBase<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Bind the first variable at `r`, folding every column straight into `R`.
    ///
    /// Each pair of cells crosses into `R` and folds there:
    ///
    /// ```text
    ///     R::from(lo) + R::from(r) * (R::from(hi) - R::from(lo))
    /// ```
    ///
    /// The repeat-last tails fold in the challenge field and cross afterwards.
    /// The alpha powers, the lookup coefficients, and the selector prefix cross once here.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_into<R>(self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        R: Field + From<EF>,
    {
        let challenge = R::from(r);
        self.fold_pairs_into(r, |lo, hi| {
            let lo = R::from(EF::from(lo));
            lo + challenge * (R::from(EF::from(hi)) - lo)
        })
    }

    /// Bind the first variable at `r` into `R` without a product per cell.
    ///
    /// Every pair of cells folds through two table lookups, see [`SubfieldFoldTables`].
    /// When the tables cannot be built, the pairs fold as in [`Self::fold_into`].
    ///
    /// # Precondition
    ///
    /// The first round found this stage to fit `S`, as [`Self::fits_subfield`] reports.
    /// Every cell then lies in `S`, so every lookup lands on a filled entry.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_subfield_into<S, R>(self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        S: Field,
        F: HasSubfield<S>,
        R: Field + From<EF>,
    {
        debug_assert!(self.fits_subfield());
        let Some(tables) = SubfieldFoldTables::<R>::new::<S, F, EF>(r) else {
            tracing::debug!("the subfield's elements share a fold table index");
            return self.fold_into(r);
        };
        self.fold_pairs_into(r, |lo, hi| {
            tables.low[table_index(lo)] + tables.scaled[table_index(hi - lo)]
        })
    }

    /// Bind the first variable at `r`, folding each pair of cells into `R` with `fold_pair`.
    #[allow(clippy::option_if_let_else)]
    fn fold_pairs_into<R, U>(
        mut self,
        r: EF,
        fold_pair: U,
    ) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        R: Field + From<EF>,
        U: Fn(F, F) -> R + Sync,
    {
        let next_tail = self.fold_claims_and_tails(r);

        // Columns already fold in parallel, so a short column folds on its own thread.
        let columns = self.fold_each_column(|column: ColumnView<'_, F>| {
            let half = column.len() / 2;
            if let Some(values) = column.as_dense() {
                let (lo, hi) = values.split_at(half);
                let fold_cells = |(&lo, &hi): (&F, &F)| fold_pair(lo, hi);
                Poly::new(if values.len() < PARALLEL_FOLD_CELLS {
                    lo.iter().zip(hi).map(fold_cells).collect()
                } else {
                    lo.par_iter().zip(hi).map(fold_cells).collect()
                })
            } else {
                Poly::new(
                    (0..half)
                        .map(|row| fold_pair(column.value(row), column.value(row + half)))
                        .collect(),
                )
            }
        });

        let lift = |values: &[EF]| {
            values
                .iter()
                .map(|&value| R::from(value))
                .collect::<Vec<_>>()
        };
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
            columns: ExtColumns::Scalar(columns),
            next_tail: lift(&next_tail),
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

impl<'air, 'data, A, F, EF, R> RoundStateExt<'air, 'data, A, F, EF, R>
where
    F: Field,
    EF: ExtensionField<F> + From<R>,
    R: Field + From<EF>,
{
    /// Evaluate this round's polynomial with every row value in `R`.
    ///
    /// The eq weights cross into `R` here, once per round, split across threads past the length
    /// at which a column fold would split.
    ///
    /// A lane group of `R` covers several residual rows, so the round runs one constraint pass
    /// per group while the residual rows still fill one. Rows too few to fill a group, and a
    /// target whose packing holds one element, take one row per pass.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_repr(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, R, R>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, R, R>>
            + for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let lanes = R::Packing::WIDTH;
        if lanes > 1 && self.num_evals() / 2 >= lanes {
            return self.round_poly_repr_lanes(eq_suffix);
        }

        let weights = eq_suffix.as_slice();
        let lift = |&weight: &EF| R::from(weight);
        let eq_suffix = Poly::new(if weights.len() < PARALLEL_FOLD_CELLS {
            weights.iter().map(lift).collect()
        } else {
            weights.par_iter().map(lift).collect()
        });
        self.round_poly_unpacked(&eq_suffix)
    }

    /// Evaluate this round's polynomial one lane group of residual rows at a time.
    ///
    /// The columns stay one element per row, so a lane group gathers the consecutive rows it
    /// covers:
    ///
    /// ```text
    ///     rows   : x_0  x_1  x_2  x_3 ...
    ///     lanes  : |------- one lane group -------|
    /// ```
    ///
    /// Every lane carries the eq weight of its own row, so the per-node sums stay lane-wise.
    /// The lanes of each sum are added, and that sum crosses into the challenge field, once the
    /// fold is over.
    ///
    /// # Panics
    ///
    /// Panics if the residual row pairs do not fill a whole number of lane groups.
    #[tracing::instrument(skip_all, level = "debug")]
    fn round_poly_repr_lanes(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let lanes = R::Packing::WIDTH;
        let width = self.width();
        let half = self.num_evals() / 2;
        let groups = half / lanes;
        assert_eq!(
            groups * lanes,
            half,
            "residual rows must fill whole lane groups"
        );

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

        // A lane group's eq weights cross into `R` where its rows are read, one group at a time.
        let weights = eq_suffix.as_slice();
        let scratch = (0..groups)
            .into_par_iter()
            .with_min_len(rows_per_task(groups))
            .par_fold_reduce(
                || {
                    PackedScratch::<PackedRepr<F, R>, PackedRepr<F, R>>::new(
                        &constraint_degrees,
                        &interaction_degrees,
                        width,
                    )
                },
                |scratch, group| {
                    let eq_suffix = lane_group(|lane| R::from(weights[group * lanes + lane]));
                    self.accumulate_lanes(scratch, group * lanes, eq_suffix, &round)
                },
                |mut lhs, rhs| {
                    lhs.constraint_evals
                        .iter_mut()
                        .zip(rhs.constraint_evals)
                        .for_each(|(lhs, rhs)| add_slice(lhs, &rhs));
                    lhs.interaction_evals
                        .iter_mut()
                        .zip(rhs.interaction_evals)
                        .for_each(|(lhs, rhs)| add_slice(lhs, &rhs));
                    lhs
                },
            );
        finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.lookup_scale,
            &sum_lanes::<F, R, EF>(scratch.constraint_evals),
            &sum_lanes::<F, R, EF>(scratch.interaction_evals),
            self.tau.as_slice()[self.round],
        )
    }

    /// The round-wide values every lane group of this round reads, broadcast to all lanes.
    pub(super) fn lane_round(&self) -> LaneRound<PackedRepr<F, R>>
    where
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
    {
        let broadcast = |value: R| PackedExt::new(R::Packing::broadcast(value));
        LaneRound {
            schedule: node_schedule::<EF>(evaluated_nodes(&self.slots, self.degree(), true))
                .into_iter()
                .map(|(node, step)| (node, step.map(|step| broadcast(R::from(step)))))
                .collect(),
            next_columns: next_row_runs(&self.slots),
            alpha: broadcast(self.alpha),
            alpha_powers: self
                .alpha_powers
                .iter()
                .map(|powers| powers.iter().copied().map(broadcast).collect())
                .collect(),
            coupling: InteractionCoupling {
                links: self
                    .coupling
                    .links
                    .iter()
                    .map(|link| link.map(broadcast))
                    .collect(),
                theta_beta_powers: self
                    .coupling
                    .theta_beta_powers
                    .iter()
                    .copied()
                    .map(broadcast)
                    .collect(),
            },
        }
    }

    /// Add one lane group's eq-weighted evaluations at every node of the round to `scratch`.
    ///
    /// Lane `j` reads residual row `s + j`.
    fn accumulate_lanes(
        &self,
        mut scratch: PackedScratch<PackedRepr<F, R>, PackedRepr<F, R>>,
        s: usize,
        eq_suffix: PackedRepr<F, R>,
        round: &LaneRound<PackedRepr<F, R>>,
    ) -> PackedScratch<PackedRepr<F, R>, PackedRepr<F, R>>
    where
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let columns = self.columns.as_scalar();
        for ((local, local_delta), column) in scratch
            .local_point
            .iter_mut()
            .zip(scratch.local_diff.iter_mut())
            .zip(columns)
        {
            let column = column.as_slice();
            let local_lo = lane_group(|lane| column[s + lane]);
            let local_hi = lane_group(|lane| column[s + half + lane]);
            *local = local_lo;
            *local_delta = local_hi - local_lo;
        }
        for run in &round.next_columns {
            for (((next, next_delta), column), next_tail) in scratch.next_point[run.clone()]
                .iter_mut()
                .zip(scratch.next_diff[run.clone()].iter_mut())
                .zip(&columns[run.clone()])
                .zip(&self.next_tail[run.clone()])
            {
                let column = column.as_slice();
                let next_lo = lane_group(|lane| column[s + lane + 1]);
                let next_hi = lane_group(|lane| {
                    // Past the last residual row, the repeat-last tail stands in.
                    let row = s + half + lane + 1;
                    if row < num_evals {
                        column[row]
                    } else {
                        *next_tail
                    }
                });
                *next = next_lo;
                *next_delta = next_hi - next_lo;
            }
        }

        self.walk_lane_nodes(&mut scratch, s, eq_suffix, round);
        scratch
    }

    /// Step one lane group through every node of the round, adding its evaluations there.
    ///
    /// The group's values at node zero, and the step from each node to the next, are already in
    /// `scratch`.
    ///
    /// Never inlined: the AIR evaluation needs a large stack frame, which inside the parallel
    /// fold would be reserved again at every level of Rayon's recursive split.
    #[inline(never)]
    pub(super) fn walk_lane_nodes(
        &self,
        scratch: &mut PackedScratch<PackedRepr<F, R>, PackedRepr<F, R>>,
        s: usize,
        eq_suffix: PackedRepr<F, R>,
        round: &LaneRound<PackedRepr<F, R>>,
    ) where
        R: Algebra<F>,
        R::Packing: Algebra<F::Packing>,
        A: for<'b> Air<MultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, PackedRepr<F, R>, PackedRepr<F, R>>>,
    {
        let num_evals = self.num_evals();
        let half = num_evals / 2;
        let (boundary, boundary_diff) = BoundaryEvals::row_pair_with_prefix_lanes::<R::Packing>(
            s,
            half,
            num_evals,
            self.boundary,
        );
        let lift = |evals: BoundaryEvals<R::Packing>| {
            BoundaryEvals::new(
                PackedExt::new(evals.first),
                PackedExt::new(evals.last),
                PackedExt::new(evals.transition),
            )
        };
        let (mut boundary, boundary_diff) = (lift(boundary), lift(boundary_diff));

        for &(node, step) in &round.schedule {
            match step {
                NodeStep::Unit(count) => {
                    for _ in 0..count {
                        scratch.add_diffs(&round.next_columns);
                        boundary += boundary_diff;
                    }
                }
                NodeStep::Scaled(step) => {
                    scratch.add_scaled_diffs(step, &round.next_columns);
                    boundary.add_scaled(boundary_diff, step);
                }
            }
            for slot in &self.slots {
                let enabled = slot.enabled_families(node, true);
                if !enabled.constraints && enabled.interaction.is_none() {
                    continue;
                }
                let folder = MultilinearFolder::new(
                    &scratch.local_point[slot.main_offset..slot.main_offset + slot.main_width],
                    &scratch.next_point[slot.main_offset..slot.main_offset + slot.main_width],
                    boundary,
                    self.public_values[slot.stage_index],
                    round.alpha,
                )
                .with_alpha_powers(&round.alpha_powers[slot.stage_index])
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
                let evaluations = evaluate_air_families(folder, &round.coupling, enabled, slot.air);
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
    }

    /// Bind the next variable at `r`, folding the scalar columns in `R`.
    ///
    /// # Panics
    ///
    /// Panics if the columns are packed, which a stage folded into `R` never is.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_repr(&mut self, r: EF) {
        self.fold_claims_and_tails(r);

        let r = R::from(r);
        match &mut self.columns {
            ExtColumns::Scalar(cols) => cols
                .par_iter_mut()
                .for_each(|col| col.fix_prefix_var_mut(r)),
            ExtColumns::Packed(_) | ExtColumns::Sliced(_) => {
                unreachable!("a stage folded into R keeps scalar columns")
            }
        }

        self.boundary.apply(r);
        self.round += 1;
    }
}

#[cfg(test)]
mod tests;
