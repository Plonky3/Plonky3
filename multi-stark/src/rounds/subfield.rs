//! First-round kernel that evaluates the AIRs inside a small subfield.
//!
//! A binary trace often holds only bits, and the first round's interpolation steps can lie in a
//! small subfield too:
//!
//! ```text
//!     cells                        : bits, so in GF(2) inside GF(4)
//!     degree-3 first-round schedule: 0 -(X_0)-> node 2 -(1)-> node 3, steps in GF(4)
//! ```
//!
//! Each row then evaluates every expression in the subfield.
//! Only the alpha-batched value is lifted, and it meets one general product with its eq weight.
//! Every value equals the generic kernel's, so the round polynomial does too.

use alloc::vec::Vec;
use core::ops::Range;

use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field, HasSubfield, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::Table;

use super::{
    NodeStep, RoundStateBase, add_scaled_slice, add_slice, evaluated_nodes, finish_round,
    next_row_runs, node_schedule,
};
use crate::folder::MultilinearFolder;
use crate::selectors::BoundaryEvals;
use crate::subfield::{SubfieldAcc, SubfieldVar};

/// Cells of one column the eligibility scan tests at once.
///
/// The scan stops at the first chunk holding a cell outside the subfield.
const SCAN_CHUNK_CELLS: usize = 1 << 16;

/// Powers of the subfield generator the embedding check compares at most.
const MAX_EMBEDDING_POWERS: usize = 1 << 8;

/// Residual rows one parallel task of the subfield kernel evaluates.
///
/// A subfield row is cheap enough that scheduling one task per row would dominate it.
const ROWS_PER_TASK: usize = 16;

/// What every row of one subfield pass shares.
struct SubfieldRows<'a, S, EF> {
    /// Distance from a low residual row to its matching high row.
    half: usize,
    /// Interpolation nodes to evaluate, each with the step that reaches it inside `S`.
    schedule: &'a [(usize, NodeStep<S>)],
    /// Every successor column run of the stage, in merged-buffer order.
    next_columns: Vec<Range<usize>>,
    /// The constraint-batching scalar.
    alpha: SubfieldAcc<EF, S>,
    /// Descending alpha powers for each AIR, one per constraint the folder batches.
    alpha_powers: Vec<Vec<SubfieldAcc<EF, S>>>,
}

/// Per-worker scratch for the subfield first-round kernel.
struct SubfieldScratch<F, S, EF> {
    /// Unweighted ordinary-constraint evaluations for each AIR at its native nodes.
    constraint_evals: Vec<Vec<EF>>,
    /// Current-row value of each column at the active interpolation node.
    local_point: Vec<SubfieldVar<F, S>>,
    /// Difference between the high and low current-row values.
    local_diff: Vec<SubfieldVar<F, S>>,
    /// Successor-row value of each column at the active interpolation node.
    ///
    /// Zero for every column no AIR reads on the next row.
    next_point: Vec<SubfieldVar<F, S>>,
    /// Difference between the high and low successor-row values.
    next_diff: Vec<SubfieldVar<F, S>>,
    /// Whether any batched value this worker evaluated was poisoned.
    poisoned: bool,
}

impl<F, S: Field, EF: Field> SubfieldScratch<F, S, EF> {
    fn new(constraint_degrees: &[usize], width: usize) -> Self {
        Self {
            constraint_evals: constraint_degrees
                .iter()
                .copied()
                .map(EF::zero_vec)
                .collect(),
            local_point: SubfieldVar::zero_vec(width),
            local_diff: SubfieldVar::zero_vec(width),
            next_point: SubfieldVar::zero_vec(width),
            next_diff: SubfieldVar::zero_vec(width),
            poisoned: false,
        }
    }

    /// Load one column group's current-row pairs at `row` into the buffer from `offset` on.
    ///
    /// Each column contributes its low value and its step to row `row + half`.
    fn fill_local(&mut self, offset: usize, table: &Table<F>, row: usize, half: usize)
    where
        F: HasSubfield<S>,
    {
        let end = offset + table.num_polys();
        for ((local, local_delta), column) in self.local_point[offset..end]
            .iter_mut()
            .zip(self.local_diff[offset..end].iter_mut())
            .zip(table.iter_polys())
        {
            let local_lo = SubfieldVar::narrow(column[row]);
            *local = local_lo;
            *local_delta = SubfieldVar::narrow(column[row + half]) - local_lo;
        }
    }

    /// Load the successor pairs at `row` of the columns inside `runs`.
    ///
    /// The group's first column sits at `offset`. Past the last row, the last row repeats.
    fn fill_next(
        &mut self,
        offset: usize,
        table: &Table<F>,
        runs: &[Range<usize>],
        row: usize,
        half: usize,
    ) where
        F: HasSubfield<S>,
    {
        let height = 2 * half;
        for run in runs {
            for ((next, next_delta), column) in self.next_point[run.clone()]
                .iter_mut()
                .zip(self.next_diff[run.clone()].iter_mut())
                .zip(table.iter_polys().skip(run.start - offset))
            {
                let next_lo = SubfieldVar::narrow(column[row + 1]);
                let next_hi = column[(row + half + 1).min(height - 1)];
                *next = next_lo;
                *next_delta = SubfieldVar::narrow(next_hi) - next_lo;
            }
        }
    }

    /// Step every current-row column, and the successor columns inside `next_columns`.
    fn add_diffs(&mut self, next_columns: &[Range<usize>]) {
        add_slice(&mut self.local_point, &self.local_diff);
        for run in next_columns {
            add_slice(
                &mut self.next_point[run.clone()],
                &self.next_diff[run.clone()],
            );
        }
    }

    /// Scaled twin of [`Self::add_diffs`].
    fn add_scaled_diffs(&mut self, step: SubfieldVar<F, S>, next_columns: &[Range<usize>]) {
        add_scaled_slice(&mut self.local_point, &self.local_diff, step);
        for run in next_columns {
            add_scaled_slice(
                &mut self.next_point[run.clone()],
                &self.next_diff[run.clone()],
                step,
            );
        }
    }

    /// Add another worker's sums into this one, poison included.
    fn merge(mut self, other: Self) -> Self {
        self.constraint_evals
            .iter_mut()
            .zip(other.constraint_evals)
            .for_each(|(lhs, rhs)| EF::add_slices(lhs, &rhs));
        self.poisoned |= other.poisoned;
        self
    }
}

impl<A, F, EF> RoundStateBase<'_, '_, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Evaluate the first round polynomial inside the subfield `S`, when the stage fits it.
    ///
    /// # Returns
    ///
    /// - `Some`: exactly what [`Self::round_poly`] returns.
    /// - `None`: the stage does not fit `S`, or an AIR constant outside `S` poisoned a row.
    ///   No state has changed, so the caller runs the generic kernel instead.
    ///
    /// A poisoned row usually shows on the first row, so that row is probed alone first.
    /// The full pass still checks every row it evaluates.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_subfield<S>(&mut self, eq_suffix: &Poly<EF>) -> Option<Vec<EF>>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S>,
        A: for<'b> Air<MultilinearFolder<'b, F, SubfieldVar<F, S>, SubfieldAcc<EF, S>>>,
    {
        let schedule = self.subfield_schedule::<S>()?;
        let eq_suffix = eq_suffix.as_slice();
        if self.subfield_pass(&eq_suffix[..1], &schedule).poisoned {
            return fall_back("an AIR constant outside the subfield reached the first row");
        }
        let scratch = self.subfield_pass(eq_suffix, &schedule);
        if scratch.poisoned {
            return fall_back("an AIR constant outside the subfield reached a row");
        }

        // A subfield stage declares no lookup, so it has no lookup group to fill.
        Some(finish_round(
            &mut self.constraint_groups,
            &mut self.interaction_groups,
            &self.betas,
            self.eta,
            &scratch.constraint_evals,
            &[],
            self.tau.as_slice()[0],
        ))
    }

    /// The first-round node schedule with every step inside `S`, if the stage fits `S`.
    ///
    /// The stage fits when all of these hold, checked cheapest first:
    ///
    /// - no AIR declares a lookup, since the lookup links are not evaluated in `S`;
    /// - `EF` embeds `S` the way it embeds `F`'s copy of `S`, so both lifts agree;
    /// - every interpolation step of the first round lies in `S`;
    /// - every public value lies in `S`;
    /// - every periodic value lies in `S`, read from the AIR's period vectors;
    /// - every main and preprocessed cell lies in `S`.
    ///
    /// AIR constants are not checked here: they reach the rows as poison instead.
    /// Each failed condition emits a `debug` event naming it.
    #[tracing::instrument(skip_all, level = "debug")]
    fn subfield_schedule<S>(&self) -> Option<Vec<(usize, NodeStep<S>)>>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S>,
    {
        if self.slots.iter().any(|slot| slot.interaction.is_some()) {
            return fall_back("an AIR declares a lookup");
        }

        if !embeddings_agree::<S, F, EF>() {
            return fall_back("the challenge field embeds the subfield differently");
        }

        let Some(schedule) = node_schedule::<F>(evaluated_nodes(&self.slots, self.degree(), false))
            .into_iter()
            .map(|(node, step)| {
                let step = match step {
                    NodeStep::Unit(count) => NodeStep::Unit(count),
                    NodeStep::Scaled(step) => NodeStep::Scaled(step.as_subfield()?),
                };
                Some((node, step))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return fall_back("an interpolation step lies outside the subfield");
        };

        if !self
            .public_values
            .iter()
            .all(|values| F::all_in_subfield(values))
        {
            return fall_back("a public value lies outside the subfield");
        }

        // A materialized periodic column repeats its period vector, so the vector holds every
        // value the column does.
        if !self.slots.iter().all(|slot| {
            slot.air
                .periodic_columns()
                .iter()
                .all(|period| F::all_in_subfield(period))
        }) {
            return fall_back("a periodic value lies outside the subfield");
        }

        let columns = self
            .tables
            .iter()
            .copied()
            .chain(self.preprocessed.iter().flatten().copied())
            .flat_map(Table::iter_polys)
            .collect::<Vec<_>>();
        if !columns.par_iter().all(|column| {
            column
                .par_chunks(SCAN_CHUNK_CELLS)
                .all(|chunk| F::all_in_subfield(chunk))
        }) {
            return fall_back("a main or preprocessed cell lies outside the subfield");
        }
        Some(schedule)
    }

    /// Sum the eq-weighted constraint values of the leading residual rows inside `S`.
    ///
    /// Row `i` is weighted by `eq_suffix[i]`, so a prefix of the eq table evaluates a prefix of
    /// the rows. Every cell narrows into `S` on the way in, poisoning the rows it reaches when
    /// it does not fit.
    fn subfield_pass<S>(
        &self,
        eq_suffix: &[EF],
        schedule: &[(usize, NodeStep<S>)],
    ) -> SubfieldScratch<F, S, EF>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S>,
        A: for<'b> Air<MultilinearFolder<'b, F, SubfieldVar<F, S>, SubfieldAcc<EF, S>>>,
    {
        let rows = SubfieldRows {
            half: self.num_evals() / 2,
            schedule,
            next_columns: next_row_runs(&self.slots),
            alpha: SubfieldAcc::new(self.alpha),
            alpha_powers: self
                .alpha_powers
                .iter()
                .map(|powers| powers.iter().copied().map(SubfieldAcc::new).collect())
                .collect(),
        };
        let width = self.total_width();
        let constraint_degrees = self
            .slots
            .iter()
            .map(|slot| slot.constraint_degree)
            .collect::<Vec<_>>();

        eq_suffix
            .par_chunks(ROWS_PER_TASK)
            .enumerate()
            .par_fold_reduce(
                || SubfieldScratch::new(&constraint_degrees, width),
                |mut scratch, (task, eq_suffix)| {
                    for (offset, &eq) in eq_suffix.iter().enumerate() {
                        self.subfield_row(&rows, task * ROWS_PER_TASK + offset, eq, &mut scratch);
                    }
                    scratch
                },
                SubfieldScratch::merge,
            )
    }

    /// Add one residual row's eq-weighted constraint values at every scheduled node.
    fn subfield_row<S>(
        &self,
        rows: &SubfieldRows<'_, S, EF>,
        row: usize,
        eq: EF,
        scratch: &mut SubfieldScratch<F, S, EF>,
    ) where
        S: Field,
        F: HasSubfield<S>,
        EF: HasSubfield<S>,
        A: for<'b> Air<MultilinearFolder<'b, F, SubfieldVar<F, S>, SubfieldAcc<EF, S>>>,
    {
        let half = rows.half;
        for slot in &self.slots {
            let main = self.tables[slot.stage_index];
            scratch.fill_local(slot.main_offset, main, row, half);
            scratch.fill_next(slot.main_offset, main, &slot.main_next_columns, row, half);
            if let Some(preprocessed) = self.preprocessed[slot.stage_index] {
                scratch.fill_local(slot.preprocessed_offset, preprocessed, row, half);
                scratch.fill_next(
                    slot.preprocessed_offset,
                    preprocessed,
                    &slot.preprocessed_next_columns,
                    row,
                    half,
                );
            }
            if let Some(periodic) = self.periodic[slot.stage_index].as_ref() {
                scratch.fill_local(slot.periodic_offset, periodic, row, half);
            }
        }

        // Selector indicators are zero or one, so they lie in every subfield.
        let (low, diff) = BoundaryEvals::<S>::row_pair(row, half, 2 * half);
        let lift = |evals: BoundaryEvals<S>| {
            BoundaryEvals::new(
                SubfieldVar::new(evals.first),
                SubfieldVar::new(evals.last),
                SubfieldVar::new(evals.transition),
            )
        };
        let (mut boundary, boundary_diff) = (lift(low), lift(diff));

        for &(node, step) in rows.schedule {
            match step {
                NodeStep::Unit(count) => {
                    for _ in 0..count {
                        scratch.add_diffs(&rows.next_columns);
                        boundary += boundary_diff;
                    }
                }
                NodeStep::Scaled(step) => {
                    let step = SubfieldVar::new(step);
                    scratch.add_scaled_diffs(step, &rows.next_columns);
                    boundary.add_scaled(boundary_diff, step);
                }
            }
            let eval_index = if node == 0 { 0 } else { node - 1 };
            for slot in &self.slots {
                if !slot.enabled_families(node, false).constraints {
                    continue;
                }
                let main = slot.main_offset..slot.main_offset + slot.main_width;
                let preprocessed =
                    slot.preprocessed_offset..slot.preprocessed_offset + slot.preprocessed_width;
                let periodic = slot.periodic_offset..slot.periodic_offset + slot.periodic_width;
                let constraints = MultilinearFolder::new(
                    &scratch.local_point[main.clone()],
                    &scratch.next_point[main],
                    boundary,
                    self.public_values[slot.stage_index],
                    rows.alpha,
                )
                .with_alpha_powers(&rows.alpha_powers[slot.stage_index])
                .with_preprocessed(
                    &scratch.local_point[preprocessed.clone()],
                    &scratch.next_point[preprocessed],
                )
                .with_periodic(&scratch.local_point[periodic])
                .eval_air(slot.air);
                scratch.constraint_evals[slot.stage_index][eval_index] += eq * constraints.value();
                scratch.poisoned |= constraints.is_poisoned();
            }
        }
    }
}

/// Decline the subfield kernel for a stage's first round, recording why.
fn fall_back<T>(reason: &'static str) -> Option<T> {
    tracing::debug!(reason, "the first round falls back to the generic kernel");
    None
}

/// Whether `EF` embeds `S` the way it embeds `F`'s copy of `S`.
///
/// Two ring embeddings that agree on a generator of the multiplicative group agree on all of `S`.
/// The powers of the generator are compared anyway, every one of them for a subfield of at most
/// [`MAX_EMBEDDING_POWERS`] elements, so a lift that is not a ring homomorphism shows up too.
fn embeddings_agree<S, F, EF>() -> bool
where
    S: Field,
    F: HasSubfield<S>,
    EF: ExtensionField<F> + HasSubfield<S>,
{
    let mut power = S::ONE;
    for _ in 0..MAX_EMBEDDING_POWERS {
        if EF::from(power) != EF::from(F::from(power)) {
            return false;
        }
        power *= S::GENERATOR;
        if power == S::ONE {
            break;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::TowerLevel;
    use p3_multilinear_util::point::Point;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::lookup::{AirLinkInstance, AirLinkLookup};
    use crate::rounds::{Stage, StageCoupling};
    use crate::zerocheck::backend_tests::{FixtureAir, Gf4, Instance, Tower, gf4, outside};
    use crate::zerocheck::get_air_profile;

    /// Activate one stage from instances of equal height and hand its first-round state to `body`.
    ///
    /// The challenges are fixed random elements of the tower.
    fn with_state<R>(
        instances: &[Instance],
        coupling: StageCoupling<Tower>,
        body: impl FnOnce(&mut RoundStateBase<'_, '_, FixtureAir, Tower, Tower>, &Poly<Tower>) -> R,
    ) -> R {
        let main = instances
            .iter()
            .map(Instance::main_table)
            .collect::<Vec<_>>();
        let preprocessed = instances
            .iter()
            .map(Instance::preprocessed_table)
            .collect::<Vec<_>>();
        let stage = Stage::new(
            instances.iter().map(|instance| &instance.air).collect(),
            instances
                .iter()
                .map(|instance| instance.public_values.as_slice())
                .collect(),
            (0..instances.len()).collect(),
            preprocessed.iter().map(Option::as_ref).collect(),
            main.iter().collect(),
            instances
                .iter()
                .map(|instance| get_air_profile::<Tower, Tower, _>(&instance.air))
                .collect(),
            coupling,
        );

        let mut rng = SmallRng::seed_from_u64(0x5B);
        let tau = Point::rand(&mut rng, stage.num_vars);
        let eq_suffix = Poly::new_from_point(&tau.as_slice()[1..], Tower::ONE);
        let betas = (0..instances.len()).map(|_| rng.random()).collect();
        let mut state = RoundStateBase::new(stage, rng.random(), rng.random(), betas, tau);
        body(&mut state, &eq_suffix)
    }

    fn no_lookups() -> StageCoupling<Tower> {
        StageCoupling::new(BTreeMap::new(), BTreeMap::new(), vec![])
    }

    /// The subfield kernel's first round polynomial, beside the generic kernel's.
    fn first_rounds(
        instances: &[Instance],
        coupling: StageCoupling<Tower>,
    ) -> (Option<Vec<Tower>>, Vec<Tower>) {
        with_state(instances, coupling, |state, eq_suffix| {
            let subfield = state.round_poly_subfield::<Gf4>(eq_suffix);
            (subfield, state.round_poly(eq_suffix))
        })
    }

    #[test]
    fn a_fitting_stage_runs_in_the_subfield_and_matches_the_generic_kernel() {
        // Residual rows per height: one row, part of a task, one full task, several tasks.
        //
        // An honest two-row pair can have a zero round polynomial, so its product cell breaks it.
        let mut pair = Instance::honest(FixtureAir::Pair, 2, 1);
        pair.main.values[2] = gf4(3);
        let stages = [
            vec![pair],
            vec![
                Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 4, 2),
                Instance::honest(FixtureAir::Pair, 4, 3),
            ],
            vec![Instance::honest(FixtureAir::Gate { scale: gf4(2) }, 32, 4)],
            vec![
                Instance::honest(FixtureAir::Pair, 128, 5),
                Instance::honest(FixtureAir::Gate { scale: gf4(3) }, 128, 6),
            ],
        ];
        for instances in stages {
            let (subfield, generic) = first_rounds(&instances, no_lookups());
            assert!(generic.iter().any(|value| *value != Tower::ZERO));
            assert_eq!(subfield, Some(generic));
        }
    }

    #[test]
    fn each_misfit_falls_back() {
        let gate = || Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 7);
        let mut cell = gate();
        // Column e of row 5.
        cell.main.values[4 * 5 + 3] = outside();
        let mut public = gate();
        public.public_values[1] = outside();
        let constant = Instance::honest(
            FixtureAir::Gate {
                scale: Tower::from_repr(5),
            },
            16,
            8,
        );
        let quartic = Instance::honest(FixtureAir::Quartic, 16, 9);

        for (name, instance) in [
            ("cell", cell),
            ("public value", public),
            ("constant", constant),
            ("interpolation step", quartic),
        ] {
            let (subfield, _) = first_rounds(&[instance], no_lookups());
            assert_eq!(subfield, None, "{name}");
        }
    }

    #[test]
    fn a_lookup_stage_falls_back() {
        let link = AirLinkInstance {
            num_local_lookups: 1,
            lookups: vec![AirLinkLookup {
                theta_bus_offset: gf4(1),
                block_weights: vec![gf4(2), gf4(3)],
            }],
        };
        let coupling = StageCoupling::new(
            BTreeMap::from([(0, gf4(1))]),
            BTreeMap::from([(0, link)]),
            vec![gf4(2)],
        );
        let (subfield, _) = first_rounds(&[Instance::honest(FixtureAir::Link, 16, 10)], coupling);
        assert_eq!(subfield, None);
    }

    #[test]
    fn an_out_of_subfield_constant_passes_the_check_and_poisons_the_probe() {
        let instance = Instance::honest(
            FixtureAir::Gate {
                scale: Tower::from_repr(5),
            },
            16,
            11,
        );
        with_state(&[instance], no_lookups(), |state, eq_suffix| {
            let schedule = state
                .subfield_schedule::<Gf4>()
                .expect("every cell, public value, and step fits");
            assert!(
                state
                    .subfield_pass(&eq_suffix.as_slice()[..1], &schedule)
                    .poisoned
            );
        });
    }

    #[test]
    fn the_full_pass_catches_a_misfit_the_probe_does_not_read() {
        let gate = || Instance::honest(FixtureAir::Gate { scale: Tower::ONE }, 16, 12);
        let schedule = with_state(&[gate()], no_lookups(), |state, _| {
            state
                .subfield_schedule::<Gf4>()
                .expect("an honest gate fits")
        });

        // Only row 5 reads column e of row 5, and the probe reads row 0 alone.
        let mut dirty = gate();
        dirty.main.values[4 * 5 + 3] = outside();
        with_state(&[dirty], no_lookups(), |state, eq_suffix| {
            assert!(state.subfield_schedule::<Gf4>().is_none());
            let eq_suffix = eq_suffix.as_slice();
            assert!(!state.subfield_pass(&eq_suffix[..1], &schedule).poisoned);
            assert!(state.subfield_pass(eq_suffix, &schedule).poisoned);
        });
    }
}
