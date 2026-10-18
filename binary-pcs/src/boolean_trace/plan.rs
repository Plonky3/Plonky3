//! Planning an opening: where each column lands, which route discharges it, and where
//! each value sits.

use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Selector, Table, TablePlacement, plan_stacked_layout};
use p3_sumcheck::{OpeningEvals, OpeningPointMismatch, OpeningProtocol, TableShape};

use super::{BooleanTraceCommitment, BooleanTraceCommitmentError};
use crate::boolean::{BitOpening, BitReadings, BooleanBackend};
use crate::boolean_trace_transcript::ColumnBatchShape;

/// Validate cell arithmetic and the padded witness before the shared infallible planner allocates selectors.
/// With this bound every planner offset is no larger than the cell count.
fn checked_layout_arity(shapes: &[TableShape]) -> Option<usize> {
    let cells = shapes.iter().try_fold(0usize, |total, shape| {
        let rows = 1usize.checked_shl(shape.num_variables().try_into().ok()?)?;
        total.checked_add(rows.checked_mul(shape.width())?)
    })?;
    let padded = cells.checked_next_power_of_two()?;
    Some(padded.trailing_zeros() as usize)
}

/// Count public batches, including empty ones, without wrapping allocation lengths.
pub(super) fn opening_count(protocol: &OpeningProtocol) -> Option<usize> {
    protocol
        .iter_openings()
        .try_fold(0usize, |count, _| count.checked_add(1))
}

fn checked_stacked_layout(shapes: &[TableShape]) -> Option<(usize, Vec<TablePlacement>)> {
    checked_layout_arity(shapes)?;
    Some(plan_stacked_layout(shapes))
}

impl<EF, B> BooleanTraceCommitment<EF, B>
where
    EF: BitCoordinates + ExtensionField<B::Val>,
    B: BooleanBackend<EF>,
    B::Val: TranscriptField + TowerLevel,
{
    /// Where each table's columns land in the bit witness, planned from the shapes.
    ///
    /// # Errors
    ///
    /// Returns an error unless the shapes stack to the committed arity.
    pub(super) fn placements(
        &self,
        shapes: &[TableShape],
    ) -> Result<Vec<TablePlacement>, BooleanTraceCommitmentError<B::Error>> {
        // Prover and verifier both plan from the public shapes, so neither picks its own.
        let arity = checked_layout_arity(shapes)
            .ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)?;
        if arity != self.num_variables() {
            return Err(BooleanTraceCommitmentError::StackedArity {
                expected: self.num_variables(),
                actual: arity,
            });
        }
        // Reject a mismatched arity before allocating one selector per public column.
        checked_stacked_layout(shapes)
            .map(|(_, placements)| placements)
            .ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)
    }

    /// Validate protocol counts and arity without allocating placements or sampling.
    pub(super) fn validate_protocol(
        &self,
        protocol: &OpeningProtocol,
    ) -> Result<(), BooleanTraceCommitmentError<B::Error>> {
        protocol
            .checked_num_cells()
            .ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)?;
        opening_count(protocol).ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)?;
        value_count(protocol).ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)?;
        let actual = checked_layout_arity(&protocol.table_shapes())
            .ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)?;
        if actual != self.num_variables() {
            return Err(BooleanTraceCommitmentError::StackedArity {
                expected: self.num_variables(),
                actual,
            });
        }
        Ok(())
    }

    /// The bit claims the per-column route raises, in transcript order.
    ///
    /// One claim per column a batch reads, that batch's point prefixed by the slot address,
    /// asking for whichever readings the claim's entry in the plan names.
    pub(super) fn bit_openings(
        protocol: &OpeningProtocol,
        claims: &[ColumnClaim],
        points: &[Point<EF>],
        placements: &[TablePlacement],
    ) -> Vec<BitOpening<EF>> {
        let shapes = protocol.table_shapes();

        // Placements arrive largest table first, so index them by the table each one owns.
        let mut by_table = alloc::vec![None; shapes.len()];
        for placement in placements {
            by_table[placement.idx()] = Some(placement);
        }

        claims
            .iter()
            .map(|claim| {
                let placement =
                    by_table[claim.table].expect("the planner places every supplied shape");
                BitOpening {
                    // Slot address as the leading coordinates, the row point as the trailing ones.
                    point: placement.selectors()[claim.column].lift_prefix(&points[claim.opening]),
                    // The successor view steps within the rows, never into the slot address.
                    row_variables: shapes[claim.table].num_variables(),
                    current: claim.current_at.is_some(),
                    next: claim.next_at.is_some(),
                }
            })
            .collect()
    }

    /// Validate all public opening metadata without constructing per-column claims.
    pub(super) fn validate_opening(
        &self,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
    ) -> Result<Vec<TablePlacement>, BooleanTraceCommitmentError<B::Error>> {
        self.validate_protocol(protocol)?;
        let placements = self.placements(&protocol.table_shapes())?;
        protocol
            .check_points(points)
            .map_err(|mismatch| match mismatch {
                OpeningPointMismatch::Count { expected, actual } => {
                    BooleanTraceCommitmentError::PointCount { expected, actual }
                }
                OpeningPointMismatch::Arity {
                    table,
                    expected,
                    actual,
                } => BooleanTraceCommitmentError::PointArity {
                    table,
                    expected,
                    actual,
                },
            })?;
        Ok(placements)
    }

    /// Validate retained source shapes before any sampled point or opening transcript is used.
    pub(super) fn validate_source_shapes(
        tables: &[Table<B::Val>],
        protocol: &OpeningProtocol,
    ) -> Result<(), BooleanTraceCommitmentError<B::Error>> {
        let expected = protocol.table_shapes();
        if tables.len() != expected.len() {
            return Err(BooleanTraceCommitmentError::TableCountMismatch {
                expected: expected.len(),
                actual: tables.len(),
            });
        }
        for (table, expected) in expected.iter().copied().enumerate() {
            let actual = tables[table].shape();
            if actual != expected {
                return Err(BooleanTraceCommitmentError::TableShapeMismatch {
                    table,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Check that supplied values name one run per batch, each with its batch's shape.
    pub(super) fn validate_known(
        protocol: &OpeningProtocol,
        known: &[Option<OpeningEvals<EF>>],
    ) -> Result<(), BooleanTraceCommitmentError<B::Error>> {
        let count =
            opening_count(protocol).ok_or(BooleanTraceCommitmentError::InvalidOpeningProtocol)?;
        if known.len() != count {
            return Err(BooleanTraceCommitmentError::KnownValueCount {
                expected: count,
                actual: known.len(),
            });
        }
        for (batch, ((_, request), known)) in protocol.iter_openings().zip(known).enumerate() {
            if known
                .as_ref()
                .is_some_and(|known| !request.has_same_shape(known))
            {
                return Err(BooleanTraceCommitmentError::KnownValueShape { batch });
            }
        }
        Ok(())
    }
}

/// How one opening protocol is discharged against the bit commitment.
///
/// A function of the protocol alone: the security assessment, the prover and the verifier
/// read the same resolution.
#[derive(Clone, Debug)]
pub(super) enum OpeningRoute {
    /// Every batch reads all of its table at the current row, and either none or all of it
    /// one row ahead: one reduction per aligned column block, over one shared column point.
    Batched(Vec<TableRun>),
    /// Protocols with no profitable complete view: one reduction per column read.
    PerColumn(ClaimPlan),
    /// Complete current views batched separately from residual ordinary claims.
    Hybrid {
        runs: Vec<TableRun>,
        residual: Vec<ColumnClaim>,
    },
}

/// The batches of one table, run through one column-batching transcript.
#[derive(Clone, Debug)]
pub(super) struct TableRun {
    /// Table every batch of the run opens.
    pub(super) table: usize,
    /// First batch of this run in protocol order.
    pub(super) opening: usize,
    /// First value of this run in protocol order.
    pub(super) value_offset: usize,
    /// Shape the run's transcript binds.
    pub(super) shape: ColumnBatchShape,
    /// Aligned blocks the table's column slots split into, each one reduction per batch.
    pub(super) blocks: Vec<ColumnBlock>,
}

/// One aligned run of a table's column slots, read as one claim at a lifted point.
///
/// ```text
///     slots   [prefix * 2^j, (prefix + 1) * 2^j)
///     claim   sum_i eq(u_j, i) * column_{first + i}(r)  =  W(prefix, u_j, r)
/// ```
///
/// A slot past the table is past every table too, so the witness holds zero there.
#[derive(Clone, Copy, Debug)]
pub(super) struct ColumnBlock {
    /// Address of the block: the slot bits above its own column variables.
    pub(super) prefix: Selector,
    /// Column of the table the block starts at.
    pub(super) first: usize,
    /// Column variables the block spans, so `2^variables` slots.
    pub(super) variables: usize,
}

impl OpeningRoute {
    /// Resolve the route a protocol takes.
    pub(super) fn new(protocol: &OpeningProtocol) -> Option<Self> {
        protocol.checked_num_cells()?;
        opening_count(protocol)?;
        checked_layout_arity(&protocol.table_shapes())?;
        value_count(protocol)?;
        let route = if let Some(runs) = batched_runs(protocol) {
            Self::Batched(runs)
        } else if let Some(route) = hybrid_route(protocol) {
            route
        } else {
            Self::PerColumn(ClaimPlan::of(protocol)?)
        };
        // Check every run and security count before any transcript can be started.
        for run in route.runs() {
            run.shape.interaction_count()?;
            run.value_offset.checked_add(run.shape.run_len()?)?;
            run.opening.checked_add(run.shape.num_batches)?;
        }
        route.num_reductions()?;
        route.column_claim_count()?;
        Some(route)
    }

    /// Column-batched runs, including those beside ordinary residual claims.
    pub(super) fn runs(&self) -> &[TableRun] {
        match self {
            Self::Batched(runs) | Self::Hybrid { runs, .. } => runs,
            Self::PerColumn(_) => &[],
        }
    }

    /// Reductions the bit commitment answers on this route.
    pub(super) fn num_reductions(&self) -> Option<usize> {
        let blocks = self.runs().iter().try_fold(0usize, |total, run| {
            total.checked_add(run.shape.num_batches.checked_mul(run.blocks.len())?)
        })?;
        match self {
            Self::Batched(_) => Some(blocks),
            Self::PerColumn(plan) => Some(plan.claims().len()),
            Self::Hybrid { residual, .. } => blocks.checked_add(residual.len()),
        }
    }

    /// Every block/view combination that draws a column point, checked separately from the ordinary residual claims the backend charges.
    pub(super) fn column_claim_count(&self) -> Option<usize> {
        self.runs().iter().try_fold(0usize, |total, run| {
            let claims = run
                .shape
                .num_batches
                .checked_mul(run.shape.num_views())?
                .checked_mul(run.blocks.len())?;
            total.checked_add(claims)
        })
    }

    /// Whether some reduction reads a successor view over more rows than one element absorbs.
    pub(super) fn successor_tensors(&self, shapes: &[TableShape], absorbed: usize) -> bool {
        match self {
            Self::Batched(runs) => runs
                .iter()
                .any(|run| run.shape.next && run.shape.table_variables > absorbed),
            Self::Hybrid { runs, residual } => {
                runs.iter()
                    .any(|run| run.shape.next && run.shape.table_variables > absorbed)
                    || residual.iter().any(|claim| {
                        claim.next_at.is_some() && shapes[claim.table].num_variables() > absorbed
                    })
            }
            Self::PerColumn(plan) => plan.claims().iter().any(|claim| {
                claim.next_at.is_some() && shapes[claim.table].num_variables() > absorbed
            }),
        }
    }
}

/// The complete shape the batched route handles, if the protocol has one.
///
/// Every batch reads its table's whole width at the current row.
///
/// One row ahead it reads none of it or all of it, the same way across the table's batches.
fn batched_runs(protocol: &OpeningProtocol) -> Option<Vec<TableRun>> {
    let shapes = protocol.table_shapes();
    if opening_count(protocol)? == 0 {
        return None;
    }
    let (arity, placements) = checked_stacked_layout(&shapes)?;
    let used = protocol.checked_num_cells()?;
    let mut by_table = alloc::vec![None; shapes.len()];
    for placement in &placements {
        by_table[placement.idx()] = Some(placement);
    }

    let mut runs: Vec<TableRun> = Vec::new();
    let mut value_offset = 0usize;
    for (opening, (table, batch)) in protocol.iter_openings().enumerate() {
        let width = shapes[table].width();
        let complete = |read: &[usize]| read.iter().copied().eq(0..width);
        let next = !batch.next().is_empty();
        if !complete(batch.current()) || (next && !complete(batch.next())) {
            return None;
        }
        match runs.last_mut() {
            Some(run) if run.table == table => {
                // One transcript binds one view set, so a table's batches agree on it.
                if run.shape.next != next {
                    return None;
                }
                run.shape.num_batches = run.shape.num_batches.checked_add(1)?;
            }
            _ => {
                let placement = by_table[table]?;
                runs.push(TableRun {
                    table,
                    opening,
                    value_offset,
                    shape: ColumnBatchShape {
                        table_variables: shapes[table].num_variables(),
                        width,
                        num_batches: 1,
                        next,
                        sparse_next: Vec::new(),
                    },
                    blocks: column_blocks(placement, shapes[table], arity, used)?,
                });
            }
        }
        value_offset = value_offset
            .checked_add(batch.current().len())?
            .checked_add(batch.next().len())?;
    }
    Some(runs)
}

/// Batch a complete current view only when its blocks plus sparse successor claims cost less than the original shared current/successor per-column reductions.
fn hybrid_route(protocol: &OpeningProtocol) -> Option<OpeningRoute> {
    let shapes = protocol.table_shapes();
    let (arity, placements) = checked_stacked_layout(&shapes)?;
    let used = protocol.checked_num_cells()?;
    let mut by_table = alloc::vec![None; shapes.len()];
    for placement in &placements {
        by_table[placement.idx()] = Some(placement);
    }
    let mut runs = Vec::new();
    let mut selected = alloc::vec![None; opening_count(protocol)?];
    let mut offset = 0usize;
    for (opening, (table, batch)) in protocol.iter_openings().enumerate() {
        let width = shapes[table].width();
        let complete = |read: &[usize]| read.iter().copied().eq(0..width);
        if complete(batch.current()) {
            let blocks = column_blocks(by_table[table]?, shapes[table], arity, used)?;
            let next = complete(batch.next());
            let residual_count = if next { 0 } else { batch.next().len() };
            if blocks.len().checked_add(residual_count)? < width {
                selected[opening] = Some(next);
                runs.push(TableRun {
                    table,
                    opening,
                    value_offset: offset,
                    blocks,
                    shape: ColumnBatchShape {
                        table_variables: shapes[table].num_variables(),
                        width,
                        num_batches: 1,
                        next,
                        sparse_next: if next {
                            Vec::new()
                        } else {
                            batch.next().to_vec()
                        },
                    },
                });
            }
        }
        offset = offset
            .checked_add(batch.current().len())?
            .checked_add(batch.next().len())?;
    }
    if runs.is_empty() {
        return None;
    }
    // Complete current vectors are consumed by the selected runs.
    // Their sparse successor positions remain ordinary claims; all other batches stay intact.
    let residual = column_claims(protocol)?
        .into_iter()
        .filter_map(|mut claim| match selected[claim.opening] {
            Some(true) => None,
            Some(false) => {
                claim.current_at = None;
                claim.next_at.map(|_| claim)
            }
            None => Some(claim),
        })
        .collect();
    Some(OpeningRoute::Hybrid { runs, residual })
}

/// Split a table's column slots into aligned blocks, largest first.
///
/// A block may run past the table only where no table follows it.
///
/// So one table alone is one block, the width padded to a power of two.
fn column_blocks(
    placement: &TablePlacement,
    shape: TableShape,
    arity: usize,
    used: usize,
) -> Option<Vec<ColumnBlock>> {
    let rows = shape.num_variables();
    let selectors = placement.selectors();
    let start = selectors.first()?.index();
    // The planner lays one table's columns out back to back.
    if !selectors
        .iter()
        .enumerate()
        .all(|(column, selector)| Some(selector.index()) == start.checked_add(column))
    {
        return None;
    }
    let slots = 1usize.checked_shl(arity.checked_sub(rows)?.try_into().ok()?)?;
    let end = start.checked_add(shape.width())?;
    let row_count = 1usize.checked_shl(rows.try_into().ok()?)?;
    let free_tail = end.checked_mul(row_count)? >= used;
    let column_variables = shape.width().checked_next_power_of_two()?.trailing_zeros() as usize;

    let mut blocks = Vec::new();
    let mut position = start;
    while position < end {
        let mut variables = column_variables.min(position.trailing_zeros() as usize);
        loop {
            let stop = position.checked_add(1usize.checked_shl(variables.try_into().ok()?)?)?;
            if stop <= end || (free_tail && stop <= slots) || variables == 0 {
                break;
            }
            variables -= 1;
        }
        blocks.push(ColumnBlock {
            prefix: Selector::new(arity - rows - variables, position >> variables),
            first: position - start,
            variables,
        });
        position = position.checked_add(1usize.checked_shl(variables.try_into().ok()?)?)?;
    }
    Some(blocks)
}

/// One bit claim of the per-column route: the column it reads, and where its values sit.
#[derive(Clone, Copy, Debug)]
pub(super) struct ColumnClaim {
    /// Table the claim's batch opens.
    pub(super) table: usize,
    /// Column of that table the claim reads.
    pub(super) column: usize,
    /// Batch the claim belongs to, whose point it is lifted by.
    pub(super) opening: usize,
    /// Where the reading at the point sits in the value run, when it is asked for.
    pub(super) current_at: Option<usize>,
    /// Where the reading one row ahead sits in the value run, when it is asked for.
    pub(super) next_at: Option<usize>,
}

/// One claim per column a batch reads, batches in protocol order.
///
/// A batch contributes one claim per entry of its current list, then one per successor
/// entry no current claim already answers. Each claim carries the positions of its own
/// values in the run, which is every batch's current values followed by its next ones.
pub(super) fn column_claims(protocol: &OpeningProtocol) -> Option<Vec<ColumnClaim>> {
    value_count(protocol)?;
    let mut claims = Vec::new();
    let mut cursor = 0usize;
    for (opening, (table, batch)) in protocol.iter_openings().enumerate() {
        let next_cursor = cursor.checked_add(batch.current().len())?;
        // Every successor entry is answered exactly once, so no claimed value goes unchecked.
        let mut answered = alloc::vec![false; batch.next().len()];
        for (offset, &column) in batch.current().iter().enumerate() {
            let at = batch
                .next()
                .iter()
                .zip(&answered)
                .position(|(&other, &taken)| other == column && !taken);
            if let Some(at) = at {
                answered[at] = true;
            }
            claims.push(ColumnClaim {
                table,
                column,
                opening,
                current_at: Some(cursor.checked_add(offset)?),
                next_at: if let Some(at) = at {
                    Some(next_cursor.checked_add(at)?)
                } else {
                    None
                },
            });
        }
        for (at, &column) in batch.next().iter().enumerate() {
            if !answered[at] {
                claims.push(ColumnClaim {
                    table,
                    column,
                    opening,
                    current_at: None,
                    next_at: Some(next_cursor.checked_add(at)?),
                });
            }
        }
        cursor = next_cursor.checked_add(batch.next().len())?;
    }
    Some(claims)
}

/// Values one protocol opens: per batch, its current readings then its successor ones.
pub(super) fn value_count(protocol: &OpeningProtocol) -> Option<usize> {
    protocol
        .iter_openings()
        .try_fold(0usize, |total, (_, batch)| {
            total
                .checked_add(batch.current().len())?
                .checked_add(batch.next().len())
        })
}

/// The per-column claims of one protocol, together with the length of the value run they lay
/// out.
///
/// A plan exists only when its claims write each position of that run exactly once, so the
/// conversions between readings and values below touch every position and no other.
#[derive(Clone, Debug)]
pub(super) struct ClaimPlan {
    /// The claims, in transcript order.
    claims: Vec<ColumnClaim>,
    /// Values in the run the claims lay out.
    len: usize,
}

impl ClaimPlan {
    /// The plan the per-column route raises for a protocol.
    pub(super) fn of(protocol: &OpeningProtocol) -> Option<Self> {
        Self::new(column_claims(protocol)?, value_count(protocol)?)
    }

    /// A plan over a run of `len` values, if the claims write each position exactly once.
    ///
    /// A position two claims write is one reading the plan answers twice, and a position no
    /// claim writes is one no reading fills.
    pub(super) fn new(claims: Vec<ColumnClaim>, len: usize) -> Option<Self> {
        let mut written = alloc::vec![false; len];
        for claim in &claims {
            for at in [claim.current_at, claim.next_at].into_iter().flatten() {
                if at >= len || core::mem::replace(&mut written[at], true) {
                    return None;
                }
            }
        }
        written
            .into_iter()
            .all(|written| written)
            .then_some(Self { claims, len })
    }

    /// The claims, in transcript order.
    pub(super) const fn claims(&self) -> &[ColumnClaim] {
        self.claims.as_slice()
    }

    /// Lay the readings every claim came back with out in the protocol's value order.
    pub(super) fn values<EF: Field>(&self, readings: &[BitReadings<EF>]) -> Vec<EF> {
        // The plan writes every position, so none keeps the zero the run starts from.
        let mut values = alloc::vec![EF::ZERO; self.len];
        for (claim, reading) in self.claims.iter().zip(readings) {
            if let Some(at) = claim.current_at {
                values[at] = reading
                    .current
                    .expect("a claim asking for the reading at the point carries it");
            }
            if let Some(at) = claim.next_at {
                values[at] = reading
                    .next
                    .expect("a claim asking for the reading one row ahead carries it");
            }
        }
        values
    }

    /// The readings every claim asks for, read back out of the protocol's value order.
    pub(super) fn readings<EF: Field>(&self, values: &[EF]) -> Vec<BitReadings<EF>> {
        self.claims
            .iter()
            .map(|claim| BitReadings {
                current: claim.current_at.map(|at| values[at]),
                next: claim.next_at.map(|at| values[at]),
            })
            .collect()
    }
}

/// Split the flat value run back into each batch's current and successor values.
pub(super) fn opening_evals<EF: Field>(
    protocol: &OpeningProtocol,
    values: &[EF],
) -> Vec<OpeningEvals<EF>> {
    let mut evals =
        Vec::with_capacity(opening_count(protocol).expect("the protocol count was checked"));
    let mut cursor = 0usize;
    for (_, batch) in protocol.iter_openings() {
        let next_cursor = cursor + batch.current().len();
        let end = next_cursor + batch.next().len();
        evals.push(OpeningEvals::new(
            values[cursor..next_cursor].to_vec(),
            values[next_cursor..end].to_vec(),
        ));
        cursor = end;
    }
    evals
}

/// One point per opening batch, drawn from the transcript in batch order.
///
/// The sponge speaks the packing level `F`, and each coordinate is an element of `EF`.
pub(super) fn sample_points<F, EF, Challenger>(
    protocol: &OpeningProtocol,
    challenger: &mut Challenger,
) -> Vec<Point<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    // Coordinates are drawn in the order the batches stream, one batch's point at a time.
    let shapes = protocol.table_shapes();
    protocol
        .iter_openings()
        .map(|(table, _)| {
            let num_variables = shapes[table].num_variables();
            Point::new(
                (0..num_variables)
                    .map(|_| challenger.sample_algebra_element())
                    .collect(),
            )
        })
        .collect()
}
