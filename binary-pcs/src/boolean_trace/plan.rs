//! Planning an opening: where each column lands, which route discharges it, and where
//! each value sits.

use alloc::vec::Vec;

use p3_binary_dft::EncodableLevel;
use p3_binary_field::TowerLevel;
use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Selector, Table, TablePlacement, plan_stacked_layout};
use p3_sumcheck::{OpeningEvals, OpeningPointMismatch, OpeningProtocol, TableShape};

use super::{BooleanTraceCommitment, BooleanTraceCommitmentError};
use crate::boolean::{BitOpening, BitReadings, BooleanBackend};
use crate::boolean_trace_transcript::ColumnBatchShape;
use crate::fold::{ChallengeField, FoldAlphabet};
use crate::packing::Coordinates;

impl<EF, B> BooleanTraceCommitment<EF, B>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    B: BooleanBackend<EF>,
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
        let (arity, placements) = plan_stacked_layout(shapes);
        if arity != self.num_variables() {
            return Err(BooleanTraceCommitmentError::StackedArity {
                expected: self.num_variables(),
                actual: arity,
            });
        }
        Ok(placements)
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
        tables: &[Table<EF>],
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
    /// Any other protocol: one reduction per column read, in transcript order.
    PerColumn(ClaimPlan),
}

/// The batches of one table, run through one column-batching transcript.
#[derive(Clone, Debug)]
pub(super) struct TableRun {
    /// Table every batch of the run opens.
    pub(super) table: usize,
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
    pub(super) fn new(protocol: &OpeningProtocol) -> Self {
        batched_runs(protocol)
            .map_or_else(|| Self::PerColumn(ClaimPlan::of(protocol)), Self::Batched)
    }

    /// Reductions the bit commitment answers on this route.
    pub(super) fn num_reductions(&self) -> usize {
        match self {
            Self::Batched(runs) => runs
                .iter()
                .map(|run| run.shape.num_batches * run.blocks.len())
                .sum(),
            Self::PerColumn(plan) => plan.claims().len(),
        }
    }

    /// Whether some reduction reads a successor view over more rows than one element absorbs.
    pub(super) fn successor_tensors(&self, shapes: &[TableShape], absorbed: usize) -> bool {
        match self {
            Self::Batched(runs) => runs
                .iter()
                .any(|run| run.shape.next && run.shape.table_variables > absorbed),
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
    if protocol.num_openings() == 0 {
        return None;
    }
    let (arity, placements) = plan_stacked_layout(&shapes);
    let used = shapes
        .iter()
        .map(|shape| shape.width() << shape.num_variables())
        .sum::<usize>();
    let mut by_table = alloc::vec![None; shapes.len()];
    for placement in &placements {
        by_table[placement.idx()] = Some(placement);
    }

    let mut runs: Vec<TableRun> = Vec::new();
    for (table, batch) in protocol.iter_openings() {
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
                run.shape.num_batches += 1;
            }
            _ => {
                let placement = by_table[table]?;
                runs.push(TableRun {
                    table,
                    shape: ColumnBatchShape {
                        table_variables: shapes[table].num_variables(),
                        width,
                        num_batches: 1,
                        next,
                    },
                    blocks: column_blocks(placement, shapes[table], arity, used)?,
                });
            }
        }
    }
    Some(runs)
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
        .all(|(column, selector)| selector.index() == start + column)
    {
        return None;
    }
    let slots = 1usize << (arity - rows);
    let end = start + shape.width();
    let free_tail = end << rows >= used;
    let column_variables = shape.width().next_power_of_two().trailing_zeros() as usize;

    let mut blocks = Vec::new();
    let mut position = start;
    while position < end {
        let mut variables = column_variables.min(position.trailing_zeros() as usize);
        loop {
            let stop = position + (1 << variables);
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
        position += 1 << variables;
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
pub(super) fn column_claims(protocol: &OpeningProtocol) -> Vec<ColumnClaim> {
    let mut claims = Vec::new();
    let mut cursor = 0;
    for (opening, (table, batch)) in protocol.iter_openings().enumerate() {
        let next_cursor = cursor + batch.current().len();
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
                current_at: Some(cursor + offset),
                next_at: at.map(|at| next_cursor + at),
            });
        }
        for (at, &column) in batch.next().iter().enumerate() {
            if !answered[at] {
                claims.push(ColumnClaim {
                    table,
                    column,
                    opening,
                    current_at: None,
                    next_at: Some(next_cursor + at),
                });
            }
        }
        cursor = next_cursor + batch.next().len();
    }
    claims
}

/// Values one protocol opens: per batch, its current readings then its successor ones.
pub(super) fn value_count(protocol: &OpeningProtocol) -> usize {
    protocol.iter_openings().map(|(_, batch)| batch.len()).sum()
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
    pub(super) fn of(protocol: &OpeningProtocol) -> Self {
        Self::new(column_claims(protocol), value_count(protocol))
            .expect("the per-column claims write every value position exactly once")
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
    let mut evals = Vec::with_capacity(protocol.num_openings());
    let mut cursor = 0;
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
pub(super) fn sample_points<EF, Challenger>(
    protocol: &OpeningProtocol,
    challenger: &mut Challenger,
) -> Vec<Point<EF>>
where
    EF: Field,
    Challenger: FieldChallenger<EF>,
{
    // Coordinates are drawn in the order the batches stream, one batch's point at a time.
    let shapes = protocol.table_shapes();
    protocol
        .iter_openings()
        .map(|(table, _)| {
            let num_variables = shapes[table].num_variables();
            Point::new((0..num_variables).map(|_| challenger.sample()).collect())
        })
        .collect()
}
