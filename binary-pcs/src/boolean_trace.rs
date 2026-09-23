//! A batched trace of Boolean columns, committed as bits and opened as columns.
//!
//! ```text
//!     in    one table per AIR instance, cells in {0, 1}
//!     out   one bit witness, committed packed, opened column by column
//! ```
//!
//! # From a column claim to a bit claim
//!
//! Every column occupies one contiguous run of the bit witness.
//!
//! A run starts at a multiple of its own length, so its address is a bit string.
//!
//! ```text
//!     column of slot s, 2^a rows  ->  bits [s * 2^a, (s + 1) * 2^a)
//!     its value at a point r      ->  the witness at (bits of s, r)
//! ```
//!
//! The slot address therefore prefixes the row point, and nothing is re-indexed.
//!
//! The planner that assigns the slots is the one the commitment schemes use.
//!
//! # Work stays inside the column
//!
//! The slot prefix is fixed before the ring-switch sumcheck starts.
//!
//! A complete batch reads every column of its table at the current row.
//!
//! One row ahead it reads no column or every column, as the table's other batches do.
//!
//! A protocol of complete batches combines columns at a fresh column point `u`.
//!
//! The point is drawn after the batch's claimed values are bound.
//!
//! A table's slots split into aligned blocks, and each block is one lifted point:
//!
//! ```text
//!     block of 2^j slots at prefix p   ->  W(p, u_j, r) = sum_i eq(u_j, i) * column_i(r)
//! ```
//!
//! A block may run past the table only where no table follows, so a lone table is one block.
//!
//! Any other protocol keeps one reduction per opened column.
//!
//! If one element holds `2^d_log` bits, a column folds `2^max(a - d_log, 0)` elements.
//!
//! # Booleanity is still free
//!
//! The packing is a bijection between bit strings and the elements a commitment holds.
//!
//! No commitment exists to a function the hypercube sends outside `{0, 1}`.
//!
//! A cell outside `{0, 1}` is refused at commitment rather than range-checked.
//!
//! The opened values are the bits, so a proof whose own trace disagrees with them fails.
//!
//! # What a batch opens
//!
//! A batch names columns read at the current row, and columns read one row ahead.
//!
//! The successor view is repeat-last: row `x` reads row `x + 1`, and the last row itself.
//!
//! It weights the column by the equality table shifted up by one row, which the
//! reduction underneath answers beside the reading at the point.
//!
//! The step stays inside the table's own row variables, so it never enters a slot address.
//!
//! ```text
//!     every batch complete    ->  one reduction per aligned block per batch
//!     anything else           ->  one reduction per column read
//! ```
//!
//! A column named by both views of a batch is one reduction answering both readings.
//!
//! The values of a batch are its current ones, in the order the batch names them,
//! followed by its next ones in the order the batch names those.
//!
//! An AIR whose constraints read only the current row names no successor column.

mod evaluate;
mod gather;
mod plan;

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_dft::EncodableLevel;
use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_sumcheck::{
    OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use self::plan::{OpeningRoute, TableRun, opening_evals, sample_points, value_count};
use crate::boolean::{
    BitOpening, BitReadings, BooleanBackend, BooleanMultilinearPcs, BooleanPcs, BooleanPcsError,
    BooleanProof,
};
use crate::boolean_trace_transcript::{
    ColumnBatchProverTranscript, ColumnBatchShape, ColumnBatchVerifierTranscript,
};
use crate::fold::{ChallengeField, FoldAlphabet};
use crate::packing::Coordinates;
use crate::params::BinaryPcsConfig;
use crate::prover::BinaryPcsProverData;

/// Bits one word of the staging buffer holds.
const WORD_BITS: usize = 64;

/// A commitment to a batch of Boolean trace tables, packed into one bit witness.
///
/// The committed object is the bits, so the codeword is as short as the alphabet allows.
///
/// The tables themselves are retained, for a prover that evaluates constraints over them.
pub struct BooleanTraceCommitment<EF, B> {
    /// The bit commitment every column claim is discharged against.
    inner: B,
    /// Marker tying the commitment to its challenge field; carries no runtime state.
    _marker: PhantomData<EF>,
}

/// The trace commitment discharged through the folding-only bit commitment.
pub type BooleanTracePcs<EF, MT, MX> = BooleanTraceCommitment<EF, BooleanPcs<EF, MT, MX>>;

impl<EF, MT, MX> BooleanTraceCommitment<EF, BooleanPcs<EF, MT, MX>>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
{
    /// Build a commitment over a batch of tables stacking to `num_variables` bits.
    ///
    /// The arity is the one the layout planner derives from the table shapes.
    ///
    /// # Errors
    ///
    /// Returns an error unless the schedule was derived for the committed alphabet.
    ///
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        num_variables: usize,
    ) -> Result<Self, BooleanTraceError<EF, MT::Error>> {
        BooleanPcs::new(config, mmcs, round_mmcs, num_variables)
            .map(Self::from_commitment)
            .map_err(BooleanTraceCommitmentError::Boolean)
    }
}

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
    /// Stack trace tables into an already-built bit commitment.
    pub const fn from_commitment(inner: B) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Variables the stacked bit witness has, so `2^n` bits in all.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    /// The claims one batch raises: one per aligned block of its table's columns.
    ///
    /// Block `b` reads its columns at the first `j_b` coordinates of the column point.
    fn block_claims<'a>(
        run: &'a TableRun,
        point: &'a Point<EF>,
        column_point: &'a Point<EF>,
        current: &'a [EF],
        successor: &'a [EF],
    ) -> impl Iterator<Item = (BitOpening<EF>, BitReadings<EF>)> + 'a {
        let next = run.shape.next;
        run.blocks.iter().map(move |block| {
            let columns = block.first..(block.first + (1 << block.variables)).min(current.len());
            let block_point = Point::new(column_point.as_slice()[..block.variables].to_vec());
            let readings = BitReadings {
                current: Some(Self::combine_columns(
                    &current[columns.clone()],
                    &block_point,
                )),
                next: next.then(|| Self::combine_columns(&successor[columns], &block_point)),
            };
            let mut local = block_point;
            local.extend(point);
            let opening = BitOpening {
                point: block.prefix.lift_prefix(&local),
                row_variables: run.shape.table_variables,
                current: true,
                next,
            };
            (opening, readings)
        })
    }

    /// Evaluate the zero-padded column-value vector at its sampled column point.
    fn combine_columns(values: &[EF], column_point: &Point<EF>) -> EF {
        let padded_len = 1usize << column_point.num_variables();
        let mut padded = Vec::with_capacity(padded_len);
        padded.extend_from_slice(values);
        padded.resize(padded_len, EF::ZERO);
        SplitEq::<EF, EF>::new_packed(column_point, EF::ONE).eval_ext(Poly::new(padded).as_view())
    }
}

/// The committed trace tables, held until the commitment is opened.
pub struct BooleanTraceCommitmentData<EF: Field, D> {
    /// Prover data of the bit commitment underneath.
    inner: D,
    /// Source tables, lent back to whatever evaluates constraints over them.
    tables: Vec<Table<EF>>,
}

/// The retained data of a trace behind the folding-only bit commitment.
pub type BooleanTraceData<EF, MT> = BooleanTraceCommitmentData<EF, BinaryPcsProverData<EF, EF, MT>>;

impl<EF: Field, D> Clone for BooleanTraceCommitmentData<EF, D>
where
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tables: self.tables.clone(),
        }
    }
}

impl<EF: Field, D> BooleanTraceCommitmentData<EF, D> {
    /// One committed table, in the order the tables were supplied.
    ///
    /// # Panics
    ///
    /// Panics on a table index this commitment does not hold.
    #[must_use]
    pub fn table(&self, index: usize) -> &Table<EF> {
        &self.tables[index]
    }
}

/// One opening of a Boolean trace: the column values, and the bit proof behind them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "EF: TowerLevel, P: Serialize",
    deserialize = "EF: TowerLevel, P: Deserialize<'de>"
))]
pub struct BooleanTraceCommitmentProof<EF: Field, P> {
    /// One value per column a batch reads, its current ones first, batches in transcript order.
    pub values: Vec<EF>,
    /// The bit commitment's own proof, answering for every value at once.
    pub opening: P,
}

/// One opening of a trace behind the folding-only bit commitment.
pub type BooleanTraceProof<EF, MT, MX> = BooleanTraceCommitmentProof<EF, BooleanProof<EF, MT, MX>>;

/// Why a trace behind the folding-only bit commitment could not be committed or opened.
pub type BooleanTraceError<EF, MmcsError> =
    BooleanTraceCommitmentError<BooleanPcsError<EF, MmcsError>>;

/// Why a Boolean trace could not be committed or opened.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanTraceCommitmentError<E> {
    /// The bit commitment underneath refused the witness or the opening.
    #[error(transparent)]
    Boolean(E),

    /// The table shapes do not stack to the arity this commitment holds.
    #[error("the shapes stack to {actual} variables, the commitment holds {expected}")]
    StackedArity {
        /// Variables the commitment holds.
        expected: usize,
        /// Variables the shapes stack to.
        actual: usize,
    },

    /// The retained source tables do not have the exact shapes promised by the opening protocol.
    #[error("table {table} has shape {actual:?}, protocol requires {expected:?}")]
    TableShapeMismatch {
        /// Table whose retained shape disagreed with the protocol.
        table: usize,
        /// Shape the protocol describes.
        expected: TableShape,
        /// Shape the committed prover data retains.
        actual: TableShape,
    },

    /// The protocol and retained source table lists have different lengths.
    #[error("the prover retains {actual} tables, while the opening protocol describes {expected}")]
    TableCountMismatch {
        /// Number of tables described by the protocol.
        expected: usize,
        /// Number of tables retained by the prover.
        actual: usize,
    },

    /// A table's bit region is wider than the table itself.
    #[error("table {table} declares {bits} bit columns out of {width}")]
    BitRegionWidth {
        /// Table whose bit region overflows it.
        table: usize,
        /// Columns the bit region names.
        bits: usize,
        /// Columns the table has.
        width: usize,
    },

    /// A trace cell holds neither zero nor one, so it addresses no bit.
    #[error("column {column} of table {table} holds a cell outside the two Boolean values")]
    NonBooleanCell {
        /// Table holding the cell.
        table: usize,
        /// Column holding the cell.
        column: usize,
    },

    /// One point per opening batch is required.
    #[error("{actual} points against {expected} opening batches")]
    PointCount {
        /// Batches the protocol schedules.
        expected: usize,
        /// Points supplied.
        actual: usize,
    },

    /// A point does not name the rows of the table it opens.
    #[error("the point for table {table} names {actual} variables, expected {expected}")]
    PointArity {
        /// Table the point opens.
        table: usize,
        /// Variables that table's rows need.
        expected: usize,
        /// Variables the point names.
        actual: usize,
    },

    /// The proof carries a different number of values than the protocol opens.
    ///
    /// A column read at both rows carries one reading per row, so the count is readings,
    /// one per value position.
    #[error("the proof carries {actual} values against {expected} opened readings")]
    ValueCount {
        /// Readings the protocol opens, one per value position.
        expected: usize,
        /// Values the proof carries.
        actual: usize,
    },

    /// A typed column-batching transcript could not replay its proof shape.
    #[error(transparent)]
    ColumnBatchTranscript(#[from] p3_challenger::fs::TranscriptError),

    /// The inner opening disagreed with the prover's independently computed batch value.
    #[error("column batch {batch} returned an aggregate value different from its claimed columns")]
    ColumnBatchValueMismatch {
        /// Batch whose aggregate value disagreed.
        batch: usize,
    },
}

impl<EF, B, Challenger> MultilinearPcs<EF, Challenger> for BooleanTraceCommitment<EF, B>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    B: BooleanMultilinearPcs<EF, Challenger>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<B::Commitment>,
{
    type Val = EF;
    type Commitment = B::Commitment;
    type ProverData = BooleanTraceCommitmentData<EF, B::ProverData>;
    type Proof = BooleanTraceCommitmentProof<EF, B::Proof>;
    type Error = BooleanTraceCommitmentError<B::Error>;
    type ProverError = BooleanTraceCommitmentError<B::Error>;
    type Witness = Vec<Table<EF>>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.num_variables()
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        // Gathering runs before the transcript is touched, so a refusal leaves it alone.
        let bits =
            tracing::info_span!("gather boolean bits").in_scope(|| self.gather_bits(&witness))?;
        let (commitment, inner) = self
            .inner
            .commit_bits(&bits, challenger)
            .map_err(BooleanTraceCommitmentError::Boolean)?;
        Ok((
            commitment,
            BooleanTraceCommitmentData {
                inner,
                tables: witness,
            },
        ))
    }

    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        self.inner.observe_commitment(commitment, challenger);
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        Self::validate_source_shapes(&prover_data.tables, &protocol)?;
        // The sampled convention draws every batch's point from the transcript.
        let points = sample_points(&protocol, challenger);
        self.open_at(prover_data, &protocol, &points, challenger)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        // Committing bound the root on the prover's sponge, so this binds it here.
        //
        // The points are then sampled from that sponge, as the prover sampled them.
        //
        // The prescribed-point entry point is the one that leaves the binding to its caller.
        self.observe_commitment(commitment, challenger);

        // The same draws the prover made, in the same order, before anything is checked.
        let points = sample_points(&protocol, challenger);
        self.verify_at(commitment, proof, &protocol, &points, challenger)
            .map(|_| ())
    }
}

impl<EF, B, Challenger> PrescribedPointPcs<EF, Challenger> for BooleanTraceCommitment<EF, B>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    B: BooleanMultilinearPcs<EF, Challenger>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<B::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        // A protocol this scheme would refuse gets no assessment, so a caller fails closed.
        let shapes = protocol.table_shapes();
        if plan_stacked_layout(&shapes).0 != self.num_variables() {
            return None;
        }
        // A reduction sends carry and last once its successor view outruns one element.
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        let route = OpeningRoute::new(protocol);
        let mut security = self.inner.readings_security(
            route.num_reductions(),
            route.successor_tensors(&shapes, absorbed),
        )?;
        if let OpeningRoute::Batched(runs) = route {
            // Every combined claim of a batch reads one column point, so each is charged.
            //
            // A table split into several blocks raises one combined claim per block and view.
            //
            // The batching challenge is drawn before any candidate has been named.
            //
            // So it pays the same list the ring-switch reduction below it paid for.
            //
            // A union bound taken once does not shrink the set the next draw faces.
            let claims = runs
                .iter()
                .map(|run| run.shape.num_batches * run.shape.num_views() * run.blocks.len())
                .sum();
            let column_variables = runs
                .iter()
                .map(|run| run.shape.column_variables())
                .max()
                .unwrap_or(0);
            security.charge_reduction(p3_security::multilinear::column_batch_term(
                claims,
                column_variables,
                EF::bits(),
            ));
        }
        Some(security)
    }

    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        // Every shape and every point is checked before the transcript moves.
        Self::validate_source_shapes(&prover_data.tables, protocol)?;
        let placements = self.validate_opening(protocol, points)?;
        let runs = match OpeningRoute::new(protocol) {
            OpeningRoute::PerColumn(plan) => {
                let openings = Self::bit_openings(protocol, plan.claims(), points, &placements);
                let (readings, opening) = self
                    .inner
                    .open_readings(prover_data.inner, &openings, challenger)
                    .map_err(BooleanTraceCommitmentError::Boolean)?;
                let values = plan.values(&readings);
                return Ok(BooleanTraceCommitmentProof { values, opening });
            }
            OpeningRoute::Batched(runs) => runs,
        };

        let BooleanTraceCommitmentData { inner, tables } = prover_data;
        let mut values = Vec::with_capacity(value_count(protocol));
        let mut openings = Vec::new();
        let mut expected = Vec::new();
        let mut batch_points = points.iter();
        for run in &runs {
            let ColumnBatchShape {
                width,
                next,
                num_batches,
                ..
            } = run.shape;
            let run_points: Vec<_> = batch_points.by_ref().take(num_batches).collect();
            let offset = values.len();
            tracing::info_span!("evaluate boolean columns", width, next).in_scope(|| {
                for point in &run_points {
                    let (current, successor) =
                        Self::evaluate_views(&tables[run.table], point, next);
                    values.extend(current);
                    values.extend(successor);
                }
            });

            let mut transcript = ColumnBatchProverTranscript::new(challenger, run.shape);
            let chunks = values[offset..].chunks_exact(run.shape.values_per_batch());
            for (point, batch_values) in run_points.into_iter().zip(chunks) {
                let (current, successor) = batch_values.split_at(width);
                // Both value runs are bound before the point that combines either of them.
                let column_point = transcript.batch(point, current, successor);
                for (opening, readings) in
                    Self::block_claims(run, point, &column_point, current, successor)
                {
                    openings.push(opening);
                    expected.push(readings);
                }
            }
            transcript.finish();
        }

        let (readings, opening) = self
            .inner
            .open_readings(inner, &openings, challenger)
            .map_err(BooleanTraceCommitmentError::Boolean)?;
        if readings.len() != expected.len() {
            return Err(BooleanTraceCommitmentError::ColumnBatchValueMismatch {
                batch: expected.len(),
            });
        }
        for (batch, (actual, expected)) in readings.iter().zip(&expected).enumerate() {
            if actual != expected {
                return Err(BooleanTraceCommitmentError::ColumnBatchValueMismatch { batch });
            }
        }
        Ok(BooleanTraceCommitmentProof { values, opening })
    }

    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<EF>>, Self::Error> {
        // Every shape, point and value count is checked before the transcript moves.
        let placements = self.validate_opening(protocol, points)?;
        let expected_values = value_count(protocol);
        if proof.values.len() != expected_values {
            return Err(BooleanTraceCommitmentError::ValueCount {
                expected: expected_values,
                actual: proof.values.len(),
            });
        }

        let runs = match OpeningRoute::new(protocol) {
            OpeningRoute::PerColumn(plan) => {
                let openings = Self::bit_openings(protocol, plan.claims(), points, &placements);

                // One bit proof answers for every column of every batch at once.
                self.inner
                    .verify_readings(
                        commitment,
                        &openings,
                        &plan.readings(&proof.values),
                        &proof.opening,
                        challenger,
                    )
                    .map_err(BooleanTraceCommitmentError::Boolean)?;
                return Ok(opening_evals(protocol, &proof.values));
            }
            OpeningRoute::Batched(runs) => runs,
        };

        let mut openings = Vec::new();
        let mut readings = Vec::new();
        let mut batch_points = points.iter();
        let mut batch_values = proof.values.as_slice();
        for run in &runs {
            let run_len = run.shape.values_per_batch() * run.shape.num_batches;
            let (run_values, rest) = batch_values.split_at(run_len);
            batch_values = rest;
            let mut transcript = ColumnBatchVerifierTranscript::new(challenger, run.shape);
            for (point, values) in batch_points
                .by_ref()
                .take(run.shape.num_batches)
                .zip(run_values.chunks_exact(run.shape.values_per_batch()))
            {
                let (current, successor) = values.split_at(run.shape.width);
                // Both value runs are bound before the point that combines either of them.
                let column_point = transcript.batch(point, current, successor)?;
                for (opening, reading) in
                    Self::block_claims(run, point, &column_point, current, successor)
                {
                    openings.push(opening);
                    readings.push(reading);
                }
            }
            transcript.finish();
        }

        // One bit proof answers for every batched claim at once.
        self.inner
            .verify_readings(commitment, &openings, &readings, &proof.opening, challenger)
            .map_err(BooleanTraceCommitmentError::Boolean)?;

        Ok(opening_evals(protocol, &proof.values))
    }
}

#[cfg(test)]
mod tests;
