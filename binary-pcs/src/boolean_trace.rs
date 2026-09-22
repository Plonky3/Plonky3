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
//! The slot prefix is fixed before the ring-switch sumcheck starts. For a complete opening
//! of one table, every column at the current row and either no column or every column at
//! the next row, all its columns are combined at one fresh random column point after their
//! claimed values are bound, so each batch uses one ring switch.
//! Other valid protocols retain one reduction per opened column.
//!
//! If one element holds `2^d_log` bits, a column folds `2^max(a - d_log, 0)` elements.
//!
//! Opening `W` equal-height columns shares the row equality weights across columns. The
//! optimized complete-table route scans those weights once per batch and opens one padded
//! stacked point; subset, reordered, mixed-height and mixed-view protocols use the
//! per-column route.
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
//!     one table, every batch reading all columns    ->  one reduction per batch
//!     of both views, or of the current alone            over one shared column point
//!     anything else                                 ->  one reduction per column read
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

use self::plan::{
    OpeningRoute, claim_readings, claim_values, opening_evals, sample_points, value_count,
};
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
        if let OpeningRoute::Batched(shape) = route {
            // Both combined claims of a batch read one column point, so each is charged.
            //
            // The batching challenge is drawn before any candidate has been named.
            security.charge_reduction(p3_security::multilinear::column_batch_term(
                shape.num_batches * shape.num_views(),
                shape.column_variables(),
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
        let shape = match OpeningRoute::new(protocol) {
            OpeningRoute::PerColumn(claims) => {
                let openings = Self::bit_openings(protocol, &claims, points, &placements);
                let (readings, opening) = self
                    .inner
                    .open_readings(prover_data.inner, &openings, challenger)
                    .map_err(BooleanTraceCommitmentError::Boolean)?;
                let values = claim_values(&claims, &readings, value_count(protocol));
                return Ok(BooleanTraceCommitmentProof { values, opening });
            }
            OpeningRoute::Batched(shape) => shape,
        };

        let BooleanTraceCommitmentData { inner, tables } = prover_data;
        let ColumnBatchShape { width, next, .. } = shape;
        let run = shape.values_per_batch();
        let mut values = Vec::with_capacity(run * shape.num_batches);
        tracing::info_span!("evaluate boolean columns", width, next).in_scope(|| {
            for point in points {
                let (current, successor) = Self::evaluate_views(&tables[0], point, next);
                values.extend(current);
                values.extend(successor);
            }
        });

        let mut transcript = ColumnBatchProverTranscript::new(challenger, shape);
        let mut openings = Vec::with_capacity(shape.num_batches);
        let mut expected = Vec::with_capacity(shape.num_batches);
        for (point, batch_values) in points.iter().zip(values.chunks_exact(run)) {
            let (current, successor) = batch_values.split_at(width);
            // Both value runs are bound before the point that combines either of them.
            let column_point = transcript.batch(point, current, successor);
            expected.push(BitReadings {
                current: Some(Self::combine_columns(current, &column_point)),
                next: next.then(|| Self::combine_columns(successor, &column_point)),
            });
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            openings.push(BitOpening {
                point: lifted_point,
                row_variables: shape.table_variables,
                current: true,
                next,
            });
        }
        transcript.finish();

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

        let shape = match OpeningRoute::new(protocol) {
            OpeningRoute::PerColumn(claims) => {
                let openings = Self::bit_openings(protocol, &claims, points, &placements);

                // One bit proof answers for every column of every batch at once.
                self.inner
                    .verify_readings(
                        commitment,
                        &openings,
                        &claim_readings(&claims, &proof.values),
                        &proof.opening,
                        challenger,
                    )
                    .map_err(BooleanTraceCommitmentError::Boolean)?;
                return Ok(opening_evals(protocol, &proof.values));
            }
            OpeningRoute::Batched(shape) => shape,
        };

        let ColumnBatchShape { width, next, .. } = shape;
        let run = shape.values_per_batch();
        let mut transcript = ColumnBatchVerifierTranscript::new(challenger, shape);
        let mut openings = Vec::with_capacity(shape.num_batches);
        let mut readings = Vec::with_capacity(shape.num_batches);
        for (point, batch_values) in points.iter().zip(proof.values.chunks_exact(run)) {
            let (current, successor) = batch_values.split_at(width);
            // Both value runs are bound before the point that combines either of them.
            let column_point = transcript.batch(point, current, successor)?;
            readings.push(BitReadings {
                current: Some(Self::combine_columns(current, &column_point)),
                next: next.then(|| Self::combine_columns(successor, &column_point)),
            });
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            openings.push(BitOpening {
                point: lifted_point,
                row_variables: shape.table_variables,
                current: true,
                next,
            });
        }
        transcript.finish();

        // One bit proof answers for every batched claim at once.
        self.inner
            .verify_readings(commitment, &openings, &readings, &proof.opening, challenger)
            .map_err(BooleanTraceCommitmentError::Boolean)?;

        Ok(opening_evals(protocol, &proof.values))
    }
}

#[cfg(test)]
mod tests;
