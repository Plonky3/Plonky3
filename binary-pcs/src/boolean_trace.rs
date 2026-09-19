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
//! The slot prefix is fixed before the ring-switch sumcheck starts. For a complete
//! current-row opening of one table, all its columns are combined at one fresh random
//! column point after their claimed values are bound, so the batch uses one ring switch.
//! Other valid protocols retain one reduction per selected column.
//!
//! If one element holds `2^d_log` bits, a column folds `2^max(a - d_log, 0)` elements.
//!
//! Opening `W` equal-height columns shares the row equality weights across columns. The
//! optimized complete-table route scans those weights once per batch and opens one padded
//! stacked point; subset, reordered, mixed-height, and successor-free fallback protocols
//! continue to use their existing per-column route.
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
//! # What this does not open
//!
//! A batch may name columns read at the current row, and columns read one row ahead.
//!
//! The reduction underneath turns an evaluation at a point into one about the packing.
//!
//! Reading one row ahead is no evaluation at a point.
//!
//! It weights the column by an equality table shifted up by one row.
//!
//! Such a batch is refused rather than answered at the wrong point.
//!
//! An AIR whose constraints read only the current row asks for none.

use alloc::vec::Vec;

use p3_binary_dft::EncodableLevel;
use p3_binary_field::{PackedGf2, PackedGf2x64, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::layout::{Table, TablePlacement, plan_stacked_layout};
use p3_sumcheck::{
    OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::boolean::{BooleanMultilinearPcs, BooleanPcs, BooleanPcsError, BooleanProof};
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
pub struct BooleanTracePcs<EF: EncodableLevel, MT, MX> {
    /// The bit commitment every column claim is discharged against.
    inner: BooleanPcs<EF, MT, MX>,
}

impl<EF, MT, MX> BooleanTracePcs<EF, MT, MX>
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
    /// Returns an error unless the schedule was derived for `(EF, EF)`.
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        num_variables: usize,
    ) -> Result<Self, BooleanTraceError<EF, MT::Error>> {
        BooleanPcs::new(config, mmcs, round_mmcs, num_variables)
            .map(|inner| Self { inner })
            .map_err(BooleanTraceError::Boolean)
    }

    /// Variables the stacked bit witness has, so `2^n` bits in all.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    /// Where each table's columns land in the bit witness, planned from the shapes.
    ///
    /// # Errors
    ///
    /// Returns an error unless the shapes stack to the committed arity.
    fn placements(
        &self,
        shapes: &[TableShape],
    ) -> Result<Vec<TablePlacement>, BooleanTraceError<EF, MT::Error>> {
        // Prover and verifier both plan from the public shapes, so neither picks its own.
        let (arity, placements) = plan_stacked_layout(shapes);
        if arity != self.num_variables() {
            return Err(BooleanTraceError::StackedArity {
                expected: self.num_variables(),
                actual: arity,
            });
        }
        Ok(placements)
    }

    /// The bit-witness points one opening protocol asks about, in transcript order.
    ///
    /// One point per column of every batch, that batch's point prefixed by the slot address.
    ///
    /// # Errors
    ///
    /// - The shapes do not stack to the committed arity.
    /// - A batch reads a row ahead, which no evaluation at a point answers.
    /// - The point count or a point's arity disagrees with the protocol.
    fn opening_points(
        &self,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
    ) -> Result<Vec<Point<EF>>, BooleanTraceError<EF, MT::Error>> {
        let shapes = protocol.table_shapes();
        let placements = self.validate_opening(protocol, points)?;

        // Placements arrive largest table first, so index them by the table each one owns.
        let mut by_table = alloc::vec![None; shapes.len()];
        for placement in &placements {
            by_table[placement.idx()] = Some(placement);
        }

        let mut lifted = Vec::with_capacity(points.len());
        for ((table, batch), point) in protocol.iter_openings().zip(points) {
            let placement = by_table[table].expect("the planner places every supplied shape");

            // Slot address as the leading coordinates, the row point as the trailing ones.
            for &column in batch.current() {
                lifted.push(placement.selectors()[column].lift_prefix(point));
            }
        }
        Ok(lifted)
    }

    /// Validate all public opening metadata without constructing per-column points.
    fn validate_opening(
        &self,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
    ) -> Result<Vec<TablePlacement>, BooleanTraceError<EF, MT::Error>> {
        let shapes = protocol.table_shapes();
        let placements = self.placements(&shapes)?;
        if points.len() != protocol.num_openings() {
            return Err(BooleanTraceError::PointCount {
                expected: protocol.num_openings(),
                actual: points.len(),
            });
        }

        for ((table, batch), point) in protocol.iter_openings().zip(points) {
            if !batch.next().is_empty() {
                return Err(BooleanTraceError::SuccessorView { table });
            }
            if point.num_variables() != shapes[table].num_variables() {
                return Err(BooleanTraceError::PointArity {
                    table,
                    expected: shapes[table].num_variables(),
                    actual: point.num_variables(),
                });
            }
        }
        Ok(placements)
    }

    /// Validate retained source shapes before any sampled point or opening transcript is used.
    fn validate_source_shapes(
        tables: &[Table<EF>],
        protocol: &OpeningProtocol,
    ) -> Result<(), BooleanTraceError<EF, MT::Error>> {
        let expected = protocol.table_shapes();
        if tables.len() != expected.len() {
            return Err(BooleanTraceError::TableCountMismatch {
                expected: expected.len(),
                actual: tables.len(),
            });
        }
        for (table, expected) in expected.iter().copied().enumerate() {
            let actual = tables[table].shape();
            if actual != expected {
                return Err(BooleanTraceError::TableShapeMismatch {
                    table,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// Bit claims one protocol raises, or nothing for a protocol this scheme refuses.
    fn opening_claim_count(&self, protocol: &OpeningProtocol) -> Option<usize> {
        let shapes = protocol.table_shapes();
        // A shape set that stacks elsewhere describes some other commitment.
        if plan_stacked_layout(&shapes).0 != self.num_variables() {
            return None;
        }
        // One reduction per opened column, and no assessment at all for a successor view.
        protocol
            .iter_openings()
            .map(|(_, batch)| batch.next().is_empty().then(|| batch.current().len()))
            .sum()
    }

    /// Whether this protocol is the complete single-table shape the optimized route handles.
    fn batched_width(&self, protocol: &OpeningProtocol) -> Option<usize> {
        let shapes = protocol.table_shapes();
        if shapes.len() != 1 || protocol.num_openings() == 0 {
            return None;
        }
        let width = shapes[0].width();
        let columns = (0..width).collect::<Vec<_>>();
        if protocol.iter_openings().all(|(table, batch)| {
            table == 0 && batch.next().is_empty() && batch.current() == columns.as_slice()
        }) {
            Some(width)
        } else {
            None
        }
    }

    /// Evaluate every column at one row point while sharing the equality weights.
    fn evaluate_columns(table: &Table<EF>, point: &Point<EF>) -> Vec<EF> {
        let weights = SplitEq::<EF, EF>::new_packed(point, EF::ONE);
        let row_weights = table.packed_bits().map(|_| weights.materialize());
        (0..table.shape().width())
            .map(|column| {
                let view = table.column(column);
                if view.as_dense().is_some() {
                    weights.eval_base(table.poly(column))
                } else {
                    let row_weights = row_weights
                        .as_ref()
                        .expect("packed columns require materialized row weights");
                    let mut result = EF::ZERO;
                    for word in 0..view.len().div_ceil(WORD_BITS) {
                        let mut bits = view
                            .boolean_word(word)
                            .expect("packed column must expose every source word");
                        while bits != 0 {
                            let bit = bits.trailing_zeros() as usize;
                            result += row_weights.as_slice()[word * WORD_BITS + bit];
                            bits &= bits - 1;
                        }
                    }
                    result
                }
            })
            .collect()
    }

    /// Evaluate the zero-padded column-value vector at its sampled column point.
    fn combine_columns(values: &[EF], column_point: &Point<EF>) -> EF {
        let padded_len = 1usize << column_point.num_variables();
        let mut padded = Vec::with_capacity(padded_len);
        padded.extend_from_slice(values);
        padded.resize(padded_len, EF::ZERO);
        SplitEq::<EF, EF>::new_packed(column_point, EF::ONE).eval_ext(Poly::new(padded).as_view())
    }

    /// Gather the Boolean cells of every table into one bit witness.
    ///
    /// # Errors
    ///
    /// - The shapes do not stack to the committed arity.
    /// - A cell holds neither zero nor one, so it addresses no bit.
    fn gather_bits(
        &self,
        tables: &[Table<EF>],
    ) -> Result<Vec<PackedGf2x64>, BooleanTraceError<EF, MT::Error>> {
        let shapes = tables.iter().map(Table::shape).collect::<Vec<_>>();
        let placements = self.placements(&shapes)?;

        // One word per sixty-four bits of the padded witness, the tail left at zero.
        let mut words = alloc::vec![0u64; 1 << (self.num_variables() - 6)];

        for placement in &placements {
            let table = &tables[placement.idx()];
            let column_len = 1usize << table.num_variables();
            for (column, selector) in placement.selectors().iter().enumerate() {
                // A slot starts at a multiple of its own length, counted in cells.
                let offset = selector.index() * column_len;
                let refuse = || BooleanTraceError::NonBooleanCell {
                    table: placement.idx(),
                    column,
                };

                let view = table.column(column);
                if let Some(cells) = view.as_dense() {
                    if column_len >= WORD_BITS {
                        // The slot is word aligned and fills whole words, so none is read back.
                        let (runs, rest) = cells.as_chunks::<WORD_BITS>();
                        debug_assert!(rest.is_empty(), "a power-of-two column fills whole words");
                        for (word, run) in words[offset / WORD_BITS..].iter_mut().zip(runs) {
                            *word = pack_word(run).ok_or_else(refuse)?;
                        }
                    } else {
                        // A shorter column sits inside one word, so its bits are set in place.
                        for (cell, &value) in cells.iter().enumerate() {
                            let bit = bit_of(value).ok_or_else(refuse)?;
                            let index = offset + cell;
                            words[index / WORD_BITS] |= bit << (index % WORD_BITS);
                        }
                    }
                } else if column_len >= WORD_BITS {
                    let source_words = column_len / WORD_BITS;
                    for word in 0..source_words {
                        words[offset / WORD_BITS + word] = view
                            .boolean_word(word)
                            .expect("packed column must expose every source word");
                    }
                } else {
                    let bits = view
                        .boolean_word(0)
                        .expect("packed short column must expose its source word");
                    for cell in 0..column_len {
                        let index = offset + cell;
                        words[index / WORD_BITS] |= ((bits >> cell) & 1) << (index % WORD_BITS);
                    }
                }
            }
        }

        // Lane `j` of a block is bit `j` of its word, which is the packing's own convention.
        Ok(words.into_iter().map(PackedGf2::new).collect())
    }
}

/// The bit of a cell holding zero or one, and nothing for any other value.
#[inline]
fn bit_of<EF: Field>(value: EF) -> Option<u64> {
    // A Boolean trace holds field zero and field one, and no third value.
    if value == EF::ZERO {
        Some(0)
    } else if value == EF::ONE {
        Some(1)
    } else {
        None
    }
}

/// Sixty-four Boolean cells as one word, the lowest cell in the lowest bit.
#[inline]
fn pack_word<EF: Field>(cells: &[EF]) -> Option<u64> {
    // Fold the run into a word; one non-Boolean cell leaves the whole word undefined.
    cells
        .iter()
        .enumerate()
        .try_fold(0u64, |word, (lane, &value)| {
            Some(word | (bit_of(value)? << lane))
        })
}

/// One point per opening batch, drawn from the transcript in batch order.
fn sample_points<EF, Challenger>(
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

/// The committed trace tables, held until the commitment is opened.
pub struct BooleanTraceData<EF: Field, MT: Mmcs<EF>> {
    /// Prover data of the bit commitment underneath.
    inner: BinaryPcsProverData<EF, EF, MT>,
    /// Source tables, lent back to whatever evaluates constraints over them.
    tables: Vec<Table<EF>>,
}

impl<EF: Field, MT: Mmcs<EF>> Clone for BooleanTraceData<EF, MT>
where
    BinaryPcsProverData<EF, EF, MT>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tables: self.tables.clone(),
        }
    }
}

impl<EF: Field, MT: Mmcs<EF>> BooleanTraceData<EF, MT> {
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
    serialize = "EF: TowerLevel, MT::Commitment: Serialize, MT::MultiProof: Serialize, MX::Commitment: Serialize, MX::MultiProof: Serialize",
    deserialize = "EF: TowerLevel, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>, MX::Commitment: Deserialize<'de>, MX::MultiProof: Deserialize<'de>"
))]
pub struct BooleanTraceProof<EF: Field, MT: Mmcs<EF>, MX: Mmcs<EF>> {
    /// One value per opened column, batches in transcript order.
    pub values: Vec<EF>,
    /// The bit commitment's own proof, answering for every value at once.
    pub opening: BooleanProof<EF, MT, MX>,
}

/// Why a Boolean trace could not be committed or opened.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanTraceError<EF, MmcsError> {
    /// The bit commitment underneath refused the witness or the opening.
    #[error(transparent)]
    Boolean(BooleanPcsError<EF, MmcsError>),

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

    /// A batch reads one row ahead, which no evaluation at a point answers.
    #[error("table {table} asks for a successor view, which this commitment cannot open")]
    SuccessorView {
        /// Table whose batch asked for it.
        table: usize,
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
    #[error("the proof carries {actual} values against {expected} opened columns")]
    ValueCount {
        /// Columns the protocol opens.
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

impl<EF, MT, MX, Challenger> MultilinearPcs<EF, Challenger> for BooleanTracePcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    type Val = EF;
    type Commitment = MT::Commitment;
    type ProverData = BooleanTraceData<EF, MT>;
    type Proof = BooleanTraceProof<EF, MT, MX>;
    type Error = BooleanTraceError<EF, MT::Error>;
    type ProverError = BooleanTraceError<EF, MT::Error>;
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
        let bits = self.gather_bits(&witness)?;
        let (commitment, inner) = self
            .inner
            .commit_bits(&bits, challenger)
            .map_err(BooleanTraceError::Boolean)?;
        Ok((
            commitment,
            BooleanTraceData {
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

impl<EF, MT, MX, Challenger> PrescribedPointPcs<EF, Challenger> for BooleanTracePcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        // A protocol this scheme would refuse gets no assessment, so a caller fails closed.
        let shapes = protocol.table_shapes();
        if plan_stacked_layout(&shapes).0 != self.num_variables() {
            return None;
        }
        self.batched_width(protocol).map_or_else(
            || {
                self.opening_claim_count(protocol)
                    .map(|claims| self.inner.opening_security(claims))
            },
            |width| {
                let batches = protocol.num_openings();
                let k = width.next_power_of_two().trailing_zeros() as usize;
                let mut security = self.inner.opening_security(batches);
                security
                    .terms
                    .push(p3_security::multilinear::column_batch_term(
                        batches,
                        k,
                        EF::bits(),
                    ));
                Some(security)
            },
        )
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
        self.validate_opening(protocol, points)?;
        let Some(width) = self.batched_width(protocol) else {
            let lifted = self.opening_points(protocol, points)?;
            let (values, opening) = self
                .inner
                .open_at_points(prover_data.inner, &lifted, challenger)
                .map_err(BooleanTraceError::Boolean)?;
            return Ok(BooleanTraceProof { values, opening });
        };

        let BooleanTraceData { inner, tables } = prover_data;
        let shape = ColumnBatchShape {
            table_variables: protocol.table_shapes()[0].num_variables(),
            width,
            num_batches: protocol.num_openings(),
        };
        let mut values = Vec::with_capacity(width * shape.num_batches);
        for point in points {
            values.extend(Self::evaluate_columns(&tables[0], point));
        }

        let mut transcript = ColumnBatchProverTranscript::new(challenger, shape);
        let mut batched_points = Vec::with_capacity(shape.num_batches);
        let mut expected = Vec::with_capacity(shape.num_batches);
        for (batch, (point, batch_values)) in
            points.iter().zip(values.chunks_exact(width)).enumerate()
        {
            let column_point = transcript.batch(point, batch_values);
            expected.push(Self::combine_columns(batch_values, &column_point));
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            batched_points.push(lifted_point);
            debug_assert_eq!(batch, batched_points.len() - 1);
        }
        transcript.finish();

        let (actual, opening) = self
            .inner
            .open_at_points(inner, &batched_points, challenger)
            .map_err(BooleanTraceError::Boolean)?;
        if actual.len() != expected.len() {
            return Err(BooleanTraceError::ColumnBatchValueMismatch {
                batch: expected.len(),
            });
        }
        for (batch, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            if actual != expected {
                return Err(BooleanTraceError::ColumnBatchValueMismatch { batch });
            }
        }
        Ok(BooleanTraceProof { values, opening })
    }

    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<EF>>, Self::Error> {
        self.validate_opening(protocol, points)?;
        let Some(width) = self.batched_width(protocol) else {
            let lifted = self.opening_points(protocol, points)?;
            if proof.values.len() != lifted.len() {
                return Err(BooleanTraceError::ValueCount {
                    expected: lifted.len(),
                    actual: proof.values.len(),
                });
            }

            // One bit proof answers for every column of every batch at once.
            self.inner
                .verify_at_points(
                    commitment,
                    &lifted,
                    &proof.values,
                    &proof.opening,
                    challenger,
                )
                .map_err(BooleanTraceError::Boolean)?;

            // Split the flat value run back into one batch of current-row values per opening.
            let mut evals = Vec::with_capacity(protocol.num_openings());
            let mut cursor = 0;
            for (_, batch) in protocol.iter_openings() {
                let width = batch.current().len();
                evals.push(OpeningEvals::new(
                    proof.values[cursor..cursor + width].to_vec(),
                    Vec::new(),
                ));
                cursor += width;
            }
            return Ok(evals);
        };

        let expected_values = width * protocol.num_openings();
        if proof.values.len() != expected_values {
            return Err(BooleanTraceError::ValueCount {
                expected: expected_values,
                actual: proof.values.len(),
            });
        }

        let shape = ColumnBatchShape {
            table_variables: protocol.table_shapes()[0].num_variables(),
            width,
            num_batches: protocol.num_openings(),
        };
        let mut transcript = ColumnBatchVerifierTranscript::new(challenger, shape);
        let mut batched_points = Vec::with_capacity(shape.num_batches);
        let mut combined_values = Vec::with_capacity(shape.num_batches);
        for (point, batch_values) in points.iter().zip(proof.values.chunks_exact(width)) {
            let column_point = transcript.batch(point, batch_values)?;
            combined_values.push(Self::combine_columns(batch_values, &column_point));
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            batched_points.push(lifted_point);
        }
        transcript.finish();

        // One bit proof answers for every batched point at once.
        self.inner
            .verify_at_points(
                commitment,
                &batched_points,
                &combined_values,
                &proof.opening,
                challenger,
            )
            .map_err(BooleanTraceError::Boolean)?;

        // Split the flat value run back into one batch of current-row values per opening.
        let mut evals = Vec::with_capacity(protocol.num_openings());
        let mut cursor = 0;
        for (_, batch) in protocol.iter_openings() {
            let width = batch.current().len();
            evals.push(OpeningEvals::new(
                proof.values[cursor..cursor + width].to_vec(),
                Vec::new(),
            ));
            cursor += width;
        }
        Ok(evals)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::{OpeningBatch, PrescribedPointPcs, TableSpec};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::params::BinaryPcsParams;
    use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs};

    type EF = BinaryField128;

    /// Columns every fixture table holds.
    const FIXTURE_WIDTH: usize = 2;

    /// A Boolean table of `2^log_height` rows and two columns.
    fn table(seed: u64, log_height: usize) -> Table<EF> {
        let rows = 1usize << log_height;
        let mut rng = SmallRng::seed_from_u64(seed);
        let cells = (0..FIXTURE_WIDTH * rows)
            .map(|_| EF::from_bool(rng.random::<bool>()))
            .collect();
        Table::new(RowMajorMatrix::new(cells, rows))
    }

    /// A Boolean table with an explicitly chosen width.
    fn table_with_width(seed: u64, log_height: usize, width: usize) -> Table<EF> {
        let rows = 1usize << log_height;
        let mut rng = SmallRng::seed_from_u64(seed);
        let cells = (0..width * rows)
            .map(|_| EF::from_bool(rng.random::<bool>()))
            .collect();
        Table::new(RowMajorMatrix::new(cells, rows))
    }

    fn packed_table(table: &Table<EF>) -> Table<EF> {
        let height = 1usize << table.num_variables();
        let words = (0..height.div_ceil(WORD_BITS))
            .flat_map(|block| {
                (0..table.num_polys()).map(move |column| {
                    (0..WORD_BITS).fold(0u64, |word, lane| {
                        let row = block * WORD_BITS + lane;
                        if row < height && table.column(column).value(row) == EF::ONE {
                            word | (1u64 << lane)
                        } else {
                            word
                        }
                    })
                })
            })
            .collect();
        Table::from_packed_bits(
            RowMajorMatrix::new(words, table.num_polys()),
            table.num_variables(),
        )
    }

    /// A commitment over the batch these shapes describe.
    fn pcs(shapes: &[TableShape]) -> BooleanTracePcs<EF, MyMmcs, MyMmcs> {
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };
        let (arity, _) = plan_stacked_layout(shapes);
        // The packing absorbs seven variables, so the commitment holds the rest.
        let config = BinaryPcsConfig::try_new::<EF, EF>(arity - 7, params).unwrap();
        BooleanTracePcs::new(config, mmcs(), mmcs(), arity).unwrap()
    }

    /// Every column of every table, opened at one point per table.
    fn protocol(shapes: &[TableShape]) -> OpeningProtocol {
        OpeningProtocol::new(
            shapes
                .iter()
                .map(|shape| {
                    TableSpec::new(
                        *shape,
                        alloc::vec![OpeningBatch::new((0..shape.width()).collect(), Vec::new())],
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn an_opened_column_is_the_column_the_table_holds() {
        // Invariant: lifting a row point by a slot address reads that column and no other.
        //
        // Fixture state: three tables, so the slots are neither all one arity nor all aligned.
        //
        //     - table 0   2^10 rows, 2 columns
        //     - table 1   2^8  rows, 2 columns
        //     - table 2   2^4  rows, 2 columns
        //
        // The stack is log2_ceil(2048 + 512 + 32) = 12 variables wide.
        //
        // The third table is what puts a column shorter than one staging word in the batch.
        //
        //     - 2^10 and 2^8 rows  ->  whole words, written by the aligned run
        //     - 2^4 rows           ->  sixteen bits inside one word, set in place
        //
        // A wrong shift on that second path would place the column somewhere else.
        //
        // Every opened value is checked against the column's own multilinear.
        // That reference shares nothing with the packing or with the reduction.
        let shapes = [
            TableShape::new(10, FIXTURE_WIDTH),
            TableShape::new(8, FIXTURE_WIDTH),
            TableShape::new(4, FIXTURE_WIDTH),
        ];
        let tables = alloc::vec![table(0xB100, 10), table(0xB101, 8), table(0xB103, 4)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);
        assert_eq!(scheme.num_variables(), 12);

        let mut rng = SmallRng::seed_from_u64(0xB102);
        let points = alloc::vec![
            Point::<EF>::rand(&mut rng, 10),
            Point::<EF>::rand(&mut rng, 8),
            Point::<EF>::rand(&mut rng, 4),
        ];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(tables.clone(), &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();

        // One batch per table, each carrying that table's whole width and no successor value.
        assert_eq!(evals.len(), shapes.len());
        for (index, batch) in evals.iter().enumerate() {
            assert!(batch.next().is_empty());
            for (column, &value) in batch.current().iter().enumerate() {
                let reference = Poly::new(tables[index].poly(column).as_slice().to_vec());
                assert_eq!(
                    value,
                    reference.eval_base(&points[index]),
                    "{index}/{column}"
                );
            }
        }
    }

    #[test]
    fn packed_table_round_trip_matches_dense_openings() {
        for (log_height, width, seed) in [(5, 8, 0xB10A), (10, FIXTURE_WIDTH, 0xB10C)] {
            let shapes = [TableShape::new(log_height, width)];
            let dense = table_with_width(seed, log_height, width);
            let packed = packed_table(&dense);
            let scheme = pcs(&shapes);
            let protocol = protocol(&shapes);
            let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(seed + 1), log_height);

            let mut dense_challenger = challenger();
            let (dense_commitment, dense_data) =
                scheme.commit(vec![dense], &mut dense_challenger).unwrap();
            let dense_proof = scheme
                .open_at(
                    dense_data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut dense_challenger,
                )
                .unwrap();
            let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);
            let mut packed_challenger = challenger();
            let (packed_commitment, packed_data) =
                scheme.commit(vec![packed], &mut packed_challenger).unwrap();
            let packed_proof = scheme
                .open_at(
                    packed_data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut packed_challenger,
                )
                .unwrap();
            let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

            assert_eq!(dense_commitment, packed_commitment);
            assert_eq!(
                postcard::to_allocvec(&dense_proof).unwrap(),
                postcard::to_allocvec(&packed_proof).unwrap()
            );
            assert_eq!(dense_after_open, packed_after_open);
            let mut verifier = challenger();
            scheme.observe_commitment(&packed_commitment, &mut verifier);
            scheme
                .verify_at(
                    &packed_commitment,
                    &packed_proof,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier,
                )
                .unwrap();
        }
    }

    #[test]
    fn packed_mixed_heights_and_widths_match_dense_proof_bytes() {
        let shapes = [
            TableShape::new(6, 3),
            TableShape::new(4, 5),
            TableShape::new(7, 2),
        ];
        let dense = vec![
            table_with_width(0xB10C, 6, 3),
            table_with_width(0xB10D, 4, 5),
            table_with_width(0xB10E, 7, 2),
        ];
        let packed = dense.iter().map(packed_table).collect::<Vec<_>>();
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);
        let mut rng = SmallRng::seed_from_u64(0xB10F);
        let points = shapes
            .iter()
            .map(|shape| Point::<EF>::rand(&mut rng, shape.num_variables()))
            .collect::<Vec<_>>();

        let mut dense_challenger = challenger();
        let (dense_commitment, dense_data) = scheme.commit(dense, &mut dense_challenger).unwrap();
        let dense_proof = scheme
            .open_at(dense_data, &protocol, &points, &mut dense_challenger)
            .unwrap();
        let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);

        let mut packed_challenger = challenger();
        let (packed_commitment, packed_data) =
            scheme.commit(packed, &mut packed_challenger).unwrap();
        let packed_proof = scheme
            .open_at(packed_data, &protocol, &points, &mut packed_challenger)
            .unwrap();
        let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

        assert_eq!(dense_commitment, packed_commitment);
        assert_eq!(
            postcard::to_allocvec(&dense_proof).unwrap(),
            postcard::to_allocvec(&packed_proof).unwrap()
        );
        assert_eq!(dense_after_open, packed_after_open);
    }

    #[test]
    fn the_sampled_path_draws_the_same_points_on_both_sides() {
        // Invariant: the sampled convention needs no point to cross the wire.
        // Each side binds the commitment itself.
        //
        //     commit  ->  binds the root, then the caller draws nothing of its own
        //     verify  ->  binds the root, then samples the points the prover sampled
        //
        // A caller that bound the root by hand would bind it twice and sample elsewhere.
        // Both sides therefore run on a fresh sponge, as every other scheme here expects.
        //
        // Fixture state: one table of 2^10 rows and two columns, so the stack has arity 11.
        let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(alloc::vec![table(0xB200, 10)], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open(data, protocol.clone(), &mut prover_chal)
            .unwrap();

        let mut verifier_chal = challenger();
        scheme
            .verify(&commitment, &proof, &mut verifier_chal, protocol.clone())
            .unwrap();

        // Binding it a second time moves every later draw, so the two sides split.
        let mut double_bound = challenger();
        scheme.observe_commitment(&commitment, &mut double_bound);
        assert!(
            scheme
                .verify(&commitment, &proof, &mut double_bound, protocol)
                .is_err()
        );
    }

    #[test]
    fn a_disagreeing_shape_or_point_count_is_refused() {
        // Invariant: the shape agreement is checked before the transcript moves at all.
        let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
        let scheme = pcs(&shapes);

        // Mutation: one column fewer changes the retained table shape.
        let narrow = [TableShape::new(10, 1)];
        let mut chal = challenger();
        let (_, data) = scheme
            .commit(alloc::vec![table(0xB300, 10)], &mut chal)
            .unwrap();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB301), 10);
        let Err(error) = scheme.open_at(
            data,
            &protocol(&narrow),
            core::slice::from_ref(&point),
            &mut chal,
        ) else {
            panic!("a shape set stacking elsewhere describes another commitment")
        };
        let BooleanTraceError::TableShapeMismatch {
            table: table_index,
            expected,
            actual,
        } = error
        else {
            panic!("the retained source shape must be checked before stacking")
        };
        assert_eq!(table_index, 0);
        assert_eq!(expected, TableShape::new(10, 1));
        assert_eq!(actual, TableShape::new(10, FIXTURE_WIDTH));

        // Mutation: no point at all, against a protocol scheduling one batch.
        let (_, data) = scheme
            .commit(alloc::vec![table(0xB300, 10)], &mut chal)
            .unwrap();
        let Err(error) = scheme.open_at(data, &protocol(&shapes), &[], &mut chal) else {
            panic!("a batch with no point is opened nowhere")
        };
        assert!(matches!(
            error,
            BooleanTraceError::PointCount {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn a_proof_short_of_one_value_is_refused() {
        // Invariant: the value run must cover every opened column, one value each.
        //
        // Fixture state: one table of two columns, so an honest proof carries two values.
        //
        // Mutation: drop the last value, leaving one against two opened columns.
        let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(alloc::vec![table(0xB400, 10)], &mut prover_chal)
            .unwrap();
        let points = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB401), 10)];
        let mut proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        proof.values.pop();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let Err(error) =
            scheme.verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        else {
            panic!("a proof answering for one column cannot answer for two")
        };
        assert!(matches!(
            error,
            BooleanTraceError::ValueCount {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn a_single_table_batch_uses_one_reduction_for_all_columns() {
        // Invariant: one complete current-row batch for one table is discharged by one
        // column-point ring switch, regardless of the table width.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
        )]);
        let table = table_with_width(0xB500, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB501), 8);

        let mut prover_chal = challenger();
        let (_, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &[point], &mut prover_chal)
            .unwrap();

        assert_eq!(proof.values.len(), 3);
        assert_eq!(proof.opening.reductions.len(), 1);
    }

    #[test]
    fn batched_values_are_bound_before_the_column_point() {
        // Changing a claimed column value changes the verifier's column point and therefore
        // cannot be repaired by reusing the original one-reduction proof.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
        )]);
        let table = table_with_width(0xB502, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB503), 8);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let mut proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();
        proof.values[0] += EF::ONE;

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        assert!(
            scheme
                .verify_at(&commitment, &proof, &protocol, &[point], &mut verifier_chal)
                .is_err()
        );
    }

    #[test]
    fn complete_batches_support_multiple_points_and_width_one() {
        // Width one has no column-point coordinates; two opening batches still use two
        // independent inner reductions and preserve the ordinary value return format.
        let shape = TableShape::new(8, 1);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![
                OpeningBatch::new(vec![0], Vec::new()),
                OpeningBatch::new(vec![0], Vec::new()),
            ],
        )]);
        let table = table_with_width(0xB504, 8, 1);
        let mut rng = SmallRng::seed_from_u64(0xB505);
        let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        assert_eq!(proof.values.len(), 2);
        assert_eq!(proof.opening.reductions.len(), 2);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();
        for (point, eval) in points.iter().zip(evals) {
            let reference = Poly::new(table.poly(0).as_slice().to_vec());
            assert_eq!(eval.current()[0], reference.eval_base(point));
        }
    }

    #[test]
    fn optimized_security_charges_batches_and_column_coordinates() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
            ],
        )]);

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol)
        .unwrap();
        let batching = security
            .terms
            .iter()
            .find(|term| term.label == "column-batching")
            .unwrap();
        assert!((batching.bits.bits() - 126.0).abs() < 1e-9);
        let ring_switch = security
            .terms
            .iter()
            .find(|term| term.label == "bit-ring-switch")
            .unwrap();
        assert!(ring_switch.bits.bits().is_finite());
    }

    #[test]
    fn optimized_security_rejects_a_protocol_with_the_wrong_stacked_arity() {
        let committed_shape = TableShape::new(8, 3);
        let scheme = pcs(&[committed_shape]);
        let protocol_shape = TableShape::new(8, 2);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            protocol_shape,
            vec![OpeningBatch::new(vec![0, 1], Vec::new())],
        )]);

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol);
        assert!(security.is_none());
    }

    #[test]
    fn non_power_two_complete_batches_round_trip_below_and_above_packing_width() {
        // Widths three and five exercise zero padding, while heights below and above 128
        // rows cover the short-column-in-word and whole-word packing layouts.
        for (width, log_height, num_batches, seed) in [
            (3, 6, 2, 0xB506),
            (3, 8, 2, 0xB507),
            (5, 6, 1, 0xB508),
            (5, 8, 1, 0xB509),
        ] {
            let shape = TableShape::new(log_height, width);
            let scheme = pcs(&[shape]);
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                shape,
                (0..num_batches)
                    .map(|_| OpeningBatch::new((0..width).collect(), Vec::new()))
                    .collect(),
            )]);
            let table = table_with_width(seed, log_height, width);
            let mut rng = SmallRng::seed_from_u64(seed + 1);
            let points = (0..num_batches)
                .map(|_| Point::<EF>::rand(&mut rng, log_height))
                .collect::<Vec<_>>();

            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table.clone()], &mut prover_chal)
                .unwrap();
            let proof = scheme
                .open_at(data, &protocol, &points, &mut prover_chal)
                .unwrap();
            assert_eq!(proof.opening.reductions.len(), num_batches);

            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            let evals = scheme
                .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
                .unwrap();
            for (batch, point) in points.iter().enumerate() {
                for column in 0..width {
                    let reference = Poly::new(table.poly(column).as_slice().to_vec());
                    assert_eq!(evals[batch].current()[column], reference.eval_base(point));
                }
            }
        }
    }

    #[test]
    fn optimized_proofs_reject_each_column_tampering_reduction_tampering_and_point_reordering() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
            ],
        )]);
        let table = table_with_width(0xB50A, 8, 3);
        let mut rng = SmallRng::seed_from_u64(0xB50B);
        let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];
        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();

        // Untampered control.
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();

        for index in 0..proof.values.len() {
            let mut tampered = proof.clone();
            tampered.values[index] += EF::ONE;
            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            assert!(
                scheme
                    .verify_at(
                        &commitment,
                        &tampered,
                        &protocol,
                        &points,
                        &mut verifier_chal
                    )
                    .is_err()
            );
        }

        let mut tampered = proof.clone();
        tampered.opening.reductions.pop();
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        assert!(
            scheme
                .verify_at(
                    &commitment,
                    &tampered,
                    &protocol,
                    &points,
                    &mut verifier_chal
                )
                .is_err()
        );

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let reordered = vec![points[1].clone(), points[0].clone()];
        assert!(
            scheme
                .verify_at(
                    &commitment,
                    &proof,
                    &protocol,
                    &reordered,
                    &mut verifier_chal
                )
                .is_err()
        );
    }

    #[test]
    fn optimized_shape_errors_leave_the_prover_transcript_untouched() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], vec![0])],
        )]);
        let mut prover_chal = challenger();
        let (_, data) = scheme
            .commit(vec![table_with_width(0xB50C, 8, 3)], &mut prover_chal)
            .unwrap();
        let mut expected = prover_chal.clone();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB50D), 8);
        let error = match scheme.open_at(data, &protocol, &[point], &mut prover_chal) {
            Ok(_) => panic!("successor views must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BooleanTraceError::SuccessorView { table: 0 }
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
            p3_challenger::CanSample::<EF>::sample(&mut expected),
        );

        let valid_protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
        )]);
        let mut prover_chal = challenger();
        let (_, data) = scheme
            .commit(vec![table_with_width(0xB510, 8, 3)], &mut prover_chal)
            .unwrap();
        let mut expected = prover_chal.clone();
        let bad_point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB511), 7);
        let error = match scheme.open_at(data, &valid_protocol, &[bad_point], &mut prover_chal) {
            Ok(_) => panic!("a point with the wrong arity must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BooleanTraceError::PointArity { table: 0, .. }
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
            p3_challenger::CanSample::<EF>::sample(&mut expected),
        );

        let narrow_shape = TableShape::new(8, 2);
        let narrow_protocol = OpeningProtocol::new(vec![TableSpec::new(
            narrow_shape,
            vec![OpeningBatch::new(vec![0, 1], Vec::new())],
        )]);
        let mut prover_chal = challenger();
        let (_, data) = scheme
            .commit(vec![table_with_width(0xB512, 8, 3)], &mut prover_chal)
            .unwrap();
        let mut expected = prover_chal.clone();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB513), 8);
        let error = match scheme.open_at(data, &narrow_protocol, &[point], &mut prover_chal) {
            Ok(_) => panic!("a protocol with the wrong stacked arity must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BooleanTraceError::TableShapeMismatch { .. }
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
            p3_challenger::CanSample::<EF>::sample(&mut expected),
        );
    }

    #[test]
    fn source_shape_errors_precede_sampling_for_open_and_open_at() {
        for (committed_shapes, protocol_shapes, expected) in [
            (
                vec![TableShape::new(8, 3)],
                vec![TableShape::new(8, 4)],
                "shape",
            ),
            (
                vec![TableShape::new(8, 3)],
                vec![TableShape::new(9, 2)],
                "shape",
            ),
            (
                vec![TableShape::new(8, 2), TableShape::new(8, 2)],
                vec![TableShape::new(8, 4)],
                "count",
            ),
        ] {
            let scheme = pcs(&committed_shapes);
            let tables = committed_shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    table_with_width(0xB520 + index as u64, shape.num_variables(), shape.width())
                })
                .collect();
            let sampled_protocol = protocol(&protocol_shapes);
            let mut sampled_challenger = challenger();
            let (_, data) = scheme.commit(tables, &mut sampled_challenger).unwrap();
            let mut expected_challenger = sampled_challenger.clone();
            let error = match scheme.open(data, sampled_protocol, &mut sampled_challenger) {
                Ok(_) => panic!("a source shape mismatch must be rejected before sampling"),
                Err(error) => error,
            };
            match expected {
                "shape" => assert!(matches!(
                    error,
                    BooleanTraceError::TableShapeMismatch { .. }
                )),
                "count" => assert!(matches!(
                    error,
                    BooleanTraceError::TableCountMismatch { .. }
                )),
                _ => unreachable!(),
            }
            assert_eq!(
                p3_challenger::CanSample::<EF>::sample(&mut sampled_challenger),
                p3_challenger::CanSample::<EF>::sample(&mut expected_challenger),
            );

            let protocol = protocol(&protocol_shapes);
            let points = protocol_shapes
                .iter()
                .map(|shape| {
                    Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB530), shape.num_variables())
                })
                .collect::<Vec<_>>();
            let mut challenger = challenger();
            let (_, data) = scheme
                .commit(
                    committed_shapes
                        .iter()
                        .enumerate()
                        .map(|(index, shape)| {
                            table_with_width(
                                0xB540 + index as u64,
                                shape.num_variables(),
                                shape.width(),
                            )
                        })
                        .collect(),
                    &mut challenger,
                )
                .unwrap();
            let mut expected_challenger = challenger.clone();
            let error = match scheme.open_at(data, &protocol, &points, &mut challenger) {
                Ok(_) => panic!("a source shape mismatch must be rejected before opening"),
                Err(error) => error,
            };
            assert!(matches!(
                (expected, error),
                ("shape", BooleanTraceError::TableShapeMismatch { .. })
                    | ("count", BooleanTraceError::TableCountMismatch { .. })
            ));
            assert_eq!(
                p3_challenger::CanSample::<EF>::sample(&mut challenger),
                p3_challenger::CanSample::<EF>::sample(&mut expected_challenger),
            );
        }
    }

    #[test]
    fn reordered_subset_batches_use_the_fallback_column_route() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![2, 0], Vec::new())],
        )]);
        let table = table_with_width(0xB50E, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB50F), 8);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();
        assert_eq!(proof.values.len(), 2);
        assert_eq!(proof.opening.reductions.len(), 2);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_chal,
            )
            .unwrap();
        assert_eq!(
            evals[0].current(),
            &[
                Poly::new(table.poly(2).as_slice().to_vec()).eval_base(&point),
                Poly::new(table.poly(0).as_slice().to_vec()).eval_base(&point),
            ]
        );
    }
}
