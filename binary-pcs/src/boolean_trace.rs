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
//! If one element holds `2^d_log` bits, a column folds `2^max(a - d_log, 0)` elements.
//!
//! Opening `W` equal-height columns costs `W * 2^max(a - d_log, 0)` field work.
//! It does not scan the `W`-column stack once per column.
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
use p3_sumcheck::layout::{Table, TablePlacement, plan_stacked_layout};
use p3_sumcheck::{
    OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::boolean::{BooleanMultilinearPcs, BooleanPcs, BooleanPcsError, BooleanProof};
use crate::fold::FoldAlphabet;
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
    EF: EncodableLevel + TranscriptField + TowerLevel + FoldAlphabet<EF> + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
{
    /// Build a commitment over a batch of tables stacking to `num_variables` bits.
    ///
    /// The arity is the one the layout planner derives from the table shapes.
    ///
    /// # Errors
    ///
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
        let placements = self.placements(&shapes)?;
        if points.len() != protocol.num_openings() {
            return Err(BooleanTraceError::PointCount {
                expected: protocol.num_openings(),
                actual: points.len(),
            });
        }

        // Placements arrive largest table first, so index them by the table each one owns.
        let mut by_table = alloc::vec![None; shapes.len()];
        for placement in &placements {
            by_table[placement.idx()] = Some(placement);
        }

        let mut lifted = Vec::with_capacity(points.len());
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
            let placement = by_table[table].expect("the planner places every supplied shape");

            // Slot address as the leading coordinates, the row point as the trailing ones.
            for &column in batch.current() {
                lifted.push(placement.selectors()[column].lift_prefix(point));
            }
        }
        Ok(lifted)
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
                let view = table.poly(column);
                let cells = view.as_slice();
                let refuse = || BooleanTraceError::NonBooleanCell {
                    table: placement.idx(),
                    column,
                };

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
}

impl<EF, MT, MX, Challenger> MultilinearPcs<EF, Challenger> for BooleanTracePcs<EF, MT, MX>
where
    EF: EncodableLevel + TranscriptField + TowerLevel + FoldAlphabet<EF> + Coordinates,
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
    EF: EncodableLevel + TranscriptField + TowerLevel + FoldAlphabet<EF> + Coordinates,
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
        self.opening_claim_count(protocol)
            .map(|claims| self.inner.opening_security(claims))
    }

    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        // Every shape and every point is checked before the transcript moves.
        let lifted = self.opening_points(protocol, points)?;
        let (values, opening) = self
            .inner
            .open_at_points(prover_data.inner, &lifted, challenger)
            .map_err(BooleanTraceError::Boolean)?;
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
        Ok(evals)
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::{OpeningBatch, TableSpec};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::params::BinaryPcsParams;
    use crate::test_util::{MyMmcs, challenger, mmcs};

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

        // Mutation: one column fewer stacks to arity 10, not the 11 committed.
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
        assert!(matches!(
            error,
            BooleanTraceError::StackedArity {
                expected: 11,
                actual: 10
            }
        ));

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
}
