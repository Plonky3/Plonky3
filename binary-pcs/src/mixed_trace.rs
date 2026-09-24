//! Tables that hold both bit columns and field-element columns, committed as one bit witness.
//!
//! ```text
//!     row j   [ b_0 ... b_{n-1} | v_0 ... v_{m-1} ]
//!               bit region        dense region
//! ```
//!
//! # Committing the dense region
//!
//! A dense cell is a run of `F_2` coordinates, as its bytes already lay it out.
//!
//! Each dense column becomes one bit column per coordinate:
//!
//! ```text
//!     v(j)  =  sum_k c_k(j) * e_k        e_k the element with only coordinate k set
//! ```
//!
//! A dense cell therefore commits exactly the bits of one field element.
//!
//! The bit region commits one bit per cell, as the Boolean trace commitment does.
//!
//! # Opening the dense region
//!
//! The multilinear extension is linear, so the identity above holds at every point:
//!
//! ```text
//!     v(r)  =  sum_k c_k(r) * e_k
//! ```
//!
//! A claim on a dense column is answered by the claims on its coordinate columns.
//!
//! The verifier recombines them itself, so the recombination draws nothing.
//!
//! Each batch is widened to its whole committed table, which makes it complete.
//!
//! So the coordinate columns share one column-batching reduction per aligned block.
//!
//! # Soundness
//!
//! The coordinate map is a bijection between field elements and bit strings.
//!
//! So every dense column has exactly one committed preimage.
//!
//! The recombination is deterministic, so it adds no term to the error.
//!
//! Every draw is made by the Boolean trace commitment underneath, which charges it.
//!
//! That includes the column point of each batch, charged once per block and view.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use p3_binary_field::{BinaryField8, BitCoordinates, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use p3_sumcheck::{
    OpeningBatch, OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs,
    TableShape, TableSpec,
};

use crate::boolean::{BooleanMultilinearPcs, BooleanPcs};
use crate::boolean_trace::{
    BooleanTraceCommitment, BooleanTraceCommitmentData, BooleanTraceCommitmentError,
    BooleanTraceCommitmentProof,
};
use crate::packing::{Coordinates, coordinate_bytes, pack};

/// Rows one packed word holds.
const WORD_BITS: usize = 64;

/// The coordinate basis of `EF`: element `k` has only coordinate `k` set.
///
/// A cell `v` is `sum_k c_k * e_k`, where `c_k` is bit `k` of its little-endian bytes.
///
/// A word view over these elements reads bits as the cell whose low coordinates they are.
#[must_use]
pub fn coordinate_basis<EF: Coordinates>() -> Vec<EF> {
    let bytes = EF::COORDINATES / 8;
    (0..EF::COORDINATES)
        .map(|k| {
            let mut cells = vec![BinaryField8::from_le_bytes([0]); bytes];
            cells[k / 8] = BinaryField8::from_le_bytes([1 << (k % 8)]);
            pack::<BinaryField8, EF>(&cells)[0]
        })
        .collect()
}

/// The shapes a batch commits once every dense column is split into its coordinates.
///
/// # Panics
///
/// Panics if the two slices differ in length, or a bit region is wider than its table.
#[must_use]
pub fn committed_shapes<EF: Coordinates>(
    shapes: &[TableShape],
    boolean_columns: &[usize],
) -> Vec<TableShape> {
    assert_eq!(
        shapes.len(),
        boolean_columns.len(),
        "one bit region per table"
    );
    shapes
        .iter()
        .zip(boolean_columns)
        .map(|(shape, &bits)| {
            assert!(bits <= shape.width(), "a bit region fits inside its table");
            let width = bits + (shape.width() - bits) * EF::COORDINATES;
            TableShape::new(shape.num_variables(), width)
        })
        .collect()
}

/// Committed columns one source column opens, in the order they are committed.
///
/// A bit column opens itself, and a dense column opens one column per coordinate.
const fn committed_columns<EF: Coordinates>(bits: usize, column: usize) -> Range<usize> {
    if column < bits {
        column..column + 1
    } else {
        let start = bits + (column - bits) * EF::COORDINATES;
        start..start + EF::COORDINATES
    }
}

/// A commitment to tables with a bit region and a dense region, one bit witness underneath.
///
/// Table `t` holds bits in its first `boolean_columns[t]` columns, field elements after.
///
/// The split is part of the statement, so prover and verifier build it from the same tables.
pub struct MixedTraceCommitment<EF, B> {
    /// The Boolean trace commitment every claim is discharged against.
    inner: BooleanTraceCommitment<EF, B>,
    /// Width of each table's bit region, in table order.
    boolean_columns: Vec<usize>,
    /// The coordinate basis dense claims are recombined over.
    basis: Vec<EF>,
}

/// The mixed trace commitment discharged through the folding-only bit commitment.
pub type MixedTracePcs<EF, MT, MX> = MixedTraceCommitment<EF, BooleanPcs<EF, MT, MX>>;

impl<EF: Coordinates, B> MixedTraceCommitment<EF, B> {
    /// Split each table at its bit region, and commit through `inner`.
    ///
    /// `inner` must hold the arity [`committed_shapes`] stacks to.
    pub fn new(inner: BooleanTraceCommitment<EF, B>, boolean_columns: Vec<usize>) -> Self {
        Self {
            inner,
            boolean_columns,
            basis: coordinate_basis(),
        }
    }

    /// Width of each table's bit region, in table order.
    #[must_use]
    pub fn boolean_columns(&self) -> &[usize] {
        &self.boolean_columns
    }

    /// The Boolean trace commitment underneath.
    #[must_use]
    pub const fn inner(&self) -> &BooleanTraceCommitment<EF, B> {
        &self.inner
    }
}

impl<EF: Field + Coordinates, B> MixedTraceCommitment<EF, B> {
    /// Refuse a protocol whose table count or widths disagree with the bit regions.
    fn check_regions<E>(
        &self,
        shapes: &[TableShape],
    ) -> Result<(), BooleanTraceCommitmentError<E>> {
        if shapes.len() != self.boolean_columns.len() {
            return Err(BooleanTraceCommitmentError::TableCountMismatch {
                expected: self.boolean_columns.len(),
                actual: shapes.len(),
            });
        }
        for (table, (shape, &bits)) in shapes.iter().zip(&self.boolean_columns).enumerate() {
            if bits > shape.width() {
                return Err(BooleanTraceCommitmentError::BitRegionWidth {
                    table,
                    bits,
                    width: shape.width(),
                });
            }
        }
        Ok(())
    }

    /// The same schedule over the committed columns, each batch widened to its whole table.
    ///
    /// A table read one row ahead anywhere is read one row ahead in all of its batches.
    ///
    /// Every batch is then complete, so each table costs one reduction per aligned column block.
    ///
    /// The extra readings are opened and verified like the others, and dropped on the way back.
    fn expand_protocol(&self, protocol: &OpeningProtocol) -> OpeningProtocol {
        let shapes = protocol.table_shapes();
        let committed = committed_shapes::<EF>(&shapes, &self.boolean_columns);
        let mut successor = vec![false; shapes.len()];
        for (table, batch) in protocol.iter_openings() {
            successor[table] |= !batch.next().is_empty();
        }
        let mut schedules = vec![Vec::new(); shapes.len()];
        for (table, _) in protocol.iter_openings() {
            let all: Vec<usize> = (0..committed[table].width()).collect();
            let next = if successor[table] {
                all.clone()
            } else {
                Vec::new()
            };
            schedules[table].push(OpeningBatch::new(all, next));
        }
        OpeningProtocol::new(
            committed
                .into_iter()
                .zip(schedules)
                .map(|(shape, schedule)| TableSpec::new(shape, schedule))
                .collect(),
        )
    }

    /// Read each requested column back out of its table's committed readings.
    ///
    /// A dense column folds its coordinate readings into one value.
    fn recombine(
        &self,
        protocol: &OpeningProtocol,
        evals: &[OpeningEvals<EF>],
    ) -> Vec<OpeningEvals<EF>> {
        protocol
            .iter_openings()
            .zip(evals)
            .map(|((table, batch), committed)| {
                let bits = self.boolean_columns[table];
                let side = |columns: &[usize], values: &[EF]| {
                    columns
                        .iter()
                        .map(|&column| {
                            let range = committed_columns::<EF>(bits, column);
                            if column < bits {
                                values[range.start]
                            } else {
                                // v(r) = sum_k c_k(r) * e_k
                                values[range]
                                    .iter()
                                    .zip(&self.basis)
                                    .map(|(&value, &element)| value * element)
                                    .sum()
                            }
                        })
                        .collect()
                };
                OpeningBatch::new(
                    side(batch.current(), committed.current()),
                    side(batch.next(), committed.next()),
                )
            })
            .collect()
    }

    /// Lay one table out as packed bits: its bit region, then each dense cell's coordinates.
    fn expand_table<E>(
        &self,
        index: usize,
        table: &Table<EF>,
    ) -> Result<Table<EF>, BooleanTraceCommitmentError<E>> {
        let num_variables = table.num_variables();
        let height = 1usize << num_variables;
        let width = table.num_polys();
        let bits = self.boolean_columns[index];
        let committed = bits + (width - bits) * EF::COORDINATES;
        let blocks = height.div_ceil(WORD_BITS);
        let mut words = vec![0u64; blocks * committed];

        // Each block row holds sixty-four rows of every committed column, filled on its own.
        let refused = words
            .par_chunks_mut(committed)
            .enumerate()
            .filter_map(|(block, row)| {
                let rows = block * WORD_BITS..((block + 1) * WORD_BITS).min(height);
                for (column, slot) in row[..bits].iter_mut().enumerate() {
                    let view = table.column(column);
                    *slot = match view.boolean_word(block) {
                        Some(word) => word,
                        None => {
                            let mut word = 0;
                            for (lane, r) in rows.clone().enumerate() {
                                let value = view.value(r);
                                if value == EF::ONE {
                                    word |= 1 << lane;
                                } else if value != EF::ZERO {
                                    return Some(column);
                                }
                            }
                            word
                        }
                    };
                }
                for column in bits..width {
                    let view = table.column(column);
                    let out = &mut row[committed_columns::<EF>(bits, column)];
                    for (lane, r) in rows.clone().enumerate() {
                        let value = view.value(r);
                        for (byte_index, &byte) in coordinate_bytes(core::slice::from_ref(&value))
                            .iter()
                            .enumerate()
                        {
                            // Only the set coordinates are visited.
                            let mut rest = byte;
                            while rest != 0 {
                                let bit = rest.trailing_zeros() as usize;
                                out[byte_index * 8 + bit] |= 1 << lane;
                                rest &= rest - 1;
                            }
                        }
                    }
                }
                None
            })
            .min();
        if let Some(column) = refused {
            return Err(BooleanTraceCommitmentError::NonBooleanCell {
                table: index,
                column,
            });
        }
        Ok(Table::from_packed_bits(
            RowMajorMatrix::new(words, committed),
            num_variables,
        ))
    }
}

/// The committed tables, held until the commitment is opened.
pub struct MixedTraceData<EF: Field, D> {
    /// Prover data of the Boolean trace commitment underneath.
    inner: BooleanTraceCommitmentData<EF, D>,
    /// Source tables, lent back to whatever evaluates constraints over them.
    tables: Vec<Table<EF>>,
}

impl<EF: Field, D: Clone> Clone for MixedTraceData<EF, D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tables: self.tables.clone(),
        }
    }
}

impl<EF: Field, D> MixedTraceData<EF, D> {
    /// One source table, in the order the tables were supplied.
    ///
    /// # Panics
    ///
    /// Panics on a table index this commitment does not hold.
    #[must_use]
    pub fn table(&self, index: usize) -> &Table<EF> {
        &self.tables[index]
    }
}

impl<EF, B, Challenger> MultilinearPcs<EF, Challenger> for MixedTraceCommitment<EF, B>
where
    EF: BitCoordinates + TranscriptField + TowerLevel + Coordinates,
    B: BooleanMultilinearPcs<EF, Challenger, Val = EF>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<B::Commitment>,
{
    type Val = EF;
    type Commitment = B::Commitment;
    type ProverData = MixedTraceData<EF, B::ProverData>;
    type Proof = BooleanTraceCommitmentProof<EF, B::Proof>;
    type Error = BooleanTraceCommitmentError<B::Error>;
    type ProverError = BooleanTraceCommitmentError<B::Error>;
    type Witness = Vec<Table<EF>>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.inner.num_variables()
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        // Every refusal comes back before the transcript moves.
        let shapes: Vec<_> = witness.iter().map(Table::shape).collect();
        self.check_regions(&shapes)?;
        let committed = tracing::info_span!("expand mixed tables").in_scope(|| {
            witness
                .iter()
                .enumerate()
                .map(|(index, table)| self.expand_table(index, table))
                .collect::<Result<Vec<_>, _>>()
        })?;
        let (commitment, inner) = self.inner.commit(committed, challenger)?;
        Ok((
            commitment,
            MixedTraceData {
                inner,
                tables: witness,
            },
        ))
    }

    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        MultilinearPcs::observe_commitment(&self.inner, commitment, challenger);
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        self.check_regions(&protocol.table_shapes())?;
        let committed = self.expand_protocol(&protocol);
        self.inner.open(prover_data.inner, committed, challenger)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        self.check_regions(&protocol.table_shapes())?;
        let committed = self.expand_protocol(&protocol);
        self.inner.verify(commitment, proof, challenger, committed)
    }
}

impl<EF, B, Challenger> PrescribedPointPcs<EF, Challenger> for MixedTraceCommitment<EF, B>
where
    EF: BitCoordinates + TranscriptField + TowerLevel + Coordinates,
    B: BooleanMultilinearPcs<EF, Challenger, Val = EF>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<B::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        // The recombination draws nothing, so the committed protocol's evidence is the whole.
        self.check_regions::<B::Error>(&protocol.table_shapes())
            .ok()?;
        self.inner
            .prescribed_security(&self.expand_protocol(protocol))
    }

    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        self.check_regions(&protocol.table_shapes())?;
        let committed = self.expand_protocol(protocol);
        self.inner
            .open_at(prover_data.inner, &committed, points, challenger)
    }

    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<EF>>, Self::Error> {
        self.check_regions(&protocol.table_shapes())?;
        let committed = self.expand_protocol(protocol);
        let evals = self
            .inner
            .verify_at(commitment, proof, &committed, points, challenger)?;
        Ok(self.recombine(protocol, &evals))
    }
}

#[cfg(test)]
mod tests;
