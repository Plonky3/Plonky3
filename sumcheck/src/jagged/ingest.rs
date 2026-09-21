//! Reading a live trace into the vector a jagged commitment binds.
//!
//! A producer declares the shape its cells arrive in, and that declaration decides every pass.
//!
//! Every conversion is charged to the report the ingestion returns.

use alloc::vec::Vec;
use core::ops::Deref;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;

use super::error::{JaggedIngestError, JaggedLayoutError};
use super::layout::JaggedLayout;

/// Bits one packed word carries.
const WORD_BITS: usize = 64;

/// Work one producer owes before its cells sit in committed order.
///
/// The ordering runs from free to most expensive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversionPass {
    /// The cells are already the committed vector, and nothing reads them.
    Shared,
    /// The cells are contiguous elsewhere and move verbatim.
    Copied,
    /// The cells are interleaved with their neighbours and move one stride at a time.
    Gathered,
    /// The cells carry one bit each and widen to a field element apiece.
    Widened,
}

/// How the cells of one sparse column reach the ingestion pass.
///
/// The variant is the contract, so a producer cannot present a conversion as free.
#[derive(Clone, Copy, Debug)]
pub enum ColumnSource<'a, F> {
    /// Cells of one column, consecutive and in row order.
    Dense(&'a [F]),
    /// Cells of one column drawn from a larger block at a fixed step.
    ///
    /// This is the shape a row-major trace presents one of its columns in.
    Interleaved {
        /// Block the column is drawn from.
        cells: &'a [F],
        /// Position of the first cell of the column inside the block.
        first: usize,
        /// Distance between two consecutive cells of the column.
        stride: usize,
        /// Number of live cells in the column.
        height: usize,
    },
    /// Cells carrying one bit each, sixty-four to a word, least significant bit first.
    Bits {
        /// Packed words holding the column.
        words: &'a [u64],
        /// Number of live cells in the column.
        height: usize,
    },
    /// Consecutive pieces of one column, in row order.
    Chunked(&'a [Self]),
}

impl<F: Field> ColumnSource<'_, F> {
    /// Returns the number of live cells this source carries.
    #[must_use]
    pub fn height(&self) -> usize {
        match self {
            Self::Dense(cells) => cells.len(),
            Self::Interleaved { height, .. } | Self::Bits { height, .. } => *height,
            Self::Chunked(parts) => parts.iter().map(Self::height).sum(),
        }
    }

    /// Returns the most expensive conversion this source needs.
    #[must_use]
    pub fn pass(&self) -> ConversionPass {
        match self {
            Self::Dense(_) => ConversionPass::Copied,
            Self::Interleaved { .. } => ConversionPass::Gathered,
            Self::Bits { .. } => ConversionPass::Widened,
            Self::Chunked(parts) => parts
                .iter()
                .map(Self::pass)
                .max()
                .unwrap_or(ConversionPass::Copied),
        }
    }

    /// Rejects a source that cannot supply the cells it declares.
    fn validate(&self, column: usize) -> Result<(), JaggedIngestError> {
        match self {
            Self::Dense(_) => Ok(()),
            Self::Interleaved {
                cells,
                first,
                stride,
                height,
            } => {
                // An empty column reads nothing, whatever the step.
                if *height == 0 {
                    return Ok(());
                }
                if *stride == 0 {
                    return Err(JaggedIngestError::ZeroStride { column });
                }

                // The last cell is the only one that can leave the block.
                let last = first
                    .checked_add((height - 1).saturating_mul(*stride))
                    .ok_or(JaggedIngestError::StrideOutOfRange {
                        column,
                        required: usize::MAX,
                        available: cells.len(),
                    })?;
                if last >= cells.len() {
                    return Err(JaggedIngestError::StrideOutOfRange {
                        column,
                        required: last + 1,
                        available: cells.len(),
                    });
                }
                Ok(())
            }
            Self::Bits { words, height } => {
                let required = height.div_ceil(WORD_BITS);
                if words.len() < required {
                    return Err(JaggedIngestError::PackedWordsTooShort {
                        column,
                        required,
                        available: words.len(),
                    });
                }
                Ok(())
            }
            Self::Chunked(parts) => parts.iter().try_for_each(|part| part.validate(column)),
        }
    }

    /// Writes every live cell of this source in row order.
    ///
    /// The destination is exactly as long as the source is tall.
    fn write_into(&self, out: &mut [F]) {
        match self {
            Self::Dense(cells) => out.copy_from_slice(cells),
            Self::Interleaved {
                cells,
                first,
                stride,
                ..
            } => {
                for (row, slot) in out.iter_mut().enumerate() {
                    *slot = cells[first + row * stride];
                }
            }
            Self::Bits { words, .. } => {
                for (row, slot) in out.iter_mut().enumerate() {
                    let bit = (words[row / WORD_BITS] >> (row % WORD_BITS)) & 1;
                    *slot = F::from_bool(bit == 1);
                }
            }
            Self::Chunked(parts) => {
                let mut rest = out;
                for part in *parts {
                    let (head, tail) = rest.split_at_mut(part.height());
                    part.write_into(head);
                    rest = tail;
                }
            }
        }
    }
}

/// How a machine hands one whole jagged trace to the commitment.
#[derive(Clone, Copy, Debug)]
pub enum TraceSource<'a, F> {
    /// The committed vector itself, already concatenated and at least as long as the envelope.
    ///
    /// This is the only shape that costs nothing.
    Committed(&'a [F]),
    /// One source per sparse column, in the order the geometry concatenates them.
    Columns(&'a [ColumnSource<'a, F>]),
}

/// Live cells charged to each conversion, and the cells the envelope adds on top.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IngestReport {
    /// Live cells the committed vector shares with its producer.
    shared: usize,
    /// Live cells that moved verbatim.
    copied: usize,
    /// Live cells gathered one stride at a time.
    gathered: usize,
    /// Live cells widened from a single bit.
    widened: usize,
    /// Cells written only to reach the power-of-two envelope.
    envelope: usize,
}

impl IngestReport {
    /// Charges live cells to one conversion.
    const fn charge(&mut self, pass: ConversionPass, cells: usize) {
        let counter = match pass {
            ConversionPass::Shared => &mut self.shared,
            ConversionPass::Copied => &mut self.copied,
            ConversionPass::Gathered => &mut self.gathered,
            ConversionPass::Widened => &mut self.widened,
        };
        *counter += cells;
    }

    /// Returns the live cells charged to one conversion.
    #[must_use]
    pub const fn charged(&self, pass: ConversionPass) -> usize {
        match pass {
            ConversionPass::Shared => self.shared,
            ConversionPass::Copied => self.copied,
            ConversionPass::Gathered => self.gathered,
            ConversionPass::Widened => self.widened,
        }
    }

    /// Returns the live cells of the trace.
    #[must_use]
    pub const fn live(&self) -> usize {
        self.shared + self.copied + self.gathered + self.widened
    }

    /// Returns the live cells that a conversion had to touch.
    #[must_use]
    pub const fn converted(&self) -> usize {
        self.copied + self.gathered + self.widened
    }

    /// Returns the cells the power-of-two envelope adds beyond the live area.
    #[must_use]
    pub const fn envelope(&self) -> usize {
        self.envelope
    }

    /// Returns whether the committed vector was taken without reading a cell.
    #[must_use]
    pub const fn is_zero_copy(&self) -> bool {
        self.converted() == 0 && self.envelope == 0
    }
}

/// The committed vector, borrowed whenever its producer already owned it.
///
/// The variant is the evidence of whether a pass ran.
#[derive(Clone, Debug)]
pub enum JaggedWitness<'a, F> {
    /// The producer's own cells, used where they lie.
    Shared(&'a [F]),
    /// Cells written by an ingestion pass.
    Assembled(Vec<F>),
}

impl<F> Deref for JaggedWitness<'_, F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        match self {
            Self::Shared(cells) => cells,
            Self::Assembled(cells) => cells,
        }
    }
}

impl JaggedLayout {
    /// Builds the geometry one source per column implies.
    ///
    /// # Errors
    ///
    /// Returns whatever the geometry rejects about the heights the sources declare.
    pub fn from_columns<F: Field>(
        row_variables: usize,
        columns: &[ColumnSource<'_, F>],
    ) -> Result<Self, JaggedLayoutError> {
        let heights = columns.iter().map(ColumnSource::height).collect::<Vec<_>>();
        Self::new(row_variables, &heights)
    }

    /// Reads one trace into the vector a commitment to this geometry binds.
    ///
    /// # Errors
    ///
    /// - The number of sources differs from the number of columns.
    /// - A source supplies a different height than the geometry reserves.
    /// - A source cannot supply the cells it declares.
    /// - A pre-concatenated vector is shorter than the envelope.
    pub fn ingest<'a, F: Field>(
        &self,
        source: TraceSource<'a, F>,
    ) -> Result<(JaggedWitness<'a, F>, IngestReport), JaggedIngestError> {
        let capacity = self.dense_capacity();
        let mut report = IngestReport::default();

        let columns = match source {
            TraceSource::Committed(cells) => {
                if cells.len() < capacity {
                    return Err(JaggedIngestError::CommittedVectorTooShort {
                        required: capacity,
                        available: cells.len(),
                    });
                }

                // Padding is unconstrained, so whatever follows the live area rides along untouched.
                report.charge(ConversionPass::Shared, self.area());
                return Ok((JaggedWitness::Shared(&cells[..capacity]), report));
            }
            TraceSource::Columns(columns) => columns,
        };

        if columns.len() != self.num_columns() {
            return Err(JaggedIngestError::ColumnCountMismatch {
                expected: self.num_columns(),
                actual: columns.len(),
            });
        }
        for (index, column) in columns.iter().enumerate() {
            let declared = self.column_height(index);
            let actual = column.height();
            if actual != declared {
                return Err(JaggedIngestError::ColumnHeightMismatch {
                    column: index,
                    expected: declared,
                    actual,
                });
            }
            column.validate(index)?;
            report.charge(column.pass(), actual);
        }

        // The envelope is zeroed once, and only the live prefix is written over.
        report.envelope = capacity - self.area();
        let mut witness = F::zero_vec(capacity);

        // Each column owns one interval, so every destination below is disjoint.
        let mut slots = Vec::with_capacity(columns.len());
        let mut rest = &mut witness[..self.area()];
        for column in columns {
            let (head, tail) = rest.split_at_mut(column.height());
            slots.push(head);
            rest = tail;
        }
        slots
            .into_par_iter()
            .zip(columns.par_iter())
            .for_each(|(slot, column)| column.write_into(slot));

        Ok((JaggedWitness::Assembled(witness), report))
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;
    use crate::tests::F;

    // The cells one column logically carries, read straight off the producer's own arrays.
    // Nothing here calls the ingestion, so the two cannot agree by sharing a mistake.
    fn expected_cells(source: &ColumnSource<'_, F>) -> Vec<F> {
        match source {
            ColumnSource::Dense(cells) => cells.to_vec(),
            ColumnSource::Interleaved {
                cells,
                first,
                stride,
                height,
            } => (0..*height)
                .map(|row| cells[first + row * stride])
                .collect(),
            ColumnSource::Bits { words, height } => (0..*height)
                .map(|row| {
                    let word = words[row / 64];
                    F::from_bool(word & (1 << (row % 64)) != 0)
                })
                .collect(),
            ColumnSource::Chunked(parts) => parts.iter().flat_map(expected_cells).collect(),
        }
    }

    fn cells(values: &[u64]) -> Vec<F> {
        values.iter().map(|&value| F::from_u64(value)).collect()
    }

    #[test]
    fn a_producer_that_already_owns_the_committed_vector_pays_nothing() {
        // Fixture state: four columns of nine live cells inside a sixteen-cell envelope.
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let committed = cells(&(1..=16).collect::<Vec<_>>());

        let (witness, report) = layout.ingest(TraceSource::Committed(&committed)).unwrap();
        assert!(matches!(witness, JaggedWitness::Shared(_)));
        assert_eq!(&*witness, &committed[..]);
        assert!(report.is_zero_copy());
        assert_eq!(report.live(), 9);
        assert_eq!(report.charged(ConversionPass::Shared), 9);
        assert_eq!(report.converted(), 0);
        assert_eq!(report.envelope(), 0);

        // The vector may run past the envelope, and the surplus is neither read nor committed.
        let long = cells(&(1..=32).collect::<Vec<_>>());
        let (witness, _) = layout.ingest(TraceSource::Committed(&long)).unwrap();
        assert_eq!(witness.len(), 16);
    }

    #[test]
    fn every_arrival_shape_lands_on_the_same_committed_vector() {
        // One logical trace of three columns, presented four different ways.
        //
        //     column 0   cells 1, 2, 3
        //     column 1   empty
        //
        //     column 2   cells 4, 5
        //     column 3   cells 6, 7, 8, 9
        let layout = JaggedLayout::new(3, &[3, 0, 2, 4]).unwrap();
        let expected = cells(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0]);

        let first = cells(&[1, 2, 3]);
        let third = cells(&[4, 5]);
        let fourth = cells(&[6, 7, 8, 9]);
        let dense = [
            ColumnSource::Dense(&first),
            ColumnSource::Dense(&[]),
            ColumnSource::Dense(&third),
            ColumnSource::Dense(&fourth),
        ];
        let (witness, report) = layout.ingest(TraceSource::Columns(&dense)).unwrap();
        assert_eq!(&*witness, &expected[..]);
        assert_eq!(report.charged(ConversionPass::Copied), 9);
        assert_eq!(report.envelope(), 7);
        assert!(!report.is_zero_copy());

        // The same trace as one row-major block of four columns and four rows.
        let block = cells(&[1, 0, 4, 6, 2, 0, 5, 7, 3, 0, 0, 8, 0, 0, 0, 9]);
        let interleaved = [0usize, 1, 2, 3].map(|column| ColumnSource::Interleaved {
            cells: &block,
            first: column,
            stride: 4,
            height: layout.column_height(column),
        });
        let (witness, report) = layout.ingest(TraceSource::Columns(&interleaved)).unwrap();
        assert_eq!(&*witness, &expected[..]);
        assert_eq!(report.charged(ConversionPass::Gathered), 9);

        // One column split into two consecutive pieces reads as one column.
        let head = cells(&[1, 2]);
        let tail = cells(&[3]);
        let parts = [ColumnSource::Dense(&head), ColumnSource::Dense(&tail)];
        let chunked = [
            ColumnSource::Chunked(&parts),
            ColumnSource::Dense(&[]),
            ColumnSource::Dense(&third),
            ColumnSource::Dense(&fourth),
        ];
        let (witness, _) = layout.ingest(TraceSource::Columns(&chunked)).unwrap();
        assert_eq!(&*witness, &expected[..]);
    }

    #[test]
    fn a_packed_column_widens_its_bits_least_significant_first() {
        // Nine bits of one word spell out 1, 0, 1, 1, 0, 0, 0, 1, 1 from the bottom up.
        let layout = JaggedLayout::new(4, &[9, 0]).unwrap();
        let words = [0b1_1000_1101u64];
        let sources: [ColumnSource<'_, F>; 2] = [
            ColumnSource::Bits {
                words: &words,
                height: 9,
            },
            ColumnSource::Dense(&[]),
        ];

        let (witness, report) = layout.ingest(TraceSource::Columns(&sources)).unwrap();
        assert_eq!(
            witness[..9],
            cells(&[1, 0, 1, 1, 0, 0, 0, 1, 1])[..],
            "bits read from the least significant end of the word"
        );
        assert_eq!(report.charged(ConversionPass::Widened), 9);
        assert_eq!(report.envelope(), 7);
    }

    #[test]
    fn the_declared_shape_names_the_pass_a_producer_pays_for() {
        let bits = [0u64];
        let dense = cells(&[1, 2]);
        let parts = [
            ColumnSource::Dense(&dense),
            ColumnSource::Bits {
                words: &bits,
                height: 1,
            },
        ];

        assert_eq!(ColumnSource::Dense(&dense).pass(), ConversionPass::Copied);
        assert_eq!(
            ColumnSource::<F>::Bits {
                words: &bits,
                height: 1
            }
            .pass(),
            ConversionPass::Widened
        );
        // A mixed column is charged at the price of its most expensive piece.
        assert_eq!(
            ColumnSource::Chunked(&parts).pass(),
            ConversionPass::Widened
        );
        assert_eq!(ColumnSource::Chunked(&parts).height(), 3);
    }

    #[test]
    fn a_trace_the_geometry_cannot_read_is_refused() {
        let layout = JaggedLayout::new(3, &[3, 0, 2, 4]).unwrap();
        let three = cells(&[1, 2, 3]);

        // One source short of the column count.
        let short = [ColumnSource::Dense(&three)];
        assert_eq!(
            layout.ingest(TraceSource::Columns(&short)).err(),
            Some(JaggedIngestError::ColumnCountMismatch {
                expected: 4,
                actual: 1
            })
        );

        // A column that supplies the wrong number of cells.
        let wrong = [
            ColumnSource::Dense(&three),
            ColumnSource::Dense(&[]),
            ColumnSource::Dense(&three),
            ColumnSource::Dense(&three),
        ];
        assert_eq!(
            layout.ingest(TraceSource::Columns(&wrong)).err(),
            Some(JaggedIngestError::ColumnHeightMismatch {
                column: 2,
                expected: 2,
                actual: 3
            })
        );

        // An interleaved column whose last cell falls outside its block.
        let block = cells(&[1, 2, 3, 4, 5, 6]);
        let over = [
            ColumnSource::Dense(&three),
            ColumnSource::Dense(&[]),
            ColumnSource::Dense(&cells(&[1, 2])),
            ColumnSource::Interleaved {
                cells: &block,
                first: 1,
                stride: 2,
                height: 4,
            },
        ];
        assert_eq!(
            layout.ingest(TraceSource::Columns(&over)).err(),
            Some(JaggedIngestError::StrideOutOfRange {
                column: 3,
                required: 8,
                available: 6
            })
        );

        // An interleaved column with no step between its cells.
        let stalled = [
            ColumnSource::Dense(&three),
            ColumnSource::Dense(&[]),
            ColumnSource::Dense(&cells(&[1, 2])),
            ColumnSource::Interleaved {
                cells: &block,
                first: 0,
                stride: 0,
                height: 4,
            },
        ];
        assert_eq!(
            layout.ingest(TraceSource::Columns(&stalled)).err(),
            Some(JaggedIngestError::ZeroStride { column: 3 })
        );

        // A packed column short of a word.
        let words = [0u64];
        let packed = [
            ColumnSource::Dense(&three),
            ColumnSource::Dense(&[]),
            ColumnSource::Bits {
                words: &[],
                height: 2,
            },
            ColumnSource::Bits {
                words: &words,
                height: 4,
            },
        ];
        assert_eq!(
            layout.ingest(TraceSource::Columns(&packed)).err(),
            Some(JaggedIngestError::PackedWordsTooShort {
                column: 2,
                required: 1,
                available: 0
            })
        );

        // A pre-concatenated trace shorter than the envelope.
        let committed = cells(&[1, 2, 3]);
        assert_eq!(
            layout.ingest(TraceSource::Committed(&committed)).err(),
            Some(JaggedIngestError::CommittedVectorTooShort {
                required: 16,
                available: 3
            })
        );
    }

    #[test]
    fn a_geometry_reads_its_heights_off_the_sources() {
        let first = cells(&[1, 2, 3]);
        let words = [0u64];
        let sources = [
            ColumnSource::Dense(&first),
            ColumnSource::Bits {
                words: &words,
                height: 5,
            },
        ];

        let layout = JaggedLayout::from_columns(3, &sources).unwrap();
        assert_eq!(layout.column_height(0), 3);
        assert_eq!(layout.column_height(1), 5);
        assert_eq!(layout.area(), 8);
    }

    proptest! {
        #[test]
        fn the_live_prefix_is_the_concatenation_of_the_columns(
            heights in prop::collection::vec(0usize..=9, 4),
            shapes in prop::collection::vec(0usize..3, 4),
            values in prop::collection::vec(any::<u64>(), 64),
        ) {
            let layout = JaggedLayout::new(4, &heights).unwrap();
            let block = cells(&values);
            let words = values;

            // Each column is presented in one of the three single-piece shapes.
            let sources = heights
                .iter()
                .enumerate()
                .map(|(column, &height)| match shapes[column] {
                    0 => ColumnSource::Dense(&block[..height]),
                    1 => ColumnSource::Interleaved {
                        cells: &block,
                        first: column,
                        stride: 4,
                        height,
                    },
                    _ => ColumnSource::Bits {
                        words: &words,
                        height,
                    },
                })
                .collect::<Vec<_>>();

            let (witness, report) = layout.ingest(TraceSource::Columns(&sources)).unwrap();
            let expected = sources.iter().flat_map(expected_cells).collect::<Vec<_>>();

            prop_assert_eq!(witness.len(), layout.dense_capacity());
            prop_assert_eq!(&witness[..layout.area()], &expected[..]);
            prop_assert!(witness[layout.area()..].iter().all(|cell| *cell == F::ZERO));
            prop_assert_eq!(report.live(), layout.area());
            prop_assert_eq!(report.converted(), layout.area());
            prop_assert_eq!(report.envelope(), layout.dense_capacity() - layout.area());
        }
    }
}
