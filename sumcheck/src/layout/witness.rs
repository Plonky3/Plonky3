//! Physical placement of source tables inside the stacked committed polynomial.

use alloc::vec::Vec;

use p3_field::{Field, PackedValue};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyView};
use p3_util::reverse_bits_len;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use thiserror::Error;
use tracing::info_span;

use crate::layout::plan::{LayoutShape, plan_layout, plan_stacked_layout};
use crate::table::TableShape;

/// Entries per task when one column is copied into its stacked slot.
///
/// Enough that each task outweighs the fork-join overhead, so short columns copy in one piece.
const STACK_COPY_CHUNK: usize = 1 << 16;

/// Identifies one slot inside the stacked polynomial.
#[derive(Debug, Clone, Copy)]
pub struct Selector {
    /// Bit-width of the slot address, carved from the stacked variables.
    num_variables: usize,
    /// Slot index, interpreted as an integer in `0..2^num_variables`.
    index: usize,
}

impl Selector {
    /// Builds a selector over `num_variables` bits pointing at slot `index`.
    ///
    /// # Panics
    ///
    /// - Slot index must fit in `num_variables` bits.
    pub const fn new(num_variables: usize, index: usize) -> Self {
        // Bounds check: slot index must address a valid hypercube point.
        assert!(index < (1 << num_variables));
        Self {
            num_variables,
            index,
        }
    }

    /// Returns the hypercube point that addresses this slot.
    #[inline(always)]
    pub fn point<F: Field>(&self) -> Point<F> {
        Point::hypercube(self.index, self.num_variables)
    }

    /// Returns the number of selector bits.
    #[inline(always)]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the slot index.
    #[inline(always)]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Reverses this selector's bitstring in place.
    #[inline(always)]
    pub const fn reverse(&mut self) {
        self.index = reverse_bits_len(self.index, self.num_variables);
    }

    /// Prefixes `other` with the selector bits.
    #[inline(always)]
    pub fn lift_prefix<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        // Expand the slot index as selector bits; single allocation.
        let mut out: Point<Ext> = self.point();
        // Append the local coordinates to finish the stacked-space point.
        out.extend(other);
        out
    }

    /// Appends the selector bits after the local coordinates.
    #[inline(always)]
    pub fn lift_suffix<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        let mut out = other.clone();
        out.extend(&self.point());
        out
    }
}

/// A column-major table of multilinear polynomials sharing a common arity.
///
/// The backing matrix is row-major, with one row per polynomial. Each row is
/// therefore one trace column in evaluation order.
///
/// # Invariants
///
/// - At least one column.
/// - Every column has the same number of variables.
#[derive(Debug, Clone)]
enum TableStorage<F: Field> {
    Dense(RowMajorMatrix<F>),
    Boolean {
        words: RowMajorMatrix<u64>,
        num_variables: usize,
    },
}

/// A borrowed, representation-independent view of one logical table column.
///
/// Dense columns borrow their field cells. Packed columns borrow the source words and decode
/// individual Boolean cells on demand; no expanded column is cached.
#[derive(Clone, Copy)]
pub enum ColumnView<'a, F: Field> {
    Dense(&'a [F]),
    Boolean {
        words: &'a RowMajorMatrix<u64>,
        column: usize,
        len: usize,
    },
}

impl<F: Field> core::fmt::Debug for ColumnView<'_, F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ColumnView")
            .field("len", &self.len())
            .finish()
    }
}

/// Iterator over the logical field values of a [`ColumnView`].
pub struct ColumnValues<'a, F: Field> {
    view: ColumnView<'a, F>,
    range: core::ops::Range<usize>,
}

impl<F: Field> Iterator for ColumnValues<'_, F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|row| self.view.value(row))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.range.len();
        (len, Some(len))
    }
}

impl<F: Field> DoubleEndedIterator for ColumnValues<'_, F> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|row| self.view.value(row))
    }
}

impl<F: Field> ExactSizeIterator for ColumnValues<'_, F> {}

impl<'a, F: Field> ColumnView<'a, F> {
    /// Number of logical rows in this column.
    #[inline]
    pub const fn len(self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Boolean { len, .. } => len,
        }
    }

    /// Whether this column has no logical rows.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Returns one logical field cell.
    #[inline]
    pub fn value(self, row: usize) -> F {
        assert!(row < self.len(), "column row out of bounds");
        match self {
            Self::Dense(values) => values[row],
            Self::Boolean { words, column, .. } => F::from_bool(
                ((words.values[(row / 64) * words.width + column] >> (row % 64)) & 1) != 0,
            ),
        }
    }

    /// Iterates over logical field cells without materializing a column.
    #[inline]
    pub const fn values(self) -> ColumnValues<'a, F> {
        ColumnValues {
            view: self,
            range: 0..self.len(),
        }
    }

    /// Returns the borrowed dense slice, if this column is dense.
    #[inline]
    pub const fn as_dense(self) -> Option<&'a [F]> {
        match self {
            Self::Dense(values) => Some(values),
            Self::Boolean { .. } => None,
        }
    }

    /// Returns one packed field lane group, decoding bits when necessary.
    #[inline]
    pub fn packed_at(self, row: usize) -> F::Packing {
        F::Packing::from_fn(|lane| self.value(row + lane))
    }

    /// Returns the packed source word for a Boolean column.
    #[inline]
    pub fn boolean_word(self, word: usize) -> Option<u64> {
        match self {
            Self::Boolean { words, column, .. } => words
                .values
                .get(word.checked_mul(words.width)? + column)
                .copied(),
            Self::Dense(_) => None,
        }
    }

    /// Copies logical cells into an explicitly supplied dense destination.
    pub fn copy_into(self, destination: &mut [F]) {
        assert_eq!(
            destination.len(),
            self.len(),
            "column destination has wrong length"
        );
        if let Some(values) = self.as_dense() {
            destination.copy_from_slice(values);
        } else {
            for (destination, value) in destination.iter_mut().zip(self.values()) {
                *destination = value;
            }
        }
    }
}

/// A column-major table of multilinear polynomials sharing a common arity.
///
/// Dense tables retain the original field-cell layout. Boolean tables keep one `u64` word per
/// 64 logical rows and expose [`ColumnView`] for consumers that can operate without decoding the
/// full trace.
#[derive(Debug, Clone)]
pub struct Table<F: Field> {
    storage: TableStorage<F>,
}

impl<F: Field> Table<F> {
    /// Creates a table from a row-major matrix with one polynomial per row.
    ///
    /// # Panics
    ///
    /// - Matrix must have at least one row (one polynomial).
    /// - Row width must be a power of two, since each row is a hypercube evaluation table.
    pub fn new(columns: RowMajorMatrix<F>) -> Self {
        assert!(
            columns.width.is_power_of_two(),
            "table row width must be a power of two"
        );
        assert!(columns.height() > 0, "table must have at least one column");
        Self {
            storage: TableStorage::Dense(columns),
        }
    }

    /// Creates a packed Boolean table from words in physical row-block-major order.
    ///
    /// The matrix width is the logical column count. Physical row `w` stores rows
    /// `64*w..64*w+63`, with bit zero holding the first row. The logical height is `2^num_variables`.
    ///
    /// # Panics
    ///
    /// Panics when the dimensions overflow, the matrix width is zero, the word count is wrong,
    /// or unused high bits are set.
    pub fn from_packed_bits(words: RowMajorMatrix<u64>, num_variables: usize) -> Self {
        assert!(
            num_variables < usize::BITS as usize,
            "packed table height overflows usize"
        );
        let height = 1usize
            .checked_shl(num_variables as u32)
            .expect("packed table height overflows usize");
        assert!(
            words.width > 0,
            "packed table must have at least one column"
        );
        let blocks = height
            .checked_add(63)
            .expect("packed table block count overflows usize")
            / 64;
        let expected = words
            .width
            .checked_mul(blocks)
            .expect("packed table word count overflows usize");
        assert_eq!(
            words.values.len(),
            expected,
            "packed table must contain exactly width * ceil(height / 64) words"
        );
        if !height.is_multiple_of(64) {
            let used = height % 64;
            let high_bits = !((1u64 << used) - 1);
            let last = &words.values[(blocks - 1) * words.width..blocks * words.width];
            assert!(
                last.iter().all(|word| word & high_bits == 0),
                "packed table padding bits must be zero"
            );
        }
        Self {
            storage: TableStorage::Boolean {
                words,
                num_variables,
            },
        }
    }

    /// Creates a zero-filled table.
    ///
    /// # Panics
    ///
    /// - `num_polys` must be nonzero.
    pub fn zero(num_polys: usize, num_variables: usize) -> Self {
        Self::new(RowMajorMatrix::new(
            F::zero_vec(num_polys * (1 << num_variables)),
            1 << num_variables,
        ))
    }

    /// Samples a table whose rows are independent random polynomials.
    ///
    /// # Panics
    ///
    /// - `num_polys` must be nonzero.
    pub fn rand<R: Rng>(rng: &mut R, num_polys: usize, num_variables: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self::new(RowMajorMatrix::rand(rng, num_polys, 1 << num_variables))
    }

    /// Iterates over the table rows, one polynomial evaluation slice per row.
    ///
    /// This accessor is for dense storage. Packed Boolean tables must use [`Self::columns`] or
    /// [`Self::column`] to avoid materializing a field-cell trace.
    ///
    /// # Panics
    ///
    /// Panics when called on packed Boolean storage.
    pub fn iter_polys(&self) -> impl DoubleEndedIterator<Item = &[F]> {
        self.dense_matrix().row_slices()
    }

    /// Iterates over the table rows in parallel, one polynomial evaluation slice per row.
    ///
    /// This accessor is for dense storage. Packed Boolean tables must use [`Self::par_columns`]
    /// or [`Self::columns`] to avoid materializing a field-cell trace.
    ///
    /// # Panics
    ///
    /// Panics when called on packed Boolean storage.
    pub fn par_iter_polys(&self) -> impl IndexedParallelIterator<Item = &[F]> {
        self.dense_matrix().par_row_slices()
    }

    /// Returns the polynomial at column `id`.
    ///
    /// This accessor is for dense storage. Packed Boolean tables must use [`Self::column`] to
    /// read a borrowed representation-independent view.
    ///
    /// # Panics
    ///
    /// Panics when called on packed Boolean storage.
    pub fn poly(&self, id: usize) -> PolyView<'_, F> {
        let dense = self.dense_matrix();
        let start = id * dense.width;
        PolyView::new(&dense.values[start..start + dense.width])
    }

    /// Returns one column without materializing packed storage.
    pub fn column(&self, id: usize) -> ColumnView<'_, F> {
        assert!(id < self.num_polys(), "table column out of bounds");
        match &self.storage {
            TableStorage::Dense(columns) => {
                let start = id * columns.width;
                ColumnView::Dense(&columns.values[start..start + columns.width])
            }
            TableStorage::Boolean {
                words,
                num_variables,
            } => ColumnView::Boolean {
                words,
                column: id,
                len: 1usize << num_variables,
            },
        }
    }

    /// Iterates over borrowed representation-independent columns.
    pub fn columns(&self) -> impl DoubleEndedIterator<Item = ColumnView<'_, F>> {
        (0..self.num_polys()).map(|column| self.column(column))
    }

    /// Iterates over borrowed representation-independent columns in parallel.
    pub fn par_columns(&self) -> impl IndexedParallelIterator<Item = ColumnView<'_, F>> {
        (0..self.num_polys())
            .into_par_iter()
            .map(|column| self.column(column))
    }

    /// Returns the packed backing matrix, if this table is Boolean-packed.
    pub const fn packed_bits(&self) -> Option<&RowMajorMatrix<u64>> {
        match &self.storage {
            TableStorage::Dense(_) => None,
            TableStorage::Boolean { words, .. } => Some(words),
        }
    }

    /// Converts packed storage into the dense field-cell representation explicitly.
    ///
    /// An already dense table is returned without copying.
    pub fn into_dense(self) -> Self {
        match self.storage {
            TableStorage::Dense(_) => self,
            TableStorage::Boolean {
                words,
                num_variables,
            } => {
                let height = 1usize << num_variables;
                let width = words.width;
                let mut values = F::zero_vec(width * height);
                for column in 0..width {
                    for row in 0..height {
                        values[column * height + row] = F::from_bool(
                            ((words.values[(row / 64) * width + column] >> (row % 64)) & 1) != 0,
                        );
                    }
                }
                Self::new(RowMajorMatrix::new(values, height))
            }
        }
    }

    fn dense_matrix(&self) -> &RowMajorMatrix<F> {
        match &self.storage {
            TableStorage::Dense(columns) => columns,
            TableStorage::Boolean { .. } => {
                panic!("dense table access is unavailable for packed Boolean storage")
            }
        }
    }

    /// Returns the number of columns.
    pub fn num_polys(&self) -> usize {
        match &self.storage {
            TableStorage::Dense(columns) => columns.height(),
            TableStorage::Boolean { words, .. } => words.width,
        }
    }

    /// Returns the shared number of variables.
    pub const fn num_variables(&self) -> usize {
        match &self.storage {
            TableStorage::Dense(columns) => columns.width.ilog2() as usize,
            TableStorage::Boolean { num_variables, .. } => *num_variables,
        }
    }

    /// Returns the verifier shape of this table.
    pub fn shape(&self) -> TableShape {
        TableShape::new(self.num_variables(), self.num_polys())
    }

    /// Pads every column with zeros until the table has at least `num_variables`.
    fn pad_zeros(&mut self, num_variables: usize) {
        let current_num_variables = self.num_variables();
        if current_num_variables < num_variables {
            match &mut self.storage {
                TableStorage::Dense(columns) => columns
                    .widen_right((1 << num_variables) - (1 << current_num_variables), F::ZERO),
                TableStorage::Boolean { .. } => {
                    panic!("packed tables must be converted to dense before padding")
                }
            }
        }
    }
}

/// A source that writes one table directly into its final contiguous suffix-layout slots.
///
/// Destinations are in source-column order and exclude zero-padding rows.
pub trait SuffixTableSource<F: Field>: Sync {
    /// Returns the source's logical shape.
    fn shape(&self) -> TableShape;

    /// Writes every logical column through its checked destination.
    fn fill(&self, columns: &mut [ColumnOut<'_, F>]);
}

impl<F: Field> SuffixTableSource<F> for Table<F> {
    fn shape(&self) -> TableShape {
        self.shape()
    }

    fn fill(&self, columns: &mut [ColumnOut<'_, F>]) {
        columns
            .par_iter_mut()
            .zip(self.par_columns())
            .for_each(|(out, source)| out.copy_from_view(source));
    }
}

/// One logical column's disjoint destination inside a suffix-layout stack.
///
/// The backing allocation is initialized; coverage is tracked per column.
pub struct ColumnOut<'a, F> {
    /// Logical rows assigned to one source column.
    values: &'a mut [F],
    /// Whether the producer called the checked writer for this column.
    written: bool,
}

impl<'a, F: Field> ColumnOut<'a, F> {
    /// Returns the number of logical rows the producer must write.
    pub const fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this destination contains no logical rows.
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Gives a producer exclusive access to the complete logical column.
    ///
    /// Coverage is recorded after the writer returns.
    pub fn write_with(&mut self, writer: impl FnOnce(&mut [F])) {
        writer(self.values);
        self.written = true;
    }

    /// Copies a dense logical column into this destination and records coverage.
    ///
    /// # Panics
    ///
    /// Panics when the source length differs from the planned logical height.
    pub fn copy_from_slice(&mut self, source: &[F])
    where
        F: Copy,
    {
        self.write_with(|destination| destination.copy_from_slice(source));
    }

    /// Copies a representation-independent source column into this destination.
    pub fn copy_from_view(&mut self, source: ColumnView<'_, F>) {
        self.write_with(|destination| source.copy_into(destination));
    }
}

/// A reusable immutable placement plan for directly filled suffix-layout witnesses.
#[derive(Debug, Clone)]
pub struct SuffixLayoutPlan {
    /// Logical dimensions expected from each source.
    source_shapes: Vec<TableShape>,
    /// Dimensions after extending short columns with zeroes.
    committed_shapes: Vec<TableShape>,
    /// Column locations inside the final stacked polynomial.
    placements: Vec<TablePlacement>,
    /// Arity of the final stacked polynomial.
    num_variables: usize,
    /// Number of suffix variables reserved for preprocessing.
    folding: usize,
}

/// Dimensions that cannot form a suffix-layout witness.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SuffixLayoutPlanError {
    /// No source tables were provided.
    #[error("a suffix-layout witness requires at least one table")]
    NoTables,
    /// The preprocessing depth cannot be represented.
    #[error("suffix-layout preprocessing depth {folding} overflows usize")]
    FoldingOverflow {
        /// Rejected depth.
        folding: usize,
    },
    /// The normalized cell count does not fit in `usize`.
    #[error("suffix-layout table cells overflow usize")]
    CellCountOverflow,
}

/// A source batch that disagrees with its immutable placement plan.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SuffixFillError {
    /// The source list and planned table list have different lengths.
    #[error("suffix-layout fill received {actual} tables, expected {expected}")]
    TableCount {
        /// Planned table count.
        expected: usize,
        /// Supplied table count.
        actual: usize,
    },
    /// One source reports dimensions different from its planned logical dimensions.
    #[error("suffix-layout table {table} shape does not match its plan")]
    ShapeMismatch {
        /// Table index.
        table: usize,
        /// Planned shape.
        expected: TableShape,
        /// Supplied shape.
        actual: TableShape,
    },
    /// A producer returned without calling the checked writer for one column.
    #[error("suffix-layout table {table} did not write column {column}")]
    UnwrittenColumn {
        /// Table index.
        table: usize,
        /// Column index.
        column: usize,
    },
}

/// A directly filled suffix-layout polynomial and its verifier-reconstructible metadata.
#[derive(Debug, Clone)]
pub struct FilledSuffixWitness<'a, F: Field> {
    /// Placement metadata used to build the polynomial.
    plan: &'a SuffixLayoutPlan,
    /// Final stacked polynomial filled in place.
    poly: Poly<F>,
}

impl SuffixLayoutPlan {
    /// Builds a checked placement plan from logical table shapes.
    ///
    /// Tables shorter than the preprocessing depth receive an inaccessible zero suffix.
    pub fn new(
        source_shapes: Vec<TableShape>,
        folding: usize,
    ) -> Result<Self, SuffixLayoutPlanError> {
        if source_shapes.is_empty() {
            return Err(SuffixLayoutPlanError::NoTables);
        }
        if folding >= usize::BITS as usize {
            return Err(SuffixLayoutPlanError::FoldingOverflow { folding });
        }

        let committed_shapes = source_shapes
            .iter()
            .map(|shape| TableShape::new(shape.num_variables().max(folding), shape.width()))
            .collect::<Vec<_>>();

        let total_cells = committed_shapes.iter().try_fold(0usize, |total, shape| {
            let rows = 1usize
                .checked_shl(shape.num_variables() as u32)
                .ok_or(SuffixLayoutPlanError::CellCountOverflow)?;
            let cells = rows
                .checked_mul(shape.width())
                .ok_or(SuffixLayoutPlanError::CellCountOverflow)?;
            total
                .checked_add(cells)
                .ok_or(SuffixLayoutPlanError::CellCountOverflow)
        })?;
        total_cells
            .checked_next_power_of_two()
            .ok_or(SuffixLayoutPlanError::CellCountOverflow)?;

        let (num_variables, placements) = plan_stacked_layout(&committed_shapes);
        Ok(Self {
            source_shapes,
            committed_shapes,
            placements,
            num_variables,
            folding,
        })
    }

    /// Returns the logical shapes producers must supply.
    pub fn source_shapes(&self) -> &[TableShape] {
        &self.source_shapes
    }

    /// Returns the committed shapes after explicit suffix-zero normalization.
    pub fn committed_shapes(&self) -> &[TableShape] {
        &self.committed_shapes
    }

    /// Returns the immutable largest-first column placements.
    pub fn placements(&self) -> &[TablePlacement] {
        &self.placements
    }

    /// Returns the arity of the final stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Fills one initialized final allocation directly from heterogeneous sources.
    ///
    /// Each producer receives all of its disjoint logical column windows together.
    /// Unused stack tail and preprocessing suffixes remain zero.
    pub fn fill<'a, F: Field>(
        &'a self,
        sources: &[&dyn SuffixTableSource<F>],
    ) -> Result<FilledSuffixWitness<'a, F>, SuffixFillError> {
        if sources.len() != self.source_shapes.len() {
            return Err(SuffixFillError::TableCount {
                expected: self.source_shapes.len(),
                actual: sources.len(),
            });
        }
        for (table, (source, expected)) in sources.iter().zip(&self.source_shapes).enumerate() {
            let actual = source.shape();
            if actual != *expected {
                return Err(SuffixFillError::ShapeMismatch {
                    table,
                    expected: *expected,
                    actual,
                });
            }
        }

        // Both normalized column suffixes and the unused stack tail stay zero.
        let mut poly = Poly::<F>::zero(self.num_variables);
        let slots = column_slots_for_shapes(
            &self.placements,
            &self.committed_shapes,
            0,
            poly.as_mut_slice(),
        );

        let mut ordered = slots;
        ordered.sort_unstable_by_key(|&(_, table, column)| (table, column));
        let mut outputs = ordered
            .into_iter()
            .map(|(slot, table, _)| {
                let logical_len = 1usize << self.source_shapes[table].num_variables();
                let (logical, _) = slot.split_at_mut(logical_len);
                ColumnOut {
                    values: logical,
                    written: false,
                }
            })
            .collect::<Vec<_>>();

        let mut rest = outputs.as_mut_slice();
        let mut table_outputs = Vec::with_capacity(self.source_shapes.len());
        for shape in &self.source_shapes {
            let (columns, tail) = rest.split_at_mut(shape.width());
            table_outputs.push(columns);
            rest = tail;
        }
        debug_assert!(rest.is_empty());

        table_outputs
            .into_par_iter()
            .zip(sources.par_iter())
            .for_each(|(columns, source)| source.fill(columns));

        let mut first_column = 0;
        for (table, shape) in self.source_shapes.iter().enumerate() {
            let columns = &outputs[first_column..first_column + shape.width()];
            if let Some(column) = columns.iter().position(|column| !column.written) {
                return Err(SuffixFillError::UnwrittenColumn { table, column });
            }
            first_column += shape.width();
        }

        Ok(FilledSuffixWitness { plan: self, poly })
    }
}

impl<F: Field> FilledSuffixWitness<'_, F> {
    /// Returns the committed table shapes in caller order.
    pub fn table_shapes(&self) -> &[TableShape] {
        self.plan.committed_shapes()
    }

    /// Returns the largest-first column placements.
    pub fn placements(&self) -> &[TablePlacement] {
        self.plan.placements()
    }

    /// Returns the arity of the final stacked polynomial.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Returns the directly filled stacked polynomial.
    pub const fn poly(&self) -> &Poly<F> {
        &self.poly
    }

    /// Consumes the witness and returns its stacked polynomial.
    pub fn into_poly(self) -> Poly<F> {
        self.poly
    }

    /// Builds the dense table view required by the suffix-layout prover.
    ///
    /// The table values are copied from the committed polynomial.
    /// Callers cannot supply a second representation that disagrees with the commitment.
    pub fn into_witness(self) -> Witness<F> {
        // Suffix rounds need one dense table per logical source.
        // Reconstructing them from the stack keeps the committed polynomial authoritative.
        let mut values_by_table = self
            .plan
            .committed_shapes
            .iter()
            .map(|shape| Vec::with_capacity((1usize << shape.num_variables()) * shape.width()))
            .collect::<Vec<_>>();

        // Each selector addresses one complete committed column slot.
        for placement in &self.plan.placements {
            let shape = self.plan.committed_shapes[placement.idx()];
            let rows = 1usize << shape.num_variables();
            for selector in placement.selectors() {
                let offset = selector.index() << shape.num_variables();
                values_by_table[placement.idx()]
                    .extend_from_slice(&self.poly.as_slice()[offset..offset + rows]);
            }
        }

        // Source order is restored after the largest-first placement walk.
        let tables = values_by_table
            .into_iter()
            .zip(&self.plan.committed_shapes)
            .map(|(values, shape)| {
                Table::new(RowMajorMatrix::new(values, 1usize << shape.num_variables()))
            })
            .collect();

        Witness {
            tables,
            placements: self.plan.placements.clone(),
            num_variables: self.plan.num_variables,
            folding: self.plan.folding,
            poly: None,
        }
    }
}

/// Splits one stacked buffer into column slots using shape-only metadata.
fn column_slots_for_shapes<'a, T>(
    placements: &[TablePlacement],
    shapes: &[TableShape],
    num_folded: usize,
    out: &'a mut [T],
) -> Vec<(&'a mut [T], usize, usize)> {
    let mut ranges = placements
        .iter()
        .flat_map(|placement| {
            let num_variables_table = shapes[placement.idx()].num_variables();
            assert!(num_folded <= num_variables_table);
            let log_len = num_variables_table - num_folded;
            placement
                .selectors()
                .iter()
                .enumerate()
                .map(move |(column, selector)| {
                    (
                        selector.index() << log_len,
                        1 << log_len,
                        placement.idx(),
                        column,
                    )
                })
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|&(offset, ..)| offset);

    let mut slots = Vec::with_capacity(ranges.len());
    let mut rest = out;
    let mut consumed = 0;
    for (offset, len, table, column) in ranges {
        let (_, tail) = core::mem::take(&mut rest).split_at_mut(offset - consumed);
        let (slot, tail) = tail.split_at_mut(len);
        slots.push((slot, table, column));
        rest = tail;
        consumed = offset + len;
    }
    slots
}

/// Splits `out` into one disjoint slot per placed column, after `num_folded` suffix variables.
///
/// Column `(table, poly)` owns the slot starting at `selector.index() << (n - num_folded)`, of
/// length `2^(n - num_folded)`, where `n` is the table's number of variables.
///
/// # Returns
///
/// One `(slot, table index, column index)` triple per column, in increasing slot order.
///
/// # Panics
///
/// - `num_folded` exceeds a placed table's number of variables.
pub(crate) fn column_slots<'a, T, F: Field>(
    placements: &[TablePlacement],
    tables: &[Table<F>],
    num_folded: usize,
    out: &'a mut [T],
) -> Vec<(&'a mut [T], usize, usize)> {
    let shapes = tables.iter().map(Table::shape).collect::<Vec<_>>();
    column_slots_for_shapes(placements, &shapes, num_folded, out)
}

/// Placement metadata for one table inside the stacked polynomial.
#[derive(Debug, Clone)]
pub struct TablePlacement {
    /// Source table index this placement refers back to.
    pub(super) idx: usize,
    /// One selector per column, addressing the column's slot.
    pub(super) selectors: Vec<Selector>,
}

impl TablePlacement {
    /// Creates placement metadata for table index `idx` with the given selectors.
    pub const fn new(idx: usize, selectors: Vec<Selector>) -> Self {
        Self { idx, selectors }
    }

    /// Reverses every selector bitstring in this placement.
    pub fn reverse_selectors(&mut self) {
        self.selectors.iter_mut().for_each(Selector::reverse);
    }

    /// Returns the number of columns placed for this table.
    pub const fn num_polys(&self) -> usize {
        self.selectors.len()
    }

    /// Returns the source table index.
    pub const fn idx(&self) -> usize {
        self.idx
    }

    /// Returns the selector assigned to each column.
    pub fn selectors(&self) -> &[Selector] {
        &self.selectors
    }
}

/// Owns the source tables together with their placement in the stacked polynomial.
#[derive(Debug, Clone)]
pub struct Witness<F: Field> {
    /// Source tables behind the stacked polynomial.
    pub(super) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(super) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(super) num_variables: usize,
    /// Preprocessing depth (number of rounds the protocol folds upfront).
    pub(super) folding: usize,
    /// Stacked committed polynomial, retained only by layouts whose prover reads it back.
    pub(super) poly: Option<Poly<F>>,
}

impl<F: Field> Witness<F> {
    /// Plans the stacked layout of the given tables, one contiguous slot per column.
    ///
    /// # Algorithm
    ///
    /// - Sort tables by arity ascending; reverse-iterate to place largest first.
    /// - Each column occupies one slot of size `2^arity`.
    /// - Selector bit-width equals the stacked arity minus the table arity.
    /// - Total stacked size is rounded up to the next power of two.
    ///
    /// The stacked polynomial is written on demand by [`Self::write_stacked_slots`], straight
    /// into whatever buffer consumes it.
    ///
    /// # Panics
    ///
    /// - Table list must be non-empty.
    /// - Tables below the preprocessing depth are zero-padded to that depth.
    #[tracing::instrument(skip_all)]
    pub fn new(mut tables: Vec<Table<F>>, folding: usize) -> Self {
        // Precondition: need at least one source table to stack.
        assert!(
            !tables.is_empty(),
            "Witness requires at least one source table"
        );
        // The ordinary stacked PCS stores field slices in its retained witness. Explicitly cross
        // this dense boundary here; BooleanTracePcs source tables never construct a Witness.
        tables = tables.into_iter().map(Table::into_dense).collect();
        // Normalize small tables to the committed arity used by the protocol.
        tables.iter_mut().for_each(|table| table.pad_zeros(folding));

        // Delegate slot assignment to the shared planner (same routine as the verifier).
        let shapes: Vec<LayoutShape> = tables
            .iter()
            .map(|t| LayoutShape {
                arity: t.num_variables(),
                width: t.num_polys(),
            })
            .collect();
        let (num_variables, placements) = plan_layout(&shapes);

        Self {
            tables,
            placements,
            num_variables,
            folding,
            poly: None,
        }
    }

    /// Stacks the given tables with local variables before selector variables.
    ///
    /// # Layout
    ///
    /// Current `new` stores each column contiguously as:
    ///
    /// ```text
    ///     P(selector_bits, local_bits)
    /// ```
    ///
    /// This constructor stores each column strided by selector bits as:
    ///
    /// ```text
    ///     P(local_bits, selector_bits)
    /// ```
    ///
    /// Local table evaluation order is preserved. Only selector bitstrings are
    /// reversed from the prefix-oriented planner so mixed-arity selector codes
    /// remain suffix-disjoint.
    ///
    /// # Panics
    ///
    /// - Table list must be non-empty.
    /// - Tables below the preprocessing depth are zero-padded to that depth.
    #[tracing::instrument(skip_all)]
    pub fn new_interleaved(mut tables: Vec<Table<F>>, folding: usize) -> Self {
        assert!(
            !tables.is_empty(),
            "Witness requires at least one source table"
        );
        tables = tables.into_iter().map(Table::into_dense).collect();
        tables.iter_mut().for_each(|table| table.pad_zeros(folding));

        let shapes: Vec<LayoutShape> = tables
            .iter()
            .map(|t| LayoutShape {
                arity: t.num_variables(),
                width: t.num_polys(),
            })
            .collect();
        let (num_variables, mut placements) = plan_layout(&shapes);
        placements
            .iter_mut()
            .for_each(TablePlacement::reverse_selectors);

        let mut stacked = Poly::<F>::zero(num_variables);

        for placement in &placements {
            let table = &tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let poly = table.poly(poly_idx);

                for (local_idx, &value) in poly.as_slice().iter().enumerate() {
                    let dst = (local_idx << selector.num_variables) | selector.index;
                    stacked.as_mut_slice()[dst] = value;
                }
            }
        }

        Self {
            tables,
            placements,
            num_variables,
            folding,
            poly: Some(stacked),
        }
    }

    /// Writes the contiguous stacked layout into a zeroed buffer over the whole hypercube.
    ///
    /// Each source column is copied into its planner-assigned slot. Slots are disjoint, so
    /// columns copy independently in parallel, and a tall column in parallel chunks. Cells
    /// outside every slot are left untouched, so the unused tail keeps the caller's zeros.
    ///
    /// # Panics
    ///
    /// - `out` must hold one cell per evaluation of the stacked polynomial.
    /// - Debug builds also require `out` to arrive zeroed: whatever this call leaves
    ///   untouched is committed verbatim, so a dirty destination is a wrong commitment
    ///   that no length check can see.
    pub(super) fn write_stacked_slots(&self, out: &mut [F]) {
        assert_eq!(
            out.len(),
            1 << self.num_variables,
            "stacked destination must cover the whole hypercube"
        );
        debug_assert!(
            out.iter().all(|value| *value == F::ZERO),
            "cells outside every slot are left untouched, so the destination must arrive zeroed"
        );
        info_span!("stack").in_scope(|| {
            column_slots(&self.placements, &self.tables, 0, out)
                .into_par_iter()
                .for_each(|(slot, table_idx, poly_idx)| {
                    slot.par_chunks_mut(STACK_COPY_CHUNK)
                        .zip(
                            self.tables[table_idx]
                                .poly(poly_idx)
                                .as_slice()
                                .par_chunks(STACK_COPY_CHUNK),
                        )
                        .for_each(|(slot, column)| slot.copy_from_slice(column));
                });
        });
    }

    /// Returns the retained stacked polynomial.
    ///
    /// # Panics
    ///
    /// - The layout behind this witness does not retain a stacked polynomial.
    pub(super) const fn retained_poly(&self) -> &Poly<F> {
        self.poly
            .as_ref()
            .expect("this layout retains its stacked polynomial")
    }

    /// Materializes the stacked committed polynomial into a fresh allocation.
    ///
    /// Layouts that retain the stack clone it; the others rebuild it from the source tables
    /// and their placements.
    pub fn stacked_poly(&self) -> Poly<F> {
        self.poly.clone().unwrap_or_else(|| {
            let mut stacked = Poly::<F>::zero(self.num_variables);
            self.write_stacked_slots(stacked.as_mut_slice());
            stacked
        })
    }

    /// Returns the number of variables of the stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns verifier table shapes after witness normalization/padding.
    pub fn table_shapes(&self) -> Vec<TableShape> {
        self.tables.iter().map(Table::shape).collect()
    }

    /// Splits the witness into its owned components for downstream use.
    pub(super) fn into_parts(self) -> WitnessParts<F> {
        // Hand each field to the caller verbatim; no normalisation is needed.
        WitnessParts {
            tables: self.tables,
            placements: self.placements,
            num_variables: self.num_variables,
            folding: self.folding,
            poly: self.poly,
        }
    }
}

/// Owned components of a stacked-layout commitment.
#[derive(Debug, Clone)]
pub(super) struct WitnessParts<F: Field> {
    /// Source tables stacked into the committed polynomial.
    pub(super) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(super) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(super) num_variables: usize,
    /// Number of preprocessing rounds folded upfront.
    pub(super) folding: usize,
    /// Stacked committed polynomial, when the layout retains one.
    pub(super) poly: Option<Poly<F>>,
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_util::log2_ceil_usize;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::layout::{Layout, PrefixProver, SuffixProver};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    struct ChunkedSource {
        /// Logical dimensions exposed to the placement plan.
        shape: TableShape,
        /// Fragmented values stored by column and then by chunk.
        columns: Vec<Vec<Vec<F>>>,
    }

    impl SuffixTableSource<F> for ChunkedSource {
        fn shape(&self) -> TableShape {
            self.shape
        }

        fn fill(&self, columns: &mut [ColumnOut<'_, F>]) {
            for (out, chunks) in columns.iter_mut().zip(&self.columns) {
                out.write_with(|destination| {
                    let mut offset = 0;
                    for chunk in chunks {
                        let end = offset + chunk.len();
                        destination[offset..end].copy_from_slice(chunk);
                        offset = end;
                    }
                    assert_eq!(offset, destination.len());
                });
            }
        }
    }

    struct BitPackedSource {
        /// Logical dimensions exposed to the placement plan.
        shape: TableShape,
        /// Packed source words stored column by column.
        columns: Vec<Vec<u64>>,
    }

    impl SuffixTableSource<F> for BitPackedSource {
        fn shape(&self) -> TableShape {
            self.shape
        }

        fn fill(&self, columns: &mut [ColumnOut<'_, F>]) {
            for (out, words) in columns.iter_mut().zip(&self.columns) {
                out.write_with(|destination| {
                    for (row, value) in destination.iter_mut().enumerate() {
                        let bit = (words[row / 64] >> (row % 64)) & 1;
                        *value = F::from_u64(bit);
                    }
                });
            }
        }
    }

    struct ShortSource {
        /// Logical dimensions exposed to the placement plan.
        shape: TableShape,
        /// Logical destination length observed by the producer.
        observed_len: AtomicUsize,
    }

    impl SuffixTableSource<F> for ShortSource {
        fn shape(&self) -> TableShape {
            self.shape
        }

        fn fill(&self, columns: &mut [ColumnOut<'_, F>]) {
            self.observed_len.store(columns[0].len(), Ordering::Relaxed);
            columns[0].write_with(|destination| destination.fill(F::ONE));
        }
    }

    struct IncompleteSource {
        /// Logical dimensions exposed to the placement plan.
        shape: TableShape,
    }

    impl SuffixTableSource<F> for IncompleteSource {
        fn shape(&self) -> TableShape {
            self.shape
        }

        fn fill(&self, columns: &mut [ColumnOut<'_, F>]) {
            columns[0].write_with(|destination| destination.fill(F::ONE));
        }
    }

    fn placement_addresses(placements: &[TablePlacement]) -> Vec<(usize, Vec<(usize, usize)>)> {
        placements
            .iter()
            .map(|placement| {
                (
                    placement.idx(),
                    placement
                        .selectors()
                        .iter()
                        .map(|selector| (selector.num_variables(), selector.index()))
                        .collect(),
                )
            })
            .collect()
    }

    fn assert_direct_fill_matches_dense(tables: &[Table<F>], folding: usize) {
        let legacy = Witness::new(tables.to_vec(), folding);
        let plan = SuffixLayoutPlan::new(tables.iter().map(Table::shape).collect(), folding)
            .expect("the fixture dimensions fit a suffix layout");
        let sources = tables
            .iter()
            .map(|table| table as &dyn SuffixTableSource<F>)
            .collect::<Vec<_>>();
        let direct = plan.fill(&sources).expect("every dense column is written");

        assert_eq!(direct.poly().as_slice(), legacy.stacked_poly().as_slice());
        assert_eq!(direct.table_shapes(), legacy.table_shapes());
        assert_eq!(
            placement_addresses(direct.placements()),
            placement_addresses(&legacy.placements)
        );
        let compatible = direct.into_witness();
        assert_eq!(
            compatible.stacked_poly().as_slice(),
            legacy.stacked_poly().as_slice()
        );
        assert_eq!(compatible.table_shapes(), legacy.table_shapes());
        assert_eq!(compatible.tables.len(), legacy.tables.len());
        for (compatible_table, legacy_table) in compatible.tables.iter().zip(&legacy.tables) {
            for (compatible_poly, legacy_poly) in
                compatible_table.iter_polys().zip(legacy_table.iter_polys())
            {
                assert_eq!(compatible_poly, legacy_poly);
            }
        }
    }

    #[test]
    fn selector_new_stores_num_variables_and_index() {
        // Invariant:
        //     Constructor stores both fields verbatim.
        //
        // Fixture state:
        //     num_variables = 3, index = 5
        let sel = Selector::new(3, 5);

        // Check: getters return what was passed in.
        assert_eq!(sel.num_variables(), 3);
        assert_eq!(sel.index(), 5);
    }

    #[test]
    #[should_panic]
    fn selector_new_panics_on_index_out_of_range() {
        // Invariant:
        //     A slot index that does not fit in num_variables bits is rejected.
        //
        // Fixture state:
        //     num_variables = 2 → legal indices are 0..=3; use 4 to trigger panic.
        let _ = Selector::new(2, 4);
    }

    #[test]
    fn selector_point_returns_boolean_vector_of_right_length() {
        // Invariant:
        //     point() returns a num_variables-long vector of 0/1 field elements.
        //
        // Fixture state:
        //     num_variables = 3, index = 5 = 0b101 → point bits are {1, 0, 1}.
        let sel = Selector::new(3, 5);
        let point: Point<F> = sel.point();

        // Check: length matches the bit-width.
        assert_eq!(point.num_variables(), 3);

        // Check: every coordinate is either 0 or 1.
        for &bit in point.as_slice() {
            assert!(bit == F::ZERO || bit == F::ONE);
        }
    }

    #[test]
    fn selector_lift_prefixes_selector_bits_onto_other() {
        // Invariant:
        //     lift(other) returns a point with the selector bits prepended,
        //     total length = selector.num_variables() + other.num_variables().
        //
        // Fixture state:
        //     selector: 3-bit, index 5 → 3 prefix coordinates.
        //     other:    2-variable point over EF.
        let sel = Selector::new(3, 5);
        let other: Point<EF> = Point::new(vec![EF::from_u64(7), EF::from_u64(11)]);
        let lifted = sel.lift_prefix(&other);

        // Check: total length is the concatenation length.
        assert_eq!(lifted.num_variables(), 5);

        // Check: the suffix matches the original point element-wise.
        for i in 0..2 {
            assert_eq!(lifted.as_slice()[3 + i], other.as_slice()[i]);
        }
    }

    #[test]
    fn selector_lift_suffix_appends_selector_bits_after_other() {
        // Invariant:
        //     lift_suffix(other) returns a point with the local coordinates
        //     first and the selector bits appended, total length =
        //     other.num_variables() + selector.num_variables().
        //
        // Fixture state:
        //     selector: 3-bit, index 5 = 0b101 → bits {1, 0, 1}.
        //     other:    2-variable point over EF.
        let sel = Selector::new(3, 5);
        let other: Point<EF> = Point::new(vec![EF::from_u64(7), EF::from_u64(11)]);
        let lifted = sel.lift_suffix(&other);

        // Check: total length is the concatenation length.
        assert_eq!(lifted.num_variables(), 5);

        // Check: the prefix matches the original point element-wise.
        assert_eq!(lifted.as_slice()[0], other.as_slice()[0]);
        assert_eq!(lifted.as_slice()[1], other.as_slice()[1]);

        // Check: the suffix is the boolean expansion of the slot index.
        let selector_bits: Point<EF> = sel.point();
        assert_eq!(lifted.as_slice()[2], selector_bits.as_slice()[0]);
        assert_eq!(lifted.as_slice()[3], selector_bits.as_slice()[1]);
        assert_eq!(lifted.as_slice()[4], selector_bits.as_slice()[2]);
    }

    #[test]
    fn selector_reverse_swaps_msb_and_lsb_within_width() {
        // Invariant:
        //     reverse() flips the slot index bitstring within num_variables bits,
        //     leaving num_variables itself untouched.
        //
        // Fixture state:
        //     num_variables = 4, index = 0b0010 = 2 → reversed: 0b0100 = 4.
        let mut sel = Selector::new(4, 0b0010);
        sel.reverse();

        // Check: bit-width is preserved.
        assert_eq!(sel.num_variables(), 4);
        // Check: index has been bit-reversed within those 4 bits.
        assert_eq!(sel.index(), 0b0100);
    }

    #[test]
    fn selector_reverse_is_idempotent_under_double_application() {
        // Invariant:
        //     Reversing a selector twice restores the original index.
        //
        // Fixture state:
        //     num_variables = 5, index = 13 → reverse → reverse → 13.
        let mut sel = Selector::new(5, 13);
        let original_index = sel.index();

        sel.reverse();
        sel.reverse();

        assert_eq!(sel.num_variables(), 5);
        assert_eq!(sel.index(), original_index);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn table_new_panics_on_non_power_of_two_width() {
        // Invariant:
        //     Each row is a hypercube evaluation table, so its width must be 2^k.
        //
        // Fixture state:
        //     one row of width 3 → not a power of two → must panic.
        let _ = Table::new(RowMajorMatrix::new(F::zero_vec(3), 3));
    }

    #[test]
    #[should_panic(expected = "at least one column")]
    fn table_new_panics_on_zero_columns() {
        // Invariant:
        //     A table with no columns (no rows) is rejected.
        //
        // Fixture state:
        //     empty buffer with width 2 → height 0 → must panic.
        let _ = Table::new(RowMajorMatrix::new(Vec::<F>::new(), 2));
    }

    #[test]
    fn table_accessors_report_shape() {
        // Invariant:
        //     num_polys, num_variables, size, and poly(id) all agree with the input.
        //
        // Fixture state:
        //     two columns of arity 3 → num_polys = 2, num_variables = 3, size = 2^3 * 2 = 16.
        let table = Table::<F>::zero(2, 3);

        // Check: all shape queries match the fixture.
        assert_eq!(table.num_polys(), 2);
        assert_eq!(table.num_variables(), 3);
        // Check: column lookup returns a ref to the i-th poly with matching arity.
        assert_eq!(table.poly(0).num_variables(), 3);
    }

    #[test]
    fn packed_table_column_views_preserve_row_major_bit_order() {
        let words = RowMajorMatrix::new(vec![0b1010_0101u64, 0b1100_0011u64], 2);
        let table = Table::<F>::from_packed_bits(words, 3);

        assert_eq!(table.num_polys(), 2);
        assert_eq!(table.num_variables(), 3);
        assert!(table.packed_bits().is_some());
        assert_eq!(table.column(0).len(), 8);
        assert_eq!(table.column(1).len(), 8);
        assert_eq!(table.column(0).boolean_word(0), Some(0b1010_0101));
        assert_eq!(table.column(1).boolean_word(0), Some(0b1100_0011));
        assert_eq!(
            table.column(0).values().collect::<Vec<_>>(),
            (0..8)
                .map(|row| F::from_bool(((0b1010_0101u64 >> row) & 1) != 0))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            table
                .columns()
                .map(|column| column.values().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![
                table.column(0).values().collect::<Vec<_>>(),
                table.column(1).values().collect::<Vec<_>>(),
            ]
        );
    }

    #[test]
    fn packed_table_explicit_dense_conversion_matches_boolean_columns() {
        let table = Table::<F>::from_packed_bits(
            RowMajorMatrix::new(vec![0b0000_1101u64, 0b0000_1011u64], 2),
            3,
        );
        let dense = table.clone().into_dense();

        assert!(dense.packed_bits().is_none());
        for column in 0..2 {
            assert_eq!(
                dense.column(column).values().collect::<Vec<_>>(),
                table.column(column).values().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn packed_table_geometry_and_suffix_fill_cover_word_boundaries() {
        for num_variables in [0, 1, 5, 6, 7] {
            let height = 1usize << num_variables;
            let width = 3;
            let words = (0..height.div_ceil(64))
                .flat_map(|block| {
                    (0..width).map(move |column| {
                        (0..64).fold(0u64, |word, lane| {
                            let row = block * 64 + lane;
                            if row < height && (row + column * 3) % 5 < 2 {
                                word | (1u64 << lane)
                            } else {
                                word
                            }
                        })
                    })
                })
                .collect::<Vec<_>>();
            let packed =
                Table::<F>::from_packed_bits(RowMajorMatrix::new(words, width), num_variables);
            for column in 0..width {
                for row in 0..height {
                    let expected = if (row + column * 3) % 5 < 2 {
                        F::ONE
                    } else {
                        F::ZERO
                    };
                    assert_eq!(packed.column(column).value(row), expected);
                }
            }
            let dense = packed.clone().into_dense();
            let plan = SuffixLayoutPlan::new(vec![packed.shape()], 0).unwrap();
            let direct = plan.fill(&[&packed]).unwrap();
            let legacy = Witness::new(vec![dense], 0);
            assert_eq!(direct.poly().as_slice(), legacy.stacked_poly().as_slice());
            assert_eq!(direct.table_shapes(), legacy.table_shapes());
        }
    }

    #[test]
    #[should_panic(expected = "padding bits")]
    fn packed_table_rejects_nonzero_padding_bits() {
        let _ = Table::<F>::from_packed_bits(RowMajorMatrix::new(vec![0b1000_0001u64], 1), 2);
    }

    #[test]
    #[should_panic(expected = "exactly")]
    fn packed_table_rejects_wrong_physical_matrix_size() {
        let _ = Table::<F>::from_packed_bits(RowMajorMatrix::new(vec![0u64], 1), 7);
    }

    #[test]
    #[should_panic(expected = "at least one column")]
    fn packed_table_rejects_zero_columns() {
        let _ = Table::<F>::from_packed_bits(RowMajorMatrix::new(Vec::new(), 0), 0);
    }

    #[test]
    #[should_panic(expected = "height overflows")]
    fn packed_table_rejects_height_overflow() {
        let _ =
            Table::<F>::from_packed_bits(RowMajorMatrix::new(vec![0u64], 1), usize::BITS as usize);
    }

    #[test]
    fn table_placement_accessors_return_stored_values() {
        // Invariant:
        //     TablePlacement forwards the table index and the selector slice.
        //
        // Fixture state:
        //     idx = 7, two selectors with num_variables = 2.
        let selectors = vec![Selector::new(2, 0), Selector::new(2, 1)];
        let placement = TablePlacement::new(7, selectors);

        // Check: idx() forwards the constructor argument.
        assert_eq!(placement.idx(), 7);
        // Check: num_polys() matches the selector count.
        assert_eq!(placement.num_polys(), 2);
        // Check: selectors() exposes the underlying slice.
        assert_eq!(placement.selectors().len(), 2);
    }

    #[test]
    fn table_placement_reverse_selectors_flips_each_in_place() {
        // Invariant:
        //     reverse_selectors() bit-reverses every selector index in place,
        //     leaving each selector's width and the placement's table index
        //     untouched.
        //
        // Fixture state:
        //     idx = 4, three selectors of width 3 with indices {0b001, 0b010, 0b110}.
        //     Reversed indices within 3 bits: {0b100, 0b010, 0b011} = {4, 2, 3}.
        let selectors = vec![
            Selector::new(3, 0b001),
            Selector::new(3, 0b010),
            Selector::new(3, 0b110),
        ];
        let mut placement = TablePlacement::new(4, selectors);

        placement.reverse_selectors();

        // Check: the table index is preserved.
        assert_eq!(placement.idx(), 4);
        // Check: every selector's width is preserved.
        for selector in placement.selectors() {
            assert_eq!(selector.num_variables(), 3);
        }
        // Check: each selector index has been bit-reversed within 3 bits.
        let indices: Vec<usize> = placement.selectors().iter().map(Selector::index).collect();
        assert_eq!(indices, vec![0b100, 0b010, 0b011]);
    }

    #[test]
    fn table_shape_reports_arity_and_width() {
        // Invariant:
        //     shape() returns the (num_variables, width) pair derived from the
        //     table's columns.
        //
        // Fixture state:
        //     three columns of arity 4 → expected shape: (4, 3).
        let table = Table::<F>::zero(3, 4);

        let shape = table.shape();

        assert_eq!(shape.num_variables(), 4);
        assert_eq!(shape.width(), 3);
    }

    // Builds a deterministic witness with two tables of arities (4, 3) and
    // two columns each. Used by several tests to avoid repeating the setup.
    fn fixture_witness() -> Witness<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        // Table 0: arity 3, two columns.
        let t0 = Table::rand(&mut rng, 2, 3);
        // Table 1: arity 4, two columns.
        let t1 = Table::rand(&mut rng, 2, 4);
        Witness::new(vec![t0, t1], 1)
    }

    fn fixture_interleaved_witness() -> Witness<F> {
        let mut rng = SmallRng::seed_from_u64(1);
        // Table 0: arity 3, two columns.
        let t0 = Table::rand(&mut rng, 2, 3);
        // Table 1: arity 4, two columns.
        let t1 = Table::rand(&mut rng, 2, 4);
        Witness::new_interleaved(vec![t0, t1], 1)
    }

    #[test]
    fn witness_new_places_largest_table_first() {
        // Invariant:
        //     Placement order is tables sorted by arity ascending, reversed.
        //
        // Fixture state:
        //     table 0: arity 3, table 1: arity 4.
        //     expected placement order: [table 1 (arity 4), table 0 (arity 3)]
        let w = fixture_witness();

        // Check: first placement targets the larger table.
        assert_eq!(w.placements[0].idx(), 1);
        // Check: second placement targets the smaller table.
        assert_eq!(w.placements[1].idx(), 0);
    }

    #[test]
    fn witness_num_variables_is_stacked_arity_rounded_up() {
        // Invariant:
        //     num_variables equals log2_ceil of total stacked size.
        //
        // Fixture state:
        //     total size = 2^3 * 2 + 2^4 * 2 = 16 + 32 = 48 → ceil(log2) = 6.
        let w = fixture_witness();
        assert_eq!(w.num_variables(), 6);
    }

    #[test]
    fn witness_stacks_column_evals_into_slots() {
        // Invariant:
        //     Every column's evaluations appear in the stacked polynomial
        //     at the offset computed by its selector.
        //
        // Fixture state:
        //     two columns per table, two tables; each column occupies its
        //     own slot; destination = selector.index << arity.
        let w = fixture_witness();
        let stacked = w.stacked_poly();

        // Walk every placement; for each column, compare the slot slice to the source column.
        for placement in &w.placements {
            let table = &w.tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let col = table.poly(poly_idx);
                let dst = selector.index() << col.num_variables();
                let slot = &stacked.as_slice()[dst..dst + col.num_evals()];
                // Check: slot contents match the source column evaluations.
                assert_eq!(slot, col.as_slice());
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic = "the destination must arrive zeroed"]
    fn stacking_rejects_a_destination_that_is_not_zeroed() {
        // Invariant:
        //     Cells outside every slot are committed exactly as they arrive, so a destination
        //     that is not already zeroed is a wrong commitment no length check can see.
        let w = fixture_witness();
        let mut out = vec![F::ONE; 1 << w.num_variables()];
        w.write_stacked_slots(&mut out);
    }

    #[test]
    fn witness_new_interleaves_by_suffix_selectors() {
        // Invariant:
        //     Local-first stacking stores P(local_bits, selector_bits), preserving
        //     each table's local evaluation order.
        //
        // Fixture state:
        //     A has arity 2: [a0, a1, a2, a3]
        //     B has arity 1: [b0, b1]
        //     Expected storage: [a0, b0, a1, 0, a2, b1, a3, 0].
        let a0 = F::from_u64(10);
        let a1 = F::from_u64(11);
        let a2 = F::from_u64(12);
        let a3 = F::from_u64(13);
        let b0 = F::from_u64(20);
        let b1 = F::from_u64(21);

        let table_a = Table::new(RowMajorMatrix::new(vec![a0, a1, a2, a3], 4));
        let table_b = Table::new(RowMajorMatrix::new(vec![b0, b1], 2));
        let witness = Witness::new_interleaved(vec![table_a, table_b], 0);

        assert_eq!(witness.num_variables(), 3);
        assert_eq!(
            witness.stacked_poly().as_slice(),
            &[a0, b0, a1, F::ZERO, a2, b1, a3, F::ZERO],
        );
    }

    #[test]
    fn witness_new_pads_tables_below_folding() {
        // Invariant:
        //     A table smaller than the preprocessing depth is committed as the
        //     zero-padded polynomial with arity equal to folding.
        let a0 = F::from_u64(10);
        let a1 = F::from_u64(11);
        let table = Table::new(RowMajorMatrix::new(vec![a0, a1], 2));

        let witness = Witness::new(vec![table], 3);

        assert_eq!(witness.tables[0].num_variables(), 3);
        assert_eq!(witness.num_variables(), 3);
        assert_eq!(
            witness.tables[0].poly(0).as_slice(),
            &[a0, a1, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        );
        assert_eq!(
            witness.stacked_poly().as_slice(),
            &[a0, a1, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        );
    }

    #[test]
    fn witness_table_shapes_returns_shapes_in_source_order() {
        // Invariant:
        //     table_shapes() returns one shape per source table, in the order
        //     the witness was constructed (independent of placement order).
        //
        // Fixture state:
        //     table 0: arity 3, two columns → shape (3, 2).
        //     table 1: arity 4, two columns → shape (4, 2).
        //     Source order: [table 0, table 1].
        let w = fixture_witness();

        let shapes = w.table_shapes();

        assert_eq!(shapes, vec![TableShape::new(3, 2), TableShape::new(4, 2)],);
    }

    #[test]
    fn witness_stacked_poly_returns_the_stacked_polynomial() {
        // Invariant:
        //     stacked_poly() returns the stacked committed polynomial. Its arity equals
        //     num_variables(), and it covers that hypercube with the unused tail zeroed.
        //
        // Fixture state:
        //     two-table fixture; expected stacked arity = 6; occupied size = 48.
        let w = fixture_witness();

        let stacked = w.stacked_poly();

        assert_eq!(stacked.num_variables(), w.num_variables());
        assert_eq!(stacked.as_slice().len(), 1 << w.num_variables());
        assert!(stacked.as_slice()[48..].iter().all(|&v| v == F::ZERO));
    }

    #[test]
    fn witness_into_parts_preserves_every_field() {
        // Invariant:
        //     into_parts() destructures the witness while preserving every
        //     field byte-for-byte: source tables (in source order), placement
        //     metadata, stacked arity, folding depth, and stacked polynomial.
        //
        // Fixture state:
        //     two-table fixture; folding = 1.
        let w = fixture_witness();
        // Snapshot every field before consumption.
        let expected_num_variables = w.num_variables();
        let expected_folding = w.folding;
        let expected_table_shapes = w.table_shapes();
        let expected_placement_idx: Vec<usize> =
            w.placements.iter().map(TablePlacement::idx).collect();
        let expected_poly = w.poly.clone();

        let parts = w.into_parts();

        // Check: scalar fields survive the move.
        assert_eq!(parts.num_variables, expected_num_variables);
        assert_eq!(parts.folding, expected_folding);
        // Check: source tables are carried over in source order.
        let actual_table_shapes: Vec<TableShape> = parts.tables.iter().map(Table::shape).collect();
        assert_eq!(actual_table_shapes, expected_table_shapes);
        // Check: placement order and table indices survive verbatim.
        let actual_placement_idx: Vec<usize> =
            parts.placements.iter().map(TablePlacement::idx).collect();
        assert_eq!(actual_placement_idx, expected_placement_idx);
        // Check: the retained stacked polynomial survives the move verbatim.
        assert_eq!(
            parts.poly.map(|poly| poly.as_slice().to_vec()),
            expected_poly.map(|poly| poly.as_slice().to_vec())
        );
    }

    #[test]
    fn prefix_prover_from_witness_carries_stacked_state() {
        // Invariant:
        //     Handing the witness to the prefix prover preserves the
        //     stacked polynomial and the per-table shapes.
        let w = fixture_interleaved_witness();
        let stacked_copy = w.stacked_poly();
        let num_variables = w.num_variables();

        // Build a prefix-mode prover from the witness.
        let prover = PrefixProver::<F, EF>::from_witness(w);

        // Check: arity matches the original stacked polynomial.
        assert_eq!(prover.num_variables(), num_variables);
        // Check: the committed polynomial is bit-for-bit identical.
        assert_eq!(prover.poly.as_slice(), stacked_copy.as_slice());
    }

    #[test]
    #[should_panic = "this layout retains its stacked polynomial"]
    fn prefix_prover_refuses_a_contiguously_stacked_witness() {
        // Invariant:
        //     The prefix prover folds the stacked polynomial directly, so it only accepts a
        //     witness built by its own constructor. A contiguously stacked witness carries a
        //     different variable order and no retained polynomial, and is refused outright
        //     rather than silently folded in the wrong order.
        let _ = PrefixProver::<F, EF>::from_witness(fixture_witness());
    }

    #[test]
    fn suffix_prover_from_witness_carries_stacked_state() {
        // Invariant:
        //     Handing the witness to the suffix prover preserves the
        //     stacked arity and the per-table data layout.
        //
        //     The suffix prover does not retain the stacked polynomial —
        //     it walks per-table evaluations on demand — so this test only
        //     checks the structural fields that survive the move.
        let w = fixture_witness();
        let num_variables = w.num_variables();
        let table_shapes = w.table_shapes();

        // Build a suffix-mode prover from the witness.
        let prover = SuffixProver::<F, EF>::from_witness(w);

        // Check: stacked arity matches.
        assert_eq!(prover.num_variables(), num_variables);
        // Check: every source table is carried over with its original shape.
        let prover_shapes: Vec<TableShape> =
            prover.claims.tables.iter().map(Table::shape).collect();
        assert_eq!(prover_shapes, table_shapes);
    }

    #[test]
    fn direct_dense_fill_matches_the_existing_suffix_witness() {
        let mut rng = SmallRng::seed_from_u64(0xD3E5E);
        let tables = vec![Table::rand(&mut rng, 3, 8), Table::rand(&mut rng, 2, 8)];

        assert_direct_fill_matches_dense(&tables, 0);
    }

    #[test]
    fn direct_jagged_fill_matches_the_existing_suffix_witness() {
        let mut rng = SmallRng::seed_from_u64(0x1A663D);
        let tables = vec![
            Table::rand(&mut rng, 3, 4),
            Table::rand(&mut rng, 1, 9),
            Table::rand(&mut rng, 2, 6),
        ];

        assert_direct_fill_matches_dense(&tables, 0);
    }

    #[test]
    fn fragmented_columns_fill_the_same_final_stack_as_dense_columns() {
        let first = (0..32).map(F::from_usize).collect::<Vec<_>>();
        let second = (100..132).map(F::from_usize).collect::<Vec<_>>();
        let dense = Table::new(RowMajorMatrix::new(
            first.iter().chain(&second).copied().collect(),
            32,
        ));
        let source = ChunkedSource {
            shape: TableShape::new(5, 2),
            columns: vec![
                vec![
                    first[..3].to_vec(),
                    first[3..19].to_vec(),
                    first[19..].to_vec(),
                ],
                vec![second[..17].to_vec(), second[17..].to_vec()],
            ],
        };
        let plan = SuffixLayoutPlan::new(vec![source.shape], 0).unwrap();
        let direct = plan.fill(&[&source]).unwrap();
        let legacy = Witness::new(vec![dense], 0);

        assert_eq!(direct.poly().as_slice(), legacy.stacked_poly().as_slice());
    }

    #[test]
    fn packed_bits_decode_directly_into_the_dense_reference_layout() {
        let words = vec![
            vec![0x0123_4567_89AB_CDEF, 0xF0F0_0F0F_AAAA_5555],
            vec![0xDEAD_BEEF_CAFE_BABE, 0x8000_0000_0000_0001],
        ];
        let scalar = words
            .iter()
            .flat_map(|column| {
                (0..128).map(move |row| F::from_u64((column[row / 64] >> (row % 64)) & 1))
            })
            .collect::<Vec<_>>();
        let dense = Table::new(RowMajorMatrix::new(scalar, 128));
        let source = BitPackedSource {
            shape: TableShape::new(7, 2),
            columns: words,
        };
        let plan = SuffixLayoutPlan::new(vec![source.shape], 0).unwrap();
        let direct = plan.fill(&[&source]).unwrap();
        let legacy = Witness::new(vec![dense], 0);

        assert_eq!(direct.poly().as_slice(), legacy.stacked_poly().as_slice());
    }

    #[test]
    fn heterogeneous_sources_share_one_checked_fill() {
        // One batch covers the three producer layouts expected at the ingestion boundary.
        // The folding depth also places logical arities on both sides of normalization.
        let dense = Table::new(RowMajorMatrix::new((0..4).map(F::from_usize).collect(), 2));
        let chunked_values = (20..52).map(F::from_usize).collect::<Vec<_>>();
        let chunked = ChunkedSource {
            shape: TableShape::new(5, 1),
            columns: vec![vec![
                chunked_values[..7].to_vec(),
                chunked_values[7..].to_vec(),
            ]],
        };
        let packed_words = vec![vec![0xA5], vec![0x3C], vec![0x81]];
        let packed = BitPackedSource {
            shape: TableShape::new(3, 3),
            columns: packed_words.clone(),
        };

        // Dense references define the exact polynomial expected from each custom source.
        let chunked_table = Table::new(RowMajorMatrix::new(chunked_values, 32));
        let packed_table = Table::new(RowMajorMatrix::new(
            packed_words
                .iter()
                .flat_map(|words| {
                    (0..8).map(|row| F::from_u64((words[row / 64] >> (row % 64)) & 1))
                })
                .collect(),
            8,
        ));
        let tables = vec![dense.clone(), chunked_table, packed_table];
        let plan = SuffixLayoutPlan::new(tables.iter().map(Table::shape).collect(), 3).unwrap();

        // The direct stack must equal the established dense path for the whole mixed batch.
        let direct = plan.fill(&[&dense, &chunked, &packed]).unwrap();
        let legacy = Witness::new(tables, 3);
        assert_eq!(direct.poly().as_slice(), legacy.stacked_poly().as_slice());
        let compatible = direct.into_witness();
        assert_eq!(
            compatible.stacked_poly().as_slice(),
            legacy.stacked_poly().as_slice()
        );
        assert_eq!(compatible.table_shapes(), legacy.table_shapes());
    }

    #[test]
    fn preprocessing_padding_is_zero_and_hidden_from_the_producer() {
        let source = ShortSource {
            shape: TableShape::new(2, 1),
            observed_len: AtomicUsize::new(0),
        };
        let plan = SuffixLayoutPlan::new(vec![source.shape], 4).unwrap();
        let direct = plan.fill(&[&source]).unwrap();

        assert_eq!(source.observed_len.load(Ordering::Relaxed), 4);
        assert_eq!(direct.table_shapes(), &[TableShape::new(4, 1)]);
        assert_eq!(&direct.poly().as_slice()[..4], &[F::ONE; 4]);
        assert!(
            direct.poly().as_slice()[4..]
                .iter()
                .all(|&value| value == F::ZERO)
        );
    }

    #[test]
    fn an_unwritten_column_is_rejected_after_safe_zero_initialization() {
        let source = IncompleteSource {
            shape: TableShape::new(3, 2),
        };
        let plan = SuffixLayoutPlan::new(vec![source.shape], 0).unwrap();
        let error = plan
            .fill(&[&source])
            .expect_err("the second column has no completed write");

        assert_eq!(
            error,
            SuffixFillError::UnwrittenColumn {
                table: 0,
                column: 1,
            }
        );
    }

    #[test]
    fn source_shape_drift_is_rejected_before_filling() {
        let source = IncompleteSource {
            shape: TableShape::new(3, 2),
        };
        let plan = SuffixLayoutPlan::new(vec![TableShape::new(3, 1)], 0).unwrap();
        let error = plan
            .fill(&[&source])
            .expect_err("the source width differs from the planned width");

        assert_eq!(
            error,
            SuffixFillError::ShapeMismatch {
                table: 0,
                expected: TableShape::new(3, 1),
                actual: source.shape,
            }
        );
    }

    #[test]
    fn an_empty_plan_is_rejected() {
        let error = SuffixLayoutPlan::new(Vec::new(), 0)
            .expect_err("an empty table batch has no committed statement");

        assert_eq!(error, SuffixLayoutPlanError::NoTables);
    }

    #[test]
    fn a_rounded_stack_larger_than_the_address_space_is_rejected() {
        let shape = TableShape::new(usize::BITS as usize - 1, 2);
        let error = SuffixLayoutPlan::new(vec![shape], 0)
            .expect_err("the two column cubes overflow the address space");

        assert_eq!(error, SuffixLayoutPlanError::CellCountOverflow);
    }

    #[test]
    fn a_source_count_mismatch_is_rejected_before_allocation() {
        let plan = SuffixLayoutPlan::new(vec![TableShape::new(3, 1)], 0).unwrap();
        let error = plan
            .fill::<F>(&[])
            .expect_err("the planned table has no source");

        assert_eq!(
            error,
            SuffixFillError::TableCount {
                expected: 1,
                actual: 0,
            }
        );
    }

    // Proptest strategy: random table shapes within safe bounds.
    //
    //     1..=3 tables, each with arity in 2..=5 and 1..=3 columns.
    //     folding = 1 fits since every arity is at least 2.
    fn arb_table_shapes() -> impl Strategy<Value = Vec<(usize, usize)>> {
        prop::collection::vec((2usize..=5, 1usize..=3), 1..=3)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

        // Invariant:
        //     For any valid set of table shapes:
        //     - every column's evaluations live at selector.index << arity
        //     - num_variables equals log2_ceil of total size
        //     - trailing slots past the concatenation stay zero
        #[test]
        fn witness_stacks_columns_and_zeros_the_tail(shapes in arb_table_shapes()) {
            let mut rng = SmallRng::seed_from_u64(123);

            // Build one table per (arity, width) entry with random evaluations.
            let tables: Vec<Table<F>> = shapes
                .iter()
                .map(|&(arity, width)| {
                    Table::rand(&mut rng, width, arity)
                })
                .collect();

            // Total stacked size (before power-of-two rounding).
            let total_used: usize = tables
                .iter()
                .map(|t| (1 << t.num_variables()) * t.num_polys())
                .sum();

            // Folding = 1 is always safe since the strategy guarantees arity >= 2.
            let witness = Witness::new(tables, 1);

            // Check: stacked arity equals log2_ceil of total occupied size.
            assert_eq!(witness.num_variables(), log2_ceil_usize(total_used));
            let stacked = witness.stacked_poly();

            // Check: each column's evaluations land at the predicted slot.
            let mut used = 0usize;
            for placement in &witness.placements {
                let table = &witness.tables[placement.idx()];
                for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                    let col = table.poly(poly_idx);
                    let dst = selector.index() << col.num_variables();
                    let slot = &stacked.as_slice()[dst..dst + col.num_evals()];
                    assert_eq!(slot, col.as_slice());
                    used += col.num_evals();
                }
            }

            // Check: the counted occupied region matches total_used.
            assert_eq!(used, total_used);

            // Check: every entry past the concatenation is the zero element.
            let stacked_len = 1usize << witness.num_variables();
            for &v in &stacked.as_slice()[used..stacked_len] {
                // The specific region of "used" is contiguous here only because
                // placements are emitted largest-first and slot offsets are the
                // cursor value. That matches the "unused tail stays zero" rule.
                assert_eq!(v, F::ZERO);
            }
        }

        #[test]
        fn direct_fill_matches_dense_with_normalization(
            shapes in arb_table_shapes(),
            folding in 0usize..=6,
        ) {
            // Randomized dimensions exercise placement changes caused by preprocessing padding.
            let mut rng = SmallRng::seed_from_u64(0xD1EC7);
            let tables = shapes
                .iter()
                .map(|&(arity, width)| Table::rand(&mut rng, width, arity))
                .collect::<Vec<_>>();

            // Both paths must commit the same cells and derive the same placement metadata.
            assert_direct_fill_matches_dense(&tables, folding);
        }
    }
}
