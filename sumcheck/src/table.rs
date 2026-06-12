use alloc::vec::Vec;

use p3_matrix::Dimensions;
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

/// Shape-only description of one verifier table.
///
/// # Contents
///
/// - Row count, always a power of two.
/// - Column count, strictly positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableShape(Dimensions);

impl TableShape {
    /// Builds a table shape of `2^k` rows and `width` columns.
    ///
    /// # Panics
    ///
    /// - Column count must be at least one.
    /// - Log row count must fit in the target's pointer width.
    pub const fn new(num_variables: usize, width: usize) -> Self {
        // Positive column count.
        assert!(width > 0);
        // Bound on the shift to rule out `1 << k` overflow on the current target.
        assert!(num_variables < usize::BITS as usize);
        // Expand the log row count to the concrete row count.
        Self(Dimensions {
            width,
            height: 1 << num_variables,
        })
    }

    /// Returns the number of variables per column.
    pub const fn num_variables(&self) -> usize {
        log2_ceil_usize(self.0.height)
    }

    /// Returns the number of columns.
    pub const fn width(&self) -> usize {
        self.0.width
    }
}

/// Opening batch sampled at one transcript point for a table.
///
/// # Overview
///
/// - One batch records what is opened at a single sampled point.
/// - It holds two column-index lists evaluated at that same point.
/// - Transcript and evaluation order is always the first list, then the second.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpeningBatch<T> {
    /// Columns evaluated directly at the sampled point.
    current: Vec<T>,
    /// Columns evaluated through the repeat-last successor view at the same point.
    next: Vec<T>,
}

impl<T> OpeningBatch<T> {
    /// Builds an opening batch from the direct list and the successor-view list.
    ///
    /// # Arguments
    ///
    /// - `current`: columns evaluated directly at the sampled point.
    /// - `next`: columns evaluated through the repeat-last successor view.
    ///
    /// # Panics
    ///
    /// Panics if both lists are empty.
    pub fn new(current: Vec<T>, next: Vec<T>) -> Self {
        // Invariant: a batch must name at least one column on either side.
        // An empty batch would consume a transcript point yet open nothing.
        assert!(
            !current.is_empty() || !next.is_empty(),
            "opening batch must name at least one column"
        );
        Self { current, next }
    }

    /// Returns the columns evaluated directly at the sampled point.
    pub fn current(&self) -> &[T] {
        &self.current
    }

    /// Returns the columns evaluated through the repeat-last successor view.
    pub fn next(&self) -> &[T] {
        &self.next
    }

    /// Returns the total number of entries across both lists.
    pub const fn len(&self) -> usize {
        // Both sides share the same sampled point, so the batch size is their sum.
        self.current.len() + self.next.len()
    }

    /// Returns true when neither list holds an entry.
    pub const fn is_empty(&self) -> bool {
        self.current.is_empty() && self.next.is_empty()
    }

    /// Returns true when both per-side list lengths match the other batch.
    ///
    /// # Arguments
    ///
    /// - `other`: the batch whose per-side lengths are compared against this one.
    pub const fn has_same_shape<U>(&self, other: &OpeningBatch<U>) -> bool {
        // Shape is the pair of side lengths; entry values are irrelevant here.
        self.current.len() == other.current.len() && self.next.len() == other.next.len()
    }
}

impl<T: Clone> OpeningBatch<T> {
    /// Flattens both lists into one vector in transcript order.
    ///
    /// # Returns
    ///
    /// - The direct entries first.
    /// - The successor-view entries second.
    pub fn to_vec(&self) -> Vec<T> {
        // Walk in transcript order (direct entries, then successor-view entries)
        // and copy each entry into a flat vector.
        self.current.iter().chain(&self.next).cloned().collect()
    }
}

/// Column indices requested at one sampled point for a table.
///
/// # Overview
///
/// - One request names the columns opened at a single sampled point.
/// - Each side may name direct columns, successor-view columns, or both.
pub type OpeningRequest = OpeningBatch<usize>;

/// Claimed column evaluations matching one request at a single sampled point.
pub type OpeningEvals<EF> = OpeningBatch<EF>;

/// Sequence of per-point opening requests for one table.
///
/// One entry per sampled point, in transcript order.
pub type PointSchedule = Vec<OpeningRequest>;

/// Description of a table used to build randomized stacked-sumcheck witnesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSpec {
    /// Logical table shape used to generate the witness.
    ///
    /// The committed shape may be padded later if the table has fewer variables
    /// than the first folding round consumes.
    shape: TableShape,
    /// Opening schedule local to this table.
    point_schedule: PointSchedule,
}

impl TableSpec {
    /// Builds a table spec from a shape and point-local opening schedule.
    ///
    /// # Panics
    ///
    /// - Every scheduled polynomial index must be less than the table width.
    /// - A column may not repeat within one side of a batch.
    pub fn new(shape: TableShape, point_schedule: PointSchedule) -> Self {
        // Invariant: every requested column must address a real column of the table.
        // Both sides of each batch index into the same set of table columns,
        // so direct and successor-view indices are validated against one width.
        assert!(point_schedule.iter().all(|batch| {
            batch
                .current()
                .iter()
                .chain(batch.next())
                .all(|&poly_idx| poly_idx < shape.width())
        }));
        // Invariant: a column appears at most once per side of a batch.
        // Opening one column on both sides is intended.
        // Repeating it within one side only burns a challenge power on a duplicate claim.
        let side_has_no_repeat = |cols: &[usize]| {
            cols.iter()
                .enumerate()
                .all(|(i, col)| !cols[..i].contains(col))
        };
        assert!(
            point_schedule.iter().all(
                |batch| side_has_no_repeat(batch.current()) && side_has_no_repeat(batch.next())
            ),
            "an opening batch must not repeat a column within its current or next side"
        );
        Self {
            shape,
            point_schedule,
        }
    }

    /// Returns the logical table shape.
    pub const fn shape(&self) -> &TableShape {
        &self.shape
    }

    /// Returns the point-local opening schedule.
    pub const fn point_schedule(&self) -> &PointSchedule {
        &self.point_schedule
    }

    /// Pads this table shape to at least `min_num_variables`.
    pub const fn pad_to_min_num_variables(&mut self, min_num_variables: usize) {
        if self.shape.num_variables() < min_num_variables {
            self.shape = TableShape::new(min_num_variables, self.shape.width());
        }
    }
}

/// Public protocol describing committed tables and their opening schedule.
///
/// This is the shape agreement between prover and verifier. The prover must
/// commit to a witness whose table shapes match this protocol, and later opens
/// exactly the point-local column batches listed here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpeningProtocol(Vec<TableSpec>);

impl OpeningProtocol {
    /// Builds an opening protocol from table specs.
    pub const fn new(tables: Vec<TableSpec>) -> Self {
        Self(tables)
    }

    /// Returns the table shapes in protocol order.
    pub fn table_shapes(&self) -> Vec<TableShape> {
        self.0.iter().map(|table| *table.shape()).collect()
    }

    /// Pads every table shape to at least `min_num_variables`.
    ///
    /// The column index lists in each schedule are unchanged.
    /// Padding only appends zero rows to the committed table.
    ///
    /// The successor view is always taken over the padded space.
    /// Padding therefore moves the repeated boundary row into the zero pad.
    /// The last real row's successor then reads a pad row instead of repeating itself.
    /// A successor opening after padding is a different polynomial identity than over the unpadded table.
    /// Prover and verifier still agree, since both work over the padded space.
    pub fn pad_to_min_num_variables(mut self, min_num_variables: usize) -> Self {
        self.0
            .iter_mut()
            .for_each(|table| table.pad_to_min_num_variables(min_num_variables));
        self
    }

    /// Returns the total number of point-local opening batches.
    pub fn num_openings(&self) -> usize {
        self.0
            .iter()
            .map(|table| table.point_schedule().len())
            .sum()
    }

    /// Iterates over all opening batches in transcript order.
    pub fn iter_openings(&self) -> impl Iterator<Item = (usize, &OpeningRequest)> {
        // Walk tables in protocol order, then each table's points in schedule order.
        // The flat stream pairs every request with the index of its owning table.
        //
        //     table 0: [req, req]   table 1: [req]
        //          -> (0, req), (0, req), (1, req)
        self.0.iter().enumerate().flat_map(|(table_idx, table)| {
            table
                .point_schedule()
                .iter()
                .map(move |batch| (table_idx, batch))
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    // Single-table protocol: arity 3, two columns, two opening points.
    //
    //     point 0: direct columns {0, 1}
    //     point 1: direct column  {0}
    //
    // Total opening count is 2; the two batches stream in transcript order
    // against table index 0.
    fn single_table_protocol() -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(3, 2),
            vec![
                OpeningBatch::new(vec![0, 1], Vec::new()),
                OpeningBatch::new(vec![0], Vec::new()),
            ],
        )])
    }

    // Two-table protocol with distinct shapes and schedules.
    //
    //     table 0: arity 3, two cols.   one point, direct {0, 1}
    //     table 1: arity 4, three cols. point a direct {0, 2}; point b direct {1}
    //
    // Total opening count is 3; the stream is
    // (0, direct [0, 1]), (1, direct [0, 2]), (1, direct [1]).
    fn two_table_protocol() -> OpeningProtocol {
        OpeningProtocol::new(vec![
            TableSpec::new(
                TableShape::new(3, 2),
                vec![OpeningBatch::new(vec![0, 1], Vec::new())],
            ),
            TableSpec::new(
                TableShape::new(4, 3),
                vec![
                    OpeningBatch::new(vec![0, 2], Vec::new()),
                    OpeningBatch::new(vec![1], Vec::new()),
                ],
            ),
        ])
    }

    #[test]
    fn opening_protocol_table_shapes_returns_shapes_in_protocol_order() {
        // Invariant:
        //     table_shapes() returns one shape per table spec, in the order
        //     they were handed to the constructor.
        //
        // Fixture state:
        //     two-table protocol with shapes (3, 2) and (4, 3).
        let protocol = two_table_protocol();

        let shapes = protocol.table_shapes();

        assert_eq!(shapes, vec![TableShape::new(3, 2), TableShape::new(4, 3)],);
    }

    #[test]
    fn opening_protocol_num_openings_sums_per_table_schedules() {
        // Invariant:
        //     num_openings() equals the sum of point-schedule lengths across
        //     every table.
        //
        // Fixture state:
        //     table 0: 1 schedule entry.
        //     table 1: 2 schedule entries.
        //     Expected total: 3.
        let protocol = two_table_protocol();
        assert_eq!(protocol.num_openings(), 3);

        // Edge case: empty protocol → no openings.
        let empty = OpeningProtocol::new(vec![]);
        assert_eq!(empty.num_openings(), 0);
    }

    #[test]
    fn opening_protocol_iter_openings_yields_batches_in_transcript_order() {
        // Invariant:
        //     The opening stream walks tables in protocol order, then each
        //     table's points in schedule order, pairing every batch with the
        //     index of its owning table.
        //
        // Fixture state:
        //     two-table protocol; every batch has direct columns and an empty
        //     successor-view side.
        //
        // Each tuple is (table index, direct columns, successor-view columns).
        let protocol = two_table_protocol();

        let collected: Vec<(usize, Vec<usize>, Vec<usize>)> = protocol
            .iter_openings()
            .map(|(table_idx, batch)| (table_idx, batch.current().to_vec(), batch.next().to_vec()))
            .collect();

        assert_eq!(
            collected,
            vec![
                (0, vec![0, 1], vec![]),
                (1, vec![0, 2], vec![]),
                (1, vec![1], vec![]),
            ],
        );
    }

    #[test]
    fn opening_protocol_pad_to_min_num_variables_grows_small_tables() {
        // Invariant:
        //     pad_to_min_num_variables raises every table arity below the
        //     given floor while leaving widths and the opening schedule
        //     untouched.
        //
        // Fixture state:
        //     table 0: arity 3, two cols.
        //     table 1: arity 4, three cols.
        //     Floor: 5 → both tables widen to arity 5.
        let padded = two_table_protocol().pad_to_min_num_variables(5);

        let shapes = padded.table_shapes();
        assert_eq!(shapes[0], TableShape::new(5, 2));
        assert_eq!(shapes[1], TableShape::new(5, 3));
        // Schedules survive the pad untouched.
        assert_eq!(padded.num_openings(), 3);
    }

    #[test]
    fn opening_protocol_pad_to_min_num_variables_leaves_large_tables_alone() {
        // Invariant:
        //     pad_to_min_num_variables is a one-way clamp: tables already at
        //     or above the floor are not modified.
        //
        // Fixture state:
        //     single table of arity 3; floor 2 → no change.
        let original = single_table_protocol();
        let original_shapes = original.table_shapes();

        let padded = original.pad_to_min_num_variables(2);

        assert_eq!(padded.table_shapes(), original_shapes);
    }

    #[test]
    fn table_spec_accessors_forward_constructor_arguments() {
        // Invariant:
        //     A spec stores the shape and the schedule verbatim; the
        //     accessors return references to the stored data.
        //
        // Fixture state:
        //     shape (3, 2), schedule [[0, 1], [0]].
        let shape = TableShape::new(3, 2);
        let schedule = vec![
            OpeningBatch::new(vec![0, 1], Vec::new()),
            OpeningBatch::new(vec![0], Vec::new()),
        ];
        let spec = TableSpec::new(shape, schedule.clone());

        assert_eq!(*spec.shape(), shape);
        assert_eq!(spec.point_schedule(), &schedule);
    }

    #[test]
    fn table_spec_accepts_mixed_current_next_batches() {
        // Invariant:
        //     One sampled point may carry both direct openings and
        //     successor-view openings.
        //     The batch length is the sum of the two side lengths.
        //
        // Fixture state:
        //     width 3, two points.
        //
        //     point 0: direct {0, 2}, successor-view {1}  -> len 3
        //     point 1: direct {},     successor-view {2}  -> len 1
        let spec = TableSpec::new(
            TableShape::new(3, 3),
            vec![
                OpeningBatch::new(vec![0, 2], vec![1]),
                OpeningBatch::new(Vec::new(), vec![2]),
            ],
        );

        let batches = spec.point_schedule();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].current(), &[0, 2]);
        assert_eq!(batches[0].next(), &[1]);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].current(), &[]);
        assert_eq!(batches[1].next(), &[2]);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    #[should_panic]
    fn table_spec_new_panics_on_out_of_range_poly_idx() {
        // Invariant:
        //     A schedule that addresses a column past the table width is
        //     rejected at construction time, not deferred to use.
        //
        // Fixture state:
        //     width = 2, schedule references column 2 (only 0 and 1 are valid).
        let _ = TableSpec::new(
            TableShape::new(3, 2),
            vec![OpeningBatch::new(vec![2], Vec::new())],
        );
    }

    #[test]
    #[should_panic]
    fn table_spec_new_panics_on_out_of_range_next_poly_idx() {
        // Invariant:
        //     The successor-view side is validated against the table width too,
        //     not just the direct side.
        //
        // Fixture state:
        //     width = 2; successor-view names column 2 (only 0 and 1 are valid).
        let _ = TableSpec::new(
            TableShape::new(3, 2),
            vec![OpeningBatch::new(vec![0], vec![2])],
        );
    }

    #[test]
    #[should_panic]
    fn table_spec_new_panics_on_duplicate_column_in_one_side() {
        // Invariant:
        //     A column may not repeat within one side of a batch.
        //
        // Fixture state:
        //     width = 2; the direct side names column 0 twice.
        let _ = TableSpec::new(
            TableShape::new(3, 2),
            vec![OpeningBatch::new(vec![0, 0], Vec::new())],
        );
    }

    #[test]
    fn table_spec_new_accepts_same_column_across_both_sides() {
        // Invariant:
        //     Opening one column on both sides (direct and successor-view) is
        //     intended and must be accepted; only within-side repeats are rejected.
        let spec = TableSpec::new(
            TableShape::new(3, 2),
            vec![OpeningBatch::new(vec![0], vec![0])],
        );
        assert_eq!(spec.point_schedule().len(), 1);
    }

    #[test]
    #[should_panic]
    fn opening_batch_new_panics_on_empty_batch() {
        // Invariant:
        //     A batch with no columns on either side is rejected at construction.
        //     Such a batch would consume a transcript point yet open nothing.
        let _: OpeningBatch<usize> = OpeningBatch::new(vec![], vec![]);
    }

    #[test]
    fn table_spec_pad_to_min_num_variables_grows_small_arity() {
        // Invariant:
        //     The in-place padder raises the table's arity but never lowers it,
        //     and never touches the column count.
        //
        // Fixture state:
        //     start arity 2, width 3; pad to floor 4 → arity becomes 4.
        let mut spec = TableSpec::new(
            TableShape::new(2, 3),
            vec![OpeningBatch::new(vec![0], Vec::new())],
        );

        spec.pad_to_min_num_variables(4);

        assert_eq!(spec.shape().num_variables(), 4);
        assert_eq!(spec.shape().width(), 3);
    }

    #[test]
    fn table_spec_pad_to_min_num_variables_leaves_larger_arity_alone() {
        // Invariant:
        //     A spec already wider than the floor is left unchanged.
        //
        // Fixture state:
        //     start arity 5, width 1; pad to floor 3 → no change.
        let mut spec = TableSpec::new(
            TableShape::new(5, 1),
            vec![OpeningBatch::new(vec![0], Vec::new())],
        );

        spec.pad_to_min_num_variables(3);

        assert_eq!(spec.shape().num_variables(), 5);
        assert_eq!(spec.shape().width(), 1);
    }
}
