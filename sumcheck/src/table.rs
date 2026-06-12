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

/// One point-local opening batch for a table.
///
/// A batch represents one sampled transcript point. `current` columns are
/// opened at that point; `next` columns are opened through the repeat-last Next
/// view at the same point. Evaluation order is always current first, then next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpeningBatch<T> {
    current: Vec<T>,
    next: Vec<T>,
}

impl<T> OpeningBatch<T> {
    /// Builds an opening batch from current and Next column lists.
    ///
    /// # Panics
    ///
    /// Panics if both lists are empty.
    pub fn new(current: Vec<T>, next: Vec<T>) -> Self {
        assert!(
            !current.is_empty() || !next.is_empty(),
            "opening batch must name at least one column"
        );
        Self { current, next }
    }

    /// Returns current entries.
    pub fn current(&self) -> &[T] {
        &self.current
    }

    /// Returns Next entries.
    pub fn next(&self) -> &[T] {
        &self.next
    }

    /// Returns the total number of entries in this batch.
    pub const fn len(&self) -> usize {
        self.current.len() + self.next.len()
    }

    /// Returns true if this batch has no entries.
    pub const fn is_empty(&self) -> bool {
        self.current.is_empty() && self.next.is_empty()
    }

    /// Iterates over current entries first, then Next entries.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.current.iter().chain(&self.next)
    }

    /// Returns true if the current/Next split has the same lengths as `other`.
    pub const fn has_same_shape<U>(&self, other: &OpeningBatch<U>) -> bool {
        self.current.len() == other.current.len() && self.next.len() == other.next.len()
    }
}

impl<T: Clone> OpeningBatch<T> {
    /// Flattens entries in transcript order: current first, then Next.
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

/// Point-local opening schedule for one table.
///
/// Each entry corresponds to one sampled point for the table. Every entry may
/// request current openings, repeat-last Next openings, or both.
pub type OpeningRequest = OpeningBatch<usize>;

/// Claimed evaluations for one point-local opening request.
pub type OpeningEvals<EF> = OpeningBatch<EF>;

/// Point-local opening schedule for one table.
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
    pub fn new(shape: TableShape, point_schedule: PointSchedule) -> Self {
        assert!(point_schedule.iter().all(|batch| {
            batch
                .current()
                .iter()
                .chain(batch.next())
                .all(|&poly_idx| poly_idx < shape.width())
        }));
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
    /// Opening schedules are unchanged because padding only adds zero rows to
    /// the committed table.
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
    //     point 0: current columns {0, 1}
    //     point 1: current column  {0}
    //
    // Yields num_openings() == 2 and iter_openings() emits the two
    // batches in transcript order against table index 0.
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
    //     table 0: arity 3, two cols. Schedule: current {0, 1} once.
    //     table 1: arity 4, three cols. Schedule: current {0, 2}; current {1}.
    //
    // Yields num_openings() == 3 and iter_openings() emits the three
    // batches as (0, current [0, 1]), (1, current [0, 2]), (1, current [1]).
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
        //     iter_openings() walks tables in protocol order, then walks
        //     each table's point schedule in insertion order, emitting
        //     (table_idx, batch) per scheduled point.
        //
        // Fixture state:
        //     two-table protocol with mixed schedules.
        //     Expected stream: current-only batches for [0, 1], [0, 2], [1].
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
        //     One scheduled point may request current and Next openings over
        //     the same sampled point, and the batch length is their sum.
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
        //     Next column indices are validated against table width too.
        let _ = TableSpec::new(
            TableShape::new(3, 2),
            vec![OpeningBatch::new(vec![0], vec![2])],
        );
    }

    #[test]
    #[should_panic]
    fn opening_batch_new_panics_on_empty_batch() {
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
