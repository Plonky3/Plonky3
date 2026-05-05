use alloc::vec::Vec;

use p3_matrix::Dimensions;
use p3_util::log2_ceil_usize;

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

/// Point-local opening schedule for one table.
///
/// Each outer entry corresponds to one sampled point for the table. The inner
/// vector lists the column indices opened at that point.
pub type PointSchedule = Vec<Vec<usize>>;

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

    /// Returns the table specs in protocol order.
    pub fn tables(&self) -> &[TableSpec] {
        &self.0
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
    pub fn iter_openings(&self) -> impl Iterator<Item = (usize, &[usize])> {
        self.0.iter().enumerate().flat_map(|(table_idx, table)| {
            table
                .point_schedule()
                .iter()
                .map(move |polys| (table_idx, polys.as_slice()))
        })
    }
}

/// Claimed evaluations for an [`OpeningProtocol`].
#[derive(Debug, Clone, Default)]
pub struct OpeningClaims<EF> {
    protocol: OpeningProtocol,
    evals: Vec<Vec<EF>>,
}

impl<EF> OpeningClaims<EF> {
    /// Attaches evaluations to a protocol.
    ///
    /// # Panics
    ///
    /// - The number of evaluation batches must match the protocol.
    /// - Each evaluation batch must match the number of opened columns at the
    ///   corresponding point.
    pub fn new(protocol: OpeningProtocol, evals: Vec<Vec<EF>>) -> Self {
        assert_eq!(protocol.num_openings(), evals.len());
        assert!(
            protocol
                .iter_openings()
                .zip(&evals)
                .all(|((_, polys), evals)| polys.len() == evals.len())
        );
        Self { protocol, evals }
    }

    /// Returns the protocol these evaluations claim against.
    pub const fn protocol(&self) -> &OpeningProtocol {
        &self.protocol
    }

    /// Iterates over all opening claims in transcript order.
    pub fn iter_openings(&self) -> impl Iterator<Item = (usize, &[usize], &[EF])> {
        self.protocol
            .iter_openings()
            .zip(&self.evals)
            .map(|((table_idx, polys), evals)| (table_idx, polys, evals.as_slice()))
    }
}

impl TableSpec {
    /// Builds a table spec from a shape and point-local opening schedule.
    ///
    /// # Panics
    ///
    /// - Every scheduled polynomial index must be less than the table width.
    pub fn new(shape: TableShape, point_schedule: PointSchedule) -> Self {
        assert!(
            point_schedule
                .iter()
                .flatten()
                .all(|&poly_idx| poly_idx < shape.width())
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

    /// Number of variables after applying the minimum first-round folding size.
    pub fn committed_num_variables(&self, folding: usize) -> usize {
        self.shape.num_variables().max(folding)
    }
}
