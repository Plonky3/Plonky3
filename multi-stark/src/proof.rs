//! Proof data and opening shapes for multilinear AIR verification.

use alloc::vec;
use alloc::vec::Vec;

use p3_sumcheck::generic_degree::GenericDegreeProof;
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};

use crate::config::{Commitment, MultiStarkConfig, PcsProof};

/// A complete multilinear AIR proof.
///
/// The three parts are checked in order against one shared transcript:
/// - the commitment binds the trace columns.
/// - the sumcheck reduces the AIR constraint to one bound-point claim.
/// - the opening proves the trace columns at that point.
pub struct MultiStarkProof<C: MultiStarkConfig> {
    /// Commitment to the trace columns.
    pub commitment: Commitment<C>,
    /// Zerocheck sumcheck transcript for the alpha-batched constraint.
    pub sumcheck: GenericDegreeProof<C::Val, C::Challenge>,
    /// Commitment opening at the bound sumcheck point.
    pub opening: PcsProof<C>,
}

/// Build the single-table opening protocol shared by the prover and verifier.
///
/// Version one commits a single table.
/// That table is the whole execution trace.
///
/// It opens at the bound sumcheck point:
/// - every current-row column.
/// - every successor-view column read by the AIR.
///
/// The prover and verifier build this identically, so their opening transcripts agree.
///
/// # Arguments
///
/// - Trace arity, with one variable per row bit.
/// - Number of trace columns.
/// - Column indices read through the successor view.
pub(crate) fn single_table_protocol(
    log_height: usize,
    width: usize,
    next_columns: &[usize],
) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_height, width),
        vec![OpeningBatch::new(
            (0..width).collect::<Vec<_>>(),
            next_columns.to_vec(),
        )],
    )])
}
